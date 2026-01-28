# app.py — Trading Scanner Dashboard with Streamlit + SQLite + OANDA/yfinance
# Primary: OANDA v20 API for candles; fallback to yfinance
# Enhanced: pullbacks, HTF momentum, configurable filters, safety checks

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Set, Optional
import logging
import time
from contextlib import contextmanager
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.exceptions import V20Error

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

@dataclass
class Config:
    ACCOUNT_SIZE_USD: float = 10000
    RISK_PER_TRADE_PCT: float = 0.01
    ATR_SL_MULT: float = 1.5
    TARGET_RR: float = 2.0
    MIN_ADX: float = 20.0
    MIN_ATR_RATIO_PCT: float = 0.3
    MAX_SIGNAL_AGE_DAYS: int = 3
    MIN_HTF_ATR_PCT: float = 0.8
    ENABLE_DI_CONFIRM: bool = True
    ENABLE_RSI_FILTER: bool = True
    ENABLE_PULLBACK_FILTER: bool = True
    PULLBACK_ATR_MAX: float = 1.2
    AGGRESSIVE_MODE: bool = False

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

DB_FILE = "signals.db"

PAIRS = [
    "EURUSD=X", "USDJPY=X", "GBPUSD=X",
    "AUDUSD=X", "USDCHF=X", "USDCAD=X",
    "GC=F"
]

INSTRUMENT_MAP = {
    "EURUSD=X": "EUR_USD", "USDJPY=X": "USD_JPY", "GBPUSD=X": "GBP_USD",
    "AUDUSD=X": "AUD_USD", "USDCHF=X": "USD_CHF", "USDCAD=X": "USD_CAD",
    "GC=F": "XAU_USD"
}

# ────────────────────────────────────────────────
# SESSION STATE & OANDA INIT
# ────────────────────────────────────────────────

if "config" not in st.session_state:
    st.session_state.config = Config().to_dict()
if "last_scan" not in st.session_state:
    st.session_state.last_scan = None
if "oanda_api" not in st.session_state:
    try:
        st.session_state.oanda_api = API(
            access_token=st.secrets["OANDA"]["access_token"],
            environment=st.secrets["OANDA"]["environment"]
        )
        log.info("OANDA API initialized successfully")
    except Exception as e:
        st.warning(f"OANDA init failed (will fallback to yfinance): {e}")
        st.session_state.oanda_api = None

def get_config() -> Config:
    return Config.from_dict(st.session_state.config)

def update_config(**kwargs):
    st.session_state.config.update(kwargs)

# ────────────────────────────────────────────────
# DATABASE (unchanged)
# ────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id TEXT PRIMARY KEY,
            symbol_raw TEXT NOT NULL,
            instrument TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry REAL NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            units INTEGER NOT NULL,
            open_time TEXT NOT NULL,
            outcome TEXT DEFAULT '',
            close_time TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            confidence REAL DEFAULT 50.0,
            current_price REAL,
            pnl_pips REAL
        )
        """)
        conn.commit()

def load_signals() -> pd.DataFrame:
    init_db()
    try:
        with get_db() as conn:
            df = pd.read_sql("SELECT * FROM signals ORDER BY open_time DESC", conn)
        df = df.fillna({'notes':'','outcome':'','confidence':50.0})
        return df
    except Exception as e:
        log.error(f"Load error: {e}")
        return pd.DataFrame()

def save_signal(sig: Dict) -> bool:
    try:
        with get_db() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO signals
            (id, symbol_raw, instrument, direction, entry, sl, tp, units,
             open_time, outcome, close_time, notes, confidence, current_price, pnl_pips)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                sig["id"], sig["symbol_raw"], sig["instrument"], sig["direction"],
                sig["entry"], sig["sl"], sig["tp"], sig["units"],
                sig["open_time"], sig.get("outcome",""), sig.get("close_time",""),
                sig.get("notes",""), sig.get("confidence",50.0),
                sig.get("current_price"), sig.get("pnl_pips")
            ))
            conn.commit()
        return True
    except Exception as e:
        log.error(f"Save error: {e}")
        return False

# ────────────────────────────────────────────────
# DATA FETCH + INDICATORS (OANDA primary, yfinance fallback)
# ────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_with_indicators(symbol: str, period: str, interval: str):
    instrument = INSTRUMENT_MAP.get(symbol)
    if not instrument:
        log.warning(f"No OANDA instrument mapped for {symbol}")
        return None

    # Try OANDA first
    api = st.session_state.oanda_api
    df = None
    if api:
        try:
            granularity_map = {"1d": "D", "4h": "H4"}
            if interval not in granularity_map:
                raise ValueError(f"Unsupported interval: {interval}")
            granularity = granularity_map[interval]

            count_map = {"1y_1d": 600, "6mo_4h": 2000}
            count = count_map.get(f"{period}_{interval}", 1000)

            params = {
                "count": count,
                "price": "M",  # mid prices
                "granularity": granularity,
            }

            r = InstrumentsCandles(instrument=instrument, params=params)
            resp = api.request(r)
            candles = resp.get("candles", [])

            if not candles:
                raise ValueError("No candles returned from OANDA")

            rows = []
            for c in candles:
                mid = c.get("mid")
                if mid:
                    rows.append({
                        "open": float(mid["o"]),
                        "high": float(mid["h"]),
                        "low": float(mid["l"]),
                        "close": float(mid["c"]),
                        "time": c["time"],
                    })

            df = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df = df[["open", "high", "low", "close"]]
            log.info(f"Fetched {len(df)} candles from OANDA for {instrument} ({period}, {interval})")
        except (V20Error, Exception) as e:
            log.warning(f"OANDA fetch failed for {instrument}: {e} — falling back to yfinance")

    # Fallback to yfinance if OANDA failed or not available
    if df is None or df.empty:
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df.empty:
                log.warning(f"yfinance empty for {symbol}")
                return None
            df = df[['Open','High','Low','Close']].copy()
            df.columns = ['open','high','low','close']
            log.info(f"Fallback: Fetched from yfinance for {symbol}")
        except Exception as e:
            log.error(f"yfinance fallback failed for {symbol}: {e}")
            return None

    if df is None or df.empty:
        return None

    # ─── Indicators (same as original) ───
    # EMAs
    for span, name in [(9,'ema_fast'), (21,'ema_slow'), (50,'ema_tf'), (200,'ema_ts')]:
        df[name] = df['close'].ewm(span=span, adjust=False).mean()

    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()

    # ADX + DI
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    tr_s = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_s
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_s
    di_sum = (plus_di + minus_di).replace(0, 1e-6)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    df = df.dropna()
    if df.empty:
        log.warning(f"No valid rows after dropna for {symbol} ({period}, {interval})")
        return None

    return df

# ────────────────────────────────────────────────
# SIGNAL ENGINE (unchanged)
# ────────────────────────────────────────────────
# ... (your find_signal function remains the same; it uses the df from fetch)

# ────────────────────────────────────────────────
# HISTORICAL + NEAR-MISS SCAN (unchanged)
# ────────────────────────────────────────────────
# ... (run_historical_scan unchanged)

# ────────────────────────────────────────────────
# LIVE SCAN (unchanged)
# ────────────────────────────────────────────────
# ... (run_scan unchanged)

# ────────────────────────────────────────────────
# UI (unchanged, but add optional status if needed)
# ────────────────────────────────────────────────

st.set_page_config(page_title="Trend Scanner", layout="wide")
st.title("📈 Trend Following Scanner")

with st.sidebar:
    st.header("Controls")

    # Optional: show OANDA status
    if st.session_state.oanda_api:
        st.success("OANDA connected")
    else:
        st.warning("OANDA not connected — using yfinance fallback")

    # ... rest of your sidebar unchanged (Run Scan, filters, historical scan, etc.)

# ────────────────────────────────────────────────
# MAIN CONTENT (unchanged)
# ────────────────────────────────────────────────

st.header("Saved Signals")

df = load_signals()
if df.empty:
    st.info("No confirmed signals yet. Try 'Run Scan Now' or check historical opportunities.")
else:
    st.dataframe(df, use_container_width=True)
