# app.py — Trading Scanner Dashboard with Streamlit + SQLite + OANDA + Email
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
import time

try:
    from tpqoa import tpqoa
except ImportError:
    tpqoa = None

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
    ENABLE_PULLBACK_FILTER: bool = True      # new
    PULLBACK_ATR_MAX: float = 1.2           # how close to EMA for pullback
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

PIP_SIZES = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "USD_CHF": 0.0001, "USD_CAD": 0.0001, "USD_JPY": 0.01,
    "XAU_USD": 0.1
}

PIP_VALUE_PER_LOT = {
    "EUR_USD": 10.0, "GBP_USD": 10.0, "AUD_USD": 10.0,
    "USD_JPY": lambda p: 1000.0 / p,
    "USD_CHF": lambda p: 10.0 / p,
    "USD_CAD": lambda p: 10.0 / p,
    "XAU_USD": 1.0,
}

# ────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────

if "config" not in st.session_state:
    st.session_state.config = Config().to_dict()
if "last_scan" not in st.session_state:
    st.session_state.last_scan = None
if "oanda_error" not in st.session_state:
    st.session_state.oanda_error = None

def get_config() -> Config:
    return Config.from_dict(st.session_state.config)

def update_config(**kwargs):
    st.session_state.config.update(kwargs)

# ────────────────────────────────────────────────
# DATABASE
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
# DATA FETCH + INDICATORS
# ────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_with_indicators(symbol: str, period: str, interval: str):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None

        df = df[['Open','High','Low','Close']].copy()
        df.columns = ['open','high','low','close']

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

        return df.dropna()
    except Exception as e:
        log.error(f"Fetch+indicators error {symbol}: {e}")
        return None

# ────────────────────────────────────────────────
# SIGNAL ENGINE
# ────────────────────────────────────────────────

def find_signal(symbol: str, htf: pd.DataFrame, ltf: pd.DataFrame, seen: Set[str], config: Config) -> Optional[Dict]:
    if len(htf) < 200 or len(ltf) < 50:
        return None

    last_htf = htf.iloc[-1]

    # Trend filter
    trend = None
    if last_htf['close'] > last_htf['ema_tf'] > last_htf['ema_ts']:
        trend = "UP"
    elif last_htf['close'] < last_htf['ema_tf'] < last_htf['ema_ts']:
        trend = "DOWN"
    if not trend:
        return None

    # HTF momentum confirmation
    if trend == "UP" and last_htf['ema_fast'] < last_htf['ema_slow']:
        return None
    if trend == "DOWN" and last_htf['ema_fast'] > last_htf['ema_slow']:
        return None

    # Aggressive scaling
    min_adx = config.MIN_ADX * (0.65 if config.AGGRESSIVE_MODE else 1.0)
    min_atr_pct = config.MIN_ATR_RATIO_PCT * (0.6 if config.AGGRESSIVE_MODE else 1.0)
    max_age = config.MAX_SIGNAL_AGE_DAYS * (2.5 if config.AGGRESSIVE_MODE else 1.0)

    # Find crossover
    ltf = ltf.copy()
    ltf['bull_cross'] = (ltf['ema_fast'] > ltf['ema_slow']) & (ltf['ema_fast'].shift(1) <= ltf['ema_slow'].shift(1))
    ltf['bear_cross'] = (ltf['ema_fast'] < ltf['ema_slow']) & (ltf['ema_fast'].shift(1) >= ltf['ema_slow'].shift(1))

    mask = ltf['bull_cross'] if trend == "UP" else ltf['bear_cross']
    crosses = ltf[mask]
    if crosses.empty:
        return None

    # Take most recent
    sig_idx = crosses.index[-1]
    sig = ltf.loc[sig_idx]

    # Age check
    age_days = (ltf.index[-1] - sig_idx).total_seconds() / 86400
    if age_days > max_age:
        return None

    # Core filters
    if sig['adx'] < min_adx:
        return None
    if (sig['atr'] / sig['close']) * 100 < min_atr_pct:
        return None

    # DI confirmation
    if config.ENABLE_DI_CONFIRM:
        if trend == "UP" and sig['plus_di'] <= sig['minus_di']:
            return None
        if trend == "DOWN" and sig['minus_di'] <= sig['plus_di']:
            return None

    # RSI filter
    if config.ENABLE_RSI_FILTER:
        if trend == "UP" and sig['rsi'] > 70:
            return None
        if trend == "DOWN" and sig['rsi'] < 30:
            return None

    # Pullback filter (optional)
    if config.ENABLE_PULLBACK_FILTER:
        dist_to_ema = abs(sig['close'] - sig['ema_fast']) / sig['atr']
        if dist_to_ema > config.PULLBACK_ATR_MAX:
            return None  # too far from EMA → no pullback

    # Safety: avoid extreme overextension
    if abs(sig['close'] - sig['ema_slow']) / sig['atr'] > 2.0:
        return None

    # Entry = next candle open
    if len(ltf) <= ltf.index.get_loc(sig_idx) + 1:
        return None
    entry = float(ltf.iloc[ltf.index.get_loc(sig_idx) + 1]['open'])

    atr = sig['atr']
    instrument = INSTRUMENT_MAP[symbol]
    sl_mult = 1.2 if instrument == "XAU_USD" else config.ATR_SL_MULT

    if trend == "UP":
        sl = entry - atr * sl_mult
        tp = entry + (entry - sl) * config.TARGET_RR
        direction = "LONG"
    else:
        sl = entry + atr * sl_mult
        tp = entry - (sl - entry) * config.TARGET_RR
        direction = "SHORT"

    units = 1000  # placeholder – replace with your position_size function

    confidence = min(
        sig['adx'] + 
        (sig['plus_di'] - sig['minus_di'] if trend == "UP" else sig['minus_di'] - sig['plus_di']) * 0.5 +
        (10 if 40 < sig['rsi'] < 60 else 0) -
        (10 if sig['adx'] > 45 else 0),
        100
    )

    sid = f"{symbol}_{sig_idx.strftime('%Y%m%d_%H%M')}_{direction}"
    if sid in seen:
        return None
    seen.add(sid)

    return {
        "id": sid,
        "symbol_raw": symbol,
        "instrument": instrument,
        "direction": direction,
        "entry": entry,
        "sl": float(sl),
        "tp": float(tp),
        "units": units,
        "open_time": datetime.now(timezone.utc).isoformat(),
        "confidence": round(confidence, 1)
    }

# ────────────────────────────────────────────────
# SCAN
# ────────────────────────────────────────────────

def run_scan():
    cfg = get_config()
    df = load_signals()
    seen = set(df['id']) if not df.empty else set()
    new_signals = []

    for symbol in PAIRS:
        htf = fetch_with_indicators(symbol, "1y", "1d")
        ltf = fetch_with_indicators(symbol, "6mo", "4h")
        if htf is None or ltf is None:
            continue

        sig = find_signal(symbol, htf, ltf, seen, cfg)
        if sig and save_signal(sig):
            new_signals.append(sig)

    return new_signals

# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────

st.set_page_config(page_title="Trend Scanner", layout="wide")
st.title("📈 Trend Following Scanner")

with st.sidebar:
    st.header("Controls")

    if st.button("Run Scan Now", type="primary", use_container_width=True):
        with st.spinner("Scanning markets..."):
            new = run_scan()
            st.session_state.last_scan = datetime.now()
        if new:
            st.success(f"Found {len(new)} new signal(s)")
        else:
            st.info("No new signals this run")

    if st.session_state.last_scan:
        age = datetime.now() - st.session_state.last_scan
        st.caption(f"Last scan: {st.session_state.last_scan.strftime('%H:%M:%S')} ({age.seconds//60} min ago)")

    st.divider()
    st.subheader("Signal Filters (Testing)")

    aggressive = st.checkbox("Aggressive Mode (more signals)", value=get_config().AGGRESSIVE_MODE)
    update_config(AGGRESSIVE_MODE=aggressive)

    st.checkbox("Require DI Direction", value=get_config().ENABLE_DI_CONFIRM, key="di")
    st.checkbox("Skip Extreme RSI", value=get_config().ENABLE_RSI_FILTER, key="rsi")
    st.checkbox("Require Pullback to EMA", value=get_config().ENABLE_PULLBACK_FILTER, key="pullback")

    update_config(
        ENABLE_DI_CONFIRM=st.session_state.di,
        ENABLE_RSI_FILTER=st.session_state.rsi,
        ENABLE_PULLBACK_FILTER=st.session_state.pullback
    )

    if st.button("Reset Filters"):
        st.session_state.config = Config().to_dict()
        st.rerun()

    st.divider()
    st.subheader("Auto-refresh")
    if st.checkbox("Auto-refresh every 60s"):
        time.sleep(60)
        st.rerun()

# Main content
df = load_signals()
if df.empty:
    st.info("No signals yet. Click 'Run Scan Now' to start.")
else:
    st.dataframe(df, use_container_width=True)
