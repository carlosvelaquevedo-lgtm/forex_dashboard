# app.py — Trend Following Scanner (FULL MERGED VERSION)
# Streamlit + SQLite + OANDA (primary) / yfinance fallback
# Includes:
# - Live scan
# - FIXED historical scan
# - Filter rejection heatmap

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

PAIRS = [
    "EURUSD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "AUDUSD=X",
    "USDCHF=X",
    "USDCAD=X",
    "GC=F"
]
assert isinstance(PAIRS, list) and len(PAIRS) > 0, "PAIRS is not defined correctly"
st.write("Pairs loaded:", PAIRS)


# ────────────────────────────────────────────────
# LOGGING
# ────────────────────────────────────────────────

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

PAIRS = [
    "EURUSD=X", "USDJPY=X", "GBPUSD=X",
    "AUDUSD=X", "USDCHF=X", "USDCAD=X",
    "GC=F"
]

INSTRUMENT_MAP = {
    "EURUSD=X": "EUR_USD",
    "USDJPY=X": "USD_JPY",
    "GBPUSD=X": "GBP_USD",
    "AUDUSD=X": "AUD_USD",
    "USDCHF=X": "USD_CHF",
    "USDCAD=X": "USD_CAD",
    "GC=F": "XAU_USD"
}

DB_FILE = "signals.db"

# ────────────────────────────────────────────────
# SESSION STATE
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
    except Exception:
        st.session_state.oanda_api = None

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
            symbol_raw TEXT,
            instrument TEXT,
            direction TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            units INTEGER,
            open_time TEXT,
            confidence REAL
        )
        """)
        conn.commit()

def load_signals():
    init_db()
    with get_db() as conn:
        return pd.read_sql("SELECT * FROM signals ORDER BY open_time DESC", conn)

def save_signal(sig: Dict):
    with get_db() as conn:
        conn.execute("""
        INSERT OR REPLACE INTO signals
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            sig["id"], sig["symbol_raw"], sig["instrument"], sig["direction"],
            sig["entry"], sig["sl"], sig["tp"], sig["units"],
            sig["open_time"], sig["confidence"]
        ))
        conn.commit()

# ────────────────────────────────────────────────
# DATA FETCH + INDICATORS
# ────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_with_indicators(symbol: str, period: str, interval: str):
    instrument = INSTRUMENT_MAP.get(symbol)
    df = None

    if st.session_state.oanda_api:
        try:
            gran = {"1d": "D", "4h": "H4"}[interval]
            params = {"count": 1500, "granularity": gran, "price": "M"}
            r = InstrumentsCandles(instrument=instrument, params=params)
            resp = st.session_state.oanda_api.request(r)
            rows = [{
                "time": pd.to_datetime(c["time"]),
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
            } for c in resp["candles"] if c["mid"]]
            df = pd.DataFrame(rows).set_index("time")
        except Exception:
            df = None

    if df is None or df.empty:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df = df[["Open", "High", "Low", "Close"]]
        df.columns = ["open", "high", "low", "close"]

    for span, name in [(9,"ema_fast"), (21,"ema_slow"), (50,"ema_tf"), (200,"ema_ts")]:
        df[name] = df["close"].ewm(span=span, adjust=False).mean()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    tr_s = tr.ewm(alpha=1/14, adjust=False).mean()
    df["plus_di"] = 100 * pd.Series(plus_dm).ewm(alpha=1/14).mean() / tr_s
    df["minus_di"] = 100 * pd.Series(minus_dm).ewm(alpha=1/14).mean() / tr_s
    dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"])
    df["adx"] = dx.ewm(alpha=1/14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(span=14).mean()
    loss = -delta.clip(upper=0).ewm(span=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df.dropna()

# ────────────────────────────────────────────────
# LIVE SCAN
# ────────────────────────────────────────────────
def run_scan():
    results = []

    for pair in PAIRS:
        try:
            htf = fetch_data(pair, Config.HTF)
            ltf = fetch_data(pair, Config.LTF)

            # ─────────────────────────────────────────
            # HARD GUARDS (this fixes your crash)
            # ─────────────────────────────────────────
            if htf is None or ltf is None:
                logger.warning(f"{pair}: data fetch returned None")
                continue

            if htf.empty or ltf.empty:
                logger.warning(f"{pair}: empty dataframe (htf={len(htf)}, ltf={len(ltf)})")
                continue

            if len(htf) < Config.MIN_BARS or len(ltf) < Config.MIN_BARS:
                logger.warning(f"{pair}: not enough bars")
                continue
            # ─────────────────────────────────────────

            htf = compute_indicators(htf)
            ltf = compute_indicators(ltf)

            # Guard again AFTER indicators (NaNs)
            if ltf.dropna().empty:
                logger.warning(f"{pair}: indicators produced NaNs only")
                continue

            last = ltf.iloc[-1]   # ← now safe
            trend = last["trend"]

            signal = generate_signal(htf, ltf)

            if signal:
                results.append(signal)

        except Exception as e:
            logger.exception(f"{pair}: scan failed → {e}")

    return results

# ────────────────────────────────────────────────
# HISTORICAL SCAN + HEATMAP (FIXED)
# ────────────────────────────────────────────────
def run_historical_scan(lookback_days: int):
    signals = []
    rejections = {}

    for pair in PAIRS:
        rejections[pair] = {}

        try:
            htf = fetch_data(pair, Config.HTF)
            ltf = fetch_data(pair, Config.LTF)

            # ─────────────────────────────────────────
            # HARD DATA GUARDS
            # ─────────────────────────────────────────
            if htf is None or ltf is None:
                rejections[pair]["no_data"] = rejections[pair].get("no_data", 0) + 1
                continue

            if htf.empty or ltf.empty:
                rejections[pair]["empty_df"] = rejections[pair].get("empty_df", 0) + 1
                continue

            # Ensure datetime index
            if not isinstance(ltf.index, pd.DatetimeIndex):
                ltf.index = pd.to_datetime(ltf.index, errors="coerce")

            ltf = ltf.sort_index()
            htf = htf.sort_index()

            if ltf.index.isna().all():
                rejections[pair]["bad_index"] = rejections[pair].get("bad_index", 0) + 1
                continue

            if len(ltf) < Config.MIN_BARS or len(htf) < Config.MIN_BARS:
                rejections[pair]["not_enough_bars"] = rejections[pair].get("not_enough_bars", 0) + 1
                continue
            # ─────────────────────────────────────────

            htf = compute_indicators(htf)
            ltf = compute_indicators(ltf)

            # Drop NaNs AFTER indicators
            ltf = ltf.dropna()
            htf = htf.dropna()

            if ltf.empty:
                rejections[pair]["nan_after_indicators"] = rejections[pair].get("nan_after_indicators", 0) + 1
                continue

            # ─────────────────────────────────────────
            # SAFE LOOKBACK WINDOW
            # ─────────────────────────────────────────
            last_ts = ltf.index.max()
            cutoff = last_ts - pd.Timedelta(days=lookback_days)

            recent = ltf[ltf.index >= cutoff]

            if recent.empty:
                rejections[pair]["no_recent_data"] = rejections[pair].get("no_recent_data", 0) + 1
                continue
            # ─────────────────────────────────────────

            for i in range(len(recent)):
                sub_ltf = recent.iloc[: i + 1]
                sub_htf = htf[htf.index <= sub_ltf.index[-1]]

                if sub_htf.empty:
                    rejections[pair]["no_htf_alignment"] = rejections[pair].get("no_htf_alignment", 0) + 1
                    continue

                signal = generate_signal(sub_htf, sub_ltf)
                if signal:
                    signals.append(signal)

        except Exception as e:
            rejections[pair]["exception"] = rejections[pair].get("exception", 0) + 1
            logger.exception(f"{pair}: historical scan failed → {e}")

    return signals, rejections

# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────

st.set_page_config("Trend Scanner", layout="wide")
st.title("📈 Trend Following Scanner")

with st.sidebar:
    if st.button("Run Scan Now"):
        with st.spinner("Scanning..."):
            res = run_scan()
            st.success(f"{len(res)} new signals")

    st.subheader("Historical Scan")
    lookback = st.slider("Lookback days", 1, 14, 3)

    if st.button("Scan Historical"):
        sigs, rej = run_historical_scan(lookback)
        if sigs:
            st.dataframe(pd.DataFrame(sigs))
        if rej:
            st.subheader("📊 Filter Rejection Heatmap")
            df = pd.DataFrame.from_dict(rej, orient="index", columns=["Count"])
            st.bar_chart(df)

st.header("Saved Signals")
df = load_signals()
if df.empty:
    st.info("No saved signals yet")
else:
    st.dataframe(df, use_container_width=True)

