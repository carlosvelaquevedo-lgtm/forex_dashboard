# app.py — Trend Following Scanner (FULL MERGED VERSION - FIXED v3)
# Streamlit + SQLite + OANDA (primary) / yfinance fallback
#
# v3 Features:
# - Backtest stats with win probability
# - Expanded DB schema (adx, atr_ratio, rsi, units_formatted)
# - Signal freshness filter / cleanup
# - Current market context (price, % to SL/TP, pips to SL/TP)
# - Optimized historical scan (step interval)
# - Scan duration / timestamp tracking
# - Color-coded direction in tables
# - Streamlined scan results display

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Dict, Set, Optional, List, Tuple
import logging
import time
import random
from contextlib import contextmanager
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.pricing import PricingInfo# ────────────────────────────────────────────────
# LOGGING
# ────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)# ────────────────────────────────────────────────
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
    # Timeframe settings
    HTF: str = "1d"
    LTF: str = "4h"
    MIN_BARS: int = 50
    # Historical scan optimization
    HIST_SCAN_STEP: int = 6  # Check every N bars instead of every bardef to_dict(self):
    return self.__dict__

@classmethod
def from_dict(cls, d):
    return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})# ────────────────────────────────────────────────
# PAIRS AND INSTRUMENT MAPPING
# ────────────────────────────────────────────────

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
REVERSE_INSTRUMENT_MAP = {v: k for k, v in INSTRUMENT_MAP.items()}# Pip multipliers and values for position sizing
INSTRUMENT_SPECS = {
    "EURUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5},
    "USDJPY=X": {"pip_multiplier": 100, "pip_value_per_lot": 1000 / 150, "type": "forex", "decimals": 3},
    "GBPUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5},
    "AUDUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5},
    "USDCHF=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5},
    "USDCAD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0 / 1.36, "type": "forex", "decimals": 5},
    "GC=F": {"pip_multiplier": 1, "pip_value_per_lot": 100.0, "type": "commodity", "decimals": 2},
}# yfinance period limits by interval
YFINANCE_LIMITS = {
    "1d": {"max_period": "2y", "max_days": 730},
    "4h": {"max_period": "60d", "max_days": 60},
    "1h": {"max_period": "730d", "max_days": 730},
    "15m": {"max_period": "60d", "max_days": 60},
}DB_FILE = "signals.db"# ────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────

if "config" not in st.session_state:
    st.session_state.config = Config().to_dict()if "last_scan" not in st.session_state:
    st.session_state.last_scan = Noneif "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = Noneif "last_scan_duration" not in st.session_state:
    st.session_state.last_scan_duration = Noneif "oanda_available" not in st.session_state:
    st.session_state.oanda_available = Falseif "oanda_account_id" not in st.session_state:
    st.session_state.oanda_account_id = Noneif "oanda_api" not in st.session_state:
    try:
        st.session_state.oanda_api = API(
            access_token=st.secrets["OANDA"]["access_token"],
            environment=st.secrets["OANDA"]["environment"]
        )
        st.session_state.oanda_available = True
        st.session_state.oanda_account_id = st.secrets["OANDA"].get("account_id")
    except Exception:
        st.session_state.oanda_api = None
        st.session_state.oanda_available = Falsedef get_config() -> Config:
    return Config.from_dict(st.session_state.config)def update_config(**kwargs):
    st.session_state.config.update(kwargs)# ────────────────────────────────────────────────
# DATABASE (EXPANDED SCHEMA)
# ────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()def init_db():
    with get_db() as conn:
        # Create signals table with expanded schema
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
            units_formatted TEXT,
            open_time TEXT,
            confidence REAL,
            adx REAL,
            atr_ratio REAL,
            rsi REAL,
            status TEXT DEFAULT 'active',
            outcome TEXT,
            close_price REAL,
            close_time TEXT,
            pnl_pips REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)    # Create backtest results table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS backtest_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_time TEXT,
        lookback_days INTEGER,
        total_signals INTEGER,
        wins INTEGER,
        losses INTEGER,
        win_rate REAL,
        avg_rr REAL,
        total_pips REAL,
        config_json TEXT
    )
    """)
    
    conn.commit()
    
    # Migration: add new columns if they don't exist
    _migrate_db(conn)def _migrate_db(conn):
    """Add new columns to existing tables if needed."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(signals)")
    existing_cols = {row[1] for row in cursor.fetchall()}new_cols = [
    ("units_formatted", "TEXT"),
    ("adx", "REAL"),
    ("atr_ratio", "REAL"),
    ("rsi", "REAL"),
    ("status", "TEXT DEFAULT 'active'"),
    ("outcome", "TEXT"),
    ("close_price", "REAL"),
    ("close_time", "TEXT"),
    ("pnl_pips", "REAL"),
    ("created_at", "TEXT"),
]

for col_name, col_type in new_cols:
    if col_name not in existing_cols:
        try:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}")
            logger.info(f"Added column {col_name} to signals table")
        except Exception as e:
            logger.warning(f"Could not add column {col_name}: {e}")

conn.commit()def load_signals(status_filter: str = None) -> pd.DataFrame:
    init_db()
    with get_db() as conn:
        if status_filter:
            return pd.read_sql(
                "SELECT * FROM signals WHERE status = ? ORDER BY open_time DESC",
                conn, params=(status_filter,)
            )
        return pd.read_sql("SELECT * FROM signals ORDER BY open_time DESC", conn)def save_signal(sig: Dict):
    with get_db() as conn:
        conn.execute("""
        INSERT OR REPLACE INTO signals
        (id, symbol_raw, instrument, direction, entry, sl, tp, units, units_formatted,
         open_time, confidence, adx, atr_ratio, rsi, status, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sig["id"], sig["symbol_raw"], sig["instrument"], sig["direction"],
            sig["entry"], sig["sl"], sig["tp"], sig["units"], sig.get("units_formatted", ""),
            sig["open_time"], sig["confidence"], sig.get("adx"), sig.get("atr_ratio"),
            sig.get("rsi"), sig.get("status", "active"), datetime.utcnow().isoformat()
        ))
        conn.commit()def update_signal_status(signal_id: str, status: str, outcome: str = None, 
                         close_price: float = None, pnl_pips: float = None):
    with get_db() as conn:
        conn.execute("""
        UPDATE signals 
        SET status = ?, outcome = ?, close_price = ?, close_time = ?, pnl_pips = ?
        WHERE id = ?
        """, (status, outcome, close_price, datetime.utcnow().isoformat(), pnl_pips, signal_id))
        conn.commit()def archive_old_signals(max_age_days: int) -> int:
    """Archive signals older than max_age_days. Returns count of archived signals."""
    cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
    with get_db() as conn:
        cursor = conn.execute("""
        UPDATE signals 
        SET status = 'archived' 
        WHERE status = 'active' AND open_time < ?
        """, (cutoff,))
        conn.commit()
        return cursor.rowcountdef delete_archived_signals() -> int:
    """Delete all archived signals. Returns count of deleted signals."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM signals WHERE status = 'archived'")
        conn.commit()
        return cursor.rowcountdef save_backtest_result(result: Dict):
    with get_db() as conn:
        conn.execute("""
        INSERT INTO backtest_results 
        (run_time, lookback_days, total_signals, wins, losses, win_rate, avg_rr, total_pips, config_json)
        VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            datetime.utcnow().isoformat(),
            result["lookback_days"],
            result["total_signals"],
            result["wins"],
            result["losses"],
            result["win_rate"],
            result["avg_rr"],
            result["total_pips"],
            str(result.get("config", {}))
        ))
        conn.commit()def load_backtest_history() -> pd.DataFrame:
    init_db()
    with get_db() as conn:
        return pd.read_sql("SELECT * FROM backtest_results ORDER BY run_time DESC LIMIT 20", conn)# ────────────────────────────────────────────────
# TIMEZONE NORMALIZATION
# ────────────────────────────────────────────────

def normalize_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df# ────────────────────────────────────────────────
# DATA FETCH
# ────────────────────────────────────────────────

def check_yfinance_limits(interval: str, lookback_days: int) -> Tuple[bool, str]:
    limits = YFINANCE_LIMITS.get(interval, {"max_days": 60})
    max_days = limits["max_days"]
    if lookback_days > max_days:
        return False, f"yfinance only supports ~{max_days} days for {interval} interval."
    return True, ""def fetch_data(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    instrument = INSTRUMENT_MAP.get(symbol)
    df = Noneinterval_map = {
    "1d": {"gran": "D", "yf_interval": "1d", "yf_period": "2y"},
    "4h": {"gran": "H4", "yf_interval": "4h", "yf_period": "60d"},
    "1h": {"gran": "H1", "yf_interval": "1h", "yf_period": "730d"},
}

settings = interval_map.get(interval, {"gran": "D", "yf_interval": "1d", "yf_period": "2y"})

# Try OANDA first
if st.session_state.oanda_api and instrument:
    try:
        params = {"count": 1500, "granularity": settings["gran"], "price": "M"}
        r = InstrumentsCandles(instrument=instrument, params=params)
        resp = st.session_state.oanda_api.request(r)
        rows = [{
            "time": pd.to_datetime(c["time"]),
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
        } for c in resp.get("candles", []) if c.get("mid")]
        if rows:
            df = pd.DataFrame(rows).set_index("time")
            df = normalize_to_utc(df)
    except Exception as e:
        logger.warning(f"OANDA fetch failed for {symbol}: {e}")
        df = None

# Fallback to yfinance
if df is None or df.empty:
    try:
        df = yf.download(
            symbol,
            period=settings["yf_period"],
            interval=settings["yf_interval"],
            progress=False
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.columns = ["open", "high", "low", "close"]
        df = normalize_to_utc(df)
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {symbol}: {e}")
        return None

return dfdef get_current_price(symbol: str) -> Optional[float]:
    """Fetch the current/latest price for a symbol."""
    instrument = INSTRUMENT_MAP.get(symbol)# Try OANDA pricing endpoint first
if st.session_state.oanda_api and instrument and st.session_state.oanda_account_id:
    try:
        params = {"instruments": instrument}
        r = PricingInfo(accountID=st.session_state.oanda_account_id, params=params)
        resp = st.session_state.oanda_api.request(r)
        prices = resp.get("prices", [])
        if prices:
            bid = float(prices[0]["bids"][0]["price"])
            ask = float(prices[0]["asks"][0]["price"])
            return (bid + ask) / 2
    except Exception as e:
        logger.warning(f"OANDA pricing failed for {symbol}: {e}")

# Fallback: get latest candle close
try:
    df = fetch_data(symbol, "1h")
    if df is not None and not df.empty:
        return df["close"].iloc[-1]
except Exception:
    pass

return Nonedef get_current_prices_batch(symbols: List[str]) -> Dict[str, float]:
    """Fetch current prices for multiple symbols."""
    prices = {}# Try OANDA batch pricing
if st.session_state.oanda_api and st.session_state.oanda_account_id:
    try:
        instruments = [INSTRUMENT_MAP.get(s) for s in symbols if s in INSTRUMENT_MAP]
        instruments = [i for i in instruments if i]
        if instruments:
            params = {"instruments": ",".join(instruments)}
            r = PricingInfo(accountID=st.session_state.oanda_account_id, params=params)
            resp = st.session_state.oanda_api.request(r)
            for price_data in resp.get("prices", []):
                inst = price_data.get("instrument")
                symbol = REVERSE_INSTRUMENT_MAP.get(inst)
                if symbol:
                    bid = float(price_data["bids"][0]["price"])
                    ask = float(price_data["asks"][0]["price"])
                    prices[symbol] = (bid + ask) / 2
    except Exception as e:
        logger.warning(f"OANDA batch pricing failed: {e}")

# Fill missing with yfinance
for symbol in symbols:
    if symbol not in prices:
        price = get_current_price(symbol)
        if price:
            prices[symbol] = price

return prices# ────────────────────────────────────────────────
# INDICATORS
# ────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return dfdf = df.copy()

# EMAs
for span, name in [(9, "ema_fast"), (21, "ema_slow"), (50, "ema_tf"), (200, "ema_ts")]:
    df[name] = df["close"].ewm(span=span, adjust=False).mean()

# ATR
tr = pd.concat([
    df["high"] - df["low"],
    (df["high"] - df["close"].shift()).abs(),
    (df["low"] - df["close"].shift()).abs()
], axis=1).max(axis=1)
df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

# ADX and DI
up = df["high"].diff()
down = -df["low"].diff()
plus_dm = np.where((up > down) & (up > 0), up, 0)
minus_dm = np.where((down > up) & (down > 0), down, 0)
tr_s = tr.ewm(alpha=1/14, adjust=False).mean()
df["plus_di"] = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_s
df["minus_di"] = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_s
dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"])
df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()

# RSI
delta = df["close"].diff()
gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
rs = gain / loss.replace(0, np.nan)
df["rsi"] = 100 - (100 / (1 + rs))

# Trend direction
df["trend"] = np.where(df["ema_fast"] > df["ema_slow"], "bullish",
                       np.where(df["ema_fast"] < df["ema_slow"], "bearish", "neutral"))

# ATR ratio
df["atr_ratio"] = (df["atr"] / df["close"]) * 100

return df# ────────────────────────────────────────────────
# POSITION SIZING
# ────────────────────────────────────────────────

def calculate_position_size(symbol: str, entry: float, sl: float, risk_usd: float) -> int:
    specs = INSTRUMENT_SPECS.get(symbol)
    if specs is None:
        specs = {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex"}sl_distance = abs(entry - sl)
if sl_distance == 0:
    return 0

if specs["type"] == "forex":
    sl_pips = sl_distance * specs["pip_multiplier"]
    if sl_pips == 0:
        return 0
    pip_value_per_unit = specs["pip_value_per_lot"] / 100000
    units = int(risk_usd / (sl_pips * pip_value_per_unit))
    return units

elif specs["type"] == "commodity":
    risk_per_contract = sl_distance * specs["pip_value_per_lot"]
    if risk_per_contract == 0:
        return 0
    contracts = int(risk_usd / risk_per_contract)
    return max(contracts, 0)

return 0 def format_position_size(symbol: str, units: int) -> str:
    specs = INSTRUMENT_SPECS.get(symbol, {"type": "forex"})
    if specs["type"] == "commodity":
        return f"{units} contract(s)"
    if units >= 100000:
        return f"{units / 100000:.2f} std lot ({units:,})"
    elif units >= 10000:
        return f"{units / 10000:.1f} mini ({units:,})"
    elif units >= 1000:
        return f"{units / 1000:.1f} micro ({units:,})"
    return f"{units:,} units"def price_to_pips(symbol: str, price_distance: float) -> float:
    """Convert price distance to pips."""
    specs = INSTRUMENT_SPECS.get(symbol, {"pip_multiplier": 10000})
    return price_distance * specs["pip_multiplier"]# ────────────────────────────────────────────────
# SIGNAL ID GENERATION
# ────────────────────────────────────────────────

def generate_signal_id(symbol: str, direction: str, timestamp: pd.Timestamp) -> str:
    ts_str = timestamp.strftime('%Y%m%d_%H%M%S')
    random_suffix = f"{int(time.time() * 1000) % 10000:04d}_{random.randint(1000, 9999)}"
    return f"{symbol}_{direction}_{ts_str}_{random_suffix}"# ────────────────────────────────────────────────
# SIGNAL GENERATION
# ────────────────────────────────────────────────

def generate_signal(htf: pd.DataFrame, ltf: pd.DataFrame, symbol: str) -> Optional[Dict]:
    cfg = get_config()if htf.empty or ltf.empty:
    return None

htf_last = htf.iloc[-1]
ltf_last = ltf.iloc[-1]

htf_trend = htf_last.get("trend", "neutral")
if htf_trend == "neutral":
    return None

adx_val = ltf_last.get("adx", 0)
if pd.isna(adx_val) or adx_val < cfg.MIN_ADX:
    return None

atr_ratio = ltf_last.get("atr_ratio", 0)
if pd.isna(atr_ratio) or atr_ratio < cfg.MIN_ATR_RATIO_PCT:
    return None

direction = None
if htf_trend == "bullish":
    if cfg.ENABLE_DI_CONFIRM:
        if ltf_last.get("plus_di", 0) <= ltf_last.get("minus_di", 0):
            return None
    if cfg.ENABLE_RSI_FILTER:
        if ltf_last.get("rsi", 50) > 70:
            return None
    direction = "LONG"

elif htf_trend == "bearish":
    if cfg.ENABLE_DI_CONFIRM:
        if ltf_last.get("minus_di", 0) <= ltf_last.get("plus_di", 0):
            return None
    if cfg.ENABLE_RSI_FILTER:
        if ltf_last.get("rsi", 50) < 30:
            return None
    direction = "SHORT"

if direction is None:
    return None

if cfg.ENABLE_PULLBACK_FILTER:
    atr = ltf_last.get("atr", 0)
    ema_slow = ltf_last.get("ema_slow", ltf_last["close"])
    pullback_dist = abs(ltf_last["close"] - ema_slow)
    if atr > 0 and pullback_dist > cfg.PULLBACK_ATR_MAX * atr:
        return None

entry = ltf_last["close"]
atr = ltf_last.get("atr", entry * 0.01)

if direction == "LONG":
    sl = entry - (cfg.ATR_SL_MULT * atr)
    tp = entry + (cfg.ATR_SL_MULT * atr * cfg.TARGET_RR)
else:
    sl = entry + (cfg.ATR_SL_MULT * atr)
    tp = entry - (cfg.ATR_SL_MULT * atr * cfg.TARGET_RR)

risk_amount = cfg.ACCOUNT_SIZE_USD * cfg.RISK_PER_TRADE_PCT
units = calculate_position_size(symbol, entry, sl, risk_amount)
confidence = min(100, (adx_val / cfg.MIN_ADX) * 50 + (atr_ratio / cfg.MIN_ATR_RATIO_PCT) * 25)
signal_id = generate_signal_id(symbol, direction, ltf.index[-1])

return {
    "id": signal_id,
    "symbol_raw": symbol,
    "instrument": INSTRUMENT_MAP.get(symbol, symbol),
    "direction": direction,
    "entry": round(entry, 5),
    "sl": round(sl, 5),
    "tp": round(tp, 5),
    "units": units,
    "units_formatted": format_position_size(symbol, units),
    "open_time": ltf.index[-1].isoformat(),
    "confidence": round(confidence, 1),
    "adx": round(adx_val, 1),
    "atr_ratio": round(atr_ratio, 3),
    "rsi": round(ltf_last.get("rsi", 50), 1),
    "status": "active",
}# ────────────────────────────────────────────────
# MARKET CONTEXT
# ────────────────────────────────────────────────

def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add current price, % to SL/TP, pips to SL/TP columns."""
    if df.empty:
        return dfdf = df.copy()

# Get unique symbols
symbols = df["symbol_raw"].unique().tolist()
current_prices = get_current_prices_batch(symbols)

# Add context columns
df["current_price"] = df["symbol_raw"].map(current_prices)

# Calculate distances
def calc_context(row):
    if pd.isna(row.get("current_price")):
        return pd.Series({
            "pct_to_sl": None, "pct_to_tp": None,
            "pips_to_sl": None, "pips_to_tp": None,
            "status_emoji": ""
        })
    
    current = row["current_price"]
    entry = row["entry"]
    sl = row["sl"]
    tp = row["tp"]
    direction = row["direction"]
    symbol = row["symbol_raw"]
    
    # Calculate distances
    if direction == "LONG":
        dist_to_sl = current - sl
        dist_to_tp = tp - current
        pnl_pct = ((current - entry) / entry) * 100
    else:
        dist_to_sl = sl - current
        dist_to_tp = current - tp
        pnl_pct = ((entry - current) / entry) * 100
    
    pct_to_sl = (dist_to_sl / current) * 100 if current else None
    pct_to_tp = (dist_to_tp / current) * 100 if current else None
    
    pips_to_sl = price_to_pips(symbol, dist_to_sl)
    pips_to_tp = price_to_pips(symbol, dist_to_tp)
    
    # Status emoji
    if dist_to_sl <= 0:
        status = ""  # Hit SL
    elif dist_to_tp <= 0:
        status = ""  # Hit TP
    elif pnl_pct > 0:
        status = ""  # In profit
    else:
        status = ""  # In loss
    
    return pd.Series({
        "pct_to_sl": round(pct_to_sl, 2) if pct_to_sl else None,
        "pct_to_tp": round(pct_to_tp, 2) if pct_to_tp else None,
        "pips_to_sl": round(pips_to_sl, 1) if pips_to_sl else None,
        "pips_to_tp": round(pips_to_tp, 1) if pips_to_tp else None,
        "status_emoji": status
    })

context_df = df.apply(calc_context, axis=1)
df = pd.concat([df, context_df], axis=1)

return df# ────────────────────────────────────────────────
# BACKTEST ENGINE
# ────────────────────────────────────────────────

def backtest_signal(signal: Dict, future_data: pd.DataFrame) -> Dict:
    """
    Backtest a single signal against future price data.
    Returns outcome: 'win', 'loss', or 'open'.
    """
    if future_data.empty:
        return {"outcome": "open", "exit_price": None, "pnl_pips": 0}entry = signal["entry"]
sl = signal["sl"]
tp = signal["tp"]
direction = signal["direction"]
symbol = signal["symbol_raw"]

for idx, row in future_data.iterrows():
    high = row["high"]
    low = row["low"]
    
    if direction == "LONG":
        # Check SL hit first (conservative)
        if low <= sl:
            pnl_pips = price_to_pips(symbol, sl - entry)
            return {"outcome": "loss", "exit_price": sl, "pnl_pips": pnl_pips}
        # Check TP hit
        if high >= tp:
            pnl_pips = price_to_pips(symbol, tp - entry)
            return {"outcome": "win", "exit_price": tp, "pnl_pips": pnl_pips}
    else:  # SHORT
        # Check SL hit first
        if high >= sl:
            pnl_pips = price_to_pips(symbol, entry - sl)
            return {"outcome": "loss", "exit_price": sl, "pnl_pips": pnl_pips}
        # Check TP hit
        if low <= tp:
            pnl_pips = price_to_pips(symbol, entry - tp)
            return {"outcome": "win", "exit_price": tp, "pnl_pips": pnl_pips}

# Neither SL nor TP hit
last_close = future_data["close"].iloc[-1]
if direction == "LONG":
    pnl_pips = price_to_pips(symbol, last_close - entry)
else:
    pnl_pips = price_to_pips(symbol, entry - last_close)

return {"outcome": "open", "exit_price": last_close, "pnl_pips": pnl_pips}def calculate_backtest_stats(signals: List[Dict], data_cache: Dict[str, pd.DataFrame]) -> Dict:
    """
    Calculate backtest statistics for a list of signals.
    """
    if not signals:
        return {
            "total_signals": 0,
            "wins": 0,
            "losses": 0,
            "open": 0,
            "win_rate": 0,
            "avg_rr": 0,
            "total_pips": 0,
            "avg_pips_per_trade": 0,
            "profit_factor": 0,
            "signals_with_outcomes": []
        }wins = 0
losses = 0
open_trades = 0
total_pips = 0
gross_profit = 0
gross_loss = 0
signals_with_outcomes = []

for signal in signals:
    symbol = signal["symbol_raw"]
    signal_time = pd.to_datetime(signal["open_time"])
    
    # Get future data after signal
    if symbol in data_cache:
        full_data = data_cache[symbol]
        future_data = full_data[full_data.index > signal_time]
    else:
        future_data = pd.DataFrame()
    
    result = backtest_signal(signal, future_data)
    
    signal_copy = signal.copy()
    signal_copy.update(result)
    signals_with_outcomes.append(signal_copy)
    
    if result["outcome"] == "win":
        wins += 1
        gross_profit += result["pnl_pips"]
    elif result["outcome"] == "loss":
        losses += 1
        gross_loss += abs(result["pnl_pips"])
    else:
        open_trades += 1
    
    total_pips += result["pnl_pips"]

total_closed = wins + losses
win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
avg_pips = total_pips / len(signals) if signals else 0
profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

# Calculate average R:R achieved
cfg = get_config()
target_rr = cfg.TARGET_RR
avg_rr = (win_rate / 100 * target_rr) - ((100 - win_rate) / 100 * 1) if total_closed > 0 else 0

return {
    "total_signals": len(signals),
    "wins": wins,
    "losses": losses,
    "open": open_trades,
    "win_rate": round(win_rate, 1),
    "avg_rr": round(avg_rr, 2),
    "total_pips": round(total_pips, 1),
    "avg_pips_per_trade": round(avg_pips, 1),
    "profit_factor": round(profit_factor, 2),
    "gross_profit": round(gross_profit, 1),
    "gross_loss": round(gross_loss, 1),
    "signals_with_outcomes": signals_with_outcomes
}# ────────────────────────────────────────────────
# WIN PROBABILITY ESTIMATION
# ────────────────────────────────────────────────

def estimate_win_probability(signal: Dict, historical_stats: Dict = None) -> float:
    """
    Estimate win probability for a signal based on:
    - Confidence score
    - Historical win rate (if available)
    - ADX strength
    - RSI position
    """
    base_prob = 50.0  # Start at 50%# Adjust for confidence (0-100 maps to ±10%)
confidence = signal.get("confidence", 50)
base_prob += (confidence - 50) * 0.2

# Adjust for ADX strength
adx = signal.get("adx", 20)
if adx > 30:
    base_prob += 5
elif adx > 40:
    base_prob += 10

# Adjust for RSI (favor middle range)
rsi = signal.get("rsi", 50)
if 40 <= rsi <= 60:
    base_prob += 5
elif rsi < 30 or rsi > 70:
    base_prob -= 5

# Incorporate historical win rate if available
if historical_stats and historical_stats.get("win_rate", 0) > 0:
    hist_rate = historical_stats["win_rate"]
    # Weighted average: 60% estimate, 40% historical
    base_prob = base_prob * 0.6 + hist_rate * 0.4

return max(10, min(90, round(base_prob, 1)))# ────────────────────────────────────────────────
# LIVE SCAN
# ────────────────────────────────────────────────

def run_scan() -> Tuple[List[Dict], float]:
    """Run a live scan. Returns (signals, duration_seconds)."""
    cfg = get_config()
    results = []
    start_time = time.time()for pair in PAIRS:
    try:
        htf = fetch_data(pair, cfg.HTF)
        ltf = fetch_data(pair, cfg.LTF)

        if htf is None or ltf is None:
            continue
        if htf.empty or ltf.empty:
            continue
        if len(htf) < cfg.MIN_BARS or len(ltf) < cfg.MIN_BARS:
            continue

        htf = compute_indicators(htf)
        ltf = compute_indicators(ltf)
        htf = htf.dropna()
        ltf = ltf.dropna()

        if ltf.empty:
            continue

        signal = generate_signal(htf, ltf, pair)

        if signal:
            # Add win probability
            signal["win_prob"] = estimate_win_probability(signal)
            results.append(signal)
            save_signal(signal)

    except Exception as e:
        logger.exception(f"{pair}: scan failed → {e}")

duration = time.time() - start_time
return results, duration# ────────────────────────────────────────────────
# HISTORICAL SCAN (OPTIMIZED)
# ────────────────────────────────────────────────

def run_historical_scan(lookback_days: int) -> Tuple[List[Dict], Dict, List[str], Dict, float]:
    """
    Run historical scan with backtest.
    Returns (signals, rejections, warnings, backtest_stats, duration).
    """
    cfg = get_config()
    signals = []
    rejections = {}
    warnings = []
    data_cache = {}  # Cache data for backtesting
    start_time = time.time()# Check yfinance limits
if not st.session_state.oanda_available:
    is_ok, warning = check_yfinance_limits(cfg.LTF, lookback_days)
    if not is_ok:
        warnings.append(f" {warning}")

for pair in PAIRS:
    rejections[pair] = {}

    try:
        htf = fetch_data(pair, cfg.HTF)
        ltf = fetch_data(pair, cfg.LTF)

        if htf is None or ltf is None:
            rejections[pair]["no_data"] = 1
            continue
        if htf.empty or ltf.empty:
            rejections[pair]["empty_df"] = 1
            continue

        if not isinstance(ltf.index, pd.DatetimeIndex):
            ltf.index = pd.to_datetime(ltf.index, errors="coerce")
        if not isinstance(htf.index, pd.DatetimeIndex):
            htf.index = pd.to_datetime(htf.index, errors="coerce")

        ltf = ltf.sort_index()
        htf = htf.sort_index()

        if len(ltf) < cfg.MIN_BARS or len(htf) < cfg.MIN_BARS:
            rejections[pair]["not_enough_bars"] = 1
            continue

        htf = compute_indicators(htf)
        ltf = compute_indicators(ltf)
        ltf = ltf.dropna()
        htf = htf.dropna()

        if ltf.empty:
            rejections[pair]["nan_after_indicators"] = 1
            continue

        # Cache for backtesting
        data_cache[pair] = ltf

        # Safe lookback window
        last_ts = ltf.index.max()
        cutoff = last_ts - pd.Timedelta(days=lookback_days)
        recent = ltf[ltf.index >= cutoff]

        if recent.empty:
            rejections[pair]["no_recent_data"] = 1
            continue

        # OPTIMIZED: Step through bars instead of every bar
        step = cfg.HIST_SCAN_STEP
        indices = list(range(0, len(recent), step))
        if len(recent) - 1 not in indices:
            indices.append(len(recent) - 1)

        for i in indices:
            sub_ltf = recent.iloc[: i + 1].copy()
            sub_htf = htf[htf.index <= sub_ltf.index[-1]].copy()

            if sub_htf.empty:
                continue

            signal = generate_signal(sub_htf, sub_ltf, pair)
            if signal:
                signals.append(signal)

    except Exception as e:
        rejections[pair]["exception"] = 1
        logger.exception(f"{pair}: historical scan failed → {e}")

# Calculate backtest statistics
backtest_stats = calculate_backtest_stats(signals, data_cache)

# Add win probability to each signal
for signal in backtest_stats["signals_with_outcomes"]:
    signal["win_prob"] = estimate_win_probability(signal, backtest_stats)

# Save backtest result
if signals:
    save_backtest_result({
        "lookback_days": lookback_days,
        "total_signals": backtest_stats["total_signals"],
        "wins": backtest_stats["wins"],
        "losses": backtest_stats["losses"],
        "win_rate": backtest_stats["win_rate"],
        "avg_rr": backtest_stats["avg_rr"],
        "total_pips": backtest_stats["total_pips"],
        "config": cfg.to_dict()
    })

duration = time.time() - start_time
return signals, rejections, warnings, backtest_stats, durationdef build_rejection_heatmap(rejections: Dict) -> pd.DataFrame:
    all_reasons = set()
    for pair_rej in rejections.values():
        all_reasons.update(pair_rej.keys())data = []
for pair, rej_dict in rejections.items():
    row = {"pair": pair}
    for reason in all_reasons:
        row[reason] = rej_dict.get(reason, 0)
    data.append(row)

df = pd.DataFrame(data)
if not df.empty:
    df = df.set_index("pair")
return df# ────────────────────────────────────────────────
# STYLING HELPERS
# ────────────────────────────────────────────────

def style_direction(val):
    """Color direction column: green for LONG, red for SHORT."""
    if val == "LONG":
        return "background-color: #c6efce; color: #006100"
    elif val == "SHORT":
        return "background-color: #ffc7ce; color: #9c0006"
    return ""def style_outcome(val):
    """Color outcome column."""
    if val == "win":
        return "background-color: #c6efce; color: #006100"
    elif val == "loss":
        return "background-color: #ffc7ce; color: #9c0006"
    return ""def format_signals_dataframe(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """Format and filter signals dataframe for display."""
    if df.empty:
        return dfif columns:
    # Only include columns that exist
    columns = [c for c in columns if c in df.columns]
    df = df[columns]

return df# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────

st.set_page_config("Trend Scanner", layout="wide")
st.title(" Trend Following Scanner")# Initialize DB
init_db()# Status indicators
col_status1, col_status2, col_status3 = st.columns(3)
with col_status1:
    if st.session_state.oanda_available:
        st.success(" OANDA Connected")
    else:
        st.warning(" yfinance fallback")with col_status2:
    if st.session_state.last_scan_time:
        st.info(f" Last scan: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")with col_status3:
    if st.session_state.last_scan_duration:
        st.info(f" Duration: {st.session_state.last_scan_duration:.1f}s")# Sidebar
with st.sidebar:
    st.header(" Configuration")cfg = get_config()

with st.expander("Scan Settings", expanded=False):
    min_adx = st.slider("Min ADX", 10.0, 40.0, cfg.MIN_ADX, 1.0)
    min_atr = st.slider("Min ATR Ratio %", 0.1, 1.0, cfg.MIN_ATR_RATIO_PCT, 0.05)
    atr_sl = st.slider("ATR SL Multiplier", 1.0, 3.0, cfg.ATR_SL_MULT, 0.1)
    target_rr = st.slider("Target R:R", 1.0, 5.0, cfg.TARGET_RR, 0.5)
    hist_step = st.slider("Hist Scan Step", 1, 12, cfg.HIST_SCAN_STEP, 1,
                          help="Check every N bars in historical scan (higher = faster)")

    di_confirm = st.checkbox("Enable DI Confirmation", cfg.ENABLE_DI_CONFIRM)
    rsi_filter = st.checkbox("Enable RSI Filter", cfg.ENABLE_RSI_FILTER)
    pullback_filter = st.checkbox("Enable Pullback Filter", cfg.ENABLE_PULLBACK_FILTER)

    if st.button("Update Settings"):
        update_config(
            MIN_ADX=min_adx, MIN_ATR_RATIO_PCT=min_atr, ATR_SL_MULT=atr_sl,
            TARGET_RR=target_rr, HIST_SCAN_STEP=hist_step,
            ENABLE_DI_CONFIRM=di_confirm, ENABLE_RSI_FILTER=rsi_filter,
            ENABLE_PULLBACK_FILTER=pullback_filter,
        )
        st.success("Settings updated!")

with st.expander("Account Settings", expanded=False):
    account_size = st.number_input("Account Size (USD)", 1000, 1000000, int(cfg.ACCOUNT_SIZE_USD), 1000)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, cfg.RISK_PER_TRADE_PCT * 100, 0.5) / 100

    if st.button("Update Account"):
        update_config(ACCOUNT_SIZE_USD=account_size, RISK_PER_TRADE_PCT=risk_pct)
        st.success("Account settings updated!")

st.divider()

# Live scan
st.subheader(" Live Scan")
if st.button("Run Scan Now", type="primary"):
    with st.spinner("Scanning markets..."):
        res, duration = run_scan()
        st.session_state.last_scan = res
        st.session_state.last_scan_time = datetime.now()
        st.session_state.last_scan_duration = duration
        if res:
            st.success(f" {len(res)} signal(s) found!")
        else:
            st.info("No signals found")

st.divider()

# Historical scan
st.subheader(" Historical Scan")
if not st.session_state.oanda_available:
    max_days = YFINANCE_LIMITS.get(cfg.LTF, {}).get("max_days", 60)
    st.caption(f" Limit: ~{max_days} days for {cfg.LTF}")
    lookback = st.slider("Lookback days", 1, min(14, max_days), 3)
else:
    lookback = st.slider("Lookback days", 1, 30, 3)

if st.button("Scan Historical"):
    with st.spinner(f"Scanning last {lookback} days..."):
        sigs, rej, warns, stats, duration = run_historical_scan(lookback)
        for w in warns:
            st.warning(w)
        st.session_state.hist_signals = sigs
        st.session_state.hist_rejections = rej
        st.session_state.hist_stats = stats
        st.session_state.last_scan_duration = duration
        st.session_state.last_scan_time = datetime.now()
        if sigs:
            st.success(f"Found {len(sigs)} signals")
        else:
            st.info("No signals found")

st.divider()

# Signal management
st.subheader(" Signal Management")
max_age = st.number_input("Max Signal Age (days)", 1, 30, cfg.MAX_SIGNAL_AGE_DAYS)

col_arch1, col_arch2 = st.columns(2)
with col_arch1:
    if st.button("Archive Old"):
        count = archive_old_signals(max_age)
        st.success(f"Archived {count} signals")
with col_arch2:
    if st.button("Delete Archived"):
        count = delete_archived_signals()
        st.success(f"Deleted {count} signals")# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([" Live Signals", " Historical / Backtest", " All Signals", " Stats"])# ────────────────────────────────────────────────
# TAB 1: Live Signals
# ────────────────────────────────────────────────
with tab1:
    st.header("Live Scan Results")if st.session_state.last_scan:
    scan_df = pd.DataFrame(st.session_state.last_scan)
    
    # Display columns
    display_cols = ["instrument", "direction", "entry", "sl", "tp", "units", 
                    "confidence", "adx", "atr_ratio", "rsi", "win_prob"]
    display_cols = [c for c in display_cols if c in scan_df.columns]
    
    styled_df = scan_df[display_cols].style.applymap(
        style_direction, subset=["direction"]
    )
    st.dataframe(styled_df, use_container_width=True)
    
    # Market context for live signals
    if st.checkbox("Show Market Context", value=True):
        with st.spinner("Fetching current prices..."):
            context_df = add_market_context(scan_df)
            context_cols = ["instrument", "direction", "entry", "current_price", 
                           "pct_to_sl", "pct_to_tp", "pips_to_sl", "pips_to_tp", "status_emoji"]
            context_cols = [c for c in context_cols if c in context_df.columns]
            st.subheader(" Market Context")
            st.dataframe(context_df[context_cols], use_container_width=True)
else:
    st.info("No live scan results. Click 'Run Scan Now' to scan markets.")# ────────────────────────────────────────────────
# TAB 2: Historical / Backtest
# ────────────────────────────────────────────────
with tab2:
    st.header("Historical Scan & Backtest Results")if "hist_stats" in st.session_state and st.session_state.hist_stats:
    stats = st.session_state.hist_stats
    
    # Stats cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Signals", stats["total_signals"])
    with col2:
        st.metric("Win Rate", f"{stats['win_rate']}%")
    with col3:
        st.metric("Wins / Losses", f"{stats['wins']} / {stats['losses']}")
    with col4:
        st.metric("Total Pips", f"{stats['total_pips']}")
    with col5:
        st.metric("Profit Factor", f"{stats['profit_factor']}")
    
    st.divider()
    
    # Detailed stats
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(" Performance Metrics")
        metrics_data = {
            "Metric": ["Win Rate", "Avg Pips/Trade", "Profit Factor", "Gross Profit", "Gross Loss", "Open Trades"],
            "Value": [
                f"{stats['win_rate']}%",
                f"{stats['avg_pips_per_trade']} pips",
                f"{stats['profit_factor']}",
                f"{stats['gross_profit']} pips",
                f"-{stats['gross_loss']} pips",
                f"{stats['open']}"
            ]
        }
        st.table(pd.DataFrame(metrics_data))
    
    with col_b:
        # Win/Loss pie chart
        st.subheader(" Win/Loss Distribution")
        if stats["wins"] + stats["losses"] > 0:
            chart_data = pd.DataFrame({
                "Outcome": ["Wins", "Losses", "Open"],
                "Count": [stats["wins"], stats["losses"], stats["open"]]
            })
            st.bar_chart(chart_data.set_index("Outcome"))
    
    st.divider()
    
    # Signals with outcomes
    if stats["signals_with_outcomes"]:
        st.subheader(" Signals with Outcomes")
        signals_df = pd.DataFrame(stats["signals_with_outcomes"])
        
        display_cols = ["instrument", "direction", "entry", "sl", "tp", "units",
                       "confidence", "adx", "atr_ratio", "rsi", "outcome", "pnl_pips", "win_prob"]
        display_cols = [c for c in display_cols if c in signals_df.columns]
        
        styled_df = signals_df[display_cols].style.applymap(
            style_direction, subset=["direction"]
        ).applymap(
            style_outcome, subset=["outcome"] if "outcome" in display_cols else []
        )
        st.dataframe(styled_df, use_container_width=True)
else:
    st.info("No historical scan results. Click 'Scan Historical' to run a backtest.")

# Rejection heatmap
if "hist_rejections" in st.session_state and st.session_state.hist_rejections:
    with st.expander(" Filter Rejection Heatmap"):
        rej_df = build_rejection_heatmap(st.session_state.hist_rejections)
        if not rej_df.empty and rej_df.sum().sum() > 0:
            st.dataframe(rej_df, use_container_width=True)
            st.bar_chart(rej_df.sum())# ────────────────────────────────────────────────
# TAB 3: All Signals
# ────────────────────────────────────────────────
with tab3:
    st.header("All Saved Signals")# Filter options
status_filter = st.selectbox("Filter by Status", ["All", "active", "archived", "closed"])

if status_filter == "All":
    df = load_signals()
else:
    df = load_signals(status_filter)

if df.empty:
    st.info("No signals found.")
else:
    # Add market context
    if st.checkbox("Show Market Context", value=False, key="all_signals_context"):
        with st.spinner("Fetching current prices..."):
            df = add_market_context(df)
    
    # Style and display
    display_cols = ["instrument", "direction", "entry", "sl", "tp", "units", "units_formatted",
                   "confidence", "adx", "atr_ratio", "rsi", "status", "open_time"]
    
    if "current_price" in df.columns:
        display_cols.extend(["current_price", "pips_to_sl", "pips_to_tp", "status_emoji"])
    
    display_cols = [c for c in display_cols if c in df.columns]
    
    styled_df = df[display_cols].style.applymap(
        style_direction, subset=["direction"]
    )
    st.dataframe(styled_df, use_container_width=True)
    
    st.caption(f"Total: {len(df)} signals")# ────────────────────────────────────────────────
# TAB 4: Stats
# ────────────────────────────────────────────────
with tab4:
    st.header(" Statistics & History")col1, col2 = st.columns(2)

with col1:
    st.subheader("Signal Summary")
    all_signals = load_signals()
    if not all_signals.empty:
        st.metric("Total Signals", len(all_signals))
        st.metric("Active", len(all_signals[all_signals["status"] == "active"]))
        st.metric("Archived", len(all_signals[all_signals["status"] == "archived"]))
        
        # Direction breakdown
        st.divider()
        st.write("**By Direction:**")
        direction_counts = all_signals["direction"].value_counts()
        st.bar_chart(direction_counts)
        
        # By instrument
        st.write("**By Instrument:**")
        instrument_counts = all_signals["instrument"].value_counts()
        st.bar_chart(instrument_counts)

with col2:
    st.subheader("Backtest History")
    bt_history = load_backtest_history()
    if not bt_history.empty:
        st.dataframe(bt_history[["run_time", "lookback_days", "total_signals", 
                                 "win_rate", "total_pips", "profit_factor"]], 
                    use_container_width=True)
    else:
        st.info("No backtest history yet.")

# Position sizing info
st.divider()
st.subheader(" Account & Risk Settings")
cfg = get_config()
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Account Size", f"${cfg.ACCOUNT_SIZE_USD:,.0f}")
with col_b:
    st.metric("Risk per Trade", f"{cfg.RISK_PER_TRADE_PCT * 100:.1f}%")
with col_c:
    st.metric("Risk Amount", f"${cfg.ACCOUNT_SIZE_USD * cfg.RISK_PER_TRADE_PCT:,.0f}")

