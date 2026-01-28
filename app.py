# app.py — Trend Following Scanner (FULL MERGED VERSION - FIXED v2)
# Streamlit + SQLite + OANDA (primary) / yfinance fallback
# Includes:
# - Live scan
# - Historical scan
# - Filter rejection heatmap
#
# v2 Fixes:
# - yfinance interval/period limits with warnings
# - Correct forex position sizing (pip-based)
# - Unique signal IDs with timestamp + random suffix
# - Timezone normalization (all data to UTC)

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Set, Optional, List, Tuple
import logging
import time
import random
from contextlib import contextmanager

import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

# ────────────────────────────────────────────────
# LOGGING
# ────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

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
    # Timeframe settings
    HTF: str = "1d"
    LTF: str = "4h"
    MIN_BARS: int = 50

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


# ────────────────────────────────────────────────
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

# Pip multipliers and values for position sizing
# pip_multiplier: how many price units = 1 pip
# pip_value_per_lot: USD value of 1 pip per standard lot (100k units)
INSTRUMENT_SPECS = {
    "EURUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex"},
    "USDJPY=X": {"pip_multiplier": 100, "pip_value_per_lot": 1000 / 150, "type": "forex"},  # ~6.67 at 150 rate
    "GBPUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex"},
    "AUDUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex"},
    "USDCHF=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex"},  # approximate
    "USDCAD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0 / 1.36, "type": "forex"},  # ~7.35 at 1.36
    "GC=F": {"pip_multiplier": 1, "pip_value_per_lot": 100.0, "type": "commodity"},  # gold: $1 move = $100/contract
}

# yfinance period limits by interval
YFINANCE_LIMITS = {
    "1d": {"max_period": "2y", "max_days": 730},
    "4h": {"max_period": "60d", "max_days": 60},
    "1h": {"max_period": "730d", "max_days": 730},  # often less reliable
    "15m": {"max_period": "60d", "max_days": 60},
}

DB_FILE = "signals.db"

# ────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────

if "config" not in st.session_state:
    st.session_state.config = Config().to_dict()

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

if "oanda_available" not in st.session_state:
    st.session_state.oanda_available = False

if "oanda_api" not in st.session_state:
    try:
        st.session_state.oanda_api = API(
            access_token=st.secrets["OANDA"]["access_token"],
            environment=st.secrets["OANDA"]["environment"]
        )
        st.session_state.oanda_available = True
    except Exception:
        st.session_state.oanda_api = None
        st.session_state.oanda_available = False


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
# TIMEZONE NORMALIZATION
# ────────────────────────────────────────────────

def normalize_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame index to UTC timezone.
    Handles both timezone-aware and naive timestamps.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # If timezone-aware, convert to UTC then remove tz info
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    else:
        # Assume naive timestamps are already UTC (OANDA) or treat as UTC
        # This is a simplification; yfinance may return local exchange time
        pass

    return df


# ────────────────────────────────────────────────
# DATA FETCH
# ────────────────────────────────────────────────

def check_yfinance_limits(interval: str, lookback_days: int) -> Tuple[bool, str]:
    """
    Check if yfinance can provide enough data for the requested lookback.
    Returns (is_ok, warning_message).
    """
    limits = YFINANCE_LIMITS.get(interval, {"max_days": 60})
    max_days = limits["max_days"]

    if lookback_days > max_days:
        return False, f"yfinance only supports ~{max_days} days for {interval} interval. Requested: {lookback_days} days."
    return True, ""


def fetch_data(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLC data for a symbol at a given interval.
    Uses OANDA if available, falls back to yfinance.
    All data is normalized to UTC.
    """
    instrument = INSTRUMENT_MAP.get(symbol)
    df = None

    # Map interval to OANDA granularity and yfinance settings
    interval_map = {
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
            # Handle multi-index columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close"]].copy()
            df.columns = ["open", "high", "low", "close"]
            df = normalize_to_utc(df)
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return None

    return df


@st.cache_data(ttl=300)
def fetch_with_indicators(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Fetch data and compute all indicators in one cached call.
    """
    df = fetch_data(symbol, interval)
    if df is None or df.empty:
        return None
    return compute_indicators(df)


# ────────────────────────────────────────────────
# INDICATORS
# ────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators on a DataFrame.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

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

    # ATR ratio (volatility filter)
    df["atr_ratio"] = (df["atr"] / df["close"]) * 100

    return df


# ────────────────────────────────────────────────
# POSITION SIZING (FIXED)
# ────────────────────────────────────────────────

def calculate_position_size(symbol: str, entry: float, sl: float, risk_usd: float) -> int:
    """
    Calculate position size in units based on:
    - Symbol specs (pip multiplier, pip value)
    - Entry and stop loss prices
    - Risk amount in USD

    Returns units (for OANDA) or contracts.
    """
    specs = INSTRUMENT_SPECS.get(symbol)

    if specs is None:
        # Fallback: assume forex-like with 10000 pip multiplier
        logger.warning(f"No specs for {symbol}, using default forex sizing")
        specs = {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex"}

    sl_distance = abs(entry - sl)

    if sl_distance == 0:
        return 0

    if specs["type"] == "forex":
        # Convert price distance to pips
        sl_pips = sl_distance * specs["pip_multiplier"]

        if sl_pips == 0:
            return 0

        # pip_value_per_lot is USD per pip per 100,000 units (1 standard lot)
        # units = (risk_usd / (sl_pips * pip_value_per_pip_per_unit))
        # pip_value_per_unit = pip_value_per_lot / 100000
        pip_value_per_unit = specs["pip_value_per_lot"] / 100000

        units = int(risk_usd / (sl_pips * pip_value_per_unit))

        return units

    elif specs["type"] == "commodity":
        # For gold (XAU/USD): $1 move = $100 per contract
        # sl_distance is in dollars (e.g., $20)
        # Risk per contract = sl_distance * pip_value_per_lot
        risk_per_contract = sl_distance * specs["pip_value_per_lot"]

        if risk_per_contract == 0:
            return 0

        contracts = int(risk_usd / risk_per_contract)
        return max(contracts, 0)

    return 0


def format_position_size(symbol: str, units: int) -> str:
    """
    Format position size for display (lots, mini-lots, micro-lots, or contracts).
    """
    specs = INSTRUMENT_SPECS.get(symbol, {"type": "forex"})

    if specs["type"] == "commodity":
        return f"{units} contract(s)"

    # Forex: convert units to lot notation
    if units >= 100000:
        lots = units / 100000
        return f"{lots:.2f} standard lot(s) ({units:,} units)"
    elif units >= 10000:
        mini_lots = units / 10000
        return f"{mini_lots:.1f} mini lot(s) ({units:,} units)"
    elif units >= 1000:
        micro_lots = units / 1000
        return f"{micro_lots:.1f} micro lot(s) ({units:,} units)"
    else:
        return f"{units:,} units"


# ────────────────────────────────────────────────
# SIGNAL ID GENERATION (FIXED)
# ────────────────────────────────────────────────

def generate_signal_id(symbol: str, direction: str, timestamp: pd.Timestamp) -> str:
    """
    Generate a unique signal ID using:
    - Symbol
    - Direction
    - Timestamp (to the second)
    - Random suffix for uniqueness
    """
    ts_str = timestamp.strftime('%Y%m%d_%H%M%S')
    random_suffix = f"{int(time.time() * 1000) % 10000:04d}_{random.randint(1000, 9999)}"
    return f"{symbol}_{direction}_{ts_str}_{random_suffix}"


# ────────────────────────────────────────────────
# SIGNAL GENERATION
# ────────────────────────────────────────────────

def generate_signal(htf: pd.DataFrame, ltf: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    Generate a trading signal based on HTF trend alignment and LTF entry.
    Returns a signal dict or None.
    """
    cfg = get_config()

    if htf.empty or ltf.empty:
        return None

    htf_last = htf.iloc[-1]
    ltf_last = ltf.iloc[-1]

    # Determine HTF trend
    htf_trend = htf_last.get("trend", "neutral")
    if htf_trend == "neutral":
        return None

    # Check ADX filter
    adx_val = ltf_last.get("adx", 0)
    if pd.isna(adx_val) or adx_val < cfg.MIN_ADX:
        return None

    # Check ATR ratio filter
    atr_ratio = ltf_last.get("atr_ratio", 0)
    if pd.isna(atr_ratio) or atr_ratio < cfg.MIN_ATR_RATIO_PCT:
        return None

    # Determine direction
    direction = None
    if htf_trend == "bullish":
        # DI confirmation for long
        if cfg.ENABLE_DI_CONFIRM:
            if ltf_last.get("plus_di", 0) <= ltf_last.get("minus_di", 0):
                return None
        # RSI filter (not overbought)
        if cfg.ENABLE_RSI_FILTER:
            if ltf_last.get("rsi", 50) > 70:
                return None
        direction = "LONG"

    elif htf_trend == "bearish":
        # DI confirmation for short
        if cfg.ENABLE_DI_CONFIRM:
            if ltf_last.get("minus_di", 0) <= ltf_last.get("plus_di", 0):
                return None
        # RSI filter (not oversold)
        if cfg.ENABLE_RSI_FILTER:
            if ltf_last.get("rsi", 50) < 30:
                return None
        direction = "SHORT"

    if direction is None:
        return None

    # Pullback filter
    if cfg.ENABLE_PULLBACK_FILTER:
        atr = ltf_last.get("atr", 0)
        ema_slow = ltf_last.get("ema_slow", ltf_last["close"])
        pullback_dist = abs(ltf_last["close"] - ema_slow)
        if atr > 0 and pullback_dist > cfg.PULLBACK_ATR_MAX * atr:
            return None

    # Calculate entry, SL, TP
    entry = ltf_last["close"]
    atr = ltf_last.get("atr", entry * 0.01)  # fallback to 1% if no ATR

    if direction == "LONG":
        sl = entry - (cfg.ATR_SL_MULT * atr)
        tp = entry + (cfg.ATR_SL_MULT * atr * cfg.TARGET_RR)
    else:
        sl = entry + (cfg.ATR_SL_MULT * atr)
        tp = entry - (cfg.ATR_SL_MULT * atr * cfg.TARGET_RR)

    # Position sizing (FIXED)
    risk_amount = cfg.ACCOUNT_SIZE_USD * cfg.RISK_PER_TRADE_PCT
    units = calculate_position_size(symbol, entry, sl, risk_amount)

    # Confidence score (simple heuristic)
    confidence = min(100, (adx_val / cfg.MIN_ADX) * 50 + (atr_ratio / cfg.MIN_ATR_RATIO_PCT) * 25)

    # Generate unique signal ID (FIXED)
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
    }


# ────────────────────────────────────────────────
# LIVE SCAN
# ────────────────────────────────────────────────

def run_scan() -> List[Dict]:
    """
    Run a live scan across all pairs.
    """
    cfg = get_config()
    results = []

    for pair in PAIRS:
        try:
            htf = fetch_data(pair, cfg.HTF)
            ltf = fetch_data(pair, cfg.LTF)

            # Hard guards
            if htf is None or ltf is None:
                logger.warning(f"{pair}: data fetch returned None")
                continue

            if htf.empty or ltf.empty:
                logger.warning(f"{pair}: empty dataframe (htf={len(htf)}, ltf={len(ltf)})")
                continue

            if len(htf) < cfg.MIN_BARS or len(ltf) < cfg.MIN_BARS:
                logger.warning(f"{pair}: not enough bars")
                continue

            htf = compute_indicators(htf)
            ltf = compute_indicators(ltf)

            # Guard again AFTER indicators (NaNs)
            htf = htf.dropna()
            ltf = ltf.dropna()

            if ltf.empty:
                logger.warning(f"{pair}: indicators produced NaNs only")
                continue

            signal = generate_signal(htf, ltf, pair)

            if signal:
                results.append(signal)
                save_signal(signal)

        except Exception as e:
            logger.exception(f"{pair}: scan failed → {e}")

    return results


# ────────────────────────────────────────────────
# HISTORICAL SCAN + HEATMAP
# ────────────────────────────────────────────────

def run_historical_scan(lookback_days: int) -> Tuple[List[Dict], Dict, List[str]]:
    """
    Run a historical scan to backtest signals over the lookback period.
    Returns (signals, rejections, warnings) tuple.
    """
    cfg = get_config()
    signals = []
    rejections = {}
    warnings = []

    # Check yfinance limits if OANDA not available
    if not st.session_state.oanda_available:
        is_ok, warning = check_yfinance_limits(cfg.LTF, lookback_days)
        if not is_ok:
            warnings.append(f"⚠️ {warning}")
            warnings.append("Consider enabling OANDA API for longer historical scans.")

    for pair in PAIRS:
        rejections[pair] = {}

        try:
            htf = fetch_data(pair, cfg.HTF)
            ltf = fetch_data(pair, cfg.LTF)

            # Hard data guards
            if htf is None or ltf is None:
                rejections[pair]["no_data"] = rejections[pair].get("no_data", 0) + 1
                continue

            if htf.empty or ltf.empty:
                rejections[pair]["empty_df"] = rejections[pair].get("empty_df", 0) + 1
                continue

            # Ensure datetime index (already normalized to UTC in fetch_data)
            if not isinstance(ltf.index, pd.DatetimeIndex):
                ltf.index = pd.to_datetime(ltf.index, errors="coerce")
            if not isinstance(htf.index, pd.DatetimeIndex):
                htf.index = pd.to_datetime(htf.index, errors="coerce")

            ltf = ltf.sort_index()
            htf = htf.sort_index()

            if ltf.index.isna().all():
                rejections[pair]["bad_index"] = rejections[pair].get("bad_index", 0) + 1
                continue

            if len(ltf) < cfg.MIN_BARS or len(htf) < cfg.MIN_BARS:
                rejections[pair]["not_enough_bars"] = rejections[pair].get("not_enough_bars", 0) + 1
                continue

            htf = compute_indicators(htf)
            ltf = compute_indicators(ltf)

            # Drop NaNs AFTER indicators
            ltf = ltf.dropna()
            htf = htf.dropna()

            if ltf.empty:
                rejections[pair]["nan_after_indicators"] = rejections[pair].get("nan_after_indicators", 0) + 1
                continue

            # Safe lookback window
            last_ts = ltf.index.max()
            cutoff = last_ts - pd.Timedelta(days=lookback_days)
            recent = ltf[ltf.index >= cutoff]

            if recent.empty:
                rejections[pair]["no_recent_data"] = rejections[pair].get("no_recent_data", 0) + 1
                continue

            # Iterate through historical bars
            for i in range(len(recent)):
                sub_ltf = recent.iloc[: i + 1].copy()
                sub_htf = htf[htf.index <= sub_ltf.index[-1]].copy()

                if sub_htf.empty:
                    rejections[pair]["no_htf_alignment"] = rejections[pair].get("no_htf_alignment", 0) + 1
                    continue

                signal = generate_signal(sub_htf, sub_ltf, pair)
                if signal:
                    signals.append(signal)

        except Exception as e:
            rejections[pair]["exception"] = rejections[pair].get("exception", 0) + 1
            logger.exception(f"{pair}: historical scan failed → {e}")

    return signals, rejections, warnings


def build_rejection_heatmap(rejections: Dict) -> pd.DataFrame:
    """
    Build a DataFrame suitable for heatmap visualization from rejections dict.
    """
    # Collect all unique rejection reasons
    all_reasons = set()
    for pair_rej in rejections.values():
        all_reasons.update(pair_rej.keys())

    # Build matrix
    data = []
    for pair, rej_dict in rejections.items():
        row = {"pair": pair}
        for reason in all_reasons:
            row[reason] = rej_dict.get(reason, 0)
        data.append(row)

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.set_index("pair")
    return df


# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────

st.set_page_config("Trend Scanner", layout="wide")
st.title("📈 Trend Following Scanner")

# Status indicators
col_status1, col_status2 = st.columns(2)
with col_status1:
    if st.session_state.oanda_available:
        st.success("✅ OANDA API Connected")
    else:
        st.warning("⚠️ OANDA not available — using yfinance (limited history)")

with col_status2:
    st.write("**Pairs:**", ", ".join(PAIRS))

with st.sidebar:
    st.header("⚙️ Configuration")

    cfg = get_config()

    # Settings expander
    with st.expander("Scan Settings", expanded=False):
        min_adx = st.slider("Min ADX", 10.0, 40.0, cfg.MIN_ADX, 1.0)
        min_atr = st.slider("Min ATR Ratio %", 0.1, 1.0, cfg.MIN_ATR_RATIO_PCT, 0.05)
        atr_sl = st.slider("ATR SL Multiplier", 1.0, 3.0, cfg.ATR_SL_MULT, 0.1)
        target_rr = st.slider("Target R:R", 1.0, 5.0, cfg.TARGET_RR, 0.5)

        di_confirm = st.checkbox("Enable DI Confirmation", cfg.ENABLE_DI_CONFIRM)
        rsi_filter = st.checkbox("Enable RSI Filter", cfg.ENABLE_RSI_FILTER)
        pullback_filter = st.checkbox("Enable Pullback Filter", cfg.ENABLE_PULLBACK_FILTER)

        if st.button("Update Settings"):
            update_config(
                MIN_ADX=min_adx,
                MIN_ATR_RATIO_PCT=min_atr,
                ATR_SL_MULT=atr_sl,
                TARGET_RR=target_rr,
                ENABLE_DI_CONFIRM=di_confirm,
                ENABLE_RSI_FILTER=rsi_filter,
                ENABLE_PULLBACK_FILTER=pullback_filter,
            )
            st.success("Settings updated!")

    with st.expander("Account Settings", expanded=False):
        account_size = st.number_input("Account Size (USD)", 1000, 1000000, int(cfg.ACCOUNT_SIZE_USD), 1000)
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, cfg.RISK_PER_TRADE_PCT * 100, 0.5) / 100

        if st.button("Update Account"):
            update_config(
                ACCOUNT_SIZE_USD=account_size,
                RISK_PER_TRADE_PCT=risk_pct,
            )
            st.success("Account settings updated!")

    st.divider()

    # Live scan
    st.subheader("🔴 Live Scan")
    if st.button("Run Scan Now", type="primary"):
        with st.spinner("Scanning markets..."):
            res = run_scan()
            st.session_state.last_scan = res
            if res:
                st.success(f"✅ {len(res)} new signal(s) found!")
            else:
                st.info("No signals found matching criteria")

    st.divider()

    # Historical scan
    st.subheader("📊 Historical Scan")

    # Show limit warning if no OANDA
    if not st.session_state.oanda_available:
        max_days = YFINANCE_LIMITS.get(cfg.LTF, {}).get("max_days", 60)
        st.caption(f"⚠️ yfinance limit for {cfg.LTF}: ~{max_days} days")
        lookback = st.slider("Lookback days", 1, min(14, max_days), 3)
    else:
        lookback = st.slider("Lookback days", 1, 30, 3)

    if st.button("Scan Historical"):
        with st.spinner(f"Scanning last {lookback} days..."):
            sigs, rej, warns = run_historical_scan(lookback)

            # Show warnings
            for w in warns:
                st.warning(w)

            if sigs:
                st.success(f"Found {len(sigs)} historical signals")
                st.session_state.hist_signals = sigs
            else:
                st.info("No historical signals found")

            st.session_state.hist_rejections = rej

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Saved Signals")
    df = load_signals()
    if df.empty:
        st.info("No saved signals yet. Run a scan to generate signals.")
    else:
        st.dataframe(df, use_container_width=True)

    # Show last scan results
    if st.session_state.last_scan:
        st.subheader("🔵 Latest Scan Results")
        scan_df = pd.DataFrame(st.session_state.last_scan)
        st.dataframe(scan_df, use_container_width=True)

    # Show historical signals
    if "hist_signals" in st.session_state and st.session_state.hist_signals:
        st.subheader("📈 Historical Signals")
        hist_df = pd.DataFrame(st.session_state.hist_signals)
        st.dataframe(hist_df, use_container_width=True)

with col2:
    # Rejection heatmap
    if "hist_rejections" in st.session_state and st.session_state.hist_rejections:
        st.subheader("🔥 Filter Rejection Heatmap")
        rej_df = build_rejection_heatmap(st.session_state.hist_rejections)
        if not rej_df.empty and rej_df.sum().sum() > 0:
            st.dataframe(rej_df, use_container_width=True)

            # Bar chart of total rejections by reason
            totals = rej_df.sum()
            if totals.sum() > 0:
                st.bar_chart(totals)
        else:
            st.info("No rejections recorded")

    # Quick stats
    st.subheader("📊 Quick Stats")
    saved_df = load_signals()
    if not saved_df.empty:
        st.metric("Total Signals", len(saved_df))
        st.metric("Long Signals", len(saved_df[saved_df["direction"] == "LONG"]))
        st.metric("Short Signals", len(saved_df[saved_df["direction"] == "SHORT"]))
    else:
        st.info("No data yet")

    # Position sizing info
    st.subheader("💰 Position Sizing")
    cfg = get_config()
    st.caption(f"Account: ${cfg.ACCOUNT_SIZE_USD:,.0f}")
    st.caption(f"Risk/Trade: {cfg.RISK_PER_TRADE_PCT * 100:.1f}% (${cfg.ACCOUNT_SIZE_USD * cfg.RISK_PER_TRADE_PCT:,.0f})")
