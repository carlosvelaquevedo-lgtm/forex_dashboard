# app.py — Trend Following Scanner (FULL MERGED VERSION - v4)
# Streamlit + SQLite + OANDA (primary) / yfinance fallback
#
# v4 Features:
# - Backtest realism: spread + slippage simulation
# - Max hold days to prevent look-ahead bias
# - Outcome persistence with "Update Outcomes" button
# - Improved current price reliability (5m fallback)
# - Calibrated win probability (capped, realistic)
# - MAE/MFE tracking for trade analysis
# - Corrected gold pip convention (0.1 = 1 pip)
# - Parallel scanning with ThreadPoolExecutor

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
from oandapyV20.endpoints.pricing import PricingInfo

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
    # Historical scan optimization
    HIST_SCAN_STEP: int = 6
    # v4: Backtest realism
    SPREAD_PIPS: float = 1.5  # Typical spread in pips
    SLIPPAGE_PIPS: float = 0.5  # Typical slippage in pips
    MAX_HOLD_DAYS: int = 30  # Max days to hold a trade in backtest

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

REVERSE_INSTRUMENT_MAP = {v: k for k, v in INSTRUMENT_MAP.items()}

# v4: Corrected pip multipliers
# Gold: 0.1 move = 1 pip (industry standard), so multiplier = 10
# JPY pairs: 0.01 move = 1 pip, so multiplier = 100
# Other forex: 0.0001 move = 1 pip, so multiplier = 10000
INSTRUMENT_SPECS = {
    "EURUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5, "spread_pips": 1.0},
    "USDJPY=X": {"pip_multiplier": 100, "pip_value_per_lot": 1000 / 150, "type": "forex", "decimals": 3, "spread_pips": 1.2},
    "GBPUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5, "spread_pips": 1.5},
    "AUDUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5, "spread_pips": 1.2},
    "USDCHF=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5, "spread_pips": 1.5},
    "USDCAD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0 / 1.36, "type": "forex", "decimals": 5, "spread_pips": 1.5},
    # v4: Gold corrected - 0.10 move = 1 pip (10 pips per $1 move)
    "GC=F": {"pip_multiplier": 10, "pip_value_per_lot": 10.0, "type": "commodity", "decimals": 2, "spread_pips": 3.0},
}

# yfinance period limits by interval
YFINANCE_LIMITS = {
    "1d": {"max_period": "2y", "max_days": 730},
    "4h": {"max_period": "60d", "max_days": 60},
    "1h": {"max_period": "730d", "max_days": 730},
    "15m": {"max_period": "60d", "max_days": 60},
    "5m": {"max_period": "60d", "max_days": 60},
}

DB_FILE = "signals.db"

# ────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────

if "config" not in st.session_state:
    st.session_state.config = Config().to_dict()

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None

if "last_scan_duration" not in st.session_state:
    st.session_state.last_scan_duration = None

if "oanda_available" not in st.session_state:
    st.session_state.oanda_available = False

if "oanda_account_id" not in st.session_state:
    st.session_state.oanda_account_id = None

if "oanda_api" not in st.session_state:
    try:
        st.session_state.oanda_api = API(
            access_token=st.secrets["OANDA"]["access_token"],
            environment=st.secrets["OANDA"]["environment"]
        )
        st.session_state.oanda_available = True
        st.session_state.oanda_account_id = st.secrets["OANDA"].get("account_id")
    except Exception:
        st.session_state.oanda_api = None
        st.session_state.oanda_available = False


def get_config() -> Config:
    return Config.from_dict(st.session_state.config)


def update_config(**kwargs):
    st.session_state.config.update(kwargs)


# ────────────────────────────────────────────────
# DATABASE (EXPANDED SCHEMA v4)
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
            units_formatted TEXT,
            open_time TEXT,
            confidence REAL,
            adx REAL,
            atr_ratio REAL,
            rsi REAL,
            win_prob REAL,
            status TEXT DEFAULT 'active',
            outcome TEXT,
            close_price REAL,
            close_time TEXT,
            pnl_pips REAL,
            mae_pips REAL,
            mfe_pips REAL,
            hold_bars INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
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
            avg_mae REAL,
            avg_mfe REAL,
            profit_factor REAL,
            config_json TEXT
        )
        """)
        
        conn.commit()
        _migrate_db(conn)


def _migrate_db(conn):
    """Add new columns to existing tables if needed."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(signals)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    
    new_cols = [
        ("units_formatted", "TEXT"),
        ("adx", "REAL"),
        ("atr_ratio", "REAL"),
        ("rsi", "REAL"),
        ("win_prob", "REAL"),
        ("status", "TEXT DEFAULT 'active'"),
        ("outcome", "TEXT"),
        ("close_price", "REAL"),
        ("close_time", "TEXT"),
        ("pnl_pips", "REAL"),
        ("mae_pips", "REAL"),
        ("mfe_pips", "REAL"),
        ("hold_bars", "INTEGER"),
        ("created_at", "TEXT"),
    ]
    
    for col_name, col_type in new_cols:
        if col_name not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass
    
    # Migrate backtest_results table
    cursor.execute("PRAGMA table_info(backtest_results)")
    bt_cols = {row[1] for row in cursor.fetchall()}
    
    bt_new_cols = [("avg_mae", "REAL"), ("avg_mfe", "REAL"), ("profit_factor", "REAL")]
    for col_name, col_type in bt_new_cols:
        if col_name not in bt_cols:
            try:
                conn.execute(f"ALTER TABLE backtest_results ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass
    
    conn.commit()


def load_signals(status_filter: str = None) -> pd.DataFrame:
    init_db()
    with get_db() as conn:
        if status_filter:
            return pd.read_sql(
                "SELECT * FROM signals WHERE status = ? ORDER BY open_time DESC",
                conn, params=(status_filter,)
            )
        return pd.read_sql("SELECT * FROM signals ORDER BY open_time DESC", conn)


def save_signal(sig: Dict):
    with get_db() as conn:
        conn.execute("""
        INSERT OR REPLACE INTO signals
        (id, symbol_raw, instrument, direction, entry, sl, tp, units, units_formatted,
         open_time, confidence, adx, atr_ratio, rsi, win_prob, status, outcome, close_price,
         pnl_pips, mae_pips, mfe_pips, hold_bars, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sig["id"], sig["symbol_raw"], sig["instrument"], sig["direction"],
            sig["entry"], sig["sl"], sig["tp"], sig["units"], sig.get("units_formatted", ""),
            sig["open_time"], sig["confidence"], sig.get("adx"), sig.get("atr_ratio"),
            sig.get("rsi"), sig.get("win_prob"), sig.get("status", "active"), sig.get("outcome"),
            sig.get("close_price"), sig.get("pnl_pips"), sig.get("mae_pips"),
            sig.get("mfe_pips"), sig.get("hold_bars"), datetime.utcnow().isoformat()
        ))
        conn.commit()


def update_signal_outcome(signal_id: str, outcome: str, close_price: float, 
                          pnl_pips: float, mae_pips: float, mfe_pips: float, hold_bars: int):
    """Update signal with backtest outcome."""
    with get_db() as conn:
        status = "closed" if outcome in ["win", "loss"] else "active"
        conn.execute("""
        UPDATE signals 
        SET status = ?, outcome = ?, close_price = ?, close_time = ?, 
            pnl_pips = ?, mae_pips = ?, mfe_pips = ?, hold_bars = ?
        WHERE id = ?
        """, (status, outcome, close_price, datetime.utcnow().isoformat(), 
              pnl_pips, mae_pips, mfe_pips, hold_bars, signal_id))
        conn.commit()


def archive_old_signals(max_age_days: int) -> int:
    cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
    with get_db() as conn:
        cursor = conn.execute("""
        UPDATE signals SET status = 'archived' 
        WHERE status = 'active' AND open_time < ?
        """, (cutoff,))
        conn.commit()
        return cursor.rowcount


def delete_archived_signals() -> int:
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM signals WHERE status = 'archived'")
        conn.commit()
        return cursor.rowcount


def save_backtest_result(result: Dict):
    with get_db() as conn:
        conn.execute("""
        INSERT INTO backtest_results 
        (run_time, lookback_days, total_signals, wins, losses, win_rate, avg_rr, 
         total_pips, avg_mae, avg_mfe, profit_factor, config_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.utcnow().isoformat(),
            result["lookback_days"],
            result["total_signals"],
            result["wins"],
            result["losses"],
            result["win_rate"],
            result["avg_rr"],
            result["total_pips"],
            result.get("avg_mae", 0),
            result.get("avg_mfe", 0),
            result.get("profit_factor", 0),
            str(result.get("config", {}))
        ))
        conn.commit()


def load_backtest_history() -> pd.DataFrame:
    init_db()
    with get_db() as conn:
        return pd.read_sql("SELECT * FROM backtest_results ORDER BY run_time DESC LIMIT 20", conn)


# ────────────────────────────────────────────────
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
    return df


# ────────────────────────────────────────────────
# DATA FETCH (v4: with 5m option for pricing)
# ────────────────────────────────────────────────

def check_yfinance_limits(interval: str, lookback_days: int) -> Tuple[bool, str]:
    limits = YFINANCE_LIMITS.get(interval, {"max_days": 60})
    max_days = limits["max_days"]
    if lookback_days > max_days:
        return False, f"yfinance only supports ~{max_days} days for {interval} interval."
    return True, ""


def fetch_data(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch OHLC data with OANDA primary, yfinance fallback."""
    instrument = INSTRUMENT_MAP.get(symbol)
    df = None

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
                logger.info(f"OANDA: {symbol} {interval} → {len(df)} bars")
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
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close"]].copy()
            df.columns = ["open", "high", "low", "close"]
            df = normalize_to_utc(df)
            logger.info(f"yfinance: {symbol} {interval} → {len(df)} bars")
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return None

    return df


def test_oanda_connection() -> Dict:
    """Test OANDA connection and return diagnostic info."""
    results = {
        "connected": st.session_state.oanda_available,
        "api_object": st.session_state.oanda_api is not None,
        "account_id": st.session_state.oanda_account_id,
        "test_results": []
    }
    
    if not st.session_state.oanda_api:
        results["error"] = "No API object"
        return results
    
    # Test with EUR_USD
    test_instrument = "EUR_USD"
    try:
        params = {"count": 10, "granularity": "H4", "price": "M"}
        r = InstrumentsCandles(instrument=test_instrument, params=params)
        resp = st.session_state.oanda_api.request(r)
        candles = resp.get("candles", [])
        results["test_results"].append({
            "instrument": test_instrument,
            "candles_received": len(candles),
            "status": "✅ OK" if candles else "❌ No data"
        })
        if candles:
            results["sample_candle"] = candles[0]
    except Exception as e:
        results["test_results"].append({
            "instrument": test_instrument,
            "status": f"❌ Error: {str(e)}"
        })
        results["error"] = str(e)
    
    return results


def get_current_price(symbol: str) -> Optional[float]:
    """Fetch current price - prefers OANDA pricing, then 5m candle, then 1h."""
    instrument = INSTRUMENT_MAP.get(symbol)
    
    # Try OANDA pricing endpoint
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
        except Exception:
            pass
    
    # v4: Try 5m candle for more recent price
    try:
        df = fetch_data(symbol, "5m")
        if df is not None and not df.empty:
            return df["close"].iloc[-1]
    except Exception:
        pass
    
    # Fallback to 1h
    try:
        df = fetch_data(symbol, "1h")
        if df is not None and not df.empty:
            return df["close"].iloc[-1]
    except Exception:
        pass
    
    return None


def get_current_prices_batch(symbols: List[str]) -> Dict[str, float]:
    """Fetch current prices for multiple symbols."""
    prices = {}
    
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
        except Exception:
            pass
    
    for symbol in symbols:
        if symbol not in prices:
            price = get_current_price(symbol)
            if price:
                prices[symbol] = price
    
    return prices


# ────────────────────────────────────────────────
# INDICATORS
# ────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    for span, name in [(9, "ema_fast"), (21, "ema_slow"), (50, "ema_tf"), (200, "ema_ts")]:
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
    df["plus_di"] = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_s
    df["minus_di"] = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_s
    dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"])
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["trend"] = np.where(df["ema_fast"] > df["ema_slow"], "bullish",
                           np.where(df["ema_fast"] < df["ema_slow"], "bearish", "neutral"))
    df["atr_ratio"] = (df["atr"] / df["close"]) * 100

    return df


# ────────────────────────────────────────────────
# POSITION SIZING
# ────────────────────────────────────────────────

def calculate_position_size(symbol: str, entry: float, sl: float, risk_usd: float) -> int:
    specs = INSTRUMENT_SPECS.get(symbol)
    if specs is None:
        specs = {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex"}

    sl_distance = abs(entry - sl)
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
        # v4: Gold corrected - pip_multiplier=10 means 0.1 move = 1 pip
        sl_pips = sl_distance * specs["pip_multiplier"]
        if sl_pips == 0:
            return 0
        # For gold: $1 per pip per 1oz, standard lot = 100oz
        pip_value_per_unit = specs["pip_value_per_lot"] / 100
        units = int(risk_usd / (sl_pips * pip_value_per_unit))
        return max(units, 0)

    return 0


def format_position_size(symbol: str, units: int) -> str:
    specs = INSTRUMENT_SPECS.get(symbol, {"type": "forex"})
    if specs["type"] == "commodity":
        if units >= 100:
            return f"{units / 100:.2f} lot ({units} oz)"
        return f"{units} oz"
    if units >= 100000:
        return f"{units / 100000:.2f} std ({units:,})"
    elif units >= 10000:
        return f"{units / 10000:.1f} mini ({units:,})"
    elif units >= 1000:
        return f"{units / 1000:.1f} micro ({units:,})"
    return f"{units:,} units"


def price_to_pips(symbol: str, price_distance: float) -> float:
    """Convert price distance to pips."""
    specs = INSTRUMENT_SPECS.get(symbol, {"pip_multiplier": 10000})
    return price_distance * specs["pip_multiplier"]


def pips_to_price(symbol: str, pips: float) -> float:
    """Convert pips to price distance."""
    specs = INSTRUMENT_SPECS.get(symbol, {"pip_multiplier": 10000})
    return pips / specs["pip_multiplier"]


# ────────────────────────────────────────────────
# SIGNAL ID GENERATION
# ────────────────────────────────────────────────

def generate_signal_id(symbol: str, direction: str, timestamp: pd.Timestamp) -> str:
    ts_str = timestamp.strftime('%Y%m%d_%H%M%S')
    random_suffix = f"{int(time.time() * 1000) % 10000:04d}_{random.randint(1000, 9999)}"
    return f"{symbol}_{direction}_{ts_str}_{random_suffix}"


# ────────────────────────────────────────────────
# SIGNAL GENERATION
# ────────────────────────────────────────────────

def generate_signal(htf: pd.DataFrame, ltf: pd.DataFrame, symbol: str) -> Optional[Dict]:
    cfg = get_config()

    if htf.empty or ltf.empty:
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
    }


def generate_signal_with_reason(htf: pd.DataFrame, ltf: pd.DataFrame, symbol: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Generate signal and return (signal, rejection_reason).
    If signal is generated, rejection_reason is None.
    If rejected, signal is None and rejection_reason explains why.
    """
    cfg = get_config()

    if htf.empty or ltf.empty:
        return None, "empty_data"

    htf_last = htf.iloc[-1]
    ltf_last = ltf.iloc[-1]

    htf_trend = htf_last.get("trend", "neutral")
    if htf_trend == "neutral":
        return None, "neutral_trend"

    adx_val = ltf_last.get("adx", 0)
    if pd.isna(adx_val) or adx_val < cfg.MIN_ADX:
        return None, "low_adx"

    atr_ratio = ltf_last.get("atr_ratio", 0)
    if pd.isna(atr_ratio) or atr_ratio < cfg.MIN_ATR_RATIO_PCT:
        return None, "low_atr_ratio"

    direction = None
    if htf_trend == "bullish":
        if cfg.ENABLE_DI_CONFIRM:
            if ltf_last.get("plus_di", 0) <= ltf_last.get("minus_di", 0):
                return None, "di_confirm_fail"
        if cfg.ENABLE_RSI_FILTER:
            if ltf_last.get("rsi", 50) > 70:
                return None, "rsi_overbought"
        direction = "LONG"
    elif htf_trend == "bearish":
        if cfg.ENABLE_DI_CONFIRM:
            if ltf_last.get("minus_di", 0) <= ltf_last.get("plus_di", 0):
                return None, "di_confirm_fail"
        if cfg.ENABLE_RSI_FILTER:
            if ltf_last.get("rsi", 50) < 30:
                return None, "rsi_oversold"
        direction = "SHORT"

    if direction is None:
        return None, "no_direction"

    if cfg.ENABLE_PULLBACK_FILTER:
        atr = ltf_last.get("atr", 0)
        ema_slow = ltf_last.get("ema_slow", ltf_last["close"])
        pullback_dist = abs(ltf_last["close"] - ema_slow)
        if atr > 0 and pullback_dist > cfg.PULLBACK_ATR_MAX * atr:
            return None, "pullback_filter_fail"

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

    signal = {
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
    }
    
    return signal, None


# ────────────────────────────────────────────────
# MARKET CONTEXT
# ────────────────────────────────────────────────

def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.copy()
    symbols = df["symbol_raw"].unique().tolist()
    current_prices = get_current_prices_batch(symbols)
    df["current_price"] = df["symbol_raw"].map(current_prices)
    
    def calc_context(row):
        if pd.isna(row.get("current_price")):
            return pd.Series({
                "pct_to_sl": None, "pct_to_tp": None,
                "pips_to_sl": None, "pips_to_tp": None, "status_emoji": "❓"
            })
        
        current = row["current_price"]
        entry = row["entry"]
        sl = row["sl"]
        tp = row["tp"]
        direction = row["direction"]
        symbol = row["symbol_raw"]
        
        if direction == "LONG":
            dist_to_sl = current - sl
            dist_to_tp = tp - current
        else:
            dist_to_sl = sl - current
            dist_to_tp = current - tp
        
        pct_to_sl = (dist_to_sl / current) * 100 if current else None
        pct_to_tp = (dist_to_tp / current) * 100 if current else None
        pips_to_sl = price_to_pips(symbol, dist_to_sl)
        pips_to_tp = price_to_pips(symbol, dist_to_tp)
        
        if dist_to_sl <= 0:
            status = "🔴"
        elif dist_to_tp <= 0:
            status = "🟢"
        elif (direction == "LONG" and current > entry) or (direction == "SHORT" and current < entry):
            status = "📈"
        else:
            status = "📉"
        
        return pd.Series({
            "pct_to_sl": round(pct_to_sl, 2) if pct_to_sl else None,
            "pct_to_tp": round(pct_to_tp, 2) if pct_to_tp else None,
            "pips_to_sl": round(pips_to_sl, 1) if pips_to_sl else None,
            "pips_to_tp": round(pips_to_tp, 1) if pips_to_tp else None,
            "status_emoji": status
        })
    
    context_df = df.apply(calc_context, axis=1)
    return pd.concat([df, context_df], axis=1)


# ────────────────────────────────────────────────
# BACKTEST ENGINE (v4: Realistic)
# ────────────────────────────────────────────────

def backtest_signal(signal: Dict, future_data: pd.DataFrame, 
                    spread_pips: float = 1.5, slippage_pips: float = 0.5,
                    max_hold_bars: int = None) -> Dict:
    """
    v4: Realistic backtest with spread, slippage, MAE/MFE tracking.
    """
    if future_data.empty:
        return {
            "outcome": "open", "exit_price": None, "pnl_pips": 0,
            "mae_pips": 0, "mfe_pips": 0, "hold_bars": 0
        }
    
    entry = signal["entry"]
    sl = signal["sl"]
    tp = signal["tp"]
    direction = signal["direction"]
    symbol = signal["symbol_raw"]
    
    # v4: Get instrument-specific spread if available
    specs = INSTRUMENT_SPECS.get(symbol, {})
    actual_spread = specs.get("spread_pips", spread_pips)
    total_cost_pips = actual_spread + slippage_pips
    
    # v4: Adjust SL/TP for spread and slippage
    spread_price = pips_to_price(symbol, total_cost_pips)
    
    if direction == "LONG":
        effective_sl = sl - pips_to_price(symbol, slippage_pips)  # SL hit worse due to slippage
        effective_tp = tp - pips_to_price(symbol, actual_spread)  # TP hit at bid (minus spread)
    else:
        effective_sl = sl + pips_to_price(symbol, slippage_pips)
        effective_tp = tp + pips_to_price(symbol, actual_spread)
    
    # v4: Limit future data to max hold period
    if max_hold_bars and len(future_data) > max_hold_bars:
        future_data = future_data.iloc[:max_hold_bars]
    
    # Track MAE (max adverse excursion) and MFE (max favorable excursion)
    mae = 0  # Worst drawdown in pips
    mfe = 0  # Best profit in pips
    
    for bar_idx, (idx, row) in enumerate(future_data.iterrows()):
        high = row["high"]
        low = row["low"]
        
        # Calculate excursions
        if direction == "LONG":
            current_favorable = high - entry
            current_adverse = entry - low
        else:
            current_favorable = entry - low
            current_adverse = high - entry
        
        mfe = max(mfe, price_to_pips(symbol, current_favorable))
        mae = max(mae, price_to_pips(symbol, current_adverse))
        
        # Check SL/TP hits
        if direction == "LONG":
            if low <= effective_sl:
                pnl_pips = price_to_pips(symbol, effective_sl - entry)
                return {
                    "outcome": "loss", "exit_price": effective_sl, 
                    "pnl_pips": round(pnl_pips, 1),
                    "mae_pips": round(mae, 1), "mfe_pips": round(mfe, 1),
                    "hold_bars": bar_idx + 1
                }
            if high >= effective_tp:
                pnl_pips = price_to_pips(symbol, effective_tp - entry)
                return {
                    "outcome": "win", "exit_price": effective_tp,
                    "pnl_pips": round(pnl_pips, 1),
                    "mae_pips": round(mae, 1), "mfe_pips": round(mfe, 1),
                    "hold_bars": bar_idx + 1
                }
        else:
            if high >= effective_sl:
                pnl_pips = price_to_pips(symbol, entry - effective_sl)
                return {
                    "outcome": "loss", "exit_price": effective_sl,
                    "pnl_pips": round(pnl_pips, 1),
                    "mae_pips": round(mae, 1), "mfe_pips": round(mfe, 1),
                    "hold_bars": bar_idx + 1
                }
            if low <= effective_tp:
                pnl_pips = price_to_pips(symbol, entry - effective_tp)
                return {
                    "outcome": "win", "exit_price": effective_tp,
                    "pnl_pips": round(pnl_pips, 1),
                    "mae_pips": round(mae, 1), "mfe_pips": round(mfe, 1),
                    "hold_bars": bar_idx + 1
                }
    
    # Trade still open - calculate current P&L
    last_close = future_data["close"].iloc[-1]
    if direction == "LONG":
        pnl_pips = price_to_pips(symbol, last_close - entry) - total_cost_pips
    else:
        pnl_pips = price_to_pips(symbol, entry - last_close) - total_cost_pips
    
    return {
        "outcome": "open", "exit_price": last_close,
        "pnl_pips": round(pnl_pips, 1),
        "mae_pips": round(mae, 1), "mfe_pips": round(mfe, 1),
        "hold_bars": len(future_data)
    }


def calculate_backtest_stats(signals: List[Dict], data_cache: Dict[str, pd.DataFrame]) -> Dict:
    """Calculate backtest statistics with MAE/MFE."""
    cfg = get_config()
    
    if not signals:
        return {
            "total_signals": 0, "wins": 0, "losses": 0, "open": 0,
            "win_rate": 0, "avg_rr": 0, "total_pips": 0,
            "avg_pips_per_trade": 0, "profit_factor": 0,
            "avg_mae": 0, "avg_mfe": 0,
            "avg_hold_bars": 0, "signals_with_outcomes": []
        }
    
    wins = losses = open_trades = 0
    total_pips = gross_profit = gross_loss = 0
    total_mae = total_mfe = total_hold_bars = 0
    signals_with_outcomes = []
    
    # Convert max hold days to bars (approximate)
    bars_per_day = {"1d": 1, "4h": 6, "1h": 24}.get(cfg.LTF, 6)
    max_hold_bars = cfg.MAX_HOLD_DAYS * bars_per_day
    
    for signal in signals:
        symbol = signal["symbol_raw"]
        signal_time = pd.to_datetime(signal["open_time"])
        
        if symbol in data_cache:
            full_data = data_cache[symbol]
            future_data = full_data[full_data.index > signal_time]
        else:
            future_data = pd.DataFrame()
        
        result = backtest_signal(
            signal, future_data,
            spread_pips=cfg.SPREAD_PIPS,
            slippage_pips=cfg.SLIPPAGE_PIPS,
            max_hold_bars=max_hold_bars
        )
        
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
        total_mae += result.get("mae_pips", 0)
        total_mfe += result.get("mfe_pips", 0)
        total_hold_bars += result.get("hold_bars", 0)
    
    total_closed = wins + losses
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
    avg_pips = total_pips / len(signals) if signals else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
    avg_mae = total_mae / len(signals) if signals else 0
    avg_mfe = total_mfe / len(signals) if signals else 0
    avg_hold = total_hold_bars / len(signals) if signals else 0
    
    avg_rr = (win_rate / 100 * cfg.TARGET_RR) - ((100 - win_rate) / 100 * 1) if total_closed > 0 else 0
    
    return {
        "total_signals": len(signals),
        "wins": wins, "losses": losses, "open": open_trades,
        "win_rate": round(win_rate, 1),
        "avg_rr": round(avg_rr, 2),
        "total_pips": round(total_pips, 1),
        "avg_pips_per_trade": round(avg_pips, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
        "gross_profit": round(gross_profit, 1),
        "gross_loss": round(gross_loss, 1),
        "avg_mae": round(avg_mae, 1),
        "avg_mfe": round(avg_mfe, 1),
        "avg_hold_bars": round(avg_hold, 1),
        "signals_with_outcomes": signals_with_outcomes
    }


# ────────────────────────────────────────────────
# WIN PROBABILITY (v4: Calibrated)
# ────────────────────────────────────────────────

def estimate_win_probability(signal: Dict, historical_stats: Dict = None) -> float:
    """
    v4: Calibrated win probability - capped at 65% unless very strong setup.
    """
    base_prob = 45.0  # Start conservative
    
    # ADX contribution (max +15%)
    adx = signal.get("adx", 20)
    if adx >= 40:
        base_prob += 15
    elif adx >= 30:
        base_prob += 10
    elif adx >= 25:
        base_prob += 5
    
    # RSI contribution - favor middle range (max +5%)
    rsi = signal.get("rsi", 50)
    if 40 <= rsi <= 60:
        base_prob += 5
    elif rsi < 25 or rsi > 75:
        base_prob -= 5
    
    # Confidence contribution (max +5%)
    confidence = signal.get("confidence", 50)
    if confidence >= 75:
        base_prob += 5
    elif confidence >= 60:
        base_prob += 2
    
    # Historical calibration (weighted blend)
    if historical_stats and historical_stats.get("win_rate", 0) > 0:
        hist_rate = historical_stats["win_rate"]
        # Only boost if historical is strong AND current setup is decent
        if hist_rate >= 55 and adx >= 30:
            base_prob = base_prob * 0.5 + hist_rate * 0.5
        else:
            base_prob = base_prob * 0.7 + hist_rate * 0.3
    
    # v4: Hard cap at 65% unless exceptional setup
    max_prob = 65.0
    if adx >= 40 and historical_stats and historical_stats.get("win_rate", 0) >= 60:
        max_prob = 70.0  # Allow up to 70% for exceptional setups
    
    return max(20, min(max_prob, round(base_prob, 1)))


# ────────────────────────────────────────────────
# LIVE SCAN
# ────────────────────────────────────────────────

def run_scan() -> Tuple[List[Dict], float]:
    """Run live scan - sequential fetching."""
    cfg = get_config()
    results = []
    start_time = time.time()
    
    for pair in PAIRS:
        try:
            htf = fetch_data(pair, cfg.HTF)
            ltf = fetch_data(pair, cfg.LTF)
            signal = _process_pair_for_signal(pair, htf, ltf, cfg)
            if signal:
                results.append(signal)
                save_signal(signal)
        except Exception as e:
            logger.exception(f"{pair}: scan failed → {e}")
    
    duration = time.time() - start_time
    return results, duration


def _process_pair_for_signal(pair: str, htf: pd.DataFrame, ltf: pd.DataFrame, cfg: Config) -> Optional[Dict]:
    """Process a single pair and return signal if valid."""
    if htf is None or ltf is None:
        return None
    if htf.empty or ltf.empty:
        return None
    if len(htf) < cfg.MIN_BARS or len(ltf) < cfg.MIN_BARS:
        return None
    
    htf = compute_indicators(htf)
    ltf = compute_indicators(ltf)
    htf = htf.dropna()
    ltf = ltf.dropna()
    
    if ltf.empty:
        return None
    
    signal = generate_signal(htf, ltf, pair)
    if signal:
        signal["win_prob"] = estimate_win_probability(signal)
    return signal


# ────────────────────────────────────────────────
# HISTORICAL SCAN (v4: Optimized)
# ────────────────────────────────────────────────

def run_historical_scan(lookback_days: int) -> Tuple[List[Dict], Dict, List[str], Dict, float]:
    """Run historical scan with backtest."""
    cfg = get_config()
    signals = []
    rejections = {}
    warnings = []
    data_cache = {}
    start_time = time.time()
    
    # Track signal generation rejections
    filter_rejections = {
        "no_data": 0, "empty_df": 0, "not_enough_bars": 0,
        "nan_after_indicators": 0, "no_recent_data": 0,
        "neutral_trend": 0, "low_adx": 0, "low_atr_ratio": 0,
        "di_confirm_fail": 0, "rsi_filter_fail": 0, "pullback_filter_fail": 0,
        "no_htf_alignment": 0, "exception": 0
    }

    if not st.session_state.oanda_available:
        is_ok, warning = check_yfinance_limits(cfg.LTF, lookback_days)
        if not is_ok:
            warnings.append(f"⚠️ {warning}")

    # Sequential data fetching (parallel disabled - st.session_state not thread-safe)
    pair_data = {}
    for pair in PAIRS:
        pair_data[pair] = (fetch_data(pair, cfg.HTF), fetch_data(pair, cfg.LTF))

    for pair in PAIRS:
        rejections[pair] = {}
        
        try:
            htf, ltf = pair_data.get(pair, (None, None))
            
            if htf is None or ltf is None:
                rejections[pair]["no_data"] = 1
                filter_rejections["no_data"] += 1
                continue
            if htf.empty or ltf.empty:
                rejections[pair]["empty_df"] = 1
                filter_rejections["empty_df"] += 1
                continue

            if not isinstance(ltf.index, pd.DatetimeIndex):
                ltf.index = pd.to_datetime(ltf.index, errors="coerce")
            if not isinstance(htf.index, pd.DatetimeIndex):
                htf.index = pd.to_datetime(htf.index, errors="coerce")

            ltf = ltf.sort_index()
            htf = htf.sort_index()

            if len(ltf) < cfg.MIN_BARS or len(htf) < cfg.MIN_BARS:
                rejections[pair]["not_enough_bars"] = 1
                filter_rejections["not_enough_bars"] += 1
                continue

            htf = compute_indicators(htf)
            ltf = compute_indicators(ltf)
            ltf = ltf.dropna()
            htf = htf.dropna()

            if ltf.empty:
                rejections[pair]["nan_after_indicators"] = 1
                filter_rejections["nan_after_indicators"] += 1
                continue

            data_cache[pair] = ltf

            last_ts = ltf.index.max()
            cutoff = last_ts - pd.Timedelta(days=lookback_days)
            recent = ltf[ltf.index >= cutoff]

            if recent.empty:
                rejections[pair]["no_recent_data"] = 1
                filter_rejections["no_recent_data"] += 1
                continue

            step = cfg.HIST_SCAN_STEP
            indices = list(range(0, len(recent), step))
            if len(recent) - 1 not in indices:
                indices.append(len(recent) - 1)

            bars_checked = 0
            for i in indices:
                bars_checked += 1
                sub_ltf = recent.iloc[: i + 1].copy()
                sub_htf = htf[htf.index <= sub_ltf.index[-1]].copy()

                if sub_htf.empty:
                    filter_rejections["no_htf_alignment"] += 1
                    continue

                signal, rejection_reason = generate_signal_with_reason(sub_htf, sub_ltf, pair)
                if signal:
                    signals.append(signal)
                elif rejection_reason:
                    filter_rejections[rejection_reason] = filter_rejections.get(rejection_reason, 0) + 1
            
            # Store bars checked for debugging
            rejections[pair]["bars_checked"] = bars_checked

        except Exception as e:
            rejections[pair]["exception"] = 1
            filter_rejections["exception"] += 1
            logger.exception(f"{pair}: historical scan failed → {e}")

    # Add filter rejections summary to warnings for visibility
    active_rejections = {k: v for k, v in filter_rejections.items() if v > 0}
    if active_rejections and not signals:
        warnings.append(f"Filter rejections: {active_rejections}")

    backtest_stats = calculate_backtest_stats(signals, data_cache)
    backtest_stats["filter_rejections"] = filter_rejections
    
    for signal in backtest_stats["signals_with_outcomes"]:
        signal["win_prob"] = estimate_win_probability(signal, backtest_stats)
    
    if signals:
        save_backtest_result({
            "lookback_days": lookback_days,
            "total_signals": backtest_stats["total_signals"],
            "wins": backtest_stats["wins"],
            "losses": backtest_stats["losses"],
            "win_rate": backtest_stats["win_rate"],
            "avg_rr": backtest_stats["avg_rr"],
            "total_pips": backtest_stats["total_pips"],
            "avg_mae": backtest_stats.get("avg_mae", 0),
            "avg_mfe": backtest_stats.get("avg_mfe", 0),
            "profit_factor": backtest_stats["profit_factor"] if backtest_stats["profit_factor"] != "∞" else 999,
            "config": cfg.to_dict()
        })

    duration = time.time() - start_time
    return signals, rejections, warnings, backtest_stats, duration


# ────────────────────────────────────────────────
# UPDATE OUTCOMES (v4: New feature)
# ────────────────────────────────────────────────

def update_active_signal_outcomes() -> Tuple[int, int]:
    """
    Re-evaluate active signals against latest data.
    Returns (updated_count, still_open_count).
    """
    cfg = get_config()
    active_signals = load_signals(status_filter="active")
    
    if active_signals.empty:
        return 0, 0
    
    updated = 0
    still_open = 0
    
    # Get max hold bars
    bars_per_day = {"1d": 1, "4h": 6, "1h": 24}.get(cfg.LTF, 6)
    max_hold_bars = cfg.MAX_HOLD_DAYS * bars_per_day
    
    for _, row in active_signals.iterrows():
        symbol = row["symbol_raw"]
        signal_time = pd.to_datetime(row["open_time"])
        
        # Fetch latest data
        ltf = fetch_data(symbol, cfg.LTF)
        if ltf is None or ltf.empty:
            continue
        
        ltf = compute_indicators(ltf)
        ltf = ltf.dropna()
        
        future_data = ltf[ltf.index > signal_time]
        
        signal_dict = row.to_dict()
        result = backtest_signal(
            signal_dict, future_data,
            spread_pips=cfg.SPREAD_PIPS,
            slippage_pips=cfg.SLIPPAGE_PIPS,
            max_hold_bars=max_hold_bars
        )
        
        if result["outcome"] in ["win", "loss"]:
            update_signal_outcome(
                row["id"], result["outcome"], result["exit_price"],
                result["pnl_pips"], result["mae_pips"], result["mfe_pips"],
                result["hold_bars"]
            )
            updated += 1
        else:
            still_open += 1
    
    return updated, still_open


def build_rejection_heatmap(rejections: Dict) -> pd.DataFrame:
    all_reasons = set()
    for pair_rej in rejections.values():
        all_reasons.update(pair_rej.keys())

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


def format_dataframe_decimals(df: pd.DataFrame, decimal_places: int = 2) -> pd.DataFrame:
    """
    Format all numeric columns in a dataframe to specified decimal places.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32', 'float']:
            df[col] = df[col].round(decimal_places)
    
    return df


# Columns that should have 5 decimal places (price-related)
PRICE_COLUMNS = {'entry', 'sl', 'tp', 'current_price', 'close_price', 'exit_price'}

# Columns that should have 1 decimal place (pips, percentages, scores)  
ONE_DECIMAL_COLUMNS = {'pnl_pips', 'mae_pips', 'mfe_pips', 'pips_to_sl', 'pips_to_tp', 
                       'confidence', 'adx', 'rsi', 'win_prob', 'win_rate', 'pct_to_sl', 'pct_to_tp'}


def style_dataframe(df: pd.DataFrame, direction_col: str = "direction", 
                    outcome_col: str = "outcome") -> pd.io.formats.style.Styler:
    """
    Apply standard styling to dataframe:
    - Price columns (entry, sl, tp, current_price): 5 decimal places
    - Pip/score columns: 1 decimal place
    - Other numeric: 2 decimal places
    - Color direction (green LONG / red SHORT)
    - Color outcome (green win / red loss)
    """
    if df.empty:
        return df.style
    
    df = df.copy()
    
    # Apply styling
    styler = df.style
    
    if direction_col in df.columns:
        styler = styler.applymap(style_direction, subset=[direction_col])
    
    if outcome_col in df.columns and outcome_col in df.columns:
        styler = styler.applymap(style_outcome, subset=[outcome_col])
    
    # Build format dictionary with appropriate decimal places per column
    format_dict = {}
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32', 'float']:
            if col in PRICE_COLUMNS:
                format_dict[col] = '{:.5f}'
            elif col in ONE_DECIMAL_COLUMNS:
                format_dict[col] = '{:.1f}'
            else:
                format_dict[col] = '{:.2f}'
    
    if format_dict:
        styler = styler.format(format_dict)
    
    return styler


# ────────────────────────────────────────────────
# STYLING HELPERS
# ────────────────────────────────────────────────

def style_direction(val):
    if val == "LONG":
        return "background-color: #c6efce; color: #006100"
    elif val == "SHORT":
        return "background-color: #ffc7ce; color: #9c0006"
    return ""


def style_outcome(val):
    if val == "win":
        return "background-color: #c6efce; color: #006100"
    elif val == "loss":
        return "background-color: #ffc7ce; color: #9c0006"
    return ""


# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────

st.set_page_config("Trend Scanner v4", layout="wide")
st.title("📈 Trend Following Scanner v4")

init_db()

# Status bar
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    if st.session_state.oanda_available:
        st.success("✅ OANDA")
    else:
        st.warning("⚠️ yfinance")
with col_s2:
    if st.session_state.last_scan_time:
        st.info(f"🕐 {st.session_state.last_scan_time.strftime('%H:%M:%S')}")
with col_s3:
    if st.session_state.last_scan_duration:
        st.info(f"⏱️ {st.session_state.last_scan_duration:.1f}s")
with col_s4:
    cfg = get_config()
    st.caption(f"Spread: {cfg.SPREAD_PIPS} | Slip: {cfg.SLIPPAGE_PIPS} pips")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    cfg = get_config()

    with st.expander("Scan Settings", expanded=False):
        min_adx = st.slider("Min ADX", 10.0, 40.0, cfg.MIN_ADX, 1.0)
        min_atr = st.slider("Min ATR Ratio %", 0.1, 1.0, cfg.MIN_ATR_RATIO_PCT, 0.05)
        atr_sl = st.slider("ATR SL Mult", 1.0, 3.0, cfg.ATR_SL_MULT, 0.1)
        target_rr = st.slider("Target R:R", 1.0, 5.0, cfg.TARGET_RR, 0.5)
        hist_step = st.slider("Hist Scan Step", 1, 12, cfg.HIST_SCAN_STEP, 1)
        
        di_confirm = st.checkbox("DI Confirmation", cfg.ENABLE_DI_CONFIRM)
        rsi_filter = st.checkbox("RSI Filter", cfg.ENABLE_RSI_FILTER)
        pullback_filter = st.checkbox("Pullback Filter", cfg.ENABLE_PULLBACK_FILTER)

        if st.button("Update Scan Settings"):
            update_config(
                MIN_ADX=min_adx, MIN_ATR_RATIO_PCT=min_atr, ATR_SL_MULT=atr_sl,
                TARGET_RR=target_rr, HIST_SCAN_STEP=hist_step,
                ENABLE_DI_CONFIRM=di_confirm, ENABLE_RSI_FILTER=rsi_filter,
                ENABLE_PULLBACK_FILTER=pullback_filter,
            )
            st.success("Updated!")

    with st.expander("Backtest Realism (v4)", expanded=False):
        spread_pips = st.slider("Spread (pips)", 0.0, 5.0, cfg.SPREAD_PIPS, 0.5)
        slippage_pips = st.slider("Slippage (pips)", 0.0, 3.0, cfg.SLIPPAGE_PIPS, 0.1)
        max_hold = st.slider("Max Hold (days)", 7, 90, cfg.MAX_HOLD_DAYS, 1)
        
        if st.button("Update Backtest Settings"):
            update_config(SPREAD_PIPS=spread_pips, SLIPPAGE_PIPS=slippage_pips, MAX_HOLD_DAYS=max_hold)
            st.success("Updated!")

    with st.expander("Account", expanded=False):
        account_size = st.number_input("Account ($)", 1000, 1000000, int(cfg.ACCOUNT_SIZE_USD), 1000)
        risk_pct = st.slider("Risk %", 0.5, 5.0, cfg.RISK_PER_TRADE_PCT * 100, 0.5) / 100
        if st.button("Update Account"):
            update_config(ACCOUNT_SIZE_USD=account_size, RISK_PER_TRADE_PCT=risk_pct)
            st.success("Updated!")

    st.divider()

    # Live scan
    st.subheader("🔴 Live Scan")
    if st.button("Run Scan Now", type="primary"):
        with st.spinner("Scanning..."):
            res, duration = run_scan()
            st.session_state.last_scan = res
            st.session_state.last_scan_time = datetime.now()
            st.session_state.last_scan_duration = duration
            if res:
                st.success(f"✅ {len(res)} signal(s)")
            else:
                st.info("No signals")

    st.divider()

    # Historical scan
    st.subheader("📊 Historical")
    if not st.session_state.oanda_available:
        max_days = YFINANCE_LIMITS.get(cfg.LTF, {}).get("max_days", 60)
        lookback = st.slider("Lookback", 1, min(14, max_days), 3)
    else:
        lookback = st.slider("Lookback", 1, 30, 3)

    if st.button("Scan Historical"):
        with st.spinner(f"Scanning {lookback} days..."):
            sigs, rej, warns, stats, duration = run_historical_scan(lookback)
            for w in warns:
                st.warning(w)
            st.session_state.hist_signals = sigs
            st.session_state.hist_rejections = rej
            st.session_state.hist_stats = stats
            st.session_state.last_scan_duration = duration
            st.session_state.last_scan_time = datetime.now()
            
            if sigs:
                st.success(f"✅ {len(sigs)} signals found")
            else:
                st.warning("No signals found - check rejections in Backtest tab")

    st.divider()

    # Signal management
    st.subheader("🗂️ Management")
    
    # v4: Update outcomes button
    if st.button("🔄 Update Outcomes"):
        with st.spinner("Updating..."):
            updated, still_open = update_active_signal_outcomes()
            st.success(f"Updated: {updated}, Open: {still_open}")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if st.button("Archive Old"):
            count = archive_old_signals(cfg.MAX_SIGNAL_AGE_DAYS)
            st.success(f"Archived: {count}")
    with col_m2:
        if st.button("Delete Arch"):
            count = delete_archived_signals()
            st.success(f"Deleted: {count}")
    
    # v4: Refresh prices button
    if st.button("🔄 Refresh Prices"):
        st.session_state.price_refresh_time = datetime.now()
        st.success("Prices will refresh on next view")
    
    st.divider()
    
    # Diagnostic section
    with st.expander("🔧 Diagnostics"):
        st.write(f"**OANDA Connected:** {st.session_state.oanda_available}")
        st.write(f"**Account ID:** {st.session_state.oanda_account_id}")
        st.write(f"**HTF:** {cfg.HTF}, **LTF:** {cfg.LTF}")
        st.write(f"**MIN_BARS:** {cfg.MIN_BARS}, **MIN_ADX:** {cfg.MIN_ADX}")
        
        if st.button("🔌 Test OANDA Connection"):
            with st.spinner("Testing OANDA..."):
                test_result = test_oanda_connection()
                st.json(test_result)
        
        if st.button("📊 Test Data Fetch"):
            st.write("Testing data fetch for all pairs...")
            results = []
            for pair in PAIRS:
                with st.spinner(f"Fetching {pair}..."):
                    htf = fetch_data(pair, cfg.HTF)
                    ltf = fetch_data(pair, cfg.LTF)
                    htf_status = f"✅ {len(htf)} bars" if htf is not None and not htf.empty else "❌ Failed"
                    ltf_status = f"✅ {len(ltf)} bars" if ltf is not None and not ltf.empty else "❌ Failed"
                    results.append({
                        "Pair": pair,
                        "Instrument": INSTRUMENT_MAP.get(pair, "?"),
                        f"HTF ({cfg.HTF})": htf_status,
                        f"LTF ({cfg.LTF})": ltf_status
                    })
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        if st.button("🧪 Test Single Pair (EUR_USD)"):
            with st.spinner("Testing EUR_USD..."):
                st.write("**Testing OANDA H4 fetch for EUR_USD...**")
                try:
                    params = {"count": 10, "granularity": "H4", "price": "M"}
                    r = InstrumentsCandles(instrument="EUR_USD", params=params)
                    resp = st.session_state.oanda_api.request(r)
                    candles = resp.get("candles", [])
                    st.write(f"Received {len(candles)} candles")
                    if candles:
                        st.write("**Sample candle:**")
                        st.json(candles[0])
                except Exception as e:
                    st.error(f"Error: {e}")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Live", "📊 Backtest", "📋 Signals", "📉 Stats"])

# TAB 1: Live
with tab1:
    st.header("Live Scan Results")
    
    if st.session_state.last_scan:
        scan_df = pd.DataFrame(st.session_state.last_scan)
        
        display_cols = ["instrument", "direction", "entry", "sl", "tp", "units", 
                        "confidence", "adx", "atr_ratio", "rsi", "win_prob"]
        display_cols = [c for c in display_cols if c in scan_df.columns]
        
        styled = style_dataframe(scan_df[display_cols], direction_col="direction", outcome_col="")
        st.dataframe(styled, use_container_width=True)
        
        if st.checkbox("Show Market Context", value=True):
            with st.spinner("Fetching prices..."):
                context_df = add_market_context(scan_df)
                ctx_cols = ["instrument", "direction", "entry", "current_price", 
                           "pips_to_sl", "pips_to_tp", "status_emoji"]
                ctx_cols = [c for c in ctx_cols if c in context_df.columns]
                st.subheader("📍 Market Context")
                styled_ctx = style_dataframe(context_df[ctx_cols], direction_col="direction", outcome_col="")
                st.dataframe(styled_ctx, use_container_width=True)
    else:
        st.info("Run a scan to see results")

# TAB 2: Backtest
with tab2:
    st.header("Backtest Results")
    
    if "hist_stats" in st.session_state and st.session_state.hist_stats:
        stats = st.session_state.hist_stats
        
        # Check if we have signals
        if stats["total_signals"] > 0:
            # v4: Enhanced stats with MAE/MFE
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Signals", stats["total_signals"])
            with c2:
                st.metric("Win Rate", f"{stats['win_rate']}%")
            with c3:
                st.metric("W/L", f"{stats['wins']}/{stats['losses']}")
            with c4:
                st.metric("Pips", f"{stats['total_pips']}")
            with c5:
                st.metric("PF", f"{stats['profit_factor']}")
            with c6:
                st.metric("Avg Hold", f"{stats.get('avg_hold_bars', 0)} bars")
            
            # v4: MAE/MFE metrics
            st.divider()
            c_a, c_b, c_c, c_d = st.columns(4)
            with c_a:
                st.metric("Avg MAE", f"{stats.get('avg_mae', 0)} pips", help="Max Adverse Excursion")
            with c_b:
                st.metric("Avg MFE", f"{stats.get('avg_mfe', 0)} pips", help="Max Favorable Excursion")
            with c_c:
                st.metric("Gross Profit", f"{stats.get('gross_profit', 0)} pips")
            with c_d:
                st.metric("Gross Loss", f"-{stats.get('gross_loss', 0)} pips")
            
            st.divider()
            
            if stats.get("signals_with_outcomes"):
                st.subheader("📋 Signals with Outcomes")
                sig_df = pd.DataFrame(stats["signals_with_outcomes"])
                
                cols = ["instrument", "direction", "entry", "sl", "tp", "units",
                       "confidence", "adx", "atr_ratio", "rsi", "outcome", 
                       "pnl_pips", "mae_pips", "mfe_pips", "hold_bars", "win_prob"]
                cols = [c for c in cols if c in sig_df.columns]
                
                styled = style_dataframe(sig_df[cols], direction_col="direction", outcome_col="outcome")
                st.dataframe(styled, use_container_width=True)
        else:
            # No signals found - show filter rejection analysis
            st.warning("⚠️ No signals found in the historical scan")
            
            # Show filter rejections if available
            if "filter_rejections" in stats:
                st.subheader("🔍 Why no signals? Filter Rejection Analysis")
                
                filter_rej = stats["filter_rejections"]
                active_rejections = {k: v for k, v in filter_rej.items() if v > 0}
                
                if active_rejections:
                    # Create a nice display
                    rej_df = pd.DataFrame([
                        {"Filter": k, "Rejections": v} 
                        for k, v in sorted(active_rejections.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(rej_df, use_container_width=True)
                    
                    # Provide suggestions based on rejections
                    st.subheader("💡 Suggestions")
                    suggestions = []
                    
                    if filter_rej.get("low_adx", 0) > 10:
                        suggestions.append(f"• **Low ADX** ({filter_rej['low_adx']} rejections): Try lowering MIN_ADX from current value (markets may be ranging)")
                    if filter_rej.get("low_atr_ratio", 0) > 10:
                        suggestions.append(f"• **Low ATR Ratio** ({filter_rej['low_atr_ratio']} rejections): Try lowering MIN_ATR_RATIO_PCT (low volatility)")
                    if filter_rej.get("di_confirm_fail", 0) > 10:
                        suggestions.append(f"• **DI Confirmation** ({filter_rej['di_confirm_fail']} rejections): Try disabling ENABLE_DI_CONFIRM")
                    if filter_rej.get("rsi_overbought", 0) + filter_rej.get("rsi_oversold", 0) > 10:
                        suggestions.append(f"• **RSI Filter** rejections: Try disabling ENABLE_RSI_FILTER")
                    if filter_rej.get("pullback_filter_fail", 0) > 10:
                        suggestions.append(f"• **Pullback Filter** ({filter_rej['pullback_filter_fail']} rejections): Try disabling ENABLE_PULLBACK_FILTER or increasing PULLBACK_ATR_MAX")
                    if filter_rej.get("neutral_trend", 0) > 10:
                        suggestions.append(f"• **Neutral Trend** ({filter_rej['neutral_trend']} rejections): Markets may be sideways, no clear trend")
                    if filter_rej.get("no_data", 0) > 0 or filter_rej.get("empty_df", 0) > 0:
                        suggestions.append("• **Data Issues**: Check your OANDA API connection or try reducing lookback period")
                    
                    if suggestions:
                        for s in suggestions:
                            st.markdown(s)
                    else:
                        st.info("Filters are working normally, but market conditions didn't produce signals.")
                else:
                    st.info("No specific filter rejections recorded.")
    else:
        st.info("Run historical scan to see backtest results")
    
    if "hist_rejections" in st.session_state:
        with st.expander("🔥 Per-Pair Rejections"):
            rej_df = build_rejection_heatmap(st.session_state.hist_rejections)
            if not rej_df.empty:
                st.dataframe(rej_df)

# TAB 3: All Signals
with tab3:
    st.header("All Signals")
    
    status_filter = st.selectbox("Status", ["All", "active", "closed", "archived"])
    df = load_signals() if status_filter == "All" else load_signals(status_filter)
    
    if df.empty:
        st.info("No signals")
    else:
        if st.checkbox("Market Context", key="all_ctx"):
            with st.spinner("Fetching..."):
                df = add_market_context(df)
        
        cols = ["instrument", "direction", "entry", "sl", "tp", "units",
               "confidence", "adx", "atr_ratio", "rsi", "win_prob", "status", "outcome",
               "pnl_pips", "mae_pips", "mfe_pips", "open_time"]
        if "current_price" in df.columns:
            cols.insert(5, "current_price")
        if "pips_to_sl" in df.columns:
            cols.insert(cols.index("pnl_pips"), "pips_to_sl")
        if "pips_to_tp" in df.columns:
            cols.insert(cols.index("pnl_pips"), "pips_to_tp")
        cols = [c for c in cols if c in df.columns]
        
        styled = style_dataframe(df[cols], direction_col="direction", outcome_col="outcome")
        st.dataframe(styled, use_container_width=True)
        st.caption(f"Total: {len(df)}")

# TAB 4: Stats
with tab4:
    st.header("Statistics")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Signal Summary")
        all_sig = load_signals()
        if not all_sig.empty:
            st.metric("Total", len(all_sig))
            st.metric("Active", len(all_sig[all_sig["status"] == "active"]))
            st.metric("Closed", len(all_sig[all_sig["status"] == "closed"]))
            
            st.divider()
            st.write("**By Direction**")
            st.bar_chart(all_sig["direction"].value_counts())
            
            st.write("**By Instrument**")
            st.bar_chart(all_sig["instrument"].value_counts())
    
    with c2:
        st.subheader("Backtest History")
        bt_hist = load_backtest_history()
        if not bt_hist.empty:
            show_cols = ["run_time", "lookback_days", "total_signals", 
                        "win_rate", "total_pips", "profit_factor", "avg_mae", "avg_mfe"]
            show_cols = [c for c in show_cols if c in bt_hist.columns]
            formatted_hist = format_dataframe_decimals(bt_hist[show_cols], 2)
            st.dataframe(formatted_hist, use_container_width=True)
        else:
            st.info("No history")
    
    st.divider()
    st.subheader("💰 Account")
    cfg = get_config()
    ca, cb, cc = st.columns(3)
    with ca:
        st.metric("Size", f"${cfg.ACCOUNT_SIZE_USD:,.0f}")
    with cb:
        st.metric("Risk", f"{cfg.RISK_PER_TRADE_PCT * 100:.1f}%")
    with cc:
        st.metric("Risk $", f"${cfg.ACCOUNT_SIZE_USD * cfg.RISK_PER_TRADE_PCT:,.0f}")
    
    # v4: Backtest settings display
    st.subheader("🎯 Backtest Settings")
    cd, ce, cf = st.columns(3)
    with cd:
        st.metric("Spread", f"{cfg.SPREAD_PIPS} pips")
    with ce:
        st.metric("Slippage", f"{cfg.SLIPPAGE_PIPS} pips")
    with cf:
        st.metric("Max Hold", f"{cfg.MAX_HOLD_DAYS} days")
