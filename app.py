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
    HIST_SCAN_STEP: int = 6  # Check every N bars instead of every bar

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

# Pip multipliers and values for position sizing
INSTRUMENT_SPECS = {
    "EURUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5},
    "USDJPY=X": {"pip_multiplier": 100, "pip_value_per_lot": 1000 / 150, "type": "forex", "decimals": 3},
    "GBPUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5},
    "AUDUSD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5},
    "USDCHF=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0, "type": "forex", "decimals": 5},
    "USDCAD=X": {"pip_multiplier": 10000, "pip_value_per_lot": 10.0 / 1.36, "type": "forex", "decimals": 5},
    "GC=F": {"pip_multiplier": 1, "pip_value_per_lot": 100.0, "type": "commodity", "decimals": 2},
}

# yfinance period limits by interval
YFINANCE_LIMITS = {
    "1d": {"max_period": "2y",
