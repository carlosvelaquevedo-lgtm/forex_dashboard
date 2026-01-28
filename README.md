# Trend Following Scanner Dashboard

A Streamlit app that scans forex pairs + gold for trend-following signals using:
- Higher timeframe trend filter (EMA 50/200)
- Lower timeframe EMA 9/21 crossover
- ADX strength, ATR volatility, DI direction, RSI filter
- Configurable aggressive mode for testing

Features:
- Real-time OANDA price & account summary (optional)
- Email notifications on new signals
- Manual trade outcome tracking (win/loss/breakeven)
- SQLite persistence
