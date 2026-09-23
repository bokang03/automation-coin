@echo off
cd /d "%~dp0"
.venv\Scripts\python -m aoabot.backtest data\candles_XBTUSD_1h.parquet data\candles_XBTUSD_1h_2021_2026.parquet %*
