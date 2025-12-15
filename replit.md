# FTMO 200K Trading Bot

> **For a comprehensive understanding of this project, start with `PROJECT_OVERVIEW.md`**

## Overview
This project is a **MetaTrader 5 FTMO 200K Trading Bot** with an integrated **optimization system**. The project contains two main components:

1. **main_live_bot.py** - A standalone MT5 live trading bot designed for FTMO 200K challenge accounts. It runs 24/7 on a Windows VM with MetaTrader 5, executing trades based on a rigorously backtested "7 Confluence Pillars" strategy.

2. **ftmo_challenge_analyzer.py** - An optimization engine that backtests main_live_bot against 2024 historical data, runs multiple optimization iterations, and updates the best-performing parameters to main_live_bot.py automatically.

## Project Structure

### Core Trading Files
- `main_live_bot.py` - The primary live trading bot (runs on Windows VM with MT5)
- `strategy_core.py` - Core strategy logic (7 Confluence Pillars)
- `ftmo_config.py` - FTMO-specific configuration and risk parameters

### Optimization System
- `ftmo_challenge_analyzer.py` - Backtests and optimizes main_live_bot.py using 2024 data
- `ftmo_optimization_backups/` - Backup copies of each optimization iteration

### Data & Analysis
- `data/ohlcv/` - Historical OHLCV data for backtesting (2023-2024)
- `ftmo_analysis_output/` - Analysis results, trade logs, and performance reports

### Supporting Modules
- `tradr/` - Trading infrastructure (MT5 client, risk manager, data providers)
- `src/` - Additional strategy and backtest modules
- `data_provider.py` - OANDA API data fetching

## Trading Strategy
The bot employs a "7 Confluence Pillars" strategy:
1. HTF Bias (Monthly/Weekly/Daily trend)
2. Location (S/R zones)
3. Fibonacci (Golden Pocket)
4. Liquidity (Sweep near equal highs/lows)
5. Structure (BOS/CHoCH alignment)
6. Confirmation (4H candle patterns)
7. Risk:Reward (Min 1:1)

Trades execute only when >= 4 pillars align with valid R:R ratio.

## Risk Management
- Dynamic position sizing (0.75-0.95% base risk)
- 5 concurrent trade limit, 6 pending order limit
- Pre-trade FTMO rule violation checks
- 5 risk modes: Aggressive, Normal, Conservative, Ultra-Safe, Halted

## Supported Assets
34 assets including:
- Forex: Majors + Cross pairs
- Metals: XAUUSD, XAGUSD
- Crypto: BTCUSD, ETHUSD
- Indices: US500, US100

## Environment Variables Required
```
MT5_SERVER=FTMO-Demo
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_account_id
```

## Running the Project

### Live Trading (Windows VM)
```bash
python main_live_bot.py
```

### Run Optimization
```bash
python ftmo_challenge_analyzer.py
```

### Web Status Server (Replit)
```bash
python main.py
```

## User Preferences
- Strategy must use EXACT SAME logic as backtests
- Bot trades independently (no external dependencies required)
- Pre-trade risk checks to prevent FTMO rule violations
