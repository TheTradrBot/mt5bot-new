# FTMO 200K Trading Bot

> **For a comprehensive understanding of this project, start with `PROJECT_OVERVIEW.md`**

## Overview
This project is a **production-hardened MetaTrader 5 FTMO 200K Trading Bot** with professional-grade **parameter optimization**. The system has been upgraded to ensure production safety with accurate position sizing, no source code mutation, and robust live trading features.

### Two Main Components

1. **main_live_bot.py** - A standalone MT5 live trading bot designed for FTMO 200K challenge accounts. It runs 24/7 on a Windows VM with MetaTrader 5, executing trades based on a rigorously backtested "7 Confluence Pillars" strategy. **Loads all tunable parameters from `params/current_params.json` at startup.**

2. **ftmo_challenge_analyzer.py** - An optimization engine that backtests the strategy against 2024 historical data, runs walk-forward optimization, and **saves optimized parameters to `params/current_params.json`** (does NOT modify source code).

## Key Improvements (Post-Fix)

| Issue | Status |
|-------|--------|
| Position sizing pip values | **FIXED** - Symbol-specific for all 34 assets |
| Source code mutation | **FIXED** - Optimizer saves to JSON only |
| Transaction costs | **FIXED** - Spread + slippage in backtests |
| Look-ahead bias | **FIXED** - Timestamp-based MTF alignment |
| MT5 connectivity | **FIXED** - Auto-reconnection with exponential backoff |
| Spread validation | **FIXED** - Pre-trade checks enforced |

## Project Structure

### Core Trading Files
- `main_live_bot.py` - Live trading bot (loads params from JSON)
- `strategy_core.py` - Core strategy logic (7 Confluence Pillars)
- `ftmo_config.py` - FTMO-specific configuration

### Parameter Management
- `params/current_params.json` - **Single source of truth** for all tunable parameters
- `params/params_loader.py` - Parameter loading utilities

### Optimization System
- `ftmo_challenge_analyzer.py` - Backtests and saves params to JSON
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

Trades execute only when >= 5 pillars align with valid R:R ratio.

### Blueprint V2 Enhancements (December 2025)
- **Proper Fib Anchoring**: Body-to-body anchors (red close → green open for bullish N, opposite for bearish V)
- **Mitigated S/R Zones**: Broken-then-retested levels with 1-2% proximity filter
- **Structural Frameworks**: Ascending/descending channel detection on Daily
- **Displacement Filter**: Strong BOS/CHoCH candles beyond structure (1.5x ATR minimum)
- **Candle Rejection**: Pinbar/engulfing patterns at S/R zones
- **RSI Divergence**: Bullish/bearish divergence detection
- **Bollinger Mean Reversion**: Band touch confluence signals

## Risk Management
- Accurate symbol-specific position sizing (all 34 assets safe)
- Dynamic risk per trade (0.5-1.0%)
- Concurrent trade limits
- Pre-trade FTMO rule violation checks
- Pre-trade spread validation
- 5 risk modes: Aggressive, Normal, Conservative, Ultra-Safe, Halted

## Support/Resistance Level Detection
The project includes `detect_sr_levels.py` which provides TradingView-style horizontal S/R line detection:

- Uses scipy.signal.find_peaks for swing high/low identification
- Clusters similar levels within asset-specific tolerances (5-10 pips for major FX)
- Counts touches with reversal confirmation (minimum 3 touches required)
- Each candle contributes maximum 1 touch (prevents double-counting)
- Calculates time-decayed strength scores

### Generated S/R Files
- Location: `data/sr_levels/`
- Format: `{SYMBOL}_{TIMEFRAME}_sr.json` (e.g., EURUSD_MN_sr.json)
- Timeframes: Weekly (W1) and Monthly (MN)
- Summary: `data/sr_levels/all_sr_summary.json`

### Using S/R Levels in Strategy
```python
from detect_sr_levels import load_sr_levels
levels = load_sr_levels("EURUSD", "MN")  # Returns list of level dicts
```

## Supported Assets
34 assets including:
- Forex: Majors + Cross pairs (28)
- Metals: XAUUSD, XAGUSD (2)
- Crypto: BTCUSD, ETHUSD (2)
- Indices: SPX500, NAS100 (2)

## Environment Variables Required
```
MT5_SERVER=FTMO-Demo
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_account_id
```

## Running the Project

## Optimization & Backtesting

The optimizer uses professional quant best practices:

- **TRAINING PERIOD**: January 1, 2024 – September 30, 2024 (in-sample optimization)
- **VALIDATION PERIOD**: October 1, 2024 – December 31, 2024 (out-of-sample test)
- **FINAL BACKTEST**: Full year 2024 with best parameters

All trades from the final full-year backtest are exported to:
`ftmo_analysis_output/all_trades_2024_full.csv`

Parameters are saved to `params/current_params.json`

### Step 1: Generate Parameters (Replit)
```bash
python ftmo_challenge_analyzer.py
```
This runs 5 Optuna trials (quick test mode), validates on Oct-Dec 2024, and saves optimized parameters to `params/current_params.json`.

### Step 2: Live Trading (Windows VM)
```bash
python main_live_bot.py
```
The bot loads parameters from `params/current_params.json` automatically.

### Web Status Server (Replit)
```bash
python main.py
```

## User Preferences
- Strategy uses EXACT SAME logic as backtests
- Bot trades independently (no external dependencies required)
- Pre-trade risk checks to prevent FTMO rule violations
- Parameters loaded from JSON (no hardcoded values in source)

## Assessment

**Current Rating**: 7.5-8/10  
**Status**: Ready for paper trading; monitor closely on live challenge.
