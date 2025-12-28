# MT5 FTMO Trading Bot

Automated MetaTrader 5 trading bot for FTMO 200K Challenge accounts. Uses a 7-Pillar Confluence system with multi-timeframe analysis.

## Quick Start

```bash
# Run optimization (recommended: use helper script for background runs)
./run_optimization.sh --single --trials 100  # TPE (auto-logs to ftmo_analysis_output/TPE/run.log)
./run_optimization.sh --multi --trials 100   # NSGA-II (auto-logs to ftmo_analysis_output/NSGA/run.log)

# Or run directly
python ftmo_challenge_analyzer.py --single --trials 100  # TPE single-objective (recommended)
python ftmo_challenge_analyzer.py --multi --trials 100   # NSGA-II multi-objective

# Monitor live progress
tail -f ftmo_analysis_output/TPE/run.log          # Complete output
tail -f ftmo_analysis_output/TPE/optimization.log # Trial results only

# Check optimization status
python ftmo_challenge_analyzer.py --status

# Show current configuration
python ftmo_challenge_analyzer.py --config

# Run live bot (Windows VM with MT5)
python main_live_bot.py
```

## Recent Updates (Dec 28, 2025)

### New Features
- ✨ **FTMOComplianceTracker**: FTMO compliance tracking with daily DD (4.5%), total DD (9%), streak halt
- ✨ **Parameter expansion**: 25+ optimizable parameters (TP scaling, 6 filter toggles, ADX regime)
- ✨ **TP scaling**: tp1/2/3_r_multiple (1.0-6.0R) and tp1/2/3_close_pct (0.15-0.40) for dynamic profit taking
- ✨ **Filter toggles**: 6 new filters (HTF, structure, Fibonacci, confirmation, displacement, candle rejection)
- ✨ **Successful test**: 5 trials generated 800-1400 trades each, best: +549R (Sharpe 2.80)

### Critical Bug Fixes
- ✅ **0-trade bug**: Fixed aggressive filters/compliance penalties causing 0 trades - now 800-1400/trial
- ✅ **params_loader.py**: Removed obsolete `liquidity_sweep_lookback` parameter
- ✅ **Metric calculations**: Fixed win_rate (4700%→47%), Calmar ratio (0.00→proper values), total_return units
- ✅ **Optimization logs**: Fixed R=0.0 display bug for losing trials
- ✅ **Trade exports**: All 34 symbols now appear in CSV outputs

### Performance Improvements
- ⚡ **Validation runs**: Only execute on top 5 trials at end (not every best trial)
- ⚡ **Faster optimization**: ~10x speedup by removing redundant backtests

### Configuration
- 🔧 **ADX filter**: Disabled (incompatible with current strategy baseline)
- 🔧 **Filter toggles**: Hard-coded to False during optimization (baseline establishment)

## Project Structure

```
├── strategy_core.py          # Core trading logic (7 Confluence Pillars)
├── ftmo_challenge_analyzer.py # Optuna optimization & backtesting
├── main_live_bot.py          # Live MT5 trading entry point
├── config.py                 # Account settings, CONTRACT_SPECS
├── ftmo_config.py            # FTMO challenge rules & risk limits
├── docs/                     # Documentation (system guide, strategy analysis)
├── scripts/                  # Utility scripts (monitoring, debugging)
├── params/                   # Optimized parameters (current_params.json)
└── data/ohlcv/               # Historical OHLCV data (2003-2025)
```

## Optimization & Backtesting

The optimizer uses professional quant best practices:

- **TRAINING PERIOD**: January 1, 2024 – September 30, 2024 (in-sample optimization)
- **VALIDATION PERIOD**: October 1, 2024 – December 31, 2024 (out-of-sample test)
- **FINAL BACKTEST**: Full year 2024 (December fully open for trading)
- **ADX > 25 trend-strength filter** applied to avoid ranging markets.

All trades from the final backtest are exported to:
`ftmo_analysis_output/all_trades_2024_full.csv`

Parameters are saved to `params/current_params.json`

Optimization is resumable and can be checked with: `python ftmo_challenge_analyzer.py --status`


## Documentation

### Core Documentation (Auto-Updated)
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete system architecture & data flow (28KB)
- **[STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md)** - Trading strategy deep dive with current parameters (11KB)
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Complete API documentation for all modules (46KB)
- **[DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)** - Installation, deployment & troubleshooting (8.3KB)
- **[OPTIMIZATION_FLOW.md](docs/OPTIMIZATION_FLOW.md)** - 4-phase optimization process (5.3KB)
- **[BASELINE_ANALYSIS.md](docs/BASELINE_ANALYSIS.md)** - Comprehensive performance analysis & roadmap (NEW)
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Version history & recent changes (2.2KB)

### Legacy Documentation
- **[PROFESSIONAL_SYSTEM_GUIDE.md](docs/PROFESSIONAL_SYSTEM_GUIDE.md)** - Original system guide
- **[TRADING_STRATEGY_ANALYSIS.txt](docs/TRADING_STRATEGY_ANALYSIS.txt)** - Strategy analysis notes

### Quick References
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - AI assistant context & commands

**📝 Documentation Auto-Update**: All core docs are auto-generated from source code on every commit via GitHub Actions. Manual update: `python scripts/update_docs.py`

---

**Last Documentation Update**: 2025-12-28 14:19:50  
**Auto-generated**: Run `python scripts/update_docs.py` to regenerate docs
