# MT5 FTMO Trading Bot

Automated MetaTrader 5 trading bot for FTMO 200K Challenge accounts. Uses a 7-Pillar Confluence system with ADX regime detection.

## Quick Start

```bash
# Run NSGA-II multi-objective optimization (recommended)
python ftmo_challenge_analyzer.py --multi --trials 100

# Run TPE single-objective optimization (faster)
python ftmo_challenge_analyzer.py --single --trials 100

# Enable ADX regime filtering
python ftmo_challenge_analyzer.py --multi --adx --trials 100

# Check optimization status
python ftmo_challenge_analyzer.py --status

# Show current configuration
python ftmo_challenge_analyzer.py --config

# Run live bot (Windows VM with MT5)
python main_live_bot.py
```

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

The optimizer uses professional quant best practices with **two optimization modes**:

### NSGA-II Multi-Objective (Default)
- Optimizes 3 objectives simultaneously: Total R, Sharpe Ratio, Win Rate
- Produces Pareto frontier with multiple optimal solutions
- Better for balancing profit vs risk (critical for FTMO 10% max drawdown)
- Output: `ftmo_analysis_output/NSGA/`

### TPE Single-Objective
- Optimizes composite score: R + Sharpe_bonus + PF_bonus + WR_bonus
- Faster, simpler scoring function
- Output: `ftmo_analysis_output/TPE/`

### Data Partitioning
- **TRAINING PERIOD**: January 1, 2023 – September 30, 2024 (in-sample optimization)
- **VALIDATION PERIOD**: October 1, 2024 – December 26, 2025 (out-of-sample test)
- **FINAL BACKTEST**: Full period 2023-2025 (December fully open for trading)
- **ADX regime filter** (optional): Adapts strategy to trend/range/transition modes

All trades from the final backtest are exported to mode-specific directories.
Parameters are saved to `params/current_params.json`.
Optimization is resumable via SQLite database: `ftmo_optimization.db`

## Documentation

### Core Documentation (Auto-Updated)
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete system architecture & data flow (28KB)
- **[STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md)** - Trading strategy deep dive with current parameters (11KB)
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Complete API documentation for all modules (46KB)
- **[DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)** - Installation, deployment & troubleshooting (8.3KB)
- **[OPTIMIZATION_FLOW.md](docs/OPTIMIZATION_FLOW.md)** - 4-phase optimization process (5.3KB)
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Version history & recent changes (2.2KB)

### Legacy Documentation
- **[PROFESSIONAL_SYSTEM_GUIDE.md](docs/PROFESSIONAL_SYSTEM_GUIDE.md)** - Original system guide
- **[TRADING_STRATEGY_ANALYSIS.txt](docs/TRADING_STRATEGY_ANALYSIS.txt)** - Strategy analysis notes

### Quick References
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - AI assistant context & commands

**📝 Documentation Auto-Update**: All core docs are auto-generated from source code on every commit via GitHub Actions. Manual update: `python scripts/update_docs.py`

---

**Last Documentation Update**: 2025-12-28 13:10:13  
**Auto-generated**: Run `python scripts/update_docs.py` to regenerate docs
