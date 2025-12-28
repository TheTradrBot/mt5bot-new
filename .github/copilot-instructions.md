# MT5 FTMO Trading Bot - AI Agent Instructions

## Project Overview
Automated MetaTrader 5 trading bot for FTMO 200K Challenge accounts. Two-environment architecture:
- **Live Bot** (`main_live_bot.py`): Runs on Windows VM with MT5 installed
- **Optimizer** (`ftmo_challenge_analyzer.py`): Runs anywhere (Replit/local) - no MT5 required

## Architecture & Data Flow

```
params/optimization_config.json  ← Optimization mode settings (multi-obj, ADX, etc.)
params/current_params.json       ← Optimized strategy parameters
         ↑                            ↓
ftmo_challenge_analyzer.py      main_live_bot.py
(Optuna optimization)           (loads params at startup)
         ↑
data/ohlcv/{SYMBOL}_{TF}_2003_2025.csv  (historical data)
```

### Key Modules
| File | Purpose |
|------|---------|
| `strategy_core.py` | Trading strategy logic - 7 Confluence Pillars, regime detection |
| `params/params_loader.py` | Load/save optimized parameters from JSON |
| `params/optimization_config.py` | Unified optimization config (DB path, mode toggles) |
| `config.py` | Account settings, CONTRACT_SPECS (pip values), tradable symbols |
| `ftmo_config.py` | FTMO challenge rules, risk limits, TP/SL settings |
| `symbol_mapping.py` | OANDA ↔ FTMO symbol conversion (`EUR_USD` → `EURUSD`) |
| `tradr/mt5/client.py` | MT5 API wrapper (Windows only) |
| `tradr/risk/manager.py` | FTMO drawdown tracking, pre-trade risk checks |

## Critical Conventions

### Symbol Format
- **Config/data files**: OANDA format with underscores (`EUR_USD`, `XAU_USD`)
- **MT5 execution**: FTMO format (`EURUSD`, `XAUUSD`, `US500.cash`)
- Always use `symbol_mapping.py` for conversions

### Parameters - NEVER Hardcode
```python
# ✅ CORRECT: Load from params loader
from params.params_loader import load_strategy_params
params = load_strategy_params()

# ❌ WRONG: Hardcoding in source files
MIN_CONFLUENCE = 5  # Don't do this
```

### Pip Values - Symbol-Specific
Different instruments have different pip sizes. Always use `get_contract_specs()`:
- Standard forex: `0.0001` (4 decimal)
- JPY pairs: `0.01` (2 decimal)
- Gold (XAUUSD): `0.01`
- Crypto (BTCUSD): `1.0`

### Multi-Timeframe Data
Prevent look-ahead bias by slicing HTF data to reference timestamp:
```python
# strategy_core.py pattern - always use _slice_htf_by_timestamp()
htf_candles = _slice_htf_by_timestamp(weekly_candles, current_daily_dt)
```

## Development Commands

### Run Optimization (resumable)
```bash
python ftmo_challenge_analyzer.py             # Run/resume optimization
python ftmo_challenge_analyzer.py --status    # Check progress
python ftmo_challenge_analyzer.py --config    # Show current configuration
python ftmo_challenge_analyzer.py --trials 100  # Set trial count
python ftmo_challenge_analyzer.py --multi     # Use NSGA-II multi-objective
python ftmo_challenge_analyzer.py --adx       # Enable ADX regime filtering
```
Uses Optuna with SQLite storage (`ftmo_optimization.db`) for crash-resistant optimization.
Configuration loaded from `params/optimization_config.json`.

### Run Live Bot (Windows VM only)
```bash
# Requires .env with MT5_SERVER, MT5_LOGIN, MT5_PASSWORD
python main_live_bot.py
```

### Background Optimization
```bash
nohup python ftmo_challenge_analyzer.py > optimization_output.log 2>&1 &
tail -f ftmo_optimization_progress.txt  # Monitor progress
```

## FTMO Challenge Rules (hardcoded limits)
- Max daily loss: **5%** (halt at 4.2%)
- Max total drawdown: **10%** (emergency at 7%)
- Phase 1 target: **10%**, Phase 2: **5%**
- Risk per trade: typically 0.3-1.0% (from params)

## File Locations
- Historical data: `data/ohlcv/{SYMBOL}_{TF}_2003_2025.csv`
- Optimized params: `params/current_params.json`
- Backtest output: `ftmo_analysis_output/`
- Logs: `logs/tradr_live.log`
- Documentation: `docs/` (system guide, strategy analysis)
- Utility scripts: `scripts/` (optimization monitoring, debug tools)

## Testing Strategy Changes
1. Modify `strategy_core.py` (contains `compute_confluence()`, `simulate_trades()`)
2. Run optimizer: `python ftmo_challenge_analyzer.py --trials 50`
3. Check `ftmo_analysis_output/` for trade CSVs and performance metrics
4. Verify OOS (out-of-sample) performance matches training period

## Common Patterns

### Adding a New Indicator Filter
```python
# In strategy_core.py StrategyParams dataclass
use_my_filter: bool = False
my_threshold: float = 0.5

# In compute_confluence() function
if params.use_my_filter and my_indicator < params.my_threshold:
    return Signal(...)  # Skip or adjust
```

### Adding to Optimization
```python
# In ftmo_challenge_analyzer.py objective function
my_param = trial.suggest_float("my_param", 0.1, 2.0)
```
