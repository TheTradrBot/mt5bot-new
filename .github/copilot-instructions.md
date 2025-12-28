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

### Recent Bug Fixes (Dec 28, 2025)
**IMPORTANT**: The following bugs were recently fixed - avoid reintroducing:

1. **FTMOComplianceTracker**: Implemented compliance tracking class with daily DD (4.5%), total DD (9%), streak halt (999)
   - Metrics-only mode for backtesting (no trade filtering)
   - Returns (trades, compliance_report) tuple from run_full_period_backtest
   - Hard constraints: TP ordering (tp1<tp2<tp3), close-sum ≤85%, ADX threshold ordering
2. **Parameter expansion**: Expanded search space from 17→25+ parameters:
   - TP scaling: tp1/2/3_r_multiple (1.0-6.0R) and tp1/2/3_close_pct (0.15-0.40)
   - Filter toggles: 6 new filters (HTF, structure, Fibonacci, confirmation, displacement, candle rejection)
   - All filter toggles HARD-CODED to False during optimization (baseline establishment)
3. **0-trade bug fix**: Initial implementation filtered all trades due to:
   - Aggressive filter toggles set to True
   - Compliance penalty rejecting trials with DD breaches
   - Streak halt (7) filtering 889/897 trades
   - FIX: Filters disabled, compliance penalty removed, streak halt set to 999
4. **params_loader.py**: Removed `liquidity_sweep_lookback` parameter (doesn't exist in StrategyParams)
5. **professional_quant_suite.py**:
   - Win rate: Remove duplicate `* 100` (already percentage)
   - Calmar ratio: Use `max_drawdown_pct` not `max_drawdown` (USD)
   - Total return: Return USD value, not percentage
6. **ftmo_challenge_analyzer.py**:
   - Quarterly stats must be calculated BEFORE early return for losing trials
   - Use `overall_stats['r_total']` not `user_attrs.get('total_r')` for logging
   - ADX filter disabled: `require_adx_filter=False` everywhere

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

**Recommended: Use helper script for background runs**
```bash
./run_optimization.sh --single --trials 100  # TPE (logs to ftmo_analysis_output/TPE/run.log)
./run_optimization.sh --multi --trials 100   # NSGA-II (logs to ftmo_analysis_output/NSGA/run.log)
tail -f ftmo_analysis_output/TPE/run.log     # Monitor complete output
```

**Direct Python execution**
```bash
python ftmo_challenge_analyzer.py             # Run/resume optimization
python ftmo_challenge_analyzer.py --status    # Check progress
python ftmo_challenge_analyzer.py --config    # Show current configuration
python ftmo_challenge_analyzer.py --trials 100  # Set trial count
python ftmo_challenge_analyzer.py --multi     # Use NSGA-II multi-objective
python ftmo_challenge_analyzer.py --single    # Use TPE single-objective
python ftmo_challenge_analyzer.py --adx       # Enable ADX regime filtering
```
Uses Optuna with SQLite storage (`ftmo_optimization.db`) for crash-resistant optimization.
Configuration loaded from `params/optimization_config.json`.

**Output Structure:**
- NSGA-II runs: `ftmo_analysis_output/NSGA/` (run.log + optimization.log + CSVs)
- TPE runs: `ftmo_analysis_output/TPE/` (run.log + optimization.log + CSVs)
- `run.log`: Complete console output (all debug info, asset processing)
- `optimization.log`: Trial results only (clean, structured)
- Each mode has its own optimization.log and CSV files

### Run Live Bot (Windows VM only)
```bash
# Requires .env with MT5_SERVER, MT5_LOGIN, MT5_PASSWORD
python main_live_bot.py
```

### Background Optimization
```bash
# Recommended: Use helper script
./run_optimization.sh --single --trials 100  # Auto-logs to ftmo_analysis_output/TPE/run.log

# Manual nohup
nohup python ftmo_challenge_analyzer.py > ftmo_analysis_output/TPE/run.log 2>&1 &
tail -f ftmo_analysis_output/TPE/optimization.log  # Monitor TPE progress
tail -f ftmo_analysis_output/NSGA/optimization.log # Monitor NSGA-II progress
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
- Documentation: `docs/` (system guide, strategy analysis, compliance tracking)
- Utility scripts: `scripts/` (optimization monitoring, debug tools)
- New docs: `docs/COMPLIANCE_TRACKING_IMPLEMENTATION.md` (FTMOComplianceTracker guide)

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
