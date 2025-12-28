# AI Assistant Quick Start Guide

**Purpose**: This file helps AI assistants (GitHub Copilot, ChatGPT, Claude, etc.) quickly understand the MT5 FTMO Trading Bot project.

**Last Updated**: 2025-12-28 (Auto-updated on every commit)

---

## 🎯 Project Summary in 30 Seconds

**What**: Automated MetaTrader 5 trading bot for FTMO $200K Challenge accounts  
**Strategy**: 7-Pillar Confluence System with ADX regime detection  
**Optimization**: Optuna (TPE/NSGA-II) with 25+ parameters  
**Deployment**: Windows (live bot) + Linux (optimizer)  
**Performance**: ~48% win rate, ~40% annual return (target: 60%+ WR, 100%+ return)

---

## 📁 Essential Files (Read These First)

### 1. Core Trading Logic
- **`strategy_core.py`** (1500 lines) - Complete trading strategy implementation
  - `compute_confluence()` - Main entry signal logic (7 pillars)
  - `simulate_trades()` - Backtest engine
  - `detect_regime()` - ADX-based market classification

### 2. Optimization Engine
- **`ftmo_challenge_analyzer.py`** (2700 lines) - Parameter optimization & backtesting
  - Dual-mode: TPE (fast) or NSGA-II (multi-objective)
  - 25+ parameter search space
  - Training/Validation/Full-period backtests
  - Saves results to `ftmo_analysis_output/NSGA/` or `TPE/`

### 3. Live Trading Bot
- **`main_live_bot.py`** (Windows only) - Production MT5 execution
  - Loads params from `params/current_params.json`
  - Scans 34 assets every 4 hours
  - FTMO risk management (max 10% DD, 5% daily loss)

### 4. Configuration
- **`params/current_params.json`** - Active strategy parameters (loaded by live bot)
- **`params/optimization_config.json`** - Optimization settings (DB path, modes, trials)
- **`config.py`** - Contract specs (pip values for 34 assets)
- **`ftmo_config.py`** - FTMO challenge rules & limits

---

## 🗂️ Directory Structure

```
mt5bot-new/
├── Core Files (CRITICAL - Never hardcode values)
│   ├── strategy_core.py           # Trading strategy (7 pillars)
│   ├── ftmo_challenge_analyzer.py # Optimization engine
│   ├── main_live_bot.py           # Live MT5 bot (Windows)
│   ├── config.py                  # Contract specs, symbols
│   └── ftmo_config.py             # FTMO rules
│
├── params/ (PARAMETER MANAGEMENT)
│   ├── current_params.json        # Active params (loaded by bot)
│   ├── optimization_config.json   # Optimization settings
│   ├── params_loader.py           # Load/save utilities
│   └── history/                   # Timestamped backups
│
├── tradr/ (MODULES)
│   ├── mt5/client.py              # MT5 API wrapper
│   ├── risk/manager.py            # FTMO compliance checks
│   └── utils/output_manager.py    # Result file management
│
├── data/ohlcv/ (HISTORICAL DATA)
│   └── {SYMBOL}_{TF}_2003_2025.csv  # 34 assets × 4 timeframes
│
├── ftmo_analysis_output/ (RESULTS)
│   ├── NSGA/                      # Multi-objective runs
│   │   ├── optimization.log       # Real-time progress
│   │   └── best_trades_*.csv      # Trade exports
│   └── TPE/                       # Single-objective runs
│
├── docs/ (AUTO-GENERATED DOCUMENTATION)
│   ├── ARCHITECTURE.md            # System design (28KB)
│   ├── STRATEGY_GUIDE.md          # Trading strategy (11KB)
│   ├── API_REFERENCE.md           # Code API (46KB)
│   ├── DEPLOYMENT_GUIDE.md        # Setup guide (8KB)
│   └── CHANGELOG.md               # Version history (2KB)
│
└── scripts/ (UTILITIES)
    ├── update_docs.py             # Auto-generate documentation
    └── pre-commit-hook.sh         # Auto-update on commit
```

---

## 🔑 Key Concepts for AI Understanding

### 1. Parameter Management (CRITICAL)
**Rule**: NEVER hardcode strategy parameters in source code.

**Correct**:
```python
from params.params_loader import load_strategy_params
params = load_strategy_params()
min_conf = params['min_confluence_score']  # Loaded from JSON
```

**Wrong**:
```python
MIN_CONFLUENCE = 4  # ❌ NEVER DO THIS
```

**Why**: Optimizer saves parameters to `current_params.json`. Live bot loads from JSON. Hardcoding breaks the optimization → live bot workflow.

### 2. Symbol Format Conversion
**OANDA Format** (data files, config): `EUR_USD`, `XAU_USD`  
**FTMO Format** (MT5 execution): `EURUSD`, `XAUUSD`

**Use**:
```python
from symbol_mapping import oanda_to_ftmo, ftmo_to_oanda
```

### 3. Pip Values (Symbol-Specific)
Different symbols have different pip sizes:
- Forex standard: `0.0001` (EURUSD, GBPUSD)
- JPY pairs: `0.01` (USDJPY, EURJPY)
- Gold: `0.01` (XAUUSD)
- Crypto: `1.0` (BTCUSD)

**Always use**:
```python
from tradr.risk.position_sizing import get_contract_specs
specs = get_contract_specs(symbol)
pip_value = specs['pip_value']
```

### 4. Multi-Timeframe Data (Look-Ahead Bias Prevention)
When using weekly/monthly candles with daily data, slice HTF data to reference timestamp:

```python
from strategy_core import _slice_htf_by_timestamp
htf_candles = _slice_htf_by_timestamp(weekly_candles, current_daily_dt)
```

### 5. Optimization Modes
**TPE (Single-Objective)**:
```bash
python ftmo_challenge_analyzer.py --single --trials 100
```
- Faster (~2.5h for 100 trials)
- Optimizes composite score
- Output: `ftmo_analysis_output/TPE/`

**NSGA-II (Multi-Objective)** (Recommended):
```bash
python ftmo_challenge_analyzer.py --multi --trials 100
```
- Optimizes 3 objectives: Total R, Sharpe, Win Rate
- Finds Pareto frontier (multiple optimal solutions)
- Better for balancing profit vs risk (FTMO 10% DD limit)
- Output: `ftmo_analysis_output/NSGA/`

---

## 🚀 Common AI Tasks & How to Handle Them

### Task 1: "Add a new indicator/filter to the strategy"

**Steps**:
1. Modify `strategy_core.py`:
   - Add parameter to `StrategyParams` dataclass
   - Implement filter logic in `compute_confluence()`
   - Add to pillar scoring system

2. Update parameter space in `ftmo_challenge_analyzer.py`:
   ```python
   'my_new_param': trial.suggest_float('my_new_param', 0.1, 2.0, step=0.1)
   ```

3. Run optimization:
   ```bash
   python ftmo_challenge_analyzer.py --multi --trials 100
   ```

4. Documentation auto-updates on commit (no manual editing needed)

### Task 2: "Fix a bug or improve risk management"

**Steps**:
1. Identify affected module:
   - Trading logic: `strategy_core.py`
   - Risk checks: `tradr/risk/manager.py`
   - Position sizing: `tradr/risk/position_sizing.py`
   - MT5 connection: `tradr/mt5/client.py`

2. Make changes with proper error handling

3. Test locally:
   ```python
   python quick_test_trades.py  # Run small backtest
   ```

4. Commit with descriptive message:
   ```bash
   git commit -m "fix: Add spread validation before trade execution"
   ```

### Task 3: "Optimize parameters"

**Quick run** (testing):
```bash
python ftmo_challenge_analyzer.py --multi --trials 10
```

**Production run** (overnight):
```bash
nohup python ftmo_challenge_analyzer.py --multi --adx --trials 500 > opt.log 2>&1 &
tail -f ftmo_analysis_output/NSGA/optimization.log
```

**Check results**:
```bash
cat ftmo_analysis_output/NSGA/optimization_report.csv
cat params/current_params.json
```

### Task 4: "Deploy to production"

**Development** (Linux):
1. Optimize parameters
2. Verify results (`params/current_params.json` updated)
3. Commit and push

**Production** (Windows VM):
1. Stop `main_live_bot.py`
2. `git pull origin main`
3. Verify `params/current_params.json` updated
4. Restart `python main_live_bot.py`
5. Monitor `logs/tradr_live.log`

---

## 📊 Understanding the Output

### Optimization Log (`ftmo_analysis_output/NSGA/optimization.log`)
```
🏆 NEW BEST - Trial #42 [2025-12-28 14:23:15]
  Score: 85.30 | R: +45.2 | Sharpe: 1.852
  Win Rate: 52.3% | PF: 1.92 | Trades: 156
  Profit: $45,200.00 | Max DD: 6.12%
  [Validation] R: +38.1 | WR: 49.8% | $38,100.00
```

**Metrics**:
- **Score**: Composite optimization score (higher = better)
- **R**: Total profit in risk units (45.2R = 45.2× average risk)
- **Sharpe**: Risk-adjusted return (>1.5 = good, >2.0 = excellent)
- **WR**: Win rate percentage (target: 60%+)
- **PF**: Profit factor (gross profit / gross loss, target: >2.0)
- **DD**: Max drawdown (FTMO limit: 10%)
- **Validation**: OOS performance (prevents overfitting)

### Trade CSV (`best_trades_final.csv`)
```csv
trade_num,symbol,direction,entry_date,entry_price,sl,tp,exit_date,exit_price,profit_r,profit_usd
1,EURUSD,BUY,2023-01-15,1.0850,1.0800,1.1000,2023-01-20,1.0920,1.4,1400.00
2,GBPUSD,SELL,2023-01-22,1.2300,1.2350,1.2150,2023-01-25,1.2180,2.4,2400.00
...
```

---

## 🛠️ Troubleshooting for AI

### Issue: "Import error: No module named 'MetaTrader5'"
**Cause**: Running on Linux (MT5 is Windows-only)  
**Fix**: This is expected. Live bot (`main_live_bot.py`) only runs on Windows. Optimizer runs anywhere.

### Issue: "Parameter not found in current_params.json"
**Cause**: Added new parameter to code but didn't run optimization  
**Fix**: Either:
1. Add default value to `params/current_params.json` manually
2. Run optimization to generate new params

### Issue: "Optuna study locked"
**Cause**: Another optimization process is running  
**Fix**: Check `ps aux | grep ftmo_challenge_analyzer` and kill if needed

### Issue: "Look-ahead bias in backtest"
**Cause**: Using future data from higher timeframes  
**Fix**: Always use `_slice_htf_by_timestamp()` when accessing weekly/monthly candles

---

## 📚 Documentation Quick Links

**For comprehensive understanding**:
1. **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design, data flow, components
2. **[STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md)** - Trading strategy details
3. **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Function signatures & usage
4. **[DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)** - Setup & deployment
5. **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - AI context

**For quick reference**:
- `README.md` - Project overview & quick start
- `scripts/README.md` - Utility scripts documentation
- `docs/CHANGELOG.md` - Recent changes

---

## 🤖 Documentation Auto-Update System

**All documentation is auto-generated from source code on every commit.**

### How it works:
1. **Docstrings** → API_REFERENCE.md (via AST parsing)
2. **current_params.json** → STRATEGY_GUIDE.md (live parameters)
3. **Git commits** → CHANGELOG.md (version history)
4. **File structure** → ARCHITECTURE.md (directory tree)

### Triggers:
- **Automatic**: GitHub Actions on push to `main`
- **Local**: Pre-commit hook (if installed)
- **Manual**: `python scripts/update_docs.py`

### When to update manually:
- After adding new functions with docstrings
- After optimizing parameters (updates strategy guide)
- After major architectural changes

**Command**: `python scripts/update_docs.py && git add docs/ && git commit -m "docs: Update"`

---

## ⚡ Quick Commands Reference

```bash
# OPTIMIZATION
python ftmo_challenge_analyzer.py --multi --trials 100     # NSGA-II (recommended)
python ftmo_challenge_analyzer.py --single --trials 100    # TPE (faster)
python ftmo_challenge_analyzer.py --status                 # Check progress
python ftmo_challenge_analyzer.py --config                 # Show config

# LIVE TRADING (Windows only)
python main_live_bot.py                                    # Start bot

# DOCUMENTATION
python scripts/update_docs.py                              # Update all docs
python scripts/update_docs.py --file STRATEGY              # Update specific doc

# VALIDATION
python scripts/validate_setup.py                           # Pre-deployment checks

# MONITORING
tail -f ftmo_analysis_output/NSGA/optimization.log         # Watch optimization
tail -f logs/tradr_live.log                                # Watch live bot
```

---

## 🎓 Learning Path for New AI Assistants

**Day 1**: Understand the basics
- Read this file completely
- Read `README.md`
- Read `.github/copilot-instructions.md`

**Day 2**: Understand the strategy
- Read `docs/STRATEGY_GUIDE.md`
- Examine `strategy_core.py` - focus on `compute_confluence()`
- Understand 7-pillar scoring system

**Day 3**: Understand optimization
- Read `docs/OPTIMIZATION_FLOW.md`
- Examine `ftmo_challenge_analyzer.py` - focus on objective functions
- Understand TPE vs NSGA-II

**Day 4**: Understand deployment
- Read `docs/DEPLOYMENT_GUIDE.md`
- Understand Windows (live) vs Linux (optimizer) split
- Learn parameter update workflow

**Day 5**: Practice
- Make a small change (add a comment to `strategy_core.py`)
- Run `python scripts/update_docs.py`
- See how documentation auto-updates
- Commit and observe GitHub Actions

---

## 🔐 Critical Rules (NEVER VIOLATE)

1. **NEVER hardcode parameters** - Always load from `current_params.json`
2. **NEVER mutate source code during optimization** - Only modify JSON files
3. **ALWAYS use symbol-specific pip values** - Never assume `0.0001`
4. **ALWAYS prevent look-ahead bias** - Slice HTF data to reference timestamp
5. **NEVER skip FTMO risk checks** - Max 10% DD, 5% daily loss are hard limits
6. **ALWAYS test before production** - Run `quick_test_trades.py` first
7. **NEVER commit without updating docs** - Run `update_docs.py` or use pre-commit hook

---

**Last Updated**: 2025-12-28 13:15:00  
**Auto-Updated By**: Documentation system (scripts/update_docs.py)  
**Manual Update**: Run `python scripts/update_docs.py` and commit result

**For human developers**: This file is AI-optimized. For comprehensive docs, see `docs/ARCHITECTURE.md`
