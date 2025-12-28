# MT5 FTMO Trading Bot - Complete System Architecture

**Last Updated**: 2025-12-28  
**Version**: 3.0 (Unified Optimization Config + Smart NSGA-II Flow)

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Optimization System](#optimization-system)
5. [Parameter Management](#parameter-management)
6. [Risk Management](#risk-management)
7. [Output Management](#output-management)
8. [Deployment Architecture](#deployment-architecture)

---

## System Overview

### Purpose
Automated trading system for FTMO 200K Challenge accounts using a 7-Pillar Confluence strategy with ADX regime detection and professional quantitative optimization.

### Key Metrics (Current Performance)
- **Account Size**: $200,000 USD
- **Win Rate**: ~48% (target: 60%+)
- **Annual Return**: ~40% (target: 100%+)
- **Max Drawdown**: <10% (FTMO hard limit)
- **Risk per Trade**: 0.2-1.0% (optimized)

### Technology Stack
- **Language**: Python 3.11+
- **Trading Platform**: MetaTrader 5 (Windows)
- **Optimization**: Optuna 3.x (TPE + NSGA-II)
- **Data Storage**: SQLite (optimization state), CSV (historical OHLCV)
- **Deployment**: Windows VM (live bot) + Linux/Replit (optimizer)

---

## Component Architecture

```
mt5bot-new/
├── main_live_bot.py              # PRODUCTION: Live MT5 trading execution
├── ftmo_challenge_analyzer.py    # OPTIMIZATION: Backtest & parameter tuning
├── strategy_core.py              # CORE: Trading strategy logic
├── config.py                     # SETTINGS: Account, symbols, contract specs
├── ftmo_config.py                # FTMO: Challenge rules, risk limits
├── symbol_mapping.py             # UTILS: OANDA ↔ FTMO symbol conversion
│
├── params/                       # PARAMETER MANAGEMENT
│   ├── optimization_config.py    # Unified config loader
│   ├── optimization_config.json  # Configuration file (DB, modes, trials)
│   ├── current_params.json       # Active strategy parameters
│   ├── params_loader.py          # Load/save parameter utilities
│   └── history/                  # Parameter version history
│
├── tradr/                        # CORE MODULES
│   ├── mt5/
│   │   ├── client.py             # MT5 API wrapper (Windows only)
│   │   └── reconnect.py          # Exponential backoff reconnection
│   ├── risk/
│   │   ├── manager.py            # FTMO drawdown tracking
│   │   └── position_sizing.py   # Lot size calculations
│   ├── strategy/
│   │   └── confluence.py         # 7-Pillar confluence system
│   └── utils/
│       └── output_manager.py     # Centralized output file management
│
├── data/                         # HISTORICAL DATA
│   ├── ohlcv/                    # OHLCV CSV files (2003-2025)
│   │   ├── EURUSD_D1_2003_2025.csv
│   │   ├── EURUSD_H4_2003_2025.csv
│   │   └── ... (34 assets × 4 timeframes)
│   ├── sr_cache/                 # Support/Resistance cache
│   └── sr_levels/                # S/R level database
│
├── ftmo_analysis_output/         # OPTIMIZATION RESULTS
│   ├── NSGA/                     # Multi-objective runs
│   │   ├── optimization.log      # Real-time NSGA-II progress
│   │   ├── best_trades_*.csv     # Trade exports
│   │   └── optimization_report.csv
│   └── TPE/                      # Single-objective runs
│       ├── optimization.log      # Real-time TPE progress
│       └── ... (same structure)
│
├── docs/                         # DOCUMENTATION
│   ├── ARCHITECTURE.md           # This file (system design)
│   ├── STRATEGY_GUIDE.md         # Trading strategy deep dive
│   ├── OPTIMIZATION_FLOW.md      # Optimization process
│   ├── API_REFERENCE.md          # Code API documentation
│   └── DEPLOYMENT_GUIDE.md       # Setup & deployment instructions
│
├── scripts/                      # UTILITIES
│   ├── update_docs.py            # Auto-generate documentation
│   ├── monitor_optimization.sh   # Watch optimization progress
│   └── validate_setup.py         # Pre-deployment validation
│
└── .github/
    ├── copilot-instructions.md   # AI assistant context
    └── workflows/
        └── update-docs.yml       # Auto-update docs on commit
```

---

## Data Flow

### 1. Historical Data Pipeline

```
OANDA API / Data Provider
         ↓
data/ohlcv/{SYMBOL}_{TIMEFRAME}_2003_2025.csv
         ↓
ftmo_challenge_analyzer.py (loads via load_ohlcv_data())
         ↓
Backtest Engine (simulate_trades())
         ↓
Trade Objects with entry/exit/profit
         ↓
OutputManager (ftmo_analysis_output/NSGA/ or TPE/)
```

### 2. Optimization → Live Bot Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: OPTIMIZATION (Linux/Replit)                         │
├─────────────────────────────────────────────────────────────┤
│ ftmo_challenge_analyzer.py                                  │
│   ├── Load params/optimization_config.json                  │
│   ├── Initialize Optuna study (SQLite DB)                   │
│   ├── Run trials (TPE or NSGA-II)                           │
│   │   ├── Suggest parameters                                │
│   │   ├── Backtest 2023-2024 (training)                     │
│   │   ├── Validate Oct-Dec 2024 (OOS)                       │
│   │   └── Score: R + Sharpe + Win Rate                      │
│   ├── Select best trial (OOS validation)                    │
│   └── SAVE → params/current_params.json                     │
└─────────────────────────────────────────────────────────────┘
                        ↓ Git sync
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: LIVE TRADING (Windows VM)                           │
├─────────────────────────────────────────────────────────────┤
│ main_live_bot.py                                            │
│   ├── LOAD params/current_params.json (startup)             │
│   ├── Connect to MT5 (FTMO broker)                          │
│   ├── Scan 34 assets every 4 hours                          │
│   ├── compute_confluence() with loaded params               │
│   │   ├── Check ADX regime (Trend/Range/Transition)         │
│   │   ├── Calculate 7-pillar confluence score               │
│   │   └── Apply parameter thresholds                        │
│   ├── Place pending orders (if signal valid)                │
│   ├── Manage positions                                      │
│   │   ├── Partial exits at loaded params['trail_activation_r']│
│   │   ├── ATR trailing stop (params['atr_trail_multiplier'])│
│   │   └── FTMO risk checks (RiskManager)                    │
│   └── Log to logs/tradr_live.log                            │
└─────────────────────────────────────────────────────────────┘
```

### 3. Parameter Update Cycle

```
Developer modifies strategy_core.py (new indicator/filter)
         ↓
Run: python ftmo_challenge_analyzer.py --multi --trials 100
         ↓
Optuna explores parameter space (25+ parameters)
         ↓
Top-5 OOS validation (prevents overfitting)
         ↓
Best OOS params → params/current_params.json
         ↓
Git commit & push
         ↓
Windows VM: git pull
         ↓
Restart main_live_bot.py (loads new params)
```

---

## Optimization System

### Dual-Mode Architecture

#### Mode 1: TPE Single-Objective (Faster)
```bash
python ftmo_challenge_analyzer.py --single --trials 100
```

**Algorithm**: Tree-structured Parzen Estimator (Bayesian optimization)  
**Objective**: Composite score = R + Sharpe_bonus + PF_bonus + WR_bonus - penalties  
**Speed**: ~1.5 min/trial  
**Output**: `ftmo_analysis_output/TPE/`

**Scoring Formula**:
```python
score = total_r
if sharpe_ratio > 1.5: score += 10
if sharpe_ratio > 2.0: score += 20
if profit_factor > 1.8: score += 15
if win_rate > 55: score += 20
if win_rate > 60: score += 30
if negative_quarters > 0: score -= (negative_quarters * 10)
if max_drawdown_pct > 15: score -= 50
```

#### Mode 2: NSGA-II Multi-Objective (Recommended for FTMO)
```bash
python ftmo_challenge_analyzer.py --multi --trials 100
```

**Algorithm**: Non-dominated Sorting Genetic Algorithm II  
**Objectives**:
1. **Maximize Total R** (profit in risk units)
2. **Maximize Sharpe Ratio** (risk-adjusted returns)
3. **Maximize Win Rate** (consistency)

**Speed**: ~1.5 min/trial  
**Output**: `ftmo_analysis_output/NSGA/`  
**Advantage**: Finds Pareto frontier - multiple optimal solutions balancing profit vs risk

**4-Phase Smart Flow** (prevents overfitting):
1. **Training** (1-2h): All trials on Jan 2023 - Sep 2024 data
2. **Top-5 OOS Validation** (10m): Validate on Oct-Dec 2024
3. **Selection**: Best OOS performer (not best training score)
4. **Final Backtest**: Full 2023-2025 verification

### Configuration File: `params/optimization_config.json`

```json
{
  "db_path": "sqlite:///ftmo_optimization.db",
  "study_name": "ftmo_unified_study",
  "use_multi_objective": true,
  "use_adx_regime_filter": true,
  "use_adx_slope_rising": false,
  "use_partial_exits": true,
  "use_atr_trailing": true,
  "use_volatility_sizing": false,
  "n_trials": 500,
  "n_startup_trials": 20,
  "timeout_hours": 48.0,
  "objectives": ["total_r", "sharpe_ratio", "win_rate"],
  "training_start": "2024-01-01",
  "training_end": "2024-09-30",
  "validation_start": "2024-10-01",
  "validation_end": "2024-12-31"
}
```

**CLI Overrides**:
- `--multi`: Force NSGA-II (even if config says TPE)
- `--single`: Force TPE (even if config says NSGA-II)
- `--adx`: Enable ADX regime filter
- `--trials N`: Set trial count
- `--config`: Show current configuration
- `--status`: Check optimization progress

### Parameter Space (25+ Parameters)

```python
# Core Risk Management
risk_per_trade_pct: 0.2-1.0%       # Position sizing
max_concurrent_trades: 3-10        # Portfolio limit

# Confluence Thresholds
min_confluence_score: 2-6          # Entry requirement (7 pillars)
min_quality_factors: 1-3           # Quality filter level

# ADX Regime Detection
adx_trend_threshold: 18-30         # Trend mode trigger
adx_range_threshold: 12-22         # Range mode trigger
trend_min_confluence: 3-6          # Required pillars in trend mode
range_min_confluence: 2-5          # Required pillars in range mode

# RSI Filters (Range Mode)
rsi_oversold_range: 15-35          # Oversold threshold
rsi_overbought_range: 65-85        # Overbought threshold

# Volatility Filters
atr_min_percentile: 30-80          # ATR percentile filter
atr_volatility_ratio: 0.5-1.5      # Current ATR vs historical

# Position Management
atr_trail_multiplier: 1.0-3.0      # Trailing stop distance
trail_activation_r: 1.0-3.0        # When to activate trail
partial_exit_at_1r: true/false     # Take profit at 1R
partial_exit_pct: 0.25-0.75        # Percentage to exit

# Seasonal Adjustments
december_atr_multiplier: 1.0-2.5   # December volatility boost
summer_risk_multiplier: 0.5-1.0    # Q3 risk reduction

# Asset-Specific
volatile_asset_boost: 1.0-2.0      # Boost for crypto/indices
use_fibonacci_zones: true/false    # Fibonacci confluence
fib_tolerance_pct: 0.5-2.0         # Fib zone width
```

### Database Schema

**SQLite Storage** (`ftmo_optimization.db`):
- **studies**: Study metadata (name, sampler, directions)
- **trials**: Trial results (number, state, values, params)
- **trial_params**: Parameter values per trial
- **trial_values**: Objective values per trial (multi-obj)

**Resumability**:
```python
storage = optuna.storages.RDBStorage(
    url="sqlite:///ftmo_optimization.db",
    heartbeat_interval=60,
    grace_period=120,
    failed_trial_callback=RetryFailedTrialCallback()
)
```

Crashes = no data loss, optimization continues from last completed trial.

---

## Parameter Management

### File Structure

```
params/
├── optimization_config.py        # Config loader class
├── optimization_config.json      # Runtime configuration
├── current_params.json           # Active parameters (live bot)
├── params_loader.py              # Utility functions
└── history/                      # Version history
    ├── params_20251228_123045.json
    ├── params_20251227_145622.json
    └── ... (timestamped backups)
```

### `current_params.json` Structure

```json
{
  "risk_per_trade_pct": 0.65,
  "min_confluence_score": 4,
  "min_quality_factors": 2,
  "adx_trend_threshold": 24,
  "adx_range_threshold": 16,
  "trend_min_confluence": 5,
  "range_min_confluence": 3,
  "rsi_oversold_range": 28,
  "rsi_overbought_range": 72,
  "atr_min_percentile": 55,
  "atr_trail_multiplier": 2.25,
  "trail_activation_r": 1.8,
  "partial_exit_at_1r": true,
  "partial_exit_pct": 0.5,
  "december_atr_multiplier": 1.75,
  "volatile_asset_boost": 1.4,
  "use_fibonacci_zones": true,
  "fib_tolerance_pct": 1.2,
  "max_concurrent_trades": 6,
  "atr_volatility_ratio": 1.1,
  "summer_risk_multiplier": 0.75
}
```

### Loading in Live Bot

```python
# main_live_bot.py
from params.params_loader import load_strategy_params

params = load_strategy_params()  # Loads current_params.json
min_conf = params['min_confluence_score']  # Use in strategy logic
```

### Saving from Optimizer

```python
# ftmo_challenge_analyzer.py
from params.params_loader import save_optimized_params

best_params = {
    'risk_per_trade_pct': trial.suggest_float('risk_per_trade_pct', 0.2, 1.0),
    # ... all 25+ parameters
}

save_optimized_params(
    params=best_params,
    metrics={'total_r': 45.2, 'sharpe': 1.85, 'win_rate': 52.3},
    backup=True  # Saves to history/ folder with timestamp
)
```

**CRITICAL**: Parameters are NEVER hardcoded in source files. Always load from JSON.

---

## Risk Management

### FTMO Challenge Rules (Hardcoded Limits)

**ftmo_config.py**:
```python
FTMO_CONFIG = {
    "account_size": 200000,
    "max_daily_loss_pct": 5.0,      # $10,000 max daily loss
    "max_total_drawdown_pct": 10.0,  # $20,000 max total drawdown
    "phase_1_target_pct": 10.0,      # $20,000 profit target
    "phase_2_target_pct": 5.0,       # $10,000 profit target
    "emergency_stop_pct": 7.0,       # Emergency halt at 7% DD
    "daily_halt_pct": 4.2            # Halt trading at 4.2% daily loss
}
```

### Pre-Trade Risk Checks

**tradr/risk/manager.py**:
```python
class RiskManager:
    def can_trade(self, symbol: str, risk_pct: float) -> Tuple[bool, str]:
        # Check 1: Daily loss limit
        if daily_loss_pct > 4.2:
            return False, "Daily loss limit exceeded"
        
        # Check 2: Total drawdown
        if total_drawdown_pct > 7.0:
            return False, "Emergency drawdown stop"
        
        # Check 3: Spread validation
        if spread > 2.0 * avg_spread:
            return False, "Abnormal spread"
        
        # Check 4: Concurrent positions
        if open_positions >= max_concurrent_trades:
            return False, "Max positions reached"
        
        return True, "OK"
```

### Position Sizing

**tradr/risk/position_sizing.py**:
```python
def calculate_lot_size(
    account_size: float,
    risk_pct: float,
    sl_pips: float,
    symbol: str
) -> float:
    """
    Calculate lot size based on:
    - Account size ($200K)
    - Risk per trade (0.2-1.0%)
    - Stop loss distance (pips)
    - Symbol-specific pip value
    
    Example:
        Account: $200,000
        Risk: 0.5% = $1,000
        SL: 50 pips
        EURUSD pip value: $10/pip (standard lot)
        
        Lot size = $1,000 / (50 pips × $10) = 2.0 lots
    """
    risk_amount = account_size * (risk_pct / 100)
    pip_value = get_contract_specs(symbol)['pip_value']
    
    lot_size = risk_amount / (sl_pips * pip_value)
    return round(lot_size, 2)
```

### Contract Specifications

**config.py** - Symbol-specific pip values:
```python
CONTRACT_SPECS = {
    # Forex (Standard: 0.0001)
    "EURUSD": {"pip_size": 0.0001, "pip_value": 10.0},
    "GBPUSD": {"pip_size": 0.0001, "pip_value": 10.0},
    
    # JPY Pairs (2 decimal)
    "USDJPY": {"pip_size": 0.01, "pip_value": 9.17},
    "EURJPY": {"pip_size": 0.01, "pip_value": 9.17},
    
    # Gold (2 decimal)
    "XAUUSD": {"pip_size": 0.01, "pip_value": 0.01},
    
    # Crypto (0 decimal)
    "BTCUSD": {"pip_size": 1.0, "pip_value": 1.0},
    
    # Indices
    "US500.cash": {"pip_size": 0.01, "pip_value": 0.01}
}
```

---

## Output Management

### Directory Structure

```
ftmo_analysis_output/
├── NSGA/                          # Multi-objective optimization runs
│   ├── optimization.log           # Real-time NSGA-II progress
│   ├── best_trades_training.csv   # Training period trades
│   ├── best_trades_validation.csv # Validation period trades
│   ├── best_trades_final.csv      # Full period trades
│   ├── monthly_stats.csv          # Monthly breakdown
│   ├── symbol_performance.csv     # Per-symbol statistics
│   └── optimization_report.csv    # Final summary report
│
└── TPE/                           # Single-objective optimization runs
    └── ... (same structure as NSGA/)
```

### OutputManager Class

**tradr/utils/output_manager.py**:

```python
om = OutputManager(optimization_mode="NSGA")

# Log trial progress
om.log_trial(
    trial_number=42,
    score=85.3,
    total_r=45.2,
    sharpe_ratio=1.85,
    win_rate=52.3,
    profit_factor=1.92,
    total_trades=156,
    profit_usd=45200,
    val_metrics={'total_r': 38.1, 'win_rate': 49.8}
)

# Save best trial trades
om.save_best_trial_trades(
    training_trades=train_trades,
    validation_trades=val_trades,
    final_trades=full_trades
)

# Generate reports
om.generate_monthly_stats(trades, "training", risk_pct=0.5)
om.generate_symbol_performance(trades)
om.generate_final_report(best_params, train_metrics, val_metrics, final_metrics)
```

### Log File Format

**ftmo_analysis_output/NSGA/optimization.log**:
```
================================================================================
FTMO OPTIMIZATION LOG - NSGA
Started: 2025-12-28 14:30:15
================================================================================

Trial #1 [2025-12-28 14:31:42]
  Score: 65.30 | R: +32.5 | Sharpe: 1.456
  Win Rate: 48.2% | PF: 1.65 | Trades: 143
  Profit: $32,500.00 | Max DD: 8.45%

--------------------------------------------------------------------------------
🏆 NEW BEST - Trial #12 [2025-12-28 14:45:23]
--------------------------------------------------------------------------------
  Score: 85.30 | R: +45.2 | Sharpe: 1.852
  Win Rate: 52.3% | PF: 1.92 | Trades: 156
  Profit: $45,200.00 | Max DD: 6.12%
  [Validation] R: +38.1 | WR: 49.8% | $38,100.00

Trial #13 [2025-12-28 14:47:08]
  Score: 72.10 | R: +38.7 | Sharpe: 1.623
  ...
```

---

## Deployment Architecture

### Development Environment (Linux/Replit)
```
Purpose: Optimization, backtesting, development
OS: Ubuntu 24.04 LTS (dev container)
Python: 3.11+
Requirements: requirements.txt (no MT5 dependencies)

Key Files:
- ftmo_challenge_analyzer.py
- strategy_core.py
- data/ohlcv/*.csv
- params/current_params.json

Workflow:
1. Edit strategy code
2. Run optimization: python ftmo_challenge_analyzer.py --multi --trials 100
3. Review results: cat ftmo_analysis_output/NSGA/optimization.log
4. Commit changes: git commit && git push
```

### Production Environment (Windows VM)
```
Purpose: Live MT5 trading
OS: Windows 10/11
Python: 3.11+
MT5: MetaTrader 5 terminal (FTMO broker)
Requirements: requirements.txt + MetaTrader5 package

Key Files:
- main_live_bot.py
- params/current_params.json (loaded at startup)
- .env (MT5_SERVER, MT5_LOGIN, MT5_PASSWORD)

Workflow:
1. Git pull latest changes
2. Verify params: cat params/current_params.json
3. Run bot: python main_live_bot.py
4. Monitor: tail -f logs/tradr_live.log
```

### Continuous Deployment

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Development (Linux)                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Modify strategy_core.py                                  │
│ 2. Run optimization (100-200 trials)                        │
│ 3. Validate results (Sharpe > 1.5, WR > 50%)                │
│ 4. git commit -m "feat: Add RSI divergence filter"          │
│ 5. git push origin main                                     │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Staging (Optional)                                  │
├─────────────────────────────────────────────────────────────┤
│ 1. Run Monte Carlo simulation (1000 runs)                   │
│ 2. Walk-forward validation (rolling windows)                │
│ 3. Parameter sensitivity analysis                           │
│ 4. Verify FTMO compliance (DD < 10%, daily < 5%)            │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Production (Windows VM)                             │
├─────────────────────────────────────────────────────────────┤
│ 1. Stop live bot                                            │
│ 2. git pull origin main                                     │
│ 3. Verify params/current_params.json updated                │
│ 4. Restart main_live_bot.py                                 │
│ 5. Monitor first 24h closely                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Monitoring & Maintenance

### Real-Time Monitoring

**Optimization Progress**:
```bash
# Watch live optimization log
tail -f ftmo_analysis_output/NSGA/optimization.log

# Check Optuna database status
python ftmo_challenge_analyzer.py --status

# Monitor system resources
htop  # CPU/RAM usage
```

**Live Bot Monitoring**:
```bash
# Watch trading log
tail -f logs/tradr_live.log

# Check MT5 connection
python -c "from tradr.mt5.client import MT5Client; MT5Client().initialize()"

# Monitor account status
python scripts/check_account_status.py
```

### Health Checks

**Pre-Deployment Validation** (`scripts/validate_setup.py`):
```python
✓ params/current_params.json exists and valid
✓ All 25 required parameters present
✓ Parameter values within valid ranges
✓ Historical data files present (34 assets × 4 TFs)
✓ MT5 connection successful (Windows only)
✓ FTMO risk limits configured correctly
✓ Contract specs loaded for all tradable assets
```

### Backup Strategy

1. **Parameter History**: Auto-saved to `params/history/` on each optimization
2. **Database Backups**: SQLite `.db` file backed up daily
3. **Trade Logs**: CSV exports stored in `ftmo_analysis_output/`
4. **Git Repository**: All code + params tracked in version control

---

## Performance Benchmarks

### Optimization Speed

| Trials | TPE Time | NSGA-II Time | Speedup |
|--------|----------|--------------|---------|
| 50     | 1.2h     | 1.5h         | 0.8x    |
| 100    | 2.3h     | 2.5h         | 0.92x   |
| 200    | 4.5h     | 4.8h         | 0.94x   |
| 500    | 10.5h    | 11.2h        | 0.94x   |

**Hardware**: 4-core CPU, 8GB RAM, SSD storage

### Backtest Coverage

- **Assets**: 34 (forex + metals + indices + crypto)
- **Timeframes**: 4 (D1, H4, W1, MN)
- **Data Points**: ~15,000 daily candles per asset
- **Total Candles**: 34 × 4 × 15,000 = ~2 million candles
- **Backtest Period**: 3 years (2023-2025)
- **Training Trades**: ~800-1200 per optimization
- **Validation Trades**: ~300-500 per optimization

---

## Troubleshooting

### Common Issues

**Issue**: "No module named 'MetaTrader5'"  
**Solution**: Windows only - install via `pip install MetaTrader5`

**Issue**: "Optuna study not found"  
**Solution**: Delete `ftmo_optimization.db` to start fresh

**Issue**: "Historical data missing for symbol X"  
**Solution**: Run `python update_csvs.py` to download missing data

**Issue**: "MT5 connection failed"  
**Solution**: Check `.env` file, verify MT5 terminal running, check credentials

**Issue**: "NSGA-II slower than expected"  
**Solution**: Reduce `n_startup_trials` in `optimization_config.json`

### Debug Mode

```bash
# Enable verbose logging
export TRADR_DEBUG=1
python ftmo_challenge_analyzer.py --trials 5

# Check parameter loading
python -c "from params.params_loader import load_strategy_params; print(load_strategy_params())"

# Validate OHLCV data
python scripts/validate_data.py
```

---

## Future Enhancements

### Planned Features
- [ ] Machine Learning trade filter (Random Forest classifier)
- [ ] Real-time Telegram notifications
- [ ] Web dashboard for monitoring (Flask/Streamlit)
- [ ] Multi-account support (Phase 1 + Phase 2 + Funded)
- [ ] Cloud deployment (AWS Lambda for optimization)
- [ ] Automated parameter drift detection
- [ ] Enhanced regime detection (HMM-based)

### Research Areas
- [ ] Deep learning for S/R level detection
- [ ] Reinforcement learning for position sizing
- [ ] Alternative data sources (sentiment, CoT)
- [ ] Multi-strategy portfolio optimization

---

## References

### External Documentation
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MetaTrader 5 Python API](https://www.mql5.com/en/docs/integration/python_metatrader5)
- [FTMO Challenge Rules](https://ftmo.com/en/faq/)

### Internal Documentation
- [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md) - 7-Pillar Confluence system
- [OPTIMIZATION_FLOW.md](OPTIMIZATION_FLOW.md) - Optimization process
- [API_REFERENCE.md](API_REFERENCE.md) - Code API docs
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Setup instructions

---

**Maintained by**: AI-assisted development team  
**Auto-update**: Triggered on code commits via `.github/workflows/update-docs.yml`  
**Manual update**: Run `python scripts/update_docs.py`
