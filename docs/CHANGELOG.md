# Changelog

**Last Updated**: 2025-12-28  
**Auto-generated**: From git commits

---

## Recent Changes (Session: Dec 28, 2025)

### New Features
- **FTMOComplianceTracker**: Implemented FTMO compliance tracking class with:
  - Daily drawdown halt (4.5% threshold, 5% FTMO limit)
  - Total drawdown halt (9% threshold, 10% FTMO limit)
  - Consecutive loss streak halt (disabled: 999 losses)
  - Metrics-only mode for backtesting (no trade filtering)
  - Comprehensive compliance reporting
- **Parameter expansion**: Expanded optimization search space to 25+ parameters:
  - TP scaling: tp1/2/3_r_multiple (1.0-6.0R) and tp1/2/3_close_pct (0.15-0.40)
  - Filter toggles: 6 new optimizable filters (HTF, structure, Fibonacci, confirmation, displacement, candle rejection)
  - Hard constraints: TP ordering (tp1<tp2<tp3), close-sum ≤85%, ADX threshold ordering
- **Successful optimization**: 5-trial test run generated 800-1400 trades/trial (was 0 before fix)
  - Best training: +194R (1127 trades, 43% WR)
  - Best validation: +327R (498 trades, 33% WR)
  - Final backtest: +549R (1394 trades, 28% WR, Sharpe 2.80)

### Bug Fixes
- **CRITICAL**: Fixed 0-trade bug caused by aggressive filter toggles and compliance penalties
  - Filter toggles now hard-coded to False during optimization (baseline establishment)
  - Compliance penalty check disabled (trades rejected for DD breaches)
  - Consecutive loss halt set to 999 (effectively disabled)
  - Compliance tracking changed from filtering to metrics-only mode
- **CRITICAL**: Fixed `params_loader.py` - removed obsolete `liquidity_sweep_lookback` parameter causing optimizer crashes
- **CRITICAL**: Fixed metric calculations in `professional_quant_suite.py`:
  - Win rate: Removed duplicate `* 100` multiplication (was showing 4700% instead of 47%)
  - Calmar ratio: Fixed unit mismatch - now uses `max_drawdown_pct` instead of USD
  - Total return: Now returns USD value instead of percentage
- **CRITICAL**: Fixed quarterly stats display for losing trials - stats now shown even when R < 0
- **CRITICAL**: Fixed optimization.log showing incorrect R=0.0 for losing trials - now uses `overall_stats['r_total']`
- Disabled ADX filter completely (`require_adx_filter=False`) - incompatible with current strategy

### Features
- Removed validation/final backtest runs on every new best trial - now only runs once at end for top 5 trials (massive speedup)
- All 34 symbols now appear in output CSVs (fixed trade combination bug)

### Configuration Changes
- ADX regime filter disabled in `optimization_config.json` and all backtest calls
- Updated `params_loader.py` with complete StrategyParams mapping (60+ parameters)

### Documentation
- [CURRENT] docs: Add comprehensive baseline performance analysis (BASELINE_ANALYSIS.md) (2025-12-28)
- [00e8d26] docs: Add AI Assistant Quick Start Guide (2025-12-28)

### Refactoring
- [2210195] refactor: Remove december_atr_multiplier parameter and legacy logging (2025-12-28)
- [cdd082f] refactor: Final cleanup - organize codebase structure (2025-12-28)
- [b4e01f6] refactor: Clean output system with human-readable log (2025-12-28)
- [6e0ad97] refactor: Unified optimization config system (2025-12-28)
- [81b0940] refactor: reorganize project structure for better maintainability (2025-12-28)


---

## Version History

### v3.2 (2025-12-28)
**Update**: Baseline Performance Analysis

**New Documentation**:
- ✅ **BASELINE_ANALYSIS.md**: Comprehensive 15-25 page technical analysis
  - Baseline performance metrics (Trial #0: 25.9% DD, 48.6% WR, +99.88R)
  - Architecture breakdown (15-flag confluence system, ADX regime detection)
  - Parameter space mapping (19 current, 30+ hardcoded, 14 disabled filters)
  - Improvement roadmap (P0-P3 priorities with impact estimates)
  - Code quality assessment and performance projections

**Key Findings**:
- ⚠️ CRITICAL: 25.9% max drawdown exceeds FTMO 10% limit
- ⚠️ HIGH: 12+ trading filters currently disabled
- ⚠️ HIGH: Q3 seasonality problem (-80R July-September)
- 🎯 Top 3 priorities: Drawdown protection, filter enablement, Q3 fix

---

### v3.1 (2025-12-28)
**Update**: History Archiving + Trading Filters

**New Features**:
- ✅ History archiving at END of run (run_001, run_002, etc.)
- ✅ Holiday filter: blocks trades on Jan 1, Dec 24-25, Good Friday
- ✅ Opposing position filter: prevents FTMO hedging rule violations
- ✅ run_optimization.sh helper script for background runs
- ✅ Sync best_score with Optuna database on resume

**Bug Fixes**:
- Fixed CSV export to use correct output directory
- Fixed false "NEW BEST" messages when resuming studies

---

### v3.0 (2025-12-28)
**Major Update**: Unified Optimization Config + Smart NSGA-II Flow

**New Features**:
- ✅ Unified config system (params/optimization_config.json)
- ✅ Smart NSGA-II flow with OOS validation
- ✅ Separate NSGA/TPE output directories
- ✅ --single and --multi CLI flags
- ✅ OutputManager with mode-specific logging
- ✅ Comprehensive documentation auto-update system

**Optimizations**:
- 25+ parameter search space (was 15)
- Top-5 Pareto OOS validation (prevents overfitting)
- Fixed Optuna step divisibility warnings
- Default config: NSGA-II + ADX regime filtering

**Breaking Changes**:
- None (backwards compatible)

---

### v2.5 (2025-12-26)
**Update**: Parameter Space Expansion

**Changes**:
- Expanded parameter ranges (confluence 2-6, risk 0.2-1.0%)
- Added summer_risk_multiplier (Q3 drawdown protection)
- Added max_concurrent_trades limit

---

### v2.0 (2025-12-20)
**Major Update**: Regime-Adaptive Trading

**Features**:
- ADX regime detection (Trend/Range/Transition)
- Regime-specific confluence requirements
- RSI filters for range mode
- Partial exits and ATR trailing stops

---

### v1.0 (2024-06-15)
**Initial Release**: Production-Ready FTMO Bot

**Core Features**:
- 7-Pillar Confluence System
- Optuna TPE optimization
- FTMO risk management
- MT5 integration (Windows)
- 34 tradable assets

---

**Full commit history**: Run `git log --oneline` in repository root
