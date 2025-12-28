# Unified Smart Optimization Flow

## Overview

This document describes the optimized NSGA-II flow that combines speed with robustness.

```
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: NSGA-II Multi-Objective (Training Only)                   │
│ • All trials run training backtest only (speed optimization)       │
│ • Output: Pareto frontier with 5-15 non-dominated solutions        │
│ • Time: ~1-2 hours for 100 trials                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Top-5 Pareto Validation (OOS Check)                       │
│ • Run Oct-Dec 2024 backtest on top 5 Pareto trials                 │
│ • Filter: OOS R > 0, WR > 45%, no blowup                           │
│ • Time: ~10 minutes                                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Top-3 Robustness Check                                    │
│ • Monte Carlo (500 sims): 95th percentile drawdown                 │
│ • Walk-Forward (3 windows): parameter stability                    │
│ • Time: ~15 minutes                                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Final Selection & Export                                  │
│ • Combined score = OOS_R * 0.4 + MC_robust * 0.3 + WF_stable * 0.3 │
│ • Full 2023-2025 backtest with winner                              │
│ • Export to params/current_params.json                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Why This Design?

### Speed Optimization
- Training-only during exploration (NSGA-II phase)
- Validation only on top 5 candidates
- Monte Carlo only on top 3 finalists

### Robustness Checks
- OOS validation prevents overfitting
- Monte Carlo tests strategy under random trade order
- Walk-forward validates parameter stability over time

## Time Estimates

| Trials | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Total |
|--------|---------|---------|---------|---------|-------|
| 50     | 1h      | 5m      | 10m     | 5m      | ~1.5h |
| 100    | 2h      | 10m     | 15m     | 5m      | ~2.5h |
| 200    | 4h      | 10m     | 15m     | 5m      | ~4.5h |
| 500    | 10h     | 10m     | 15m     | 5m      | ~11h  |

## Usage

```bash
# Default: NSGA-II with smart validation
python ftmo_challenge_analyzer.py --multi --trials 100

# With ADX regime filtering enabled
python ftmo_challenge_analyzer.py --multi --adx --trials 100

# Background run
nohup python ftmo_challenge_analyzer.py --multi --adx --trials 200 > opt.log 2>&1 &
```
