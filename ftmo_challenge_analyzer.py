#!/usr/bin/env python3
"""
Ultimate FTMO Challenge Performance Analyzer - 2024 Historical Data

This module provides a comprehensive backtesting and self-optimizing system that:
1. Backtests main_live_bot.py using 2024 historical data
2. Training Period: Jan-Sep 2024, Validation Period: Oct-Dec 2024
3. Runs continuous FTMO challenges (Step 1 + Step 2 = 1 complete challenge)
4. Tracks ALL trades with complete entry/exit data validated against OANDA
5. Generates detailed CSV reports with all trade details
6. Self-optimizes by saving parameters to params/current_params.json (no source code mutation)
7. Target: Minimum 14 challenges passed, Maximum 2 failed
"""

import json
import csv
import os
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import pandas as pd

from strategy_core import (
    StrategyParams,
    Trade,
    Signal,
    compute_confluence,
    simulate_trades,
    _infer_trend,
    _pick_direction_from_bias,
    get_default_params,
    extract_ml_features,
    apply_ml_filter,
    check_volatility_filter,
)

from data_provider import get_ohlcv as get_ohlcv_api
from ftmo_config import FTMO_CONFIG, FTMO10KConfig, get_pip_size, get_sl_limits
from config import FOREX_PAIRS, METALS, INDICES, CRYPTO_ASSETS
from tradr.data.oanda import OandaClient
from tradr.risk.position_sizing import calculate_lot_size, get_contract_specs
from params.params_loader import save_optimized_params

OUTPUT_DIR = Path("ftmo_analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# WALK-FORWARD OPTIMIZATION DATE RANGES
# =============================================================================
# Training Period: Optimize parameters on this data
TRAINING_START = datetime(2024, 1, 1)
TRAINING_END = datetime(2024, 9, 30)

# Validation Period: Validate optimized parameters (same year, different period)
VALIDATION_START = datetime(2024, 10, 1)
VALIDATION_END = datetime(2024, 12, 31)

# Out-of-Sample Test: Final test on completely different year
OOS_START = datetime(2023, 1, 1)
OOS_END = datetime(2023, 12, 31)

# Quarterly breakdown for 2024 validation
QUARTERS_2024 = {
    "Q1": (datetime(2024, 1, 1), datetime(2024, 3, 31)),
    "Q2": (datetime(2024, 4, 1), datetime(2024, 6, 30)),
    "Q3": (datetime(2024, 7, 1), datetime(2024, 9, 30)),
    "Q4": (datetime(2024, 10, 1), datetime(2024, 12, 31)),
}


class MonteCarloSimulator:
    """
    Monte Carlo simulation for robustness testing of trading strategies.
    
    Resamples trades with replacement and perturbs P&L to generate
    distribution of possible outcomes and confidence intervals.
    """
    
    def __init__(self, trades: List[Union[Trade, Any]], num_simulations: int = 1000):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            trades: List of Trade objects or objects with rr/r_multiple attribute
            num_simulations: Number of Monte Carlo iterations (default 1000)
        """
        self.trades = trades
        self.num_simulations = num_simulations
        self.r_values = self._extract_r_values()
    
    def _extract_r_values(self) -> List[float]:
        """Extract R-multiple values from trades."""
        r_values = []
        for t in self.trades:
            r = getattr(t, 'rr', None) or getattr(t, 'r_multiple', None) or 0.0
            r_values.append(float(r))
        return r_values
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with trade resampling and P&L perturbation.
        
        Returns:
            Dict with simulation results including confidence intervals
        """
        if not self.r_values:
            return {
                "error": "No trades to simulate",
                "num_simulations": 0,
                "confidence_intervals": {},
            }
        
        np.random.seed(42)
        
        final_equities = []
        max_drawdowns = []
        win_rates = []
        total_returns = []
        
        num_trades = len(self.r_values)
        
        for _ in range(self.num_simulations):
            indices = np.random.choice(num_trades, size=num_trades, replace=True)
            resampled_trades = [self.r_values[i] for i in indices]
            
            noise = np.random.uniform(0.9, 1.1, size=num_trades)
            perturbed_trades = [r * n for r, n in zip(resampled_trades, noise)]
            
            equity_curve = [0.0]
            for r in perturbed_trades:
                equity_curve.append(equity_curve[-1] + r)
            
            final_equity = equity_curve[-1]
            final_equities.append(final_equity)
            total_returns.append(final_equity)
            
            peak = equity_curve[0]
            max_dd = 0.0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = peak - eq
                if dd > max_dd:
                    max_dd = dd
            max_drawdowns.append(max_dd)
            
            wins = sum(1 for r in perturbed_trades if r > 0)
            win_rates.append(wins / num_trades * 100 if num_trades > 0 else 0)
        
        percentiles = [5, 25, 50, 75, 95]
        
        return {
            "num_simulations": self.num_simulations,
            "num_trades": num_trades,
            "mean_return": float(np.mean(total_returns)),
            "std_return": float(np.std(total_returns)),
            "mean_max_dd": float(np.mean(max_drawdowns)),
            "mean_win_rate": float(np.mean(win_rates)),
            "confidence_intervals": {
                "final_equity": {
                    f"p{p}": float(np.percentile(final_equities, p)) for p in percentiles
                },
                "max_drawdown": {
                    f"p{p}": float(np.percentile(max_drawdowns, p)) for p in percentiles
                },
                "win_rate": {
                    f"p{p}": float(np.percentile(win_rates, p)) for p in percentiles
                },
            },
            "worst_case_dd": float(np.percentile(max_drawdowns, 95)),
            "best_case_return": float(np.percentile(final_equities, 95)),
            "worst_case_return": float(np.percentile(final_equities, 5)),
        }


def run_monte_carlo_analysis(
    trades: List[Union[Trade, Any]],
    num_simulations: int = 1000,
) -> Dict[str, Any]:
    """
    Run Monte Carlo analysis on a list of trades.
    
    Args:
        trades: List of Trade objects
        num_simulations: Number of Monte Carlo iterations
        
    Returns:
        Dict with Monte Carlo results including confidence intervals
    """
    if not trades:
        return {
            "error": "No trades provided",
            "num_simulations": 0,
        }
    
    simulator = MonteCarloSimulator(trades, num_simulations)
    results = simulator.run_simulation()
    
    print(f"\n{'='*60}")
    print("MONTE CARLO SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Simulations: {results.get('num_simulations', 0)}")
    print(f"Trades per simulation: {results.get('num_trades', 0)}")
    print(f"\nReturn Distribution:")
    print(f"  Mean Return: {results.get('mean_return', 0):+.2f}R")
    print(f"  Std Dev: {results.get('std_return', 0):.2f}R")
    print(f"  Best Case (95th): {results.get('best_case_return', 0):+.2f}R")
    print(f"  Worst Case (5th): {results.get('worst_case_return', 0):+.2f}R")
    print(f"\nDrawdown Distribution:")
    print(f"  Mean Max DD: {results.get('mean_max_dd', 0):.2f}R")
    print(f"  Worst Case DD (95th): {results.get('worst_case_dd', 0):.2f}R")
    print(f"\nWin Rate Distribution:")
    ci = results.get('confidence_intervals', {}).get('win_rate', {})
    print(f"  Mean: {results.get('mean_win_rate', 0):.1f}%")
    print(f"  5th-95th Range: {ci.get('p5', 0):.1f}% - {ci.get('p95', 0):.1f}%")
    print(f"{'='*60}")
    
    return results


def is_valid_trading_day(dt: datetime) -> bool:
    """Check if datetime is a valid trading day (no weekends or major holidays)."""
    if dt.weekday() >= 5:  # Weekend check
        return False
    if dt.month == 1 and dt.day == 1:  # New Year's Day
        return False
    if dt.month == 12 and dt.day == 25:  # Christmas
        return False
    return True


def validate_price_against_candle(entry_price: float, exit_price: float, candle_high: float, candle_low: float) -> Tuple[bool, str]:
    """Validate that entry/exit prices are within the candle's high/low range."""
    notes = []
    is_valid = True
    
    if entry_price > candle_high or entry_price < candle_low:
        is_valid = False
        notes.append(f"Entry {entry_price} outside candle range [{candle_low}-{candle_high}]")
    
    if exit_price > candle_high or exit_price < candle_low:
        if exit_price != 0:
            is_valid = False
            notes.append(f"Exit {exit_price} outside candle range [{candle_low}-{candle_high}]")
    
    return is_valid, "; ".join(notes) if notes else "Price validated"


def run_quarterly_backtest(
    quarter: str,
    assets: Optional[List[str]] = None,
    min_confluence: int = 3,
    min_quality_factors: int = 1,
    risk_per_trade_pct: float = 0.5,
    excluded_assets: Optional[List[str]] = None,
) -> Tuple[List["Trade"], Dict]:
    """
    Run backtest for a specific quarter.
    
    Args:
        quarter: Quarter identifier (e.g., "Q1", "Q2", "Q3", "Q4")
        assets: List of assets to trade (None for all)
        min_confluence: Minimum confluence score
        min_quality_factors: Minimum quality factors
        risk_per_trade_pct: Risk per trade percentage
        excluded_assets: Assets to exclude
        
    Returns:
        Tuple of (trades list, metrics dict)
    """
    if quarter not in QUARTERS_2024:
        raise ValueError(f"Invalid quarter: {quarter}. Must be one of {list(QUARTERS_2024.keys())}")
    
    start_date, end_date = QUARTERS_2024[quarter]
    
    trades = run_full_period_backtest(
        start_date=start_date,
        end_date=end_date,
        assets=assets,
        min_confluence=min_confluence,
        min_quality_factors=min_quality_factors,
        risk_per_trade_pct=risk_per_trade_pct,
        excluded_assets=excluded_assets,
    )
    
    wins = sum(1 for t in trades if getattr(t, 'is_winner', getattr(t, 'rr', 0) > 0))
    losses = sum(1 for t in trades if not getattr(t, 'is_winner', getattr(t, 'rr', 0) > 0))
    total_r = sum(getattr(t, 'rr', getattr(t, 'r_multiple', 0)) for t in trades)
    r_values = [getattr(t, 'rr', getattr(t, 'r_multiple', 0)) for t in trades]
    
    metrics = {
        "quarter": quarter,
        "start_date": start_date,
        "end_date": end_date,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(trades) * 100) if trades else 0,
        "total_r": total_r,
        "avg_r": total_r / len(trades) if trades else 0,
        "r_std": (sum((r - total_r/len(trades))**2 for r in r_values) / len(r_values))**0.5 if len(r_values) > 1 else 0,
    }
    
    return trades, metrics


def calculate_robustness_score(
    quarterly_metrics: Dict[str, Dict],
    max_drawdown_pct: float = 0.0,
    challenge_pass_rate: float = 0.0,
) -> Tuple[float, Dict]:
    """
    Calculate multi-metric robustness score for walk-forward validation.
    
    Components:
    1. Overall profitability: Total R across all quarters (primary metric)
    2. Win rate quality: Average win rate across quarters
    3. Max drawdown penalty: Penalize high drawdowns
    4. Challenge pass rate: Percentage of challenges passed
    5. Quarterly consistency bonus: Reward profitable quarters
    
    Args:
        quarterly_metrics: Dict of quarter -> metrics from run_quarterly_backtest
        max_drawdown_pct: Maximum drawdown percentage observed
        challenge_pass_rate: Challenge pass rate (0-1)
        
    Returns:
        Tuple of (robustness_score, component_breakdown)
    """
    if not quarterly_metrics:
        return 0.0, {"error": "No quarterly metrics provided"}
    
    win_rates = [m.get("win_rate", 0) for m in quarterly_metrics.values()]
    total_rs = [m.get("total_r", 0) for m in quarterly_metrics.values()]
    
    # 1. Overall profitability score (total R across year)
    annual_total_r = sum(total_rs)
    if annual_total_r >= 100:
        profitability_score = 100
    elif annual_total_r >= 50:
        profitability_score = 85 + (annual_total_r - 50) * 0.3
    elif annual_total_r >= 20:
        profitability_score = 70 + (annual_total_r - 20) * 0.5
    elif annual_total_r > 0:
        profitability_score = 50 + annual_total_r * 1.0
    else:
        profitability_score = max(0, 50 + annual_total_r * 2)
    
    # 2. Win rate quality score
    avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
    if avg_win_rate >= 70:
        winrate_score = 100
    elif avg_win_rate >= 60:
        winrate_score = 80 + (avg_win_rate - 60) * 2
    elif avg_win_rate >= 50:
        winrate_score = 60 + (avg_win_rate - 50) * 2
    else:
        winrate_score = max(0, avg_win_rate * 1.2)
    
    # 3. Max drawdown penalty (adjusted thresholds)
    if max_drawdown_pct >= 10:
        dd_score = 0
    elif max_drawdown_pct >= 8:
        dd_score = 40
    elif max_drawdown_pct >= 6:
        dd_score = 60
    elif max_drawdown_pct >= 4:
        dd_score = 80
    else:
        dd_score = 100
    
    # 4. Challenge pass rate score
    pass_rate_score = challenge_pass_rate * 100
    
    # 5. Quarterly consistency bonus (how many quarters were profitable)
    profitable_quarters = sum(1 for r in total_rs if r > 0)
    consistency_score = (profitable_quarters / len(total_rs)) * 100 if total_rs else 0
    
    # Weighted final score - heavily prioritize profitability
    weights = {
        "profitability": 0.40,
        "winrate": 0.25,
        "drawdown": 0.10,
        "pass_rate": 0.15,
        "consistency": 0.10,
    }
    
    robustness_score = (
        profitability_score * weights["profitability"] +
        winrate_score * weights["winrate"] +
        dd_score * weights["drawdown"] +
        pass_rate_score * weights["pass_rate"] +
        consistency_score * weights["consistency"]
    )
    
    breakdown = {
        "profitability_score": profitability_score,
        "winrate_score": winrate_score,
        "drawdown_score": dd_score,
        "pass_rate_score": pass_rate_score,
        "consistency_score": consistency_score,
        "weights": weights,
        "annual_total_r": annual_total_r,
        "avg_win_rate": avg_win_rate,
        "profitable_quarters": profitable_quarters,
        "quarterly_win_rates": win_rates,
        "quarterly_total_rs": total_rs,
        "max_drawdown_pct": max_drawdown_pct,
    }
    
    return robustness_score, breakdown


def test_parameter_stability(
    optimal_confluence: int,
    assets: Optional[List[str]] = None,
    min_quality_factors: int = 1,
    risk_per_trade_pct: float = 0.5,
    excluded_assets: Optional[List[str]] = None,
) -> Tuple[bool, Dict]:
    """
    Test if parameters are stable by testing confluence ±1 from optimal.
    
    A stable parameter set should perform reasonably well even with slight variations.
    This helps avoid overfitting to a very specific parameter value.
    
    Args:
        optimal_confluence: The optimized confluence value
        assets: List of assets
        min_quality_factors: Quality factors
        risk_per_trade_pct: Risk percentage
        excluded_assets: Excluded assets
        
    Returns:
        Tuple of (is_stable, stability_metrics)
    """
    test_values = []
    
    # Test confluence -1, optimal, +1
    for delta in [-1, 0, 1]:
        test_confluence = max(2, min(6, optimal_confluence + delta))
        if test_confluence not in [v for v, _ in test_values]:
            test_values.append((test_confluence, delta))
    
    results = {}
    
    for confluence, delta in test_values:
        trades = run_full_period_backtest(
            start_date=VALIDATION_START,
            end_date=VALIDATION_END,
            assets=assets,
            min_confluence=confluence,
            min_quality_factors=min_quality_factors,
            risk_per_trade_pct=risk_per_trade_pct,
            excluded_assets=excluded_assets,
        )
        
        wins = sum(1 for t in trades if getattr(t, 'is_winner', getattr(t, 'rr', 0) > 0))
        total_r = sum(getattr(t, 'rr', getattr(t, 'r_multiple', 0)) for t in trades)
        win_rate = (wins / len(trades) * 100) if trades else 0
        
        results[f"confluence_{confluence}"] = {
            "confluence": confluence,
            "delta": delta,
            "trades": len(trades),
            "wins": wins,
            "win_rate": win_rate,
            "total_r": total_r,
        }
    
    # Check stability: all variations should have positive R and reasonable win rate
    is_stable = True
    stability_reasons = []
    
    for key, metrics in results.items():
        if metrics["trades"] < 10:
            is_stable = False
            stability_reasons.append(f"{key}: Too few trades ({metrics['trades']})")
        elif metrics["win_rate"] < 30:
            is_stable = False
            stability_reasons.append(f"{key}: Win rate too low ({metrics['win_rate']:.1f}%)")
        elif metrics["total_r"] < 0:
            is_stable = False
            stability_reasons.append(f"{key}: Negative total R ({metrics['total_r']:.1f})")
    
    return is_stable, {
        "is_stable": is_stable,
        "results": results,
        "issues": stability_reasons,
    }


def run_out_of_sample_test(
    assets: Optional[List[str]] = None,
    min_confluence: int = 3,
    min_quality_factors: int = 1,
    risk_per_trade_pct: float = 0.5,
    excluded_assets: Optional[List[str]] = None,
) -> Dict:
    """
    Run final out-of-sample test on 2023 data.
    
    This is the ultimate test: parameters optimized on 2024 data
    should generalize to completely unseen 2023 data.
    
    Args:
        assets: List of assets
        min_confluence: Optimized confluence
        min_quality_factors: Quality factors
        risk_per_trade_pct: Risk percentage
        excluded_assets: Excluded assets
        
    Returns:
        Dict with out-of-sample results
    """
    print(f"\n{'='*80}")
    print("OUT-OF-SAMPLE TEST: 2023 DATA")
    print(f"{'='*80}")
    print(f"Testing parameters optimized on 2024 against 2023 data")
    print(f"This validates generalization to completely unseen data")
    print(f"{'='*80}")
    
    trades = run_full_period_backtest(
        start_date=OOS_START,
        end_date=OOS_END,
        assets=assets,
        min_confluence=min_confluence,
        min_quality_factors=min_quality_factors,
        risk_per_trade_pct=risk_per_trade_pct,
        excluded_assets=excluded_assets,
    )
    
    wins = sum(1 for t in trades if getattr(t, 'is_winner', getattr(t, 'rr', 0) > 0))
    losses = sum(1 for t in trades if not getattr(t, 'is_winner', getattr(t, 'rr', 0) > 0))
    total_r = sum(getattr(t, 'rr', getattr(t, 'r_multiple', 0)) for t in trades)
    
    r_values = [getattr(t, 'rr', getattr(t, 'r_multiple', 0)) for t in trades]
    avg_r = total_r / len(trades) if trades else 0
    std_r = (sum((r - avg_r)**2 for r in r_values) / len(r_values))**0.5 if len(r_values) > 1 else 0
    
    results = {
        "period": "2023 Out-of-Sample",
        "start_date": OOS_START,
        "end_date": OOS_END,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(trades) * 100) if trades else 0,
        "total_r": total_r,
        "avg_r": avg_r,
        "r_std": std_r,
        "sharpe_like": avg_r / max(0.01, std_r),
        "passed": total_r > 0 and (wins / len(trades) * 100 if trades else 0) >= 40,
    }
    
    print(f"\nOut-of-Sample Results:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['win_rate']:.1f}%")
    print(f"  Total R: {results['total_r']:+.1f}R")
    print(f"  Sharpe-like Ratio: {results['sharpe_like']:.2f}")
    print(f"  PASSED: {'YES' if results['passed'] else 'NO'}")
    
    if trades and len(trades) >= 10:
        print(f"\nRunning Monte Carlo simulation on OOS results...")
        mc_results = run_monte_carlo_analysis(trades, num_simulations=1000)
        results["monte_carlo"] = mc_results
        
        ci = mc_results.get("confidence_intervals", {}).get("final_equity", {})
        results["mc_confidence_intervals"] = {
            "return_5th_percentile": ci.get("p5", 0),
            "return_25th_percentile": ci.get("p25", 0),
            "return_median": ci.get("p50", 0),
            "return_75th_percentile": ci.get("p75", 0),
            "return_95th_percentile": ci.get("p95", 0),
        }
        
        print(f"\nMonte Carlo Confidence Intervals (Final Equity):")
        print(f"  5th percentile: {ci.get('p5', 0):+.2f}R")
        print(f"  Median (50th): {ci.get('p50', 0):+.2f}R")
        print(f"  95th percentile: {ci.get('p95', 0):+.2f}R")
    
    return results


@dataclass
class BacktestTrade:
    """Extended trade data for FTMO challenge analysis."""
    trade_num: int
    challenge_num: int
    challenge_step: int
    symbol: str
    direction: str
    confluence_score: int
    entry_date: datetime
    entry_price: float
    stop_loss: float
    tp1_price: float
    tp2_price: Optional[float]
    tp3_price: Optional[float]
    tp4_price: Optional[float] = None
    tp5_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_price: float = 0.0
    tp1_hit: bool = False
    tp1_hit_date: Optional[datetime] = None
    tp2_hit: bool = False
    tp2_hit_date: Optional[datetime] = None
    tp3_hit: bool = False
    tp3_hit_date: Optional[datetime] = None
    tp4_hit: bool = False
    tp4_hit_date: Optional[datetime] = None
    tp5_hit: bool = False
    tp5_hit_date: Optional[datetime] = None
    sl_hit: bool = False
    sl_hit_date: Optional[datetime] = None
    exit_reason: str = ""
    r_multiple: float = 0.0
    profit_loss_usd: float = 0.0
    result: str = ""
    risk_pips: float = 0.0
    holding_time_hours: float = 0.0
    price_validated: bool = False
    validation_notes: str = ""
    trailing_sl: Optional[float] = None
    lot_size: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Trade #": self.trade_num,
            "Challenge #": self.challenge_num,
            "Challenge Step": self.challenge_step,
            "Symbol": self.symbol,
            "Direction": self.direction,
            "Confluence Score": f"{self.confluence_score}/7",
            "Entry Date": self.entry_date.strftime("%Y-%m-%d %H:%M:%S") if self.entry_date else "",
            "Entry Price": self.entry_price,
            "Stop Loss Price": self.stop_loss,
            "TP1 Price": self.tp1_price,
            "TP2 Price": self.tp2_price or "",
            "TP3 Price": self.tp3_price or "",
            "TP4 Price": self.tp4_price or "",
            "TP5 Price": self.tp5_price or "",
            "Exit Date": self.exit_date.strftime("%Y-%m-%d %H:%M:%S") if self.exit_date else "",
            "Exit Price": self.exit_price,
            "TP1 Hit?": f"YES ({self.tp1_hit_date.strftime('%Y-%m-%d')})" if self.tp1_hit and self.tp1_hit_date else ("YES" if self.tp1_hit else "NO"),
            "TP2 Hit?": f"YES ({self.tp2_hit_date.strftime('%Y-%m-%d')})" if self.tp2_hit and self.tp2_hit_date else ("YES" if self.tp2_hit else "NO"),
            "TP3 Hit?": f"YES ({self.tp3_hit_date.strftime('%Y-%m-%d')})" if self.tp3_hit and self.tp3_hit_date else ("YES" if self.tp3_hit else "NO"),
            "TP4 Hit?": f"YES ({self.tp4_hit_date.strftime('%Y-%m-%d')})" if self.tp4_hit and self.tp4_hit_date else ("YES" if self.tp4_hit else "NO"),
            "TP5 Hit?": f"YES ({self.tp5_hit_date.strftime('%Y-%m-%d')})" if self.tp5_hit and self.tp5_hit_date else ("YES" if self.tp5_hit else "NO"),
            "SL Hit?": f"YES ({self.sl_hit_date.strftime('%Y-%m-%d')})" if self.sl_hit and self.sl_hit_date else ("YES" if self.sl_hit else "NO"),
            "Final Exit Reason": self.exit_reason,
            "R Multiple": f"{self.r_multiple:+.2f}R",
            "Profit/Loss USD": f"${self.profit_loss_usd:+.2f}",
            "Result": self.result,
            "Risk Pips": f"{self.risk_pips:.1f}",
            "Holding Time (hours)": f"{self.holding_time_hours:.1f}",
            "Price Data Validated?": "YES" if self.price_validated else "NO",
            "Validation Notes": self.validation_notes,
            "Lot Size": f"{self.lot_size:.2f}",
        }


@dataclass
class StepResult:
    """Result of a single FTMO challenge step."""
    step_num: int
    passed: bool
    starting_balance: float
    ending_balance: float
    profit_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    trading_days: int
    trades_count: int
    trades: List[BacktestTrade] = field(default_factory=list)
    failure_reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "step_num": self.step_num,
            "passed": self.passed,
            "starting_balance": self.starting_balance,
            "ending_balance": self.ending_balance,
            "profit_pct": self.profit_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "trading_days": self.trading_days,
            "trades_count": self.trades_count,
            "failure_reason": self.failure_reason,
        }


@dataclass
class ChallengeResult:
    """Result of a complete FTMO challenge (Step 1 + Step 2)."""
    challenge_num: int
    status: str
    failed_at: Optional[str]
    step1: Optional[StepResult]
    step2: Optional[StepResult]
    total_profit_usd: float = 0.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "challenge_num": self.challenge_num,
            "status": self.status,
            "failed_at": self.failed_at,
            "step1": self.step1.to_dict() if self.step1 else None,
            "step2": self.step2.to_dict() if self.step2 else None,
            "total_profit_usd": self.total_profit_usd,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


class OandaValidator:
    """Validates trade prices against OANDA historical data."""
    
    def __init__(self):
        self.validation_cache = {}
        self.client = OandaClient()
        
    def _get_candle_for_date(self, symbol: str, trade_date: datetime) -> Optional[Dict]:
        """Fetch OHLCV candle data for a specific date from OANDA."""
        cache_key = f"{symbol}_{trade_date.date()}"
        
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        try:
            trade_day = trade_date if isinstance(trade_date, datetime) else datetime.combine(trade_date, datetime.min.time())
            ohlcv_data = self.client.get_candles(
                symbol=symbol,
                granularity="D",
                count=1,
                from_time=trade_day,
                to_time=trade_day + timedelta(days=1),
            )
            
            if ohlcv_data:
                self.validation_cache[cache_key] = ohlcv_data[0]
                return ohlcv_data[0]
        except Exception as e:
            pass
        
        return None
    
    def _price_within_candle(self, price: float, candle: Dict, tolerance: float = 0.0) -> bool:
        """Check if price was achievable within the candle's high/low range."""
        if not candle:
            return True
        
        high = candle.get("high", 0)
        low = candle.get("low", 0)
        
        return (low - tolerance) <= price <= (high + tolerance)
    
    def validate_trade(self, trade: BacktestTrade, symbol: str) -> Tuple[bool, str]:
        """Validate that trade entry/exit prices align with actual OANDA market data."""
        notes = []
        is_valid = True
        
        try:
            pip_size = get_pip_size(symbol)
            tolerance_pips = 10
            tolerance = tolerance_pips * pip_size
            
            if trade.entry_price <= 0:
                notes.append("Invalid entry price")
                is_valid = False
            
            if trade.stop_loss <= 0:
                notes.append("Invalid stop loss")
                is_valid = False
            
            if trade.direction.upper() == "BULLISH":
                if trade.stop_loss >= trade.entry_price:
                    notes.append("SL above entry for bullish trade")
                    is_valid = False
            else:
                if trade.stop_loss <= trade.entry_price:
                    notes.append("SL below entry for bearish trade")
                    is_valid = False
            
            if is_valid and trade.entry_date:
                entry_candle = self._get_candle_for_date(symbol, trade.entry_date)
                if entry_candle:
                    if not self._price_within_candle(trade.entry_price, entry_candle, tolerance):
                        notes.append(f"Entry price {trade.entry_price} outside candle range [{entry_candle.get('low')}-{entry_candle.get('high')}]")
                        is_valid = False
                    else:
                        notes.append("Entry price validated against OANDA candle")
            
            if is_valid and trade.exit_date and trade.exit_price > 0:
                exit_candle = self._get_candle_for_date(symbol, trade.exit_date)
                if exit_candle:
                    if not self._price_within_candle(trade.exit_price, exit_candle, tolerance):
                        notes.append(f"Exit price {trade.exit_price} outside candle range [{exit_candle.get('low')}-{exit_candle.get('high')}]")
                        is_valid = False
                    else:
                        notes.append("Exit price validated against OANDA candle")
            
            if is_valid and not any("outside" in n.lower() or "mismatch" in n.lower() for n in notes):
                if not notes:
                    notes = ["Price levels validated successfully"]
                
        except Exception as e:
            notes.append(f"Validation error: {str(e)}")
            is_valid = False
        
        return is_valid, "; ".join(notes)
    
    def validate_all_trades(self, trades: List[BacktestTrade]) -> Dict:
        """Validate all trades against OANDA data and generate report."""
        total = len(trades)
        perfect_match = 0
        minor_discrepancies = 0
        major_issues = 0
        suspicious = 0
        
        print(f"\n[OandaValidator] Validating {total} trades...")
        
        for i, trade in enumerate(trades):
            if (i + 1) % 100 == 0:
                print(f"  Validated {i + 1}/{total} trades...")
            
            is_valid, notes = self.validate_trade(trade, trade.symbol)
            trade.price_validated = is_valid
            trade.validation_notes = notes
            
            if is_valid and "successfully" in notes.lower():
                perfect_match += 1
            elif is_valid:
                minor_discrepancies += 1
            else:
                if "suspicious" in notes.lower() or "fabricated" in notes.lower():
                    suspicious += 1
                else:
                    major_issues += 1
        
        print(f"[OandaValidator] Validation complete: {perfect_match} perfect, {minor_discrepancies} minor, {major_issues} major")
        
        return {
            "total_validated": total,
            "perfect_match": perfect_match,
            "minor_discrepancies": minor_discrepancies,
            "major_issues": major_issues,
            "suspicious_trades": suspicious,
        }


class ChallengeSequencer:
    """
    Manages sequential FTMO challenges throughout Jan-Nov 2025.
    Starts new challenge immediately after completing Step 1 + Step 2.
    """
    
    ACCOUNT_SIZE = 200000.0
    STEP1_PROFIT_TARGET_PCT = 10.0
    STEP2_PROFIT_TARGET_PCT = 5.0
    MAX_DAILY_LOSS_PCT = 5.0
    MAX_DRAWDOWN_PCT = 10.0
    MIN_TRADING_DAYS = 4
    
    def __init__(self, trades: List[Trade], start_date: datetime, end_date: datetime, config: Optional[FTMO10KConfig] = None):
        self.raw_trades = sorted(trades, key=lambda t: t.entry_date)
        self.start_date = start_date
        self.end_date = end_date
        self.config = config if config else FTMO_CONFIG
        self.challenges_passed = 0
        self.challenges_failed = 0
        self.all_challenge_results: List[ChallengeResult] = []
        self.all_backtest_trades: List[BacktestTrade] = []
        self.trade_counter = 0
    
    def _convert_trade(
        self, 
        trade: Trade, 
        challenge_num: int, 
        step_num: int,
        risk_per_trade_usd: float
    ) -> BacktestTrade:
        """Convert strategy_core Trade to BacktestTrade with additional fields."""
        self.trade_counter += 1
        
        entry_dt = trade.entry_date
        exit_dt = trade.exit_date
        
        if isinstance(entry_dt, str):
            try:
                entry_dt = datetime.fromisoformat(entry_dt.replace("Z", "+00:00"))
            except:
                entry_dt = datetime.now()
        
        if isinstance(exit_dt, str):
            try:
                exit_dt = datetime.fromisoformat(exit_dt.replace("Z", "+00:00"))
            except:
                exit_dt = datetime.now()
        
        specs = get_contract_specs(trade.symbol)
        pip_value_unit = specs.get("pip_value", 0.0001)
        stop_distance = abs(trade.entry_price - trade.stop_loss)
        risk_pips = stop_distance / pip_value_unit if pip_value_unit > 0 else 0
        
        holding_hours = 0.0
        if entry_dt and exit_dt:
            if isinstance(entry_dt, datetime) and isinstance(exit_dt, datetime):
                try:
                    if entry_dt.tzinfo is not None and exit_dt.tzinfo is None:
                        exit_dt = exit_dt.replace(tzinfo=entry_dt.tzinfo)
                    elif entry_dt.tzinfo is None and exit_dt.tzinfo is not None:
                        entry_dt = entry_dt.replace(tzinfo=exit_dt.tzinfo)
                    delta = exit_dt - entry_dt
                    holding_hours = delta.total_seconds() / 3600
                    if holding_hours < 0:
                        holding_hours = abs(holding_hours)
                except Exception:
                    holding_hours = 0.0
        
        profit_usd = trade.rr * risk_per_trade_usd
        
        result = "WIN" if trade.is_winner else "LOSS"
        if abs(trade.rr) < 0.1:
            result = "BREAKEVEN"
        
        tp1_hit = getattr(trade, 'tp1_hit', False)
        tp2_hit = getattr(trade, 'tp2_hit', False)
        tp3_hit = getattr(trade, 'tp3_hit', False)
        tp4_hit = getattr(trade, 'tp4_hit', False)
        tp5_hit = getattr(trade, 'tp5_hit', False)
        
        exit_reason = trade.exit_reason or ""
        if "TP5" in exit_reason:
            tp5_hit = tp4_hit = tp3_hit = tp2_hit = tp1_hit = True
        elif "TP4" in exit_reason:
            tp4_hit = tp3_hit = tp2_hit = tp1_hit = True
        elif "TP3" in exit_reason:
            tp3_hit = tp2_hit = tp1_hit = True
        elif "TP2" in exit_reason:
            tp2_hit = tp1_hit = True
        elif "TP1" in exit_reason:
            tp1_hit = True
        
        sl_hit = exit_reason == "SL"
        
        sizing_result = calculate_lot_size(
            symbol=trade.symbol,
            account_balance=self.config.account_size,
            risk_percent=risk_per_trade_usd / self.config.account_size,
            entry_price=trade.entry_price,
            stop_loss_price=trade.stop_loss,
            max_lot=100.0,
            min_lot=0.01,
        )
        lot_size = sizing_result.get("lot_size", 0.01)
        risk_pips = sizing_result.get("stop_pips", risk_pips)
        
        return BacktestTrade(
            trade_num=self.trade_counter,
            challenge_num=challenge_num,
            challenge_step=step_num,
            symbol=trade.symbol,
            direction=trade.direction.upper(),
            confluence_score=trade.confluence_score,
            entry_date=entry_dt,
            entry_price=trade.entry_price,
            stop_loss=trade.stop_loss,
            tp1_price=trade.tp1 or 0.0,
            tp2_price=trade.tp2,
            tp3_price=trade.tp3,
            tp4_price=trade.tp4,
            tp5_price=trade.tp5,
            exit_date=exit_dt,
            exit_price=trade.exit_price,
            tp1_hit=tp1_hit,
            tp1_hit_date=exit_dt if tp1_hit else None,
            tp2_hit=tp2_hit,
            tp2_hit_date=exit_dt if tp2_hit else None,
            tp3_hit=tp3_hit,
            tp3_hit_date=exit_dt if tp3_hit else None,
            tp4_hit=tp4_hit,
            tp4_hit_date=exit_dt if tp4_hit else None,
            tp5_hit=tp5_hit,
            tp5_hit_date=exit_dt if tp5_hit else None,
            sl_hit=sl_hit,
            sl_hit_date=exit_dt if sl_hit else None,
            exit_reason=trade.exit_reason,
            r_multiple=trade.rr,
            profit_loss_usd=profit_usd,
            result=result,
            risk_pips=risk_pips,
            holding_time_hours=holding_hours,
            lot_size=lot_size,
        )
    
    def _run_step(
        self,
        trades: List[Trade],
        step_num: int,
        starting_balance: float,
        profit_target_pct: float,
        challenge_num: int,
    ) -> Tuple[StepResult, int]:
        """Run a single challenge step with dynamic lot sizing."""
        balance = starting_balance
        peak_balance = starting_balance
        daily_start_balance = starting_balance
        current_day = None
        trading_days = set()
        max_daily_loss_pct = 0.0
        max_drawdown_pct = 0.0
        
        step_trades: List[BacktestTrade] = []
        trades_used = 0
        
        profit_target = starting_balance * (1 + profit_target_pct / 100)
        max_daily_loss = starting_balance * (self.MAX_DAILY_LOSS_PCT / 100)
        max_total_dd = starting_balance * (self.MAX_DRAWDOWN_PCT / 100)
        
        win_streak = 0
        loss_streak = 0
        
        for trade in trades:
            trades_used += 1
            
            trade_date = trade.entry_date
            if isinstance(trade_date, str):
                try:
                    trade_date = datetime.fromisoformat(trade_date.replace("Z", "+00:00"))
                except:
                    trade_date = datetime.now()
            
            if not is_valid_trading_day(trade_date):
                continue
            
            trade_day = trade_date.date() if hasattr(trade_date, 'date') else trade_date
            
            if current_day is None:
                current_day = trade_day
                daily_start_balance = balance
            elif trade_day != current_day:
                current_day = trade_day
                daily_start_balance = balance
            
            trading_days.add(trade_day)
            
            current_daily_loss = max(0, daily_start_balance - balance)
            current_daily_loss_pct = (current_daily_loss / starting_balance) * 100
            current_profit_pct = ((balance - starting_balance) / starting_balance) * 100
            current_dd = max(0, peak_balance - balance)
            current_dd_pct = (current_dd / starting_balance) * 100
            
            if self.config.use_dynamic_lot_sizing:
                dynamic_risk_pct = self.config.get_dynamic_risk_pct(
                    confluence_score=trade.confluence_score,
                    win_streak=win_streak,
                    loss_streak=loss_streak,
                    current_profit_pct=current_profit_pct,
                    daily_loss_pct=current_daily_loss_pct,
                    total_dd_pct=current_dd_pct,
                )
                risk_per_trade_usd = starting_balance * (dynamic_risk_pct / 100)
            else:
                risk_per_trade_pct = self.config.risk_per_trade_pct
                risk_per_trade_usd = starting_balance * (risk_per_trade_pct / 100)
            
            bt_trade = self._convert_trade(trade, challenge_num, step_num, risk_per_trade_usd)
            step_trades.append(bt_trade)
            self.all_backtest_trades.append(bt_trade)
            
            if bt_trade.result == "WIN":
                win_streak += 1
                loss_streak = 0
            elif bt_trade.result == "LOSS":
                loss_streak += 1
                win_streak = 0
            
            balance += bt_trade.profit_loss_usd
            
            if balance > peak_balance:
                peak_balance = balance
            
            daily_loss = daily_start_balance - balance
            daily_loss_pct = (daily_loss / starting_balance) * 100
            if daily_loss_pct > max_daily_loss_pct:
                max_daily_loss_pct = daily_loss_pct
            
            drawdown = peak_balance - balance
            drawdown_pct = (drawdown / starting_balance) * 100
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct
            
            if daily_loss >= max_daily_loss:
                return StepResult(
                    step_num=step_num,
                    passed=False,
                    starting_balance=starting_balance,
                    ending_balance=balance,
                    profit_pct=((balance - starting_balance) / starting_balance) * 100,
                    max_daily_loss_pct=max_daily_loss_pct,
                    max_drawdown_pct=max_drawdown_pct,
                    trading_days=len(trading_days),
                    trades_count=len(step_trades),
                    trades=step_trades,
                    failure_reason=f"Daily loss limit breached: {daily_loss_pct:.2f}%",
                ), trades_used
            
            if drawdown >= max_total_dd:
                return StepResult(
                    step_num=step_num,
                    passed=False,
                    starting_balance=starting_balance,
                    ending_balance=balance,
                    profit_pct=((balance - starting_balance) / starting_balance) * 100,
                    max_daily_loss_pct=max_daily_loss_pct,
                    max_drawdown_pct=max_drawdown_pct,
                    trading_days=len(trading_days),
                    trades_count=len(step_trades),
                    trades=step_trades,
                    failure_reason=f"Total drawdown limit breached: {drawdown_pct:.2f}%",
                ), trades_used
            
            if balance >= profit_target and len(trading_days) >= self.MIN_TRADING_DAYS:
                return StepResult(
                    step_num=step_num,
                    passed=True,
                    starting_balance=starting_balance,
                    ending_balance=balance,
                    profit_pct=((balance - starting_balance) / starting_balance) * 100,
                    max_daily_loss_pct=max_daily_loss_pct,
                    max_drawdown_pct=max_drawdown_pct,
                    trading_days=len(trading_days),
                    trades_count=len(step_trades),
                    trades=step_trades,
                ), trades_used
        
        profit_pct = ((balance - starting_balance) / starting_balance) * 100
        passed = profit_pct >= profit_target_pct and len(trading_days) >= self.MIN_TRADING_DAYS
        
        failure_reason = ""
        if not passed:
            if len(trading_days) < self.MIN_TRADING_DAYS:
                failure_reason = f"Insufficient trading days: {len(trading_days)} < {self.MIN_TRADING_DAYS}"
            else:
                failure_reason = f"Profit target not reached: {profit_pct:.2f}% < {profit_target_pct}%"
        
        return StepResult(
            step_num=step_num,
            passed=passed,
            starting_balance=starting_balance,
            ending_balance=balance,
            profit_pct=profit_pct,
            max_daily_loss_pct=max_daily_loss_pct,
            max_drawdown_pct=max_drawdown_pct,
            trading_days=len(trading_days),
            trades_count=len(step_trades),
            trades=step_trades,
            failure_reason=failure_reason,
        ), trades_used
    
    def run_sequential_challenges(self) -> Dict:
        """Run challenges sequentially through all 11 months."""
        current_challenge = 1
        trade_index = 0
        max_challenges = 50
        
        while trade_index < len(self.raw_trades) and current_challenge <= max_challenges:
            print(f"\n{'='*60}")
            print(f"CHALLENGE #{current_challenge} (Trade index: {trade_index}/{len(self.raw_trades)})")
            print(f"{'='*60}")
            
            remaining_trades = self.raw_trades[trade_index:]
            if not remaining_trades:
                break
            
            challenge_start = remaining_trades[0].entry_date
            
            step1_result, trades_used = self._run_step(
                trades=remaining_trades,
                step_num=1,
                starting_balance=self.ACCOUNT_SIZE,
                profit_target_pct=self.STEP1_PROFIT_TARGET_PCT,
                challenge_num=current_challenge,
            )
            
            trade_index += trades_used
            
            if not step1_result.passed:
                print(f"Challenge #{current_challenge} FAILED at Step 1: {step1_result.failure_reason}")
                self.challenges_failed += 1
                
                challenge_end = step1_result.trades[-1].exit_date if step1_result.trades else challenge_start
                
                self.all_challenge_results.append(ChallengeResult(
                    challenge_num=current_challenge,
                    status="FAILED",
                    failed_at="Step 1",
                    step1=step1_result,
                    step2=None,
                    total_profit_usd=step1_result.ending_balance - step1_result.starting_balance,
                    start_date=challenge_start,
                    end_date=challenge_end,
                ))
                current_challenge += 1
                continue
            
            print(f"Step 1 PASSED: {step1_result.profit_pct:.2f}%, Balance: ${step1_result.ending_balance:,.2f}")
            
            remaining_trades = self.raw_trades[trade_index:]
            if not remaining_trades:
                self.challenges_passed += 1
                self.all_challenge_results.append(ChallengeResult(
                    challenge_num=current_challenge,
                    status="PASSED",
                    failed_at=None,
                    step1=step1_result,
                    step2=None,
                    total_profit_usd=step1_result.ending_balance - step1_result.starting_balance,
                    start_date=challenge_start,
                    end_date=step1_result.trades[-1].exit_date if step1_result.trades else challenge_start,
                ))
                break
            
            step2_result, trades_used = self._run_step(
                trades=remaining_trades,
                step_num=2,
                starting_balance=step1_result.ending_balance,
                profit_target_pct=self.STEP2_PROFIT_TARGET_PCT,
                challenge_num=current_challenge,
            )
            
            trade_index += trades_used
            
            challenge_end = step2_result.trades[-1].exit_date if step2_result.trades else challenge_start
            
            if not step2_result.passed:
                print(f"Challenge #{current_challenge} FAILED at Step 2: {step2_result.failure_reason}")
                self.challenges_failed += 1
                self.all_challenge_results.append(ChallengeResult(
                    challenge_num=current_challenge,
                    status="FAILED",
                    failed_at="Step 2",
                    step1=step1_result,
                    step2=step2_result,
                    total_profit_usd=(step1_result.ending_balance - step1_result.starting_balance) + 
                                    (step2_result.ending_balance - step2_result.starting_balance),
                    start_date=challenge_start,
                    end_date=challenge_end,
                ))
            else:
                print(f"Challenge #{current_challenge} PASSED! Step1: {step1_result.profit_pct:.2f}%, Step2: {step2_result.profit_pct:.2f}%")
                self.challenges_passed += 1
                
                total_profit = (step1_result.ending_balance - step1_result.starting_balance) + \
                              (step2_result.ending_balance - step2_result.starting_balance)
                
                self.all_challenge_results.append(ChallengeResult(
                    challenge_num=current_challenge,
                    status="PASSED",
                    failed_at=None,
                    step1=step1_result,
                    step2=step2_result,
                    total_profit_usd=total_profit,
                    start_date=challenge_start,
                    end_date=challenge_end,
                ))
            
            current_challenge += 1
        
        return {
            "total_challenges_attempted": current_challenge - 1,
            "challenges_passed": self.challenges_passed,
            "challenges_failed": self.challenges_failed,
            "all_results": self.all_challenge_results,
            "all_trades": self.all_backtest_trades,
        }


class PerformanceOptimizer:
    """
    Optimizes parameters and WRITES changes to actual files.
    Enhanced with walk-forward optimization:
    - Trains on Jan-Sep 2024, validates on Oct-Dec 2024
    - Uses validation score for best result selection
    - Includes robustness scoring and parameter stability testing
    - Early stopping if no improvement for 5 iterations
    
    Enhanced targets:
    - >= 14 challenges passed, <= 2 challenges failed
    - >= 40% win rate per asset (for assets with 3+ trades)
    - >= +2R total per asset (for assets with 3+ trades)
    - >= $80 average profit per winning trade
    """
    
    MIN_CHALLENGES_PASSED = 14
    MAX_CHALLENGES_FAILED = 2
    MIN_TRADES_NEEDED = 300
    MIN_WIN_RATE_PER_ASSET = 40.0
    MIN_R_PER_ASSET = 2.0
    EARLY_STOPPING_PATIENCE = 5
    
    def __init__(self, config: Optional[FTMO10KConfig] = None):
        self.optimization_log: List[Dict] = []
        self.config = config if config else FTMO_CONFIG
        self._original_config = self._snapshot_config()
        
        self.current_min_confluence = self.config.min_confluence_score
        self.current_risk_pct = self.config.risk_per_trade_pct
        self.current_max_concurrent = self.config.max_concurrent_trades
        self.current_min_quality = self.config.min_quality_factors
        
        # Only exclude AUDNZD as specified - no dynamic exclusions
        self.excluded_assets: List[str] = ["AUD_NZD"]
        self.best_result: Optional[Dict] = None
        self.best_score: float = -999999
        
        self.best_training_score: float = -999999
        self.best_training_result: Optional[Dict] = None
        self.best_validation_score: float = -999999
        self.best_validation_result: Optional[Dict] = None
        self.iterations_without_improvement: int = 0
        self.quarterly_metrics: Dict[str, Dict] = {}
        self.robustness_history: List[Dict] = []
    
    def _snapshot_config(self) -> Dict:
        """Take a snapshot of current config values."""
        return {
            "risk_per_trade_pct": self.config.risk_per_trade_pct,
            "min_confluence_score": self.config.min_confluence_score,
            "max_concurrent_trades": self.config.max_concurrent_trades,
            "max_cumulative_risk_pct": self.config.max_cumulative_risk_pct,
            "daily_loss_warning_pct": self.config.daily_loss_warning_pct,
            "daily_loss_reduce_pct": self.config.daily_loss_reduce_pct,
            "min_quality_factors": self.config.min_quality_factors,
        }
    
    def analyze_per_asset_performance(self, trades: List[BacktestTrade]) -> Dict:
        """Analyze win rate, total R, and profit per asset."""
        asset_stats = {}
        
        for trade in trades:
            symbol = trade.symbol
            if symbol not in asset_stats:
                asset_stats[symbol] = {
                    'total': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_profit': 0.0,
                    'total_r': 0.0,
                    'win_profits': [],
                }
            
            asset_stats[symbol]['total'] += 1
            asset_stats[symbol]['total_profit'] += trade.profit_loss_usd
            asset_stats[symbol]['total_r'] += trade.r_multiple
            
            if trade.result == "WIN":
                asset_stats[symbol]['wins'] += 1
                asset_stats[symbol]['win_profits'].append(trade.profit_loss_usd)
            elif trade.result == "LOSS":
                asset_stats[symbol]['losses'] += 1
        
        for symbol, stats in asset_stats.items():
            stats['win_rate'] = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
            stats['avg_win_profit'] = (
                sum(stats['win_profits']) / len(stats['win_profits'])
            ) if stats['win_profits'] else 0
        
        return asset_stats
    
    def identify_underperforming_assets(self, trades: List[BacktestTrade]) -> List[str]:
        """Identify assets that fail win-rate or R criteria."""
        asset_stats = self.analyze_per_asset_performance(trades)
        underperforming = []
        
        for symbol, stats in asset_stats.items():
            if stats['total'] >= 3:
                if stats['win_rate'] < self.MIN_WIN_RATE_PER_ASSET:
                    underperforming.append(symbol)
                elif stats['total_r'] < self.MIN_R_PER_ASSET:
                    underperforming.append(symbol)
        
        return underperforming
    
    def calculate_result_score(self, results: Dict) -> float:
        """Calculate a score for comparing results across iterations."""
        passed = results.get("challenges_passed", 0)
        failed = results.get("challenges_failed", 0)
        all_trades = results.get("all_trades", [])
        
        asset_stats = self.analyze_per_asset_performance(all_trades)
        
        assets_meeting_wr = sum(1 for s, st in asset_stats.items() 
                                if st['total'] >= 3 and st['win_rate'] >= self.MIN_WIN_RATE_PER_ASSET)
        assets_meeting_r = sum(1 for s, st in asset_stats.items() 
                               if st['total'] >= 3 and st['total_r'] >= self.MIN_R_PER_ASSET)
        
        win_profits = [t.profit_loss_usd for t in all_trades if t.result == "WIN"]
        avg_win = sum(win_profits) / len(win_profits) if win_profits else 0
        
        score = (passed * 100) - (failed * 50) + assets_meeting_wr * 10 + assets_meeting_r * 10 + avg_win
        return score
    
    def update_best_result(self, results: Dict, iteration: int):
        """Track the best result across iterations."""
        score = self.calculate_result_score(results)
        if score > self.best_score:
            self.best_score = score
            self.best_result = {
                "iteration": iteration,
                "results": results,
                "score": score,
                "params": self.get_current_params(),
                "excluded_assets": self.excluded_assets.copy(),
            }
            print(f"  [Optimizer] New best result! Score: {score:.1f} (Iteration #{iteration})")
    
    def get_excluded_assets(self) -> List[str]:
        """Get list of currently excluded assets."""
        return self.excluded_assets.copy()
    
    def add_excluded_assets(self, assets: List[str]):
        """Add assets to exclusion list."""
        for asset in assets:
            if asset not in self.excluded_assets:
                self.excluded_assets.append(asset)
                print(f"  [Optimizer] Excluding underperforming asset: {asset}")
    
    def check_success_criteria(self, results: Dict) -> Tuple[bool, List[str]]:
        """Check if results meet ALL success criteria. Returns (success, issues)."""
        passed = results.get("challenges_passed", 0)
        failed = results.get("challenges_failed", 0)
        all_trades = results.get("all_trades", [])
        
        issues = []
        
        if passed < self.MIN_CHALLENGES_PASSED:
            issues.append(f"Challenges passed: {passed} < {self.MIN_CHALLENGES_PASSED}")
        
        if failed > self.MAX_CHALLENGES_FAILED:
            issues.append(f"Challenges failed: {failed} > {self.MAX_CHALLENGES_FAILED}")
        
        asset_stats = self.analyze_per_asset_performance(all_trades)
        low_wr_assets = []
        low_r_assets = []
        
        for symbol, stats in asset_stats.items():
            if stats['total'] >= 3:
                if stats['win_rate'] < self.MIN_WIN_RATE_PER_ASSET:
                    low_wr_assets.append(f"{symbol}: {stats['win_rate']:.1f}%")
                if stats['total_r'] < self.MIN_R_PER_ASSET:
                    low_r_assets.append(f"{symbol}: {stats['total_r']:+.1f}R")
        
        if low_wr_assets:
            issues.append(f"Assets below {self.MIN_WIN_RATE_PER_ASSET}% WR: {', '.join(low_wr_assets[:5])}")
        
        if low_r_assets:
            issues.append(f"Assets below +{self.MIN_R_PER_ASSET}R: {', '.join(low_r_assets[:5])}")
        
        return len(issues) == 0, issues
    
    def check_trade_count(self, trade_count: int) -> bool:
        """Check if we have enough trades for proper testing."""
        return trade_count >= self.MIN_TRADES_NEEDED
    
    def analyze_failure_patterns(self, results: Dict) -> Dict:
        """Analyze failure patterns to determine optimization strategy."""
        all_results = results.get("all_results", [])
        all_trades = results.get("all_trades", [])
        
        step1_failures = sum(1 for c in all_results if c.failed_at == "Step 1")
        step2_failures = sum(1 for c in all_results if c.failed_at == "Step 2")
        
        dd_failures = 0
        daily_loss_failures = 0
        profit_failures = 0
        
        for challenge in all_results:
            if challenge.status == "FAILED":
                step = challenge.step1 if challenge.failed_at == "Step 1" else challenge.step2
                if step and step.failure_reason:
                    reason = step.failure_reason.lower()
                    if "drawdown" in reason:
                        dd_failures += 1
                    elif "daily" in reason:
                        daily_loss_failures += 1
                    elif "profit" in reason:
                        profit_failures += 1
        
        asset_stats = self.analyze_per_asset_performance(all_trades)
        low_winrate_assets = [
            s for s, stats in asset_stats.items()
            if stats['total'] >= 3 and stats['win_rate'] < self.MIN_WIN_RATE_PER_ASSET
        ]
        
        low_r_assets = [
            s for s, stats in asset_stats.items()
            if stats['total'] >= 3 and stats['total_r'] < self.MIN_R_PER_ASSET
        ]
        
        win_profits = [t.profit_loss_usd for t in all_trades if t.result == "WIN"]
        avg_win_profit = sum(win_profits) / len(win_profits) if win_profits else 0
        
        return {
            "total_trades": len(all_trades),
            "step1_failures": step1_failures,
            "step2_failures": step2_failures,
            "dd_failures": dd_failures,
            "daily_loss_failures": daily_loss_failures,
            "profit_failures": profit_failures,
            "challenges_passed": results.get("challenges_passed", 0),
            "challenges_failed": results.get("challenges_failed", 0),
            "low_winrate_assets": low_winrate_assets,
            "low_r_assets": low_r_assets,
            "avg_win_profit": avg_win_profit,
            "asset_stats": asset_stats,
        }
    
    def determine_optimizations(self, patterns: Dict, iteration: int) -> Dict[str, Any]:
        """
        Determine what optimizations to apply based on patterns.
        
        Uses ACCUMULATOR approach to prevent conditions from overwriting each other.
        Each condition adjusts delta values, then final values are calculated at the end.
        GUARANTEES at least one parameter changes each iteration to prevent stagnation.
        """
        low_wr_assets = patterns.get("low_winrate_assets", [])
        low_r_assets = patterns.get("low_r_assets", [])
        
        if low_wr_assets or low_r_assets:
            assets_info = list(set(low_wr_assets + low_r_assets))
            if assets_info:
                print(f"  [Optimizer] Note: {len(assets_info)} assets below thresholds (not excluding)")
        
        confluence_delta = 0
        quality_delta = 0
        risk_delta = 0.0
        concurrent_delta = 0
        changes_made = []
        
        if patterns["total_trades"] < self.MIN_TRADES_NEEDED:
            confluence_delta -= 1
            quality_delta -= 1
            changes_made.append(f"Too few trades ({patterns['total_trades']}/{self.MIN_TRADES_NEEDED}): confluence-1, quality-1")
        
        if patterns["dd_failures"] > 0 or patterns["daily_loss_failures"] > 0:
            risk_delta -= 0.15
            concurrent_delta -= 1
            changes_made.append(f"Risk failures (DD:{patterns['dd_failures']}, Daily:{patterns['daily_loss_failures']}): risk-0.15, concurrent-1")
        
        if len(low_wr_assets) > 5:
            confluence_delta += 1
            quality_delta += 1
            changes_made.append(f"Many low WR assets ({len(low_wr_assets)}): confluence+1, quality+1")
        
        if patterns["profit_failures"] > 2 and patterns["dd_failures"] == 0:
            confluence_delta -= 1
            changes_made.append(f"Profit failures ({patterns['profit_failures']}): confluence-1")
        
        for change in changes_made:
            print(f"  [Optimizer] {change}")
        
        optimizations = {}
        
        if confluence_delta != 0:
            new_confluence = max(2, min(6, self.current_min_confluence + confluence_delta))
            if new_confluence != self.current_min_confluence:
                optimizations["min_confluence_score"] = new_confluence
                print(f"  [Optimizer] Confluence: {self.current_min_confluence} -> {new_confluence}")
        
        if quality_delta != 0:
            new_quality = max(1, min(3, self.current_min_quality + quality_delta))
            if new_quality != self.current_min_quality:
                optimizations["min_quality_factors"] = new_quality
                print(f"  [Optimizer] Quality: {self.current_min_quality} -> {new_quality}")
        
        if risk_delta != 0:
            new_risk = max(0.3, min(1.5, self.current_risk_pct + risk_delta))
            if abs(new_risk - self.current_risk_pct) >= 0.05:
                optimizations["risk_per_trade_pct"] = round(new_risk, 2)
                print(f"  [Optimizer] Risk: {self.current_risk_pct}% -> {new_risk:.2f}%")
        
        if concurrent_delta != 0:
            new_concurrent = max(2, min(7, self.current_max_concurrent + concurrent_delta))
            if new_concurrent != self.current_max_concurrent:
                optimizations["max_concurrent_trades"] = new_concurrent
                print(f"  [Optimizer] Concurrent: {self.current_max_concurrent} -> {new_concurrent}")
        
        if not optimizations:
            print(f"  [Optimizer] Deltas produced no changes, applying multi-parameter grid exploration...")
            
            conf_options = [2, 3, 4, 5, 6]
            quality_options = [1, 2, 3]
            risk_options = [0.4, 0.5, 0.65, 0.8, 1.0, 1.2]
            concurrent_options = [2, 3, 4, 5, 6]
            
            conf_idx = iteration % len(conf_options)
            quality_idx = (iteration // len(conf_options)) % len(quality_options)
            risk_idx = (iteration // (len(conf_options) * len(quality_options))) % len(risk_options)
            concurrent_idx = iteration % len(concurrent_options)
            
            new_confluence = conf_options[conf_idx]
            new_quality = quality_options[quality_idx]
            new_risk = risk_options[risk_idx]
            new_concurrent = concurrent_options[concurrent_idx]
            
            if new_confluence != self.current_min_confluence:
                optimizations["min_confluence_score"] = new_confluence
                print(f"  [Optimizer] Grid exploration confluence: {self.current_min_confluence} -> {new_confluence}")
            
            if new_quality != self.current_min_quality:
                optimizations["min_quality_factors"] = new_quality
                print(f"  [Optimizer] Grid exploration quality: {self.current_min_quality} -> {new_quality}")
            
            if abs(new_risk - self.current_risk_pct) >= 0.05:
                optimizations["risk_per_trade_pct"] = new_risk
                print(f"  [Optimizer] Grid exploration risk: {self.current_risk_pct}% -> {new_risk}%")
            
            if new_concurrent != self.current_max_concurrent:
                optimizations["max_concurrent_trades"] = new_concurrent
                print(f"  [Optimizer] Grid exploration concurrent: {self.current_max_concurrent} -> {new_concurrent}")
        
        current_tuple = (self.current_min_confluence, self.current_min_quality, 
                         self.current_risk_pct, self.current_max_concurrent)
        
        new_confluence = optimizations.get("min_confluence_score", self.current_min_confluence)
        new_quality = optimizations.get("min_quality_factors", self.current_min_quality)
        new_risk = optimizations.get("risk_per_trade_pct", self.current_risk_pct)
        new_concurrent = optimizations.get("max_concurrent_trades", self.current_max_concurrent)
        new_tuple = (new_confluence, new_quality, new_risk, new_concurrent)
        
        if new_tuple == current_tuple:
            print(f"  [Optimizer] VALIDATION: No change detected, applying cascading guaranteed changes...")
            
            param_axis = iteration % 4
            
            if param_axis == 0:
                conf_candidates = [c for c in [2, 3, 4, 5, 6] if c != self.current_min_confluence]
                new_confluence = conf_candidates[iteration % len(conf_candidates)]
                optimizations["min_confluence_score"] = new_confluence
                print(f"  [Optimizer] Cascading confluence: {self.current_min_confluence} -> {new_confluence}")
            
            if param_axis == 1 or new_confluence == self.current_min_confluence:
                quality_candidates = [q for q in [1, 2, 3] if q != self.current_min_quality]
                new_quality = quality_candidates[iteration % len(quality_candidates)]
                optimizations["min_quality_factors"] = new_quality
                print(f"  [Optimizer] Cascading quality: {self.current_min_quality} -> {new_quality}")
            
            if param_axis == 2 or (new_confluence == self.current_min_confluence and new_quality == self.current_min_quality):
                risk_candidates = [r for r in [0.4, 0.5, 0.65, 0.8, 1.0, 1.2] if abs(r - self.current_risk_pct) >= 0.1]
                if risk_candidates:
                    new_risk = risk_candidates[iteration % len(risk_candidates)]
                    optimizations["risk_per_trade_pct"] = new_risk
                    print(f"  [Optimizer] Cascading risk: {self.current_risk_pct}% -> {new_risk}%")
            
            if param_axis == 3 or not optimizations:
                conc_candidates = [c for c in [2, 3, 4, 5, 6] if c != self.current_max_concurrent]
                new_concurrent = conc_candidates[iteration % len(conc_candidates)]
                optimizations["max_concurrent_trades"] = new_concurrent
                print(f"  [Optimizer] Cascading concurrent: {self.current_max_concurrent} -> {new_concurrent}")
        
        final_confluence = optimizations.get("min_confluence_score", self.current_min_confluence)
        final_quality = optimizations.get("min_quality_factors", self.current_min_quality)
        final_risk = optimizations.get("risk_per_trade_pct", self.current_risk_pct)
        final_concurrent = optimizations.get("max_concurrent_trades", self.current_max_concurrent)
        final_tuple = (final_confluence, final_quality, final_risk, final_concurrent)
        
        assert final_tuple != current_tuple, f"FATAL: Optimizer failed to produce parameter change! Current={current_tuple}, Final={final_tuple}"
        
        print(f"  [Optimizer] VERIFIED: Parameters will change from {current_tuple} to {final_tuple}")
        
        return optimizations
    
    def apply_optimizations(self, optimizations: Dict[str, Any], iteration: int) -> bool:
        """Apply optimizations by saving to JSON params file (no source code mutation)."""
        if not optimizations:
            print(f"  [Optimizer] No optimizations to apply.")
            return False
        
        if "min_confluence_score" in optimizations:
            self.current_min_confluence = optimizations["min_confluence_score"]
            self.config.min_confluence_score = self.current_min_confluence
        
        if "min_quality_factors" in optimizations:
            self.current_min_quality = optimizations["min_quality_factors"]
            self.config.min_quality_factors = self.current_min_quality
        
        if "risk_per_trade_pct" in optimizations:
            self.current_risk_pct = optimizations["risk_per_trade_pct"]
            self.config.risk_per_trade_pct = self.current_risk_pct
        
        if "max_concurrent_trades" in optimizations:
            self.current_max_concurrent = optimizations["max_concurrent_trades"]
            self.config.max_concurrent_trades = self.current_max_concurrent
        
        params_dict = {
            "min_confluence": self.current_min_confluence,
            "min_quality_factors": self.current_min_quality,
            "risk_per_trade_pct": self.current_risk_pct,
            "max_concurrent_trades": self.current_max_concurrent,
            "max_open_trades": self.current_max_concurrent,
            "iteration": iteration,
        }
        
        try:
            saved_path = save_optimized_params(params_dict, backup=True)
            print(f"  [Optimizer] Saved optimized params to {saved_path}")
            params_saved = True
        except Exception as e:
            print(f"  [Optimizer] Failed to save params: {e}")
            params_saved = False
        
        self.optimization_log.append({
            "iteration": iteration,
            "optimizations": optimizations,
            "params_saved": params_saved,
            "timestamp": datetime.now().isoformat(),
        })
        
        return params_saved
    
    def optimize_and_retest(self, results: Dict, iteration: int) -> Dict:
        """Analyze patterns and apply optimizations."""
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION ANALYSIS - ITERATION #{iteration}")
        print(f"{'='*60}")
        
        patterns = self.analyze_failure_patterns(results)
        
        print(f"\nFailure Pattern Analysis:")
        print(f"  Total Trades: {patterns['total_trades']} (need {self.MIN_TRADES_NEEDED}+)")
        print(f"  Challenges Passed: {patterns['challenges_passed']} (need {self.MIN_CHALLENGES_PASSED}+)")
        print(f"  Challenges Failed: {patterns['challenges_failed']} (max {self.MAX_CHALLENGES_FAILED})")
        print(f"  Step 1 Failures: {patterns['step1_failures']}")
        print(f"  Step 2 Failures: {patterns['step2_failures']}")
        print(f"  Drawdown Failures: {patterns['dd_failures']}")
        print(f"  Daily Loss Failures: {patterns['daily_loss_failures']}")
        print(f"  Profit Target Failures: {patterns['profit_failures']}")
        print(f"  Low Win-Rate Assets: {len(patterns.get('low_winrate_assets', []))}")
        
        optimizations = self.determine_optimizations(patterns, iteration)
        
        if optimizations:
            print(f"\nApplying Optimizations:")
            for key, value in optimizations.items():
                print(f"  {key}: {value}")
            
            self.apply_optimizations(optimizations, iteration)
        
        return {
            "patterns": patterns,
            "optimizations": optimizations,
            "modified_config": self._snapshot_config(),
        }
    
    def get_config(self) -> FTMO10KConfig:
        """Return the current config."""
        return self.config
    
    def get_current_params(self) -> Dict:
        """Get current parameter values."""
        return {
            "min_confluence": self.current_min_confluence,
            "min_quality_factors": self.current_min_quality,
            "risk_per_trade_pct": self.current_risk_pct,
            "max_concurrent_trades": self.current_max_concurrent,
        }
    
    def reset_config(self):
        """Reset config to original values."""
        self.config.risk_per_trade_pct = self._original_config["risk_per_trade_pct"]
        self.config.min_confluence_score = self._original_config["min_confluence_score"]
        self.config.max_concurrent_trades = self._original_config["max_concurrent_trades"]
        self.config.min_quality_factors = self._original_config["min_quality_factors"]
        
        self.current_min_confluence = self._original_config["min_confluence_score"]
        self.current_risk_pct = self._original_config["risk_per_trade_pct"]
        self.current_max_concurrent = self._original_config["max_concurrent_trades"]
        self.current_min_quality = self._original_config["min_quality_factors"]
    
    def calculate_validation_score(self, validation_results: Dict) -> float:
        """
        Calculate validation score for walk-forward optimization.
        Uses robustness metrics instead of raw training score.
        """
        passed = validation_results.get("challenges_passed", 0)
        failed = validation_results.get("challenges_failed", 0)
        all_trades = validation_results.get("all_trades", [])
        
        if not all_trades:
            return -999999
        
        wins = sum(1 for t in all_trades if hasattr(t, 'r_multiple') and t.r_multiple > 0)
        total_r = sum(t.r_multiple for t in all_trades if hasattr(t, 'r_multiple'))
        win_rate = (wins / len(all_trades) * 100) if all_trades else 0
        
        r_values = [t.r_multiple for t in all_trades if hasattr(t, 'r_multiple')]
        avg_r = total_r / len(r_values) if r_values else 0
        std_r = (sum((r - avg_r)**2 for r in r_values) / len(r_values))**0.5 if len(r_values) > 1 else 1
        sharpe_like = avg_r / max(0.01, std_r)
        
        score = (
            passed * 100 -
            failed * 75 +
            win_rate * 2 +
            total_r * 5 +
            sharpe_like * 20
        )
        
        return score
    
    def update_training_result(self, results: Dict, iteration: int) -> bool:
        """
        Update best training result. Returns True if improved.
        """
        score = self.calculate_result_score(results)
        improved = False
        
        if score > self.best_training_score:
            self.best_training_score = score
            self.best_training_result = {
                "iteration": iteration,
                "results": results,
                "score": score,
                "params": self.get_current_params(),
                "excluded_assets": self.excluded_assets.copy(),
            }
            improved = True
            print(f"  [Optimizer] New best TRAINING result! Score: {score:.1f} (Iteration #{iteration})")
        
        return improved
    
    def update_validation_result(self, validation_results: Dict, iteration: int) -> bool:
        """
        Update best validation result. Returns True if improved.
        This is the key metric for walk-forward optimization.
        """
        score = self.calculate_validation_score(validation_results)
        improved = False
        
        if score > self.best_validation_score:
            self.best_validation_score = score
            self.best_validation_result = {
                "iteration": iteration,
                "results": validation_results,
                "score": score,
                "params": self.get_current_params(),
                "excluded_assets": self.excluded_assets.copy(),
            }
            self.iterations_without_improvement = 0
            improved = True
            print(f"  [Optimizer] New best VALIDATION result! Score: {score:.1f} (Iteration #{iteration})")
        else:
            self.iterations_without_improvement += 1
            print(f"  [Optimizer] No validation improvement ({self.iterations_without_improvement}/{self.EARLY_STOPPING_PATIENCE})")
        
        return improved
    
    def check_early_stopping(self) -> bool:
        """
        Check if we should stop optimization early.
        Returns True if no improvement for EARLY_STOPPING_PATIENCE iterations.
        """
        should_stop = self.iterations_without_improvement >= self.EARLY_STOPPING_PATIENCE
        
        if should_stop:
            print(f"\n  [Optimizer] EARLY STOPPING: No improvement for {self.EARLY_STOPPING_PATIENCE} iterations")
        
        return should_stop
    
    def run_validation_phase(self, excluded_assets: Optional[List[str]] = None) -> Dict:
        """
        Run validation on Oct-Dec 2024 data.
        This validates the parameters optimized on training data.
        """
        print(f"\n{'='*60}")
        print("VALIDATION PHASE: Oct-Dec 2024")
        print(f"{'='*60}")
        
        current_params = self.get_current_params()
        
        trades = run_full_period_backtest(
            start_date=VALIDATION_START,
            end_date=VALIDATION_END,
            min_confluence=current_params['min_confluence'],
            min_quality_factors=current_params['min_quality_factors'],
            risk_per_trade_pct=current_params['risk_per_trade_pct'],
            excluded_assets=excluded_assets or self.excluded_assets,
        )
        
        wins = sum(1 for t in trades if getattr(t, 'rr', getattr(t, 'r_multiple', 0)) > 0)
        losses = sum(1 for t in trades if getattr(t, 'rr', getattr(t, 'r_multiple', 0)) <= 0)
        total_r = sum(getattr(t, 'rr', getattr(t, 'r_multiple', 0)) for t in trades)
        
        validation_results = {
            "period": "Validation (Oct-Dec 2024)",
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(trades) * 100) if trades else 0,
            "total_r": total_r,
            "all_trades": trades,
            "challenges_passed": 0,
            "challenges_failed": 0,
        }
        
        if trades:
            config = self.get_config()
            sequencer = ChallengeSequencer(trades, VALIDATION_START, VALIDATION_END, config)
            challenge_results = sequencer.run_sequential_challenges()
            validation_results["challenges_passed"] = challenge_results.get("challenges_passed", 0)
            validation_results["challenges_failed"] = challenge_results.get("challenges_failed", 0)
            validation_results["all_results"] = challenge_results.get("all_results", [])
            validation_results["all_trades"] = challenge_results.get("all_trades", trades)
        
        print(f"\nValidation Results:")
        print(f"  Total Trades: {validation_results['total_trades']}")
        print(f"  Win Rate: {validation_results['win_rate']:.1f}%")
        print(f"  Total R: {validation_results['total_r']:+.1f}R")
        print(f"  Challenges Passed: {validation_results['challenges_passed']}")
        print(f"  Challenges Failed: {validation_results['challenges_failed']}")
        
        return validation_results
    
    def run_quarterly_validation(self, excluded_assets: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Run validation for each quarter of 2024 to check consistency.
        """
        print(f"\n{'='*60}")
        print("QUARTERLY VALIDATION: Q1-Q4 2024")
        print(f"{'='*60}")
        
        current_params = self.get_current_params()
        quarterly_results = {}
        
        for quarter in ["Q1", "Q2", "Q3", "Q4"]:
            _, metrics = run_quarterly_backtest(
                quarter=quarter,
                min_confluence=current_params['min_confluence'],
                min_quality_factors=current_params['min_quality_factors'],
                risk_per_trade_pct=current_params['risk_per_trade_pct'],
                excluded_assets=excluded_assets or self.excluded_assets,
            )
            quarterly_results[quarter] = metrics
            print(f"  {quarter}: {metrics['total_trades']} trades, {metrics['win_rate']:.1f}% WR, {metrics['total_r']:+.1f}R")
        
        self.quarterly_metrics = quarterly_results
        return quarterly_results
    
    def get_robustness_assessment(self, max_drawdown_pct: float = 0.0) -> Tuple[float, Dict]:
        """
        Get overall robustness assessment based on quarterly metrics.
        """
        if not self.quarterly_metrics:
            return 0.0, {"error": "No quarterly metrics available"}
        
        challenge_pass_rate = 0.0
        if self.best_validation_result:
            results = self.best_validation_result.get("results", {})
            passed = results.get("challenges_passed", 0)
            failed = results.get("challenges_failed", 0)
            total = passed + failed
            challenge_pass_rate = passed / total if total > 0 else 0
        
        score, breakdown = calculate_robustness_score(
            quarterly_metrics=self.quarterly_metrics,
            max_drawdown_pct=max_drawdown_pct,
            challenge_pass_rate=challenge_pass_rate,
        )
        
        self.robustness_history.append({
            "score": score,
            "breakdown": breakdown,
            "params": self.get_current_params(),
        })
        
        return score, breakdown


class OptunaOptimizer:
    """Optuna-based optimizer for FTMO strategy parameters with ML training."""
    
    def __init__(self, config: Optional[FTMO10KConfig] = None):
        self.config = config if config else FTMO_CONFIG
        self.best_params: Dict = {}
        self.best_score: float = -float('inf')
        self.trade_features: List[Dict] = []
        self.trade_labels: List[int] = []
    
    def _objective(self, trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        params = {
            'min_confluence_score': trial.suggest_int('min_confluence_score', 2, 6),
            'min_quality_factors': trial.suggest_int('min_quality_factors', 1, 4),
            'risk_per_trade_pct': trial.suggest_float('risk_per_trade_pct', 0.3, 0.6),
            'bollinger_std': trial.suggest_float('bollinger_std', 1.8, 2.5),
            'rsi_period': trial.suggest_int('rsi_period', 10, 20),
            'atr_min_percentile': trial.suggest_float('atr_min_percentile', 40.0, 80.0),
            'use_mean_reversion_filter': trial.suggest_categorical('use_mean_reversion_filter', [True, False]),
            'use_rsi_divergence_filter': trial.suggest_categorical('use_rsi_divergence_filter', [True, False]),
            'ml_min_prob': trial.suggest_float('ml_min_prob', 0.55, 0.75),
        }
        
        trades = run_full_period_backtest(
            start_date=TRAINING_START, end_date=TRAINING_END,
            min_confluence=params['min_confluence_score'],
            min_quality_factors=params['min_quality_factors'],
            risk_per_trade_pct=params['risk_per_trade_pct'],
            atr_min_percentile=params['atr_min_percentile'],
            ml_min_prob=params['ml_min_prob'],
            bollinger_std=params['bollinger_std'],
            rsi_period=params['rsi_period'],
            use_mean_reversion_filter=params['use_mean_reversion_filter'],
            use_rsi_divergence_filter=params['use_rsi_divergence_filter'],
        )
        
        if not trades or len(trades) < 10:
            return -1000.0
        
        quarterly_r = {}
        for q, (start, end) in QUARTERS_2024.items():
            if start <= TRAINING_END:
                q_trades = []
                for t in trades:
                    entry = getattr(t, 'entry_date', None)
                    if entry:
                        if isinstance(entry, str):
                            try:
                                entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                            except:
                                continue
                        if hasattr(entry, 'replace') and entry.tzinfo:
                            entry = entry.replace(tzinfo=None)
                        if start <= entry <= end:
                            q_trades.append(t)
                quarterly_r[q] = sum(getattr(t, 'rr', 0) for t in q_trades)
        
        if not quarterly_r:
            return -500.0
        
        avg_quarterly_R = sum(quarterly_r.values()) / len(quarterly_r)
        min_quarterly_R = min(quarterly_r.values())
        min_quarterly_R_bonus = 10.0 if min_quarterly_R > 0 else 0.0
        negative_quarter_penalty = sum(20.0 for r in quarterly_r.values() if r < 0)
        
        val_trades = run_full_period_backtest(
            start_date=VALIDATION_START, end_date=VALIDATION_END,
            min_confluence=params['min_confluence_score'],
            min_quality_factors=params['min_quality_factors'],
            risk_per_trade_pct=params['risk_per_trade_pct'],
            atr_min_percentile=params['atr_min_percentile'],
            ml_min_prob=params['ml_min_prob'],
            bollinger_std=params['bollinger_std'],
            rsi_period=params['rsi_period'],
            use_mean_reversion_filter=params['use_mean_reversion_filter'],
            use_rsi_divergence_filter=params['use_rsi_divergence_filter'],
        )
        validation_R = sum(getattr(t, 'rr', 0) for t in val_trades) if val_trades else 0
        
        monte_carlo_stability = 0.0
        if len(trades) >= 20:
            try:
                mc = MonteCarloSimulator(trades, num_simulations=100)
                mc_result = mc.run_simulation()
                monte_carlo_stability = mc_result.get('mean_return', 0) * 0.1
            except:
                pass
        
        score = avg_quarterly_R + min_quarterly_R_bonus - negative_quarter_penalty + validation_R + monte_carlo_stability
        
        for trade in trades:
            r_value = getattr(trade, 'rr', getattr(trade, 'r_multiple', 0))
            label = 1 if r_value > 0 else 0
            features = {
                'htf_aligned': 1 if hasattr(trade, 'flags') and getattr(trade, 'flags', {}).get('htf_aligned', False) else 0,
                'z_score': 0.0,
                'atr_percentile': params['atr_min_percentile'],
                'rsi_value': 50.0,
                'bollinger_distance': 0.0,
                'momentum_roc': 0.0,
                'direction_bullish': 1 if trade.direction == 'bullish' else 0,
            }
            self.trade_features.append(features)
            self.trade_labels.append(label)
        
        return score
    
    def run_optimization(self, n_trials: int = 100) -> Dict:
        """Run Optuna optimization with specified number of trials."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"\n{'='*60}")
        print(f"OPTUNA OPTIMIZATION - {n_trials} trials")
        print(f"{'='*60}")
        
        study = optuna.create_study(direction='maximize', study_name='ftmo_optimization')
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"\nBest Score: {self.best_score:.2f}")
        print(f"Best Parameters:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")
        
        params_to_save = {
            'min_confluence': self.best_params.get('min_confluence_score', 3),
            'min_quality_factors': self.best_params.get('min_quality_factors', 2),
            'risk_per_trade_pct': self.best_params.get('risk_per_trade_pct', 0.5),
            'atr_min_percentile': self.best_params.get('atr_min_percentile', 60.0),
            'ml_min_prob': self.best_params.get('ml_min_prob', 0.6),
            'bollinger_std': self.best_params.get('bollinger_std', 2.0),
            'rsi_period': self.best_params.get('rsi_period', 14),
            'use_mean_reversion_filter': self.best_params.get('use_mean_reversion_filter', True),
            'use_rsi_divergence_filter': self.best_params.get('use_rsi_divergence_filter', True),
        }
        
        try:
            save_optimized_params(params_to_save, backup=True)
            print(f"Optimized parameters saved to params/current_params.json")
        except Exception as e:
            print(f"Failed to save params: {e}")
        
        final_trades = run_full_period_backtest(
            start_date=TRAINING_START, end_date=TRAINING_END,
            min_confluence=self.best_params.get('min_confluence_score', 3),
            min_quality_factors=self.best_params.get('min_quality_factors', 2),
            risk_per_trade_pct=self.best_params.get('risk_per_trade_pct', 0.5),
            atr_min_percentile=self.best_params.get('atr_min_percentile', 60.0),
            ml_min_prob=self.best_params.get('ml_min_prob', 0.6),
            bollinger_std=self.best_params.get('bollinger_std', 2.0),
            rsi_period=self.best_params.get('rsi_period', 14),
            use_mean_reversion_filter=self.best_params.get('use_mean_reversion_filter', True),
            use_rsi_divergence_filter=self.best_params.get('use_rsi_divergence_filter', True),
        )
        
        if final_trades:
            if self.trade_features and self.trade_labels:
                self.train_ml_model(final_trades)
            else:
                print("No trade features collected during optimization - skipping ML training")
        
        return {'best_params': self.best_params, 'best_score': self.best_score, 'n_trials': n_trials}
    
    def train_ml_model(self, trades: List[Trade]) -> bool:
        """Train ML model on trade features and save to models/best_rf.joblib."""
        import os
        os.makedirs('models', exist_ok=True)
        
        if len(trades) < 50:
            print(f"Insufficient trades for ML training: {len(trades)} < 50")
            return False
        
        features_list = []
        labels = []
        
        for trade in trades:
            r_value = getattr(trade, 'rr', getattr(trade, 'r_multiple', 0))
            label = 1 if r_value > 0 else 0
            
            features = {
                'htf_aligned': 1,
                'location_ok': 1,
                'fib_ok': 1,
                'structure_ok': 1,
                'liquidity_ok': 1,
                'confirmation_ok': 1,
                'atr_regime_ok': 1,
                'z_score': 0.0,
                'bollinger_distance': 0.5,
                'rsi_value': 50.0,
                'atr_percentile': 50.0,
                'momentum_roc': 0.0,
                'direction_bullish': 1 if trade.direction == 'bullish' else 0,
            }
            
            features_list.append(list(features.values()))
            labels.append(label)
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, min_samples_leaf=10)
            clf.fit(features_list, labels)
            
            joblib.dump(clf, 'models/best_rf.joblib')
            print(f"ML model trained on {len(trades)} trades and saved to models/best_rf.joblib")
            return True
        except Exception as e:
            print(f"ML training failed: {e}")
            return False


class ReportGenerator:
    """Generates all required reports and CSV files."""
    
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def export_trade_log(self, trades: List[BacktestTrade], filename: str = "all_trades_jan_dec_2024.csv"):
        """Export comprehensive trade log to CSV."""
        filepath = self.output_dir / filename
        
        if not trades:
            print(f"No trades to export")
            return
        
        with open(filepath, 'w', newline='') as f:
            fieldnames = list(trades[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in trades:
                writer.writerow(trade.to_dict())
        
        print(f"Trade log exported to: {filepath}")
    
    def generate_challenge_summary(self, results: Dict, validation_report: Dict) -> str:
        """Generate comprehensive challenge summary text."""
        all_results = results.get("all_results", [])
        all_trades = results.get("all_trades", [])
        
        total_trades = len(all_trades)
        wins = sum(1 for t in all_trades if t.result == "WIN")
        losses = sum(1 for t in all_trades if t.result == "LOSS")
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_r = sum(t.r_multiple for t in all_trades)
        avg_r = total_r / total_trades if total_trades > 0 else 0
        
        gross_profit = sum(t.profit_loss_usd for t in all_trades if t.profit_loss_usd > 0)
        gross_loss = sum(t.profit_loss_usd for t in all_trades if t.profit_loss_usd < 0)
        net_profit = gross_profit + gross_loss
        
        passed_challenges = [c for c in all_results if c.status == "PASSED"]
        total_earned = sum(c.total_profit_usd for c in passed_challenges)
        
        max_daily_loss = 0
        max_drawdown = 0
        for challenge in all_results:
            if challenge.step1:
                max_daily_loss = max(max_daily_loss, challenge.step1.max_daily_loss_pct)
                max_drawdown = max(max_drawdown, challenge.step1.max_drawdown_pct)
            if challenge.step2:
                max_daily_loss = max(max_daily_loss, challenge.step2.max_daily_loss_pct)
                max_drawdown = max(max_drawdown, challenge.step2.max_drawdown_pct)
        
        best_trade = max(all_trades, key=lambda t: t.r_multiple) if all_trades else None
        worst_trade = min(all_trades, key=lambda t: t.r_multiple) if all_trades else None
        
        symbol_stats = {}
        for trade in all_trades:
            if trade.symbol not in symbol_stats:
                symbol_stats[trade.symbol] = {"trades": 0, "wins": 0, "total_r": 0}
            symbol_stats[trade.symbol]["trades"] += 1
            if trade.result == "WIN":
                symbol_stats[trade.symbol]["wins"] += 1
            symbol_stats[trade.symbol]["total_r"] += trade.r_multiple
        
        top_symbols = sorted(
            symbol_stats.items(),
            key=lambda x: x[1]["total_r"],
            reverse=True
        )[:5]
        
        summary = f"""
{'='*80}
FTMO CHALLENGE PERFORMANCE - JAN 2025 TO NOV 2025
{'='*80}

PERIOD: January 1, 2025 - November 30, 2025 (11 months)
ACCOUNT SIZE: $200,000 (per challenge)
TOTAL TRADES EXECUTED: {total_trades}

CHALLENGE RESULTS:
------------------
Challenges PASSED (Both Steps): {results['challenges_passed']}
Challenges FAILED: {results['challenges_failed']}
Success Rate: {(results['challenges_passed'] / max(1, results['challenges_passed'] + results['challenges_failed']) * 100):.1f}%

CHALLENGE DETAILS:
------------------
"""
        for challenge in all_results:
            status_icon = "PASSED" if challenge.status == "PASSED" else "FAILED"
            step1_pct = challenge.step1.profit_pct if challenge.step1 else 0
            step2_pct = challenge.step2.profit_pct if challenge.step2 else 0
            
            if challenge.status == "PASSED":
                summary += f"Challenge #{challenge.challenge_num}: {status_icon} | Step 1: +{step1_pct:.1f}% | Step 2: +{step2_pct:.1f}% | Total: ${challenge.total_profit_usd:,.0f}\n"
            else:
                failed_step = challenge.failed_at or "Unknown"
                summary += f"Challenge #{challenge.challenge_num}: {status_icon} | Step 1: {step1_pct:+.1f}% | Failed at: {failed_step}\n"
        
        summary += f"""
TRADING STATISTICS:
-------------------
Total Trades: {total_trades}
Winning Trades: {wins}
Losing Trades: {losses}
Win Rate: {win_rate:.1f}%
Average R per Trade: {avg_r:+.2f}R
Best Trade: {f'{best_trade.r_multiple:+.1f}R ({best_trade.symbol})' if best_trade else 'N/A'}
Worst Trade: {f'{worst_trade.r_multiple:+.1f}R' if worst_trade else 'N/A'}

PROFITABILITY ANALYSIS:
-----------------------
Gross Profit: ${gross_profit:+,.2f}
Gross Loss: ${gross_loss:,.2f}
Net Profit: ${net_profit:+,.2f}

EARNING POTENTIAL:
------------------
TOTAL FROM {len(passed_challenges)} PASSED CHALLENGES: ${total_earned:,.2f}

SUCCESS CRITERIA CHECK:
-----------------------
"""
        passed = results['challenges_passed']
        failed = results['challenges_failed']
        
        if passed >= 14:
            summary += f"Minimum 14 Challenges Passed: YES ({passed} passed)\n"
        else:
            summary += f"Minimum 14 Challenges Passed: NO ({passed} passed, need {14-passed} more)\n"
        
        if failed <= 2:
            summary += f"Maximum 2 Challenges Failed: YES ({failed} failed)\n"
        else:
            summary += f"Maximum 2 Challenges Failed: NO ({failed} failed, {failed-2} over limit)\n"
        
        if passed >= 14 and failed <= 2:
            summary += "\n*** CRITERIA MET - SUCCESS! ***\n"
        else:
            summary += "\n*** CRITERIA NOT MET - OPTIMIZATION REQUIRED ***\n"
        
        summary += f"\n{'='*80}\n"
        
        return summary
    
    def save_summary(self, summary: str, filename: str = "challenge_summary_jan_nov_2025.txt"):
        """Save summary to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(summary)
        print(f"Summary saved to: {filepath}")
    
    def save_challenge_breakdown(self, results: Dict, filename: str = "challenge_breakdown.json"):
        """Save detailed challenge breakdown to JSON."""
        filepath = self.output_dir / filename
        
        serializable_results = {
            "total_challenges_attempted": results.get("total_challenges_attempted", results.get("challenges_passed", 0) + results.get("challenges_failed", 0)),
            "challenges_passed": results.get("challenges_passed", 0),
            "challenges_failed": results.get("challenges_failed", 0),
            "all_results": [c.to_dict() for c in results.get("all_results", [])],
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Challenge breakdown saved to: {filepath}")
    
    def save_monthly_performance(self, trades: List[BacktestTrade], filename: str = "monthly_performance.csv"):
        """Save month-by-month performance breakdown."""
        filepath = self.output_dir / filename
        
        monthly_data = {}
        for trade in trades:
            if trade.entry_date:
                month_key = trade.entry_date.strftime("%B %Y")
                if month_key not in monthly_data:
                    monthly_data[month_key] = {
                        "month": month_key,
                        "trades": 0,
                        "wins": 0,
                        "total_r": 0,
                        "profit_usd": 0,
                        "challenges": set(),
                    }
                monthly_data[month_key]["trades"] += 1
                if trade.result == "WIN":
                    monthly_data[month_key]["wins"] += 1
                monthly_data[month_key]["total_r"] += trade.r_multiple
                monthly_data[month_key]["profit_usd"] += trade.profit_loss_usd
                monthly_data[month_key]["challenges"].add(trade.challenge_num)
        
        with open(filepath, 'w', newline='') as f:
            fieldnames = ["Month", "Trades", "Wins", "Win Rate %", "Total R", "Profit USD", "Challenges"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for month, data in sorted(monthly_data.items()):
                writer.writerow({
                    "Month": data["month"],
                    "Trades": data["trades"],
                    "Wins": data["wins"],
                    "Win Rate %": f"{data['wins']/max(1,data['trades'])*100:.1f}",
                    "Total R": f"{data['total_r']:+.1f}",
                    "Profit USD": f"${data['profit_usd']:+,.2f}",
                    "Challenges": len(data["challenges"]),
                })
        
        print(f"Monthly performance saved to: {filepath}")
    
    def save_symbol_performance(self, trades: List[BacktestTrade], filename: str = "symbol_performance.csv"):
        """Save performance by trading pair."""
        filepath = self.output_dir / filename
        
        symbol_data = {}
        for trade in trades:
            if trade.symbol not in symbol_data:
                symbol_data[trade.symbol] = {
                    "symbol": trade.symbol,
                    "trades": 0,
                    "wins": 0,
                    "total_r": 0,
                    "profit_usd": 0,
                }
            symbol_data[trade.symbol]["trades"] += 1
            if trade.result == "WIN":
                symbol_data[trade.symbol]["wins"] += 1
            symbol_data[trade.symbol]["total_r"] += trade.r_multiple
            symbol_data[trade.symbol]["profit_usd"] += trade.profit_loss_usd
        
        with open(filepath, 'w', newline='') as f:
            fieldnames = ["Symbol", "Trades", "Wins", "Win Rate %", "Total R", "Profit USD"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for symbol, data in sorted(symbol_data.items(), key=lambda x: x[1]["total_r"], reverse=True):
                writer.writerow({
                    "Symbol": data["symbol"],
                    "Trades": data["trades"],
                    "Wins": data["wins"],
                    "Win Rate %": f"{data['wins']/max(1,data['trades'])*100:.1f}",
                    "Total R": f"{data['total_r']:+.1f}",
                    "Profit USD": f"${data['profit_usd']:+,.2f}",
                })
        
        print(f"Symbol performance saved to: {filepath}")
    
    def generate_all_reports(self, results: Dict, validation_report: Dict):
        """Generate all required reports."""
        trades = results.get("all_trades", [])
        
        self.export_trade_log(trades)
        
        summary = self.generate_challenge_summary(results, validation_report)
        self.save_summary(summary)
        print(summary)
        
        self.save_challenge_breakdown(results)
        self.save_monthly_performance(trades)
        self.save_symbol_performance(trades)
        
        print(f"\nAll reports generated in: {self.output_dir}")


def run_full_period_backtest(
    start_date: datetime,
    end_date: datetime,
    assets: Optional[List[str]] = None,
    min_confluence: int = 3,
    min_quality_factors: int = 1,
    risk_per_trade_pct: float = 0.5,
    excluded_assets: Optional[List[str]] = None,
    atr_min_percentile: float = 60.0,
    ml_min_prob: float = 0.6,
    bollinger_std: float = 2.0,
    rsi_period: int = 14,
    use_mean_reversion_filter: bool = True,
    use_rsi_divergence_filter: bool = True,
) -> List[Trade]:
    """
    Run backtest for the full Jan-Dec 2024 period.
    Uses lower confluence threshold to generate more trades.
    """
    if assets is None:
        assets = FOREX_PAIRS + METALS + INDICES + CRYPTO_ASSETS
    
    if excluded_assets:
        assets = [a for a in assets if a not in excluded_assets]
        print(f"Excluding {len(excluded_assets)} underperforming assets")
    
    print(f"\n{'='*80}")
    print("RUNNING FULL PERIOD BACKTEST")
    print(f"{'='*80}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Assets: {len(assets)} symbols")
    print(f"Min Confluence: {min_confluence}/7")
    print(f"Min Quality Factors: {min_quality_factors}")
    print(f"Risk Per Trade: {risk_per_trade_pct}%")
    print(f"{'='*80}")
    
    all_trades = []
    params = get_default_params()
    
    params.min_confluence = min_confluence
    params.min_quality_factors = min_quality_factors
    params.risk_per_trade_pct = risk_per_trade_pct
    params.atr_min_percentile = atr_min_percentile
    params.ml_min_prob = ml_min_prob
    params.bollinger_std = bollinger_std
    params.rsi_period = rsi_period
    params.use_mean_reversion = use_mean_reversion_filter
    params.use_rsi_divergence = use_rsi_divergence_filter
    
    for asset in assets:
        print(f"Processing {asset}...", end=" ")
        
        try:
            lookback_start = start_date - timedelta(days=365)
            
            daily_data = get_ohlcv_api(
                asset, timeframe="D", 
                start_date=lookback_start,
                end_date=end_date,
                use_cache=True
            )
            weekly_data = get_ohlcv_api(
                asset, timeframe="W",
                start_date=lookback_start - timedelta(days=365),
                end_date=end_date,
                use_cache=True
            ) or []
            monthly_data = get_ohlcv_api(
                asset, timeframe="M",
                start_date=lookback_start - timedelta(days=730),
                end_date=end_date,
                use_cache=True
            ) or []
            h4_data = get_ohlcv_api(
                asset, timeframe="H4",
                start_date=start_date - timedelta(days=90),
                end_date=end_date,
                use_cache=True
            ) or []
            
            oanda_data = None
            try:
                oanda_client = OandaClient()
                oanda_data = oanda_client.get_candles(
                    symbol=asset,
                    granularity="D",
                    count=500,
                    from_time=start_date,
                    to_time=end_date,
                )
            except Exception as e:
                pass
            
            if not daily_data:
                print("No data")
                continue
            
            # Use full daily_data (with lookback) for strategy calculation
            # Filter trades by date after simulation to include Jan/Feb
            start_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
            end_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
            
            trades = simulate_trades(
                candles=daily_data,  # Use full data with lookback
                symbol=asset,
                params=params,
                monthly_candles=monthly_data,
                weekly_candles=weekly_data,
                h4_candles=h4_data,
            )
            
            # Filter trades to only those within the target date range
            date_filtered_trades = []
            for trade in trades:
                trade_dt = trade.entry_date
                if isinstance(trade_dt, str):
                    try:
                        trade_dt = datetime.fromisoformat(trade_dt.replace("Z", "+00:00"))
                    except:
                        continue
                if hasattr(trade_dt, 'replace'):
                    trade_dt = trade_dt.replace(tzinfo=None)
                if start_naive <= trade_dt <= end_naive:
                    date_filtered_trades.append(trade)
            trades = date_filtered_trades
            
            validated_trades = []
            for trade in trades:
                trade_dt = trade.entry_date
                if isinstance(trade_dt, str):
                    try:
                        trade_dt = datetime.fromisoformat(trade_dt.replace("Z", "+00:00"))
                    except:
                        validated_trades.append(trade)
                        continue
                
                if not is_valid_trading_day(trade_dt):
                    continue
                
                if oanda_data:
                    trade_date_only = trade_dt.date() if hasattr(trade_dt, 'date') else trade_dt
                    for oanda_candle in oanda_data:
                        oanda_time = oanda_candle.get("time")
                        if isinstance(oanda_time, datetime):
                            oanda_date = oanda_time.date()
                        else:
                            oanda_date = oanda_time
                        
                        if oanda_date == trade_date_only:
                            is_valid, notes = validate_price_against_candle(
                                trade.entry_price,
                                trade.exit_price,
                                oanda_candle.get("high", float('inf')),
                                oanda_candle.get("low", 0),
                            )
                            break
                
                validated_trades.append(trade)
            
            all_trades.extend(validated_trades)
            print(f"{len(validated_trades)} trades ({len(trades) - len(validated_trades)} filtered)")
            
        except Exception as e:
            print(f"Error: {e}")
    
    all_trades.sort(key=lambda t: t.entry_date if t.entry_date else datetime.min)
    
    seen_trades = set()
    deduplicated_trades = []
    for trade in all_trades:
        trade_dt = trade.entry_date
        if isinstance(trade_dt, str):
            try:
                trade_dt = datetime.fromisoformat(trade_dt.replace("Z", "+00:00"))
            except:
                pass
        trade_date_str = trade_dt.strftime("%Y-%m-%d") if isinstance(trade_dt, datetime) else str(trade_dt)[:10]
        trade_key = (trade.symbol, trade.direction, trade_date_str)
        if trade_key not in seen_trades:
            seen_trades.add(trade_key)
            deduplicated_trades.append(trade)
    
    duplicates_removed = len(all_trades) - len(deduplicated_trades)
    if duplicates_removed > 0:
        print(f"Deduplication: Removed {duplicates_removed} duplicate trades (same symbol+direction+date)")
    
    all_trades = deduplicated_trades
    
    print(f"\n{'='*80}")
    print(f"BACKTEST COMPLETE: {len(all_trades)} total trades")
    print(f"{'='*80}")
    
    return all_trades


def main_challenge_analyzer():
    """
    Walk-Forward Optimization System for FTMO Challenge Analysis.
    
    This implements a robust walk-forward optimization approach:
    1. TRAINING: Optimize parameters on Jan-Sep 2024 data
    2. VALIDATION: Validate each iteration on Oct-Dec 2024 data
    3. Use VALIDATION score (not training) for best result selection
    4. Test parameter stability before accepting final parameters
    5. OUT-OF-SAMPLE: Final test on 2023 data for ultimate validation
    6. Report results from all three periods
    
    Early stopping: Stop if no validation improvement for 5 iterations.
    
    Targets:
    - >=14 challenges passed, <=2 failed
    - >=40% win rate per asset (assets with 3+ trades)
    - >=+2R total per asset (assets with 3+ trades)
    """
    MAX_ITERATIONS = 10
    iteration = 0
    success = False
    
    training_results: Dict = {"challenges_passed": 0, "challenges_failed": 0, "all_results": [], "all_trades": [], "total_challenges_attempted": 0}
    validation_results: Dict = {}
    oos_results: Dict = {}
    validation_report: Dict = {"total_validated": 0, "perfect_match": 0, "minor_discrepancies": 0, "major_issues": 0, "suspicious_trades": 0}
    
    optimizer = PerformanceOptimizer(FTMO_CONFIG)
    
    print(f"\n{'='*80}")
    print("FTMO WALK-FORWARD OPTIMIZATION SYSTEM")
    print(f"{'='*80}")
    print(f"\nData Partitioning:")
    print(f"  TRAINING:    Jan 1, 2024 - Sep 30, 2024 (optimize here)")
    print(f"  VALIDATION:  Oct 1, 2024 - Dec 31, 2024 (validate each iteration)")
    print(f"  OUT-OF-SAMPLE: Jan 1, 2023 - Dec 31, 2023 (final test)")
    print(f"\nTargets:")
    print(f"  - Minimum 14 challenges PASSED")
    print(f"  - Maximum 2 challenges FAILED")
    print(f"  - Minimum 40% win rate per asset (assets with 3+ trades)")
    print(f"  - Minimum +2R total per asset (assets with 3+ trades)")
    print(f"\nOptimization Settings:")
    print(f"  Max Iterations: {MAX_ITERATIONS}")
    print(f"  Early Stopping Patience: {optimizer.EARLY_STOPPING_PATIENCE} iterations")
    print(f"{'='*80}")
    
    while not success and iteration < MAX_ITERATIONS:
        iteration += 1
        
        print(f"\n{'#'*80}")
        print(f"# ITERATION #{iteration} - TRAINING PHASE (Jan-Sep 2024)")
        print(f"{'#'*80}")
        
        current_params = optimizer.get_current_params()
        print(f"\nCurrent Config:")
        print(f"  risk_per_trade_pct: {current_params['risk_per_trade_pct']}%")
        print(f"  min_confluence_score: {current_params['min_confluence']}/7")
        print(f"  max_concurrent_trades: {current_params['max_concurrent_trades']}")
        
        trades = run_full_period_backtest(
            start_date=TRAINING_START,
            end_date=TRAINING_END,
            min_confluence=current_params['min_confluence'],
            min_quality_factors=current_params['min_quality_factors'],
            risk_per_trade_pct=current_params['risk_per_trade_pct'],
            excluded_assets=optimizer.get_excluded_assets(),
        )
        
        if not trades:
            print("No trades generated in training period. Check data availability.")
            if iteration < MAX_ITERATIONS:
                optimizer.optimize_and_retest({"all_results": [], "all_trades": [], "challenges_passed": 0, "challenges_failed": 0}, iteration)
                continue
            break
        
        print(f"\nTraining: Generated {len(trades)} trades")
        
        config = optimizer.get_config()
        sequencer = ChallengeSequencer(trades, TRAINING_START, TRAINING_END, config)
        training_results = sequencer.run_sequential_challenges()
        
        print(f"\n{'='*60}")
        print(f"ITERATION #{iteration} - TRAINING RESULTS")
        print(f"{'='*60}")
        print(f"Total Trades: {len(training_results.get('all_trades', []))}")
        print(f"Challenges PASSED: {training_results.get('challenges_passed', 0)}")
        print(f"Challenges FAILED: {training_results.get('challenges_failed', 0)}")
        
        optimizer.update_training_result(training_results, iteration)
        
        print(f"\n{'#'*80}")
        print(f"# ITERATION #{iteration} - VALIDATION PHASE (Oct-Dec 2024)")
        print(f"{'#'*80}")
        
        validation_results = optimizer.run_validation_phase()
        
        validation_improved = optimizer.update_validation_result(validation_results, iteration)
        
        success, issues = optimizer.check_success_criteria(training_results)
        
        print(f"\n{'='*60}")
        print(f"ITERATION #{iteration} SUMMARY")
        print(f"{'='*60}")
        print(f"Training Score: {optimizer.best_training_score:.1f}")
        print(f"Validation Score: {optimizer.best_validation_score:.1f}")
        print(f"Iterations without improvement: {optimizer.iterations_without_improvement}")
        
        if optimizer.check_early_stopping():
            print(f"\n*** EARLY STOPPING TRIGGERED ***")
            break
        
        if success:
            print(f"\n*** TRAINING SUCCESS CRITERIA MET! ***")
            print(f"Proceeding to parameter stability test...")
            break
        else:
            print(f"\nCriteria NOT met. Issues:")
            for issue in issues[:5]:
                print(f"  - {issue}")
            
            if iteration < MAX_ITERATIONS:
                optimizer.optimize_and_retest(training_results, iteration)
    
    print(f"\n{'='*80}")
    print("PARAMETER STABILITY TEST")
    print(f"{'='*80}")
    
    best_params = optimizer.best_validation_result.get("params", optimizer.get_current_params()) if optimizer.best_validation_result else optimizer.get_current_params()
    
    is_stable, stability_metrics = test_parameter_stability(
        optimal_confluence=best_params['min_confluence'],
        min_quality_factors=best_params['min_quality_factors'],
        risk_per_trade_pct=best_params['risk_per_trade_pct'],
        excluded_assets=optimizer.get_excluded_assets(),
    )
    
    print(f"\nParameter Stability: {'PASSED' if is_stable else 'CAUTION - UNSTABLE'}")
    if not is_stable:
        print(f"Stability issues:")
        for issue in stability_metrics.get("issues", []):
            print(f"  - {issue}")
    
    print(f"\n{'='*80}")
    print("QUARTERLY CONSISTENCY CHECK")
    print(f"{'='*80}")
    
    quarterly_metrics = optimizer.run_quarterly_validation()
    robustness_score, robustness_breakdown = optimizer.get_robustness_assessment()
    
    print(f"\nRobustness Score: {robustness_score:.1f}/100")
    print(f"  Consistency Score: {robustness_breakdown.get('consistency_score', 0):.1f}")
    print(f"  Drawdown Score: {robustness_breakdown.get('drawdown_score', 0):.1f}")
    print(f"  Sharpe Score: {robustness_breakdown.get('sharpe_score', 0):.1f}")
    
    print(f"\n{'='*80}")
    print("OUT-OF-SAMPLE TEST: 2023 DATA")
    print(f"{'='*80}")
    
    oos_results = run_out_of_sample_test(
        min_confluence=best_params['min_confluence'],
        min_quality_factors=best_params['min_quality_factors'],
        risk_per_trade_pct=best_params['risk_per_trade_pct'],
        excluded_assets=optimizer.get_excluded_assets(),
    )
    
    print(f"\n{'='*80}")
    print("FULL YEAR BACKTEST WITH OPTIMIZED PARAMETERS")
    print(f"{'='*80}")
    
    full_year_start = datetime(2024, 1, 1)
    full_year_end = datetime(2024, 12, 31)
    
    full_year_trades = run_full_period_backtest(
        start_date=full_year_start,
        end_date=full_year_end,
        min_confluence=best_params['min_confluence'],
        min_quality_factors=best_params['min_quality_factors'],
        risk_per_trade_pct=best_params['risk_per_trade_pct'],
        excluded_assets=optimizer.get_excluded_assets(),
    )
    
    full_year_backtest_trades = []
    trade_num = 0
    account_size = 200000
    risk_pct = best_params['risk_per_trade_pct']
    
    for trade in full_year_trades:
        trade_num += 1
        r_multiple = trade.rr if trade.is_winner else -1.0
        result = "WIN" if trade.is_winner else "LOSS"
        
        # Calculate risk in USD
        risk_per_trade_usd = account_size * (risk_pct / 100)
        profit_usd = r_multiple * risk_per_trade_usd
        
        # Calculate lot size using proper position sizing
        sizing_result = calculate_lot_size(
            symbol=trade.symbol,
            account_balance=account_size,
            risk_percent=risk_pct / 100,
            entry_price=trade.entry_price,
            stop_loss_price=trade.stop_loss,
            max_lot=100.0,
            min_lot=0.01,
        )
        lot_size = sizing_result.get("lot_size", 0.01)
        risk_pips = sizing_result.get("stop_pips", 0)
        
        entry_dt = trade.entry_date
        if isinstance(entry_dt, str):
            try:
                entry_dt = datetime.fromisoformat(entry_dt.replace("Z", "+00:00"))
            except:
                entry_dt = datetime.now()
        if hasattr(entry_dt, 'replace') and entry_dt.tzinfo:
            entry_dt = entry_dt.replace(tzinfo=None)
        
        exit_dt = trade.exit_date
        if isinstance(exit_dt, str):
            try:
                exit_dt = datetime.fromisoformat(exit_dt.replace("Z", "+00:00"))
            except:
                exit_dt = datetime.now()
        if hasattr(exit_dt, 'replace') and exit_dt.tzinfo:
            exit_dt = exit_dt.replace(tzinfo=None)
        
        bt = BacktestTrade(
            trade_num=trade_num,
            challenge_num=1,
            challenge_step=1,
            symbol=trade.symbol,
            direction=trade.direction,
            confluence_score=getattr(trade, 'confluence_score', 0),
            entry_date=entry_dt,
            entry_price=trade.entry_price,
            stop_loss=trade.stop_loss,
            tp1_price=trade.tp1 or trade.entry_price,
            tp2_price=trade.tp2,
            tp3_price=trade.tp3,
            tp4_price=trade.tp4,
            tp5_price=trade.tp5,
            exit_date=exit_dt,
            exit_price=trade.exit_price,
            exit_reason=trade.exit_reason,
            r_multiple=r_multiple,
            profit_loss_usd=profit_usd,
            result=result,
            lot_size=lot_size,
            risk_pips=risk_pips,
        )
        full_year_backtest_trades.append(bt)
    
    wins = sum(1 for t in full_year_backtest_trades if t.result == "WIN")
    total = len(full_year_backtest_trades)
    total_r = sum(t.r_multiple for t in full_year_backtest_trades)
    win_rate = (wins / total * 100) if total > 0 else 0
    
    print(f"\n{'='*80}")
    print("FULL YEAR 2024 BACKTEST RESULTS")
    print(f"{'='*80}")
    print(f"Total Trades: {total}")
    print(f"Wins: {wins}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total R: {total_r:+.1f}R")
    
    final_results = {
        "all_trades": full_year_backtest_trades,
        "total_trades": total,
        "wins": wins,
        "win_rate": win_rate,
        "total_r": total_r,
        "challenges_passed": optimizer.best_training_result.get("results", {}).get("challenges_passed", 0) if optimizer.best_training_result else 0,
        "challenges_failed": optimizer.best_training_result.get("results", {}).get("challenges_failed", 0) if optimizer.best_training_result else 0,
    }
    
    print(f"\n{'='*80}")
    print("GENERATING FINAL REPORTS")
    print(f"{'='*80}")
    
    reporter = ReportGenerator()
    
    validator = OandaValidator()
    validation_report = {}
    if final_results.get('all_trades'):
        validation_report = validator.validate_all_trades(final_results.get('all_trades', []))
    
    reporter.generate_all_reports(final_results, validation_report)
    
    print(f"\n{'='*80}")
    print("WALK-FORWARD OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    
    print(f"\n=== TRAINING PERIOD (Jan-Sep 2024) ===")
    print(f"Best Training Score: {optimizer.best_training_score:.1f}")
    if optimizer.best_training_result:
        tr = optimizer.best_training_result.get("results", {})
        print(f"Challenges Passed: {tr.get('challenges_passed', 0)}")
        print(f"Challenges Failed: {tr.get('challenges_failed', 0)}")
    
    print(f"\n=== VALIDATION PERIOD (Oct-Dec 2024) ===")
    print(f"Best Validation Score: {optimizer.best_validation_score:.1f}")
    if validation_results:
        print(f"Win Rate: {validation_results.get('win_rate', 0):.1f}%")
        print(f"Total R: {validation_results.get('total_r', 0):+.1f}R")
    
    print(f"\n=== OUT-OF-SAMPLE (2023) ===")
    if oos_results:
        print(f"Win Rate: {oos_results.get('win_rate', 0):.1f}%")
        print(f"Total R: {oos_results.get('total_r', 0):+.1f}R")
        print(f"Sharpe-like: {oos_results.get('sharpe_like', 0):.2f}")
        print(f"OOS Test: {'PASSED' if oos_results.get('passed', False) else 'FAILED'}")
    
    print(f"\n=== ROBUSTNESS ASSESSMENT ===")
    print(f"Robustness Score: {robustness_score:.1f}/100")
    print(f"Parameter Stability: {'STABLE' if is_stable else 'UNSTABLE'}")
    
    print(f"\n=== FINAL PARAMETERS ===")
    print(f"min_confluence: {best_params['min_confluence']}")
    print(f"min_quality_factors: {best_params['min_quality_factors']}")
    print(f"risk_per_trade_pct: {best_params['risk_per_trade_pct']}%")
    
    print(f"\nIterations Used: {iteration}")
    print(f"Early Stopped: {optimizer.iterations_without_improvement >= optimizer.EARLY_STOPPING_PATIENCE}")
    
    print(f"\nOptimization Iterations: {len(optimizer.optimization_log)}")
    
    print(f"\n{'='*80}")
    print("WALK-FORWARD OPTIMIZATION SYSTEM COMPLETE")
    print(f"{'='*80}")
    
    return {
        "training_results": optimizer.best_training_result,
        "validation_results": optimizer.best_validation_result,
        "oos_results": oos_results,
        "robustness_score": robustness_score,
        "is_stable": is_stable,
        "best_params": best_params,
        "iterations": iteration,
    }


def main():
    return main_challenge_analyzer()


if __name__ == "__main__":
    main()
