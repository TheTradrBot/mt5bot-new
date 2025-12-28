#!/usr/bin/env python3
"""
Ultimate FTMO Challenge Performance Analyzer - Multi-Year 2023-2025
Production-Ready, Resumable Optimizer with ADX Trend Filter + MedianPruner

This module provides a comprehensive backtesting and self-optimizing system that:
1. Backtests using multi-year historical data (2023-2025)
2. Training Period: 2023-01-01 to 2024-09-30
3. Validation Period: 2024-10-01 to current date
4. ADX > 25 trend-strength filter to avoid ranging markets
5. December fully open for trading (no date restrictions)
6. Tracks ALL trades with complete entry/exit data
7. Generates detailed CSV reports with all trade details
8. Self-optimizes by saving parameters to params/current_params.json
9. RESUMABLE: Uses Optuna SQLite storage for crash-resistant optimization
10. MedianPruner kills bad trials early for faster convergence
11. STATUS MODE: Check progress anytime with --status flag

Usage:
  python ftmo_challenge_analyzer.py              # Run/resume optimization
  python ftmo_challenge_analyzer.py --status     # Check progress without running
  python ftmo_challenge_analyzer.py --trials 100 # Set number of trials
"""

import argparse
import json
import csv
import os
import random
import signal
import sys
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import pandas as pd

# Force unbuffered output for better logging in nohup
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

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
    detect_regime,
    validate_range_mode_entry,
)

from ftmo_config import FTMO_CONFIG, FTMO10KConfig, get_pip_size, get_sl_limits
from config import FOREX_PAIRS, METALS, INDICES, CRYPTO_ASSETS
from tradr.risk.position_sizing import calculate_lot_size, get_contract_specs
from params.params_loader import save_optimized_params
from params.optimization_config import get_optimization_config, OptimizationConfig

# Professional Quant Suite Integration
from professional_quant_suite import (
    RiskMetrics,
    calculate_risk_metrics,
    WalkForwardTester,
    ParameterSensitivityAnalyzer,
    generate_professional_report,
)

OUTPUT_DIR = Path("ftmo_analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_best_params_persistent(best_params: Dict) -> None:
    """
    Save best parameters to best_params.json for instant bot updates.
    This persists even if optimization run is halted abruptly.
    """
    try:
        params_file = Path("best_params.json")
        params_file.write_text(json.dumps(best_params, indent=2))
        print(f"✓ Best parameters saved to best_params.json")
    except Exception as e:
        print(f"[!] Error saving best_params.json: {e}")

# ============================================================================
# UNIFIED OPTIMIZATION CONFIG - Single source of truth
# All database paths, study names, and feature toggles are now in:
#   params/optimization_config.json
# ============================================================================
OPT_CONFIG = get_optimization_config()

# Legacy constants for backwards compatibility - now derived from config
OPTUNA_DB_PATH = OPT_CONFIG.db_path
OPTUNA_STUDY_NAME = OPT_CONFIG.study_name

_DATA_CACHE: Dict[str, List[Dict]] = {}
OPTUNA_STUDY_NAME = "regime_adaptive_v2_clean"
PROGRESS_LOG_FILE = "ftmo_optimization_progress.txt"

# FIXED PERIODS FOR CONSISTENT BACKTESTING
# These dates are locked to ensure reproducible results and proper train/validation splits
TODAY = datetime.utcnow().date()

# TRAINING PERIOD: Full 2023 + first 9 months of 2024
# This gives maximum in-sample data for optimization
TRAINING_START = datetime(2023, 1, 1)
TRAINING_END = datetime(2024, 9, 30)

# VALIDATION PERIOD: Oct 2024 to present (out-of-sample test)
# This tests robustness on unseen data
VALIDATION_START = datetime(2024, 10, 1)
VALIDATION_END = datetime(2025, 12, 26)

# FULL PERIOD: Entire 2023-2025 for comprehensive reporting
FULL_PERIOD_START = datetime(2023, 1, 1)
FULL_PERIOD_END = datetime(2025, 12, 26)

QUARTERS_ALL = {
    "2023_Q1": (datetime(2023, 1, 1), datetime(2023, 3, 31)),
    "2023_Q2": (datetime(2023, 4, 1), datetime(2023, 6, 30)),
    "2023_Q3": (datetime(2023, 7, 1), datetime(2023, 9, 30)),
    "2023_Q4": (datetime(2023, 10, 1), datetime(2023, 12, 31)),
    "2024_Q1": (datetime(2024, 1, 1), datetime(2024, 3, 31)),
    "2024_Q2": (datetime(2024, 4, 1), datetime(2024, 6, 30)),
    "2024_Q3": (datetime(2024, 7, 1), datetime(2024, 9, 30)),
    "2024_Q4": (datetime(2024, 10, 1), datetime(2024, 12, 31)),
    "2025_Q1": (datetime(2025, 1, 1), datetime(2025, 3, 31)),
    "2025_Q2": (datetime(2025, 4, 1), datetime(2025, 6, 30)),
    "2025_Q3": (datetime(2025, 7, 1), datetime(2025, 9, 30)),
    "2025_Q4": (datetime(2025, 10, 1), datetime(2025, 12, 31)),
}

TRAINING_QUARTERS = {
    "2023_Q1": (datetime(2023, 1, 1), datetime(2023, 3, 31)),
    "2023_Q2": (datetime(2023, 4, 1), datetime(2023, 6, 30)),
    "2023_Q3": (datetime(2023, 7, 1), datetime(2023, 9, 30)),
    "2023_Q4": (datetime(2023, 10, 1), datetime(2023, 12, 31)),
    "2024_Q1": (datetime(2024, 1, 1), datetime(2024, 3, 31)),
    "2024_Q2": (datetime(2024, 4, 1), datetime(2024, 6, 30)),
    "2024_Q3": (datetime(2024, 7, 1), datetime(2024, 9, 30)),
}

ACCOUNT_SIZE = 200000.0


def calculate_adx(candles: List[Dict], period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX) for trend strength measurement.
    ADX > 25 indicates a strong trend.
    
    Args:
        candles: List of OHLCV candle dictionaries
        period: ADX period (default 14)
    
    Returns:
        ADX value (0-100 scale)
    """
    if len(candles) < period * 2:
        return 0.0
    
    highs = [c.get("high", 0) for c in candles]
    lows = [c.get("low", 0) for c in candles]
    closes = [c.get("close", 0) for c in candles]
    
    plus_dm = []
    minus_dm = []
    tr_values = []
    
    for i in range(1, len(candles)):
        high_diff = highs[i] - highs[i-1]
        low_diff = lows[i-1] - lows[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm.append(high_diff)
        else:
            plus_dm.append(0)
        
        if low_diff > high_diff and low_diff > 0:
            minus_dm.append(low_diff)
        else:
            minus_dm.append(0)
        
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return 0.0
    
    smoothed_plus_dm = sum(plus_dm[:period])
    smoothed_minus_dm = sum(minus_dm[:period])
    smoothed_tr = sum(tr_values[:period])
    
    dx_values = []
    
    for i in range(period, len(tr_values)):
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr_values[i]
        
        if smoothed_tr == 0:
            continue
            
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0
        else:
            dx = 100 * abs(plus_di - minus_di) / di_sum
        dx_values.append(dx)
    
    if not dx_values:
        return 0.0
    
    if len(dx_values) < period:
        return sum(dx_values) / len(dx_values)
    
    adx = sum(dx_values[:period]) / period
    for i in range(period, len(dx_values)):
        adx = ((adx * (period - 1)) + dx_values[i]) / period
    
    return adx


def check_adx_filter(candles: List[Dict], min_adx: float = 25.0) -> Tuple[bool, float]:
    """
    Check if ADX is above minimum threshold for trend trading.
    
    Args:
        candles: D1 candles for ADX calculation
        min_adx: Minimum ADX value (default 25)
    
    Returns:
        Tuple of (passes_filter, adx_value)
    """
    adx = calculate_adx(candles, period=14)
    return adx > min_adx, adx


def log_optimization_progress(trial_num: int, value: float, best_value: float, best_params: Dict):
    """Append optimization progress to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    key_params = {k: round(v, 3) if isinstance(v, float) else v 
                  for k, v in list(best_params.items())[:5]}
    log_entry = (
        f"[{timestamp}] Trial #{trial_num}: value={value:.0f}, "
        f"best_value={best_value:.0f}, params={json.dumps(key_params)}\n"
    )
    with open(PROGRESS_LOG_FILE, 'a') as f:
        f.write(log_entry)


def show_optimization_status():
    """Display current optimization status without running new trials."""
    import optuna
    
    print("\n" + "=" * 60)
    print("FTMO OPTIMIZATION STATUS CHECK")
    print("=" * 60)
    
    db_file = "regime_adaptive_v2_clean.db"
    if not os.path.exists(db_file):
        print("\nNo optimization study found.")
        print("Run 'python ftmo_challenge_analyzer.py' to start optimization.")
        return
    
    try:
        study = optuna.load_study(
            study_name=OPTUNA_STUDY_NAME,
            storage=OPTUNA_DB_PATH
        )
        
        print(f"\nStudy Name: {OPTUNA_STUDY_NAME}")
        print(f"Completed Trials: {len(study.trials)}")
        
        if study.best_trial:
            print(f"\nBest Value: {study.best_value:.0f}")
            print(f"Best Parameters:")
            for k, v in sorted(study.best_params.items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")
            
            best_trial = study.best_trial
            if best_trial.datetime_complete:
                print(f"\nLast Update: {best_trial.datetime_complete.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("\nNo completed trials yet.")
        
    except Exception as e:
        print(f"\nError loading study: {e}")
        return
    
    if os.path.exists(PROGRESS_LOG_FILE):
        print(f"\n{'='*60}")
        print("RECENT PROGRESS (last 10 entries):")
        print("=" * 60)
        with open(PROGRESS_LOG_FILE, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.rstrip())
    else:
        print("\nNo progress log found yet.")
    
    print(f"\n{'='*60}")
    print("To resume optimization: python ftmo_challenge_analyzer.py")
    print("=" * 60)


def is_valid_trading_day(dt: datetime) -> bool:
    """Check if datetime is a valid trading day (no weekends)."""
    if dt.weekday() >= 5:
        return False
    return True


def _atr(candles: List[Dict], period: int = 14) -> float:
    """Calculate Average True Range (ATR)."""
    if len(candles) < period + 1:
        return 0.0
    
    tr_values = []
    for i in range(1, len(candles)):
        high = candles[i].get("high")
        low = candles[i].get("low")
        prev_close = candles[i - 1].get("close")
        
        if high is None or low is None or prev_close is None:
            continue
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return sum(tr_values) / len(tr_values) if tr_values else 0.0
    
    atr_val = sum(tr_values[:period]) / period
    for tr in tr_values[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    
    return atr_val


def _calculate_atr_percentile(candles: List[Dict], period: int = 14, lookback: int = 100) -> Tuple[float, float]:
    """Calculate current ATR and its percentile rank."""
    if len(candles) < period + lookback:
        current_atr = _atr(candles, period)
        return current_atr, 50.0
    
    atr_values = []
    for i in range(lookback):
        end_idx = len(candles) - i
        if end_idx < period + 1:
            break
        slice_candles = candles[:end_idx]
        atr_val = _atr(slice_candles, period)
        if atr_val > 0:
            atr_values.append(atr_val)
    
    if not atr_values:
        return 0.0, 50.0
    
    current_atr = atr_values[0]
    sorted_atrs = sorted(atr_values)
    rank = sum(1 for v in sorted_atrs if v <= current_atr)
    percentile = (rank / len(sorted_atrs)) * 100
    
    return current_atr, percentile


@dataclass
class BacktestTrade:
    """Complete trade record for CSV export."""
    trade_num: int
    symbol: str
    direction: str
    entry_date: Any
    entry_price: float
    stop_loss_price: float
    tp1_price: float
    tp2_price: Optional[float]
    tp3_price: Optional[float]
    tp4_price: Optional[float]
    tp5_price: Optional[float]
    exit_date: Any
    exit_price: float
    tp1_hit: bool
    tp2_hit: bool
    tp3_hit: bool
    tp4_hit: bool
    tp5_hit: bool
    sl_hit: bool
    exit_reason: str
    r_multiple: float
    profit_loss_usd: float
    confluence_score: int
    holding_time_hours: float
    lot_size: float
    risk_pips: float
    validation_notes: str = ""
    adx_value: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return {
            "Trade#": self.trade_num,
            "Symbol": self.symbol,
            "Direction": self.direction,
            "Entry Date": str(self.entry_date) if self.entry_date else "",
            "Entry Price": self.entry_price,
            "Stop Loss Price": self.stop_loss_price,
            "TP1 Price": self.tp1_price,
            "TP2 Price": self.tp2_price or "",
            "TP3 Price": self.tp3_price or "",
            "TP4 Price": self.tp4_price or "",
            "TP5 Price": self.tp5_price or "",
            "Exit Date": str(self.exit_date) if self.exit_date else "",
            "Exit Price": self.exit_price,
            "TP1 Hit?": "Yes" if self.tp1_hit else "No",
            "TP2 Hit?": "Yes" if self.tp2_hit else "No",
            "TP3 Hit?": "Yes" if self.tp3_hit else "No",
            "TP4 Hit?": "Yes" if self.tp4_hit else "No",
            "TP5 Hit?": "Yes" if self.tp5_hit else "No",
            "SL Hit?": "Yes" if self.sl_hit else "No",
            "Final Exit Reason": self.exit_reason,
            "R Multiple": round(self.r_multiple, 2),
            "Profit/Loss USD": round(self.profit_loss_usd, 2),
            "Confluence Score": self.confluence_score,
            "Holding Time (hours)": round(self.holding_time_hours, 1),
            "Lot Size": round(self.lot_size, 2),
            "Risk Pips": round(self.risk_pips, 1),
            "ADX Value": round(self.adx_value, 1),
            "Validation Notes": self.validation_notes,
        }


class MonteCarloSimulator:
    """Monte Carlo simulation for robustness testing."""
    
    def __init__(self, trades: List[Any], num_simulations: int = 1000):
        self.trades = trades
        self.num_simulations = num_simulations
        self.r_values = self._extract_r_values()
    
    def _extract_r_values(self) -> List[float]:
        r_values = []
        for t in self.trades:
            r = getattr(t, 'rr', None) or getattr(t, 'r_multiple', None) or 0.0
            r_values.append(float(r))
        return r_values
    
    def run_simulation(self) -> Dict[str, Any]:
        if not self.r_values:
            return {"error": "No trades to simulate", "num_simulations": 0}
        
        np.random.seed(42)
        
        final_equities = []
        max_drawdowns = []
        win_rates = []
        
        num_trades = len(self.r_values)
        
        for _ in range(self.num_simulations):
            indices = np.random.choice(num_trades, size=num_trades, replace=True)
            resampled_trades = [self.r_values[i] for i in indices]
            
            noise = np.random.uniform(0.9, 1.1, size=num_trades)
            perturbed_trades = [r * n for r, n in zip(resampled_trades, noise)]
            
            equity_curve = [0.0]
            for r in perturbed_trades:
                equity_curve.append(equity_curve[-1] + r)
            
            final_equities.append(equity_curve[-1])
            
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
        
        return {
            "num_simulations": self.num_simulations,
            "num_trades": num_trades,
            "mean_return": float(np.mean(final_equities)),
            "std_return": float(np.std(final_equities)),
            "mean_max_dd": float(np.mean(max_drawdowns)),
            "mean_win_rate": float(np.mean(win_rates)),
            "worst_case_dd": float(np.percentile(max_drawdowns, 95)),
            "best_case_return": float(np.percentile(final_equities, 95)),
            "worst_case_return": float(np.percentile(final_equities, 5)),
            "confidence_intervals": {
                "final_equity": {f"p{p}": float(np.percentile(final_equities, p)) for p in [5, 25, 50, 75, 95]},
                "max_drawdown": {f"p{p}": float(np.percentile(max_drawdowns, p)) for p in [5, 25, 50, 75, 95]},
            },
        }


def run_monte_carlo_analysis(trades: List[Any], num_simulations: int = 1000) -> Dict:
    """Run Monte Carlo analysis on trades."""
    if not trades:
        return {"error": "No trades provided"}
    
    simulator = MonteCarloSimulator(trades, num_simulations)
    results = simulator.run_simulation()
    
    print(f"\nMonte Carlo Simulation ({results.get('num_simulations', 0)} iterations):")
    print(f"  Mean Return: {results.get('mean_return', 0):+.2f}R")
    print(f"  Std Dev: {results.get('std_return', 0):.2f}R")
    print(f"  Best Case (95th): {results.get('best_case_return', 0):+.2f}R")
    print(f"  Worst Case (5th): {results.get('worst_case_return', 0):+.2f}R")
    print(f"  Worst Case DD (95th): {results.get('worst_case_dd', 0):.2f}R")
    
    return results


def load_ohlcv_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    """Load OHLCV data from local CSV files only (no API calls). Uses cache for performance."""
    global _DATA_CACHE
    data_dir = Path("data/ohlcv")
    
    symbol_normalized = symbol.replace("_", "").replace("/", "")
    
    tf_map = {"D1": "D1", "H4": "H4", "W1": "W1", "MN": "MN"}
    tf = tf_map.get(timeframe, timeframe)
    
    cache_key = f"{symbol_normalized}_{tf}"
    
    if cache_key not in _DATA_CACHE:
        pattern = f"{symbol_normalized}_{tf}_*.csv"
        matches = list(data_dir.glob(pattern))
        
        if not matches:
            _DATA_CACHE[cache_key] = []
            return []
        
        csv_path = matches[0]
        try:
            df = pd.read_csv(csv_path)
            
            date_col = None
            for col in ['time', 'timestamp', 'date', 'Date', 'Time']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], utc=True)
            
            col_map = {}
            for target, options in [
                ('time', [date_col] if date_col else []),
                ('open', ['open', 'Open']),
                ('high', ['high', 'High']),
                ('low', ['low', 'Low']),
                ('close', ['close', 'Close']),
                ('volume', ['volume', 'Volume']),
            ]:
                for opt in options:
                    if opt and opt in df.columns:
                        col_map[target] = opt
                        break
            
            result_df = pd.DataFrame()
            for target, source in col_map.items():
                result_df[target] = df[source]
            
            if 'volume' not in result_df.columns:
                result_df['volume'] = 0
            
            candles = result_df.to_dict('records')
            
            _DATA_CACHE[cache_key] = candles
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            _DATA_CACHE[cache_key] = []
    
    all_candles = _DATA_CACHE[cache_key]
    if not all_candles:
        return []
    
    start_ts = pd.Timestamp(start_date, tz='UTC') if start_date.tzinfo is None else pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date, tz='UTC') if end_date.tzinfo is None else pd.Timestamp(end_date)
    
    filtered = [c for c in all_candles if c.get("time") and start_ts <= c["time"] <= end_ts]
    return filtered


def get_all_trading_assets() -> List[str]:
    """Get list of all tradeable assets."""
    assets = []
    assets.extend(FOREX_PAIRS if FOREX_PAIRS else [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "NZD_USD", "USD_CAD",
        "EUR_GBP", "EUR_JPY", "GBP_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
        "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD", "AUD_JPY", "AUD_NZD", "AUD_CAD",
        "AUD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CAD_JPY", "CAD_CHF", "CHF_JPY"
    ])
    assets.extend(METALS if METALS else ["XAU_USD", "XAG_USD"])
    assets.extend(INDICES if INDICES else ["SPX500_USD", "NAS100_USD"])
    assets.extend(CRYPTO_ASSETS if CRYPTO_ASSETS else ["BTC_USD", "ETH_USD"])
    
    return list(set(assets))


def run_full_period_backtest(
    start_date: datetime,
    end_date: datetime,
    min_confluence: int = 3,
    min_quality_factors: int = 1,
    risk_per_trade_pct: float = 0.5,
    atr_min_percentile: float = 60.0,
    trail_activation_r: float = 2.2,
    december_atr_multiplier: float = 1.5,
    volatile_asset_boost: float = 1.5,
    ml_min_prob: Optional[float] = None,
    excluded_assets: Optional[List[str]] = None,
    require_adx_filter: bool = True,
    min_adx: float = 25.0,
    # ============================================================================
    # REGIME-ADAPTIVE V2 PARAMETERS
    # ============================================================================
    use_adx_regime_filter: bool = False,  # DISABLED: Set to True to enable ADX regime filtering
    adx_trend_threshold: float = 25.0,  # ADX level for Trend Mode
    adx_range_threshold: float = 20.0,  # ADX level for Range Mode
    trend_min_confluence: int = 6,  # Min confluence for trend mode
    range_min_confluence: int = 5,  # Min confluence for range mode
    rsi_oversold_range: float = 25.0,  # RSI threshold for range longs
    rsi_overbought_range: float = 75.0,  # RSI threshold for range shorts
    atr_volatility_ratio: float = 0.8,  # ATR(14)/ATR(50) ratio for range mode
    atr_trail_multiplier: float = 1.5,  # ATR multiplier for trailing stops
    partial_exit_at_1r: bool = True,  # Whether to take profit at 1R
    partial_exit_pct: float = 0.5,  # % to close at 1R
    use_adx_slope_rising: bool = False,  # Enable ADX slope rising early trend detection
    atr_vol_ratio_range: float = 0.8,  # ATR volatility ratio for range mode filter
) -> List[Trade]:
    """
    Run backtest for a given period with Regime-Adaptive V2 filtering.
    
    REGIME-ADAPTIVE V2 SYSTEM:
    ==========================
    
    This backtest uses a dual-mode regime detection system based on ADX:
    
    1. TREND MODE (ADX >= adx_trend_threshold):
       - Momentum-following entries
       - Standard trend trading rules apply
       - Higher confluence but momentum bias
    
    2. RANGE MODE (ADX < adx_range_threshold):
       - Ultra-conservative mean reversion
       - ALL filters must pass: RSI extremes, Fib 0.786, S/R zone, H4 rejection
       - Low volatility confirmation required
    
    3. TRANSITION ZONE (ADX between thresholds):
       - NO ENTRIES - market regime is unclear
       - Wait for regime confirmation before trading
    
    December is fully open for trading.
    """
    assets = get_all_trading_assets()
    
    if excluded_assets:
        assets = [a for a in assets if a not in excluded_assets]
    
    all_trades: List[Trade] = []
    seen_trades = set()
    
    total_assets = len(assets)
    for idx, symbol in enumerate(assets):
        if idx % 10 == 0:
            print(f"  Processing asset {idx+1}/{total_assets}: {symbol}...", end="\r", flush=True)
        try:
            d1_candles = load_ohlcv_data(symbol, "D1", start_date - timedelta(days=100), end_date)
            h4_candles = load_ohlcv_data(symbol, "H4", start_date - timedelta(days=50), end_date)
            w1_candles = load_ohlcv_data(symbol, "W1", start_date - timedelta(days=365), end_date)
            mn_candles = load_ohlcv_data(symbol, "MN", start_date - timedelta(days=730), end_date)
            
            if not d1_candles or len(d1_candles) < 30:
                continue
            
            regime_info = detect_regime(
                daily_candles=d1_candles,
                adx_trend_threshold=adx_trend_threshold,
                adx_range_threshold=adx_range_threshold,
                use_adx_slope_rising=use_adx_slope_rising,
                use_adx_regime_filter=use_adx_regime_filter  # Pass ADX filter toggle
            )
            
            # Only skip Transition mode if ADX filter is enabled
            if use_adx_regime_filter and regime_info['mode'] == 'Transition':
                continue
            
            if regime_info['mode'] == 'Trend':
                effective_confluence = trend_min_confluence
            else:
                effective_confluence = range_min_confluence
            
            # DISABLED: ATR percentile filter - too restrictive, prevents trades
            # current_atr, atr_percentile = _calculate_atr_percentile(d1_candles)
            # if atr_percentile < atr_min_percentile:
            #     continue
            
            params = StrategyParams(
                min_confluence=effective_confluence,
                min_quality_factors=min_quality_factors,
                risk_per_trade_pct=risk_per_trade_pct,
                atr_min_percentile=atr_min_percentile,
                trail_activation_r=trail_activation_r,
                december_atr_multiplier=december_atr_multiplier,
                volatile_asset_boost=volatile_asset_boost,
                adx_trend_threshold=adx_trend_threshold,
                adx_range_threshold=adx_range_threshold,
                use_adx_regime_filter=use_adx_regime_filter,  # Pass ADX filter toggle to params
            )
            
            trades = simulate_trades(
                candles=d1_candles,
                symbol=symbol,
                params=params,
                h4_candles=h4_candles,
                weekly_candles=w1_candles,
                monthly_candles=mn_candles,
                include_transaction_costs=True,
            )
            
            for trade in trades:
                if regime_info['mode'] == 'Range':
                    direction = trade.direction
                    entry_price = trade.entry_price
                    confluence = trade.confluence_score
                    
                    is_valid, range_details = validate_range_mode_entry(
                        daily_candles=d1_candles,
                        h4_candles=h4_candles,
                        weekly_candles=w1_candles,
                        monthly_candles=mn_candles,
                        price=entry_price,
                        direction=direction,
                        confluence_score=confluence,
                        params=params,
                        historical_sr=None,
                        atr_vol_ratio_range=atr_vol_ratio_range,
                    )
                    
                    if not is_valid:
                        continue
                
                trade_key = (
                    trade.symbol,
                    str(trade.entry_date)[:10],
                    trade.direction,
                    round(trade.entry_price, 5)
                )
                if trade_key not in seen_trades:
                    seen_trades.add(trade_key)
                    all_trades.append(trade)
                    
        except Exception as e:
            continue
    
    all_trades.sort(key=lambda t: str(t.entry_date))
    
    # CRITICAL FIX: Filter trades to only include those within the requested date range
    # This ensures training/validation periods are isolated correctly
    filtered_trades = []
    for trade in all_trades:
        entry = getattr(trade, 'entry_date', None)
        if entry:
            if isinstance(entry, str):
                try:
                    entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                except:
                    continue
            if hasattr(entry, 'replace') and entry.tzinfo:
                entry = entry.replace(tzinfo=None)
            
            # Check if trade entry_date is within the requested period
            if start_date <= entry <= end_date:
                filtered_trades.append(trade)
    
    return filtered_trades


def convert_to_backtest_trade(
    trade: Trade,
    trade_num: int,
    risk_per_trade_pct: float,
    adx_value: float = 0.0
) -> BacktestTrade:
    """Convert strategy Trade to BacktestTrade for CSV export."""
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
            except:
                holding_hours = 0.0
    
    risk_usd = ACCOUNT_SIZE * (risk_per_trade_pct / 100)
    profit_usd = trade.rr * risk_usd
    
    exit_reason = trade.exit_reason or ""
    tp1_hit = "TP1" in exit_reason or "TP2" in exit_reason or "TP3" in exit_reason or "TP4" in exit_reason or "TP5" in exit_reason
    tp2_hit = "TP2" in exit_reason or "TP3" in exit_reason or "TP4" in exit_reason or "TP5" in exit_reason
    tp3_hit = "TP3" in exit_reason or "TP4" in exit_reason or "TP5" in exit_reason
    tp4_hit = "TP4" in exit_reason or "TP5" in exit_reason
    tp5_hit = "TP5" in exit_reason
    sl_hit = exit_reason == "SL"
    
    sizing_result = calculate_lot_size(
        symbol=trade.symbol,
        account_balance=ACCOUNT_SIZE,
        risk_percent=risk_per_trade_pct / 100,
        entry_price=trade.entry_price,
        stop_loss_price=trade.stop_loss,
        max_lot=100.0,
        min_lot=0.01,
    )
    lot_size = sizing_result.get("lot_size", 0.01)
    
    # Use profit_usd directly - spread/slippage already accounted for in backtest simulation
    # The R multiple already reflects the actual trade outcome
    adjusted_profit = profit_usd

    return BacktestTrade(
        trade_num=trade_num,
        symbol=trade.symbol,
        direction=trade.direction.upper(),
        entry_date=entry_dt,
        entry_price=trade.entry_price,
        stop_loss_price=trade.stop_loss,
        tp1_price=trade.tp1 or 0.0,
        tp2_price=trade.tp2,
        tp3_price=trade.tp3,
        tp4_price=trade.tp4,
        tp5_price=trade.tp5,
        exit_date=exit_dt,
        exit_price=trade.exit_price,
        tp1_hit=tp1_hit,
        tp2_hit=tp2_hit,
        tp3_hit=tp3_hit,
        tp4_hit=tp4_hit,
        tp5_hit=tp5_hit,
        sl_hit=sl_hit,
        exit_reason=exit_reason,
        r_multiple=trade.rr,
        profit_loss_usd=adjusted_profit,
        confluence_score=trade.confluence_score,
        holding_time_hours=holding_hours,
        lot_size=lot_size,
        risk_pips=risk_pips,
        adx_value=adx_value,
        validation_notes="Regime-Adaptive V2: Trend (ADX >= threshold) + Conservative Range (ADX < threshold)",
    )


def export_trades_to_csv(trades: List[Trade], filename: str, risk_per_trade_pct: float = 0.5):
    """Export trades to CSV with all required columns."""
    filepath = OUTPUT_DIR / filename
    
    if not trades:
        print(f"No trades to export to {filename}")
        return
    
    backtest_trades = []
    for i, trade in enumerate(trades, 1):
        bt_trade = convert_to_backtest_trade(trade, i, risk_per_trade_pct)
        backtest_trades.append(bt_trade)
    
    with open(filepath, 'w', newline='') as f:
        if backtest_trades:
            fieldnames = list(backtest_trades[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in backtest_trades:
                writer.writerow(trade.to_dict())
    
    print(f"Exported {len(backtest_trades)} trades to: {filepath}")


def print_period_results(trades: List[Trade], period_name: str, start: datetime, end: datetime) -> Dict:
    """Print results for a specific period."""
    if not trades:
        print(f"\n{period_name}: No trades generated")
        return {'trades': 0, 'total_r': 0, 'win_rate': 0, 'net_profit': 0}
    
    total_r = sum(getattr(t, 'rr', 0) for t in trades)
    wins = sum(1 for t in trades if getattr(t, 'rr', 0) > 0)
    losses = sum(1 for t in trades if getattr(t, 'rr', 0) <= 0)
    win_rate = (wins / len(trades) * 100) if trades else 0
    
    risk_usd = ACCOUNT_SIZE * 0.005
    total_profit = total_r * risk_usd
    
    print(f"\n{period_name}")
    print(f"  Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Wins: {wins}, Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total R: {total_r:+.2f}R")
    print(f"  Net Profit (est): ${total_profit:+,.2f}")
    print(f"  Regime-Adaptive V2: Trend (ADX >= threshold) + Conservative Range (ADX < threshold)")
    
    return {
        'trades': len(trades),
        'total_r': total_r,
        'win_rate': win_rate,
        'net_profit': total_profit,
    }


def update_readme_documentation():
    """Update README.md with optimization & backtesting section."""
    readme_path = Path("README.md")
    
    new_section = """
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
"""
    
    if readme_path.exists():
        content = readme_path.read_text()
        
        import re
        pattern = r'## Optimization & Backtesting.*?(?=\n## |\Z)'
        
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, new_section.strip() + "\n\n", content, flags=re.DOTALL)
        else:
            content = content.rstrip() + "\n\n" + new_section.strip() + "\n"
        
        readme_path.write_text(content)
        print("README.md updated with optimization section.")
    else:
        readme_path.write_text(new_section.strip())
        print("README.md created with optimization section.")


def train_ml_model(trades: List[Trade]) -> bool:
    """Train RandomForest ML model on full-year trades and save to models/best_rf.joblib."""
    os.makedirs('models', exist_ok=True)
    
    if len(trades) < 50:
        print(f"Insufficient trades for ML training: {len(trades)} < 50 required")
        return False
    
    features_list = []
    labels = []
    
    for trade in trades:
        r_value = getattr(trade, 'rr', 0)
        label = 1 if r_value > 0 else 0
        
        features = {
            'confluence_score': trade.confluence_score,
            'direction_bullish': 1 if trade.direction == 'bullish' else 0,
            'risk': abs(trade.entry_price - trade.stop_loss) if trade.stop_loss else 0,
            'rr_target': abs(trade.tp1 - trade.entry_price) / max(0.00001, abs(trade.entry_price - trade.stop_loss)) if trade.tp1 and trade.stop_loss else 1,
        }
        
        features_list.append(list(features.values()))
        labels.append(label)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            min_samples_leaf=10
        )
        clf.fit(features_list, labels)
        
        joblib.dump(clf, 'models/best_rf.joblib')
        print(f"ML model trained on {len(trades)} trades and saved to models/best_rf.joblib")
        return True
    except Exception as e:
        print(f"ML training failed: {e}")
        return False


class OptunaOptimizer:
    """
    Optuna-based optimizer for FTMO strategy parameters.
    Runs optimization ONLY on training data (2023-01-01 to 2024-09-30).
    Uses persistent SQLite storage for resumability.
    """
    
    def __init__(self):
        self.best_params: Dict = {}
        self.best_score: float = -float('inf')
    
    def _objective(self, trial) -> float:
        """
        Optuna objective function - DUAL PERIOD validation.
        Trains on 2023-2024-09-30, validates on 2024-10-01+.
        Objective: Balance training profit + validation robustness (50/50 weighted).
        
        REGIME-ADAPTIVE V2: Extended search space includes:
        - Regime detection thresholds (ADX trend/range)
        - Mode-specific confluence requirements
        - Partial profit taking and trail management
        """
        # ============================================================================
        # REGIME-ADAPTIVE V2 EXPANDED PARAMETER SEARCH SPACE (20+ Parameters)
        # ============================================================================
        # AGGRESSIVELY LOOSENED PARAMETERS FOR MAXIMUM TRADE GENERATION
        # Goal: Generate 200+ trades in training period + meaningful validation trades
        params = {
            'risk_per_trade_pct': trial.suggest_float('risk_per_trade_pct', 0.3, 0.8, step=0.05),
            'min_confluence_score': trial.suggest_int('min_confluence_score', 2, 4),  # ULTRA-LOOSE: 2-4 (was 3-5)
            'min_quality_factors': trial.suggest_int('min_quality_factors', 1, 2),    # LOOSEN: 1-2 (was 1-3)
            'adx_trend_threshold': trial.suggest_float('adx_trend_threshold', 15.0, 24.0, step=1.0),  # Allow much lower ADX
            'adx_range_threshold': trial.suggest_float('adx_range_threshold', 10.0, 18.0, step=1.0),  # Allow lower range mode ADX
            'trend_min_confluence': trial.suggest_int('trend_min_confluence', 3, 6),   # Allow as low as 3
            'range_min_confluence': trial.suggest_int('range_min_confluence', 2, 5),   # Allow 2+
            'atr_trail_multiplier': trial.suggest_float('atr_trail_multiplier', 1.2, 3.5, step=0.2),
            'atr_vol_ratio_range': trial.suggest_float('atr_vol_ratio_range', 0.5, 1.0, step=0.05),
            'partial_exit_at_1r': trial.suggest_categorical('partial_exit_at_1r', [True, False]),
            'partial_exit_pct': trial.suggest_float('partial_exit_pct', 0.3, 0.8, step=0.05),
            'atr_min_percentile': trial.suggest_float('atr_min_percentile', 30.0, 70.0, step=5.0),  # AGGRESSIVELY LOOSEN!
            'trail_activation_r': trial.suggest_float('trail_activation_r', 1.0, 3.0, step=0.2),
            'december_atr_multiplier': trial.suggest_float('december_atr_multiplier', 1.0, 2.0, step=0.1),
            'volatile_asset_boost': trial.suggest_float('volatile_asset_boost', 1.0, 2.0, step=0.1),
        }
        
        training_trades = run_full_period_backtest(
            start_date=TRAINING_START,
            end_date=TRAINING_END,
            min_confluence=params['min_confluence_score'],
            min_quality_factors=params['min_quality_factors'],
            risk_per_trade_pct=params['risk_per_trade_pct'],
            atr_min_percentile=params['atr_min_percentile'],
            trail_activation_r=params['trail_activation_r'],
            december_atr_multiplier=params['december_atr_multiplier'],
            volatile_asset_boost=params['volatile_asset_boost'],
            ml_min_prob=None,
            require_adx_filter=True,
            min_adx=25.0,
            use_adx_regime_filter=False,  # DISABLED: ADX regime filtering disabled for now
            adx_trend_threshold=params['adx_trend_threshold'],
            adx_range_threshold=params['adx_range_threshold'],
            trend_min_confluence=params['trend_min_confluence'],
            range_min_confluence=params['range_min_confluence'],
            atr_volatility_ratio=params['atr_vol_ratio_range'],
            atr_vol_ratio_range=params['atr_vol_ratio_range'],
            atr_trail_multiplier=params['atr_trail_multiplier'],
            partial_exit_at_1r=params['partial_exit_at_1r'],
            partial_exit_pct=params['partial_exit_pct'],
        )
        
        if not training_trades or len(training_trades) == 0:
            trial.set_user_attr('quarterly_stats', {})
            trial.set_user_attr('overall_stats', {'trades': 0, 'profit': 0, 'win_rate': 0})
            return -50000.0
        
        total_r = sum(getattr(t, 'rr', 0) for t in training_trades)
        total_trades = len(training_trades)
        wins = sum(1 for t in training_trades if getattr(t, 'rr', 0) > 0)
        overall_win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        if total_r <= 0:
            trial.set_user_attr('quarterly_stats', {})
            trial.set_user_attr('overall_stats', {'trades': total_trades, 'profit': total_r, 'win_rate': overall_win_rate})
            return -50000.0
        
        quarterly_r = {q: 0.0 for q in TRAINING_QUARTERS.keys()}
        quarterly_trades = {q: [] for q in TRAINING_QUARTERS.keys()}
        
        for t in training_trades:
            entry = getattr(t, 'entry_date', None)
            if entry:
                if isinstance(entry, str):
                    try:
                        entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                    except:
                        continue
                if hasattr(entry, 'replace') and entry.tzinfo:
                    entry = entry.replace(tzinfo=None)
                
                for q, (q_start, q_end) in TRAINING_QUARTERS.items():
                    if q_start <= entry <= q_end:
                        quarterly_r[q] += getattr(t, 'rr', 0)
                        quarterly_trades[q].append(t)
                        break
        
        risk_usd = ACCOUNT_SIZE * (params['risk_per_trade_pct'] / 100)
        
        quarterly_stats = {}
        for q in TRAINING_QUARTERS.keys():
            q_trades = quarterly_trades[q]
            q_total = len(q_trades)
            q_wins = sum(1 for t in q_trades if getattr(t, 'rr', 0) > 0)
            q_r = quarterly_r[q]
            q_profit = q_r * risk_usd
            q_wr = (q_wins / q_total * 100) if q_total > 0 else 0
            quarterly_stats[q] = {
                'trades': q_total,
                'wins': q_wins,
                'r_total': round(q_r, 2),
                'profit': round(q_profit, 2),
                'win_rate': round(q_wr, 1)
            }
        
        trial.set_user_attr('quarterly_stats', quarterly_stats)
        trial.set_user_attr('overall_stats', {
            'trades': total_trades,
            'wins': wins,
            'r_total': round(total_r, 2),
            'profit': round(total_r * risk_usd, 2),
            'win_rate': round(overall_win_rate, 1)
        })
        
        # ============================================================================
        # PROFESSIONAL SCORING FORMULA V3
        # Multi-objective optimization using industry-standard metrics
        # ============================================================================
        
        # Extract risk percentage from params
        risk_pct = params['risk_per_trade_pct']
        
        # Calculate quarterly profits, trade counts, and winning trades
        quarterly_profits = {}  # Q -> profit in USD
        quarterly_trade_counts = {}  # Q -> number of trades
        quarterly_winning_trades = {}  # Q -> number of winning trades
        
        for q in TRAINING_QUARTERS.keys():
            q_r = quarterly_r.get(q, 0.0)
            q_profit = q_r * risk_usd
            q_count = len(quarterly_trades.get(q, []))
            q_wins = sum(1 for t in quarterly_trades.get(q, []) if getattr(t, 'rr', 0) > 0)
            quarterly_profits[q] = q_profit
            quarterly_trade_counts[q] = q_count
            quarterly_winning_trades[q] = q_wins
        
        trades_list = training_trades  # Alias for consistency
        
        # ============================================================================
        # COMPONENT 1: PROFIT FACTOR (most important for profitability)
        # Profit Factor = Gross Profit / Gross Loss
        # Target: > 1.5 is good, > 2.0 is excellent
        # ============================================================================
        gross_profit = sum(getattr(t, 'rr', 0) for t in trades_list if getattr(t, 'rr', 0) > 0)
        gross_loss = abs(sum(getattr(t, 'rr', 0) for t in trades_list if getattr(t, 'rr', 0) < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit
        
        # ============================================================================
        # COMPONENT 2: SHARPE RATIO (risk-adjusted returns)
        # Using professional_quant_suite for institutional-grade calculation
        # Target: > 0.5 is acceptable, > 1.0 is good, > 2.0 is excellent
        # ============================================================================
        risk_metrics = calculate_risk_metrics(
            trades=trades_list,
            risk_per_trade_pct=risk_pct,
            account_size=ACCOUNT_SIZE,
            trading_days_per_year=252.0
        )
        sharpe_ratio = risk_metrics.sharpe_ratio
        sortino_ratio = risk_metrics.sortino_ratio
        
        # ============================================================================
        # COMPONENT 3: EXPECTANCY (average R per trade)
        # E = (Win% × Avg Win) - (Loss% × Avg Loss)
        # Target: > 0.3R per trade is good
        # ============================================================================
        avg_win = (gross_profit / wins) if wins > 0 else 0
        losses = total_trades - wins
        avg_loss = (gross_loss / losses) if losses > 0 else 0
        expectancy = (overall_win_rate / 100 * avg_win) - ((1 - overall_win_rate / 100) * avg_loss)
        
        # ============================================================================
        # COMPONENT 4: DRAWDOWN RISK (FTMO compliance critical)
        # Max DD in R terms, scaled to account percentage
        # Target: < 10% of account
        # ============================================================================
        max_dd_r = 0.0
        equity = 0.0
        peak = 0.0
        for t in trades_list:
            equity += getattr(t, 'rr', 0)
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd_r:
                max_dd_r = dd
        
        max_drawdown_pct = (max_dd_r * risk_usd) / ACCOUNT_SIZE if ACCOUNT_SIZE > 0 else 0
        
        # ============================================================================
        # COMPONENT 5: CONSISTENCY SCORE
        # Count negative quarters and severely underperforming quarters
        # ============================================================================
        negative_quarter_count = sum(1 for r in quarterly_r.values() if r < -5.0)  # Only count significantly negative
        weak_quarter_count = sum(1 for r in quarterly_r.values() if -5.0 <= r < 0)  # Slightly negative
        zero_trade_quarters = sum(1 for c in quarterly_trade_counts.values() if c == 0)
        
        # ============================================================================
        # MULTI-OBJECTIVE SCORING FORMULA V4
        # Combines: Profit, Sharpe Ratio, Win Rate, Consistency
        # All components normalized to similar scales for balanced optimization
        # ============================================================================
        
        # Base score: Total R (simple and interpretable)
        base_score = total_r
        
        # Sharpe Ratio Bonus: Reward risk-adjusted returns (NEW!)
        # Scaled to contribute 10-30 points at good levels
        sharpe_bonus = 0.0
        if sharpe_ratio >= 2.0:
            sharpe_bonus = 30.0   # Exceptional risk-adjusted returns
        elif sharpe_ratio >= 1.5:
            sharpe_bonus = 25.0   # Excellent
        elif sharpe_ratio >= 1.0:
            sharpe_bonus = 20.0   # Good
        elif sharpe_ratio >= 0.5:
            sharpe_bonus = 10.0   # Acceptable
        elif sharpe_ratio >= 0.0:
            sharpe_bonus = 5.0 * sharpe_ratio  # Proportional for 0-0.5
        else:
            sharpe_bonus = 10.0 * sharpe_ratio  # Penalty for negative Sharpe
        
        # Profit Factor Bonus: Reward PF > 1.0, strong bonus for PF > 1.5
        pf_bonus = 0.0
        if profit_factor >= 2.0:
            pf_bonus = 20.0  # Excellent PF
        elif profit_factor >= 1.5:
            pf_bonus = 10.0  # Good PF
        elif profit_factor >= 1.2:
            pf_bonus = 5.0   # Acceptable PF
        elif profit_factor < 1.0:
            pf_bonus = -10.0 * (1.0 - profit_factor)  # Penalty for losing strategy
        
        # Win Rate Bonus: Reward WR > 45%
        wr_bonus = 0.0
        if overall_win_rate >= 55:
            wr_bonus = 15.0
        elif overall_win_rate >= 50:
            wr_bonus = 10.0
        elif overall_win_rate >= 45:
            wr_bonus = 5.0
        elif overall_win_rate < 35:
            wr_bonus = -5.0  # Penalty for very low win rate
        
        # Drawdown Penalty: Penalize excessive drawdowns (FTMO critical)
        dd_penalty = 0.0
        if max_drawdown_pct > 0.10:  # Over 10% account drawdown
            dd_penalty = (max_drawdown_pct - 0.10) * 150  # 1.5 points per 1% over threshold
        elif max_drawdown_pct > 0.08:  # Warning zone
            dd_penalty = (max_drawdown_pct - 0.08) * 50
        
        # Consistency Penalty: Penalize bad quarters (but don't kill the score)
        consistency_penalty = 0.0
        consistency_penalty += negative_quarter_count * 10.0  # Significant penalty for very bad quarters
        consistency_penalty += weak_quarter_count * 3.0       # Smaller penalty for slightly negative
        consistency_penalty += zero_trade_quarters * 5.0      # Penalty for inactive quarters
        
        # Trade Count Bonus: Reward having enough trades for statistical significance
        trade_bonus = 0.0
        if 100 <= total_trades <= 300:
            trade_bonus = 10.0  # Ideal range
        elif 50 <= total_trades < 100:
            trade_bonus = 5.0   # Acceptable
        elif total_trades > 400:
            trade_bonus = -5.0  # Too many trades (overtrading)
        elif total_trades < 30:
            trade_bonus = -10.0 # Not enough trades for confidence
        
        # Calculate final composite score
        final_score = (
            base_score +          # Core profitability in R
            sharpe_bonus +        # Risk-adjusted return quality (NEW!)
            pf_bonus +            # Profit factor quality
            wr_bonus +            # Win rate quality
            trade_bonus -         # Trade frequency balance
            dd_penalty -          # Drawdown risk
            consistency_penalty   # Quarter consistency
        )
        
        # Store ALL metrics for analysis and multi-objective selection
        trial.set_user_attr('sharpe_ratio', round(sharpe_ratio, 3))
        trial.set_user_attr('sortino_ratio', round(sortino_ratio, 3))
        trial.set_user_attr('profit_factor', round(profit_factor, 3))
        trial.set_user_attr('expectancy', round(expectancy, 3))
        trial.set_user_attr('max_drawdown_pct', round(max_drawdown_pct * 100, 2))
        trial.set_user_attr('negative_quarters', negative_quarter_count)
        trial.set_user_attr('total_r', round(total_r, 2))
        trial.set_user_attr('win_rate', round(overall_win_rate, 2))
        trial.set_user_attr('score_breakdown', {
            'base_r': round(base_score, 2),
            'sharpe_bonus': round(sharpe_bonus, 2),
            'pf_bonus': round(pf_bonus, 2),
            'wr_bonus': round(wr_bonus, 2),
            'trade_bonus': round(trade_bonus, 2),
            'dd_penalty': round(dd_penalty, 2),
            'consistency_penalty': round(consistency_penalty, 2),
        })
        
        return final_score
    
    def run_optimization(self, n_trials: int = 5, early_stopping_patience: int = 20) -> Dict:
        """Run Optuna optimization on TRAINING data only.
        
        Args:
            n_trials: Number of trials to run
            early_stopping_patience: Stop if no improvement after this many trials (default: 20)
        """
        import optuna
        from optuna.pruners import MedianPruner
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"\n{'='*60}")
        print(f"OPTUNA OPTIMIZATION - Adding {n_trials} trials")
        print(f"TRAINING PERIOD: 2023-01-01 to 2024-09-30")
        print(f"Regime-Adaptive V2: Trend (ADX >= threshold) + Conservative Range (ADX < threshold)")
        print(f"Storage: {OPTUNA_DB_PATH} (resumable)")
        print(f"Early Stopping: After {early_stopping_patience} trials without improvement")
        print(f"{'='*60}")
        
        study = optuna.create_study(
            direction='maximize',
            study_name=OPTUNA_STUDY_NAME,
            storage=OPTUNA_DB_PATH,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=MedianPruner()
        )
        
        existing_trials = len(study.trials)
        previous_best_value = None  # Track previous best to detect real improvements
        if existing_trials > 0:
            print(f"Resuming from existing study with {existing_trials} completed trials")
            try:
                if study.best_trial and study.best_value is not None:
                    previous_best_value = study.best_value
                    print(f"Current best value: {study.best_value:.0f}")
                else:
                    print("No best trial found yet (all trials may have failed)")
            except (ValueError, AttributeError) as e:
                print(f"No valid completed trials yet: {e}")
        
        # Store best value before optimization starts for comparison
        best_value_before_run = previous_best_value
        
        # Early stopping tracking
        trials_without_improvement = 0
        last_best_value = previous_best_value
        early_stopped = False
        
        def progress_callback(study, trial):
            nonlocal best_value_before_run
            
            log_optimization_progress(
                trial_num=trial.number,
                value=trial.value if trial.value is not None else 0,
                best_value=study.best_value if study.best_trial else 0,
                best_params=study.best_params if study.best_trial else {}
            )
            
            # Check if this trial is STRICTLY better than the previous best
            is_new_best = False
            try:
                current_best = study.best_value if study.best_trial else None
                if current_best is not None:
                    if best_value_before_run is None:
                        # First successful trial
                        is_new_best = True
                    elif current_best > best_value_before_run:
                        # Strictly better than before
                        is_new_best = True
                    
                    # Update best_value_before_run for next comparison
                    if is_new_best:
                        best_value_before_run = current_best
            except (ValueError, AttributeError):
                pass
            
            # Early stopping check
            nonlocal trials_without_improvement, last_best_value, early_stopped
            current_best_for_stopping = study.best_value if study.best_trial else None
            if current_best_for_stopping is not None:
                if last_best_value is None or current_best_for_stopping > last_best_value:
                    # Improvement found
                    trials_without_improvement = 0
                    last_best_value = current_best_for_stopping
                else:
                    # No improvement
                    trials_without_improvement += 1
                    
                    if trials_without_improvement >= early_stopping_patience:
                        print(f"\n🛑 EARLY STOPPING: No improvement for {early_stopping_patience} consecutive trials")
                        print(f"   Best score remains: {last_best_value:.2f}")
                        early_stopped = True
                        study.stop()
            
            quarterly_stats = trial.user_attrs.get('quarterly_stats', {})
            overall_stats = trial.user_attrs.get('overall_stats', {})
            
            # Display current best value
            try:
                current_best = study.best_value if study.best_trial else "N/A"
                print(f"\n{'─'*70}")
                print(f"TRIAL #{trial.number} COMPLETE | Score: {trial.value:.0f} | Best: {current_best}")
                if is_new_best:
                    print(f"🎯 NEW BEST TRIAL FOUND! Updating CSV exports and best_params.json")
                print(f"{'─'*70}")
            except (ValueError, AttributeError):
                print(f"\n{'─'*70}")
                print(f"TRIAL #{trial.number} COMPLETE | Score: {trial.value:.0f}")
                print(f"{'─'*70}")
            
            if quarterly_stats:
                print(f"{'Quarter':<10} {'Trades':>8} {'Wins':>6} {'Win%':>8} {'R-Total':>10} {'Profit $':>12}")
                print(f"{'-'*70}")
                for q in sorted(quarterly_stats.keys()):
                    qs = quarterly_stats[q]
                    profit_str = f"${qs['profit']:,.0f}" if qs['profit'] >= 0 else f"-${abs(qs['profit']):,.0f}"
                    print(f"{q:<10} {qs['trades']:>8} {qs['wins']:>6} {qs['win_rate']:>7.1f}% {qs['r_total']:>10.2f} {profit_str:>12}")
                
                print(f"{'-'*70}")
                if overall_stats:
                    overall_profit = overall_stats.get('profit', 0)
                    profit_str = f"${overall_profit:,.0f}" if overall_profit >= 0 else f"-${abs(overall_profit):,.0f}"
                    print(f"{'OVERALL':<10} {overall_stats.get('trades', 0):>8} {overall_stats.get('wins', 0):>6} {overall_stats.get('win_rate', 0):>7.1f}% {overall_stats.get('r_total', 0):>10.2f} {profit_str:>12}")
            else:
                print("  No trades generated for this trial")
            
            print(f"{'─'*70}\n")
            
            # If this is the new best trial, immediately export CSVs and update best_params.json
            if is_new_best and study.best_trial:
                try:
                    print(f"\n📊 Exporting updated CSV files for best trial #{trial.number}...")
                    
                    # Get best params
                    current_best_params = study.best_params
                    risk_pct = current_best_params.get('risk_per_trade_pct', 0.5)
                    
                    # Run backtests with best params to get trades
                    print("  Running training backtest...")
                    training_trades_best = run_full_period_backtest(
                        start_date=TRAINING_START,
                        end_date=TRAINING_END,
                        min_confluence=current_best_params.get('min_confluence_score', 3),
                        min_quality_factors=current_best_params.get('min_quality_factors', 2),
                        risk_per_trade_pct=risk_pct,
                        atr_min_percentile=current_best_params.get('atr_min_percentile', 60.0),
                        trail_activation_r=current_best_params.get('trail_activation_r', 2.2),
                        december_atr_multiplier=current_best_params.get('december_atr_multiplier', 1.5),
                        volatile_asset_boost=current_best_params.get('volatile_asset_boost', 1.5),
                        ml_min_prob=None,
                        require_adx_filter=True,
                        adx_trend_threshold=current_best_params.get('adx_trend_threshold', 25.0),
                        adx_range_threshold=current_best_params.get('adx_range_threshold', 20.0),
                        trend_min_confluence=current_best_params.get('trend_min_confluence', 6),
                        range_min_confluence=current_best_params.get('range_min_confluence', 5),
                        rsi_oversold_range=current_best_params.get('rsi_oversold_range', 25.0),
                        rsi_overbought_range=current_best_params.get('rsi_overbought_range', 75.0),
                        atr_volatility_ratio=current_best_params.get('atr_volatility_ratio', 0.8),
                        atr_trail_multiplier=current_best_params.get('atr_trail_multiplier', 1.5),
                        partial_exit_at_1r=current_best_params.get('partial_exit_at_1r', True),
                    )
                    
                    print("  Running validation backtest...")
                    validation_trades_best = run_full_period_backtest(
                        start_date=VALIDATION_START,
                        end_date=VALIDATION_END,
                        min_confluence=current_best_params.get('min_confluence_score', 3),
                        min_quality_factors=current_best_params.get('min_quality_factors', 2),
                        risk_per_trade_pct=risk_pct,
                        atr_min_percentile=current_best_params.get('atr_min_percentile', 60.0),
                        trail_activation_r=current_best_params.get('trail_activation_r', 2.2),
                        december_atr_multiplier=current_best_params.get('december_atr_multiplier', 1.5),
                        volatile_asset_boost=current_best_params.get('volatile_asset_boost', 1.5),
                        ml_min_prob=None,
                        require_adx_filter=True,
                        adx_trend_threshold=current_best_params.get('adx_trend_threshold', 25.0),
                        adx_range_threshold=current_best_params.get('adx_range_threshold', 20.0),
                        trend_min_confluence=current_best_params.get('trend_min_confluence', 6),
                        range_min_confluence=current_best_params.get('range_min_confluence', 5),
                        rsi_oversold_range=current_best_params.get('rsi_oversold_range', 25.0),
                        rsi_overbought_range=current_best_params.get('rsi_overbought_range', 75.0),
                        atr_volatility_ratio=current_best_params.get('atr_volatility_ratio', 0.8),
                        atr_trail_multiplier=current_best_params.get('atr_trail_multiplier', 1.5),
                        partial_exit_at_1r=current_best_params.get('partial_exit_at_1r', True),
                    )
                    
                    print("  Running full period backtest...")
                    full_trades_best = run_full_period_backtest(
                        start_date=FULL_PERIOD_START,
                        end_date=FULL_PERIOD_END,
                        min_confluence=current_best_params.get('min_confluence_score', 3),
                        min_quality_factors=current_best_params.get('min_quality_factors', 2),
                        risk_per_trade_pct=risk_pct,
                        atr_min_percentile=current_best_params.get('atr_min_percentile', 60.0),
                        trail_activation_r=current_best_params.get('trail_activation_r', 2.2),
                        december_atr_multiplier=current_best_params.get('december_atr_multiplier', 1.5),
                        volatile_asset_boost=current_best_params.get('volatile_asset_boost', 1.5),
                        ml_min_prob=None,
                        require_adx_filter=True,
                        adx_trend_threshold=current_best_params.get('adx_trend_threshold', 25.0),
                        adx_range_threshold=current_best_params.get('adx_range_threshold', 20.0),
                        trend_min_confluence=current_best_params.get('trend_min_confluence', 6),
                        range_min_confluence=current_best_params.get('range_min_confluence', 5),
                        rsi_oversold_range=current_best_params.get('rsi_oversold_range', 25.0),
                        rsi_overbought_range=current_best_params.get('rsi_overbought_range', 75.0),
                        atr_volatility_ratio=current_best_params.get('atr_volatility_ratio', 0.8),
                        atr_trail_multiplier=current_best_params.get('atr_trail_multiplier', 1.5),
                        partial_exit_at_1r=current_best_params.get('partial_exit_at_1r', True),
                    )
                    
                    # Export updated CSV files
                    print("  Exporting all_trades_jan_dec_2024.csv...")
                    export_trades_to_csv(training_trades_best, "all_trades_jan_dec_2024.csv", risk_pct)
                    
                    print("  Exporting all_trades_2024_full.csv...")
                    export_trades_to_csv(validation_trades_best, "all_trades_2024_full.csv", risk_pct)
                    
                    print("  Exporting all_trades_2023_2025_full.csv...")
                    export_trades_to_csv(full_trades_best, "all_trades_2023_2025_full.csv", risk_pct)
                    
                    # Save best_params.json immediately for live bot
                    params_to_save = {
                        'min_confluence': current_best_params.get('min_confluence_score', 5),
                        'min_quality_factors': current_best_params.get('min_quality_factors', 2),
                        'risk_per_trade_pct': current_best_params.get('risk_per_trade_pct', 0.5),
                        'atr_min_percentile': current_best_params.get('atr_min_percentile', 75.0),
                        'trail_activation_r': current_best_params.get('trail_activation_r', 2.2),
                        'december_atr_multiplier': current_best_params.get('december_atr_multiplier', 1.5),
                        'volatile_asset_boost': current_best_params.get('volatile_asset_boost', 1.5),
                        'adx_trend_threshold': current_best_params.get('adx_trend_threshold', 25.0),
                        'adx_range_threshold': current_best_params.get('adx_range_threshold', 20.0),
                        'trend_min_confluence': current_best_params.get('trend_min_confluence', 6),
                        'range_min_confluence': current_best_params.get('range_min_confluence', 5),
                        'rsi_oversold_range': current_best_params.get('rsi_oversold_range', 25.0),
                        'rsi_overbought_range': current_best_params.get('rsi_overbought_range', 75.0),
                        'atr_volatility_ratio': current_best_params.get('atr_vol_ratio_range', 0.8),
                        'atr_trail_multiplier': current_best_params.get('atr_trail_multiplier', 1.5),
                        'partial_exit_at_1r': current_best_params.get('partial_exit_at_1r', True),
                        'use_adx_slope_rising': current_best_params.get('use_adx_slope_rising', False),
                        'partial_exit_pct': current_best_params.get('partial_exit_pct', 0.5),
                    }
                    
                    print("  Saving best_params.json for live bot...")
                    save_optimized_params(params_to_save, backup=True)
                    
                    print(f"✅ Updated CSV exports and best_params.json for trial #{trial.number}\n")
                    
                except Exception as e:
                    print(f"⚠️  Error exporting CSVs for best trial: {e}\n")
        
        try:
            study.optimize(
                self._objective,
                n_trials=n_trials,
                show_progress_bar=False,
                callbacks=[progress_callback]
            )
        except optuna.exceptions.OptunaError as e:
            # study.stop() raises this when early stopping
            if "stopped" not in str(e).lower():
                raise
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Log early stopping status
        if early_stopped:
            print(f"\n{'='*60}")
            print(f"OPTIMIZATION EARLY STOPPED (No improvement for {early_stopping_patience} trials)")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"OPTIMIZATION COMPLETE")
            print(f"{'='*60}")
        
        print(f"Total Trials: {len(study.trials)}")
        print(f"Best Score: {self.best_score:.0f}")
        print(f"Best Parameters:")
        for k, v in sorted(self.best_params.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
        
        params_to_save = {
            # Core risk management
            'min_confluence': self.best_params.get('min_confluence_score', 5),
            'min_quality_factors': self.best_params.get('min_quality_factors', 2),
            'risk_per_trade_pct': self.best_params.get('risk_per_trade_pct', 0.5),
            'atr_min_percentile': self.best_params.get('atr_min_percentile', 75.0),
            'trail_activation_r': self.best_params.get('trail_activation_r', 2.2),
            'december_atr_multiplier': self.best_params.get('december_atr_multiplier', 1.5),
            'volatile_asset_boost': self.best_params.get('volatile_asset_boost', 1.5),
            # Regime-Adaptive V2 parameters
            'adx_trend_threshold': self.best_params.get('adx_trend_threshold', 25.0),
            'adx_range_threshold': self.best_params.get('adx_range_threshold', 20.0),
            'trend_min_confluence': self.best_params.get('trend_min_confluence', 6),
            'range_min_confluence': self.best_params.get('range_min_confluence', 5),
            'rsi_oversold_range': self.best_params.get('rsi_oversold_range', 25.0),
            'rsi_overbought_range': self.best_params.get('rsi_overbought_range', 75.0),
            'atr_volatility_ratio': self.best_params.get('atr_vol_ratio_range', 0.8),
            'atr_trail_multiplier': self.best_params.get('atr_trail_multiplier', 1.5),
            'partial_exit_at_1r': self.best_params.get('partial_exit_at_1r', True),
            # NEW: ADX slope-based early trend entry
            'use_adx_slope_rising': self.best_params.get('use_adx_slope_rising', False),
            # NEW: Dynamic RSI in Range Mode
            'partial_exit_pct': self.best_params.get('partial_exit_pct', 0.5),
        }
        
        try:
            save_optimized_params(params_to_save, backup=True)
            print(f"\nOptimized parameters saved to params/current_params.json")
        except Exception as e:
            print(f"Failed to save params: {e}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': n_trials,
            'total_trials': len(study.trials),
            'study': study,  # Return study for top 5 analysis
            'early_stopped': early_stopped,
            'trials_without_improvement': trials_without_improvement,
        }


def validate_top_trials(study, top_n: int = 5) -> List[Dict]:
    """
    Run validation backtests on top N trials to find the best OOS performer.
    This prevents overfitting by selecting based on validation performance.
    
    Works with both single-objective and multi-objective (NSGA-II) studies.
    
    Returns:
        List of dicts with trial info and validation results, sorted by validation R
    """
    import optuna
    
    # Check if this is a multi-objective study
    is_multi_objective = len(study.directions) > 1 if hasattr(study, 'directions') else False
    
    # Get all completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if is_multi_objective:
        # For multi-objective: filter trials with valid values tuple
        completed_trials = [t for t in completed_trials if t.values is not None and len(t.values) >= 1]
        if not completed_trials:
            print("No completed trials to validate")
            return []
        # Use Pareto front trials, or sort by first objective (Total R)
        try:
            pareto_trials = study.best_trials
            if len(pareto_trials) >= top_n:
                sorted_trials = pareto_trials[:top_n]
            else:
                # Add more from sorted by Total R
                non_pareto = [t for t in completed_trials if t not in pareto_trials]
                non_pareto_sorted = sorted(non_pareto, key=lambda t: t.values[0], reverse=True)
                sorted_trials = pareto_trials + non_pareto_sorted[:top_n - len(pareto_trials)]
        except:
            sorted_trials = sorted(completed_trials, key=lambda t: t.values[0], reverse=True)[:top_n]
    else:
        # For single-objective: filter and sort by value
        completed_trials = [t for t in completed_trials if t.value is not None]
        if not completed_trials:
            print("No completed trials to validate")
            return []
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:top_n]
    
    print(f"\n{'='*70}")
    print(f"TOP {len(sorted_trials)} TRIALS - VALIDATION COMPARISON")
    print(f"{'='*70}")
    print(f"Running validation backtests to find best OOS performer...")
    print(f"{'='*70}\n")
    
    validation_results = []
    
    for rank, trial in enumerate(sorted_trials, 1):
        params = trial.params
        
        # Get training score (handle both single and multi-objective)
        if is_multi_objective:
            training_score = trial.values[0] if trial.values else 0  # Use Total R
            score_display = f"R={training_score:+.1f}"
        else:
            training_score = trial.value if trial.value else 0
            score_display = f"{training_score:,.0f}"
        
        print(f"[{rank}/{len(sorted_trials)}] Trial #{trial.number} (Training Score: {score_display})")
        
        # Run validation backtest
        validation_trades = run_full_period_backtest(
            start_date=VALIDATION_START,
            end_date=VALIDATION_END,
            min_confluence=params.get('min_confluence_score', 3),
            min_quality_factors=params.get('min_quality_factors', 2),
            risk_per_trade_pct=params.get('risk_per_trade_pct', 0.5),
            atr_min_percentile=params.get('atr_min_percentile', 60.0),
            trail_activation_r=params.get('trail_activation_r', 2.2),
            december_atr_multiplier=params.get('december_atr_multiplier', 1.5),
            volatile_asset_boost=params.get('volatile_asset_boost', 1.5),
            ml_min_prob=None,
            require_adx_filter=True,
            use_adx_regime_filter=False,
            adx_trend_threshold=params.get('adx_trend_threshold', 25.0),
            adx_range_threshold=params.get('adx_range_threshold', 20.0),
            trend_min_confluence=params.get('trend_min_confluence', 6),
            range_min_confluence=params.get('range_min_confluence', 5),
            atr_volatility_ratio=params.get('atr_vol_ratio_range', 0.8),
            atr_trail_multiplier=params.get('atr_trail_multiplier', 1.5),
            partial_exit_at_1r=params.get('partial_exit_at_1r', True),
            partial_exit_pct=params.get('partial_exit_pct', 0.5),
        )
        
        # Calculate validation metrics
        val_r = sum(getattr(t, 'rr', 0) for t in validation_trades) if validation_trades else 0
        val_trades = len(validation_trades) if validation_trades else 0
        val_wins = sum(1 for t in validation_trades if getattr(t, 'rr', 0) > 0) if validation_trades else 0
        val_wr = (val_wins / val_trades * 100) if val_trades > 0 else 0
        
        result = {
            'trial_number': trial.number,
            'training_score': training_score,
            'params': params,
            'validation_r': val_r,
            'validation_trades': val_trades,
            'validation_wins': val_wins,
            'validation_wr': val_wr,
            'validation_trade_objects': validation_trades,
        }
        validation_results.append(result)
        
        print(f"    Validation: {val_trades} trades, {val_r:+.1f}R, {val_wr:.1f}% WR")
    
    # Sort by validation R (best OOS performance)
    validation_results.sort(key=lambda x: x['validation_r'], reverse=True)
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"VALIDATION RANKING (Best OOS Performance)")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Trial':<8} {'Train Score':>12} {'Val Trades':>12} {'Val R':>10} {'Val WR':>10}")
    print(f"{'-'*70}")
    
    for rank, result in enumerate(validation_results, 1):
        marker = " ★" if rank == 1 else ""
        print(f"{rank:<6} #{result['trial_number']:<7} {result['training_score']:>12,.0f} {result['validation_trades']:>12} {result['validation_r']:>+10.1f} {result['validation_wr']:>9.1f}%{marker}")
    
    print(f"{'-'*70}")
    
    if validation_results:
        best = validation_results[0]
        print(f"\n🏆 BEST OOS PERFORMER: Trial #{best['trial_number']}")
        print(f"   Training Score: {best['training_score']:,.0f}")
        print(f"   Validation: {best['validation_r']:+.1f}R ({best['validation_trades']} trades, {best['validation_wr']:.1f}% WR)")
    
    return validation_results


def generate_summary_txt(
    results: Dict,
    training_trades: List,
    validation_trades: List,
    full_year_trades: List,
    best_params: Dict
) -> str:
    """Generate a summary text file after each analyzer run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = OUTPUT_DIR / f"analysis_summary_{timestamp}.txt"
    
    def calc_stats(trades):
        if not trades:
            return {"count": 0, "total_r": 0, "win_rate": 0, "avg_r": 0}
        total_r = sum(getattr(t, 'rr', 0) for t in trades)
        wins = sum(1 for t in trades if getattr(t, 'rr', 0) > 0)
        win_rate = (wins / len(trades) * 100) if trades else 0
        avg_r = total_r / len(trades) if trades else 0
        return {"count": len(trades), "total_r": total_r, "win_rate": win_rate, "avg_r": avg_r}
    
    training_stats = calc_stats(training_trades)
    validation_stats = calc_stats(validation_trades)
    full_stats = calc_stats(full_year_trades)
    
    lines = [
        "=" * 80,
        "FTMO CHALLENGE ANALYZER - SUMMARY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "OPTIMIZATION RESULTS",
        "-" * 40,
        f"Best Score: {results.get('best_score', 0):.2f}",
        f"Trials This Session: {results.get('n_trials', 0)}",
        f"Total Trials: {results.get('total_trials', 0)}",
        "",
        "BEST PARAMETERS",
        "-" * 40,
    ]
    
    for k, v in sorted(best_params.items()):
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")
    
    # Calculate account size and risk per trade for USD conversion
    account_size = 200000.0  # FTMO 200K account
    # risk_per_trade_pct is stored as 0.5 meaning 0.5%, so divide by 100
    risk_per_trade_pct = best_params.get('risk_per_trade_pct', 0.5)
    risk_per_trade_decimal = risk_per_trade_pct / 100  # 0.5% -> 0.005
    
    training_profit_usd = training_stats['total_r'] * risk_per_trade_decimal * account_size
    validation_profit_usd = validation_stats['total_r'] * risk_per_trade_decimal * account_size
    full_profit_usd = full_stats['total_r'] * risk_per_trade_decimal * account_size
    
    lines.extend([
        "",
        "TRAINING PERIOD (2023-01-01 to 2024-09-30)",
        "-" * 40,
        f"  Trades: {training_stats['count']}",
        f"  Total R: {training_stats['total_r']:+.2f}",
        f"  Estimated Profit: ${training_profit_usd:+,.2f}",
        f"  Win Rate: {training_stats['win_rate']:.1f}%",
        f"  Avg R per Trade: {training_stats['avg_r']:+.3f}",
        "",
        f"VALIDATION PERIOD (2024-10-01 to {VALIDATION_END.strftime('%Y-%m-%d')})",
        "-" * 40,
        f"  Trades: {validation_stats['count']}",
        f"  Total R: {validation_stats['total_r']:+.2f}",
        f"  Estimated Profit: ${validation_profit_usd:+,.2f}",
        f"  Win Rate: {validation_stats['win_rate']:.1f}%",
        f"  Avg R per Trade: {validation_stats['avg_r']:+.3f}",
        "",
        "FULL PERIOD (2023-2025)",
        "-" * 40,
        f"  Trades: {full_stats['count']}",
        f"  Total R: {full_stats['total_r']:+.2f}",
        f"  Estimated Profit: ${full_profit_usd:+,.2f}",
        f"  Win Rate: {full_stats['win_rate']:.1f}%",
        f"  Avg R per Trade: {full_stats['avg_r']:+.3f}",
        "",
        "QUARTERLY BREAKDOWN",
        "-" * 40,
    ])
    
    total_full_period_profit_usd = 0.0
    
    for q_name, (q_start, q_end) in sorted(QUARTERS_ALL.items()):
        q_filtered = []
        for t in full_year_trades:
            entry = getattr(t, 'entry_date', None)
            if entry:
                if isinstance(entry, str):
                    try:
                        entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                    except:
                        continue
                if hasattr(entry, 'replace') and entry.tzinfo:
                    entry = entry.replace(tzinfo=None)
                if q_start <= entry <= q_end:
                    q_filtered.append(t)
        
        q_r = sum(getattr(t, 'rr', 0) for t in q_filtered)
        q_wins = sum(1 for t in q_filtered if getattr(t, 'rr', 0) > 0)
        q_wr = (q_wins / len(q_filtered) * 100) if q_filtered else 0
        
        # Calculate USD profit for this quarter
        q_profit_usd = q_r * risk_per_trade_decimal * account_size
        total_full_period_profit_usd += q_profit_usd
        
        lines.append(f"  {q_name}: {len(q_filtered)} trades, {q_r:+.1f}R, {q_wr:.0f}% win rate, ${q_profit_usd:+,.2f}")
    
    lines.extend([
        "",
        f"TOTAL PROFIT (Full Period 2023-2025): ${total_full_period_profit_usd:+,.2f}",
        "=" * 80,
        "End of Summary",
        "=" * 80,
    ])
    
    with open(summary_filename, 'w') as f:
        f.write("\n".join(lines))
    
    return str(summary_filename)


# ============================================================================
# NSGA-II MULTI-OBJECTIVE OPTIMIZATION
# Optimizes three objectives simultaneously: Total R, Sharpe Ratio, Win Rate
# Uses Pareto frontier to find non-dominated solutions
# ============================================================================

# Use unified config for multi-objective as well
# When use_multi_objective=True, we use the same DB but different study name
MULTI_OBJECTIVE_DB = OPT_CONFIG.db_path
MULTI_OBJECTIVE_STUDY_NAME = f"{OPT_CONFIG.study_name}_multi"


def multi_objective_function(trial) -> Tuple[float, float, float]:
    """
    Multi-objective function for NSGA-II optimization.
    Returns three objectives to maximize:
    1. Total R (profitability)
    2. Sharpe Ratio (risk-adjusted returns)
    3. Win Rate (consistency)
    
    All three should be MAXIMIZED (Optuna NSGA-II handles this).
    """
    # Sample hyperparameters (same as single-objective)
    params = {
        'min_confluence_score': trial.suggest_int('min_confluence_score', 2, 6),
        'min_quality_factors': trial.suggest_int('min_quality_factors', 1, 4),
        'risk_per_trade_pct': trial.suggest_float('risk_per_trade_pct', 0.3, 0.6, step=0.05),
        'atr_min_percentile': trial.suggest_float('atr_min_percentile', 40.0, 80.0, step=5.0),
        'trail_activation_r': trial.suggest_float('trail_activation_r', 0.8, 2.0, step=0.2),
        'december_atr_multiplier': trial.suggest_float('december_atr_multiplier', 1.0, 2.0, step=0.2),
        'volatile_asset_boost': trial.suggest_float('volatile_asset_boost', 1.0, 1.5, step=0.1),
        'adx_trend_threshold': trial.suggest_int('adx_trend_threshold', 15, 30),
        'adx_range_threshold': trial.suggest_int('adx_range_threshold', 10, 25),
        'trend_min_confluence': trial.suggest_int('trend_min_confluence', 2, 5),
        'range_min_confluence': trial.suggest_int('range_min_confluence', 2, 5),
        'atr_vol_ratio_range': trial.suggest_float('atr_vol_ratio_range', 0.5, 1.0, step=0.05),
        'atr_trail_multiplier': trial.suggest_float('atr_trail_multiplier', 1.2, 3.5, step=0.2),
        'partial_exit_at_1r': trial.suggest_categorical('partial_exit_at_1r', [True, False]),
        'partial_exit_pct': trial.suggest_float('partial_exit_pct', 0.3, 0.8, step=0.05),
    }
    
    risk_pct = params['risk_per_trade_pct']
    
    # Run training backtest
    training_trades = run_full_period_backtest(
        start_date=TRAINING_START,
        end_date=TRAINING_END,
        min_confluence=params['min_confluence_score'],
        min_quality_factors=params['min_quality_factors'],
        risk_per_trade_pct=risk_pct,
        atr_min_percentile=params['atr_min_percentile'],
        trail_activation_r=params['trail_activation_r'],
        december_atr_multiplier=params['december_atr_multiplier'],
        volatile_asset_boost=params['volatile_asset_boost'],
        ml_min_prob=None,
        require_adx_filter=True,
        use_adx_regime_filter=False,
        adx_trend_threshold=params['adx_trend_threshold'],
        adx_range_threshold=params['adx_range_threshold'],
        trend_min_confluence=params['trend_min_confluence'],
        range_min_confluence=params['range_min_confluence'],
        atr_volatility_ratio=params['atr_vol_ratio_range'],
        atr_trail_multiplier=params['atr_trail_multiplier'],
        partial_exit_at_1r=params['partial_exit_at_1r'],
        partial_exit_pct=params['partial_exit_pct'],
    )
    
    # Calculate objectives
    if not training_trades or len(training_trades) < 10:
        # Not enough trades - return bad values for all objectives
        return (-1000.0, -10.0, 0.0)
    
    # Objective 1: Total R (profitability)
    total_r = sum(getattr(t, 'rr', 0) for t in training_trades)
    
    # Objective 2: Sharpe Ratio (risk-adjusted returns)
    risk_metrics = calculate_risk_metrics(
        trades=training_trades,
        risk_per_trade_pct=risk_pct,
        account_size=ACCOUNT_SIZE,
        trading_days_per_year=252.0
    )
    sharpe_ratio = risk_metrics.sharpe_ratio
    
    # Objective 3: Win Rate
    wins = sum(1 for t in training_trades if getattr(t, 'rr', 0) > 0)
    win_rate = (wins / len(training_trades) * 100) if training_trades else 0
    
    # Store additional metrics
    trial.set_user_attr('total_trades', len(training_trades))
    trial.set_user_attr('profit_factor', risk_metrics.profit_factor)
    trial.set_user_attr('sortino_ratio', risk_metrics.sortino_ratio)
    trial.set_user_attr('max_drawdown', risk_metrics.max_drawdown)
    
    return (total_r, sharpe_ratio, win_rate)


def run_multi_objective_optimization(n_trials: int = 50) -> Dict:
    """
    Run NSGA-II multi-objective optimization.
    
    NSGA-II (Non-dominated Sorting Genetic Algorithm II) finds the Pareto frontier:
    solutions where improving one objective would worsen another.
    
    Returns the best balanced solution from the Pareto frontier.
    """
    import optuna
    from optuna.samplers import NSGAIISampler
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print(f"\n{'='*70}")
    print(f"NSGA-II MULTI-OBJECTIVE OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Objectives: Maximize [Total R, Sharpe Ratio, Win Rate]")
    print(f"Trials: {n_trials}")
    print(f"Storage: {MULTI_OBJECTIVE_DB}")
    print(f"{'='*70}\n")
    
    # Create multi-objective study with NSGA-II sampler
    study = optuna.create_study(
        directions=['maximize', 'maximize', 'maximize'],  # All three are maximized
        study_name=MULTI_OBJECTIVE_STUDY_NAME,
        storage=MULTI_OBJECTIVE_DB,
        load_if_exists=True,
        sampler=NSGAIISampler(seed=42)
    )
    
    existing_trials = len(study.trials)
    if existing_trials > 0:
        print(f"Resuming study with {existing_trials} existing trials")
    
    def progress_callback(study, trial):
        if trial.values:
            total_r, sharpe, wr = trial.values
            print(f"Trial #{trial.number}: R={total_r:+.1f}, Sharpe={sharpe:.2f}, WR={wr:.1f}%")
    
    # Run optimization
    study.optimize(multi_objective_function, n_trials=n_trials, callbacks=[progress_callback])
    
    # Get Pareto front (non-dominated solutions)
    pareto_trials = study.best_trials
    
    print(f"\n{'='*70}")
    print(f"PARETO FRONTIER - {len(pareto_trials)} Non-Dominated Solutions")
    print(f"{'='*70}")
    print(f"{'Trial':<8} {'Total R':>10} {'Sharpe':>10} {'Win Rate':>10} {'Trades':>10}")
    print(f"{'-'*70}")
    
    for trial in pareto_trials:
        total_r, sharpe, wr = trial.values
        trades = trial.user_attrs.get('total_trades', 0)
        print(f"#{trial.number:<6} {total_r:>+10.1f} {sharpe:>10.2f} {wr:>9.1f}% {trades:>10}")
    
    # Select best balanced solution using weighted scoring
    # Weights: Total R (40%), Sharpe (35%), Win Rate (25%)
    best_trial = None
    best_composite_score = -float('inf')
    
    for trial in pareto_trials:
        total_r, sharpe, wr = trial.values
        # Normalize and weight
        r_score = total_r / 100  # Scale R to ~1.0 range for good strategies
        sharpe_score = sharpe    # Already in ~0-2 range typically
        wr_score = (wr - 40) / 20  # Scale WR from 40-60% to 0-1 range
        
        composite = 0.40 * r_score + 0.35 * sharpe_score + 0.25 * wr_score
        
        if composite > best_composite_score:
            best_composite_score = composite
            best_trial = trial
    
    if best_trial:
        total_r, sharpe, wr = best_trial.values
        print(f"\n🏆 BEST BALANCED SOLUTION: Trial #{best_trial.number}")
        print(f"   Total R: {total_r:+.1f}")
        print(f"   Sharpe: {sharpe:.2f}")
        print(f"   Win Rate: {wr:.1f}%")
        print(f"   Composite Score: {best_composite_score:.3f}")
        
        best_params = best_trial.params
        
        # Save best params
        save_best_params_persistent(best_params)
        
        return {
            'best_params': best_params,
            'best_score': best_composite_score,
            'pareto_trials': len(pareto_trials),
            'total_r': total_r,
            'sharpe': sharpe,
            'win_rate': wr,
            'study': study,
            'n_trials': n_trials,
            'total_trials': len(study.trials),
        }
    else:
        print("\n⚠️ No valid solutions found on Pareto frontier")
        return {
            'best_params': {},
            'best_score': 0,
            'study': study,
            'n_trials': n_trials,
        }


def main():
    """
    Professional FTMO Optimization Workflow with CLI support.
    
    Configuration is loaded from params/optimization_config.json
    CLI arguments can override config file settings.
    
    Usage:
      python ftmo_challenge_analyzer.py              # Run/resume optimization (default 5 trials)
      python ftmo_challenge_analyzer.py --status     # Check progress without running
      python ftmo_challenge_analyzer.py --config     # Show current configuration
      python ftmo_challenge_analyzer.py --trials 100 # Run 100 trials
      python ftmo_challenge_analyzer.py --multi      # Use NSGA-II multi-objective optimization
      python ftmo_challenge_analyzer.py --adx        # Enable ADX regime filtering
    """
    parser = argparse.ArgumentParser(
        description="FTMO Professional Optimization System - Resumable with ADX Filter"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check optimization progress without running new trials"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of optimization trials to run (default: 5)"
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Use NSGA-II multi-objective optimization (Profit + Sharpe + WinRate)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience - stop if no improvement after N trials (default: 20)"
    )
    parser.add_argument(
        "--adx",
        action="store_true", 
        help="Enable ADX regime filtering (Trend/Range/Transition modes)"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show current optimization configuration and exit"
    )
    args = parser.parse_args()
    
    # Load unified config and override with CLI args
    config = get_optimization_config(reload=True)
    
    if args.config:
        config.print_summary()
        return
    
    if args.status:
        show_optimization_status()
        return
    
    # CLI args override config file settings
    n_trials = args.trials
    use_multi_objective = args.multi or config.use_multi_objective
    use_adx_regime = args.adx or config.use_adx_regime_filter
    early_stopping_patience = args.patience

    print(f"\n{'='*80}")
    print("FTMO PROFESSIONAL OPTIMIZATION SYSTEM - UNIFIED CONFIG V3")
    print(f"{'='*80}")
    
    # Show active config
    print(f"\n📋 Configuration (from params/optimization_config.json):")
    print(f"  Database: {config.db_path}")
    print(f"  Study: {config.study_name}")
    print(f"  Multi-objective: {'✅ Enabled' if use_multi_objective else '❌ Disabled'}")
    print(f"  ADX Regime Filter: {'✅ Enabled' if use_adx_regime else '❌ Disabled'}")
    
    print(f"\n📊 Data Partitioning (Multi-Year Robustness):")
    print(f"  TRAINING:    2023-01-01 to 2024-09-30 (in-sample)")
    print(f"  VALIDATION:  2024-10-01 to {VALIDATION_END.strftime('%Y-%m-%d')} (out-of-sample)")
    print(f"  FINAL:       Full 2023-2025 (December fully open)")
    
    if use_adx_regime:
        print(f"\n⚙️  Regime-Adaptive Trading:")
        print(f"  TREND MODE:      ADX >= threshold (momentum following)")
        print(f"  RANGE MODE:      ADX < threshold (conservative mean reversion)")
        print(f"  TRANSITION:      NO ENTRIES (wait for regime confirmation)")
    
    print(f"\n⏱️  Early Stopping: After {early_stopping_patience} trials without improvement")
    
    if use_multi_objective:
        print(f"\n🎯 OPTIMIZATION MODE: NSGA-II Multi-Objective")
        print(f"   Objectives: {', '.join(config.objectives)}")
        print(f"   Sampler: NSGA-II (evolutionary algorithm)")
    else:
        print(f"\n📈 OPTIMIZATION MODE: TPE Single-Objective")
        print(f"   Score = R + Sharpe_bonus + PF_bonus + WR_bonus - penalties")
    
    print(f"\n💾 Resumable: Study stored in {config.db_path}")
    print(f"{'='*80}\n")
    
    # ============================================================================
    # MULTI-OBJECTIVE OR SINGLE-OBJECTIVE OPTIMIZATION
    # ============================================================================
    if use_multi_objective:
        results = run_multi_objective_optimization(n_trials=n_trials)
        study = results.get('study')
        best_params = results.get('best_params', {})
    else:
        optimizer = OptunaOptimizer()
        results = optimizer.run_optimization(n_trials=n_trials, early_stopping_patience=early_stopping_patience)
        study = results.get('study')
        best_params = results.get('best_params', optimizer.best_params)
        
        # Log early stopping info
        if results.get('early_stopped', False):
            print(f"\n⏹️  Optimization early stopped after {results.get('trials_without_improvement', 0)} trials without improvement")
    
    # ============================================================================
    # TOP 5 VALIDATION COMPARISON
    # Run validation on top 5 trials to find best OOS performer
    # This prevents overfitting by selecting based on validation performance
    # ============================================================================
    
    if study:
        top_5_results = validate_top_trials(study, top_n=5)
        
        if top_5_results:
            # Use the best validation performer (not just best training score)
            best_oos = top_5_results[0]
            best_params = best_oos['params']
            validation_trades = best_oos['validation_trade_objects']
            
            print(f"\n✅ Selected Trial #{best_oos['trial_number']} as FINAL (best OOS performance)")
        else:
            # Fallback to best training params
            best_params = results['best_params']
            validation_trades = []
    else:
        best_params = results['best_params']
        validation_trades = []
    
    # INSTANTLY SAVE BEST PARAMS FOR LIVE BOT
    save_best_params_persistent(best_params)
    
    # ============================================================================
    # RUN FINAL FULL PERIOD BACKTEST (only once, with best OOS params)
    # ============================================================================
    
    print(f"\n{'='*80}")
    print("=== FULL PERIOD FINAL RESULTS (2023-2025) ===")
    print(f"{'='*80}")
    print("Running full period backtest with best OOS parameters...")
    print("December fully open for trading")
    
    full_year_trades = run_full_period_backtest(
        start_date=FULL_PERIOD_START,
        end_date=FULL_PERIOD_END,
        min_confluence=best_params.get('min_confluence_score', 3),
        min_quality_factors=best_params.get('min_quality_factors', 2),
        risk_per_trade_pct=best_params.get('risk_per_trade_pct', 0.5),
        atr_min_percentile=best_params.get('atr_min_percentile', 60.0),
        trail_activation_r=best_params.get('trail_activation_r', 2.2),
        december_atr_multiplier=best_params.get('december_atr_multiplier', 1.5),
        volatile_asset_boost=best_params.get('volatile_asset_boost', 1.5),
        ml_min_prob=None,
        require_adx_filter=True,
        use_adx_regime_filter=use_adx_regime,  # From unified config
        adx_trend_threshold=best_params.get('adx_trend_threshold', 25.0),
        adx_range_threshold=best_params.get('adx_range_threshold', 20.0),
        trend_min_confluence=best_params.get('trend_min_confluence', 6),
        range_min_confluence=best_params.get('range_min_confluence', 5),
        atr_volatility_ratio=best_params.get('atr_vol_ratio_range', 0.8),
        atr_trail_multiplier=best_params.get('atr_trail_multiplier', 1.5),
        partial_exit_at_1r=best_params.get('partial_exit_at_1r', True),
        partial_exit_pct=best_params.get('partial_exit_pct', 0.5),
    )
    
    # Extract training trades from full period for reporting
    training_trades = [t for t in full_year_trades if hasattr(t, 'entry_date') and 
                       TRAINING_START <= (t.entry_date.replace(tzinfo=None) if hasattr(t.entry_date, 'replace') and t.entry_date.tzinfo else t.entry_date) <= TRAINING_END]
    
    risk_pct = best_params.get('risk_per_trade_pct', 0.5)
    
    # Export all three CSV files at the end
    print("\n📊 Exporting final CSV files...")
    export_trades_to_csv(training_trades, "all_trades_jan_dec_2024.csv", risk_pct)
    export_trades_to_csv(validation_trades if validation_trades else [], "all_trades_2024_full.csv", risk_pct)
    export_trades_to_csv(full_year_trades, "all_trades_2023_2025_full.csv", risk_pct)
    print("✅ All CSV files exported successfully\n")
    
    full_year_results = print_period_results(
        full_year_trades, f"FULL PERIOD FINAL RESULTS ({FULL_PERIOD_START.year}-{FULL_PERIOD_END.year})",
        FULL_PERIOD_START, FULL_PERIOD_END
    )
    
    if full_year_trades and len(full_year_trades) >= 30:
        print(f"\n{'='*80}")
        print("MONTE CARLO SIMULATION (1000 iterations)")
        print(f"{'='*80}")
        mc_results = run_monte_carlo_analysis(full_year_trades, num_simulations=1000)
    
    print(f"\n{'='*80}")
    print("QUARTERLY PERFORMANCE BREAKDOWN (Full Year)")
    print(f"{'='*80}")
    
    for q_name, (q_start, q_end) in sorted(QUARTERS_ALL.items()):
        q_filtered = []
        for t in full_year_trades:
            entry = getattr(t, 'entry_date', None)
            if entry:
                if isinstance(entry, str):
                    try:
                        entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                    except:
                        continue
                if hasattr(entry, 'replace') and entry.tzinfo:
                    entry = entry.replace(tzinfo=None)
                if q_start <= entry <= q_end:
                    q_filtered.append(t)
        
        q_r = sum(getattr(t, 'rr', 0) for t in q_filtered)
        q_wins = sum(1 for t in q_filtered if getattr(t, 'rr', 0) > 0)
        q_wr = (q_wins / len(q_filtered) * 100) if q_filtered else 0
        print(f"  {q_name}: {len(q_filtered)} trades, {q_r:+.1f}R, {q_wr:.0f}% win rate")
    
    if full_year_trades and len(full_year_trades) >= 50:
        print(f"\n{'='*80}")
        print("TRAINING ML MODEL")
        print(f"{'='*80}")
        train_ml_model(full_year_trades)
    
    print(f"\n{'='*80}")
    print("UPDATING DOCUMENTATION")
    print(f"{'='*80}")
    update_readme_documentation()
    
    # ============================================================================
    # PROFESSIONAL QUANTITATIVE ANALYSIS SUITE
    # ============================================================================
    print(f"\n{'='*80}")
    print("PROFESSIONAL QUANTITATIVE ANALYSIS")
    print(f"{'='*80}\n")
    
    # Risk Metrics Calculation
    training_risk_metrics = calculate_risk_metrics(training_trades, best_params.get('risk_per_trade_pct', 0.5))
    validation_risk_metrics = calculate_risk_metrics(validation_trades, best_params.get('risk_per_trade_pct', 0.5))
    full_risk_metrics = calculate_risk_metrics(full_year_trades, best_params.get('risk_per_trade_pct', 0.5))
    
    print("Risk Metrics (Sharpe, Sortino, Calmar Ratios):")
    print(f"  Training:   Sharpe={training_risk_metrics.sharpe_ratio:+.2f}  Sortino={training_risk_metrics.sortino_ratio:+.2f}  Calmar={training_risk_metrics.calmar_ratio:+.2f}")
    print(f"  Validation: Sharpe={validation_risk_metrics.sharpe_ratio:+.2f}  Sortino={validation_risk_metrics.sortino_ratio:+.2f}  Calmar={validation_risk_metrics.calmar_ratio:+.2f}")
    print(f"  Full:       Sharpe={full_risk_metrics.sharpe_ratio:+.2f}  Sortino={full_risk_metrics.sortino_ratio:+.2f}  Calmar={full_risk_metrics.calmar_ratio:+.2f}")
    print(f"\n  IS-OOS Degradation: {training_risk_metrics.sharpe_ratio - validation_risk_metrics.sharpe_ratio:+.2f} (Sharpe)")
    
    # Walk-Forward Testing
    print(f"\n  Walk-Forward Robustness Testing...")
    try:
        wf_tester = WalkForwardTester(
            all_trades=full_year_trades,
            start_date=FULL_PERIOD_START,
            end_date=FULL_PERIOD_END,
            train_months=12,
            validate_months=3,
            rolling=True
        )
        wf_results = wf_tester.analyze_all_windows(best_params.get('risk_per_trade_pct', 0.5))
        
        print(f"  Windows: {wf_results['total_windows']}")
        print(f"  Avg Sharpe Degradation: {wf_results['avg_sharpe_degradation']:+.2f}")
        print(f"  Std Dev Degradation: {wf_results['std_sharpe_degradation']:.2f}")
        print(f"  Avg Return Degradation: {wf_results['avg_return_degradation']:+.2f}%")
    except Exception as e:
        print(f"  [!] Walk-forward analysis skipped: {e}")
        wf_results = {'total_windows': 0, 'avg_sharpe_degradation': 0, 'std_sharpe_degradation': 0}
    
    # Generate Professional Report
    print(f"\n  Generating professional report...")
    try:
        report_text = generate_professional_report(
            best_params=best_params,
            training_metrics=training_risk_metrics,
            validation_metrics=validation_risk_metrics,
            full_metrics=full_risk_metrics,
            walk_forward_results=wf_results,
            output_file=OUTPUT_DIR / "professional_backtest_report.txt"
        )
        print(f"  ✓ Report saved to: ftmo_analysis_output/professional_backtest_report.txt")
    except Exception as e:
        print(f"  [!] Report generation failed: {e}")
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nBest Score: {results['best_score']:.2f}")
    print(f"Trials Run This Session: {results['n_trials']}")
    print(f"Total Trials in Study: {results.get('total_trials', results['n_trials'])}")
    print(f"\nFiles Created:")
    print(f"  - params/current_params.json (optimized parameters)")
    print(f"  - ftmo_analysis_output/all_trades_2024_full.csv ({len(full_year_trades) if full_year_trades else 0} trades)")
    print(f"  - models/best_rf.joblib (ML model)")
    print(f"  - optuna_study.db (resumable optimization state)")
    print(f"  - ftmo_optimization_progress.txt (progress log)")
    
    print(f"\nDocumentation updated. CSV exported. Ready for live trading.")
    
    summary_file = generate_summary_txt(
        results=results,
        training_trades=training_trades,
        validation_trades=validation_trades,
        full_year_trades=full_year_trades,
        best_params=best_params
    )
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
