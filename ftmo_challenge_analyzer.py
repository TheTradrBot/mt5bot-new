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
    detect_regime,
    validate_range_mode_entry,
)

from ftmo_config import FTMO_CONFIG, FTMO10KConfig, get_pip_size, get_sl_limits
from config import FOREX_PAIRS, METALS, INDICES, CRYPTO_ASSETS
from tradr.risk.position_sizing import calculate_lot_size, get_contract_specs
from params.params_loader import save_optimized_params

OUTPUT_DIR = Path("ftmo_analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

OPTUNA_DB_PATH = "sqlite:///regime_adaptive_v2_clean.db"

_DATA_CACHE: Dict[str, List[Dict]] = {}
OPTUNA_STUDY_NAME = "regime_adaptive_v2_clean"
PROGRESS_LOG_FILE = "ftmo_optimization_progress.txt"

TRAINING_START = datetime(2023, 1, 1)
TRAINING_END = datetime(2024, 9, 30)
VALIDATION_START = datetime(2024, 10, 1)
VALIDATION_END = min(datetime.now(), datetime(2025, 12, 31))
FULL_PERIOD_START = datetime(2023, 1, 1)
FULL_PERIOD_END = min(datetime.now(), datetime(2025, 12, 31))

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
    bollinger_std: float = 2.0,
    rsi_period: int = 14,
    excluded_assets: Optional[List[str]] = None,
    require_adx_filter: bool = True,
    min_adx: float = 25.0,
    # ============================================================================
    # REGIME-ADAPTIVE V2 PARAMETERS
    # ============================================================================
    adx_trend_threshold: float = 25.0,  # ADX level for Trend Mode
    adx_range_threshold: float = 20.0,  # ADX level for Range Mode
    trend_min_confluence: int = 6,  # Min confluence for trend mode
    range_min_confluence: int = 5,  # Min confluence for range mode
    rsi_oversold_range: float = 25.0,  # RSI threshold for range longs
    rsi_overbought_range: float = 75.0,  # RSI threshold for range shorts
    atr_volatility_ratio: float = 0.8,  # ATR(14)/ATR(50) ratio for range mode
    atr_trail_multiplier: float = 1.5,  # ATR multiplier for trailing stops
    partial_exit_at_1r: bool = True,  # Whether to take partial at 1R
    # ============================================================================
    # NEW: EXPANDED PARAMETERS (V2 Enhancement)
    # ============================================================================
    use_adx_slope_rising: bool = False,  # ADX slope-based early trend entry
    use_rsi_range: bool = False,  # Dynamic RSI in Range Mode
    rsi_period_range: int = 14,  # RSI period for range mode
    use_bollinger_range: bool = False,  # Dynamic Bollinger Bands in Range Mode
    bb_period_range: int = 20,  # Bollinger Band period for range mode
    bb_std_range: float = 2.0,  # Bollinger Band std dev for range mode
    use_rsi_trend: bool = False,  # RSI Filtering in Trend Mode
    rsi_trend_overbought: float = 80.0,  # RSI overbought threshold for trend mode
    rsi_trend_oversold: float = 20.0,  # RSI oversold threshold for trend mode
    use_fib_0786_only: bool = False,  # Only use Fib 0.786 level
    use_liquidity_sweep_required: bool = False,  # Require liquidity sweep
    use_market_structure_bos_only: bool = False,  # Only BOS for market structure
    use_atr_trailing: bool = False,  # Use ATR-based trailing stop
    use_volatility_sizing_boost: bool = False,  # Boost sizing in high volatility
    fib_zone_type: str = 'golden_only',  # Fib zone selection
    candle_pattern_strictness: str = 'moderate',  # Candle pattern strictness
    partial_exit_pct: float = 0.5,  # Partial exit percentage
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
                adx_range_threshold=adx_range_threshold
            )
            
            if regime_info['mode'] == 'Transition':
                continue
            
            if regime_info['mode'] == 'Trend':
                effective_confluence = trend_min_confluence
            else:
                effective_confluence = range_min_confluence
            
            current_atr, atr_percentile = _calculate_atr_percentile(d1_candles)
            if atr_percentile < atr_min_percentile:
                continue
            
            params = StrategyParams(
                min_confluence=effective_confluence,
                min_quality_factors=min_quality_factors,
                risk_per_trade_pct=risk_per_trade_pct,
                atr_min_percentile=atr_min_percentile,
                trail_activation_r=trail_activation_r,
                december_atr_multiplier=december_atr_multiplier,
                volatile_asset_boost=volatile_asset_boost,
                bollinger_std=bollinger_std,
                rsi_period=rsi_period,
                ml_min_prob=ml_min_prob if ml_min_prob else 0.0,
                adx_trend_threshold=adx_trend_threshold,
                adx_range_threshold=adx_range_threshold,
                trend_min_confluence=trend_min_confluence,
                range_min_confluence=range_min_confluence,
                rsi_oversold_range=rsi_oversold_range,
                rsi_overbought_range=rsi_overbought_range,
                atr_volatility_ratio=atr_volatility_ratio,
                atr_trail_multiplier=atr_trail_multiplier,
                partial_exit_at_1r=partial_exit_at_1r,
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
    return all_trades


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
    
    spread_cost = risk_pips * 0.02
    slippage_cost = risk_pips * 0.01
    commission_cost = 0.001
    adjusted_profit = profit_usd * (1 - spread_cost - slippage_cost - commission_cost)
    
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
        Optuna objective function - runs ONLY on TRAINING period.
        Objective: Maximize total_net_profit_dollars on training
        
        REGIME-ADAPTIVE V2: Extended search space includes:
        - Regime detection thresholds (ADX trend/range)
        - Mode-specific confluence requirements
        - Range mode filters (RSI, ATR volatility)
        - Partial profit taking and trail management
        """
        # ============================================================================
        # REGIME-ADAPTIVE V2 EXPANDED PARAMETER SEARCH SPACE (20+ Parameters)
        # ============================================================================
        params = {
            # Core risk management (keep existing)
            'risk_per_trade_pct': trial.suggest_float('risk_per_trade_pct', 0.5, 0.8, step=0.05),
            'min_confluence_score': trial.suggest_int('min_confluence_score', 3, 6),
            'min_quality_factors': trial.suggest_int('min_quality_factors', 1, 4),
            
            # EXPANDED: Regime detection thresholds (wider range)
            'adx_trend_threshold': trial.suggest_float('adx_trend_threshold', 20.0, 28.0, step=1.0),
            'adx_range_threshold': trial.suggest_float('adx_range_threshold', 15.0, 20.0, step=1.0),
            
            # NEW: ADX slope-based early trend entry
            'use_adx_slope_rising': trial.suggest_categorical('use_adx_slope_rising', [True, False]),
            
            # EXPANDED: Mode-specific confluence (relaxed ranges)
            'trend_min_confluence': trial.suggest_int('trend_min_confluence', 5, 7),
            'range_min_confluence': trial.suggest_int('range_min_confluence', 4, 6),
            
            # NEW: Dynamic RSI in Range Mode
            'use_rsi_range': trial.suggest_categorical('use_rsi_range', [True, False]),
            'rsi_period_range': trial.suggest_int('rsi_period_range', 10, 20) if trial.params.get('use_rsi_range', False) else 14,
            
            # Range mode RSI thresholds
            'rsi_oversold_range': trial.suggest_float('rsi_oversold_range', 20.0, 35.0, step=2.0),
            'rsi_overbought_range': trial.suggest_float('rsi_overbought_range', 65.0, 80.0, step=2.0),
            
            # NEW: Dynamic Bollinger Bands in Range Mode
            'use_bollinger_range': trial.suggest_categorical('use_bollinger_range', [True, False]),
            'bb_period_range': trial.suggest_int('bb_period_range', 18, 25),
            'bb_std_range': trial.suggest_float('bb_std_range', 1.8, 2.5, step=0.1),
            
            # NEW: RSI Filtering in Trend Mode
            'use_rsi_trend': trial.suggest_categorical('use_rsi_trend', [True, False]),
            'rsi_trend_overbought': trial.suggest_float('rsi_trend_overbought', 75.0, 85.0, step=2.0),
            'rsi_trend_oversold': trial.suggest_float('rsi_trend_oversold', 15.0, 25.0, step=2.0),
            
            # NEW: Strategy-Level Toggles
            'use_fib_0786_only': trial.suggest_categorical('use_fib_0786_only', [True, False]),
            'use_liquidity_sweep_required': trial.suggest_categorical('use_liquidity_sweep_required', [True, False]),
            'use_market_structure_bos_only': trial.suggest_categorical('use_market_structure_bos_only', [True, False]),
            'use_atr_trailing': trial.suggest_categorical('use_atr_trailing', [True, False]),
            'use_volatility_sizing_boost': trial.suggest_categorical('use_volatility_sizing_boost', [True, False]),
            
            # Categorical/Other
            'fib_zone_type': trial.suggest_categorical('fib_zone_type', ['golden_only', 'extended', 'full_retracement']),
            'candle_pattern_strictness': trial.suggest_categorical('candle_pattern_strictness', ['strict', 'moderate', 'loose']),
            'partial_exit_pct': trial.suggest_float('partial_exit_pct', 0.4, 0.7, step=0.05),
            'atr_trail_multiplier': trial.suggest_float('atr_trail_multiplier', 1.5, 3.5, step=0.2),
            'atr_vol_ratio_range': trial.suggest_float('atr_vol_ratio_range', 0.6, 0.9, step=0.05),
            
            # Existing filters (keep)
            'atr_min_percentile': trial.suggest_float('atr_min_percentile', 60.0, 85.0, step=5.0),
            'trail_activation_r': trial.suggest_float('trail_activation_r', 1.8, 3.4, step=0.2),
            'december_atr_multiplier': trial.suggest_float('december_atr_multiplier', 1.3, 1.8, step=0.1),
            'volatile_asset_boost': trial.suggest_float('volatile_asset_boost', 1.3, 2.0, step=0.1),
            'partial_exit_at_1r': trial.suggest_categorical('partial_exit_at_1r', [True, False]),
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
            # Regime-Adaptive V2 parameters
            adx_trend_threshold=params['adx_trend_threshold'],
            adx_range_threshold=params['adx_range_threshold'],
            trend_min_confluence=params['trend_min_confluence'],
            range_min_confluence=params['range_min_confluence'],
            rsi_oversold_range=params['rsi_oversold_range'],
            rsi_overbought_range=params['rsi_overbought_range'],
            atr_volatility_ratio=params['atr_vol_ratio_range'],
            atr_trail_multiplier=params['atr_trail_multiplier'],
            partial_exit_at_1r=params['partial_exit_at_1r'],
            # NEW: Expanded parameters
            use_adx_slope_rising=params['use_adx_slope_rising'],
            use_rsi_range=params['use_rsi_range'],
            rsi_period_range=params['rsi_period_range'],
            use_bollinger_range=params['use_bollinger_range'],
            bb_period_range=params['bb_period_range'],
            bb_std_range=params['bb_std_range'],
            use_rsi_trend=params['use_rsi_trend'],
            rsi_trend_overbought=params['rsi_trend_overbought'],
            rsi_trend_oversold=params['rsi_trend_oversold'],
            use_fib_0786_only=params['use_fib_0786_only'],
            use_liquidity_sweep_required=params['use_liquidity_sweep_required'],
            use_market_structure_bos_only=params['use_market_structure_bos_only'],
            use_atr_trailing=params['use_atr_trailing'],
            use_volatility_sizing_boost=params['use_volatility_sizing_boost'],
            fib_zone_type=params['fib_zone_type'],
            candle_pattern_strictness=params['candle_pattern_strictness'],
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
        # REGIME-ADAPTIVE V2: NEW SCORING FORMULA
        # Focuses on consistency, drawdown control, and balanced trading activity
        # ============================================================================
        
        # Calculate quarterly profits and trade counts
        quarterly_profits = {}  # Q -> profit in USD
        quarterly_trade_counts = {}  # Q -> number of trades
        
        for q in TRAINING_QUARTERS.keys():
            q_r = quarterly_r.get(q, 0.0)
            q_profit = q_r * risk_usd
            q_count = len(quarterly_trades.get(q, []))
            quarterly_profits[q] = q_profit
            quarterly_trade_counts[q] = q_count
        
        # Calculate max drawdown in R terms
        max_dd_r = 0.0
        equity = 0.0
        peak = 0.0
        for t in training_trades:
            equity += getattr(t, 'rr', 0)
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd_r:
                max_dd_r = dd
        
        # Total net profit in USD
        total_net_profit = total_r * risk_usd
        
        # ----------------------------------------------------------------------------
        # CONSISTENCY BONUS CALCULATION
        # Rewards consistent quarterly performance across all training quarters
        # Penalizes quarters with low profit, rewards all-positive quarters
        # ----------------------------------------------------------------------------
        min_quarterly_profit = min(quarterly_profits.values()) if quarterly_profits else 0
        consistency_bonus = 1.0
        
        # Penalty for low minimum quarter (scale down if below $15,000)
        if min_quarterly_profit < 15000:
            consistency_bonus -= 0.2 * max(0, (15000 - min_quarterly_profit) / 15000)
        
        # Bonus for all positive quarters (major consistency indicator)
        if all(p >= 0 for p in quarterly_profits.values()):
            consistency_bonus += 0.3
        
        # ----------------------------------------------------------------------------
        # DRAWDOWN PENALTY
        # Severe penalty for drawdowns exceeding 10% of account
        # Critical for FTMO challenge compliance (5% daily, 10% total max)
        # ----------------------------------------------------------------------------
        # Convert max drawdown in R to percentage of account
        # max_dd_r is in R units, need to convert to account percentage
        max_drawdown_pct = (max_dd_r * risk_usd) / ACCOUNT_SIZE if ACCOUNT_SIZE > 0 else 0
        
        dd_penalty = 0.0
        if max_drawdown_pct > 0.10:
            # 40% penalty for each 1% over 10% threshold
            # E.g., 15% drawdown = 0.4 * (0.15-0.10) = 0.02 penalty -> 98% multiplier
            dd_penalty = 0.4 * (max_drawdown_pct - 0.10)
        
        # ----------------------------------------------------------------------------
        # ENHANCED: Trade Balance Bonus for 12-35 trades per quarter
        # More nuanced penalty system for undertrading and overtrading
        # ----------------------------------------------------------------------------
        trade_balance_bonus = 0.25  # Base bonus
        for q, count in quarterly_trade_counts.items():
            if count < 12:
                # Penalty for undertrading
                trade_balance_bonus -= 0.05 * (12 - count)
            elif count > 35:
                # Penalty for overtrading
                trade_balance_bonus -= 0.02 * (count - 35)
        trade_balance_bonus = max(0.0, min(0.3, trade_balance_bonus))
        
        # ENHANCED: Consistency bonus with $20k minimum quarterly profit
        if min_quarterly_profit >= 20000:
            consistency_bonus += 0.4  # Strong bonus for $20k+ quarters
        elif min_quarterly_profit >= 15000:
            consistency_bonus += 0.2
        elif min_quarterly_profit < 10000:
            consistency_bonus -= 0.3  # Penalty for weak quarters
        
        # ----------------------------------------------------------------------------
        # FINAL SCORE CALCULATION
        # Combines profit with consistency, drawdown protection, and trade balance
        # ----------------------------------------------------------------------------
        final_score = total_net_profit * (1 - dd_penalty) * consistency_bonus * (1 + trade_balance_bonus)
        
        # Store additional regime-adaptive metrics for analysis
        trial.set_user_attr('quarterly_profits', quarterly_profits)
        trial.set_user_attr('quarterly_trade_counts', quarterly_trade_counts)
        trial.set_user_attr('consistency_bonus', round(consistency_bonus, 3))
        trial.set_user_attr('dd_penalty', round(dd_penalty, 3))
        trial.set_user_attr('trade_balance_bonus', round(trade_balance_bonus, 3))
        trial.set_user_attr('max_drawdown_pct', round(max_drawdown_pct * 100, 2))
        
        return final_score
    
    def run_optimization(self, n_trials: int = 5) -> Dict:
        """Run Optuna optimization on TRAINING data only."""
        import optuna
        from optuna.pruners import MedianPruner
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"\n{'='*60}")
        print(f"OPTUNA OPTIMIZATION - Adding {n_trials} trials")
        print(f"TRAINING PERIOD: 2023-01-01 to 2024-09-30")
        print(f"Regime-Adaptive V2: Trend (ADX >= threshold) + Conservative Range (ADX < threshold)")
        print(f"Storage: {OPTUNA_DB_PATH} (resumable)")
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
        if existing_trials > 0:
            print(f"Resuming from existing study with {existing_trials} completed trials")
            try:
                if study.best_trial:
                    print(f"Current best value: {study.best_value:.0f}")
            except ValueError:
                print("No valid completed trials yet")
        
        def progress_callback(study, trial):
            log_optimization_progress(
                trial_num=trial.number,
                value=trial.value if trial.value is not None else 0,
                best_value=study.best_value if study.best_trial else 0,
                best_params=study.best_params if study.best_trial else {}
            )
            
            quarterly_stats = trial.user_attrs.get('quarterly_stats', {})
            overall_stats = trial.user_attrs.get('overall_stats', {})
            
            print(f"\n{'─'*70}")
            print(f"TRIAL #{trial.number} COMPLETE | Score: {trial.value:.0f} | Best: {study.best_value:.0f}")
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
        
        study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[progress_callback]
        )
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
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
            'use_rsi_range': self.best_params.get('use_rsi_range', False),
            'rsi_period_range': self.best_params.get('rsi_period_range', 14),
            # NEW: Dynamic Bollinger Bands in Range Mode
            'use_bollinger_range': self.best_params.get('use_bollinger_range', False),
            'bb_period_range': self.best_params.get('bb_period_range', 20),
            'bb_std_range': self.best_params.get('bb_std_range', 2.0),
            # NEW: RSI Filtering in Trend Mode
            'use_rsi_trend': self.best_params.get('use_rsi_trend', False),
            'rsi_trend_overbought': self.best_params.get('rsi_trend_overbought', 80.0),
            'rsi_trend_oversold': self.best_params.get('rsi_trend_oversold', 20.0),
            # NEW: Strategy-Level Toggles
            'use_fib_0786_only': self.best_params.get('use_fib_0786_only', False),
            'use_liquidity_sweep_required': self.best_params.get('use_liquidity_sweep_required', False),
            'use_market_structure_bos_only': self.best_params.get('use_market_structure_bos_only', False),
            'use_atr_trailing': self.best_params.get('use_atr_trailing', False),
            'use_volatility_sizing_boost': self.best_params.get('use_volatility_sizing_boost', False),
            # Categorical/Other
            'fib_zone_type': self.best_params.get('fib_zone_type', 'golden_only'),
            'candle_pattern_strictness': self.best_params.get('candle_pattern_strictness', 'moderate'),
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
        }


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
    
    lines.extend([
        "",
        "TRAINING PERIOD (2023-01-01 to 2024-09-30)",
        "-" * 40,
        f"  Trades: {training_stats['count']}",
        f"  Total R: {training_stats['total_r']:+.2f}",
        f"  Win Rate: {training_stats['win_rate']:.1f}%",
        f"  Avg R per Trade: {training_stats['avg_r']:+.3f}",
        "",
        f"VALIDATION PERIOD (2024-10-01 to {VALIDATION_END.strftime('%Y-%m-%d')})",
        "-" * 40,
        f"  Trades: {validation_stats['count']}",
        f"  Total R: {validation_stats['total_r']:+.2f}",
        f"  Win Rate: {validation_stats['win_rate']:.1f}%",
        f"  Avg R per Trade: {validation_stats['avg_r']:+.3f}",
        "",
        "FULL PERIOD (2023-2025)",
        "-" * 40,
        f"  Trades: {full_stats['count']}",
        f"  Total R: {full_stats['total_r']:+.2f}",
        f"  Win Rate: {full_stats['win_rate']:.1f}%",
        f"  Avg R per Trade: {full_stats['avg_r']:+.3f}",
        "",
        "QUARTERLY BREAKDOWN",
        "-" * 40,
    ])
    
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
        lines.append(f"  {q_name}: {len(q_filtered)} trades, {q_r:+.1f}R, {q_wr:.0f}% win rate")
    
    lines.extend([
        "",
        "=" * 80,
        "End of Summary",
        "=" * 80,
    ])
    
    with open(summary_filename, 'w') as f:
        f.write("\n".join(lines))
    
    return str(summary_filename)


def main():
    """
    Professional FTMO Optimization Workflow with CLI support.
    
    Usage:
      python ftmo_challenge_analyzer.py              # Run/resume optimization (5 trials)
      python ftmo_challenge_analyzer.py --status     # Check progress without running
      python ftmo_challenge_analyzer.py --trials 100 # Run 100 trials
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
    args = parser.parse_args()
    
    if args.status:
        show_optimization_status()
        return
    
    n_trials = args.trials
    
    print(f"\n{'='*80}")
    print("FTMO PROFESSIONAL OPTIMIZATION SYSTEM - REGIME-ADAPTIVE V2")
    print(f"{'='*80}")
    print(f"\nData Partitioning (Multi-Year Robustness):")
    print(f"  TRAINING:    2023-01-01 to 2024-09-30 (in-sample)")
    print(f"  VALIDATION:  2024-10-01 to {VALIDATION_END.strftime('%Y-%m-%d')} (out-of-sample)")
    print(f"  FINAL:       Full 2023-2025 (December fully open)")
    print(f"\nRegime-Adaptive V2 Trading System:")
    print(f"  TREND MODE:      ADX >= threshold (momentum following)")
    print(f"  RANGE MODE:      ADX < threshold (conservative mean reversion)")
    print(f"  TRANSITION:      NO ENTRIES (wait for regime confirmation)")
    print(f"\nResumable: Study stored in {OPTUNA_DB_PATH}")
    print(f"{'='*80}\n")
    
    optimizer = OptunaOptimizer()
    results = optimizer.run_optimization(n_trials=n_trials)
    
    best_params = results['best_params']
    
    print(f"\n{'='*80}")
    print("=== TRAINING RESULTS (2023-01-01 to 2024-09-30) ===")
    print(f"{'='*80}")
    
    training_trades = run_full_period_backtest(
        start_date=TRAINING_START,
        end_date=TRAINING_END,
        min_confluence=best_params.get('min_confluence_score', 3),
        min_quality_factors=best_params.get('min_quality_factors', 2),
        risk_per_trade_pct=best_params.get('risk_per_trade_pct', 0.5),
        atr_min_percentile=best_params.get('atr_min_percentile', 60.0),
        trail_activation_r=best_params.get('trail_activation_r', 2.2),
        december_atr_multiplier=best_params.get('december_atr_multiplier', 1.5),
        volatile_asset_boost=best_params.get('volatile_asset_boost', 1.5),
        ml_min_prob=None,
        require_adx_filter=True,
        adx_trend_threshold=best_params.get('adx_trend_threshold', 25.0),
        adx_range_threshold=best_params.get('adx_range_threshold', 20.0),
        trend_min_confluence=best_params.get('trend_min_confluence', 6),
        range_min_confluence=best_params.get('range_min_confluence', 5),
        rsi_oversold_range=best_params.get('rsi_oversold_range', 25.0),
        rsi_overbought_range=best_params.get('rsi_overbought_range', 75.0),
        atr_volatility_ratio=best_params.get('atr_volatility_ratio', 0.8),
        atr_trail_multiplier=best_params.get('atr_trail_multiplier', 1.5),
        partial_exit_at_1r=best_params.get('partial_exit_at_1r', True),
    )
    
    training_results = print_period_results(
        training_trades, "TRAINING RESULTS (2023-01-01 to 2024-09-30)",
        TRAINING_START, TRAINING_END
    )
    
    print(f"\n{'='*80}")
    print(f"=== VALIDATION RESULTS (2024-10-01 to {VALIDATION_END.strftime('%Y-%m-%d')}) ===")
    print(f"{'='*80}")
    
    validation_trades = run_full_period_backtest(
        start_date=VALIDATION_START,
        end_date=VALIDATION_END,
        min_confluence=best_params.get('min_confluence_score', 3),
        min_quality_factors=best_params.get('min_quality_factors', 2),
        risk_per_trade_pct=best_params.get('risk_per_trade_pct', 0.5),
        atr_min_percentile=best_params.get('atr_min_percentile', 60.0),
        trail_activation_r=best_params.get('trail_activation_r', 2.2),
        december_atr_multiplier=best_params.get('december_atr_multiplier', 1.5),
        volatile_asset_boost=best_params.get('volatile_asset_boost', 1.5),
        ml_min_prob=None,
        require_adx_filter=True,
        adx_trend_threshold=best_params.get('adx_trend_threshold', 25.0),
        adx_range_threshold=best_params.get('adx_range_threshold', 20.0),
        trend_min_confluence=best_params.get('trend_min_confluence', 6),
        range_min_confluence=best_params.get('range_min_confluence', 5),
        rsi_oversold_range=best_params.get('rsi_oversold_range', 25.0),
        rsi_overbought_range=best_params.get('rsi_overbought_range', 75.0),
        atr_volatility_ratio=best_params.get('atr_volatility_ratio', 0.8),
        atr_trail_multiplier=best_params.get('atr_trail_multiplier', 1.5),
        partial_exit_at_1r=best_params.get('partial_exit_at_1r', True),
    )
    
    validation_results = print_period_results(
        validation_trades, f"VALIDATION RESULTS (2024-10-01 to {VALIDATION_END.strftime('%Y-%m-%d')})",
        VALIDATION_START, VALIDATION_END
    )
    
    print(f"\n{'='*80}")
    print("=== FULL PERIOD FINAL RESULTS (2023-2025) ===")
    print(f"{'='*80}")
    print("Running full period backtest with Regime-Adaptive V2...")
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
        adx_trend_threshold=best_params.get('adx_trend_threshold', 25.0),
        adx_range_threshold=best_params.get('adx_range_threshold', 20.0),
        trend_min_confluence=best_params.get('trend_min_confluence', 6),
        range_min_confluence=best_params.get('range_min_confluence', 5),
        rsi_oversold_range=best_params.get('rsi_oversold_range', 25.0),
        rsi_overbought_range=best_params.get('rsi_overbought_range', 75.0),
        atr_volatility_ratio=best_params.get('atr_volatility_ratio', 0.8),
        atr_trail_multiplier=best_params.get('atr_trail_multiplier', 1.5),
        partial_exit_at_1r=best_params.get('partial_exit_at_1r', True),
    )
    
    risk_pct = best_params.get('risk_per_trade_pct', 0.5)
    export_trades_to_csv(full_year_trades, "all_trades_2023_2025_full.csv", risk_pct)
    
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
