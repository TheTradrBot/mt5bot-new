"""
Support/Resistance Level Detection for FTMO Trading Bot

Detects strong historical S/R levels on WEEKLY and MONTHLY timeframes using:
- Swing high/low detection via scipy.signal.find_peaks
- Clustering similar levels within tolerance
- Touch counting with reversal confirmation
- Strength scoring based on touch count and recency

Matches the horizontal S/R lines visible in TradingView charts.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import fclusterdata

DATA_DIR = Path("data/ohlcv")
SR_OUTPUT_DIR = Path("data/sr_levels")
PLOT_DIR = Path("data/plots")

SR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ALL_ASSETS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
    "XAUUSD", "XAGUSD",
    "BTCUSD", "ETHUSD",
    "SPX500USD", "NAS100USD"
]

ASSET_CONFIG = {
    "EURUSD": {"tolerance": 0.0008, "decimals": 5},
    "GBPUSD": {"tolerance": 0.0010, "decimals": 5},
    "USDJPY": {"tolerance": 0.08, "decimals": 3},
    "USDCHF": {"tolerance": 0.0008, "decimals": 5},
    "USDCAD": {"tolerance": 0.0008, "decimals": 5},
    "AUDUSD": {"tolerance": 0.0006, "decimals": 5},
    "NZDUSD": {"tolerance": 0.0006, "decimals": 5},
    "EURGBP": {"tolerance": 0.0006, "decimals": 5},
    "EURJPY": {"tolerance": 0.10, "decimals": 3},
    "EURCHF": {"tolerance": 0.0008, "decimals": 5},
    "EURAUD": {"tolerance": 0.0012, "decimals": 5},
    "EURCAD": {"tolerance": 0.0010, "decimals": 5},
    "EURNZD": {"tolerance": 0.0012, "decimals": 5},
    "GBPJPY": {"tolerance": 0.12, "decimals": 3},
    "GBPCHF": {"tolerance": 0.0010, "decimals": 5},
    "GBPAUD": {"tolerance": 0.0015, "decimals": 5},
    "GBPCAD": {"tolerance": 0.0012, "decimals": 5},
    "GBPNZD": {"tolerance": 0.0015, "decimals": 5},
    "AUDJPY": {"tolerance": 0.08, "decimals": 3},
    "AUDCHF": {"tolerance": 0.0006, "decimals": 5},
    "AUDCAD": {"tolerance": 0.0008, "decimals": 5},
    "AUDNZD": {"tolerance": 0.0008, "decimals": 5},
    "NZDJPY": {"tolerance": 0.08, "decimals": 3},
    "NZDCHF": {"tolerance": 0.0006, "decimals": 5},
    "NZDCAD": {"tolerance": 0.0006, "decimals": 5},
    "CADJPY": {"tolerance": 0.08, "decimals": 3},
    "CADCHF": {"tolerance": 0.0006, "decimals": 5},
    "CHFJPY": {"tolerance": 0.10, "decimals": 3},
    "XAUUSD": {"tolerance": 5.0, "decimals": 2},
    "XAGUSD": {"tolerance": 0.10, "decimals": 3},
    "BTCUSD": {"tolerance": 500.0, "decimals": 0},
    "ETHUSD": {"tolerance": 25.0, "decimals": 2},
    "SPX500USD": {"tolerance": 25.0, "decimals": 1},
    "NAS100USD": {"tolerance": 50.0, "decimals": 1},
}


def get_asset_config(symbol: str) -> dict:
    """Get tolerance and decimal config for an asset."""
    clean = symbol.upper().replace("_", "")
    if clean in ASSET_CONFIG:
        return ASSET_CONFIG[clean]
    avg_price = 1.0
    return {"tolerance": avg_price * 0.005, "decimals": 5}


def load_ohlcv(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Load OHLCV data from CSV file.
    
    Args:
        symbol: e.g., "EURUSD" or "EUR_USD"
        timeframe: "W1" for weekly, "MN" for monthly
    
    Returns:
        DataFrame with OHLCV data or None
    """
    clean_symbol = symbol.upper().replace("_", "")
    
    patterns = [
        f"{clean_symbol}_{timeframe}_*.csv",
        f"{clean_symbol}_{timeframe}.csv",
    ]
    
    for pattern in patterns:
        matches = list(DATA_DIR.glob(pattern))
        if matches:
            matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            try:
                df = pd.read_csv(matches[0], parse_dates=['time'])
                df.set_index('time', inplace=True)
                df.columns = [c.capitalize() for c in df.columns]
                if 'High' not in df.columns:
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                return df
            except Exception as e:
                print(f"  Error loading {matches[0]}: {e}")
                continue
    
    return None


def detect_swing_highs(highs: np.ndarray, distance: int = 5, prominence: float = None) -> np.ndarray:
    """
    Detect swing high indices using scipy.signal.find_peaks.
    
    Args:
        highs: Array of high prices
        distance: Minimum distance between peaks (candles)
        prominence: Minimum prominence for peaks (auto-calculated if None)
    
    Returns:
        Array of indices where swing highs occur
    """
    if prominence is None:
        atr = np.mean(np.abs(np.diff(highs)))
        prominence = atr * 0.5
    
    peaks, properties = find_peaks(highs, distance=distance, prominence=prominence)
    return peaks


def detect_swing_lows(lows: np.ndarray, distance: int = 5, prominence: float = None) -> np.ndarray:
    """
    Detect swing low indices using scipy.signal.find_peaks on inverted data.
    
    Args:
        lows: Array of low prices
        distance: Minimum distance between valleys (candles)
        prominence: Minimum prominence for valleys
    
    Returns:
        Array of indices where swing lows occur
    """
    if prominence is None:
        atr = np.mean(np.abs(np.diff(lows)))
        prominence = atr * 0.5
    
    inverted = -lows
    peaks, properties = find_peaks(inverted, distance=distance, prominence=prominence)
    return peaks


def cluster_levels(levels: List[float], tolerance: float) -> Dict[float, List[float]]:
    """
    Cluster similar price levels using hierarchical clustering.
    
    Args:
        levels: List of price levels
        tolerance: Maximum distance within a cluster
    
    Returns:
        Dict mapping cluster center to list of member levels
    """
    if len(levels) < 2:
        if len(levels) == 1:
            return {levels[0]: levels}
        return {}
    
    levels_arr = np.array(levels).reshape(-1, 1)
    
    try:
        clusters = fclusterdata(levels_arr, t=tolerance, criterion='distance', method='complete')
    except Exception:
        return {levels[0]: levels}
    
    cluster_dict = {}
    for level, cluster_id in zip(levels, clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(level)
    
    result = {}
    for members in cluster_dict.values():
        center = np.mean(members)
        result[center] = members
    
    return result


def count_touches_with_reversal(
    df: pd.DataFrame,
    level: float,
    tolerance: float,
    min_reversal_factor: float = 0.5
) -> List[Dict]:
    """
    Count how many times price touched a level and reversed.
    
    A touch is confirmed when:
    - For support: Low comes within tolerance of level AND price moves up afterward
    - For resistance: High comes within tolerance of level AND price moves down afterward
    
    Each candle can only contribute ONE touch (either support OR resistance, not both).
    Priority given to the touch type with stronger reversal.
    
    Args:
        df: OHLCV DataFrame
        level: The S/R level to check
        tolerance: Distance tolerance for touch detection
        min_reversal_factor: Minimum reversal as factor of tolerance
    
    Returns:
        List of touch info dicts with timestamp, type, and price
    """
    touches = []
    highs = df['High'].values
    lows = df['Low'].values
    times = df.index.tolist()
    
    atr = np.mean(highs - lows) if len(df) > 0 else tolerance * 2
    min_reversal = max(tolerance, atr * min_reversal_factor)
    
    for i in range(len(df) - 1):
        support_touch = None
        resistance_touch = None
        
        if abs(lows[i] - level) <= tolerance:
            future_move = 0
            for j in range(i + 1, min(i + 4, len(df))):
                future_move = max(future_move, highs[j] - lows[i])
            
            if future_move >= min_reversal:
                support_touch = {
                    "timestamp": str(times[i]),
                    "type": "support",
                    "touch_price": float(lows[i]),
                    "reversal_size": float(future_move)
                }
        
        if abs(highs[i] - level) <= tolerance:
            future_move = 0
            for j in range(i + 1, min(i + 4, len(df))):
                future_move = max(future_move, highs[i] - lows[j])
            
            if future_move >= min_reversal:
                resistance_touch = {
                    "timestamp": str(times[i]),
                    "type": "resistance",
                    "touch_price": float(highs[i]),
                    "reversal_size": float(future_move)
                }
        
        if support_touch and resistance_touch:
            if support_touch["reversal_size"] >= resistance_touch["reversal_size"]:
                touches.append(support_touch)
            else:
                touches.append(resistance_touch)
        elif support_touch:
            touches.append(support_touch)
        elif resistance_touch:
            touches.append(resistance_touch)
    
    return touches


def calculate_strength_score(touches: List[Dict], decay_factor: float = 0.1) -> float:
    """
    Calculate strength score based on touch count and recency.
    
    Score = touch_count * (1 - decay for age)
    More recent touches contribute more to the score.
    
    Args:
        touches: List of touch dicts with timestamps
        decay_factor: How much to discount older touches
    
    Returns:
        Strength score as float
    """
    if not touches:
        return 0.0
    
    now = datetime.now()
    score = 0.0
    
    for touch in touches:
        try:
            touch_time = pd.to_datetime(touch["timestamp"])
            years_ago = (now - touch_time).days / 365.25
            recency_weight = max(0.2, 1.0 - (decay_factor * years_ago))
            score += recency_weight
        except:
            score += 0.5
    
    return round(score, 2)


def classify_sr_type(touches: List[Dict]) -> str:
    """
    Classify level as Support, Resistance, or S/R based on touch types.
    
    Args:
        touches: List of touch dicts
    
    Returns:
        "Support", "Resistance", or "S/R"
    """
    if not touches:
        return "S/R"
    
    support_count = sum(1 for t in touches if t["type"] == "support")
    resistance_count = sum(1 for t in touches if t["type"] == "resistance")
    
    if support_count >= resistance_count * 2:
        return "Support"
    elif resistance_count >= support_count * 2:
        return "Resistance"
    else:
        return "S/R"


def detect_sr_levels(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    min_touches: int = 2
) -> List[Dict]:
    """
    Main S/R detection function.
    
    1. Detect swing highs and lows using find_peaks
    2. Cluster similar levels
    3. Count touches with reversal confirmation
    4. Return top levels sorted by strength
    
    Args:
        df: OHLCV DataFrame
        symbol: Asset symbol
        timeframe: "W1" or "MN"
        min_touches: Minimum touches to confirm level
    
    Returns:
        List of S/R level dicts
    """
    if df is None or len(df) < 20:
        return []
    
    config = get_asset_config(symbol)
    tolerance = config["tolerance"]
    decimals = config["decimals"]
    
    highs = df['High'].values
    lows = df['Low'].values
    
    if timeframe == "MN":
        peak_distance = 3
        prominence_factor = 0.3
    else:
        peak_distance = 5
        prominence_factor = 0.2
    
    atr = np.mean(highs - lows)
    prominence = atr * prominence_factor
    
    swing_high_idx = detect_swing_highs(highs, distance=peak_distance, prominence=prominence)
    swing_low_idx = detect_swing_lows(lows, distance=peak_distance, prominence=prominence)
    
    swing_high_prices = highs[swing_high_idx].tolist()
    swing_low_prices = lows[swing_low_idx].tolist()
    
    all_swing_levels = swing_high_prices + swing_low_prices
    
    if len(all_swing_levels) == 0:
        return []
    
    clustered = cluster_levels(all_swing_levels, tolerance * 2)
    
    sr_levels = []
    
    for cluster_center, members in clustered.items():
        avg_level = round(np.mean(members), decimals)
        
        touches = count_touches_with_reversal(df, avg_level, tolerance)
        
        if len(touches) >= min_touches:
            touch_dates = [t["timestamp"] for t in touches]
            
            sr_levels.append({
                "level": avg_level,
                "type": classify_sr_type(touches),
                "touch_count": len(touches),
                "touch_dates": touch_dates,
                "strength_score": calculate_strength_score(touches),
                "first_touch": min(touch_dates) if touch_dates else None,
                "last_touch": max(touch_dates) if touch_dates else None
            })
    
    sr_levels.sort(key=lambda x: (-x["touch_count"], -x["strength_score"]))
    
    return sr_levels[:10]


def save_sr_levels(levels: List[Dict], symbol: str, timeframe: str):
    """Save S/R levels to JSON file."""
    filename = SR_OUTPUT_DIR / f"{symbol}_{timeframe}_sr.json"
    with open(filename, 'w') as f:
        json.dump(levels, f, indent=2)
    return filename


def load_sr_levels(symbol: str, timeframe: str) -> List[Dict]:
    """
    Load S/R levels from JSON file.
    
    This function can be imported and used by strategy_core.py
    to check for S/R confluence.
    
    Args:
        symbol: Asset symbol (e.g., "EURUSD")
        timeframe: "W1" for weekly, "MN" for monthly
    
    Returns:
        List of S/R level dicts
    """
    clean_symbol = symbol.upper().replace("_", "")
    filename = SR_OUTPUT_DIR / f"{clean_symbol}_{timeframe}_sr.json"
    
    if not filename.exists():
        return []
    
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def is_price_near_sr(
    price: float,
    symbol: str,
    tolerance_pct: float = 0.005
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if price is near any S/R level.
    
    Args:
        price: Current price
        symbol: Asset symbol
        tolerance_pct: Percentage tolerance (default 0.5%)
    
    Returns:
        Tuple of (is_near_sr, closest_level_info)
    """
    monthly_levels = load_sr_levels(symbol, "MN")
    weekly_levels = load_sr_levels(symbol, "W1")
    
    all_levels = monthly_levels + weekly_levels
    
    for sr in all_levels:
        level = sr["level"]
        tolerance = level * tolerance_pct
        if abs(price - level) <= tolerance:
            return True, sr
    
    return False, None


def print_results_table(levels: List[Dict], symbol: str, timeframe: str):
    """Print results in a clean table format."""
    tf_name = "Monthly" if timeframe == "MN" else "Weekly"
    print(f"\n{'='*70}")
    print(f"  {symbol} {tf_name} S/R Levels")
    print(f"{'='*70}")
    
    if not levels:
        print("  No significant S/R levels detected")
        return
    
    print(f"  {'Level':<12} {'Type':<12} {'Touches':<8} {'Strength':<10} {'Last Touch':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*10} {'-'*12}")
    
    for sr in levels:
        level = sr["level"]
        sr_type = sr["type"]
        touches = sr["touch_count"]
        strength = sr["strength_score"]
        last_touch = sr["last_touch"][:10] if sr["last_touch"] else "N/A"
        
        print(f"  {level:<12.5f} {sr_type:<12} {touches:<8} {strength:<10.2f} {last_touch:<12}")


def process_asset(symbol: str, mode: str = "historical") -> Dict[str, List[Dict]]:
    """
    Process a single asset for S/R detection.
    
    Args:
        symbol: Asset symbol
        mode: "historical" or "update"
    
    Returns:
        Dict with "weekly" and "monthly" S/R levels
    """
    results = {"weekly": [], "monthly": []}
    
    for timeframe in ["W1", "MN"]:
        df = load_ohlcv(symbol, timeframe)
        
        if df is None:
            print(f"  No data found for {symbol} {timeframe}")
            continue
        
        levels = detect_sr_levels(df, symbol, timeframe)
        
        if levels:
            save_sr_levels(levels, symbol, timeframe)
            print(f"  {symbol} {timeframe}: Found {len(levels)} S/R levels")
        else:
            print(f"  {symbol} {timeframe}: No significant levels found")
        
        key = "monthly" if timeframe == "MN" else "weekly"
        results[key] = levels
    
    return results


def process_all_assets(mode: str = "historical"):
    """Process all 34 assets for S/R detection."""
    print(f"\n{'='*70}")
    print(f"  S/R Level Detection for All Assets ({mode} mode)")
    print(f"{'='*70}")
    
    all_results = {}
    
    for i, symbol in enumerate(ALL_ASSETS, 1):
        print(f"\n[{i}/{len(ALL_ASSETS)}] Processing {symbol}...")
        results = process_asset(symbol, mode)
        all_results[symbol] = results
    
    summary_file = SR_OUTPUT_DIR / "all_sr_summary.json"
    summary = {}
    for symbol, data in all_results.items():
        summary[symbol] = {
            "weekly_count": len(data.get("weekly", [])),
            "monthly_count": len(data.get("monthly", [])),
            "weekly_levels": [l["level"] for l in data.get("weekly", [])[:5]],
            "monthly_levels": [l["level"] for l in data.get("monthly", [])[:5]]
        }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Processed: {len(all_results)} assets")
    print(f"  Output directory: {SR_OUTPUT_DIR}")
    print(f"  Summary file: {summary_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Detect S/R levels from historical OHLCV data"
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Asset symbol (e.g., EURUSD). If not provided, processes all 34 assets."
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "update"],
        default="historical",
        help="Mode: 'historical' for full analysis, 'update' for recent data only"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    if args.symbol:
        symbol = args.symbol.upper().replace("_", "")
        print(f"\nProcessing {symbol}...")
        results = process_asset(symbol, args.mode)
        
        if args.verbose or True:
            print_results_table(results["monthly"], symbol, "MN")
            print_results_table(results["weekly"], symbol, "W1")
    else:
        process_all_assets(args.mode)


if __name__ == "__main__":
    main()
