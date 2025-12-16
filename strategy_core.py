"""
Strategy Core Module for Blueprint Trader AI.

This module provides the single source of truth for trading rules,
used by both backtests and live scanning/Discord outputs.

The strategy is parameterized to allow optimization while staying
faithful to the Blueprint confluence-based approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any


def _get_candle_datetime(candle: Dict) -> Optional[datetime]:
    """Extract datetime from candle dictionary."""
    time_val = candle.get("time") or candle.get("timestamp") or candle.get("date")
    if time_val is None:
        return None
    
    try:
        if isinstance(time_val, str):
            return datetime.fromisoformat(time_val.replace("Z", "+00:00"))
        elif isinstance(time_val, datetime):
            return time_val
        else:
            return None
    except (ValueError, TypeError):
        return None


def _slice_htf_by_timestamp(
    htf_candles: Optional[List[Dict]],
    reference_dt: datetime
) -> Optional[List[Dict]]:
    """
    Slice higher-timeframe candles to only include those with timestamp <= reference.
    
    This prevents look-ahead bias by ensuring we only use HTF data
    that would have been available at the time of the reference candle.
    
    Args:
        htf_candles: List of higher-timeframe candles (weekly, monthly, etc)
        reference_dt: The reference datetime (current daily candle time)
    
    Returns:
        Sliced list of candles or None if input is None/empty
    """
    if not htf_candles:
        return None
    
    result = []
    for candle in htf_candles:
        candle_dt = _get_candle_datetime(candle)
        if candle_dt is None:
            result.append(candle)
        elif candle_dt <= reference_dt:
            result.append(candle)
        else:
            break
    
    return result if result else None


def _is_weekend(candle: Dict) -> bool:
    """Check if a candle falls on a weekend (Saturday=5, Sunday=6)."""
    dt = _get_candle_datetime(candle)
    if dt is None:
        return False
    return dt.weekday() >= 5  # 5=Saturday, 6=Sunday


@dataclass
class StrategyParams:
    """
    Strategy parameters that can be optimized.
    
    These control confluence thresholds, SL/TP ratios, filters, etc.
    """
    min_confluence: int = 6
    min_quality_factors: int = 3
    
    atr_sl_multiplier: float = 1.5
    atr_tp1_multiplier: float = 0.6
    atr_tp2_multiplier: float = 1.2
    atr_tp3_multiplier: float = 2.0
    atr_tp4_multiplier: float = 3.0
    atr_tp5_multiplier: float = 4.0
    
    fib_low: float = 0.382
    fib_high: float = 0.886
    
    structure_sl_lookback: int = 35
    liquidity_sweep_lookback: int = 12
    
    use_htf_filter: bool = True
    use_structure_filter: bool = True
    use_liquidity_filter: bool = True
    use_fib_filter: bool = True
    use_confirmation_filter: bool = True
    
    require_htf_alignment: bool = False
    require_confirmation_for_active: bool = True
    require_rr_for_active: bool = True
    
    min_rr_ratio: float = 1.0
    risk_per_trade_pct: float = 1.0
    
    cooldown_bars: int = 0
    max_open_trades: int = 3
    
    # Partial take-profit percentages (must sum to 1.0)
    tp1_close_pct: float = 0.10  # Close 10% at TP1
    tp2_close_pct: float = 0.10  # Close 10% at TP2
    tp3_close_pct: float = 0.15  # Close 15% at TP3
    tp4_close_pct: float = 0.20  # Close 20% at TP4
    tp5_close_pct: float = 0.45  # Close 45% at TP5
    
    # Quantitative enhancement filters
    use_atr_regime_filter: bool = True
    atr_min_percentile: float = 60.0
    use_zscore_filter: bool = True
    zscore_threshold: float = 1.5
    use_pattern_filter: bool = True
    
    # Blueprint V2 enhancements
    use_mitigated_sr: bool = True  # Broken then retested SR zones
    sr_proximity_pct: float = 0.02  # 1-2% proximity filter for SR entry
    use_structural_framework: bool = True  # Ascending/descending channel detection
    use_displacement_filter: bool = True  # Strong candles beyond structure
    displacement_atr_mult: float = 1.5  # Min ATR multiplier for displacement
    use_candle_rejection: bool = True  # Pinbar/engulfing at SR
    
    # Advanced quant filters
    use_rsi_divergence: bool = True
    rsi_period: int = 14
    use_momentum_filter: bool = True
    momentum_lookback: int = 10
    use_mean_reversion: bool = True
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # ML filter parameters
    ml_min_prob: float = 0.6
    
    # New FTMO challenge parameters
    trail_activation_r: float = 2.2  # Delay trailing stop activation until this R is reached
    december_atr_multiplier: float = 1.5  # Extra strict ATR threshold only in December
    volatile_asset_boost: float = 1.5  # Boost scoring for high-ATR assets
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            "min_confluence": self.min_confluence,
            "min_quality_factors": self.min_quality_factors,
            "atr_sl_multiplier": self.atr_sl_multiplier,
            "atr_tp1_multiplier": self.atr_tp1_multiplier,
            "atr_tp2_multiplier": self.atr_tp2_multiplier,
            "atr_tp3_multiplier": self.atr_tp3_multiplier,
            "fib_low": self.fib_low,
            "fib_high": self.fib_high,
            "structure_sl_lookback": self.structure_sl_lookback,
            "liquidity_sweep_lookback": self.liquidity_sweep_lookback,
            "use_htf_filter": self.use_htf_filter,
            "use_structure_filter": self.use_structure_filter,
            "use_liquidity_filter": self.use_liquidity_filter,
            "use_fib_filter": self.use_fib_filter,
            "use_confirmation_filter": self.use_confirmation_filter,
            "require_htf_alignment": self.require_htf_alignment,
            "require_confirmation_for_active": self.require_confirmation_for_active,
            "require_rr_for_active": self.require_rr_for_active,
            "min_rr_ratio": self.min_rr_ratio,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "cooldown_bars": self.cooldown_bars,
            "max_open_trades": self.max_open_trades,
            "tp1_close_pct": self.tp1_close_pct,
            "tp2_close_pct": self.tp2_close_pct,
            "tp3_close_pct": self.tp3_close_pct,
            "tp4_close_pct": self.tp4_close_pct,
            "tp5_close_pct": self.tp5_close_pct,
            "use_atr_regime_filter": self.use_atr_regime_filter,
            "atr_min_percentile": self.atr_min_percentile,
            "use_zscore_filter": self.use_zscore_filter,
            "zscore_threshold": self.zscore_threshold,
            "use_pattern_filter": self.use_pattern_filter,
            "use_mitigated_sr": self.use_mitigated_sr,
            "sr_proximity_pct": self.sr_proximity_pct,
            "use_structural_framework": self.use_structural_framework,
            "use_displacement_filter": self.use_displacement_filter,
            "displacement_atr_mult": self.displacement_atr_mult,
            "use_candle_rejection": self.use_candle_rejection,
            "use_rsi_divergence": self.use_rsi_divergence,
            "rsi_period": self.rsi_period,
            "use_momentum_filter": self.use_momentum_filter,
            "momentum_lookback": self.momentum_lookback,
            "use_mean_reversion": self.use_mean_reversion,
            "bollinger_period": self.bollinger_period,
            "bollinger_std": self.bollinger_std,
            "ml_min_prob": self.ml_min_prob,
            "trail_activation_r": self.trail_activation_r,
            "december_atr_multiplier": self.december_atr_multiplier,
            "volatile_asset_boost": self.volatile_asset_boost,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyParams":
        """Create parameters from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


@dataclass
class Signal:
    """Represents a trading signal/setup."""
    symbol: str
    direction: str
    bar_index: int
    timestamp: Any
    
    confluence_score: int = 0
    quality_factors: int = 0
    
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    tp4: Optional[float] = None
    tp5: Optional[float] = None
    
    is_active: bool = False
    is_watching: bool = False
    
    flags: Dict[str, bool] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class Trade:
    """Represents a completed trade for backtest analysis."""
    symbol: str
    direction: str
    entry_date: Any
    exit_date: Any
    entry_price: float
    exit_price: float
    stop_loss: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    tp4: Optional[float] = None
    tp5: Optional[float] = None
    
    risk: float = 0.0
    reward: float = 0.0
    rr: float = 0.0
    
    is_winner: bool = False
    exit_reason: str = ""
    
    confluence_score: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_date": str(self.entry_date),
            "exit_date": str(self.exit_date),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "tp4": self.tp4,
            "tp5": self.tp5,
            "risk": self.risk,
            "reward": self.reward,
            "rr": self.rr,
            "is_winner": self.is_winner,
            "exit_reason": self.exit_reason,
            "confluence_score": self.confluence_score,
        }


def _atr(candles: List[Dict], period: int = 14) -> float:
    """
    Calculate Average True Range (ATR).
    
    Args:
        candles: List of OHLCV candle dictionaries
        period: ATR period (default 14)
    
    Returns:
        ATR value or 0 if insufficient data
    """
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
    """
    Calculate current ATR and its percentile rank over the lookback period.
    
    Args:
        candles: List of OHLCV candle dictionaries
        period: ATR period (default 14)
        lookback: Number of days to calculate percentile over (default 100)
    
    Returns:
        Tuple of (current_atr, percentile_rank 0-100)
    """
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


def calculate_volatility_parity_risk(
    atr: float,
    base_risk_pct: float,
    reference_atr: Optional[float] = None,
    min_risk_pct: float = 0.25,
    max_risk_pct: float = 2.0,
) -> float:
    """
    Calculate volatility parity adjusted risk percentage.
    
    Volatility parity ensures equal dollar risk per ATR unit across all symbols,
    normalizing position sizes based on current volatility.
    
    Args:
        atr: Current ATR value for the symbol
        base_risk_pct: Base risk percentage (e.g., 0.5 for 0.5%)
        reference_atr: Reference ATR for normalization. If None or 0, returns base risk.
        min_risk_pct: Minimum risk percentage floor (default 0.25%)
        max_risk_pct: Maximum risk percentage cap (default 2.0%)
    
    Returns:
        Adjusted risk percentage capped between min and max
        
    Example:
        If reference_atr=0.0010 and current atr=0.0020 (2x more volatile):
        adjusted_risk = base_risk * (0.0010 / 0.0020) = base_risk * 0.5
        This reduces position size for more volatile instruments.
    """
    if atr <= 0 or reference_atr is None or reference_atr <= 0:
        return max(min_risk_pct, min(max_risk_pct, base_risk_pct))
    
    adjustment_ratio = reference_atr / atr
    adjusted_risk = base_risk_pct * adjustment_ratio
    
    return max(min_risk_pct, min(max_risk_pct, adjusted_risk))


def _detect_bullish_n_pattern(candles: List[Dict], lookback: int = 10) -> Tuple[bool, str]:
    """
    Detect Bullish N pattern: impulse up, pullback, higher low formation.
    
    Pattern criteria:
    1. Initial impulse up (significant up move)
    2. Pullback/retracement (down move that doesn't break prior low)
    3. Higher low confirmed (new low above previous swing low)
    
    Args:
        candles: List of OHLCV candle dictionaries
        lookback: Number of candles to analyze (default 10)
    
    Returns:
        Tuple of (pattern_detected, description)
    """
    if len(candles) < lookback + 5:
        return False, "Insufficient data for pattern detection"
    
    recent = candles[-(lookback + 5):]
    
    swing_highs, swing_lows = _find_pivots(recent, lookback=2)
    
    if len(swing_lows) < 2 or len(swing_highs) < 1:
        return False, "Not enough swing points"
    
    last_low = swing_lows[-1]
    prev_low = swing_lows[-2]
    last_high = swing_highs[-1] if swing_highs else recent[-1]["high"]
    
    higher_low = last_low > prev_low
    
    impulse_range = last_high - prev_low
    atr = _atr(candles, 14)
    significant_impulse = impulse_range > atr * 1.5 if atr > 0 else False
    
    current_price = recent[-1]["close"]
    pullback_depth = (last_high - current_price) / impulse_range if impulse_range > 0 else 0
    valid_pullback = 0.2 <= pullback_depth <= 0.7
    
    if higher_low and significant_impulse and valid_pullback:
        return True, f"Bullish N: Higher low at {last_low:.5f}, impulse from {prev_low:.5f}"
    elif higher_low and significant_impulse:
        return True, f"Bullish N forming: Higher low confirmed, awaiting pullback"
    elif higher_low:
        return False, "Higher low present but no clear impulse"
    else:
        return False, "No Bullish N pattern detected"


def _detect_bearish_v_pattern(candles: List[Dict], lookback: int = 10) -> Tuple[bool, str]:
    """
    Detect Bearish V pattern: impulse down, pullback, lower high formation.
    
    Pattern criteria:
    1. Initial impulse down (significant down move)
    2. Pullback/retracement (up move that doesn't break prior high)
    3. Lower high confirmed (new high below previous swing high)
    
    Args:
        candles: List of OHLCV candle dictionaries
        lookback: Number of candles to analyze (default 10)
    
    Returns:
        Tuple of (pattern_detected, description)
    """
    if len(candles) < lookback + 5:
        return False, "Insufficient data for pattern detection"
    
    recent = candles[-(lookback + 5):]
    
    swing_highs, swing_lows = _find_pivots(recent, lookback=2)
    
    if len(swing_highs) < 2 or len(swing_lows) < 1:
        return False, "Not enough swing points"
    
    last_high = swing_highs[-1]
    prev_high = swing_highs[-2]
    last_low = swing_lows[-1] if swing_lows else recent[-1]["low"]
    
    lower_high = last_high < prev_high
    
    impulse_range = prev_high - last_low
    atr = _atr(candles, 14)
    significant_impulse = impulse_range > atr * 1.5 if atr > 0 else False
    
    current_price = recent[-1]["close"]
    pullback_depth = (current_price - last_low) / impulse_range if impulse_range > 0 else 0
    valid_pullback = 0.2 <= pullback_depth <= 0.7
    
    if lower_high and significant_impulse and valid_pullback:
        return True, f"Bearish V: Lower high at {last_high:.5f}, impulse from {prev_high:.5f}"
    elif lower_high and significant_impulse:
        return True, f"Bearish V forming: Lower high confirmed, awaiting pullback"
    elif lower_high:
        return False, "Lower high present but no clear impulse"
    else:
        return False, "No Bearish V pattern detected"


def _calculate_zscore(price: float, candles: List[Dict], period: int = 20) -> float:
    """
    Calculate z-score of current price relative to moving average.
    
    Z-score measures how many standard deviations the price is from the mean.
    - Z-score < 0: Price below mean
    - Z-score > 0: Price above mean
    - |Z-score| > 2: Extreme deviation
    
    Args:
        price: Current price
        candles: List of OHLCV candle dictionaries
        period: Period for mean/std calculation (default 20)
    
    Returns:
        Z-score value (0.0 if insufficient data)
    """
    if len(candles) < period:
        return 0.0
    
    closes = [c["close"] for c in candles[-period:] if c.get("close") is not None]
    
    if len(closes) < period:
        return 0.0
    
    mean = sum(closes) / len(closes)
    
    variance = sum((x - mean) ** 2 for x in closes) / len(closes)
    std = variance ** 0.5
    
    if std == 0:
        return 0.0
    
    zscore = (price - mean) / std
    return zscore


def _find_bos_swing_for_bullish_n(candles: List[Dict], lookback: int = 20) -> Optional[Tuple[float, float, int, int]]:
    """
    Find the Bullish N pattern anchor points for Fibonacci:
    After a break of structure UP, find the swing low.
    Return fib anchors from red candle close to green candle open at swing low.
    
    Blueprint rule: For bullish N, take fibs from red candle close to green candle open.
    
    Returns:
        Tuple of (fib_start, fib_end, swing_low_idx, bos_idx) or None
    """
    if len(candles) < lookback + 10:
        return None
    
    recent = candles[-(lookback + 10):]
    
    bos_idx = None
    for i in range(len(recent) - 1, 4, -1):
        candle = recent[i]
        prev_high = max(c["high"] for c in recent[max(0, i-5):i])
        if candle["close"] > prev_high and candle["close"] > candle["open"]:
            is_strong = (candle["high"] - candle["low"]) > 0
            if is_strong:
                bos_idx = i
                break
    
    if bos_idx is None:
        return None
    
    swing_low_idx = None
    swing_low_val = float('inf')
    for i in range(bos_idx - 1, max(0, bos_idx - 10), -1):
        if recent[i]["low"] < swing_low_val:
            swing_low_val = recent[i]["low"]
            swing_low_idx = i
    
    if swing_low_idx is None:
        return None
    
    red_candle_close = None
    green_candle_open = None
    
    for i in range(swing_low_idx, min(len(recent), swing_low_idx + 3)):
        c = recent[i]
        if c["close"] < c["open"]:  # Red/bearish candle
            red_candle_close = c["close"]
        elif c["close"] > c["open"] and red_candle_close is not None:  # Green/bullish after red
            green_candle_open = c["open"]
            break
    
    if red_candle_close is None:
        red_candle_close = swing_low_val
    if green_candle_open is None:
        green_candle_open = recent[min(swing_low_idx + 1, len(recent) - 1)]["open"]
    
    fib_start = min(red_candle_close, green_candle_open)
    bos_candle = recent[bos_idx]
    fib_end = bos_candle["close"]
    
    return (fib_start, fib_end, swing_low_idx, bos_idx)


def _find_bos_swing_for_bearish_v(candles: List[Dict], lookback: int = 20) -> Optional[Tuple[float, float, int, int]]:
    """
    Find the Bearish V pattern anchor points for Fibonacci:
    After a break of structure DOWN, find the swing high.
    Return fib anchors from green candle close to red candle open at swing high.
    
    Blueprint rule: For bearish V (shorts), take fibs from green candle close to red candle open.
    
    Returns:
        Tuple of (fib_start, fib_end, swing_high_idx, bos_idx) or None
    """
    if len(candles) < lookback + 10:
        return None
    
    recent = candles[-(lookback + 10):]
    
    bos_idx = None
    for i in range(len(recent) - 1, 4, -1):
        candle = recent[i]
        prev_low = min(c["low"] for c in recent[max(0, i-5):i])
        if candle["close"] < prev_low and candle["close"] < candle["open"]:
            is_strong = (candle["high"] - candle["low"]) > 0
            if is_strong:
                bos_idx = i
                break
    
    if bos_idx is None:
        return None
    
    swing_high_idx = None
    swing_high_val = float('-inf')
    for i in range(bos_idx - 1, max(0, bos_idx - 10), -1):
        if recent[i]["high"] > swing_high_val:
            swing_high_val = recent[i]["high"]
            swing_high_idx = i
    
    if swing_high_idx is None:
        return None
    
    green_candle_close = None
    red_candle_open = None
    
    for i in range(swing_high_idx, min(len(recent), swing_high_idx + 3)):
        c = recent[i]
        if c["close"] > c["open"]:  # Green/bullish candle
            green_candle_close = c["close"]
        elif c["close"] < c["open"] and green_candle_close is not None:  # Red/bearish after green
            red_candle_open = c["open"]
            break
    
    if green_candle_close is None:
        green_candle_close = swing_high_val
    if red_candle_open is None:
        red_candle_open = recent[min(swing_high_idx + 1, len(recent) - 1)]["open"]
    
    fib_start = max(green_candle_close, red_candle_open)
    bos_candle = recent[bos_idx]
    fib_end = bos_candle["close"]
    
    return (fib_start, fib_end, swing_high_idx, bos_idx)


def _detect_mitigated_sr(candles: List[Dict], price: float, direction: str, 
                         proximity_pct: float = 0.02) -> Tuple[bool, str, Optional[float]]:
    """
    Detect mitigated S/R zones - zones that were broken then retested.
    
    A mitigated zone is more reliable because it shows the level has been
    tested, broken, and is now acting as support (former resistance) or
    resistance (former support).
    
    Args:
        candles: OHLCV candles
        price: Current price
        direction: Trade direction
        proximity_pct: How close price must be to zone (default 2%)
    
    Returns:
        Tuple of (is_at_mitigated_sr, note, sr_level)
    """
    if len(candles) < 50:
        return False, "Mitigated SR: Insufficient data", None
    
    swing_highs_with_idx = []
    swing_lows_with_idx = []
    lookback = 3
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i]["high"]
        low = candles[i]["low"]
        
        is_swing_high = all(candles[j]["high"] <= high for j in range(i - lookback, i + lookback + 1) if j != i)
        is_swing_low = all(candles[j]["low"] >= low for j in range(i - lookback, i + lookback + 1) if j != i)
        
        if is_swing_high:
            swing_highs_with_idx.append((high, i))
        if is_swing_low:
            swing_lows_with_idx.append((low, i))
    
    mitigated_levels = []
    
    for sr_level, sr_idx in swing_highs_with_idx:
        was_broken = False
        was_retested = False
        for i in range(sr_idx + 1, len(candles)):
            if candles[i]["close"] > sr_level:
                was_broken = True
            if was_broken and candles[i]["low"] <= sr_level <= candles[i]["high"]:
                was_retested = True
                break
        if was_broken and was_retested:
            mitigated_levels.append(("resistance_turned_support", sr_level))
    
    for sr_level, sr_idx in swing_lows_with_idx:
        was_broken = False
        was_retested = False
        for i in range(sr_idx + 1, len(candles)):
            if candles[i]["close"] < sr_level:
                was_broken = True
            if was_broken and candles[i]["low"] <= sr_level <= candles[i]["high"]:
                was_retested = True
                break
        if was_broken and was_retested:
            mitigated_levels.append(("support_turned_resistance", sr_level))
    
    for level_type, level in mitigated_levels:
        distance_pct = abs(price - level) / price if price > 0 else 0
        
        if distance_pct <= proximity_pct:
            if direction == "bullish" and level_type == "resistance_turned_support":
                return True, f"Mitigated SR: At RTS level {level:.5f} (within {distance_pct:.1%})", level
            elif direction == "bearish" and level_type == "support_turned_resistance":
                return True, f"Mitigated SR: At STR level {level:.5f} (within {distance_pct:.1%})", level
    
    return False, "Mitigated SR: No qualified level nearby", None


def _detect_structural_framework(candles: List[Dict], direction: str) -> Tuple[bool, str, Optional[Tuple[float, float]]]:
    """
    Detect ascending/descending channel frameworks on daily timeframe.
    
    Framework detection:
    - Ascending channel: Connect 3+ swing lows (ascending) and 3+ swing highs (ascending)
    - Descending channel: Connect 3+ swing highs (descending) and 3+ swing lows (descending)
    
    Returns:
        Tuple of (is_in_framework, note, (lower_bound, upper_bound) or None)
    """
    if len(candles) < 30:
        return False, "Framework: Insufficient data", None
    
    swing_highs, swing_lows = [], []
    lookback = 3
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i]["high"]
        low = candles[i]["low"]
        
        is_swing_high = all(candles[j]["high"] <= high for j in range(i - lookback, i + lookback + 1) if j != i)
        is_swing_low = all(candles[j]["low"] >= low for j in range(i - lookback, i + lookback + 1) if j != i)
        
        if is_swing_high:
            swing_highs.append((i, high))
        if is_swing_low:
            swing_lows.append((i, low))
    
    if len(swing_lows) < 3 or len(swing_highs) < 3:
        return False, "Framework: Not enough swing points", None
    
    recent_lows = swing_lows[-5:]
    ascending_lows = all(recent_lows[i][1] <= recent_lows[i+1][1] for i in range(len(recent_lows)-1))
    descending_lows = all(recent_lows[i][1] >= recent_lows[i+1][1] for i in range(len(recent_lows)-1))
    
    recent_highs = swing_highs[-5:]
    ascending_highs = all(recent_highs[i][1] <= recent_highs[i+1][1] for i in range(len(recent_highs)-1))
    descending_highs = all(recent_highs[i][1] >= recent_highs[i+1][1] for i in range(len(recent_highs)-1))
    
    current_price = candles[-1]["close"]
    last_low = swing_lows[-1][1] if swing_lows else candles[-1]["low"]
    last_high = swing_highs[-1][1] if swing_highs else candles[-1]["high"]
    
    if ascending_lows and ascending_highs:
        if direction == "bullish":
            price_position = (current_price - last_low) / (last_high - last_low) if last_high > last_low else 0.5
            if price_position < 0.4:
                return True, f"Framework: Ascending channel - price near lower bound ({price_position:.0%})", (last_low, last_high)
        return False, "Framework: Ascending channel but not at optimal entry", (last_low, last_high)
    
    elif descending_lows and descending_highs:
        if direction == "bearish":
            price_position = (current_price - last_low) / (last_high - last_low) if last_high > last_low else 0.5
            if price_position > 0.6:
                return True, f"Framework: Descending channel - price near upper bound ({price_position:.0%})", (last_low, last_high)
        return False, "Framework: Descending channel but not at optimal entry", (last_low, last_high)
    
    return False, "Framework: No clear channel detected", None


def _detect_displacement(candles: List[Dict], direction: str, atr_mult: float = 1.5) -> Tuple[bool, str]:
    """
    Detect displacement - strong candles beyond structure confirming the move.
    
    Displacement = large body candle that shows institutional order flow.
    Must be at least atr_mult * ATR in body size.
    
    Returns:
        Tuple of (has_displacement, note)
    """
    if len(candles) < 20:
        return False, "Displacement: Insufficient data"
    
    atr = _atr(candles, 14)
    if atr <= 0:
        return False, "Displacement: ATR calculation failed"
    
    min_body = atr * atr_mult
    
    for i in range(-5, 0):
        if abs(i) > len(candles):
            continue
        c = candles[i]
        body = abs(c["close"] - c["open"])
        
        if body >= min_body:
            if direction == "bullish" and c["close"] > c["open"]:
                return True, f"Displacement: Strong bullish candle ({body/atr:.1f}x ATR)"
            elif direction == "bearish" and c["close"] < c["open"]:
                return True, f"Displacement: Strong bearish candle ({body/atr:.1f}x ATR)"
    
    return False, "Displacement: No strong impulse candle found"


def _detect_candle_rejection(candles: List[Dict], direction: str) -> Tuple[bool, str]:
    """
    Detect pinbar or engulfing rejection patterns at current price.
    
    Returns:
        Tuple of (has_rejection, note)
    """
    if len(candles) < 3:
        return False, "Rejection: Insufficient data"
    
    curr = candles[-1]
    prev = candles[-2]
    
    body = abs(curr["close"] - curr["open"])
    full_range = curr["high"] - curr["low"]
    upper_wick = curr["high"] - max(curr["close"], curr["open"])
    lower_wick = min(curr["close"], curr["open"]) - curr["low"]
    
    if full_range > 0:
        body_ratio = body / full_range
        
        if direction == "bullish":
            if lower_wick > body * 2 and lower_wick > upper_wick * 1.5:
                return True, "Rejection: Bullish pinbar (long lower wick)"
            if curr["close"] > curr["open"] and curr["close"] > prev["high"] and curr["open"] < prev["low"]:
                return True, "Rejection: Bullish engulfing pattern"
            if curr["close"] > curr["open"] and lower_wick > body:
                return True, "Rejection: Hammer pattern"
        else:
            if upper_wick > body * 2 and upper_wick > lower_wick * 1.5:
                return True, "Rejection: Bearish pinbar (long upper wick)"
            if curr["close"] < curr["open"] and curr["open"] > prev["high"] and curr["close"] < prev["low"]:
                return True, "Rejection: Bearish engulfing pattern"
            if curr["close"] < curr["open"] and upper_wick > body:
                return True, "Rejection: Shooting star pattern"
    
    return False, "Rejection: No clear rejection pattern"


def _calculate_rsi(candles: List[Dict], period: int = 14) -> float:
    """Calculate RSI indicator."""
    if len(candles) < period + 1:
        return 50.0
    
    closes = [c["close"] for c in candles[-(period + 1):]]
    gains, losses = [], []
    
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def _detect_rsi_divergence(candles: List[Dict], direction: str, period: int = 14) -> Tuple[bool, str]:
    """
    Detect RSI divergence for confirmation.
    
    Bullish divergence: Price makes lower low but RSI makes higher low
    Bearish divergence: Price makes higher high but RSI makes lower high
    
    Returns:
        Tuple of (has_divergence, note)
    """
    if len(candles) < period + 20:
        return False, "RSI Divergence: Insufficient data"
    
    rsi_values = []
    for i in range(20):
        end_idx = len(candles) - 19 + i
        slice_candles = candles[:end_idx + 1]
        rsi = _calculate_rsi(slice_candles, period)
        rsi_values.append(rsi)
    
    price_lows = [candles[-(20-i)]["low"] for i in range(20)]
    price_highs = [candles[-(20-i)]["high"] for i in range(20)]
    
    if direction == "bullish":
        recent_price_low = min(price_lows[-10:])
        older_price_low = min(price_lows[:10])
        recent_rsi_low = min(rsi_values[-10:])
        older_rsi_low = min(rsi_values[:10])
        
        if recent_price_low < older_price_low and recent_rsi_low > older_rsi_low:
            return True, f"RSI Divergence: Bullish (price LL, RSI HL) - RSI: {rsi_values[-1]:.1f}"
    else:
        recent_price_high = max(price_highs[-10:])
        older_price_high = max(price_highs[:10])
        recent_rsi_high = max(rsi_values[-10:])
        older_rsi_high = max(rsi_values[:10])
        
        if recent_price_high > older_price_high and recent_rsi_high < older_rsi_high:
            return True, f"RSI Divergence: Bearish (price HH, RSI LH) - RSI: {rsi_values[-1]:.1f}"
    
    current_rsi = rsi_values[-1]
    if direction == "bullish" and current_rsi < 35:
        return True, f"RSI Divergence: Oversold ({current_rsi:.1f}) - potential bounce"
    elif direction == "bearish" and current_rsi > 65:
        return True, f"RSI Divergence: Overbought ({current_rsi:.1f}) - potential drop"
    
    return False, f"RSI Divergence: No divergence detected (RSI: {current_rsi:.1f})"


def _detect_momentum(candles: List[Dict], direction: str, lookback: int = 10) -> Tuple[bool, str]:
    """
    Detect momentum alignment using rate of change.
    
    Returns:
        Tuple of (momentum_aligned, note)
    """
    if len(candles) < lookback + 5:
        return False, "Momentum: Insufficient data"
    
    closes = [c["close"] for c in candles[-(lookback + 5):]]
    
    current = closes[-1]
    past = closes[-lookback]
    roc = ((current - past) / past) * 100 if past > 0 else 0
    
    short_roc = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if closes[-5] > 0 else 0
    
    if direction == "bullish":
        if roc > 0 and short_roc > 0:
            return True, f"Momentum: Bullish aligned (ROC: {roc:.2f}%, Short: {short_roc:.2f}%)"
        elif roc < -2 and short_roc > 0:
            return True, f"Momentum: Mean reversion setup (pulling back in downtrend, bounce starting)"
    else:
        if roc < 0 and short_roc < 0:
            return True, f"Momentum: Bearish aligned (ROC: {roc:.2f}%, Short: {short_roc:.2f}%)"
        elif roc > 2 and short_roc < 0:
            return True, f"Momentum: Mean reversion setup (rallying in uptrend, reversal starting)"
    
    return False, f"Momentum: Not aligned (ROC: {roc:.2f}%, Short: {short_roc:.2f}%)"


def _detect_bollinger_mean_reversion(candles: List[Dict], direction: str, 
                                     period: int = 20, std_mult: float = 2.0) -> Tuple[bool, str]:
    """
    Detect mean reversion setup using Bollinger Bands.
    
    Returns:
        Tuple of (is_mean_reversion_setup, note)
    """
    if len(candles) < period + 5:
        return False, "Bollinger: Insufficient data"
    
    closes = [c["close"] for c in candles[-period:]]
    
    mean = sum(closes) / len(closes)
    variance = sum((x - mean) ** 2 for x in closes) / len(closes)
    std = variance ** 0.5
    
    upper_band = mean + std_mult * std
    lower_band = mean - std_mult * std
    
    current_price = closes[-1]
    
    if direction == "bullish":
        if current_price <= lower_band:
            return True, f"Bollinger: At lower band ({current_price:.5f} <= {lower_band:.5f}) - bounce expected"
        elif current_price < mean:
            band_position = (current_price - lower_band) / (mean - lower_band) if mean > lower_band else 0.5
            if band_position < 0.3:
                return True, f"Bollinger: Near lower band ({band_position:.0%} from lower) - mean reversion setup"
    else:
        if current_price >= upper_band:
            return True, f"Bollinger: At upper band ({current_price:.5f} >= {upper_band:.5f}) - drop expected"
        elif current_price > mean:
            band_position = (current_price - mean) / (upper_band - mean) if upper_band > mean else 0.5
            if band_position > 0.7:
                return True, f"Bollinger: Near upper band ({band_position:.0%} from mean) - mean reversion setup"
    
    return False, f"Bollinger: Not at extreme (price: {current_price:.5f}, bands: {lower_band:.5f}-{upper_band:.5f})"


def _find_pivots(candles: List[Dict], lookback: int = 5) -> Tuple[List[float], List[float]]:
    """
    Find swing highs and swing lows in candle data.
    
    Args:
        candles: List of OHLCV candle dictionaries
        lookback: Number of bars to look back/forward for pivot identification
    
    Returns:
        Tuple of (swing_highs, swing_lows) as lists of price levels
    """
    if len(candles) < lookback * 2 + 1:
        return [], []
    
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i]["high"]
        low = candles[i]["low"]
        
        is_swing_high = True
        is_swing_low = True
        
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if candles[j]["high"] > high:
                is_swing_high = False
            if candles[j]["low"] < low:
                is_swing_low = False
        
        if is_swing_high:
            swing_highs.append(high)
        if is_swing_low:
            swing_lows.append(low)
    
    return swing_highs, swing_lows


def _infer_trend(candles: List[Dict], ema_short: int = 8, ema_long: int = 21) -> str:
    """
    Infer trend direction from candle data using EMA crossover and price action.
    
    Args:
        candles: List of OHLCV candle dictionaries
        ema_short: Short EMA period
        ema_long: Long EMA period
    
    Returns:
        "bullish", "bearish", or "mixed"
    """
    if not candles or len(candles) < ema_long + 5:
        return "mixed"
    
    closes = [c["close"] for c in candles if c.get("close") is not None]
    
    if len(closes) < ema_long + 5:
        return "mixed"
    
    def calc_ema(values: List[float], period: int) -> float:
        if len(values) < period:
            valid_values = [v for v in values if v is not None and v == v]
            return sum(valid_values) / len(valid_values) if valid_values else 0
        k = 2 / (period + 1)
        initial_values = [v for v in values[:period] if v is not None and v == v]
        if not initial_values:
            return 0
        ema = sum(initial_values) / len(initial_values)
        for price in values[period:]:
            if price is not None and price == price:
                ema = price * k + ema * (1 - k)
        return ema
    
    ema_s = calc_ema(closes, ema_short)
    ema_l = calc_ema(closes, ema_long)
    
    current_price = closes[-1]
    recent_high = max(c["high"] for c in candles[-10:])
    recent_low = min(c["low"] for c in candles[-10:])
    
    bullish_signals = 0
    bearish_signals = 0
    
    if ema_s > ema_l:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    if current_price > ema_l:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    if len(closes) >= 20:
        higher_highs = closes[-1] > max(closes[-10:-1]) if len(closes) > 10 else False
        lower_lows = closes[-1] < min(closes[-10:-1]) if len(closes) > 10 else False
        
        if higher_highs:
            bullish_signals += 1
        if lower_lows:
            bearish_signals += 1
    
    if bullish_signals > bearish_signals:
        return "bullish"
    elif bearish_signals > bullish_signals:
        return "bearish"
    else:
        return "mixed"


def _pick_direction_from_bias(
    mn_trend: str,
    wk_trend: str,
    d_trend: str
) -> Tuple[str, str, bool]:
    """
    Determine trade direction based on multi-timeframe bias.
    
    Args:
        mn_trend: Monthly trend
        wk_trend: Weekly trend
        d_trend: Daily trend
    
    Returns:
        Tuple of (direction, note, htf_aligned)
    """
    trends = [mn_trend, wk_trend, d_trend]
    bullish_count = sum(1 for t in trends if t == "bullish")
    bearish_count = sum(1 for t in trends if t == "bearish")
    
    if bullish_count >= 2:
        direction = "bullish"
        htf_aligned = mn_trend == "bullish" or wk_trend == "bullish"
        note = f"HTF bias: {mn_trend.upper()[0]}/{wk_trend.upper()[0]}/{d_trend.upper()[0]} -> Bullish"
    elif bearish_count >= 2:
        direction = "bearish"
        htf_aligned = mn_trend == "bearish" or wk_trend == "bearish"
        note = f"HTF bias: {mn_trend.upper()[0]}/{wk_trend.upper()[0]}/{d_trend.upper()[0]} -> Bearish"
    else:
        direction = d_trend if d_trend != "mixed" else "bullish"
        htf_aligned = False
        note = f"HTF bias: Mixed ({mn_trend[0].upper()}/{wk_trend[0].upper()}/{d_trend[0].upper()})"
    
    return direction, note, htf_aligned


def _location_context(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    price: float,
    direction: str,
    historical_sr: Optional[Dict[str, List[Dict]]] = None,
) -> Tuple[str, bool]:
    """
    Check if price is at a key location (support/resistance zone).
    
    Now also checks against historical S/R levels from 20+ years of data.
    
    Args:
        monthly_candles: Monthly OHLCV data
        weekly_candles: Weekly OHLCV data  
        daily_candles: Daily OHLCV data
        price: Current price
        direction: Trade direction
        historical_sr: Optional dict with 'monthly' and 'weekly' S/R level lists
    
    Returns:
        Tuple of (note, is_valid_location)
    """
    if not daily_candles or len(daily_candles) < 20:
        return "Location: Insufficient data", False
    
    highs = [c["high"] for c in daily_candles[-50:]] if len(daily_candles) >= 50 else [c["high"] for c in daily_candles]
    lows = [c["low"] for c in daily_candles[-50:]] if len(daily_candles) >= 50 else [c["low"] for c in daily_candles]
    
    recent_high = max(highs[-20:])
    recent_low = min(lows[-20:])
    range_size = recent_high - recent_low
    
    if range_size <= 0:
        return "Location: No range", False
    
    swing_highs, swing_lows = _find_pivots(daily_candles[-50:] if len(daily_candles) >= 50 else daily_candles, lookback=3)
    
    atr = _atr(daily_candles, 14)
    zone_tolerance = atr * 0.5 if atr > 0 else range_size * 0.05
    
    near_historical_sr = False
    historical_sr_note = ""
    if historical_sr:
        sr_tolerance = price * 0.005
        monthly_levels = historical_sr.get('monthly', [])
        for sr in monthly_levels[:20]:
            if abs(price - sr['level']) <= sr_tolerance:
                near_historical_sr = True
                historical_sr_note = f" (HTF MN SR: {sr['level']:.5f}, {sr['touches']} touches)"
                break
        
        if not near_historical_sr:
            weekly_levels = historical_sr.get('weekly', [])
            for sr in weekly_levels[:30]:
                if abs(price - sr['level']) <= sr_tolerance:
                    near_historical_sr = True
                    historical_sr_note = f" (HTF W1 SR: {sr['level']:.5f}, {sr['touches']} touches)"
                    break
    
    if direction == "bullish":
        near_support = any(abs(price - sl) < zone_tolerance for sl in swing_lows[-5:]) if swing_lows else False
        near_range_low = (price - recent_low) < range_size * 0.3
        
        if near_historical_sr:
            return f"Location: At historical S/R zone{historical_sr_note}", True
        elif near_support or near_range_low:
            return "Location: Near support zone", True
        else:
            return "Location: Not at key support", False
    else:
        near_resistance = any(abs(price - sh) < zone_tolerance for sh in swing_highs[-5:]) if swing_highs else False
        near_range_high = (recent_high - price) < range_size * 0.3
        
        if near_historical_sr:
            return f"Location: At historical S/R zone{historical_sr_note}", True
        elif near_resistance or near_range_high:
            return "Location: Near resistance zone", True
        else:
            return "Location: Not at key resistance", False


def _fib_context(
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    direction: str,
    price: float,
    fib_low: float = 0.382,
    fib_high: float = 0.886,
) -> Tuple[str, bool]:
    """
    Check if price is within a Fibonacci retracement zone.
    
    Returns:
        Tuple of (note, is_in_fib_zone)
    """
    try:
        candles = daily_candles if daily_candles and len(daily_candles) >= 30 else weekly_candles
        
        if not candles or len(candles) < 20:
            return "Fib: Insufficient data", False
        
        leg = _find_last_swing_leg_for_fib(candles, direction)
        
        if not leg:
            return "Fib: No clear swing leg found", False
        
        lo, hi = leg
        span = hi - lo
        
        if span <= 0:
            return "Fib: Invalid swing range", False
        
        if direction == "bullish":
            fib_382 = hi - span * 0.382
            fib_500 = hi - span * 0.5
            fib_618 = hi - span * 0.618
            fib_786 = hi - span * 0.786
            
            if fib_786 <= price <= fib_382:
                level = round((hi - price) / span, 3)
                return f"Fib: Price at {level:.1%} retracement (Golden Pocket zone)", True
            elif fib_618 <= price <= fib_500:
                return "Fib: Price at 50-61.8% zone", True
            else:
                return "Fib: Price outside retracement zone", False
        else:
            fib_382 = lo + span * 0.382
            fib_500 = lo + span * 0.5
            fib_618 = lo + span * 0.618
            fib_786 = lo + span * 0.786
            
            if fib_382 <= price <= fib_786:
                level = round((price - lo) / span, 3)
                return f"Fib: Price at {level:.1%} retracement (Golden Pocket zone)", True
            elif fib_500 <= price <= fib_618:
                return "Fib: Price at 50-61.8% zone", True
            else:
                return "Fib: Price outside retracement zone", False
    except Exception as e:
        return f"Fib: Error calculating ({type(e).__name__})", False


def _find_last_swing_leg_for_fib(candles: List[Dict], direction: str) -> Optional[Tuple[float, float]]:
    """
    Find the last swing leg for Fibonacci calculation using proper Blueprint anchoring.
    
    Blueprint rules:
    - Bullish N: After BOS up, fibs from red candle close to green candle open at swing low
    - Bearish V: After BOS down, fibs from green candle close to red candle open at swing high
    
    Returns:
        Tuple of (fib_low, fib_high) or None
    """
    if not candles or len(candles) < 20:
        return None
    
    try:
        if direction == "bullish":
            result = _find_bos_swing_for_bullish_n(candles, lookback=20)
            if result:
                fib_start, fib_end, _, _ = result
                return (fib_start, fib_end)
        else:
            result = _find_bos_swing_for_bearish_v(candles, lookback=20)
            if result:
                fib_start, fib_end, _, _ = result
                return (fib_end, fib_start)
    except Exception:
        pass
    
    try:
        swing_highs, swing_lows = _find_pivots(candles, lookback=3)
    except Exception:
        swing_highs, swing_lows = [], []
    
    if not swing_highs or not swing_lows:
        try:
            highs = [c["high"] for c in candles[-30:] if "high" in c]
            lows = [c["low"] for c in candles[-30:] if "low" in c]
            if highs and lows:
                return (min(lows), max(highs))
        except Exception:
            pass
        return None
    
    try:
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        
        if recent_highs and recent_lows:
            hi = max(recent_highs)
            lo = min(recent_lows)
            return (lo, hi)
    except Exception:
        pass
    
    return None


def _daily_liquidity_context(candles: List[Dict], price: float) -> Tuple[str, bool]:
    """
    Check for liquidity sweep or proximity to liquidity pools.
    
    Returns:
        Tuple of (note, is_near_liquidity)
    """
    try:
        if not candles or len(candles) < 10:
            return "Liquidity: Insufficient data", False
        
        lookback = min(20, len(candles))
        recent = candles[-lookback:]
        
        recent_highs = [c["high"] for c in recent if "high" in c]
        recent_lows = [c["low"] for c in recent if "low" in c]
        
        if not recent_highs or not recent_lows:
            return "Liquidity: Invalid data", False
        
        equal_highs = []
        equal_lows = []
        
        atr = _atr(candles, 14)
        tolerance = atr * 0.2 if atr > 0 else (max(recent_highs) - min(recent_lows)) * 0.02
        
        for i, h in enumerate(recent_highs):
            for j, h2 in enumerate(recent_highs):
                if i != j and abs(h - h2) < tolerance:
                    equal_highs.append(h)
                    break
        
        for i, l in enumerate(recent_lows):
            for j, l2 in enumerate(recent_lows):
                if i != j and abs(l - l2) < tolerance:
                    equal_lows.append(l)
                    break
        
        near_equal_high = any(abs(price - h) < tolerance * 2 for h in equal_highs)
        near_equal_low = any(abs(price - l) < tolerance * 2 for l in equal_lows)
        
        current = candles[-1]
        prev = candles[-2] if len(candles) >= 2 else None
        
        swept_high = False
        swept_low = False
        
        if prev:
            prev_high = max(c["high"] for c in candles[-10:-1] if "high" in c)
            prev_low = min(c["low"] for c in candles[-10:-1] if "low" in c)
            
            if current.get("high", 0) > prev_high and current.get("close", 0) < prev_high:
                swept_high = True
            if current.get("low", float("inf")) < prev_low and current.get("close", float("inf")) > prev_low:
                swept_low = True
        
        if swept_high or swept_low:
            return "Liquidity: Sweep detected", True
        elif near_equal_high or near_equal_low:
            return "Liquidity: Near equal highs/lows", True
        else:
            return "Liquidity: No clear liquidity zone", False
    except Exception as e:
        return f"Liquidity: Error ({type(e).__name__})", False


def _structure_context(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    direction: str,
) -> Tuple[bool, str]:
    """
    Check market structure alignment (BOS/CHoCH).
    
    Returns:
        Tuple of (is_aligned, note)
    """
    if not daily_candles or len(daily_candles) < 10:
        return False, "Structure: Insufficient data"
    
    swing_highs, swing_lows = _find_pivots(daily_candles[-30:], lookback=3)
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, "Structure: Not enough swing points"
    
    if direction == "bullish":
        higher_low = swing_lows[-1] > swing_lows[-2] if len(swing_lows) >= 2 else False
        higher_high = swing_highs[-1] > swing_highs[-2] if len(swing_highs) >= 2 else False
        
        bos_up = daily_candles[-1]["close"] > max(swing_highs[-3:]) if swing_highs else False
        
        if bos_up:
            return True, "Structure: BOS up confirmed"
        elif higher_low and higher_high:
            return True, "Structure: HH/HL pattern (bullish)"
        elif higher_low:
            return True, "Structure: Higher low formed"
        else:
            return False, "Structure: No bullish structure"
    else:
        lower_high = swing_highs[-1] < swing_highs[-2] if len(swing_highs) >= 2 else False
        lower_low = swing_lows[-1] < swing_lows[-2] if len(swing_lows) >= 2 else False
        
        bos_down = daily_candles[-1]["close"] < min(swing_lows[-3:]) if swing_lows else False
        
        if bos_down:
            return True, "Structure: BOS down confirmed"
        elif lower_high and lower_low:
            return True, "Structure: LH/LL pattern (bearish)"
        elif lower_high:
            return True, "Structure: Lower high formed"
        else:
            return False, "Structure: No bearish structure"


def _h4_confirmation(
    h4_candles: List[Dict],
    direction: str,
    daily_candles: List[Dict],
) -> Tuple[str, bool]:
    """
    Check for 4H timeframe confirmation (entry trigger).
    
    Returns:
        Tuple of (note, is_confirmed)
    """
    candles = h4_candles if h4_candles and len(h4_candles) >= 5 else daily_candles[-10:]
    
    if not candles or len(candles) < 3:
        return "4H: Insufficient data", False
    
    last = candles[-1]
    prev = candles[-2]
    
    body_last = abs(last["close"] - last["open"])
    range_last = last["high"] - last["low"]
    body_ratio = body_last / range_last if range_last > 0 else 0
    
    if direction == "bullish":
        bullish_candle = last["close"] > last["open"]
        engulfing = (
            last["close"] > last["open"] and
            prev["close"] < prev["open"] and
            last["close"] > prev["open"] and
            last["open"] < prev["close"]
        )
        
        lower_wick = last["open"] - last["low"] if last["close"] > last["open"] else last["close"] - last["low"]
        upper_wick = last["high"] - last["close"] if last["close"] > last["open"] else last["high"] - last["open"]
        pin_bar = lower_wick > body_last * 2 and upper_wick < body_last * 0.5
        
        bos_check = last["high"] > max(c["high"] for c in candles[-5:-1]) if len(candles) >= 5 else False
        
        if engulfing:
            return "4H: Bullish engulfing confirmed", True
        elif pin_bar:
            return "4H: Bullish pin bar (rejection)", True
        elif bos_check and bullish_candle:
            return "4H: Break of structure up", True
        elif bullish_candle and body_ratio > 0.6:
            return "4H: Strong bullish candle", True
        else:
            return "4H: Awaiting bullish confirmation", False
    else:
        bearish_candle = last["close"] < last["open"]
        engulfing = (
            last["close"] < last["open"] and
            prev["close"] > prev["open"] and
            last["close"] < prev["open"] and
            last["open"] > prev["close"]
        )
        
        upper_wick = last["high"] - last["open"] if last["close"] < last["open"] else last["high"] - last["close"]
        lower_wick = last["close"] - last["low"] if last["close"] < last["open"] else last["open"] - last["low"]
        pin_bar = upper_wick > body_last * 2 and lower_wick < body_last * 0.5
        
        bos_check = last["low"] < min(c["low"] for c in candles[-5:-1]) if len(candles) >= 5 else False
        
        if engulfing:
            return "4H: Bearish engulfing confirmed", True
        elif pin_bar:
            return "4H: Bearish pin bar (rejection)", True
        elif bos_check and bearish_candle:
            return "4H: Break of structure down", True
        elif bearish_candle and body_ratio > 0.6:
            return "4H: Strong bearish candle", True
        else:
            return "4H: Awaiting bearish confirmation", False


def _find_structure_sl(candles: List[Dict], direction: str, lookback: int = 35) -> Optional[float]:
    """
    Find structure-based stop loss level.
    
    Returns:
        Stop loss price level or None
    """
    if not candles or len(candles) < 5:
        return None
    
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    swing_highs, swing_lows = _find_pivots(recent, lookback=3)
    
    if direction == "bullish":
        if swing_lows:
            return min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
        else:
            return min(c["low"] for c in recent[-10:])
    else:
        if swing_highs:
            return max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
        else:
            return max(c["high"] for c in recent[-10:])


def _compute_confluence_flags(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    h4_candles: List[Dict],
    direction: str,
    params: Optional[StrategyParams] = None,
) -> Tuple[Dict[str, bool], Dict[str, str], Tuple]:
    """
    Compute all confluence flags for a trading setup.
    
    This is the main entry point for confluence calculation,
    used by both backtests and live scanning.
    
    Returns:
        Tuple of (flags dict, notes dict, trade_levels tuple)
    """
    return compute_confluence(
        monthly_candles, weekly_candles, daily_candles, h4_candles, direction, params
    )


def compute_confluence(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    h4_candles: List[Dict],
    direction: str,
    params: Optional[StrategyParams] = None,
    historical_sr: Optional[Dict[str, List[Dict]]] = None,
) -> Tuple[Dict[str, bool], Dict[str, str], Tuple]:
    """
    Compute confluence flags for a given setup.
    
    Uses the same core logic as strategy.py but with parameterization.
    
    Args:
        monthly_candles: Monthly OHLCV data
        weekly_candles: Weekly OHLCV data
        daily_candles: Daily OHLCV data
        h4_candles: 4H OHLCV data
        direction: Trade direction ("bullish" or "bearish")
        params: Strategy parameters (uses defaults if None)
        historical_sr: Optional dict with 'monthly' and 'weekly' S/R levels from historical data
    
    Returns:
        Tuple of (flags dict, notes dict, trade_levels tuple)
    """
    if params is None:
        params = StrategyParams()
    
    price = daily_candles[-1]["close"] if daily_candles else float("nan")
    
    mn_trend = _infer_trend(monthly_candles) if monthly_candles else "mixed"
    wk_trend = _infer_trend(weekly_candles) if weekly_candles else "mixed"
    d_trend = _infer_trend(daily_candles) if daily_candles else "mixed"
    _, htf_note_text, htf_ok = _pick_direction_from_bias(mn_trend, wk_trend, d_trend)
    
    if params.use_htf_filter:
        loc_note, loc_ok = _location_context(
            monthly_candles, weekly_candles, daily_candles, price, direction, historical_sr
        )
    else:
        loc_note, loc_ok = "Location filter disabled", True
    
    if params.use_fib_filter:
        fib_note, fib_ok = _fib_context(weekly_candles, daily_candles, direction, price)
    else:
        fib_note, fib_ok = "Fib filter disabled", True
    
    if params.use_liquidity_filter:
        liq_note, liq_ok = _daily_liquidity_context(daily_candles, price)
    else:
        liq_note, liq_ok = "Liquidity filter disabled", True
    
    if params.use_structure_filter:
        struct_ok, struct_note = _structure_context(
            monthly_candles, weekly_candles, daily_candles, direction
        )
    else:
        struct_ok, struct_note = True, "Structure filter disabled"
    
    if params.use_confirmation_filter:
        conf_note, conf_ok = _h4_confirmation(h4_candles, direction, daily_candles)
    else:
        conf_note, conf_ok = "Confirmation filter disabled", True
    
    if params.use_atr_regime_filter:
        _, atr_percentile = _calculate_atr_percentile(daily_candles, period=14, lookback=100)
        atr_regime_ok = atr_percentile >= params.atr_min_percentile
        atr_regime_note = f"ATR Regime: {atr_percentile:.1f}th percentile ({'OK' if atr_regime_ok else 'Low volatility'})"
    else:
        atr_regime_ok, atr_regime_note = True, "ATR regime filter disabled"
    
    if params.use_pattern_filter:
        if direction == "bullish":
            pattern_ok, pattern_note = _detect_bullish_n_pattern(daily_candles, lookback=10)
        else:
            pattern_ok, pattern_note = _detect_bearish_v_pattern(daily_candles, lookback=10)
    else:
        pattern_ok, pattern_note = True, "Pattern filter disabled"
    
    if params.use_zscore_filter:
        zscore = _calculate_zscore(price, daily_candles, period=20)
        if direction == "bullish":
            zscore_valid = zscore < -1.0
            zscore_note = f"Z-Score: {zscore:.2f} ({'Valid <-1.0' if zscore_valid else 'Above -1.0, not ideal for long'})"
        else:
            zscore_valid = zscore > 1.0
            zscore_note = f"Z-Score: {zscore:.2f} ({'Valid >1.0' if zscore_valid else 'Below 1.0, not ideal for short'})"
    else:
        zscore_valid, zscore_note = True, "Z-score filter disabled"
    
    # Blueprint V2 enhancements
    if params.use_mitigated_sr:
        mitigated_sr_ok, mitigated_sr_note, _ = _detect_mitigated_sr(
            daily_candles, price, direction, params.sr_proximity_pct
        )
    else:
        mitigated_sr_ok, mitigated_sr_note = True, "Mitigated SR disabled"
    
    if params.use_structural_framework:
        framework_ok, framework_note, _ = _detect_structural_framework(daily_candles, direction)
    else:
        framework_ok, framework_note = True, "Framework disabled"
    
    if params.use_displacement_filter:
        displacement_ok, displacement_note = _detect_displacement(
            daily_candles, direction, params.displacement_atr_mult
        )
    else:
        displacement_ok, displacement_note = True, "Displacement disabled"
    
    if params.use_candle_rejection:
        rejection_ok, rejection_note = _detect_candle_rejection(h4_candles if h4_candles else daily_candles, direction)
    else:
        rejection_ok, rejection_note = True, "Candle rejection disabled"
    
    # Advanced quant filters
    if params.use_rsi_divergence:
        rsi_div_ok, rsi_div_note = _detect_rsi_divergence(daily_candles, direction, params.rsi_period)
    else:
        rsi_div_ok, rsi_div_note = True, "RSI divergence disabled"
    
    if params.use_momentum_filter:
        momentum_ok, momentum_note = _detect_momentum(daily_candles, direction, params.momentum_lookback)
    else:
        momentum_ok, momentum_note = True, "Momentum filter disabled"
    
    if params.use_mean_reversion:
        mean_rev_ok, mean_rev_note = _detect_bollinger_mean_reversion(
            daily_candles, direction, params.bollinger_period, params.bollinger_std
        )
    else:
        mean_rev_ok, mean_rev_note = True, "Mean reversion disabled"
    
    rr_note, rr_ok, entry, sl, tp1, tp2, tp3, tp4, tp5 = compute_trade_levels(
        daily_candles, direction, params, h4_candles
    )
    
    flags = {
        "htf_bias": htf_ok,
        "location": loc_ok,
        "fib": fib_ok,
        "liquidity": liq_ok,
        "structure": struct_ok,
        "confirmation": conf_ok,
        "rr": rr_ok,
        "atr_regime_ok": atr_regime_ok,
        "pattern_confirmed": pattern_ok,
        "zscore_valid": zscore_valid,
        "mitigated_sr": mitigated_sr_ok,
        "framework": framework_ok,
        "displacement": displacement_ok,
        "rejection": rejection_ok,
        "rsi_divergence": rsi_div_ok,
        "momentum": momentum_ok,
        "mean_reversion": mean_rev_ok,
    }
    
    notes = {
        "htf_bias": htf_note_text,
        "location": loc_note,
        "fib": fib_note,
        "liquidity": liq_note,
        "structure": struct_note,
        "confirmation": conf_note,
        "rr": rr_note,
        "atr_regime": atr_regime_note,
        "pattern": pattern_note,
        "zscore": zscore_note,
        "mitigated_sr": mitigated_sr_note,
        "framework": framework_note,
        "displacement": displacement_note,
        "rejection": rejection_note,
        "rsi_divergence": rsi_div_note,
        "momentum": momentum_note,
        "mean_reversion": mean_rev_note,
    }
    
    trade_levels = (entry, sl, tp1, tp2, tp3, tp4, tp5)
    return flags, notes, trade_levels


def compute_trade_levels(
    daily_candles: List[Dict],
    direction: str,
    params: Optional[StrategyParams] = None,
    h4_candles: Optional[List[Dict]] = None,
) -> Tuple[str, bool, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute entry, SL, and TP levels using parameterized logic.
    
    Args:
        daily_candles: Daily OHLCV data
        direction: Trade direction
        params: Strategy parameters
        h4_candles: 4H OHLCV data for tighter SL calculation
    
    Returns:
        Tuple of (note, is_valid, entry, sl, tp1, tp2, tp3, tp4, tp5)
    """
    if params is None:
        params = StrategyParams()
    
    if not daily_candles:
        return "R/R: no data.", False, None, None, None, None, None, None, None
    
    current = daily_candles[-1]["close"]
    atr = _atr(daily_candles, 14)
    
    if atr <= 0:
        return "R/R: ATR too small.", False, None, None, None, None, None, None, None
    
    leg = _find_last_swing_leg_for_fib(daily_candles, direction)
    
    sl_candles = h4_candles if h4_candles and len(h4_candles) >= 20 else daily_candles
    h4_lookback = 20
    structure_sl = _find_structure_sl(sl_candles, direction, lookback=h4_lookback)
    
    if leg:
        lo, hi = leg
        span = hi - lo
        if span > 0:
            if direction == "bullish":
                gp_mid = hi - span * 0.618
                entry = current if abs(current - gp_mid) < atr * 0.3 else gp_mid
                
                base_sl = lo - atr * 0.5
                if structure_sl is not None:
                    sl = min(base_sl, structure_sl - atr * 0.4)
                else:
                    sl = base_sl
                
                risk = entry - sl
                
                if risk > 0:
                    tp1 = entry + risk * params.atr_tp1_multiplier
                    tp2 = entry + risk * params.atr_tp2_multiplier
                    tp3 = entry + risk * params.atr_tp3_multiplier
                    tp4 = entry + risk * 2.5
                    tp5 = entry + risk * 3.5
                    
                    note = f"R/R: Entry near {entry:.5f}, SL at {sl:.5f}"
                    return note, True, entry, sl, tp1, tp2, tp3, tp4, tp5
            else:
                gp_mid = lo + span * 0.618
                entry = current if abs(current - gp_mid) < atr * 0.3 else gp_mid
                
                base_sl = hi + atr * 0.5
                if structure_sl is not None:
                    sl = max(base_sl, structure_sl + atr * 0.4)
                else:
                    sl = base_sl
                
                risk = sl - entry
                
                if risk > 0:
                    tp1 = entry - risk * params.atr_tp1_multiplier
                    tp2 = entry - risk * params.atr_tp2_multiplier
                    tp3 = entry - risk * params.atr_tp3_multiplier
                    tp4 = entry - risk * 2.5
                    tp5 = entry - risk * 3.5
                    
                    note = f"R/R: Entry near {entry:.5f}, SL at {sl:.5f}"
                    return note, True, entry, sl, tp1, tp2, tp3, tp4, tp5
    
    entry = current
    sl_mult = params.atr_sl_multiplier
    
    if direction == "bullish":
        if structure_sl is not None:
            sl = min(entry - atr * sl_mult, structure_sl - atr * 0.4)
        else:
            sl = entry - atr * sl_mult
        risk = entry - sl
        tp1 = entry + risk * params.atr_tp1_multiplier
        tp2 = entry + risk * params.atr_tp2_multiplier
        tp3 = entry + risk * params.atr_tp3_multiplier
        tp4 = entry + risk * 2.5
        tp5 = entry + risk * 3.5
    else:
        if structure_sl is not None:
            sl = max(entry + atr * sl_mult, structure_sl + atr * 0.4)
        else:
            sl = entry + atr * sl_mult
        risk = sl - entry
        tp1 = entry - risk * params.atr_tp1_multiplier
        tp2 = entry - risk * params.atr_tp2_multiplier
        tp3 = entry - risk * params.atr_tp3_multiplier
        tp4 = entry - risk * 2.5
        tp5 = entry - risk * 3.5
    
    note = f"R/R: ATR+structure levels"
    return note, True, entry, sl, tp1, tp2, tp3, tp4, tp5


def generate_signals(
    candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
    monthly_candles: Optional[List[Dict]] = None,
    weekly_candles: Optional[List[Dict]] = None,
    h4_candles: Optional[List[Dict]] = None,
) -> List[Signal]:
    """
    Generate trading signals from historical candles.
    
    This function walks through candles sequentially (no look-ahead bias)
    and generates signals based on the Blueprint strategy rules.
    
    Args:
        candles: Daily OHLCV candles (oldest to newest)
        symbol: Asset symbol
        params: Strategy parameters
        monthly_candles: Optional monthly data (derived from daily if not provided)
        weekly_candles: Optional weekly data (derived from daily if not provided)
        h4_candles: Optional 4H data (uses daily for confirmation if not provided)
    
    Returns:
        List of Signal objects
    """
    if params is None:
        params = StrategyParams()
    
    if len(candles) < 50:
        return []
    
    signals = []
    
    for i in range(50, len(candles)):
        try:
            daily_slice = candles[:i+1]
            current_candle = candles[i]
            current_dt = _get_candle_datetime(current_candle)
            
            if current_dt is not None:
                weekly_slice = _slice_htf_by_timestamp(weekly_candles, current_dt)
                monthly_slice = _slice_htf_by_timestamp(monthly_candles, current_dt)
                h4_slice = _slice_htf_by_timestamp(h4_candles, current_dt)
            else:
                weekly_slice = weekly_candles[:i//5+1] if weekly_candles else None
                monthly_slice = monthly_candles[:i//20+1] if monthly_candles else None
                h4_slice = h4_candles[:i*6+1] if h4_candles else None
            
            mn_trend = _infer_trend(monthly_slice) if monthly_slice else _infer_trend(daily_slice[-60:])
            wk_trend = _infer_trend(weekly_slice) if weekly_slice else _infer_trend(daily_slice[-20:])
            d_trend = _infer_trend(daily_slice[-10:])
            
            direction, _, _ = _pick_direction_from_bias(mn_trend, wk_trend, d_trend)
            
            flags, notes, trade_levels = compute_confluence(
                monthly_slice or [],
                weekly_slice or [],
                daily_slice,
                h4_slice or daily_slice[-20:],
                direction,
                params,
            )
        except Exception:
            continue
        
        entry, sl, tp1, tp2, tp3, tp4, tp5 = trade_levels
        
        confluence_score = sum(1 for v in flags.values() if v)
        
        quality_factors = sum([
            flags.get("location", False),
            flags.get("fib", False),
            flags.get("liquidity", False),
            flags.get("structure", False),
            flags.get("htf_bias", False),
        ])
        
        # Apply volatile asset boost for high-volatility instruments BEFORE threshold check
        # This allows volatile assets (XAUUSD, NAS100USD, GBPJPY, BTCUSD) to more easily pass thresholds
        boosted_confluence, boosted_quality = apply_volatile_asset_boost(
            symbol,
            confluence_score,
            quality_factors,
            params.volatile_asset_boost
        )
        
        has_rr = flags.get("rr", False)
        has_confirmation = flags.get("confirmation", False)
        
        is_active = False
        is_watching = False
        
        # Use boosted scores for threshold comparison
        if boosted_confluence >= params.min_confluence and boosted_quality >= params.min_quality_factors:
            if params.require_rr_for_active and not has_rr:
                is_watching = True
            elif params.require_confirmation_for_active and not has_confirmation:
                is_watching = True
            else:
                is_active = True
        elif boosted_confluence >= params.min_confluence - 1:
            is_watching = True
        
        if is_active or is_watching:
            candle = candles[i]
            timestamp = candle.get("time") or candle.get("timestamp") or candle.get("date")
            
            signal = Signal(
                symbol=symbol,
                direction=direction,
                bar_index=i,
                timestamp=timestamp,
                confluence_score=confluence_score,
                quality_factors=quality_factors,
                entry=entry,
                stop_loss=sl,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                tp4=tp4,
                tp5=tp5,
                is_active=is_active,
                is_watching=is_watching,
                flags=flags,
                notes=notes,
            )
            signals.append(signal)
    
    return signals


def _validate_and_find_entry(
    candles: List[Dict],
    signal_bar: int,
    theoretical_entry: float,
    direction: str,
    max_wait_bars: int = 5,
) -> Tuple[Optional[int], float, bool]:
    """
    Validate entry price against actual candle data and find real entry point.
    
    For limit orders (golden pocket entries), we need to wait for price to 
    actually reach our entry level. This prevents look-ahead bias.
    
    IMPORTANT: Limit order fill logic:
    - Bullish limit order: Placed BELOW current price, fills when price drops TO or THROUGH it
      -> Entry is valid if theoretical_entry >= candle_low (price reached down to our level)
    - Bearish limit order: Placed ABOVE current price, fills when price rises TO or THROUGH it
      -> Entry is valid if theoretical_entry <= candle_high (price reached up to our level)
    
    If price never reaches the entry level, the trade is SKIPPED (not adjusted).
    
    Args:
        candles: OHLCV candles
        signal_bar: Bar index where signal was generated
        theoretical_entry: Calculated entry price from strategy
        direction: 'bullish' or 'bearish'
        max_wait_bars: Maximum bars to wait for entry (default 5)
    
    Returns:
        Tuple of (actual_entry_bar, actual_entry_price, was_adjusted)
        Returns (None, 0, False) if entry never achieved
    """
    for bar_offset in range(max_wait_bars + 1):
        check_bar = signal_bar + bar_offset
        if check_bar >= len(candles):
            break
            
        candle = candles[check_bar]
        high = candle["high"]
        low = candle["low"]
        
        if direction == "bullish":
            if low <= theoretical_entry <= high:
                return check_bar, theoretical_entry, False
            elif theoretical_entry > high:
                pass
        else:
            if low <= theoretical_entry <= high:
                return check_bar, theoretical_entry, False
            elif theoretical_entry < low:
                pass
    
    return None, 0.0, False


def simulate_trades(
    candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
    monthly_candles: Optional[List[Dict]] = None,
    weekly_candles: Optional[List[Dict]] = None,
    h4_candles: Optional[List[Dict]] = None,
    include_transaction_costs: bool = True,
) -> List[Trade]:
    """
    Simulate trades through historical candles using the Blueprint strategy.
    
    This is a walk-forward simulation with no look-ahead bias.
    Uses the same logic as live trading but runs through historical data.
    
    IMPORTANT: Supports multiple concurrent trades up to params.max_open_trades.
    Entry prices are validated against actual candle data.
    If the theoretical entry price is not available on the signal bar,
    we wait up to 5 bars for price to reach the entry level.
    
    Transaction costs (spread + slippage) are deducted from each trade
    when include_transaction_costs=True to produce realistic backtest results.
    
    Args:
        candles: Daily OHLCV candles (oldest to newest)
        symbol: Asset symbol
        params: Strategy parameters
        monthly_candles: Optional monthly data
        weekly_candles: Optional weekly data
        h4_candles: Optional 4H data
        include_transaction_costs: Whether to include spread/slippage costs (default True)
    
    Returns:
        List of completed Trade objects
    """
    if params is None:
        params = StrategyParams()
    
    transaction_cost_pips = 0.0
    pip_value = 0.0001
    
    if include_transaction_costs:
        try:
            from params.params_loader import get_transaction_costs
            from tradr.risk.position_sizing import get_contract_specs
            
            spread_pips, slippage_pips, _ = get_transaction_costs(symbol)
            transaction_cost_pips = spread_pips + slippage_pips
            
            specs = get_contract_specs(symbol)
            pip_value = specs.get("pip_value", 0.0001)
        except Exception:
            transaction_cost_pips = 2.5
            pip_value = 0.0001
    
    transaction_cost_price = transaction_cost_pips * pip_value
    
    signals = generate_signals(
        candles, symbol, params,
        monthly_candles, weekly_candles, h4_candles
    )
    
    active_signals = [s for s in signals if s.is_active]
    
    TP1_CLOSE_PCT = params.tp1_close_pct
    TP2_CLOSE_PCT = params.tp2_close_pct
    TP3_CLOSE_PCT = params.tp3_close_pct
    TP4_CLOSE_PCT = params.tp4_close_pct
    TP5_CLOSE_PCT = params.tp5_close_pct
    
    signal_to_pending_entry = {}
    for sig in active_signals:
        if sig.entry is None or sig.stop_loss is None or sig.tp1 is None:
            continue
        risk = abs(sig.entry - sig.stop_loss)
        if risk <= 0:
            continue
        signal_to_pending_entry[id(sig)] = {
            "signal": sig,
            "wait_until_bar": sig.bar_index + 5,
        }
    
    trades = []
    open_trades = []
    entered_signal_ids = set()
    
    for bar_idx in range(len(candles)):
        c = candles[bar_idx]
        high = c["high"]
        low = c["low"]
        bar_timestamp = c.get("time") or c.get("timestamp") or c.get("date")
        
        trades_to_close = []
        for ot in open_trades:
            direction = ot["direction"]
            entry_price = ot["entry_price"]
            risk = ot["risk"]
            trailing_sl = ot["trailing_sl"]
            tp1 = ot["tp1"]
            tp2 = ot["tp2"]
            tp3 = ot["tp3"]
            tp4 = ot["tp4"]
            tp5 = ot["tp5"]
            tp1_hit = ot["tp1_hit"]
            tp2_hit = ot["tp2_hit"]
            tp3_hit = ot["tp3_hit"]
            tp4_hit = ot["tp4_hit"]
            tp5_hit = ot["tp5_hit"]
            
            tp1_rr = ot["tp1_rr"]
            tp2_rr = ot["tp2_rr"]
            tp3_rr = ot["tp3_rr"]
            tp4_rr = ot["tp4_rr"]
            tp5_rr = ot["tp5_rr"]
            
            trade_closed = False
            rr = 0.0
            is_winner = False
            exit_reason = ""
            reward = 0.0
            
            if direction == "bullish":
                if low <= trailing_sl:
                    trail_rr = (trailing_sl - entry_price) / risk
                    if tp4_hit:
                        remaining_pct = TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + TP4_CLOSE_PCT * tp4_rr + remaining_pct * trail_rr
                        exit_reason = "TP4+Trail"
                        is_winner = True
                    elif tp3_hit:
                        remaining_pct = TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + remaining_pct * trail_rr
                        exit_reason = "TP3+Trail"
                        is_winner = True
                    elif tp2_hit:
                        remaining_pct = TP3_CLOSE_PCT + TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + remaining_pct * trail_rr
                        exit_reason = "TP2+Trail"
                        is_winner = rr >= 0
                    elif tp1_hit:
                        remaining_pct = TP2_CLOSE_PCT + TP3_CLOSE_PCT + TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + remaining_pct * trail_rr
                        exit_reason = "TP1+Trail"
                        is_winner = rr >= 0
                    else:
                        rr = -1.0
                        exit_reason = "SL"
                        is_winner = False
                    reward = rr * risk
                    trade_closed = True
                
                if not trade_closed and tp1 is not None and high >= tp1 and not tp1_hit:
                    ot["tp1_hit"] = True
                    tp1_hit = True
                    # Delay trailing activation until trail_activation_r is reached
                    if tp1_rr >= params.trail_activation_r:
                        ot["trailing_sl"] = entry_price
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp1_hit and tp2 is not None and high >= tp2 and not tp2_hit:
                    ot["tp2_hit"] = True
                    tp2_hit = True
                    # Only activate trailing if we've reached trail_activation_r
                    if tp2_rr >= params.trail_activation_r and tp1 is not None:
                        ot["trailing_sl"] = tp1 + 0.5 * risk
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp2_hit and tp3 is not None and high >= tp3 and not tp3_hit:
                    ot["tp3_hit"] = True
                    tp3_hit = True
                    # Only activate trailing if we've reached trail_activation_r
                    if tp3_rr >= params.trail_activation_r and tp2 is not None:
                        ot["trailing_sl"] = tp2 + 0.5 * risk
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp3_hit and tp4 is not None and high >= tp4 and not tp4_hit:
                    ot["tp4_hit"] = True
                    tp4_hit = True
                    if tp3 is not None:
                        ot["trailing_sl"] = tp3 + 0.5 * risk
                
                if not trade_closed and tp4_hit and tp5 is not None and high >= tp5 and not tp5_hit:
                    ot["tp5_hit"] = True
                    rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + TP4_CLOSE_PCT * tp4_rr + TP5_CLOSE_PCT * tp5_rr
                    reward = rr * risk
                    exit_reason = "TP5"
                    is_winner = True
                    trade_closed = True
            else:
                if high >= trailing_sl:
                    trail_rr = (entry_price - trailing_sl) / risk
                    if tp4_hit:
                        remaining_pct = TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + TP4_CLOSE_PCT * tp4_rr + remaining_pct * trail_rr
                        exit_reason = "TP4+Trail"
                        is_winner = True
                    elif tp3_hit:
                        remaining_pct = TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + remaining_pct * trail_rr
                        exit_reason = "TP3+Trail"
                        is_winner = True
                    elif tp2_hit:
                        remaining_pct = TP3_CLOSE_PCT + TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + remaining_pct * trail_rr
                        exit_reason = "TP2+Trail"
                        is_winner = rr >= 0
                    elif tp1_hit:
                        remaining_pct = TP2_CLOSE_PCT + TP3_CLOSE_PCT + TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + remaining_pct * trail_rr
                        exit_reason = "TP1+Trail"
                        is_winner = rr >= 0
                    else:
                        rr = -1.0
                        exit_reason = "SL"
                        is_winner = False
                    reward = rr * risk
                    trade_closed = True
                
                if not trade_closed and tp1 is not None and low <= tp1 and not tp1_hit:
                    ot["tp1_hit"] = True
                    tp1_hit = True
                    # Delay trailing activation until trail_activation_r is reached
                    if tp1_rr >= params.trail_activation_r:
                        ot["trailing_sl"] = entry_price
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp1_hit and tp2 is not None and low <= tp2 and not tp2_hit:
                    ot["tp2_hit"] = True
                    tp2_hit = True
                    # Only activate trailing if we've reached trail_activation_r
                    if tp2_rr >= params.trail_activation_r and tp1 is not None:
                        ot["trailing_sl"] = tp1 - 0.5 * risk
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp2_hit and tp3 is not None and low <= tp3 and not tp3_hit:
                    ot["tp3_hit"] = True
                    tp3_hit = True
                    # Only activate trailing if we've reached trail_activation_r
                    if tp3_rr >= params.trail_activation_r and tp2 is not None:
                        ot["trailing_sl"] = tp2 - 0.5 * risk
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp3_hit and tp4 is not None and low <= tp4 and not tp4_hit:
                    ot["tp4_hit"] = True
                    tp4_hit = True
                    if tp3 is not None:
                        ot["trailing_sl"] = tp3 - 0.5 * risk
                
                if not trade_closed and tp4_hit and tp5 is not None and low <= tp5 and not tp5_hit:
                    ot["tp5_hit"] = True
                    rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + TP4_CLOSE_PCT * tp4_rr + TP5_CLOSE_PCT * tp5_rr
                    reward = rr * risk
                    exit_reason = "TP5"
                    is_winner = True
                    trade_closed = True
            
            if trade_closed:
                cost_r = ot.get("transaction_cost_r", 0.0)
                adjusted_rr = rr - cost_r
                adjusted_reward = adjusted_rr * risk
                adjusted_is_winner = is_winner and adjusted_rr >= 0
                
                trade = Trade(
                    symbol=symbol,
                    direction=direction,
                    entry_date=ot["entry_timestamp"],
                    exit_date=bar_timestamp,
                    entry_price=entry_price,
                    exit_price=entry_price + adjusted_reward if direction == "bullish" else entry_price - adjusted_reward,
                    stop_loss=ot["sl"],
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    tp4=tp4,
                    tp5=tp5,
                    risk=risk,
                    reward=adjusted_reward,
                    rr=adjusted_rr,
                    is_winner=adjusted_is_winner,
                    exit_reason=exit_reason,
                    confluence_score=ot["confluence_score"],
                )
                trades.append(trade)
                trades_to_close.append(ot)
        
        for ot in trades_to_close:
            open_trades.remove(ot)
        
        if len(open_trades) < params.max_open_trades:
            for sig_id, pending in list(signal_to_pending_entry.items()):
                if sig_id in entered_signal_ids:
                    continue
                if len(open_trades) >= params.max_open_trades:
                    break
                
                sig = pending["signal"]
                if bar_idx < sig.bar_index:
                    continue
                if bar_idx > pending["wait_until_bar"]:
                    entered_signal_ids.add(sig_id)
                    continue
                
                theoretical_entry = sig.entry
                direction = sig.direction
                
                if direction == "bullish":
                    if low <= theoretical_entry <= high:
                        entry_price = theoretical_entry
                    else:
                        continue
                else:
                    if low <= theoretical_entry <= high:
                        entry_price = theoretical_entry
                    else:
                        continue
                
                sl = sig.stop_loss
                tp1 = sig.tp1
                tp2 = sig.tp2
                tp3 = sig.tp3
                tp4 = sig.tp4
                tp5 = sig.tp5
                risk = abs(entry_price - sl)
                
                if risk <= 0:
                    entered_signal_ids.add(sig_id)
                    continue
                
                # Hard volatility filter - skip trades in low volatility regimes
                # In December, apply stricter threshold
                if params.use_atr_regime_filter:
                    current_date = _get_candle_datetime(candles[bar_idx])
                    passes_vol, _ = check_volatility_filter(
                        candles[:bar_idx+1], 
                        params.atr_min_percentile,
                        current_date=current_date,
                        december_atr_multiplier=params.december_atr_multiplier
                    )
                    if not passes_vol:
                        entered_signal_ids.add(sig_id)
                        continue
                
                if params.ml_min_prob > 0:
                    features = extract_ml_features(candles[:bar_idx+1], sig.flags, direction, params)
                    if features:
                        should_trade, prob = apply_ml_filter(features, params.ml_min_prob)
                        if not should_trade:
                            entered_signal_ids.add(sig_id)
                            continue
                
                tp1_rr = (tp1 - entry_price) / risk if tp1 and direction == "bullish" else ((entry_price - tp1) / risk if tp1 else 0)
                tp2_rr = (tp2 - entry_price) / risk if tp2 and direction == "bullish" else ((entry_price - tp2) / risk if tp2 else 0)
                tp3_rr = (tp3 - entry_price) / risk if tp3 and direction == "bullish" else ((entry_price - tp3) / risk if tp3 else 0)
                tp4_rr = (tp4 - entry_price) / risk if tp4 and direction == "bullish" else ((entry_price - tp4) / risk if tp4 else 0)
                tp5_rr = (tp5 - entry_price) / risk if tp5 and direction == "bullish" else ((entry_price - tp5) / risk if tp5 else 0)
                
                cost_as_r = transaction_cost_price / risk if risk > 0 else 0.0
                
                # Apply volatile asset boost for high-volatility instruments
                boosted_confluence, _ = apply_volatile_asset_boost(
                    symbol,
                    sig.confluence_score,
                    sig.quality_factors,
                    params.volatile_asset_boost
                )
                
                open_trades.append({
                    "signal_id": sig_id,
                    "direction": direction,
                    "entry_bar": bar_idx,
                    "entry_price": entry_price,
                    "entry_timestamp": bar_timestamp,
                    "sl": sl,
                    "trailing_sl": sl,
                    "trailing_activated": False,  # Flag to track when trail_activation_r threshold is reached
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp3": tp3,
                    "tp4": tp4,
                    "tp5": tp5,
                    "risk": risk,
                    "tp1_hit": False,
                    "tp2_hit": False,
                    "tp3_hit": False,
                    "tp4_hit": False,
                    "tp5_hit": False,
                    "tp1_rr": tp1_rr,
                    "tp2_rr": tp2_rr,
                    "tp3_rr": tp3_rr,
                    "tp4_rr": tp4_rr,
                    "tp5_rr": tp5_rr,
                    "confluence_score": boosted_confluence,
                    "transaction_cost_r": cost_as_r,
                })
                entered_signal_ids.add(sig_id)
    
    return trades


def get_default_params() -> StrategyParams:
    """Get default strategy parameters - optimized for trade generation."""
    return StrategyParams(
        min_confluence=2,
        min_quality_factors=1,
        require_confirmation_for_active=False,
        require_rr_for_active=False,
        atr_sl_multiplier=1.5,
        atr_tp1_multiplier=0.6,
        atr_tp2_multiplier=1.2,
        atr_tp3_multiplier=2.0,
        atr_tp4_multiplier=3.0,
        atr_tp5_multiplier=4.0,
    )


def get_aggressive_params() -> StrategyParams:
    """Get aggressive (more trades) strategy parameters."""
    return StrategyParams(
        min_confluence=1,
        min_quality_factors=0,
        require_confirmation_for_active=False,
        require_rr_for_active=False,
        atr_sl_multiplier=1.2,
        atr_tp1_multiplier=1.0,
        atr_tp2_multiplier=2.0,
        atr_tp3_multiplier=3.0,
        atr_tp4_multiplier=4.0,
        atr_tp5_multiplier=5.0,
    )


def get_conservative_params() -> StrategyParams:
    """Get conservative (higher quality) strategy parameters."""
    return StrategyParams(
        min_confluence=4,
        min_quality_factors=2,
        require_htf_alignment=True,
        require_confirmation_for_active=True,
        require_rr_for_active=True,
        atr_sl_multiplier=1.8,
        atr_tp1_multiplier=1.0,
        atr_tp2_multiplier=2.0,
        atr_tp3_multiplier=3.0,
        atr_tp4_multiplier=4.0,
        atr_tp5_multiplier=5.0,
    )


def extract_ml_features(
    candles: List[Dict],
    flags: Dict[str, bool],
    direction: str,
    params: Optional[StrategyParams] = None,
) -> Dict[str, float]:
    """
    Extract ML features for trade filtering.
    
    Args:
        candles: List of OHLCV candle dictionaries
        flags: Confluence flags from compute_confluence
        direction: Trade direction ("bullish" or "bearish")
        params: Strategy parameters
    
    Returns:
        Dict with ML features for model input
    """
    if params is None:
        params = StrategyParams()
    
    if not candles or len(candles) < 20:
        return {}
    
    price = candles[-1].get("close", 0)
    
    z_score = _calculate_zscore(price, candles, period=20)
    
    closes = [c["close"] for c in candles[-params.bollinger_period:] if c.get("close")]
    if len(closes) >= params.bollinger_period:
        mean = sum(closes) / len(closes)
        variance = sum((x - mean) ** 2 for x in closes) / len(closes)
        std = variance ** 0.5
        upper_band = mean + params.bollinger_std * std
        lower_band = mean - params.bollinger_std * std
        band_range = upper_band - lower_band
        if band_range > 0:
            bollinger_distance = (price - lower_band) / band_range
        else:
            bollinger_distance = 0.5
    else:
        bollinger_distance = 0.5
    
    rsi_value = _calculate_rsi(candles, params.rsi_period)
    
    _, atr_percentile = _calculate_atr_percentile(candles, period=14, lookback=100)
    
    momentum_lookback = params.momentum_lookback
    if len(candles) >= momentum_lookback + 1:
        current_close = candles[-1].get("close", 0)
        past_close = candles[-(momentum_lookback + 1)].get("close", 1)
        momentum_roc = ((current_close - past_close) / past_close * 100) if past_close > 0 else 0
    else:
        momentum_roc = 0
    
    features = {
        "htf_aligned": 1 if flags.get("htf_aligned", False) else 0,
        "location_ok": 1 if flags.get("location_ok", False) else 0,
        "fib_ok": 1 if flags.get("fib_ok", False) else 0,
        "structure_ok": 1 if flags.get("structure_ok", False) else 0,
        "liquidity_ok": 1 if flags.get("liquidity_ok", False) else 0,
        "confirmation_ok": 1 if flags.get("confirmation_ok", False) else 0,
        "atr_regime_ok": 1 if flags.get("atr_regime_ok", False) else 0,
        "z_score": z_score,
        "bollinger_distance": bollinger_distance,
        "rsi_value": rsi_value,
        "atr_percentile": atr_percentile,
        "momentum_roc": momentum_roc,
        "direction_bullish": 1 if direction == "bullish" else 0,
    }
    
    return features


def apply_ml_filter(
    features: Dict[str, float],
    min_prob: float = 0.6,
) -> Tuple[bool, float]:
    """
    Apply ML model to filter trades.
    
    Args:
        features: Dict of features from extract_ml_features
        min_prob: Minimum probability threshold for trade acceptance
    
    Returns:
        Tuple of (should_trade, probability)
    """
    import os
    
    model_path = "models/best_rf.joblib"
    
    if not os.path.exists(model_path):
        return (True, 1.0)
    
    try:
        import joblib
        model = joblib.load(model_path)
        
        feature_order = [
            "htf_aligned", "location_ok", "fib_ok", "structure_ok",
            "liquidity_ok", "confirmation_ok", "atr_regime_ok",
            "z_score", "bollinger_distance", "rsi_value",
            "atr_percentile", "momentum_roc", "direction_bullish"
        ]
        
        feature_values = [[features.get(f, 0) for f in feature_order]]
        
        probas = model.predict_proba(feature_values)
        prob_profitable = probas[0][1] if len(probas[0]) > 1 else probas[0][0]
        
        should_trade = prob_profitable >= min_prob
        return (should_trade, float(prob_profitable))
        
    except Exception as e:
        return (True, 1.0)


def check_volatility_filter(
    candles: List[Dict],
    atr_min_percentile: float = 60.0,
    current_date: Optional[datetime] = None,
    december_atr_multiplier: float = 1.0,
) -> Tuple[bool, float]:
    """
    Check if current volatility is above minimum threshold.
    In December, applies december_atr_multiplier to be more strict.
    """
    _, atr_percentile = _calculate_atr_percentile(candles, period=14, lookback=100)
    
    effective_threshold = atr_min_percentile
    if current_date and current_date.month == 12:
        effective_threshold = min(95.0, atr_min_percentile * december_atr_multiplier)
    
    passes_filter = atr_percentile >= effective_threshold
    return (passes_filter, atr_percentile)


VOLATILE_ASSETS = ["XAU_USD", "XAUUSD", "NAS100_USD", "NAS100USD", "GBP_JPY", "GBPJPY", "BTC_USD", "BTCUSD"]

def apply_volatile_asset_boost(
    symbol: str,
    confluence_score: int,
    quality_factors: int,
    volatile_asset_boost: float = 1.0,
) -> Tuple[int, int]:
    """
    Apply boost to confluence/quality scores for volatile assets.
    These assets (XAUUSD, NAS100USD, GBPJPY, BTCUSD) have potential for bigger R trades.
    """
    normalized_symbol = symbol.replace("_", "").upper()
    is_volatile = any(v.replace("_", "").upper() == normalized_symbol for v in VOLATILE_ASSETS)
    
    if is_volatile and volatile_asset_boost > 1.0:
        boosted_confluence = int(confluence_score * volatile_asset_boost)
        boosted_quality = int(quality_factors * volatile_asset_boost)
        return (boosted_confluence, boosted_quality)
    
    return (confluence_score, quality_factors)
