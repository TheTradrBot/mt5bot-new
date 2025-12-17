# indicators.py
"""
Technical indicators used by Blueprint Trader AI:
- EMA
- RSI
- Bollinger Bands
- ADX with slope detection
"""

from typing import List, Optional, Tuple, Dict


def ema(values: List[float], period: int) -> Optional[float]:
    """
    Return the latest EMA value for the given period.
    values: oldest -> newest
    """
    if len(values) < period:
        return None

    k = 2 / (period + 1)
    ema_val = sum(values[:period]) / period  # simple MA start
    for price in values[period:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val


def rsi(values: List[float], period: int = 14) -> Optional[float]:
    """
    Classic RSI calculation, returns latest RSI.
    values: oldest -> newest
    """
    if len(values) <= period:
        return None

    gains = []
    losses = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)

    # initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    values: List[float],
    period: int = 20,
    std_mult: float = 2.0
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate Bollinger Bands (upper, middle, lower).
    
    Args:
        values: List of prices (oldest -> newest)
        period: Moving average period (default 20)
        std_mult: Standard deviation multiplier (default 2.0)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band) or None if insufficient data
    """
    if len(values) < period:
        return None
    
    recent = values[-period:]
    middle = sum(recent) / period
    
    variance = sum((x - middle) ** 2 for x in recent) / period
    std = variance ** 0.5
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    return (upper, middle, lower)


def calculate_adx_with_slope(
    candles: List[Dict],
    period: int = 14,
    slope_lookback: int = 3
) -> Tuple[float, float, float, float, bool]:
    """
    Calculate ADX with directional indicators and slope detection.
    
    Returns:
        Tuple of (adx, plus_di, minus_di, adx_slope, is_slope_rising)
        - adx: ADX value (0-100)
        - plus_di: +DI value
        - minus_di: -DI value
        - adx_slope: Rate of change of ADX over slope_lookback bars
        - is_slope_rising: True if ADX is rising (trend strengthening)
    """
    if len(candles) < period * 2 + slope_lookback:
        return 0.0, 0.0, 0.0, 0.0, False
    
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
        return 0.0, 0.0, 0.0, 0.0, False
    
    smoothed_plus_dm = sum(plus_dm[:period])
    smoothed_minus_dm = sum(minus_dm[:period])
    smoothed_tr = sum(tr_values[:period])
    
    dx_values = []
    plus_di_values = []
    minus_di_values = []
    
    for i in range(period, len(tr_values)):
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr_values[i]
        
        if smoothed_tr == 0:
            continue
            
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        plus_di_values.append(plus_di)
        minus_di_values.append(minus_di)
        
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0
        else:
            dx = 100 * abs(plus_di - minus_di) / di_sum
        dx_values.append(dx)
    
    if not dx_values:
        return 0.0, 0.0, 0.0, 0.0, False
    
    if len(dx_values) < period:
        adx = sum(dx_values) / len(dx_values)
        adx_values = [adx]
    else:
        adx = sum(dx_values[:period]) / period
        adx_values = [adx]
        for i in range(period, len(dx_values)):
            adx = ((adx * (period - 1)) + dx_values[i]) / period
            adx_values.append(adx)
    
    current_adx = adx_values[-1] if adx_values else 0.0
    current_plus_di = plus_di_values[-1] if plus_di_values else 0.0
    current_minus_di = minus_di_values[-1] if minus_di_values else 0.0
    
    if len(adx_values) >= slope_lookback:
        adx_slope = current_adx - adx_values[-slope_lookback]
        is_slope_rising = adx_slope > 0
    else:
        adx_slope = 0.0
        is_slope_rising = False
    
    return current_adx, current_plus_di, current_minus_di, adx_slope, is_slope_rising


def check_di_crossover(
    candles: List[Dict],
    period: int = 14,
    lookback: int = 3
) -> Tuple[bool, bool, str]:
    """
    Check for +DI/-DI crossover within the lookback period.
    
    Returns:
        Tuple of (bullish_crossover, bearish_crossover, description)
        - bullish_crossover: True if +DI crossed above -DI
        - bearish_crossover: True if -DI crossed above +DI
    """
    if len(candles) < period * 2 + lookback + 5:
        return False, False, "Insufficient data for DI crossover"
    
    di_history = []
    for i in range(lookback + 2):
        end_idx = len(candles) - lookback + i
        if end_idx < period * 2:
            continue
        slice_candles = candles[:end_idx]
        adx, plus_di, minus_di, _, _ = calculate_adx_with_slope(slice_candles, period)
        di_history.append((plus_di, minus_di))
    
    if len(di_history) < 2:
        return False, False, "Insufficient DI history"
    
    for i in range(1, len(di_history)):
        prev_plus, prev_minus = di_history[i-1]
        curr_plus, curr_minus = di_history[i]
        
        if prev_plus <= prev_minus and curr_plus > curr_minus:
            return True, False, f"Bullish DI crossover: +DI({curr_plus:.1f}) crossed above -DI({curr_minus:.1f})"
        
        if prev_plus >= prev_minus and curr_plus < curr_minus:
            return False, True, f"Bearish DI crossover: -DI({curr_minus:.1f}) crossed above +DI({curr_plus:.1f})"
    
    return False, False, "No recent DI crossover"
