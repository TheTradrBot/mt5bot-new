# Trading Strategy Guide

**Last Updated**: 2025-12-28  
**Strategy**: 7-Pillar Confluence System with ADX Regime Detection

---

## Table of Contents
1. [Strategy Overview](#strategy-overview)
2. [7 Confluence Pillars](#7-confluence-pillars)
3. [ADX Regime Detection](#adx-regime-detection)
4. [Entry Rules](#entry-rules)
5. [Exit Management](#exit-management)
6. [Risk Management](#risk-management)
7. [Current Parameters](#current-parameters)

---

## Strategy Overview

The bot uses a **multi-timeframe confluence system** that combines 7 independent signals to identify high-probability setups. Each pillar votes on trade direction and quality, with entries requiring a minimum confluence score.

### Key Principles
- **Confluence over single indicators**: Requires multiple confirmations
- **Regime-adaptive**: Different rules for trending vs ranging markets
- **Risk-first approach**: Every trade pre-validated for R:R and FTMO limits
- **Multi-timeframe**: D1 signals confirmed by H4, W1, MN context

---

## 7 Confluence Pillars

### 1. Trend Alignment (Daily → Weekly → Monthly)
**Weight**: 2 points  
**Logic**: Entry must align with higher timeframe trend direction

```python
# Weekly trend trumps daily, monthly trumps weekly
weekly_trend = _infer_trend(weekly_candles)
monthly_trend = _infer_trend(monthly_candles)

if direction != weekly_trend or direction != monthly_trend:
    pillar_score -= 2  # Reject counter-trend
```

### 2. Support/Resistance Confluence
**Weight**: 1 point  
**Logic**: Price near key S/R level

```python
# Check if entry near S/R level (within 0.5% tolerance)
if abs(entry_price - sr_level) / entry_price < 0.005:
    pillar_score += 1
```

### 3. Fibonacci Zone Alignment
**Weight**: 1 point  
**Logic**: Entry at Fibonacci retracement level (38.2%, 50%, 61.8%)

```python
# Calculate Fib levels from recent swing
fib_levels = [0.382, 0.5, 0.618]
for level in fib_levels:
    if abs(entry_price - fib_price) < fib_tolerance:
        pillar_score += 1
```

### 4. RSI Divergence
**Weight**: 1 point  
**Logic**: Price makes new high/low but RSI doesn't (reversal signal)

```python
# Bullish divergence: price lower low, RSI higher low
if price_trend == "down" and rsi_trend == "up":
    pillar_score += 1
```

### 5. ADX Trend Strength
**Weight**: 1 point  
**Logic**: ADX > threshold confirms strong trend

```python
adx_value = calculate_adx(daily_candles, period=14)
if adx_value >= params['adx_trend_threshold']:
    pillar_score += 1
```

### 6. ATR Volatility Filter
**Weight**: 1 point  
**Logic**: Current ATR above minimum percentile (avoid dead markets)

```python
current_atr = calculate_atr(daily_candles, period=14)
atr_percentile = get_atr_percentile(current_atr, historical_atr)

if atr_percentile >= params['atr_min_percentile']:
    pillar_score += 1
```

### 7. Candlestick Pattern
**Weight**: 1 point  
**Logic**: Bullish/bearish engulfing, pin bar, inside bar

```python
pattern = detect_candlestick_pattern(daily_candles[-3:])
if pattern in ['engulfing', 'pin_bar'] and pattern_direction == direction:
    pillar_score += 1
```

---

## ADX Regime Detection

The strategy adapts based on market regime:

### Regime Classification
```python
adx = calculate_adx(daily_candles, period=14)

if adx >= params['adx_trend_threshold']:
    regime = "TREND"       # Momentum following
elif adx <= params['adx_range_threshold']:
    regime = "RANGE"       # Mean reversion
else:
    regime = "TRANSITION"  # No trading
```

### Regime-Specific Rules

| Regime | Min Confluence | RSI Filter | ATR Filter | SL Distance |
|--------|----------------|------------|------------|-------------|
| **TREND** | 4 | Disabled | Strict | 1.5× ATR |
| **RANGE** | 4 | Enabled (oversold/overbought) | Relaxed | 1.0× ATR |
| **TRANSITION** | No entries | - | - | - |

---

## Entry Rules

### Minimum Requirements
1. **Confluence Score** ≥ 4 pillars
2. **Quality Factors** ≥ 1
3. **Risk:Reward** ≥ 2.5:1
4. **ADX Regime** = TREND or RANGE (not TRANSITION)
5. **Spread** < 2× average spread
6. **FTMO Limits** not breached

### Entry Execution
```python
if confluence_score >= min_confluence and can_trade:
    # Place pending order at entry price
    lot_size = calculate_lot_size(risk_pct, sl_distance, symbol)
    
    if direction == "BUY":
        place_buy_stop(entry_price, lot_size, sl, tp)
    else:
        place_sell_stop(entry_price, lot_size, sl, tp)
```

---

## Exit Management

### Take Profit Levels
1. **TP1**: 1.0R - Partial exit (35.0% position)
2. **TP2**: 2.5R - Final target

### Trailing Stop
Activated after 1.0R profit:
```python
trail_distance = atr * params['atr_trail_multiplier']  # 2.2× ATR
```

### Stop Loss
- **Initial**: Set at S/R level or 1.5× ATR from entry
- **Breakeven**: Move to entry +1 pip after 1R profit
- **Trail**: Follows price at 2.2× ATR distance

---

## Risk Management

### Position Sizing
```python
# Current setting: 0.3% per trade
risk_amount = account_size * 0.3 / 100
lot_size = risk_amount / (sl_pips * pip_value)
```

### FTMO Limits
- **Max Daily Loss**: 5.0% ($10,000) - Bot halts at 4.2%
- **Max Total Drawdown**: 10.0% ($20,000) - Bot emergency stop at 7%
- **Max Concurrent Trades**: 6

### Seasonal Adjustments
- **Summer (June-Aug)**: Risk × 0.75 (lower volatility)
- **December**: ATR × 1.7000000000000002 (holiday spike)

---

## Current Parameters

**Active Configuration** (from `params/current_params.json`):

```json
{
  "min_confluence": 2,
  "min_quality_factors": 1,
  "risk_per_trade_pct": 0.3,
  "atr_min_percentile": 55.0,
  "trail_activation_r": 1.0,
  "december_atr_multiplier": 1.7000000000000002,
  "volatile_asset_boost": 1.1,
  "adx_trend_threshold": 22.0,
  "adx_range_threshold": 14.0,
  "trend_min_confluence": 4,
  "range_min_confluence": 4,
  "rsi_oversold_range": 25.0,
  "rsi_overbought_range": 75.0,
  "atr_volatility_ratio": 0.7,
  "atr_trail_multiplier": 2.2,
  "partial_exit_at_1r": false,
  "use_adx_slope_rising": false,
  "partial_exit_pct": 0.35,
  "generated_at": "2025-12-27T09:30:42.450029Z",
  "generated_by": "ftmo_challenge_analyzer.py",
  "version": "1.0.0"
}
```

---

## Strategy Functions Reference

### Core Functions


#### `detect_regime(daily_candles: List[Dict], adx_trend_threshold: float, adx_range_threshold: float, use_adx_slope_rising: bool, use_adx_regime_filter: bool) -> Dict`

Detect market regime based on ADX (Average Directional Index).

When use_adx_regime_filter=False, the ADX filter is bypassed and all setups
are treated as Trend Mode. This allows trading without regime restrictions.

This is the core function for the Regime-Adaptive V2 trading system.
It classifies the current market into one of three regimes:

1. TREND MODE (ADX >= adx_trend_threshold):
   - Market has strong directional movement
   - Trade with momentum: use trend-following entries
   - Higher confluence requirements, larger position sizing allowed
   
2. RANGE MODE (ADX < adx_range_threshold):
   - Market is ranging/consolidating
   - Trade mean reversion: fade extremes at S/R zones
   - Ultra-conservative: ALL range mode filters must pass
   - Require RSI extremes, Fib 0.786, H4 rejection candles
   
3. TRANSITION ZONE (ADX between thresholds):
   - Market is transitioning between regimes
   - OPTIMIZED: Allow trading at 6+/7 confluence (requires high conviction)
   - This captures emerging trends early with strict confirmation

V2 Enhancement: Early Trend Detection
When use_adx_slope_rising=True, allows Trend Mode entry even when ADX is
slightly below threshold if:
- ADX is rising (slope > 0)
- ADX is within 3 points of trend threshold
- There's a recent +DI/-DI crossover (trend direction confirmation)

Args:
    daily_candles: List of D1 OHLCV candle dictionaries
    adx_trend_threshold: ADX level for trend mode (default 25.0)
    adx_range_threshold: ADX level for range mode (default 20.0)
    use_adx_slope_rising: Enable early trend detection via ADX slope (default False)

Returns:
    Dict with keys:
        'mode': str - 'Trend', 'Range', or 'Transition'
        'adx': float - Current ADX value
        'can_trade': bool - Whether entries are allowed in this regime
        'description': str - Human-readable regime description
        'adx_slope': float - ADX slope (only if use_adx_slope_rising=True)
        'di_crossover': str - DI crossover info (only if use_adx_slope_rising=True)
        'early_trend_entry': bool - True if triggered by early trend detection

Note:
    - No look-ahead bias: uses only data up to current candle
    - ADX is calculated using standard 14-period smoothing


#### `_infer_trend(candles: List[Dict], short_lookback: int, long_lookback: int) -> str`

Infer trend direction without using EMA (EMA removed).

Simple heuristic: compare short vs long simple averages and
check recent price action for higher highs / lower lows.

Returns: "bullish", "bearish" or "mixed".


#### `compute_confluence(monthly_candles: List[Dict], weekly_candles: List[Dict], daily_candles: List[Dict], h4_candles: List[Dict], direction: str, params: Optional[StrategyParams], historical_sr: Optional[Dict[str, List[Dict]]]) -> Tuple[Dict[str, bool], Dict[str, str], Tuple]`

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


#### `simulate_trades(candles: List[Dict], symbol: str, params: Optional[StrategyParams], monthly_candles: Optional[List[Dict]], weekly_candles: Optional[List[Dict]], h4_candles: Optional[List[Dict]], include_transaction_costs: bool) -> List[Trade]`

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


---

**Maintained by**: Auto-generated from source code  
**Update command**: `python scripts/update_docs.py`  
**Source**: `strategy_core.py`
