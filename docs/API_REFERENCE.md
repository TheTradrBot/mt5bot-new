# API Reference

**Last Updated**: 2025-12-28  
**Auto-generated**: Yes (run `python scripts/update_docs.py` to regenerate)

---

## Table of Contents

- [strategy_core](#strategy_core)
- [ftmo_challenge_analyzer](#ftmo_challenge_analyzer)
- [params_loader](#params_loader)
- [manager](#manager)
- [output_manager](#output_manager)

---

## strategy_core

**File**: `strategy_core.py`

### Classes

#### `StrategyParams`

Strategy parameters that can be optimized.

These control confluence thresholds, SL/TP ratios, filters, etc.

**Methods**:

- `to_dict()`: Convert parameters to dictionary....
- `from_dict()`: Create parameters from dictionary....

#### `Signal`

Represents a trading signal/setup.


#### `Trade`

Represents a completed trade for backtest analysis.

**Methods**:

- `to_dict()`: Convert trade to dictionary....

### Functions

#### `_get_candle_datetime(candle: Dict)`

**Returns**: `Optional[datetime]`

Extract datetime from candle dictionary.

---

#### `_slice_htf_by_timestamp(htf_candles: Optional[List[Dict]], reference_dt: datetime)`

**Returns**: `Optional[List[Dict]]`

Slice higher-timeframe candles to only include those with timestamp <= reference.

This prevents look-ahead bias by ensuring we only use HTF data
that would have been available at the time of the reference candle.

Args:
    htf_candles: List of higher-timeframe candles (weekly, monthly, etc)
    reference_dt: The reference datetime (current daily candle time)

Returns:
    Sliced list of candles or None if input is None/empty

---

#### `_is_weekend(candle: Dict)`

**Returns**: `bool`

Check if a candle falls on a weekend (Saturday=5, Sunday=6).

---

#### `_atr(candles: List[Dict], period: int)`

**Returns**: `float`

Calculate Average True Range (ATR).

Args:
    candles: List of OHLCV candle dictionaries
    period: ATR period (default 14)

Returns:
    ATR value or 0 if insufficient data

---

#### `_calculate_atr_percentile(candles: List[Dict], period: int, lookback: int)`

**Returns**: `Tuple[float, float]`

Calculate current ATR and its percentile rank over the lookback period.

Args:
    candles: List of OHLCV candle dictionaries
    period: ATR period (default 14)
    lookback: Number of days to calculate percentile over (default 100)

Returns:
    Tuple of (current_atr, percentile_rank 0-100)

---

#### `calculate_adx(candles: List[Dict], period: int)`

**Returns**: `float`

Calculate Average Directional Index (ADX) for trend strength measurement.

ADX measures trend strength regardless of direction:
- ADX > 25: Strong trend (good for trend following)
- ADX 20-25: Moderate trend (transition zone)
- ADX < 20: Weak trend/ranging market (consider mean reversion)

Args:
    candles: List of OHLCV candle dictionaries
    period: ADX period (default 14)

Returns:
    ADX value (0-100 scale)

---

#### `detect_regime(daily_candles: List[Dict], adx_trend_threshold: float, adx_range_threshold: float, use_adx_slope_rising: bool, use_adx_regime_filter: bool)`

**Returns**: `Dict`

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

---

#### `_check_h4_rejection_candle(h4_candles: List[Dict], direction: str)`

**Returns**: `Tuple[bool, str]`

Check for H4 rejection candle (engulfing or pinbar) at current level.

Args:
    h4_candles: List of H4 OHLCV candle dictionaries
    direction: 'bullish' or 'bearish'

Returns:
    Tuple of (has_rejection, description)

---

#### `_check_fib_786_zone(candles: List[Dict], price: float, direction: str, tolerance: float)`

**Returns**: `Tuple[bool, str]`

Check if price is in the Fib 0.786 retracement zone (±tolerance).

This is specifically for Range Mode entries which require price
to be at deep retracement levels for mean reversion setups.

Args:
    candles: List of OHLCV candle dictionaries
    price: Current price
    direction: 'bullish' or 'bearish'
    tolerance: Tolerance around 0.786 level (default ±0.05 = 0.736-0.836)

Returns:
    Tuple of (is_in_zone, description)

---

#### `validate_range_mode_entry(daily_candles: List[Dict], h4_candles: Optional[List[Dict]], weekly_candles: Optional[List[Dict]], monthly_candles: Optional[List[Dict]], price: float, direction: str, confluence_score: int, params: Optional['StrategyParams'], historical_sr: Optional[Dict[str, List[Dict]]], atr_vol_ratio_range: float)`

**Returns**: `Tuple[bool, Dict[str, Any]]`

Validate entry for Range Mode (conservative mean reversion).

Range Mode is ultra-conservative and requires ALL of the following conditions:

1. Confluence Score >= min_confluence (5-6)
   - Even in ranging markets, we need multiple confirming factors

2. Price at Major S/R Zone
   - Uses existing location helpers to check for support/resistance
   - Historical HTF levels carry more weight

3. Price in Fib 0.786 Retracement Zone (±0.05 tolerance)
   - Deep retracement = better risk/reward for mean reversion
   - 0.786 is the classic "last chance" Fib level

4. H4 Rejection Candle (Engulfing or Pinbar)
   - Confirms price rejection at the level
   - Without rejection, no entry allowed

5. ATR Volatility Filter
   - Current ATR(14) < atr_volatility_ratio * ATR(50) average
   - Low volatility = ranging market confirmation
   - High volatility = avoid (may break out)

Args:
    daily_candles: D1 OHLCV data
    h4_candles: H4 OHLCV data (for rejection candle check)
    weekly_candles: W1 OHLCV data (for location context)
    monthly_candles: MN OHLCV data (for location context)
    price: Current price
    direction: 'bullish' or 'bearish'
    confluence_score: Pre-calculated confluence score
    params: StrategyParams with range mode thresholds
    historical_sr: Optional historical S/R levels
    atr_vol_ratio_range: ATR volatility ratio threshold (default 0.8)

Returns:
    Tuple of (is_valid, details_dict)
    - is_valid: True only if ALL conditions pass
    - details_dict: Contains each check result and notes

---

#### `calculate_volatility_parity_risk(atr: float, base_risk_pct: float, reference_atr: Optional[float], min_risk_pct: float, max_risk_pct: float)`

**Returns**: `float`

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

---

#### `_detect_bullish_n_pattern(candles: List[Dict], lookback: int)`

**Returns**: `Tuple[bool, str]`

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

---

#### `_detect_bearish_v_pattern(candles: List[Dict], lookback: int)`

**Returns**: `Tuple[bool, str]`

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

---

#### `_calculate_zscore(price: float, candles: List[Dict], period: int)`

**Returns**: `float`

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

---

#### `_find_bos_swing_for_bullish_n(candles: List[Dict], lookback: int)`

**Returns**: `Optional[Tuple[float, float, int, int]]`

Find the Bullish N pattern anchor points for Fibonacci:
After a break of structure UP, find the swing low.
Return fib anchors from red candle close to green candle open at swing low.

Blueprint rule: For bullish N, take fibs from red candle close to green candle open.

Returns:
    Tuple of (fib_start, fib_end, swing_low_idx, bos_idx) or None

---

#### `_find_bos_swing_for_bearish_v(candles: List[Dict], lookback: int)`

**Returns**: `Optional[Tuple[float, float, int, int]]`

Find the Bearish V pattern anchor points for Fibonacci:
After a break of structure DOWN, find the swing high.
Return fib anchors from green candle close to red candle open at swing high.

Blueprint rule: For bearish V (shorts), take fibs from green candle close to red candle open.

Returns:
    Tuple of (fib_start, fib_end, swing_high_idx, bos_idx) or None

---

#### `_detect_mitigated_sr(candles: List[Dict], price: float, direction: str, proximity_pct: float)`

**Returns**: `Tuple[bool, str, Optional[float]]`

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

---

#### `_detect_structural_framework(candles: List[Dict], direction: str)`

**Returns**: `Tuple[bool, str, Optional[Tuple[float, float]]]`

Detect ascending/descending channel frameworks on daily timeframe.

Framework detection:
- Ascending channel: Connect 3+ swing lows (ascending) and 3+ swing highs (ascending)
- Descending channel: Connect 3+ swing highs (descending) and 3+ swing lows (descending)

Returns:
    Tuple of (is_in_framework, note, (lower_bound, upper_bound) or None)

---

#### `_detect_displacement(candles: List[Dict], direction: str, atr_mult: float)`

**Returns**: `Tuple[bool, str]`

Detect displacement - strong candles beyond structure confirming the move.

Displacement = large body candle that shows institutional order flow.
Must be at least atr_mult * ATR in body size.

Returns:
    Tuple of (has_displacement, note)

---

#### `_detect_candle_rejection(candles: List[Dict], direction: str)`

**Returns**: `Tuple[bool, str]`

Detect pinbar or engulfing rejection patterns at current price.

Returns:
    Tuple of (has_rejection, note)

---

#### `_detect_momentum(candles: List[Dict], direction: str, lookback: int)`

**Returns**: `Tuple[bool, str]`

Detect momentum alignment using rate of change.

Returns:
    Tuple of (momentum_aligned, note)

---

#### `_find_pivots(candles: List[Dict], lookback: int)`

**Returns**: `Tuple[List[float], List[float]]`

Find swing highs and swing lows in candle data.

Args:
    candles: List of OHLCV candle dictionaries
    lookback: Number of bars to look back/forward for pivot identification

Returns:
    Tuple of (swing_highs, swing_lows) as lists of price levels

---

#### `_infer_trend(candles: List[Dict], short_lookback: int, long_lookback: int)`

**Returns**: `str`

Infer trend direction without using EMA (EMA removed).

Simple heuristic: compare short vs long simple averages and
check recent price action for higher highs / lower lows.

Returns: "bullish", "bearish" or "mixed".

---

#### `_pick_direction_from_bias(mn_trend: str, wk_trend: str, d_trend: str)`

**Returns**: `Tuple[str, str, bool]`

Determine trade direction based on multi-timeframe bias.

Args:
    mn_trend: Monthly trend
    wk_trend: Weekly trend
    d_trend: Daily trend

Returns:
    Tuple of (direction, note, htf_aligned)

---

#### `_location_context(monthly_candles: List[Dict], weekly_candles: List[Dict], daily_candles: List[Dict], price: float, direction: str, historical_sr: Optional[Dict[str, List[Dict]]])`

**Returns**: `Tuple[str, bool]`

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

---

#### `_fib_context(weekly_candles: List[Dict], daily_candles: List[Dict], direction: str, price: float, fib_low: float, fib_high: float)`

**Returns**: `Tuple[str, bool]`

Check if price is within a Fibonacci retracement zone using new Fibonacci module.

Returns:
    Tuple of (note, is_in_fib_zone)

---

#### `_find_last_swing_leg_for_fib(candles: List[Dict], direction: str)`

**Returns**: `Optional[Tuple[float, float]]`

Find the last swing leg for Fibonacci calculation using proper Blueprint anchoring.

Blueprint rules:
- Bullish N: After BOS up, fibs from red candle close to green candle open at swing low
- Bearish V: After BOS down, fibs from green candle close to red candle open at swing high

Returns:
    Tuple of (fib_low, fib_high) or None

---

#### `_structure_context(monthly_candles: List[Dict], weekly_candles: List[Dict], daily_candles: List[Dict], direction: str)`

**Returns**: `Tuple[bool, str]`

Check market structure alignment (BOS/CHoCH).

Returns:
    Tuple of (is_aligned, note)

---

#### `_h4_confirmation(h4_candles: List[Dict], direction: str, daily_candles: List[Dict])`

**Returns**: `Tuple[str, bool]`

Check for 4H timeframe confirmation (entry trigger).

Returns:
    Tuple of (note, is_confirmed)

---

#### `_find_structure_sl(candles: List[Dict], direction: str, lookback: int)`

**Returns**: `Optional[float]`

Find structure-based stop loss level.

Returns:
    Stop loss price level or None

---

#### `_compute_confluence_flags(monthly_candles: List[Dict], weekly_candles: List[Dict], daily_candles: List[Dict], h4_candles: List[Dict], direction: str, params: Optional[StrategyParams])`

**Returns**: `Tuple[Dict[str, bool], Dict[str, str], Tuple]`

Compute all confluence flags for a trading setup.

This is the main entry point for confluence calculation,
used by both backtests and live scanning.

Returns:
    Tuple of (flags dict, notes dict, trade_levels tuple)

---

#### `compute_confluence(monthly_candles: List[Dict], weekly_candles: List[Dict], daily_candles: List[Dict], h4_candles: List[Dict], direction: str, params: Optional[StrategyParams], historical_sr: Optional[Dict[str, List[Dict]]])`

**Returns**: `Tuple[Dict[str, bool], Dict[str, str], Tuple]`

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

---

#### `compute_trade_levels(daily_candles: List[Dict], direction: str, params: Optional[StrategyParams], h4_candles: Optional[List[Dict]])`

**Returns**: `Tuple[str, bool, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]`

Compute entry, SL, and TP levels using parameterized logic.

Args:
    daily_candles: Daily OHLCV data
    direction: Trade direction
    params: Strategy parameters
    h4_candles: 4H OHLCV data for tighter SL calculation

Returns:
    Tuple of (note, is_valid, entry, sl, tp1, tp2, tp3, tp4, tp5)

---

#### `generate_signals(candles: List[Dict], symbol: str, params: Optional[StrategyParams], monthly_candles: Optional[List[Dict]], weekly_candles: Optional[List[Dict]], h4_candles: Optional[List[Dict]])`

**Returns**: `List[Signal]`

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

---

#### `_validate_and_find_entry(candles: List[Dict], signal_bar: int, theoretical_entry: float, direction: str, max_wait_bars: int)`

**Returns**: `Tuple[Optional[int], float, bool]`

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

---

#### `simulate_trades(candles: List[Dict], symbol: str, params: Optional[StrategyParams], monthly_candles: Optional[List[Dict]], weekly_candles: Optional[List[Dict]], h4_candles: Optional[List[Dict]], include_transaction_costs: bool)`

**Returns**: `List[Trade]`

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

#### `get_default_params()`

**Returns**: `StrategyParams`

Get default strategy parameters - optimized for trade generation.

---

#### `get_aggressive_params()`

**Returns**: `StrategyParams`

Get aggressive (more trades) strategy parameters.

---

#### `get_conservative_params()`

**Returns**: `StrategyParams`

Get conservative (higher quality) strategy parameters.

---

#### `extract_ml_features(candles: List[Dict], flags: Dict[str, bool], direction: str, params: Optional[StrategyParams])`

**Returns**: `Dict[str, float]`

Extract ML features for trade filtering.

Args:
    candles: List of OHLCV candle dictionaries
    flags: Confluence flags from compute_confluence
    direction: Trade direction ("bullish" or "bearish")
    params: Strategy parameters

Returns:
    Dict with ML features for model input

---

#### `apply_ml_filter(features: Dict[str, float], min_prob: float)`

**Returns**: `Tuple[bool, float]`

Apply ML model to filter trades.

Args:
    features: Dict of features from extract_ml_features
    min_prob: Minimum probability threshold for trade acceptance

Returns:
    Tuple of (should_trade, probability)

---

#### `check_volatility_filter(candles: List[Dict], atr_min_percentile: float, current_date: Optional[datetime], december_atr_multiplier: float)`

**Returns**: `Tuple[bool, float]`

Check if current volatility is above minimum threshold.
In December, applies december_atr_multiplier to be more strict.

---

#### `apply_volatile_asset_boost(symbol: str, confluence_score: int, quality_factors: int, volatile_asset_boost: float)`

**Returns**: `Tuple[int, int]`

Apply boost to confluence/quality scores for volatile assets.
These assets (XAUUSD, NAS100USD, GBPJPY, BTCUSD) have potential for bigger R trades.

---

#### `to_dict(self: Any)`

**Returns**: `Dict[str, Any]`

Convert parameters to dictionary.

---

#### `from_dict(cls: Any, d: Dict[str, Any])`

**Returns**: `'StrategyParams'`

Create parameters from dictionary.

---

#### `to_dict(self: Any)`

**Returns**: `Dict[str, Any]`

Convert trade to dictionary.

---

## ftmo_challenge_analyzer

**File**: `ftmo_challenge_analyzer.py`

### Classes

#### `BacktestTrade`

Complete trade record for CSV export.

**Methods**:

- `to_dict()`: Convert to dictionary for CSV export....

#### `MonteCarloSimulator`

Monte Carlo simulation for robustness testing.

**Methods**:

- `__init__()`: No description...
- `_extract_r_values()`: No description...
- `run_simulation()`: No description...

#### `OptunaOptimizer`

Optuna-based optimizer for FTMO strategy parameters.
Runs optimization ONLY on training data (2023-01-01 to 2024-09-30).
Uses persistent SQLite storage for resumability.
Uses OutputManager for structured CSV output.

**Methods**:

- `__init__()`: No description...
- `_objective()`: Optuna objective function - DUAL PERIOD validation.
Trains on 2023-2024-09-30, validates on 2024-10-...
- `run_optimization()`: Run Optuna optimization on TRAINING data only.

Args:
    n_trials: Number of trials to run
    earl...

### Functions

#### `save_best_params_persistent(best_params: Dict)`

**Returns**: `None`

Save best parameters to best_params.json for instant bot updates.
This persists even if optimization run is halted abruptly.

---

#### `calculate_adx(candles: List[Dict], period: int)`

**Returns**: `float`

Calculate Average Directional Index (ADX) for trend strength measurement.
ADX > 25 indicates a strong trend.

Args:
    candles: List of OHLCV candle dictionaries
    period: ADX period (default 14)

Returns:
    ADX value (0-100 scale)

---

#### `check_adx_filter(candles: List[Dict], min_adx: float)`

**Returns**: `Tuple[bool, float]`

Check if ADX is above minimum threshold for trend trading.

Args:
    candles: D1 candles for ADX calculation
    min_adx: Minimum ADX value (default 25)

Returns:
    Tuple of (passes_filter, adx_value)

---

#### `log_optimization_progress(trial_num: int, value: float, best_value: float, best_params: Dict)`

**Returns**: `None`

Append optimization progress to log file.

---

#### `show_optimization_status()`

**Returns**: `None`

Display current optimization status without running new trials.

---

#### `is_valid_trading_day(dt: datetime)`

**Returns**: `bool`

Check if datetime is a valid trading day (no weekends).

---

#### `_atr(candles: List[Dict], period: int)`

**Returns**: `float`

Calculate Average True Range (ATR).

---

#### `_calculate_atr_percentile(candles: List[Dict], period: int, lookback: int)`

**Returns**: `Tuple[float, float]`

Calculate current ATR and its percentile rank.

---

#### `run_monte_carlo_analysis(trades: List[Any], num_simulations: int)`

**Returns**: `Dict`

Run Monte Carlo analysis on trades.

---

#### `load_ohlcv_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime)`

**Returns**: `List[Dict]`

Load OHLCV data from local CSV files only (no API calls). Uses cache for performance.

---

#### `get_all_trading_assets()`

**Returns**: `List[str]`

Get list of all tradeable assets.

---

#### `run_full_period_backtest(start_date: datetime, end_date: datetime, min_confluence: int, min_quality_factors: int, risk_per_trade_pct: float, atr_min_percentile: float, trail_activation_r: float, december_atr_multiplier: float, volatile_asset_boost: float, ml_min_prob: Optional[float], excluded_assets: Optional[List[str]], require_adx_filter: bool, min_adx: float, use_adx_regime_filter: bool, adx_trend_threshold: float, adx_range_threshold: float, trend_min_confluence: int, range_min_confluence: int, rsi_oversold_range: float, rsi_overbought_range: float, atr_volatility_ratio: float, atr_trail_multiplier: float, partial_exit_at_1r: bool, partial_exit_pct: float, use_adx_slope_rising: bool, atr_vol_ratio_range: float)`

**Returns**: `List[Trade]`

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

---

#### `convert_to_backtest_trade(trade: Trade, trade_num: int, risk_per_trade_pct: float, adx_value: float)`

**Returns**: `BacktestTrade`

Convert strategy Trade to BacktestTrade for CSV export.

---

#### `export_trades_to_csv(trades: List[Trade], filename: str, risk_per_trade_pct: float)`

**Returns**: `None`

Export trades to CSV with all required columns.

---

#### `print_period_results(trades: List[Trade], period_name: str, start: datetime, end: datetime)`

**Returns**: `Dict`

Print results for a specific period.

---

#### `update_readme_documentation()`

**Returns**: `None`

Update README.md with optimization & backtesting section.

---

#### `train_ml_model(trades: List[Trade])`

**Returns**: `bool`

Train RandomForest ML model on full-year trades and save to models/best_rf.joblib.

---

#### `validate_top_trials(study: Any, top_n: int)`

**Returns**: `List[Dict]`

Run validation backtests on top N trials to find the best OOS performer.
This prevents overfitting by selecting based on validation performance.

Works with both single-objective and multi-objective (NSGA-II) studies.

Returns:
    List of dicts with trial info and validation results, sorted by validation R

---

#### `generate_summary_txt(results: Dict, training_trades: List, validation_trades: List, full_year_trades: List, best_params: Dict)`

**Returns**: `str`

Generate a summary text file after each analyzer run.

---

#### `multi_objective_function(trial: Any)`

**Returns**: `Tuple[float, float, float]`

Multi-objective function for NSGA-II optimization.
Returns three objectives to maximize:
1. Total R (profitability)
2. Sharpe Ratio (risk-adjusted returns)
3. Win Rate (consistency)

All three should be MAXIMIZED (Optuna NSGA-II handles this).

---

#### `run_multi_objective_optimization(n_trials: int)`

**Returns**: `Dict`

Run NSGA-II multi-objective optimization.

NSGA-II (Non-dominated Sorting Genetic Algorithm II) finds the Pareto frontier:
solutions where improving one objective would worsen another.

Returns the best balanced solution from the Pareto frontier.

---

#### `main()`

**Returns**: `None`

Professional FTMO Optimization Workflow with CLI support.

Configuration is loaded from params/optimization_config.json
CLI arguments can override config file settings.

Usage:
  python ftmo_challenge_analyzer.py              # Run/resume optimization (default 5 trials)
  python ftmo_challenge_analyzer.py --status     # Check progress without running
  python ftmo_challenge_analyzer.py --config     # Show current configuration
  python ftmo_challenge_analyzer.py --trials 100 # Run 100 trials
  python ftmo_challenge_analyzer.py --multi      # Use NSGA-II multi-objective optimization
  python ftmo_challenge_analyzer.py --single     # Use TPE single-objective optimization
  python ftmo_challenge_analyzer.py --adx        # Enable ADX regime filtering

---

#### `to_dict(self: Any)`

**Returns**: `Dict`

Convert to dictionary for CSV export.

---

#### `__init__(self: Any, trades: List[Any], num_simulations: int)`

**Returns**: `None`

No description

---

#### `_extract_r_values(self: Any)`

**Returns**: `List[float]`

No description

---

#### `run_simulation(self: Any)`

**Returns**: `Dict[str, Any]`

No description

---

#### `__init__(self: Any)`

**Returns**: `None`

No description

---

#### `_objective(self: Any, trial: Any)`

**Returns**: `float`

Optuna objective function - DUAL PERIOD validation.
Trains on 2023-2024-09-30, validates on 2024-10-01+.
Objective: Balance training profit + validation robustness (50/50 weighted).

REGIME-ADAPTIVE V2: Extended search space includes:
- Regime detection thresholds (ADX trend/range)
- Mode-specific confluence requirements
- Partial profit taking and trail management

---

#### `run_optimization(self: Any, n_trials: int, early_stopping_patience: int)`

**Returns**: `Dict`

Run Optuna optimization on TRAINING data only.

Args:
    n_trials: Number of trials to run
    early_stopping_patience: Stop if no improvement after this many trials (default: 20)

---

#### `calc_stats(trades: Any)`

**Returns**: `None`

No description

---

#### `progress_callback(study: Any, trial: Any)`

**Returns**: `None`

No description

---

#### `progress_callback(study: Any, trial: Any)`

**Returns**: `None`

No description

---

## params_loader

**File**: `params/params_loader.py`

### Classes

#### `ParamsNotFoundError`

Raised when params file doesn't exist. Run optimizer first.


### Functions

#### `load_params_dict()`

**Returns**: `Dict[str, Any]`

Load raw parameters dictionary from JSON file.

Returns:
    Dict with all parameters
    
Raises:
    ParamsNotFoundError: If params file doesn't exist

---

#### `load_strategy_params()`

**Returns**: `None`

Load optimized strategy parameters.

Returns:
    StrategyParams object with optimized values
    
Raises:
    ParamsNotFoundError: If params file doesn't exist

---

#### `get_min_confluence()`

**Returns**: `int`

Get minimum confluence score from params.

---

#### `get_max_concurrent_trades()`

**Returns**: `int`

Get maximum concurrent trades from params.

---

#### `get_risk_per_trade_pct()`

**Returns**: `float`

Get risk per trade percentage from params.

---

#### `get_transaction_costs(symbol: str)`

**Returns**: `Tuple[float, float, float]`

Get transaction costs for a symbol.

Args:
    symbol: Trading symbol (any format - EURUSD, EUR_USD, etc)
    
Returns:
    Tuple of (spread_pips, slippage_pips, commission_per_lot)

---

#### `save_optimized_params(params_dict: Dict[str, Any], backup: bool)`

**Returns**: `Path`

Save optimized parameters to JSON file.

Args:
    params_dict: Dictionary of optimized parameters
    backup: Whether to create backup in history folder
    
Returns:
    Path to saved file

---

## manager

**File**: `tradr/risk/manager.py`

### Classes

#### `OpenPosition`

Represents an open position for risk calculation.

**Methods**:

- `potential_loss_usd()`: Calculate potential loss if SL is hit.

FIXED: Now uses symbol-specific pip size instead of hardcode...

#### `DailyRecord`

Track daily PnL for profitable day counting.

**Methods**:

- `is_profitable()`: Check if day qualifies as profitable (FTMO has no minimum threshold)....

#### `ChallengeState`

Persistent state for FTMO challenge tracking.
Saved to JSON file for persistence across restarts.

**Methods**:

- `to_dict()`: No description...
- `from_dict()`: No description...
- `current_profit_pct()`: Current profit as percentage of initial balance....
- `current_drawdown_pct()`: Current drawdown from initial balance....
- `daily_loss_pct()`: Current daily loss percentage....
- `target_pct()`: Target profit percentage for current phase....
- `progress_pct()`: Progress towards target as percentage....

#### `RiskCheckResult`

Result of a pre-trade risk check.


#### `RiskManager`

Risk manager for FTMO Challenge.

Key features:
- Pre-trade simulation: Calculates worst-case DD if all SLs hit
- Dynamic lot reduction: Halves lot for each existing position
- Daily loss tracking: Blocks trades that would breach 5% daily limit
- Max drawdown tracking: Blocks trades that would breach 10% overall limit
- Emergency close: Closes all positions before hitting hard limits
- Partial take profits: Scales out at TP1, TP2, TP3

**Methods**:

- `__init__()`: No description...
- `_load_state()`: Load state from file or create new....
- `save_state()`: Save state to file....
- `sync_from_mt5()`: Sync state with actual MT5 account values.

Use this on startup to ensure risk manager uses real acc...
- `start_challenge()`: Start or restart a challenge....
- `stop_challenge()`: Stop the current challenge....
- `advance_to_phase2()`: Advance to Phase 2 after passing Phase 1....
- `_check_new_day()`: Check if it's a new trading day and reset daily tracking....
- `_calculate_total_open_risk()`: Calculate total potential loss from all open positions....
- `_simulate_worst_case_dd()`: Simulate worst-case drawdown if all open positions hit SL.

Returns:
    (daily_dd_pct, total_dd_pct...
- `check_trade()`: Pre-trade risk check with FTMO-compliant limits.

Simulates worst-case scenario where all open posit...
- `record_trade_open()`: Record a new position opening....
- `record_trade_close()`: Record a position closing....
- `get_status()`: Get current challenge status for Discord embed....
- `should_emergency_close()`: Check if positions should be closed to protect account.

Uses buffer thresholds to close BEFORE hitt...
- `calculate_pending_orders_risk()`: Calculate total risk from pending orders.

Args:
    pending_setups: List of pending setup dictionar...

### Functions

#### `potential_loss_usd(self: Any)`

**Returns**: `float`

Calculate potential loss if SL is hit.

FIXED: Now uses symbol-specific pip size instead of hardcoded 0.0001.
This is critical for JPY pairs (0.01), Gold (0.01), BTC (1.0), etc.

---

#### `is_profitable(self: Any)`

**Returns**: `bool`

Check if day qualifies as profitable (FTMO has no minimum threshold).

---

#### `to_dict(self: Any)`

**Returns**: `Dict`

No description

---

#### `from_dict(cls: Any, data: Dict)`

**Returns**: `'ChallengeState'`

No description

---

#### `current_profit_pct(self: Any)`

**Returns**: `float`

Current profit as percentage of initial balance.

---

#### `current_drawdown_pct(self: Any)`

**Returns**: `float`

Current drawdown from initial balance.

---

#### `daily_loss_pct(self: Any)`

**Returns**: `float`

Current daily loss percentage.

---

#### `target_pct(self: Any)`

**Returns**: `float`

Target profit percentage for current phase.

---

#### `progress_pct(self: Any)`

**Returns**: `float`

Progress towards target as percentage.

---

#### `__init__(self: Any, state_file: str)`

**Returns**: `None`

No description

---

#### `_load_state(self: Any)`

**Returns**: `ChallengeState`

Load state from file or create new.

---

#### `save_state(self: Any)`

**Returns**: `None`

Save state to file.

---

#### `sync_from_mt5(self: Any, balance: float, equity: float)`

**Returns**: `None`

Sync state with actual MT5 account values.

Use this on startup to ensure risk manager uses real account values
instead of potentially stale state file values.

---

#### `start_challenge(self: Any, phase: int)`

**Returns**: `None`

Start or restart a challenge.

---

#### `stop_challenge(self: Any)`

**Returns**: `None`

Stop the current challenge.

---

#### `advance_to_phase2(self: Any)`

**Returns**: `None`

Advance to Phase 2 after passing Phase 1.

---

#### `_check_new_day(self: Any)`

**Returns**: `None`

Check if it's a new trading day and reset daily tracking.

---

#### `_calculate_total_open_risk(self: Any)`

**Returns**: `float`

Calculate total potential loss from all open positions.

---

#### `_simulate_worst_case_dd(self: Any, new_trade_loss: float)`

**Returns**: `Tuple[float, float]`

Simulate worst-case drawdown if all open positions hit SL.

Returns:
    (daily_dd_pct, total_dd_pct) after simulated losses

---

#### `check_trade(self: Any, symbol: str, direction: str, entry_price: float, stop_loss_price: float, requested_lot: float, pending_orders_risk: float)`

**Returns**: `RiskCheckResult`

Pre-trade risk check with FTMO-compliant limits.

Simulates worst-case scenario where all open positions + pending + new trade hit SL.
Enforces:
- MAX_SINGLE_TRADE_RISK_PCT (1%): Hard cap per trade
- MAX_CUMULATIVE_RISK_PCT (3%): Max total open + pending risk
- Daily loss and max drawdown limits

Returns adjusted lot size that won't breach limits.

---

#### `record_trade_open(self: Any, symbol: str, direction: str, entry_price: float, stop_loss: float, lot_size: float, order_id: int)`

**Returns**: `None`

Record a new position opening.

---

#### `record_trade_close(self: Any, order_id: int, exit_price: float, pnl_usd: float)`

**Returns**: `None`

Record a position closing.

---

#### `get_status(self: Any)`

**Returns**: `Dict[str, Any]`

Get current challenge status for Discord embed.

---

#### `should_emergency_close(self: Any, current_equity: float)`

**Returns**: `Tuple[bool, str]`

Check if positions should be closed to protect account.

Uses buffer thresholds to close BEFORE hitting hard limits:
- DAILY_LOSS_BUFFER_PCT (4.5%): Close before 5% daily loss limit
- TOTAL_DD_BUFFER_PCT (9%): Close before 10% max drawdown limit

Returns:
    Tuple of (should_close, reason) where should_close is True if
    emergency close is needed and reason explains why.

---

#### `calculate_pending_orders_risk(self: Any, pending_setups: List[Dict])`

**Returns**: `float`

Calculate total risk from pending orders.

Args:
    pending_setups: List of pending setup dictionaries with 'lot_size', 
                  'entry_price', 'stop_loss', and 'symbol' keys.

Returns:
    Total potential risk in USD from all pending orders.

---

## output_manager

**File**: `tradr/utils/output_manager.py`

### Classes

#### `TrialResult`

Single trial result for logging.


#### `OutputManager`

Centralized output manager for optimization results.

Features:
- Real-time trial logging to CSV (nohup compatible)
- Best trial trade exports (training/validation/final)
- Monthly statistics breakdown
- Symbol performance analysis
- Final optimization report
- Separate directories for NSGA-II vs TPE runs

**Methods**:

- `__init__()`: Initialize OutputManager.

Args:
    output_dir: Custom output directory (optional)
    optimization...
- `_init_log_file()`: Initialize optimization log with header....
- `log_trial()`: Log a trial result to log file. Returns True if this is a new best.

Args:
    trial_number: Optuna ...
- `save_best_trial_trades()`: Save all trades from best trial to separate CSV files.

Args:
    training_trades: List of Trade obj...
- `_export_trades_csv()`: Export trades to CSV file....
- `generate_monthly_stats()`: Generate monthly statistics breakdown.

Creates/updates monthly_stats.csv with:
- Period (training/v...
- `generate_symbol_performance()`: Generate symbol performance breakdown.

Creates symbol_performance.csv with:
- Symbol
- Total trades...
- `generate_final_report()`: Generate final optimization report as CSV.

Creates optimization_report.csv with complete summary....
- `clear_output()`: Clear all output files for fresh optimization run....

### Functions

#### `get_output_manager(optimization_mode: str)`

**Returns**: `OutputManager`

Get the global OutputManager instance.

Args:
    optimization_mode: "NSGA" or "TPE" - determines subdirectory

---

#### `set_output_manager(optimization_mode: str)`

**Returns**: `None`

Explicitly set/reset the global OutputManager with specific mode.

Args:
    optimization_mode: "NSGA" or "TPE"

---

#### `__init__(self: Any, output_dir: Path, optimization_mode: str)`

**Returns**: `None`

Initialize OutputManager.

Args:
    output_dir: Custom output directory (optional)
    optimization_mode: "NSGA" or "TPE" - creates subdirectory in ftmo_analysis_output/

---

#### `_init_log_file(self: Any)`

**Returns**: `None`

Initialize optimization log with header.

---

#### `log_trial(self: Any, trial_number: int, score: float, total_r: float, sharpe_ratio: float, win_rate: float, profit_factor: float, total_trades: int, profit_usd: float, max_drawdown_pct: float, val_metrics: Optional[Dict], final_metrics: Optional[Dict])`

**Returns**: `bool`

Log a trial result to log file. Returns True if this is a new best.

Args:
    trial_number: Optuna trial number
    score: Composite optimization score
    total_r: Total R (risk units profit)
    sharpe_ratio: Sharpe ratio
    win_rate: Win rate percentage
    profit_factor: Profit factor
    total_trades: Number of trades
    profit_usd: Profit in USD (based on $200K account)
    max_drawdown_pct: Maximum drawdown percentage
    val_metrics: Validation period metrics (dict with total_r, sharpe, win_rate, profit_usd)
    final_metrics: Final period metrics (dict with total_r, sharpe, win_rate, profit_usd)

Returns:
    True if this trial is new best, False otherwise

---

#### `save_best_trial_trades(self: Any, training_trades: List[Any], validation_trades: List[Any], final_trades: List[Any], risk_pct: float, account_size: float)`

**Returns**: `None`

Save all trades from best trial to separate CSV files.

Args:
    training_trades: List of Trade objects from training period
    validation_trades: List of Trade objects from validation period
    final_trades: List of Trade objects from full period
    risk_pct: Risk per trade percentage
    account_size: Account size in USD

---

#### `_export_trades_csv(self: Any, trades: List[Any], filepath: Path, risk_pct: float, account_size: float)`

**Returns**: `None`

Export trades to CSV file.

---

#### `generate_monthly_stats(self: Any, trades: List[Any], period_name: str, risk_pct: float, account_size: float)`

**Returns**: `None`

Generate monthly statistics breakdown.

Creates/updates monthly_stats.csv with:
- Period (training/validation/final)
- Month
- Number of trades
- Win rate
- Total profit USD

---

#### `generate_symbol_performance(self: Any, trades: List[Any], risk_pct: float, account_size: float)`

**Returns**: `None`

Generate symbol performance breakdown.

Creates symbol_performance.csv with:
- Symbol
- Total trades
- Win rate
- Total profit USD
- Average R per trade

---

#### `generate_final_report(self: Any, best_params: Dict[str, Any], training_metrics: Dict[str, float], validation_metrics: Dict[str, float], final_metrics: Dict[str, float], total_trials: int, optimization_time_hours: float)`

**Returns**: `None`

Generate final optimization report as CSV.

Creates optimization_report.csv with complete summary.

---

#### `clear_output(self: Any)`

**Returns**: `None`

Clear all output files for fresh optimization run.

---


## Usage Examples

### Load Strategy Parameters
```python
from params.params_loader import load_strategy_params

params = load_strategy_params()
min_confluence = params['min_confluence_score']
```

### Run Backtest
```python
from ftmo_challenge_analyzer import run_full_period_backtest

trades = run_full_period_backtest(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    min_confluence=4,
    risk_per_trade_pct=0.5
)
```

### Check Risk Manager
```python
from tradr.risk.manager import RiskManager

rm = RiskManager(account_size=200000)
can_trade, reason = rm.can_trade(symbol="EURUSD", risk_pct=0.5)
```

---

**Auto-generated**: Run `python scripts/update_docs.py` to regenerate  
**Last update**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
