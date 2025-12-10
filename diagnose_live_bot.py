
#!/usr/bin/env python3
"""
Diagnostic script to check why live bot isn't placing trades.
Runs the same scan logic but shows WHY setups are being rejected.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from strategy_core import (
    _infer_trend,
    _pick_direction_from_bias,
    compute_confluence,
)
from symbol_mapping import ALL_TRADABLE_FTMO
from data import get_ohlcv
from ftmo_config import FTMO_CONFIG, get_pip_size
from config import SIGNAL_MODE, MIN_CONFLUENCE_STANDARD, MIN_CONFLUENCE_AGGRESSIVE

MIN_CONFLUENCE = MIN_CONFLUENCE_STANDARD if SIGNAL_MODE == "standard" else MIN_CONFLUENCE_AGGRESSIVE

def diagnose_symbol(symbol: str) -> Dict:
    """Diagnose why a symbol isn't producing a trade."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {symbol}")
    print(f"{'='*70}")
    
    reasons = []
    
    # Get data
    daily = get_ohlcv(symbol, timeframe="D", count=500, use_cache=False)
    weekly = get_ohlcv(symbol, timeframe="W", count=104, use_cache=False) or []
    monthly = get_ohlcv(symbol, timeframe="M", count=24, use_cache=False) or []
    h4 = get_ohlcv(symbol, timeframe="H4", count=500, use_cache=False) or []
    
    if not daily or len(daily) < 50:
        reasons.append(f"Insufficient daily data ({len(daily) if daily else 0} candles)")
        return {"symbol": symbol, "tradeable": False, "reasons": reasons}
    
    if not weekly or len(weekly) < 10:
        reasons.append(f"Insufficient weekly data ({len(weekly)} candles)")
        return {"symbol": symbol, "tradeable": False, "reasons": reasons}
    
    # Get trends
    mn_trend = _infer_trend(monthly) if monthly else "mixed"
    wk_trend = _infer_trend(weekly) if weekly else "mixed"
    d_trend = _infer_trend(daily) if daily else "mixed"
    
    print(f"Trends: M={mn_trend}, W={wk_trend}, D={d_trend}")
    
    direction, _, _ = _pick_direction_from_bias(mn_trend, wk_trend, d_trend)
    print(f"Direction: {direction}")
    
    # Compute confluence
    flags, notes, trade_levels = compute_confluence(
        monthly, weekly, daily, h4, direction
    )
    
    entry, sl, tp1, tp2, tp3, tp4, tp5 = trade_levels
    
    confluence_score = sum(1 for v in flags.values() if v)
    has_rr = flags.get("rr", False)
    has_location = flags.get("location", False)
    has_fib = flags.get("fib", False)
    has_liquidity = flags.get("liquidity", False)
    has_structure = flags.get("structure", False)
    has_htf_bias = flags.get("htf_bias", False)
    
    quality_factors = sum([has_location, has_fib, has_liquidity, has_structure, has_htf_bias])
    
    print(f"\nConfluence: {confluence_score}/7")
    for pillar, met in flags.items():
        marker = "✓" if met else "✗"
        print(f"  [{marker}] {pillar}")
    
    print(f"Quality Factors: {quality_factors}")
    
    # Check if signal is active
    if not has_rr:
        reasons.append("No valid R:R")
    
    if confluence_score < MIN_CONFLUENCE:
        reasons.append(f"Confluence too low ({confluence_score}/{MIN_CONFLUENCE})")
    
    if quality_factors < FTMO_CONFIG.min_quality_factors:
        reasons.append(f"Quality too low ({quality_factors}/{FTMO_CONFIG.min_quality_factors})")
    
    if not entry or not sl or not tp1:
        reasons.append("Missing entry/SL/TP levels")
        return {"symbol": symbol, "tradeable": False, "reasons": reasons}
    
    # Check current price vs entry
    current_price = daily[-1]["close"]
    risk = abs(entry - sl)
    entry_distance = abs(current_price - entry)
    entry_distance_r = entry_distance / risk if risk > 0 else 999
    
    print(f"\nPrice Analysis:")
    print(f"  Current: {current_price:.5f}")
    print(f"  Entry: {entry:.5f}")
    print(f"  Distance: {entry_distance_r:.2f}R")
    print(f"  Max allowed: {FTMO_CONFIG.max_entry_distance_r}R")
    
    if entry_distance_r > FTMO_CONFIG.max_entry_distance_r:
        reasons.append(f"Entry too far ({entry_distance_r:.2f}R > {FTMO_CONFIG.max_entry_distance_r}R)")
    
    # Check SL size
    pip_size = get_pip_size(symbol)
    sl_pips = abs(entry - sl) / pip_size
    
    print(f"\nSL Analysis:")
    print(f"  SL: {sl:.5f}")
    print(f"  SL Pips: {sl_pips:.1f}")
    print(f"  Min: {FTMO_CONFIG.min_sl_pips}, Max: {FTMO_CONFIG.max_sl_pips}")
    
    if sl_pips < FTMO_CONFIG.min_sl_pips:
        reasons.append(f"SL too tight ({sl_pips:.1f} < {FTMO_CONFIG.min_sl_pips} pips)")
    
    if sl_pips > FTMO_CONFIG.max_sl_pips:
        reasons.append(f"SL too wide ({sl_pips:.1f} > {FTMO_CONFIG.max_sl_pips} pips)")
    
    # Check if SL already hit
    if direction == "bullish" and current_price <= sl:
        reasons.append(f"Price already below SL")
    elif direction == "bearish" and current_price >= sl:
        reasons.append(f"Price already above SL")
    
    tradeable = len(reasons) == 0
    
    if tradeable:
        print(f"\n✓ TRADEABLE SETUP FOUND!")
    else:
        print(f"\n✗ NOT TRADEABLE")
        for reason in reasons:
            print(f"  - {reason}")
    
    return {
        "symbol": symbol,
        "tradeable": tradeable,
        "direction": direction,
        "confluence": confluence_score,
        "quality": quality_factors,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "current_price": current_price,
        "entry_distance_r": entry_distance_r,
        "sl_pips": sl_pips,
        "reasons": reasons
    }

def main():
    print(f"\n{'='*70}")
    print("LIVE BOT DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Mode: {SIGNAL_MODE}")
    print(f"Min Confluence: {MIN_CONFLUENCE}/7")
    print(f"Min Quality: {FTMO_CONFIG.min_quality_factors}")
    print(f"Max Entry Distance: {FTMO_CONFIG.max_entry_distance_r}R")
    print(f"SL Range: {FTMO_CONFIG.min_sl_pips}-{FTMO_CONFIG.max_sl_pips} pips")
    print(f"{'='*70}")
    
    tradeable_setups = []
    all_results = []
    
    for symbol in ALL_TRADABLE_FTMO[:10]:  # Test first 10 symbols
        result = diagnose_symbol(symbol)
        all_results.append(result)
        if result["tradeable"]:
            tradeable_setups.append(result)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Symbols tested: {len(all_results)}")
    print(f"Tradeable setups: {len(tradeable_setups)}")
    
    if tradeable_setups:
        print(f"\n✓ TRADEABLE SETUPS:")
        for setup in tradeable_setups:
            print(f"  {setup['symbol']}: {setup['direction'].upper()}, "
                  f"{setup['confluence']}/7 conf, {setup['entry_distance_r']:.2f}R away")
    else:
        print(f"\n✗ NO TRADEABLE SETUPS FOUND")
        print(f"\nMost common rejection reasons:")
        
        reason_counts = {}
        for result in all_results:
            for reason in result.get("reasons", []):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_reasons[:5]:
            print(f"  - {reason} ({count} symbols)")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
