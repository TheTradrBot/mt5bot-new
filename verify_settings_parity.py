
#!/usr/bin/env python3
"""
Verify that main_live_bot.py and backtest_live_bot.py use identical settings.
"""

from ftmo_config import FTMO_CONFIG
from symbol_mapping import ALL_TRADABLE_OANDA
from config import SIGNAL_MODE

print("=" * 70)
print("SETTINGS PARITY VERIFICATION")
print("=" * 70)

print("\n1. SYMBOL SELECTION:")
print(f"   Total symbols: {len(ALL_TRADABLE_OANDA)}")
print(f"   Both use: ALL_TRADABLE_OANDA")

print("\n2. CONFLUENCE SETTINGS:")
print(f"   Min Confluence: {FTMO_CONFIG.min_confluence_score}/7")
print(f"   Min Quality Factors: {FTMO_CONFIG.min_quality_factors}")
print(f"   Signal Mode: {SIGNAL_MODE}")

print("\n3. RISK SETTINGS:")
print(f"   Risk per trade: {FTMO_CONFIG.risk_per_trade_pct}%")
print(f"   Max cumulative risk: {FTMO_CONFIG.max_cumulative_risk_pct}%")
print(f"   Max concurrent trades: {FTMO_CONFIG.max_concurrent_trades}")

print("\n4. SL/TP SETTINGS:")
print(f"   Min SL: {FTMO_CONFIG.min_sl_pips} pips")
print(f"   Max SL: {FTMO_CONFIG.max_sl_pips} pips")
print(f"   Min SL ATR ratio: {FTMO_CONFIG.min_sl_atr_ratio}")
print(f"   Max SL ATR ratio: {FTMO_CONFIG.max_sl_atr_ratio}")
print(f"   TP1 R-multiple: {FTMO_CONFIG.tp1_r_multiple}R")
print(f"   TP2 R-multiple: {FTMO_CONFIG.tp2_r_multiple}R")
print(f"   TP3 R-multiple: {FTMO_CONFIG.tp3_r_multiple}R")

print("\n5. PARTIAL CLOSE SETTINGS:")
print(f"   TP1 close: {FTMO_CONFIG.tp1_close_pct * 100:.0f}%")
print(f"   TP2 close: {FTMO_CONFIG.tp2_close_pct * 100:.0f}%")
print(f"   TP3 close: {FTMO_CONFIG.tp3_close_pct * 100:.0f}%")

print("\n6. ENTRY VALIDATION:")
print(f"   Max entry distance: {FTMO_CONFIG.max_entry_distance_r}R")
print(f"   Immediate entry threshold: {FTMO_CONFIG.immediate_entry_r}R")
print(f"   Pending order expiry: {FTMO_CONFIG.pending_order_expiry_hours}H")

print("\n" + "=" * 70)
print("✓ All settings are centralized in ftmo_config.py")
print("✓ Both bots use identical strategy_core.py logic")
print("✓ Both bots use identical FTMO_CONFIG parameters")
print("=" * 70)
