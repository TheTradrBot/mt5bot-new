
#!/usr/bin/env python3
"""
Test backtest for all assets - 2024
This uses the same backtest engine as the Discord bot.
"""

from datetime import datetime
from backtest import run_backtest
from config import FOREX_PAIRS, METALS, INDICES, CRYPTO_ASSETS, ACCOUNT_SIZE, RISK_PER_TRADE_PCT

def main():
    # Period for 2024
    period = "Jan 2024 - Dec 2024"
    
    # All configured assets (32 total)
    all_assets = FOREX_PAIRS + METALS + INDICES + CRYPTO_ASSETS
    
    print(f"\n{'='*70}")
    print(f"BACKTEST ALL ASSETS - 2024")
    print(f"{'='*70}")
    print(f"Testing {len(all_assets)} assets")
    print(f"Period: {period}")
    print(f"Account: ${ACCOUNT_SIZE:,.0f}")
    print(f"Risk/Trade: {RISK_PER_TRADE_PCT*100:.1f}%")
    print(f"{'='*70}\n")
    
    all_results = []
    total_trades_all = 0
    total_wins_all = 0
    total_profit_all = 0.0
    
    for i, asset in enumerate(all_assets, 1):
        print(f"[{i}/{len(all_assets)}] {asset}...", end=" ")
        
        try:
            result = run_backtest(asset, period)
            
            total_trades = result.get('total_trades', 0)
            win_rate = result.get('win_rate', 0)
            net_return_pct = result.get('net_return_pct', 0)
            total_profit_usd = result.get('total_profit_usd', 0)
            trades = result.get('trades', [])
            
            if total_trades > 0:
                total_r = sum(t.get('rr', 0) for t in trades)
                
                all_results.append({
                    'asset': asset,
                    'trades': total_trades,
                    'win_rate': win_rate,
                    'total_r': total_r,
                    'return_pct': net_return_pct,
                    'profit_usd': total_profit_usd
                })
                
                total_trades_all += total_trades
                total_wins_all += sum(1 for t in trades if t.get('rr', 0) > 0)
                total_profit_all += total_profit_usd
                
                print(f"{total_trades} trades, {win_rate:.1f}% WR, {total_r:+.2f}R (${total_profit_usd:+,.0f})")
            else:
                print("No trades")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Assets tested: {len(all_assets)}")
    print(f"Assets with trades: {len(all_results)}")
    print(f"Total trades: {total_trades_all}")
    
    if total_trades_all > 0:
        avg_win_rate = (total_wins_all / total_trades_all) * 100
        print(f"Overall win rate: {avg_win_rate:.1f}%")
        print(f"Total profit: ${total_profit_all:+,.2f}")
        print(f"Total return: {(total_profit_all/ACCOUNT_SIZE)*100:+.1f}%")
        
        print(f"\nTop 10 performers:")
        sorted_results = sorted(all_results, key=lambda x: x['profit_usd'], reverse=True)
        for i, res in enumerate(sorted_results[:10], 1):
            print(f"{i:2d}. {res['asset']:<12} {res['trades']:3d} trades, "
                  f"{res['win_rate']:5.1f}% WR, {res['total_r']:+7.2f}R (${res['profit_usd']:+10,.0f})")
    else:
        print("WARNING: No trades found across all assets!")
        print("This suggests a data or configuration issue.")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
