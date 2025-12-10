
#!/usr/bin/env python3
"""
Backtest main_live_bot.py for all assets (like Discord /backtest command).
"""

from datetime import datetime, timedelta
from backtest_live_bot import backtest_live_bot
from symbol_mapping import ALL_TRADABLE_OANDA


def main():
    # Jan 2025 - Nov 2025
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 11, 30)
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING MAIN_LIVE_BOT.PY - ALL ASSETS")
    print(f"Period: Jan 2025 - Nov 2025")
    print(f"Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"End: {end_date.strftime('%Y-%m-%d')}")
    print(f"Assets: {len(ALL_TRADABLE_OANDA)} (all tradable)")
    print(f"{'='*80}\n")
    
    # Run the backtest with all tradable assets
    result = backtest_live_bot(
        start_date=start_date,
        end_date=end_date,
        symbols=ALL_TRADABLE_OANDA  # Use all 42 assets
    )
    
    if result.get("total_trades", 0) == 0:
        print("\n⚠️  No trades found in this period")
        print("\nPossible reasons:")
        print("  - Market was quiet/ranging")
        print("  - No setups met minimum confluence requirements")
        print("  - Confluence threshold may be too high for period")
        return
    
    # Print summary
    print("\n" + "="*80)
    print("BACKTEST SUMMARY - ALL ASSETS")
    print("="*80)
    print(f"Period: {result['period']}")
    print(f"Total Trades: {result['total_trades']}")
    print(f"Win Rate: {result['win_rate']:.1f}%")
    print(f"Symbols Traded: {len(result['trades_by_symbol'])}")
    
    challenge = result.get('challenge_result')
    if challenge:
        print(f"\nChallenge Performance:")
        print(f"  Challenges Started: {len(challenge.challenges)}")
        print(f"  Challenges Passed: {challenge.full_challenges_passed}")
        print(f"  Total Profit: ${challenge.total_profit_usd:+,.2f} ({challenge.total_profit_pct:+.1f}%)")
    print("="*80)


if __name__ == "__main__":
    main()
