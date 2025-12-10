
#!/usr/bin/env python3
"""
Quick backtest of main_live_bot.py for the last week.
"""

from datetime import datetime, timedelta
from backtest_live_bot import backtest_live_bot

def main():
    # Calculate last week's date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING MAIN_LIVE_BOT.PY")
    print(f"Period: Last 7 Days")
    print(f"Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"End: {end_date.strftime('%Y-%m-%d')}")
    print(f"{'='*80}\n")
    
    # Run the backtest (uses FTMO whitelist symbols from ftmo_config.py)
    result = backtest_live_bot(
        start_date=start_date,
        end_date=end_date,
        symbols=None  # Uses FTMO_CONFIG.whitelist_assets
    )
    
    if result.get("total_trades", 0) == 0:
        print("\n⚠️  No trades found in the last week")
        print("\nPossible reasons:")
        print("  - Market was quiet/ranging")
        print("  - No setups met minimum confluence requirements")
        print("  - Confluence threshold may be too high for short period")
        return
    
    # Print summary
    print("\n" + "="*80)
    print("WEEKLY BACKTEST SUMMARY")
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
        
        if challenge.rule_violations:
            print(f"\n⚠️  Rule Violations:")
            for violation in challenge.rule_violations[:3]:
                print(f"    - {violation}")
    
    print("="*80)


if __name__ == "__main__":
    main()
