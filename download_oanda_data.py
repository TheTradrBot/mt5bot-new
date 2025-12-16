"""
OANDA Historical Data Downloader

Downloads OHLCV data for all 34 trading assets from OANDA for 2023-2025.
Skips already downloaded files. Run in Shell tab for long downloads.

Usage: python download_oanda_data.py
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from data import get_ohlcv
import pandas as pd

OUTPUT_DIR = Path("data/ohlcv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "XAU_USD", "XAG_USD",
    "BTC_USD", "ETH_USD",
    "SPX500_USD", "NAS100_USD",
]

TIMEFRAMES = ["D", "H4", "W"]  # Removed M (monthly) - OANDA API often hangs/fails on monthly
TF_MAP = {"D": "D1", "H4": "H4", "W": "W1"}

def main():
    print("=" * 70)
    print("OANDA HISTORICAL DATA DOWNLOADER")
    print("=" * 70)
    print(f"Assets: {len(ASSETS)}")
    print(f"Date range: 2023-01-01 to 2025-12-16")
    print(f"Timeframes: D1, H4, W1 (Monthly excluded - OANDA API unreliable)")
    print(f"Skipping already downloaded files")
    print("=" * 70)

    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 12, 16, tzinfo=timezone.utc)

    successful = 0
    skipped = 0
    failed = 0

    for i, asset in enumerate(ASSETS, 1):
        print(f"\n[{i}/{len(ASSETS)}] {asset}")
        
        for tf in TIMEFRAMES:
            symbol = asset.replace("_", "")
            tf_name = TF_MAP[tf]
            output_path = OUTPUT_DIR / f"{symbol}_{tf_name}_2023_2025.csv"
            
            if output_path.exists():
                print(f"  {tf_name}: Already exists, skipping")
                skipped += 1
                continue
            
            try:
                candles = get_ohlcv(
                    instrument=asset,
                    timeframe=tf,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=False
                )
                
                if candles:
                    df = pd.DataFrame(candles)
                    df.set_index('time', inplace=True)
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df.to_csv(output_path)
                    print(f"  {tf_name}: {len(candles)} candles saved")
                    successful += 1
                else:
                    print(f"  {tf_name}: No data")
                    failed += 1
            except Exception as e:
                print(f"  {tf_name}: Error - {e}")
                failed += 1

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print(f"Successful: {successful} files")
    print(f"Skipped (already exist): {skipped} files")
    print(f"Failed: {failed} files")
    print("=" * 70)

if __name__ == "__main__":
    main()
