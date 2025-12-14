"""
Pre-download OHLCV data from Oanda and cache as CSVs.
This allows the FTMO analyzer to run without API calls.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from tradr.data.oanda import OandaClient
from config import FOREX_PAIRS, METALS, INDICES, CRYPTO_ASSETS

OHLCV_DIR = Path("data/ohlcv")
OHLCV_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAMES = ["H4", "D1", "W1", "MN"]
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2024, 12, 31, tzinfo=timezone.utc)

SYMBOLS = FOREX_PAIRS + METALS + INDICES + CRYPTO_ASSETS

def main():
    oanda = OandaClient()
    
    if not oanda.api_key:
        print("ERROR: OANDA_API_KEY not set")
        return
    
    print("=" * 60)
    print("PRE-DOWNLOADING OHLCV DATA FROM OANDA")
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Timeframes: {TIMEFRAMES}")
    print("=" * 60)
    
    total = len(SYMBOLS) * len(TIMEFRAMES)
    done = 0
    
    for symbol in SYMBOLS:
        norm_symbol = symbol.upper().replace("_", "").replace("/", "")
        
        for tf in TIMEFRAMES:
            done += 1
            print(f"\n[{done}/{total}] {norm_symbol} {tf}...")
            
            oanda_tf = tf.replace("1", "")
            
            try:
                candles = oanda.get_candles(
                    symbol, 
                    oanda_tf, 
                    from_time=START_DATE, 
                    to_time=END_DATE
                )
                
                if not candles:
                    print(f"  No data returned")
                    continue
                
                df = pd.DataFrame(candles)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                csv_path = OHLCV_DIR / f"{norm_symbol}_{tf}_2023_2024.csv"
                df.to_csv(csv_path)
                print(f"  Saved {len(candles)} candles to {csv_path}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    
    files = list(OHLCV_DIR.glob("*.csv"))
    print(f"Total CSV files created: {len(files)}")
    
    print("\nThe FTMO analyzer will now use cached data instead of API calls.")

if __name__ == "__main__":
    main()
