"""
Dukascopy Historical Data Downloader

Downloads OHLCV data for all 34 trading assets from Dukascopy for 2023-2025.
Timeframes: Monthly (MN), Weekly (W1), Daily (D1), 4-Hour (H4)

Usage: python download_dukascopy_data.py
"""

import os
import json
import struct
import lzma
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

DUKASCOPY_SYMBOLS = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "USDCAD": "USDCAD",
    "AUDUSD": "AUDUSD",
    "NZDUSD": "NZDUSD",
    "EURGBP": "EURGBP",
    "EURJPY": "EURJPY",
    "EURCHF": "EURCHF",
    "EURAUD": "EURAUD",
    "EURCAD": "EURCAD",
    "EURNZD": "EURNZD",
    "GBPJPY": "GBPJPY",
    "GBPCHF": "GBPCHF",
    "GBPAUD": "GBPAUD",
    "GBPCAD": "GBPCAD",
    "GBPNZD": "GBPNZD",
    "AUDJPY": "AUDJPY",
    "AUDCHF": "AUDCHF",
    "AUDCAD": "AUDCAD",
    "AUDNZD": "AUDNZD",
    "NZDJPY": "NZDJPY",
    "NZDCHF": "NZDCHF",
    "NZDCAD": "NZDCAD",
    "CADJPY": "CADJPY",
    "CADCHF": "CADCHF",
    "CHFJPY": "CHFJPY",
    "XAUUSD": "XAUUSD",
    "XAGUSD": "XAGUSD",
    "BTCUSD": "BTCUSD",
    "ETHUSD": "ETHUSD",
    "USA500IDXUSD": "SPX500",
    "USATECHIDXUSD": "NAS100",
}

ASSETS_34 = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
    "XAUUSD", "XAGUSD",
    "BTCUSD", "ETHUSD",
    "SPX500", "NAS100",
]

SYMBOL_TO_DUKASCOPY = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "USDCAD": "USDCAD",
    "AUDUSD": "AUDUSD",
    "NZDUSD": "NZDUSD",
    "EURGBP": "EURGBP",
    "EURJPY": "EURJPY",
    "EURCHF": "EURCHF",
    "EURAUD": "EURAUD",
    "EURCAD": "EURCAD",
    "EURNZD": "EURNZD",
    "GBPJPY": "GBPJPY",
    "GBPCHF": "GBPCHF",
    "GBPAUD": "GBPAUD",
    "GBPCAD": "GBPCAD",
    "GBPNZD": "GBPNZD",
    "AUDJPY": "AUDJPY",
    "AUDCHF": "AUDCHF",
    "AUDCAD": "AUDCAD",
    "AUDNZD": "AUDNZD",
    "NZDJPY": "NZDJPY",
    "NZDCHF": "NZDCHF",
    "NZDCAD": "NZDCAD",
    "CADJPY": "CADJPY",
    "CADCHF": "CADCHF",
    "CHFJPY": "CHFJPY",
    "XAUUSD": "XAUUSD",
    "XAGUSD": "XAGUSD",
    "BTCUSD": "BTCUSD",
    "ETHUSD": "ETHUSD",
    "SPX500": "USA500IDXUSD",
    "NAS100": "USATECHIDXUSD",
}

POINT_VALUES = {
    "EURUSD": 0.00001, "GBPUSD": 0.00001, "AUDUSD": 0.00001, "NZDUSD": 0.00001,
    "USDCHF": 0.00001, "USDCAD": 0.00001,
    "USDJPY": 0.001, "EURJPY": 0.001, "GBPJPY": 0.001, "AUDJPY": 0.001,
    "NZDJPY": 0.001, "CADJPY": 0.001, "CHFJPY": 0.001,
    "EURGBP": 0.00001, "EURCHF": 0.00001, "EURAUD": 0.00001, "EURCAD": 0.00001,
    "EURNZD": 0.00001,
    "GBPCHF": 0.00001, "GBPAUD": 0.00001, "GBPCAD": 0.00001, "GBPNZD": 0.00001,
    "AUDCHF": 0.00001, "AUDCAD": 0.00001, "AUDNZD": 0.00001,
    "NZDCHF": 0.00001, "NZDCAD": 0.00001, "CADCHF": 0.00001,
    "XAUUSD": 0.01, "XAGUSD": 0.001,
    "BTCUSD": 0.01, "ETHUSD": 0.01,
    "SPX500": 0.01, "NAS100": 0.01,
    "USA500IDXUSD": 0.01, "USATECHIDXUSD": 0.01,
}

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
CACHE_DIR = Path("data_cache/dukascopy")
OUTPUT_DIR = Path("data/ohlcv")


def ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_point_value(symbol: str) -> float:
    if symbol in POINT_VALUES:
        return POINT_VALUES[symbol]
    if "JPY" in symbol:
        return 0.001
    return 0.00001


def parse_bi5(data: bytes, symbol: str) -> List[Dict]:
    try:
        decompressed = lzma.decompress(data)
    except Exception:
        return []

    ticks = []
    record_size = 20
    point = get_point_value(symbol)

    for i in range(0, len(decompressed), record_size):
        if i + record_size > len(decompressed):
            break

        record = decompressed[i:i+record_size]

        try:
            timestamp_ms, ask, bid, ask_vol, bid_vol = struct.unpack('>IIIff', record)
            ticks.append({
                "timestamp_ms": timestamp_ms,
                "ask": ask * point,
                "bid": bid * point,
                "ask_volume": ask_vol,
                "bid_volume": bid_vol,
            })
        except Exception:
            continue

    return ticks


def download_hour(symbol: str, day: date, hour: int) -> List[Dict]:
    duk_symbol = SYMBOL_TO_DUKASCOPY.get(symbol, symbol)
    url = (
        f"{BASE_URL}/{duk_symbol}/"
        f"{day.year}/{day.month - 1:02d}/{day.day:02d}/"
        f"{hour:02d}h_ticks.bi5"
    )

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200 and len(response.content) > 0:
            hour_ticks = parse_bi5(response.content, symbol)

            base_time = datetime(
                day.year, day.month, day.day, hour,
                tzinfo=timezone.utc
            )

            for tick in hour_ticks:
                tick_time = base_time + timedelta(milliseconds=tick["timestamp_ms"])
                tick["time"] = tick_time
                tick["mid"] = (tick["ask"] + tick["bid"]) / 2

            return hour_ticks
    except Exception as e:
        pass

    return []


def download_day(symbol: str, day: date, verbose: bool = False) -> List[Dict]:
    all_ticks = []

    for hour in range(24):
        hour_ticks = download_hour(symbol, day, hour)
        all_ticks.extend(hour_ticks)
        time.sleep(0.05)

    if verbose and all_ticks:
        print(f"  {symbol} {day}: {len(all_ticks)} ticks")

    return all_ticks


def ticks_to_ohlcv(ticks: List[Dict], timeframe: str) -> pd.DataFrame:
    if not ticks:
        return pd.DataFrame()

    tf_map = {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1D",
        "W1": "1W-MON",
        "MN": "1ME",
    }

    freq = tf_map.get(timeframe, "1D")

    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)

    ohlcv = df['mid'].resample(freq).ohlc()
    ohlcv['volume'] = df['ask_volume'].resample(freq).sum()
    ohlcv.dropna(inplace=True)

    ohlcv.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    return ohlcv


def download_symbol_data(
    symbol: str,
    start_date: date,
    end_date: date,
    use_cache: bool = True,
    verbose: bool = True
) -> List[Dict]:
    duk_symbol = SYMBOL_TO_DUKASCOPY.get(symbol, symbol)
    all_ticks = []

    current = start_date
    total_days = (end_date - start_date).days + 1
    days_done = 0

    while current <= end_date:
        cache_path = CACHE_DIR / symbol / f"{current.year}" / f"{current.month:02d}" / f"{current.day:02d}.json"

        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    day_ticks = json.load(f)
                for tick in day_ticks:
                    tick['time'] = datetime.fromisoformat(tick['time'].replace('Z', '+00:00'))
                all_ticks.extend(day_ticks)
                days_done += 1
                current += timedelta(days=1)
                continue
            except Exception:
                pass

        day_ticks = download_day(symbol, current, verbose=False)

        if day_ticks:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                cache_data = []
                for t in day_ticks:
                    cache_data.append({
                        'time': t['time'].isoformat(),
                        'ask': t['ask'],
                        'bid': t['bid'],
                        'mid': t['mid'],
                        'ask_volume': t['ask_volume'],
                        'bid_volume': t['bid_volume'],
                    })
                with open(cache_path, 'w') as f:
                    json.dump(cache_data, f)
            except Exception:
                pass

            all_ticks.extend(day_ticks)

        days_done += 1
        if verbose and days_done % 30 == 0:
            print(f"  {symbol}: {days_done}/{total_days} days downloaded")

        current += timedelta(days=1)
        time.sleep(0.1)

    if verbose:
        print(f"  {symbol}: Total {len(all_ticks)} ticks from {start_date} to {end_date}")

    return all_ticks


def save_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str, start_year: int, end_year: int):
    if df.empty:
        return

    output_path = OUTPUT_DIR / f"{symbol}_{timeframe}_{start_year}_{end_year}.csv"
    df.to_csv(output_path)
    print(f"  Saved: {output_path} ({len(df)} candles)")


def download_and_process_symbol(
    symbol: str,
    start_date: date,
    end_date: date,
    timeframes: List[str],
    use_cache: bool = True
):
    print(f"\n{'='*60}")
    print(f"Processing {symbol}")
    print(f"{'='*60}")

    all_ticks = download_symbol_data(symbol, start_date, end_date, use_cache=use_cache)

    if not all_ticks:
        print(f"  No data available for {symbol}")
        return

    for tf in timeframes:
        ohlcv = ticks_to_ohlcv(all_ticks, tf)
        if not ohlcv.empty:
            save_ohlcv(ohlcv, symbol, tf, start_date.year, end_date.year)


def main():
    ensure_dirs()

    print("=" * 70)
    print("DUKASCOPY HISTORICAL DATA DOWNLOADER")
    print("=" * 70)
    print(f"Assets: {len(ASSETS_34)}")
    print(f"Date range: 2023-01-01 to 2025-12-16")
    print(f"Timeframes: MN (Monthly), W1 (Weekly), D1 (Daily), H4 (4-Hour)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    start_date = date(2023, 1, 1)
    end_date = date(2025, 12, 16)
    timeframes = ["MN", "W1", "D1", "H4"]

    successful = []
    failed = []

    for i, symbol in enumerate(ASSETS_34, 1):
        print(f"\n[{i}/{len(ASSETS_34)}] Processing {symbol}...")
        try:
            download_and_process_symbol(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframes=timeframes,
                use_cache=True
            )
            successful.append(symbol)
        except Exception as e:
            print(f"  ERROR: Failed to download {symbol}: {e}")
            failed.append(symbol)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Successful: {len(successful)} assets")
    print(f"Failed: {len(failed)} assets")
    if failed:
        print(f"Failed symbols: {failed}")
    print(f"\nData saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
