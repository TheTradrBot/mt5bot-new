"""
Unified Data Provider for Blueprint Trader AI.

Provides OHLCV data from multiple sources with fallback:
1. Dukascopy cached data (from data/ohlcv/ or data_cache/dukascopy/)
2. OANDA API (if Dukascopy not available)

This ensures the FTMO analyzer and backtesting can work with historical data.
"""

import os
import json
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd

from data import get_ohlcv as get_ohlcv_oanda

OHLCV_DIR = Path("data/ohlcv")
CACHE_DIR = Path("data_cache/dukascopy")

TIMEFRAME_MAP = {
    "M": "MN",
    "MN": "MN",
    "W": "W1",
    "W1": "W1",
    "D": "D1",
    "D1": "D1",
    "H4": "H4",
    "H1": "H1",
}

OANDA_SYMBOL_MAP = {
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "USDJPY": "USD_JPY",
    "USDCHF": "USD_CHF",
    "USDCAD": "USD_CAD",
    "AUDUSD": "AUD_USD",
    "NZDUSD": "NZD_USD",
    "EURGBP": "EUR_GBP",
    "EURJPY": "EUR_JPY",
    "EURCHF": "EUR_CHF",
    "EURAUD": "EUR_AUD",
    "EURCAD": "EUR_CAD",
    "EURNZD": "EUR_NZD",
    "GBPJPY": "GBP_JPY",
    "GBPCHF": "GBP_CHF",
    "GBPAUD": "GBP_AUD",
    "GBPCAD": "GBP_CAD",
    "GBPNZD": "GBP_NZD",
    "AUDJPY": "AUD_JPY",
    "AUDCHF": "AUD_CHF",
    "AUDCAD": "AUD_CAD",
    "AUDNZD": "AUD_NZD",
    "NZDJPY": "NZD_JPY",
    "NZDCHF": "NZD_CHF",
    "NZDCAD": "NZD_CAD",
    "CADJPY": "CAD_JPY",
    "CADCHF": "CAD_CHF",
    "CHFJPY": "CHF_JPY",
    "XAUUSD": "XAU_USD",
    "XAGUSD": "XAG_USD",
    "BTCUSD": "BTC_USD",
    "ETHUSD": "ETH_USD",
    "SPX500": "SPX500_USD",
    "NAS100": "NAS100_USD",
}


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper().replace("_", "").replace(".", "").replace("/", "")
    return symbol


def to_oanda_symbol(symbol: str) -> str:
    norm = normalize_symbol(symbol)
    if norm in OANDA_SYMBOL_MAP:
        return OANDA_SYMBOL_MAP[norm]
    if "_" in symbol:
        return symbol
    if len(norm) == 6:
        return f"{norm[:3]}_{norm[3:]}"
    return symbol


def load_dukascopy_ohlcv(
    symbol: str,
    timeframe: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Dict]:
    norm_symbol = normalize_symbol(symbol)
    norm_tf = TIMEFRAME_MAP.get(timeframe.upper(), timeframe.upper())
    
    for year_range in ["2023_2024", "2024_2024", "2023_2023"]:
        csv_path = OHLCV_DIR / f"{norm_symbol}_{norm_tf}_{year_range}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                candles = []
                for idx, row in df.iterrows():
                    candle_time = idx
                    if isinstance(candle_time, str):
                        candle_time = pd.to_datetime(candle_time)
                    
                    if start_date and candle_time < start_date:
                        continue
                    if end_date and candle_time > end_date:
                        continue
                    
                    candles.append({
                        "time": candle_time.to_pydatetime() if hasattr(candle_time, 'to_pydatetime') else candle_time,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row.get("Volume", 0)),
                    })
                
                if candles:
                    print(f"[DataProvider] Loaded {len(candles)} candles from Dukascopy cache for {norm_symbol} {norm_tf}")
                    return candles
            except Exception as e:
                print(f"[DataProvider] Error loading {csv_path}: {e}")
    
    return []


def get_ohlcv(
    instrument: str,
    timeframe: str = "D",
    count: int = 200,
    use_cache: bool = True,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    prefer_dukascopy: bool = True,
) -> List[Dict]:
    """
    Get OHLCV data from Dukascopy or OANDA.
    
    Args:
        instrument: Trading symbol (OANDA or standard format)
        timeframe: D, H4, W, M, etc.
        count: Number of candles (used only for OANDA fallback)
        use_cache: Whether to use caching for OANDA fallback (default True)
        start_date: Start date for historical range
        end_date: End date for historical range
        prefer_dukascopy: Try Dukascopy first (default True)
    
    Returns:
        List of candle dictionaries
    """
    if prefer_dukascopy:
        dukascopy_data = load_dukascopy_ohlcv(instrument, timeframe, start_date, end_date)
        if dukascopy_data:
            return dukascopy_data
    
    oanda_symbol = to_oanda_symbol(instrument)
    print(f"[DataProvider] Falling back to OANDA for {oanda_symbol} {timeframe}")
    
    try:
        if start_date:
            return get_ohlcv_oanda(
                instrument=oanda_symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
            )
        else:
            return get_ohlcv_oanda(
                instrument=oanda_symbol,
                timeframe=timeframe,
                count=count,
                use_cache=use_cache,
            )
    except Exception as e:
        print(f"[DataProvider] OANDA error for {oanda_symbol}: {e}")
        return []


def get_available_dukascopy_data() -> Dict[str, List[str]]:
    """List all available Dukascopy data files."""
    available = {}
    
    if OHLCV_DIR.exists():
        for csv_file in OHLCV_DIR.glob("*.csv"):
            parts = csv_file.stem.split("_")
            if len(parts) >= 2:
                symbol = parts[0]
                timeframe = parts[1] if len(parts) > 1 else "unknown"
                if symbol not in available:
                    available[symbol] = []
                available[symbol].append(timeframe)
    
    return available


if __name__ == "__main__":
    print("Data Provider - Available Dukascopy Data:")
    available = get_available_dukascopy_data()
    for symbol, timeframes in available.items():
        print(f"  {symbol}: {', '.join(timeframes)}")
    
    print("\nTesting data fetch for EURUSD D1...")
    candles = get_ohlcv("EURUSD", "D", start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 10))
    print(f"Fetched {len(candles)} candles")
    if candles:
        print(f"First candle: {candles[0]}")
