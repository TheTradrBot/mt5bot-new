"""
FTMO Configuration
Trading parameters optimized for FTMO challenge rules
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FTMOConfig:
    """FTMO Challenge Configuration"""
    
    # === ACCOUNT SETTINGS ===
    account_size: float = 100000.0  # FTMO challenge account size
    max_daily_loss: float = 5.0  # Maximum daily loss percentage (5%)
    max_total_loss: float = 10.0  # Maximum total loss percentage (10%)
    profit_target: float = 10.0  # Profit target percentage (10%)
    
    # === RISK MANAGEMENT ===
    risk_per_trade: float = 1.0  # Risk 1% per trade
    max_open_trades: int = 3  # Maximum number of open trades
    max_daily_trades: int = 5  # Maximum trades per day
    
    # === TRADING HOURS (UTC) ===
    trading_start_hour: int = 0  # Start trading at midnight UTC
    trading_end_hour: int = 23  # End trading at 11 PM UTC
    
    # === SPREAD & SLIPPAGE ===
    max_spread: float = 2.0  # Maximum spread in pips
    slippage: int = 3  # Maximum slippage in points
    
    # === POSITION SIZING ===
    use_fixed_lots: bool = False  # Use dynamic lot sizing
    fixed_lot_size: float = 0.01  # Fixed lot size if enabled
    
    # === STOP LOSS & TAKE PROFIT ===
    default_sl_pips: float = 50.0  # Default stop loss in pips
    default_tp_pips: float = 100.0  # Default take profit in pips
    use_trailing_stop: bool = True  # Enable trailing stop
    trailing_stop_pips: float = 30.0  # Trailing stop distance in pips
    
    # === TIMEFRAMES ===
    primary_timeframe: str = "H1"  # Primary analysis timeframe
    secondary_timeframe: str = "H4"  # Secondary analysis timeframe
    
    # === IMMEDIATE ENTRY SETTINGS ===
    immediate_entry_r: float = 1.5  # Risk-to-reward ratio for immediate entries
    
    # === PENDING ORDER SETTINGS ===
    pending_order_expiry_hours: float = 6.0  # Expire pending orders after 6 hours
    
    # === NEWS FILTER ===
    avoid_news: bool = True  # Avoid trading during high-impact news
    news_buffer_minutes: int = 30  # Minutes before/after news to avoid trading
    
    # === VALIDATION ===
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.risk_per_trade > 2.0:
            raise ValueError("Risk per trade cannot exceed 2% for FTMO")
        if self.max_daily_loss > 5.0:
            raise ValueError("Max daily loss cannot exceed 5% for FTMO")
        if self.max_total_loss > 10.0:
            raise ValueError("Max total loss cannot exceed 10% for FTMO")
