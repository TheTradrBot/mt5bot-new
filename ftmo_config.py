"""
FTMO 10K Challenge Configuration
Ultra-conservative settings optimized for FTMO challenge success
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FTMO10KConfig:
    """
    FTMO 10K Challenge Configuration
    
    Ultra-conservative approach prioritizing:
    1. Safety: Never breach FTMO rules (5% daily, 10% total loss)
    2. Quality: Only take highest-confluence setups (4/7)
    3. Consistency: Gradual profit accumulation over speed
    4. Proven assets: Focus on top-performing instruments
    
    Target: Pass Phase 1 (10%) in 1-2 months, Phase 2 (5%) in 2-4 weeks
    """
    
    # === ACCOUNT SETTINGS ===
    account_size: float = 100000.0  # FTMO 100K challenge size (10K uses 10000.0)
    max_daily_loss_pct: float = 5.0  # FTMO rule: Max 5% daily loss
    max_total_drawdown_pct: float = 10.0  # FTMO rule: Max 10% total drawdown
    profit_target_pct: float = 10.0  # Phase 1 target: 10%
    
    # === ULTRA-CONSERVATIVE RISK SETTINGS ===
    risk_per_trade_pct: float = 0.5  # Risk only 0.5% per trade (ultra-safe)
    max_concurrent_trades: int = 2  # Max 2 simultaneous trades (1% total risk)
    max_cumulative_risk_pct: float = 1.0  # Never exceed 1% total exposure
    max_trades_per_day: int = 3  # Limit daily trades to avoid overtrading
    max_trades_per_week: int = 10  # Weekly limit for quality over quantity
    
    # === SAFETY BUFFERS (Never get close to FTMO limits) ===
    safety_buffer_daily_pct: float = 3.0  # Stop trading at -3% daily loss (vs -5% limit)
    safety_buffer_total_pct: float = 6.0  # Stop trading at -6% total loss (vs -10% limit)
    max_losing_streak: int = 3  # Pause after 3 consecutive losses
    
    # === CONFLUENCE & QUALITY REQUIREMENTS ===
    min_confluence_score: int = 4  # Require 4/7 confluence minimum
    min_quality_factors: int = 2  # Minimum quality factors for entry
    require_trend_confluence: bool = True  # Must have trend agreement
    require_structure_confluence: bool = True  # Must have structure support
    
    # === STOP LOSS VALIDATION (Asset-Specific) ===
    min_sl_pips: float = 15.0  # Global minimum SL (overridden per asset)
    max_sl_pips: float = 200.0  # Global maximum SL (overridden per asset)
    min_sl_atr_ratio: float = 1.0  # SL must be >= 1.0 ATR
    max_sl_atr_ratio: float = 3.0  # SL must be <= 3.0 ATR
    use_atr_validation: bool = True  # Validate SL against ATR
    
    # === TAKE PROFIT SETTINGS ===
    tp1_r_multiple: float = 1.5  # TP1 at 1.5R (50% close)
    tp2_r_multiple: float = 2.5  # TP2 at 2.5R (30% close)
    tp3_r_multiple: float = 4.0  # TP3 at 4.0R (20% close)
    
    # === PARTIAL CLOSE PERCENTAGES ===
    tp1_close_pct: float = 0.50  # Close 50% at TP1
    tp2_close_pct: float = 0.30  # Close 30% at TP2
    tp3_close_pct: float = 0.20  # Close 20% at TP3 (ride winners)
    
    # === ENTRY VALIDATION ===
    max_entry_distance_r: float = 1.2  # Max 1.2R from current price
    immediate_entry_r: float = 0.3  # If entry within 0.3R, enter immediately
    
    # === PENDING ORDER SETTINGS ===
    pending_order_expiry_hours: float = 6.0  # Expire pending orders after 6 hours
    
    # === ASSET WHITELIST (Top Performers) ===
    whitelist_assets: Optional[List[str]] = None  # Set in __post_init__
    
    # === SPREAD & SLIPPAGE ===
    max_spread_pips: float = 3.0  # Skip if spread > 3 pips
    max_slippage_pips: float = 2.0  # Max acceptable slippage
    
    # === TRADING HOURS (24/7 with quality filter) ===
    trading_start_hour: int = 0  # Trade all hours
    trading_end_hour: int = 23  # But skip low-liquidity periods via spread filter
    
    # === POSITION SIZING ===
    use_fixed_lots: bool = False  # Use dynamic position sizing
    min_lot_size: float = 0.01  # Minimum trade size
    max_lot_size: float = 10.0  # Maximum trade size
    
    # === VALIDATION ===
    def __post_init__(self):
        """
        Initialize and validate configuration.
        
        Trade ALL 42 assets that meet confluence requirements.
        Whitelist disabled - confluence filtering (4/7) is sufficient.
        """
        # Trade ALL 42 assets that meet confluence requirements
        # Whitelist disabled - confluence filtering (4/7) is sufficient
        if self.whitelist_assets is None:
            self.whitelist_assets = []  # Empty list = no whitelist, trade all
        
        # Validate FTMO limits
        if self.max_daily_loss_pct > 5.0:
            raise ValueError("Max daily loss cannot exceed 5% for FTMO")
        if self.max_total_drawdown_pct > 10.0:
            raise ValueError("Max total drawdown cannot exceed 10% for FTMO")
        
        # Validate risk settings
        max_total_risk = self.max_concurrent_trades * self.risk_per_trade_pct
        if max_total_risk > self.max_cumulative_risk_pct:
            raise ValueError(
                f"Max total risk ({max_total_risk}%) exceeds limit ({self.max_cumulative_risk_pct}%)"
            )
        
        # Validate partial close percentages sum to 100%
        total_close = self.tp1_close_pct + self.tp2_close_pct + self.tp3_close_pct
        if abs(total_close - 1.0) > 0.01:
            raise ValueError(f"Partial close percentages must sum to 100% (got {total_close*100:.0f}%)")
    
    def is_asset_whitelisted(self, symbol: str) -> bool:
        """
        Check if asset is in whitelist.
        
        Empty whitelist = trade all assets (no restrictions).
        Confluence filtering (4/7) provides sufficient quality control.
        
        Args:
            symbol: Asset symbol to check
            
        Returns:
            True if asset can be traded (always True when whitelist is empty)
        """
        if not self.whitelist_assets:  # Empty list means trade all
            return True
        return symbol in self.whitelist_assets
    
    def get_risk_pct(self, account_balance: float, daily_pnl: float) -> float:
        """
        Calculate risk percentage with dynamic adjustment based on daily P&L.
        
        Reduces risk if approaching daily loss limit.
        
        Args:
            account_balance: Current account balance
            daily_pnl: Today's profit/loss in currency
            
        Returns:
            Risk percentage (0-risk_per_trade_pct)
        """
        daily_loss_pct = abs(daily_pnl / account_balance * 100) if daily_pnl < 0 else 0
        
        # Stop trading if approaching safety buffer
        if daily_loss_pct >= self.safety_buffer_daily_pct:
            return 0.0
        
        # Reduce risk if getting close to buffer
        if daily_loss_pct >= self.safety_buffer_daily_pct * 0.6:  # At 60% of buffer
            return self.risk_per_trade_pct * 0.5  # Half risk
        
        return self.risk_per_trade_pct
    
    def get_max_trades(self, current_trades: int, daily_trades: int, weekly_trades: int) -> int:
        """
        Get remaining trade capacity.
        
        Args:
            current_trades: Number of currently open trades
            daily_trades: Trades taken today
            weekly_trades: Trades taken this week
            
        Returns:
            Number of additional trades allowed (0 if limits reached)
        """
        if current_trades >= self.max_concurrent_trades:
            return 0
        if daily_trades >= self.max_trades_per_day:
            return 0
        if weekly_trades >= self.max_trades_per_week:
            return 0
        
        return min(
            self.max_concurrent_trades - current_trades,
            self.max_trades_per_day - daily_trades,
            self.max_trades_per_week - weekly_trades,
        )


# === PIP SIZE MAPPING (Asset-Specific) ===
PIP_SIZES = {
    # Forex pairs (standard: 0.0001, JPY pairs: 0.01)
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "USD_JPY": 0.01, "USD_CHF": 0.0001,
    "USD_CAD": 0.0001, "AUD_USD": 0.0001, "NZD_USD": 0.0001,
    "EUR_GBP": 0.0001, "EUR_JPY": 0.01, "EUR_CHF": 0.0001, "EUR_AUD": 0.0001,
    "EUR_CAD": 0.0001, "EUR_NZD": 0.0001,
    "GBP_JPY": 0.01, "GBP_CHF": 0.0001, "GBP_AUD": 0.0001, "GBP_CAD": 0.0001,
    "GBP_NZD": 0.0001,
    "AUD_JPY": 0.01, "AUD_CHF": 0.0001, "AUD_CAD": 0.0001, "AUD_NZD": 0.0001,
    "NZD_JPY": 0.01, "NZD_CHF": 0.0001, "NZD_CAD": 0.0001,
    "CAD_JPY": 0.01, "CAD_CHF": 0.0001, "CHF_JPY": 0.01,
    
    # Metals (0.01 for gold/silver)
    "XAU_USD": 0.01, "XAG_USD": 0.01,
    
    # Crypto (0.01)
    "BTC_USD": 0.01, "ETH_USD": 0.01,
    
    # Indices (0.01)
    "SPX500_USD": 0.01, "NAS100_USD": 0.01,
}


def get_pip_size(symbol: str) -> float:
    """
    Get pip size for a symbol.
    
    Args:
        symbol: OANDA format symbol (e.g., "EUR_USD")
        
    Returns:
        Pip size for the symbol
    """
    return PIP_SIZES.get(symbol, 0.0001)  # Default to 0.0001 if not found


def get_sl_limits(symbol: str) -> tuple[float, float]:
    """
    Get stop loss limits (min, max) in pips for a symbol.
    
    Different asset classes have different volatility profiles.
    These limits ensure stops are neither too tight nor too wide.
    
    Args:
        symbol: OANDA format symbol (e.g., "EUR_USD")
        
    Returns:
        Tuple of (min_sl_pips, max_sl_pips)
    """
    # Forex majors: Tighter stops (15-100 pips)
    if symbol in ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD"]:
        return (15.0, 100.0)
    
    # Forex crosses: Medium stops (20-120 pips)
    if "_" in symbol and symbol not in ["XAU_USD", "XAG_USD", "BTC_USD", "ETH_USD", "SPX500_USD", "NAS100_USD"]:
        return (20.0, 120.0)
    
    # Metals: Wider stops due to volatility
    if symbol in ["XAU_USD", "XAG_USD"]:
        return (30.0, 200.0)
    
    # Crypto: Very wide stops due to high volatility
    if symbol in ["BTC_USD", "ETH_USD"]:
        return (50.0, 500.0)
    
    # Indices: Wide stops
    if symbol in ["SPX500_USD", "NAS100_USD"]:
        return (30.0, 200.0)
    
    # Default for unknown symbols
    return (20.0, 150.0)


# === SINGLETON INSTANCE ===
FTMO_CONFIG = FTMO10KConfig()
