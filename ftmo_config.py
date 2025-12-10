"""
FTMO 10K Configuration - Ultra-Conservative Settings

This configuration is optimized for the FTMO $10K challenge with ultra-conservative
risk management to maximize pass probability while still achieving profit targets.

Key Safety Features:
- 0.5% risk per trade (ultra-conservative)
- Multiple safety buffers at 1.5%, 2.5%, 3.5% daily loss
- Max 2 concurrent trades, 5 trades per week
- Top 10 asset whitelist based on backtest performance
- Smart position sizing that adapts to drawdown
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class FTMO10KConfig:
    """FTMO 10K Challenge Configuration - Ultra-Conservative"""
    
    # === ACCOUNT SETTINGS ===
    account_size: float = 10000.0  # FTMO 10K challenge account size
    max_daily_loss_pct: float = 5.0  # FTMO hard limit: 5% daily loss
    max_total_drawdown_pct: float = 10.0  # FTMO hard limit: 10% max drawdown
    profit_target_pct: float = 10.0  # FTMO target: 10% profit
    
    # === SAFETY BUFFERS (Multi-layer protection) ===
    daily_loss_warning_pct: float = 1.5  # Start reducing risk at 1.5% daily loss
    daily_loss_reduce_pct: float = 2.5  # Reduce risk to 0.3% at 2.5% daily loss
    daily_loss_halt_pct: float = 3.5  # Stop all trading at 3.5% daily loss
    
    total_dd_warning_pct: float = 3.0  # Start reducing risk at 3% total DD
    total_dd_reduce_pct: float = 5.0  # Reduce risk to 0.3% at 5% total DD
    total_dd_halt_pct: float = 7.0  # Stop all trading at 7% total DD
    
    # === ULTRA-CONSERVATIVE POSITION SIZING ===
    risk_per_trade_pct: float = 0.5  # Ultra-conservative: 0.5% risk per trade
    max_cumulative_risk_pct: float = 2.5  # Max 2.5% total risk across all open trades
    
    # === TRADE LIMITS (Prevent overtrading) ===
    max_concurrent_trades: int = 2  # Max 2 positions open at once
    max_pending_orders: int = 3  # Max 3 pending orders
    max_trades_per_week: int = 5  # Max 5 trades per week (prevents burnout)
    
    # === ENTRY OPTIMIZATION ===
    max_entry_distance_r: float = 1.2  # Max 1.2R from entry (prevent chasing)
    immediate_entry_r: float = 0.8  # Use market order if within 0.8R of entry
    
    # === PENDING ORDER SETTINGS ===
    pending_order_expiry_hours: float = 6.0  # Expire pending orders after 6 hours
    
    # === STOP LOSS VALIDATION (Asset-specific limits) ===
    min_sl_pips: float = 15.0  # Minimum SL in pips (default for major pairs)
    max_sl_pips: float = 150.0  # Maximum SL in pips (default for major pairs)
    min_sl_atr_ratio: float = 1.0  # Minimum SL in ATR terms (1.0x ATR)
    max_sl_atr_ratio: float = 3.0  # Maximum SL in ATR terms (3.0x ATR)
    
    # === TAKE PROFIT TARGETS (R-multiples) ===
    tp1_r_multiple: float = 1.0  # TP1 at 1R
    tp2_r_multiple: float = 2.0  # TP2 at 2R
    tp3_r_multiple: float = 3.0  # TP3 at 3R
    
    # === PARTIAL CLOSE PERCENTAGES ===
    tp1_close_pct: float = 0.45  # Close 45% at TP1
    tp2_close_pct: float = 0.30  # Close 30% at TP2
    tp3_close_pct: float = 0.25  # Close 25% at TP3
    
    # === SIGNAL QUALITY THRESHOLDS ===
    min_confluence_score: int = 5  # Minimum 5/7 confluence pillars
    min_quality_factors: int = 2  # Minimum 2 quality factors
    
    # === PROTECTION LOOP ===
    protection_interval_sec: int = 30  # Run protection checks every 30 seconds
    
    # === ULTRA-SAFE MODE (When profitable) ===
    ultra_safe_profit_threshold_pct: float = 8.0  # Activate ultra-safe mode at 8% profit
    ultra_safe_risk_pct: float = 0.3  # Reduce to 0.3% risk when ultra-safe
    
    # === TOP 10 ASSET WHITELIST (Based on backtest performance) ===
    # Top 10 performers from backtest analysis
    # These assets showed best win rate, drawdown control, and profit factor
    whitelist_assets: List[str] = field(default_factory=lambda: [
        "SPX500_USD",  # S&P 500 - Best overall performance
        "NAS100_USD",  # Nasdaq 100 - Strong trending behavior
        "AUD_NZD",     # Low volatility, mean-reverting
        "GBP_JPY",     # High R-multiples when setup is clean
        "USD_JPY",     # Excellent liquidity, tight spreads
        "XAU_USD",     # Gold - Safe haven moves
        "CAD_JPY",     # Strong trending characteristics
        "CHF_JPY",     # Good for range-bound strategies
        "GBP_CAD",     # Clean structure, good R:R
        "EUR_GBP",     # Tight ranges, good for scalping
    ])
    
    def get_risk_pct(self, daily_loss_pct: float, total_dd_pct: float) -> float:
        """
        Get adaptive risk percentage based on current drawdown.
        
        Risk Modes:
        - Normal: 0.5% (default)
        - Warning: 0.4% (at 1.5% daily or 3% total DD)
        - Reduced: 0.3% (at 2.5% daily or 5% total DD)
        - Ultra-Safe: 0.3% (when profit >= 8%)
        - Halted: 0.0% (at 3.5% daily or 7% total DD)
        """
        # Halt trading if approaching limits
        if daily_loss_pct >= self.daily_loss_halt_pct or total_dd_pct >= self.total_dd_halt_pct:
            return 0.0
        
        # Reduced risk mode
        if daily_loss_pct >= self.daily_loss_reduce_pct or total_dd_pct >= self.total_dd_reduce_pct:
            return 0.3
        
        # Warning mode
        if daily_loss_pct >= self.daily_loss_warning_pct or total_dd_pct >= self.total_dd_warning_pct:
            return 0.4
        
        # Normal mode
        return self.risk_per_trade_pct
    
    def get_max_trades(self, profit_pct: float) -> int:
        """
        Get maximum concurrent trades based on profit level.
        
        Trade Limits:
        - Default: 2 concurrent trades
        - Ultra-Safe: 1 concurrent trade (when profit >= 8%)
        """
        # Ultra-safe mode when close to target
        if profit_pct >= self.ultra_safe_profit_threshold_pct:
            return 1
        
        return self.max_concurrent_trades
    
    def is_asset_whitelisted(self, symbol: str) -> bool:
        """Check if an asset is in the top 10 whitelist."""
        return symbol in self.whitelist_assets


# Module-level instance for easy importing
FTMO_CONFIG = FTMO10KConfig()


# === HELPER FUNCTIONS ===

def get_pip_size(symbol: str) -> float:
    """
    Get pip size for a symbol.
    
    JPY pairs: 1 pip = 0.01
    All others: 1 pip = 0.0001
    """
    if "JPY" in symbol:
        return 0.01
    return 0.0001


def get_sl_limits(symbol: str) -> Tuple[float, float]:
    """
    Get asset-specific SL limits (min_pips, max_pips).
    
    Returns:
        (min_sl_pips, max_sl_pips) for the given symbol
    
    Asset Classes:
    - Major Pairs: 15-80 pips
    - Minor/Exotic Pairs: 20-120 pips
    - Indices: 30-200 pips
    - Commodities: 25-150 pips
    """
    # Indices (higher volatility, wider stops)
    if any(idx in symbol for idx in ["SPX", "NAS", "US30", "DJ30", "GER", "UK100"]):
        return (30.0, 200.0)
    
    # Gold/Silver (commodities)
    if any(metal in symbol for metal in ["XAU", "XAG", "GOLD", "SILVER"]):
        return (25.0, 150.0)
    
    # Exotic pairs (wider spreads, more volatility)
    if any(exotic in symbol for exotic in ["TRY", "ZAR", "MXN", "NOK", "SEK", "SGD", "HKD"]):
        return (25.0, 120.0)
    
    # Minor pairs (not USD-based majors)
    if "USD" not in symbol and any(minor in symbol for minor in ["EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "JPY"]):
        return (20.0, 100.0)
    
    # Major pairs (EUR/USD, GBP/USD, USD/JPY, etc.)
    return (15.0, 80.0)
