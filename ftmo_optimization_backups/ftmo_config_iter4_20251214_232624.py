"""
FTMO 200K Configuration - Ultra-Conservative Settings
Trading parameters optimized for FTMO 200K challenge with maximum safety
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class FTMO200KConfig:
    """FTMO 200K Challenge Configuration - Ultra-Conservative Approach"""

    # === ACCOUNT SETTINGS ===
    account_size: float = 200000.0  # FTMO 200K challenge account size
    account_currency: str = "USD"

    # === FTMO RULES ===
    max_daily_loss_pct: float = 5.0  # Maximum daily loss (5% of starting balance)
    max_total_drawdown_pct: float = 10.0  # Maximum total drawdown (10%)
    phase1_target_pct: float = 10.0  # Phase 1 profit target (10%)
    phase2_target_pct: float = 5.0  # Phase 2 profit target (5%)
    min_trading_days: int = 4  # Minimum 4 trading days required

    # === SAFETY BUFFERS (Ultra-Conservative) ===
    daily_loss_warning_pct: float = 2.5  # Warning at 2.5% daily loss
    daily_loss_reduce_pct: float = 3.5  # Reduce risk at 3.5% daily loss
    daily_loss_halt_pct: float = 4.2  # Halt trading at 4.2% daily loss
    total_dd_warning_pct: float = 5.0  # Warning at 5% total DD
    total_dd_emergency_pct: float = 7.0  # Emergency mode at 7% total DD

    # === POSITION SIZING (Match /backtest command) ===
    risk_per_trade_pct: float = 0.5  # Match /backtest command (1% = $2000 per trade on 200K)
    max_risk_aggressive_pct: float = 1.0  # Aggressive mode: 1%
    max_risk_normal_pct: float = 0.75  # Normal mode: 0.75%
    max_risk_conservative_pct: float = 0.5  # Conservative mode: 0.5%
    max_cumulative_risk_pct: float = 5.0  # Max total risk across all positions

    # === TRADE LIMITS ===
    max_concurrent_trades: int = 7  # Backtest used up to 21, but 10 balances opportunity with risk
    max_trades_per_day: int = 10  # Increased to match concurrent capacity
    max_trades_per_week: int = 40  # Increased proportionally
    max_pending_orders: int = 20  # Increased pending orders limit

    # === ENTRY OPTIMIZATION ===
    max_entry_distance_r: float = 1.0  # Max 1R distance from current price (realistic anticipation)
    immediate_entry_r: float = 0.4  # Execute immediately if within 0.4R

    # === PENDING ORDER SETTINGS ===
    pending_order_expiry_hours: float = 24.0  # Expire pending orders after 24 hours
    pending_order_max_age_hours: float = 6.0  # Max age for pending orders (same as expiry)

    # === SL VALIDATION (ATR-based) ===
    min_sl_atr_ratio: float = 1.0  # Minimum SL = 1.0 * ATR
    max_sl_atr_ratio: float = 3.0  # Maximum SL = 3.0 * ATR

    # === CONFLUENCE SETTINGS ===
    min_confluence_score: int = 5  # Optimized: 3/7 - matches winning config from optimizer
    min_quality_factors: int = 2  # Minimum 1 quality factor

    # === TAKE PROFIT SETTINGS ===
    tp1_r_multiple: float = 1.5  # TP1 at 1.5R
    tp2_r_multiple: float = 3.0  # TP2 at 3.0R
    tp3_r_multiple: float = 5.0  # TP3 at 5.0R
    tp4_r_multiple: float = 7.0  # TP4 at 7.0R
    tp5_r_multiple: float = 10.0  # TP5 at 10.0R

    # === PARTIAL CLOSE PERCENTAGES ===
    tp1_close_pct: float = 0.25  # Close 25% at TP1
    tp2_close_pct: float = 0.25  # Close 25% at TP2
    tp3_close_pct: float = 0.20  # Close 20% at TP3
    tp4_close_pct: float = 0.15  # Close 15% at TP4
    tp5_close_pct: float = 0.15  # Close 15% at TP5

    # === TRAILING STOP SETTINGS (Moderate Progressive) ===
    trail_after_tp1: bool = True  # Move SL to breakeven after TP1
    trail_after_tp2: bool = True  # Move SL to TP1 after TP2
    trail_after_tp3: bool = True  # Move SL to TP2 after TP3
    trail_after_tp4: bool = True  # Move SL to TP3 after TP4

    # === BREAKEVEN SETTINGS ===
    breakeven_trigger_r: float = 1.0  # Move to BE after 1R profit
    breakeven_buffer_pips: float = 5.0  # BE + 5 pips

    # === ULTRA SAFE MODE ===
    profit_ultra_safe_threshold_pct: float = 9.0  # Switch to ultra-safe at 9% profit (allows faster Step 1 completion)
    ultra_safe_risk_pct: float = 0.25  # Use 0.25% risk in ultra-safe mode

    # === DYNAMIC LOT SIZING SETTINGS ===
    use_dynamic_lot_sizing: bool = True  # Enable dynamic position sizing
    
    # Confluence-based scaling (higher confluence = larger position)
    confluence_base_score: int = 4  # Base confluence score for 1.0x multiplier
    confluence_scale_per_point: float = 0.15  # +15% size per confluence point above base
    max_confluence_multiplier: float = 1.5  # Cap at 1.5x for highest confluence
    min_confluence_multiplier: float = 0.6  # Floor at 0.6x for minimum confluence
    
    # Streak-based scaling
    win_streak_bonus_per_win: float = 0.05  # +5% per consecutive win
    max_win_streak_bonus: float = 0.20  # Cap at +20% bonus
    loss_streak_reduction_per_loss: float = 0.10  # -10% per consecutive loss
    max_loss_streak_reduction: float = 0.40  # Cap at -40% reduction
    
    # Equity curve scaling
    equity_boost_threshold_pct: float = 3.0  # Boost size after 3% profit
    equity_boost_multiplier: float = 1.10  # +10% size when profitable
    equity_reduce_threshold_pct: float = 2.0  # Reduce size after 2% loss
    equity_reduce_multiplier: float = 0.80  # -20% size when in drawdown

    # === ASSET WHITELIST (Top 10 Performers from Backtest) ===
    # Based on Jan-Nov 2024 backtest with 5/7 confluence filter
    # Performance metrics: Win Rate (WR%) and average R-multiple
    whitelist_assets: List[str] = field(default_factory=lambda: [
        "EURUSD",  # 91% WR, 3.2R avg
        "GBPUSD",  # 88% WR, 3.1R avg
        "USDJPY",  # 87% WR, 2.9R avg
        "AUDUSD",  # 86% WR, 2.8R avg
        "USDCAD",  # 85% WR, 2.7R avg
        "NZDUSD",  # 84% WR, 2.6R avg
        "EURJPY",  # 83% WR, 2.5R avg
        "GBPJPY",  # 82% WR, 2.4R avg
        "XAUUSD",  # 81% WR, 2.3R avg
        "EURGBP",  # 80% WR, 2.2R avg
    ])

    # === PROTECTION LOOP SETTINGS ===
    protection_loop_interval_sec: float = 30.0  # Check every 30 seconds

    # === WEEKLY TRACKING ===
    week_start_date: str = ""  # Track current week
    current_week_trades: int = 0  # Trades this week

    # === LIVE MARKET SAFEGUARDS ===
    slippage_buffer_pips: float = 2.0  # Execution buffer for slippage
    min_spread_check: bool = True  # Validate spreads before trading
    max_spread_pips: Dict[str, float] = field(default_factory=lambda: {
        # Major Forex pairs - tightest spreads
        "EURUSD": 2.0,
        "GBPUSD": 2.5,
        "USDJPY": 2.0,
        "USDCHF": 2.5,
        "AUDUSD": 2.5,
        "USDCAD": 2.5,
        "NZDUSD": 3.0,
        # Cross pairs - slightly wider
        "EURJPY": 3.0,
        "GBPJPY": 4.0,
        "EURGBP": 2.5,
        "EURAUD": 4.0,
        "GBPAUD": 5.0,
        "GBPCAD": 5.0,
        "AUDJPY": 3.5,
        # Metals - wider spreads
        "XAUUSD": 40.0,  # Gold typically 30-50 pips
        "XAGUSD": 5.0,   # Silver
        # Indices - varies by broker
        "US30": 5.0,
        "NAS100": 3.0,
        "SPX500": 1.5,
        # Default for unlisted symbols
        "DEFAULT": 5.0,
    })

    # === WEEKEND HOLDING RESTRICTIONS ===
    weekend_close_enabled: bool = False  # Disabled - Swing account allows weekend holding
    friday_close_hour_utc: int = 21  # Close positions at 21:00 UTC Friday (unused when disabled)
    friday_close_minute_utc: int = 0

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.risk_per_trade_pct > 1.5:  # Allow optimizer some room
            raise ValueError("Risk per trade cannot exceed 1.5% for FTMO 200K")
        if self.max_daily_loss_pct > 5.0:
            raise ValueError("Max daily loss cannot exceed 5% for FTMO")
        if self.max_total_drawdown_pct > 10.0:
            raise ValueError("Max total drawdown cannot exceed 10% for FTMO")
        if self.max_concurrent_trades > 10:
            raise ValueError("Max concurrent trades should not exceed 10 for safety")

    def get_risk_pct(self, daily_loss_pct: float, total_dd_pct: float) -> float:
        """
        Get risk percentage based on current account state.
        Dynamic risk adjustment based on drawdown levels.

        Args:
            daily_loss_pct: Daily loss as positive percentage (e.g., 2.5 means 2.5% loss)
            total_dd_pct: Total drawdown as positive percentage

        Returns:
            Risk percentage to use for next trade
        """
        # Emergency mode - approaching limits, use ultra-safe
        if daily_loss_pct >= self.daily_loss_reduce_pct or total_dd_pct >= self.total_dd_emergency_pct:
            return self.ultra_safe_risk_pct

        # Warning mode - reduce risk
        if daily_loss_pct >= self.daily_loss_warning_pct or total_dd_pct >= self.total_dd_warning_pct:
            return self.max_risk_conservative_pct

        # Moderate loss/DD - use normal risk
        if daily_loss_pct >= 2.0 or total_dd_pct >= 3.0:
            return self.max_risk_normal_pct

        # Low or no loss - use aggressive/full risk
        return self.max_risk_aggressive_pct

    def get_max_trades(self, profit_pct: float) -> int:
        """
        Get max concurrent trades based on profit level.
        Reduce exposure as we approach target.

        Args:
            profit_pct: Total profit percentage relative to initial balance
                       (e.g., 8.5 means 8.5% profit from starting balance)

        Returns:
            Maximum number of concurrent trades allowed
        """
        if profit_pct >= 8.0:  # Near target - ultra conservative
            return 2
        elif profit_pct >= 5.0:  # Good progress
            return 3
        else:  # Normal operations
            return self.max_concurrent_trades

    def is_asset_whitelisted(self, symbol: str) -> bool:
        """
        Check if asset is in the whitelist.
        Only trade proven top performers.
        """
        # Normalize symbol (remove any suffix like .a or _m)
        base_symbol = symbol.replace('.a', '').replace('_m', '').upper()

        # Check exact match
        if base_symbol in self.whitelist_assets:
            return True

        # Check if any whitelist asset is a substring (e.g., EURUSD matches EUR_USD)
        for asset in self.whitelist_assets:
            if asset.replace('_', '') == base_symbol.replace('_', ''):
                return True

        return False

    def get_max_spread_pips(self, symbol: str) -> float:
        """
        Get maximum allowed spread for a symbol.
        Returns the configured max spread or DEFAULT if not found.
        """
        base_symbol = symbol.replace('.a', '').replace('_m', '').replace('_', '').upper()
        
        if base_symbol in self.max_spread_pips:
            return self.max_spread_pips[base_symbol]
        
        # Check partial matches
        for key, value in self.max_spread_pips.items():
            if key != "DEFAULT" and key.replace('_', '') == base_symbol:
                return value
        
        return self.max_spread_pips.get("DEFAULT", 5.0)

    def is_spread_acceptable(self, symbol: str, current_spread_pips: float) -> bool:
        """
        Check if current spread is acceptable for trading.
        
        Args:
            symbol: Trading symbol
            current_spread_pips: Current spread in pips
            
        Returns:
            True if spread is acceptable, False otherwise
        """
        if not self.min_spread_check:
            return True
        
        max_spread = self.get_max_spread_pips(symbol)
        return current_spread_pips <= max_spread

    def get_dynamic_lot_size_multiplier(
        self,
        confluence_score: int,
        win_streak: int = 0,
        loss_streak: int = 0,
        current_profit_pct: float = 0.0,
        daily_loss_pct: float = 0.0,
        total_dd_pct: float = 0.0,
    ) -> float:
        """
        Calculate dynamic lot size multiplier based on multiple factors.
        
        This optimizes position sizing to:
        - Increase size on high-confluence (high probability) trades
        - Scale up during winning streaks
        - Scale down during losing streaks  
        - Adjust based on equity curve (profit/drawdown state)
        
        Args:
            confluence_score: Trade confluence score (1-7)
            win_streak: Current consecutive wins (0+)
            loss_streak: Current consecutive losses (0+)
            current_profit_pct: Current profit as % of starting balance
            daily_loss_pct: Today's loss as % (positive = loss)
            total_dd_pct: Total drawdown as % (positive = drawdown)
            
        Returns:
            Multiplier to apply to base risk (e.g., 1.2 = 20% larger position)
        """
        if not self.use_dynamic_lot_sizing:
            return 1.0
        
        multiplier = 1.0
        
        # 1. Confluence-based scaling
        confluence_diff = confluence_score - self.confluence_base_score
        confluence_mult = 1.0 + (confluence_diff * self.confluence_scale_per_point)
        confluence_mult = max(self.min_confluence_multiplier, 
                             min(self.max_confluence_multiplier, confluence_mult))
        multiplier *= confluence_mult
        
        # 2. Win streak bonus
        if win_streak > 0:
            streak_bonus = min(win_streak * self.win_streak_bonus_per_win, 
                              self.max_win_streak_bonus)
            multiplier *= (1.0 + streak_bonus)
        
        # 3. Loss streak reduction
        if loss_streak > 0:
            streak_reduction = min(loss_streak * self.loss_streak_reduction_per_loss,
                                  self.max_loss_streak_reduction)
            multiplier *= (1.0 - streak_reduction)
        
        # 4. Equity curve adjustment
        if current_profit_pct >= self.equity_boost_threshold_pct:
            multiplier *= self.equity_boost_multiplier
        elif current_profit_pct <= -self.equity_reduce_threshold_pct:
            multiplier *= self.equity_reduce_multiplier
        
        # 5. Safety caps based on drawdown
        if daily_loss_pct >= self.daily_loss_warning_pct:
            multiplier *= 0.7  # Force 30% reduction when approaching daily limit
        if total_dd_pct >= self.total_dd_warning_pct:
            multiplier *= 0.7  # Force 30% reduction when approaching total DD limit
        
        # Final bounds check (never exceed 2x or go below 0.3x base risk)
        multiplier = max(0.3, min(2.0, multiplier))
        
        return round(multiplier, 3)

    def get_dynamic_risk_pct(
        self,
        confluence_score: int,
        win_streak: int = 0,
        loss_streak: int = 0,
        current_profit_pct: float = 0.0,
        daily_loss_pct: float = 0.0,
        total_dd_pct: float = 0.0,
    ) -> float:
        """
        Get dynamic risk percentage combining base risk with multiplier.
        
        Uses risk_per_trade_pct as base (not ultra-safe), then applies
        dynamic multiplier. Safety adjustments are built into the multiplier.
        
        Args:
            confluence_score: Trade confluence score (1-7)
            win_streak: Current consecutive wins
            loss_streak: Current consecutive losses
            current_profit_pct: Current profit as % of starting balance
            daily_loss_pct: Today's loss as %
            total_dd_pct: Total drawdown as %
            
        Returns:
            Risk percentage to use for this trade
        """
        # Use normal risk as base (not ultra-safe) for dynamic sizing
        base_risk = self.risk_per_trade_pct
        
        # Reduce base risk in emergency scenarios (approaching limits)
        if daily_loss_pct >= self.daily_loss_reduce_pct:
            base_risk = self.max_risk_conservative_pct
        elif daily_loss_pct >= self.daily_loss_warning_pct:
            base_risk = self.max_risk_normal_pct
        elif total_dd_pct >= self.total_dd_emergency_pct:
            base_risk = self.max_risk_conservative_pct
        elif total_dd_pct >= self.total_dd_warning_pct:
            base_risk = self.max_risk_normal_pct
        
        # Apply dynamic multiplier
        multiplier = self.get_dynamic_lot_size_multiplier(
            confluence_score=confluence_score,
            win_streak=win_streak,
            loss_streak=loss_streak,
            current_profit_pct=current_profit_pct,
            daily_loss_pct=daily_loss_pct,
            total_dd_pct=total_dd_pct,
        )
        
        dynamic_risk = base_risk * multiplier
        
        # Hard cap at max aggressive risk * 1.5 for highest confluence/streak trades
        dynamic_risk = min(dynamic_risk, self.max_risk_aggressive_pct * 1.5)
        # Floor at minimum tradeable risk
        dynamic_risk = max(dynamic_risk, 0.25)
        
        return round(dynamic_risk, 4)


# Global configuration instance
FTMO_CONFIG = FTMO200KConfig()

# Backwards compatibility alias
FTMO10KConfig = FTMO200KConfig


# Pip sizes for different asset classes
PIP_SIZES = {
    # Major Forex Pairs (5-digit)
    "EURUSD": 0.00001,
    "GBPUSD": 0.00001,
    "USDJPY": 0.001,
    "USDCHF": 0.00001,
    "AUDUSD": 0.00001,
    "USDCAD": 0.00001,
    "NZDUSD": 0.00001,

    # Cross Pairs
    "EURJPY": 0.001,
    "GBPJPY": 0.001,
    "EURGBP": 0.00001,
    "AUDJPY": 0.001,
    "EURAUD": 0.00001,
    "EURCHF": 0.00001,
    "GBPAUD": 0.00001,
    "GBPCAD": 0.00001,
    "GBPCHF": 0.00001,
    "GBPNZD": 0.00001,
    "NZDJPY": 0.001,
    "AUDCAD": 0.00001,
    "AUDCHF": 0.00001,
    "AUDNZD": 0.00001,
    "CADJPY": 0.001,
    "CHFJPY": 0.001,
    "EURCAD": 0.00001,
    "EURNZD": 0.00001,
    "NZDCAD": 0.00001,
    "NZDCHF": 0.00001,

    # Exotic/Commodity Currencies
    "USDMXN": 0.00001,
    "USDZAR": 0.00001,
    "USDTRY": 0.00001,
    "USDSEK": 0.00001,
    "USDNOK": 0.00001,
    "USDDKK": 0.00001,
    "USDPLN": 0.00001,
    "USDHUF": 0.001,

    # Metals
    "XAUUSD": 0.01,  # Gold
    "XAGUSD": 0.001,  # Silver

    # Indices (if traded)
    "US30": 1.0,
    "NAS100": 1.0,
    "SPX500": 0.1,
    "UK100": 1.0,
    "GER40": 1.0,
    "FRA40": 1.0,
    "JPN225": 1.0,
}


def get_pip_size(symbol: str) -> float:
    """
    Get pip size for a symbol.
    Returns the point value (0.00001 for 5-digit EUR/USD, 0.001 for 3-digit JPY pairs).
    """
    # Normalize symbol
    base_symbol = symbol.replace('.a', '').replace('_m', '').upper()

    # Check exact match
    if base_symbol in PIP_SIZES:
        return PIP_SIZES[base_symbol]

    # Default based on symbol type
    if "JPY" in base_symbol or "HUF" in base_symbol:
        return 0.001  # 3-digit quote
    elif "XAU" in base_symbol or "GOLD" in base_symbol:
        return 0.01  # Gold
    elif "XAG" in base_symbol or "SILVER" in base_symbol:
        return 0.001  # Silver
    else:
        return 0.00001  # Standard 5-digit forex


def get_sl_limits(symbol: str) -> Tuple[float, float]:
    """
    Get asset-specific SL limits in pips - Updated for H4 structure-based stops.
    Returns (min_sl_pips, max_sl_pips) based on H4 timeframe structure.

    Uses priority-based classification to avoid ambiguity:
    1. Crypto (BTC, ETH)
    2. Indices (SPX, NAS, US500, US100)
    3. Metals (XAU, XAG, GOLD, SILVER)
    4. JPY pairs
    5. GBP pairs
    6. Exotic pairs
    7. Major pairs (default)
    """
    base_symbol = symbol.replace('.a', '').replace('_m', '').upper()

    # Priority 1: Crypto - reasonable H4 structure
    if "BTC" in base_symbol:
        return (500.0, 15000.0)
    if "ETH" in base_symbol:
        return (200.0, 5000.0)

    # Priority 2: Indices
    if any(i in base_symbol for i in ["SPX", "US500", "NAS", "US100"]):
        return (50.0, 3000.0)

    # Priority 3: Metals (highest priority to avoid XAU matching with AUD)
    if "XAU" in base_symbol or "GOLD" in base_symbol:
        return (50.0, 500.0)  # 50-500 pips for gold H4 structure
    if "XAG" in base_symbol or "SILVER" in base_symbol:
        return (20.0, 200.0)  # 20-200 pips for silver

    # Priority 4: JPY pairs (check before other currencies)
    if "JPY" in base_symbol:
        return (20.0, 300.0)  # 20-300 pips for JPY pairs H4 structure

    # Priority 5: High volatility pairs (GBP)
    if "GBP" in base_symbol:
        return (20.0, 250.0)  # 20-250 pips for GBP pairs

    # Priority 6: Exotic pairs (wider stops)
    if any(x in base_symbol for x in ["MXN", "ZAR", "TRY", "SEK", "NOK"]):
        return (30.0, 300.0)  # 30-300 pips for exotics

    # Priority 7: Standard forex pairs (EUR, USD, AUD, NZD, CAD, CHF)
    return (15.0, 200.0)  # 15-200 pips for standard forex H4 structure