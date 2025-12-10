"""
FTMO 10K Challenge Configuration
Optimized settings to pass challenge while replicating backtest performance.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class FTMO10KConfig:
    """FTMO 10K Challenge optimized configuration."""

    # === ACCOUNT SETTINGS ===
    account_size: float = 10000.0
    account_currency: str = "USD"

    # === FTMO RULES (HARD LIMITS - NEVER BREACH) ===
    max_daily_loss_pct: float = 5.0
    max_total_drawdown_pct: float = 10.0
    phase1_target_pct: float = 10.0
    phase2_target_pct: float = 5.0

    # === SAFETY BUFFERS (Stop BEFORE breach) ===
    daily_loss_warning_pct: float = 2.5
    daily_loss_reduce_pct: float = 3.5
    daily_loss_halt_pct: float = 4.2
    total_dd_warning_pct: float = 5.0
    total_dd_reduce_pct: float = 7.0
    total_dd_halt_pct: float = 8.5

    # === POSITION SIZING (Conservative for 10K) ===
    risk_per_trade_pct: float = 0.75
    risk_per_trade_reduced_pct: float = 0.4
    risk_per_trade_minimal_pct: float = 0.2
    max_cumulative_risk_pct: float = 2.5

    # === TRADE LIMITS ===
    max_concurrent_trades: int = 4
    max_pending_orders: int = 5
    max_trades_per_day: int = 8
    max_trades_per_symbol: int = 1

    # === ENTRY OPTIMIZATION (Match Backtest) ===
    max_entry_distance_r: float = 1.5  # Max distance from current price to entry (in R) - LOOSENED from 0.5
    immediate_entry_r: float = 0.2  # If within this range, use market order instead of pending

    # SL validation (in pips)
    min_sl_pips: float = 10.0  # Minimum SL size - LOOSENED from 15.0
    max_sl_pips: float = 150.0  # Maximum SL size - LOOSENED from 100.0
    min_sl_atr_ratio: float = 0.3  # Min SL as ratio of ATR - LOOSENED from 0.5
    max_sl_atr_ratio: float = 3.0  # Max SL as ratio of ATR - LOOSENED from 2.5

    # === TAKE PROFIT SETTINGS (Lock profits fast) ===
    tp1_r_multiple: float = 1.0
    tp2_r_multiple: float = 2.0
    tp3_r_multiple: float = 3.0

    # === PARTIAL CLOSE PERCENTAGES ===
    tp1_close_pct: float = 0.40
    tp2_close_pct: float = 0.35
    tp3_close_pct: float = 0.25

    # === BREAKEVEN SETTINGS ===
    move_sl_to_be_after_tp1: bool = True
    be_buffer_pips: float = 2.0

    # === ULTRA SAFE MODE (When close to target) ===
    ultra_safe_profit_threshold_pct: float = 8.0
    ultra_safe_risk_pct: float = 0.25
    ultra_safe_max_trades: int = 2

    # === CONFLUENCE SETTINGS (Same as backtest) ===
    min_confluence_score: int = 4
    min_quality_factors: int = 1
    require_rr_flag: bool = True
    require_confirmation: bool = False

    # === PROTECTION LOOP ===
    protection_interval_sec: float = 20.0

    def get_risk_pct(self, daily_loss_pct: float, total_dd_pct: float) -> float:
        """Get adjusted risk percentage based on current drawdown."""
        if daily_loss_pct >= self.daily_loss_halt_pct or total_dd_pct >= self.total_dd_halt_pct:
            return 0.0
        elif daily_loss_pct >= self.daily_loss_reduce_pct or total_dd_pct >= self.total_dd_reduce_pct:
            return self.risk_per_trade_minimal_pct
        elif daily_loss_pct >= self.daily_loss_warning_pct or total_dd_pct >= self.total_dd_warning_pct:
            return self.risk_per_trade_reduced_pct
        else:
            return self.risk_per_trade_pct

    def get_max_trades(self, profit_pct: float) -> int:
        """Get max concurrent trades based on profit level."""
        if profit_pct >= self.ultra_safe_profit_threshold_pct:
            return self.ultra_safe_max_trades
        return self.max_concurrent_trades


FTMO_CONFIG = FTMO10KConfig()

PIP_SIZES = {
    "forex_jpy": 0.01,
    "forex_standard": 0.0001,
    "xauusd": 0.1,
    "xagusd": 0.01,
    "indices": 1.0,
    "crypto": 1.0,
}


def get_pip_size(symbol: str) -> float:
    """Get pip size for a symbol."""
    s = symbol.upper()
    if "JPY" in s:
        return PIP_SIZES["forex_jpy"]
    elif "XAU" in s or "GOLD" in s:
        return PIP_SIZES["xauusd"]
    elif "XAG" in s or "SILVER" in s:
        return PIP_SIZES["xagusd"]
    elif any(idx in s for idx in ["US30", "US500", "NAS100", "SPX500", "DAX", "USTEC", "DJ30"]):
        return PIP_SIZES["indices"]
    elif any(crypto in s for crypto in ["BTC", "ETH", "LTC"]):
        return PIP_SIZES["crypto"]
    else:
        return PIP_SIZES["forex_standard"]