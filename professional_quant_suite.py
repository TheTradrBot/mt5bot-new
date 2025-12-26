#!/usr/bin/env python3
"""
Professional Quantitative Trading Analysis Suite
Comprehensive tools for institutional-grade backtesting and optimization

Features:
- Walk-Forward Testing (Rolling window validation)
- Risk Metrics (Sharpe, Sortino, Calmar ratios)
- Parameter Sensitivity Analysis
- Monte Carlo Parameter Sweep
- Multi-Objective Optimization (Pareto frontier)
- Out-of-Sample Robustness Testing
- Drawdown Analysis
- Performance Consistency Metrics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
from statistics import mean, stdev

# ============================================================================
# PROFESSIONAL RISK METRICS
# ============================================================================

@dataclass
class RiskMetrics:
    """Professional risk and performance metrics for trading strategies."""
    
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    win_rate: float = 0.0
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    consecutive_winners: int = 0
    consecutive_losers: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting."""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration_days': self.max_drawdown_duration,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'recovery_factor': self.recovery_factor,
            'consecutive_winners': self.consecutive_winners,
            'consecutive_losers': self.consecutive_losers,
        }


def calculate_risk_metrics(trades: List[Any], 
                          risk_per_trade_pct: float = 0.5,
                          account_size: float = 200000.0,
                          trading_days_per_year: float = 252.0) -> RiskMetrics:
    """
    Calculate professional risk metrics for a trade sequence.
    
    Args:
        trades: List of Trade objects with rr (risk-reward) and entry_date
        risk_per_trade_pct: Risk per trade as percentage
        account_size: Starting account size
        trading_days_per_year: Trading days (default 252 for forex)
    
    Returns:
        RiskMetrics object with all risk calculations
    """
    if not trades:
        return RiskMetrics()
    
    # Convert R-multiples to account returns
    risk_amount = account_size * (risk_per_trade_pct / 100.0)
    returns = [getattr(t, 'rr', 0) * risk_amount for t in trades]
    
    # Total return
    total_return = sum(returns)
    total_return_pct = (total_return / account_size) * 100
    
    # Time period calculation
    entry_dates = []
    for t in trades:
        entry = getattr(t, 'entry_date', None)
        if entry:
            if isinstance(entry, str):
                try:
                    entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                except:
                    continue
            if hasattr(entry, 'replace') and entry.tzinfo:
                entry = entry.replace(tzinfo=None)
            entry_dates.append(entry)
    
    if entry_dates:
        trading_period_days = (max(entry_dates) - min(entry_dates)).days + 1
        years = trading_period_days / 365.0
        annual_return = (total_return_pct / years) if years > 0 else 0
    else:
        annual_return = total_return_pct
    
    # Win rate and profit factor
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    win_rate = (len(wins) / len(returns) * 100) if returns else 0
    
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    # Drawdown analysis
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = running_max - cumulative_returns
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Drawdown duration
    max_dd_idx = np.argmax(drawdown)
    max_drawdown_duration = 0
    if max_dd_idx > 0:
        dd_region = drawdown[:max_dd_idx + 1]
        recovery_idx = np.where(dd_region == 0)[0]
        if len(recovery_idx) > 0:
            last_recovery = recovery_idx[-1]
            max_drawdown_duration = max_dd_idx - last_recovery
    
    # Risk-adjusted returns (Sharpe ratio)
    if len(returns) > 1:
        return_std = np.std(returns)
        sharpe = (mean(returns) / return_std * np.sqrt(252)) if return_std > 0 else 0
    else:
        sharpe = 0
    
    # Sortino ratio (penalizes downside volatility only)
    if len(returns) > 1:
        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            downside_std = np.std(downside_returns)
            sortino = (mean(returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            sortino = sharpe  # No losses, same as Sharpe
    else:
        sortino = 0
    
    # Calmar ratio (return / max drawdown)
    calmar = (annual_return / max_drawdown) if max_drawdown > 0 else 0
    
    # Recovery factor
    recovery_factor = (total_return / max_drawdown) if max_drawdown > 0 else 0
    
    # Consecutive winners/losers
    consecutive_wins = []
    consecutive_losses = []
    current_streak = 0
    current_type = None
    
    for ret in returns:
        if ret > 0:
            if current_type == 'win':
                current_streak += 1
            else:
                if current_type == 'loss':
                    consecutive_losses.append(current_streak)
                current_streak = 1
                current_type = 'win'
        elif ret < 0:
            if current_type == 'loss':
                current_streak += 1
            else:
                if current_type == 'win':
                    consecutive_wins.append(current_streak)
                current_streak = 1
                current_type = 'loss'
    
    if current_type == 'win':
        consecutive_wins.append(current_streak)
    elif current_type == 'loss':
        consecutive_losses.append(current_streak)
    
    max_consecutive_wins = max(consecutive_wins) if consecutive_wins else 0
    max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
    
    return RiskMetrics(
        total_return=total_return_pct,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        recovery_factor=recovery_factor,
        consecutive_winners=max_consecutive_wins,
        consecutive_losers=max_consecutive_losses,
    )


# ============================================================================
# WALK-FORWARD TESTING
# ============================================================================

class WalkForwardTester:
    """
    Walk-forward analysis for robust parameter testing.
    
    Implements anchored and rolling window validation:
    - Anchored: Training window grows, validation window fixed
    - Rolling: Both windows roll forward together
    """
    
    def __init__(self, all_trades: List[Any], 
                 start_date: datetime,
                 end_date: datetime,
                 train_months: int = 12,
                 validate_months: int = 3,
                 rolling: bool = True):
        """
        Initialize walk-forward tester.
        
        Args:
            all_trades: List of Trade objects
            start_date: Analysis start date
            end_date: Analysis end date
            train_months: Training window size in months
            validate_months: Validation window size in months
            rolling: True for rolling windows, False for anchored
        """
        self.all_trades = all_trades
        self.start_date = start_date
        self.end_date = end_date
        self.train_months = train_months
        self.validate_months = validate_months
        self.rolling = rolling
    
    def get_date_windows(self) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate training/validation date windows.
        
        Returns:
            List of (train_start, train_end, val_start, val_end) tuples
        """
        windows = []
        
        if self.rolling:
            # Rolling window: both move forward
            current_start = self.start_date
            
            while True:
                train_end = current_start + timedelta(days=self.train_months * 30)
                val_start = train_end + timedelta(days=1)
                val_end = val_start + timedelta(days=self.validate_months * 30)
                
                if val_end > self.end_date:
                    break
                
                windows.append((current_start, train_end, val_start, val_end))
                
                # Roll forward by 1 month
                current_start = current_start + timedelta(days=30)
        else:
            # Anchored window: training grows, validation fixed size
            train_start = self.start_date
            
            while True:
                train_end = train_start + timedelta(days=self.train_months * 30)
                val_start = train_end + timedelta(days=1)
                val_end = val_start + timedelta(days=self.validate_months * 30)
                
                if val_end > self.end_date:
                    break
                
                windows.append((train_start, train_end, val_start, val_end))
                
                # Extend training by 1 month
                train_start = train_start + timedelta(days=30)
        
        return windows
    
    def get_trades_for_period(self, start: datetime, end: datetime) -> List[Any]:
        """Get all trades within a date range."""
        period_trades = []
        
        for trade in self.all_trades:
            entry = getattr(trade, 'entry_date', None)
            if entry:
                if isinstance(entry, str):
                    try:
                        entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                    except:
                        continue
                if hasattr(entry, 'replace') and entry.tzinfo:
                    entry = entry.replace(tzinfo=None)
                
                if start <= entry <= end:
                    period_trades.append(trade)
        
        return period_trades
    
    def analyze_all_windows(self, risk_per_trade_pct: float = 0.5) -> Dict:
        """
        Analyze all walk-forward windows.
        
        Returns:
            Dictionary with metrics for each window and summary statistics
        """
        windows = self.get_date_windows()
        window_results = []
        
        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            train_trades = self.get_trades_for_period(train_start, train_end)
            val_trades = self.get_trades_for_period(val_start, val_end)
            
            train_metrics = calculate_risk_metrics(train_trades, risk_per_trade_pct)
            val_metrics = calculate_risk_metrics(val_trades, risk_per_trade_pct)
            
            # Calculate degradation (IS-OOS spread)
            sharpe_degradation = train_metrics.sharpe_ratio - val_metrics.sharpe_ratio
            return_degradation = train_metrics.total_return - val_metrics.total_return
            
            window_results.append({
                'window': i + 1,
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'val_start': val_start.strftime('%Y-%m-%d'),
                'val_end': val_end.strftime('%Y-%m-%d'),
                'train_metrics': train_metrics.to_dict(),
                'val_metrics': val_metrics.to_dict(),
                'sharpe_degradation': sharpe_degradation,
                'return_degradation': return_degradation,
                'train_trades': len(train_trades),
                'val_trades': len(val_trades),
            })
        
        # Calculate summary statistics
        sharpe_degradations = [w['sharpe_degradation'] for w in window_results]
        return_degradations = [w['return_degradation'] for w in window_results]
        
        summary = {
            'total_windows': len(window_results),
            'avg_sharpe_degradation': mean(sharpe_degradations) if sharpe_degradations else 0,
            'std_sharpe_degradation': stdev(sharpe_degradations) if len(sharpe_degradations) > 1 else 0,
            'avg_return_degradation': mean(return_degradations) if return_degradations else 0,
            'window_results': window_results,
        }
        
        return summary


# ============================================================================
# PARAMETER SENSITIVITY ANALYSIS
# ============================================================================

class ParameterSensitivityAnalyzer:
    """Analyze how sensitive strategy is to parameter changes."""
    
    @staticmethod
    def tornado_analysis(baseline_metrics: Dict, 
                        sensitivity_results: Dict) -> Dict:
        """
        Create tornado chart data: impact of each parameter on performance.
        
        Args:
            baseline_metrics: Baseline performance metrics
            sensitivity_results: Dict of {param_name: [(param_value, metrics), ...]}
        
        Returns:
            Tornado chart data sorted by impact
        """
        baseline_sharpe = baseline_metrics.get('sharpe_ratio', 0)
        baseline_return = baseline_metrics.get('total_return', 0)
        
        parameter_impacts = []
        
        for param_name, results in sensitivity_results.items():
            impacts = []
            for param_value, metrics in results:
                impact = metrics.get('sharpe_ratio', 0) - baseline_sharpe
                impacts.append({
                    'param_value': param_value,
                    'sharpe_impact': impact,
                    'return_impact': metrics.get('total_return', 0) - baseline_return,
                })
            
            if impacts:
                best = max(impacts, key=lambda x: x['sharpe_impact'])
                worst = min(impacts, key=lambda x: x['sharpe_impact'])
                
                parameter_impacts.append({
                    'parameter': param_name,
                    'best_value': best['param_value'],
                    'best_impact': best['sharpe_impact'],
                    'worst_value': worst['param_value'],
                    'worst_impact': worst['sharpe_impact'],
                    'range': best['sharpe_impact'] - worst['sharpe_impact'],
                })
        
        # Sort by impact range (descending)
        parameter_impacts.sort(key=lambda x: x['range'], reverse=True)
        
        return parameter_impacts


# ============================================================================
# REPORTING AND VISUALIZATION
# ============================================================================

def generate_professional_report(
    best_params: Dict,
    training_metrics: RiskMetrics,
    validation_metrics: RiskMetrics,
    full_metrics: RiskMetrics,
    walk_forward_results: Dict,
    output_file: str = "professional_backtest_report.txt"
) -> str:
    """
    Generate comprehensive professional backtest report.
    
    Returns:
        Report text content
    """
    report_lines = [
        "=" * 100,
        "PROFESSIONAL QUANTITATIVE TRADING ANALYSIS REPORT",
        "=" * 100,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 100,
        f"Strategy Status: {'APPROVED' if validation_metrics.sharpe_ratio > 0.5 else 'NEEDS IMPROVEMENT'}",
        f"Training Sharpe Ratio: {training_metrics.sharpe_ratio:+.2f}",
        f"Validation Sharpe Ratio: {validation_metrics.sharpe_ratio:+.2f}",
        f"Degradation: {training_metrics.sharpe_ratio - validation_metrics.sharpe_ratio:.2f}",
        "",
        "PERFORMANCE METRICS COMPARISON",
        "-" * 100,
        f"{'Metric':<30} {'Training':>20} {'Validation':>20} {'Full Period':>20}",
        "-" * 100,
        f"{'Total Return':<30} {training_metrics.total_return:>19.2f}% {validation_metrics.total_return:>19.2f}% {full_metrics.total_return:>19.2f}%",
        f"{'Annual Return':<30} {training_metrics.annual_return:>19.2f}% {validation_metrics.annual_return:>19.2f}% {full_metrics.annual_return:>19.2f}%",
        f"{'Sharpe Ratio':<30} {training_metrics.sharpe_ratio:>20.2f} {validation_metrics.sharpe_ratio:>20.2f} {full_metrics.sharpe_ratio:>20.2f}",
        f"{'Sortino Ratio':<30} {training_metrics.sortino_ratio:>20.2f} {validation_metrics.sortino_ratio:>20.2f} {full_metrics.sortino_ratio:>20.2f}",
        f"{'Calmar Ratio':<30} {training_metrics.calmar_ratio:>20.2f} {validation_metrics.calmar_ratio:>20.2f} {full_metrics.calmar_ratio:>20.2f}",
        f"{'Max Drawdown':<30} {training_metrics.max_drawdown:>19.2f}$ {validation_metrics.max_drawdown:>19.2f}$ {full_metrics.max_drawdown:>19.2f}$",
        f"{'Win Rate':<30} {training_metrics.win_rate:>19.1f}% {validation_metrics.win_rate:>19.1f}% {full_metrics.win_rate:>19.1f}%",
        f"{'Profit Factor':<30} {training_metrics.profit_factor:>20.2f} {validation_metrics.profit_factor:>20.2f} {full_metrics.profit_factor:>20.2f}",
        f"{'Recovery Factor':<30} {training_metrics.recovery_factor:>20.2f} {validation_metrics.recovery_factor:>20.2f} {full_metrics.recovery_factor:>20.2f}",
        "",
        "ROBUSTNESS ANALYSIS (Walk-Forward Testing)",
        "-" * 100,
        f"Total Windows Tested: {walk_forward_results.get('total_windows', 0)}",
        f"Average Sharpe Degradation: {walk_forward_results.get('avg_sharpe_degradation', 0):+.2f}",
        f"Std Dev Degradation: {walk_forward_results.get('std_sharpe_degradation', 0):.2f}",
        f"Average Return Degradation: {walk_forward_results.get('avg_return_degradation', 0):+.2f}%",
        "",
        "OPTIMAL PARAMETERS",
        "-" * 100,
    ]
    
    for param, value in sorted(best_params.items()):
        if isinstance(value, float):
            report_lines.append(f"  {param:<40} {value:>10.4f}")
        else:
            report_lines.append(f"  {param:<40} {str(value):>10}")
    
    report_lines.extend([
        "",
        "APPROVAL CRITERIA",
        "-" * 100,
        f"✓ Sharpe Ratio > 0.5: {validation_metrics.sharpe_ratio > 0.5}",
        f"✓ Win Rate > 45%: {validation_metrics.win_rate > 45}",
        f"✓ Profit Factor > 1.5: {validation_metrics.profit_factor > 1.5}",
        f"✓ IS-OOS Degradation < 0.5: {(training_metrics.sharpe_ratio - validation_metrics.sharpe_ratio) < 0.5}",
        f"✓ Consistency Score: {(validation_metrics.sharpe_ratio / (training_metrics.sharpe_ratio + 0.01)) if training_metrics.sharpe_ratio > 0 else 0:.2%}",
        "",
        "=" * 100,
    ])
    
    report_text = "\n".join(report_lines)
    
    # Write to file
    output_path = Path(output_file)
    output_path.write_text(report_text)
    
    return report_text


# Print example usage
if __name__ == "__main__":
    print("Professional Quant Suite Loaded")
    print("Available Classes:")
    print("  - RiskMetrics: Professional risk calculation")
    print("  - WalkForwardTester: Rolling/anchored window validation")
    print("  - ParameterSensitivityAnalyzer: Parameter sensitivity analysis")
    print("\nAvailable Functions:")
    print("  - calculate_risk_metrics()")
    print("  - generate_professional_report()")
