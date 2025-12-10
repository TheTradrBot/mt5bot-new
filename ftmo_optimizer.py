"""
Self-Optimizing FTMO Backtest System
=====================================

Runs backtests for November 2024 and optimizes parameters until BOTH
Phase 1 AND Phase 2 of the FTMO challenge are passed.

FTMO Challenge Rules:
- Phase 1: 10% profit on $10K, max 5% daily loss, max 10% drawdown, min 4 trading days
- Phase 2: 5% profit on Phase 1 ending balance, same percentage limits

Author: Blueprint Trader AI
"""

import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random
import copy

from backtest import run_backtest
from ftmo_config import FTMO_CONFIG, FTMO10KConfig


@dataclass
class PhaseResult:
    """Result of a single FTMO challenge phase."""
    phase: int
    passed: bool
    starting_balance: float
    ending_balance: float
    profit: float
    profit_pct: float
    profit_target: float
    max_daily_loss: float
    max_daily_loss_pct: float
    daily_loss_limit: float
    max_drawdown: float
    max_drawdown_pct: float
    drawdown_limit: float
    trading_days: int
    min_trading_days: int
    total_trades: int
    win_rate: float
    daily_pnl: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class ChallengeResult:
    """Complete FTMO challenge result (both phases)."""
    passed: bool
    phase1: PhaseResult
    phase2: Optional[PhaseResult]
    total_profit: float
    total_profit_pct: float
    final_balance: float
    iteration: int
    config_snapshot: Dict


@dataclass
class OptimizableParams:
    """Parameters that can be optimized."""
    risk_per_trade_pct: float = 0.5
    min_confluence_score: int = 5
    min_quality_factors: int = 2
    tp1_r_multiple: float = 1.5
    tp2_r_multiple: float = 3.0
    tp3_r_multiple: float = 5.0
    max_entry_distance_r: float = 2.0
    min_sl_atr_ratio: float = 1.0
    max_sl_atr_ratio: float = 3.0
    max_concurrent_trades: int = 3
    daily_loss_halt_pct: float = 4.2
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_ftmo_config(cls) -> 'OptimizableParams':
        return cls(
            risk_per_trade_pct=FTMO_CONFIG.risk_per_trade_pct,
            min_confluence_score=FTMO_CONFIG.min_confluence_score,
            min_quality_factors=FTMO_CONFIG.min_quality_factors,
            tp1_r_multiple=FTMO_CONFIG.tp1_r_multiple,
            tp2_r_multiple=FTMO_CONFIG.tp2_r_multiple,
            tp3_r_multiple=FTMO_CONFIG.tp3_r_multiple,
            max_entry_distance_r=FTMO_CONFIG.max_entry_distance_r,
            min_sl_atr_ratio=FTMO_CONFIG.min_sl_atr_ratio,
            max_sl_atr_ratio=FTMO_CONFIG.max_sl_atr_ratio,
            max_concurrent_trades=FTMO_CONFIG.max_concurrent_trades,
            daily_loss_halt_pct=FTMO_CONFIG.daily_loss_halt_pct,
        )


class FTMOBacktestRunner:
    """Runs FTMO-compliant backtests with proper phase tracking."""
    
    INITIAL_BALANCE = 10000.0
    PHASE1_PROFIT_TARGET_PCT = 10.0
    PHASE2_PROFIT_TARGET_PCT = 5.0
    MAX_DAILY_LOSS_PCT = 5.0
    MAX_DRAWDOWN_PCT = 10.0
    MIN_TRADING_DAYS = 4
    
    def __init__(self, symbols: List[str] = None):
        from symbol_mapping import ALL_TRADABLE_OANDA
        self.symbols = symbols or ALL_TRADABLE_OANDA[:10]
    
    def run_november_backtest(self, params: OptimizableParams) -> Dict:
        """
        Run backtest for November 2024 using the strategy.
        Returns trade data organized by date for FTMO validation.
        """
        period = "2024-11-01 - 2024-11-30"
        all_trades = []
        
        for symbol in self.symbols:
            try:
                result = run_backtest(symbol, period)
                if result.get('trades'):
                    for trade in result['trades']:
                        trade['symbol'] = symbol
                        trade['risk_pct'] = params.risk_per_trade_pct
                    all_trades.extend(result['trades'])
            except Exception as e:
                print(f"  Warning: {symbol} backtest failed: {e}")
                continue
        
        all_trades.sort(key=lambda t: t.get('entry_date', ''))
        
        return {
            'trades': all_trades,
            'period': period,
            'symbols': self.symbols,
            'params': params.to_dict(),
        }
    
    def simulate_challenge(self, backtest_data: Dict, params: OptimizableParams) -> ChallengeResult:
        """
        Simulate the complete FTMO challenge (Phase 1 + Phase 2).
        """
        trades = backtest_data.get('trades', [])
        
        phase1_result = self._run_phase(
            trades=trades,
            phase=1,
            starting_balance=self.INITIAL_BALANCE,
            profit_target_pct=self.PHASE1_PROFIT_TARGET_PCT,
            params=params,
        )
        
        phase2_result = None
        if phase1_result.passed:
            mid_point = len(trades) // 2
            remaining_trades = trades[mid_point:] if mid_point > 0 else trades
            
            phase2_result = self._run_phase(
                trades=remaining_trades,
                phase=2,
                starting_balance=phase1_result.ending_balance,
                profit_target_pct=self.PHASE2_PROFIT_TARGET_PCT,
                params=params,
            )
        
        complete_passed = phase1_result.passed and (phase2_result.passed if phase2_result else False)
        
        final_balance = phase2_result.ending_balance if phase2_result else phase1_result.ending_balance
        total_profit = final_balance - self.INITIAL_BALANCE
        total_profit_pct = (total_profit / self.INITIAL_BALANCE) * 100
        
        return ChallengeResult(
            passed=complete_passed,
            phase1=phase1_result,
            phase2=phase2_result,
            total_profit=total_profit,
            total_profit_pct=total_profit_pct,
            final_balance=final_balance,
            iteration=0,
            config_snapshot=params.to_dict(),
        )
    
    def _run_phase(
        self,
        trades: List[Dict],
        phase: int,
        starting_balance: float,
        profit_target_pct: float,
        params: OptimizableParams,
    ) -> PhaseResult:
        """Run a single challenge phase and track all FTMO metrics."""
        
        balance = starting_balance
        equity_curve = [balance]
        daily_pnl = {}
        peak_equity = balance
        max_drawdown = 0.0
        trading_days = set()
        
        profit_target = starting_balance * (profit_target_pct / 100)
        daily_loss_limit = starting_balance * (self.MAX_DAILY_LOSS_PCT / 100)
        drawdown_limit = starting_balance * (self.MAX_DRAWDOWN_PCT / 100)
        
        breached_daily_loss = False
        breached_drawdown = False
        target_reached = False
        
        for trade in trades:
            rr = trade.get('rr', 0)
            entry_date = trade.get('entry_date', '')[:10]
            
            if entry_date:
                trading_days.add(entry_date)
            
            pnl_pct = rr * params.risk_per_trade_pct
            pnl_usd = balance * (pnl_pct / 100)
            
            balance += pnl_usd
            equity_curve.append(balance)
            
            if entry_date not in daily_pnl:
                daily_pnl[entry_date] = 0.0
            daily_pnl[entry_date] += pnl_usd
            
            if balance > peak_equity:
                peak_equity = balance
            
            current_dd = peak_equity - balance
            if current_dd > max_drawdown:
                max_drawdown = current_dd
            
            if current_dd >= drawdown_limit:
                breached_drawdown = True
                break
            
            if daily_pnl[entry_date] <= -daily_loss_limit:
                breached_daily_loss = True
                break
            
            if balance >= starting_balance + profit_target:
                target_reached = True
                break
        
        profit = balance - starting_balance
        profit_pct = (profit / starting_balance) * 100
        
        daily_pnl_list = list(daily_pnl.values()) if daily_pnl else [0.0]
        max_daily_loss = abs(min(daily_pnl_list)) if daily_pnl_list else 0.0
        max_daily_loss_pct = (max_daily_loss / starting_balance) * 100
        max_drawdown_pct = (max_drawdown / starting_balance) * 100
        
        wins = sum(1 for t in trades if t.get('rr', 0) > 0)
        win_rate = (wins / len(trades) * 100) if trades else 0.0
        
        failure_reasons = []
        if profit < profit_target:
            failure_reasons.append(f"Profit target not met: ${profit:.2f} < ${profit_target:.2f}")
        if max_daily_loss > daily_loss_limit:
            failure_reasons.append(f"Daily loss exceeded: ${max_daily_loss:.2f} > ${daily_loss_limit:.2f}")
        if max_drawdown > drawdown_limit:
            failure_reasons.append(f"Drawdown exceeded: ${max_drawdown:.2f} > ${drawdown_limit:.2f}")
        if len(trading_days) < self.MIN_TRADING_DAYS:
            failure_reasons.append(f"Not enough trading days: {len(trading_days)} < {self.MIN_TRADING_DAYS}")
        
        passed = (
            profit >= profit_target and
            max_daily_loss <= daily_loss_limit and
            max_drawdown <= drawdown_limit and
            len(trading_days) >= self.MIN_TRADING_DAYS
        )
        
        return PhaseResult(
            phase=phase,
            passed=passed,
            starting_balance=starting_balance,
            ending_balance=balance,
            profit=profit,
            profit_pct=profit_pct,
            profit_target=profit_target,
            max_daily_loss=max_daily_loss,
            max_daily_loss_pct=max_daily_loss_pct,
            daily_loss_limit=daily_loss_limit,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            drawdown_limit=drawdown_limit,
            trading_days=len(trading_days),
            min_trading_days=self.MIN_TRADING_DAYS,
            total_trades=len(trades),
            win_rate=win_rate,
            daily_pnl=daily_pnl_list,
            equity_curve=equity_curve,
            failure_reasons=failure_reasons,
        )


class ParameterOptimizer:
    """Optimizes strategy parameters based on backtest failure analysis."""
    
    PARAM_RANGES = {
        'risk_per_trade_pct': (0.25, 1.0, 0.05),
        'min_confluence_score': (4, 6, 1),
        'min_quality_factors': (1, 3, 1),
        'tp1_r_multiple': (1.0, 2.0, 0.25),
        'tp2_r_multiple': (2.0, 4.0, 0.5),
        'tp3_r_multiple': (3.0, 6.0, 0.5),
        'max_entry_distance_r': (1.0, 3.0, 0.5),
        'max_concurrent_trades': (2, 5, 1),
        'daily_loss_halt_pct': (3.0, 4.5, 0.25),
    }
    
    def __init__(self):
        self.best_params = None
        self.best_score = -float('inf')
        self.history = []
    
    def analyze_failure(self, result: ChallengeResult) -> Dict[str, str]:
        """Analyze why the challenge failed and suggest improvements."""
        analysis = {
            'primary_issue': 'unknown',
            'suggestions': [],
        }
        
        phase = result.phase1 if not result.phase1.passed else result.phase2
        if not phase:
            return analysis
        
        if phase.max_drawdown > phase.drawdown_limit:
            analysis['primary_issue'] = 'drawdown'
            analysis['suggestions'] = [
                'Reduce risk_per_trade_pct',
                'Increase min_confluence_score',
                'Reduce max_concurrent_trades',
            ]
        elif phase.max_daily_loss > phase.daily_loss_limit:
            analysis['primary_issue'] = 'daily_loss'
            analysis['suggestions'] = [
                'Reduce risk_per_trade_pct',
                'Lower daily_loss_halt_pct',
                'Reduce max_concurrent_trades',
            ]
        elif phase.profit < phase.profit_target:
            analysis['primary_issue'] = 'profit'
            if phase.win_rate < 50:
                analysis['suggestions'] = [
                    'Increase min_confluence_score',
                    'Increase min_quality_factors',
                ]
            else:
                analysis['suggestions'] = [
                    'Increase tp2_r_multiple or tp3_r_multiple',
                    'Slightly increase risk_per_trade_pct (if drawdown allows)',
                ]
        elif phase.trading_days < phase.min_trading_days:
            analysis['primary_issue'] = 'trading_days'
            analysis['suggestions'] = [
                'Decrease min_confluence_score',
                'Increase max_entry_distance_r',
            ]
        
        return analysis
    
    def optimize(
        self,
        current_params: OptimizableParams,
        result: ChallengeResult,
        iteration: int,
    ) -> OptimizableParams:
        """Generate optimized parameters based on failure analysis."""
        
        analysis = self.analyze_failure(result)
        new_params = copy.deepcopy(current_params)
        
        if analysis['primary_issue'] == 'drawdown':
            new_params.risk_per_trade_pct = max(0.25, current_params.risk_per_trade_pct - 0.1)
            new_params.max_concurrent_trades = max(2, current_params.max_concurrent_trades - 1)
            new_params.min_confluence_score = min(6, current_params.min_confluence_score + 1)
            
        elif analysis['primary_issue'] == 'daily_loss':
            new_params.risk_per_trade_pct = max(0.25, current_params.risk_per_trade_pct - 0.1)
            new_params.daily_loss_halt_pct = max(3.0, current_params.daily_loss_halt_pct - 0.3)
            new_params.max_concurrent_trades = max(2, current_params.max_concurrent_trades - 1)
            
        elif analysis['primary_issue'] == 'profit':
            phase = result.phase1 if not result.phase1.passed else result.phase2
            if phase and phase.win_rate < 50:
                new_params.min_confluence_score = min(6, current_params.min_confluence_score + 1)
            else:
                new_params.tp2_r_multiple = min(5.0, current_params.tp2_r_multiple + 0.5)
                new_params.tp3_r_multiple = min(7.0, current_params.tp3_r_multiple + 0.5)
                if result.phase1.max_drawdown_pct < 5.0:
                    new_params.risk_per_trade_pct = min(1.0, current_params.risk_per_trade_pct + 0.1)
                    
        elif analysis['primary_issue'] == 'trading_days':
            new_params.min_confluence_score = max(4, current_params.min_confluence_score - 1)
            new_params.max_entry_distance_r = min(3.0, current_params.max_entry_distance_r + 0.5)
        
        else:
            if iteration % 3 == 0:
                new_params.risk_per_trade_pct = random.uniform(0.3, 0.8)
            elif iteration % 3 == 1:
                new_params.min_confluence_score = random.randint(4, 6)
            else:
                new_params.tp2_r_multiple = random.uniform(2.5, 4.0)
        
        self.history.append({
            'iteration': iteration,
            'params': new_params.to_dict(),
            'analysis': analysis,
        })
        
        return new_params
    
    def calculate_score(self, result: ChallengeResult) -> float:
        """Calculate optimization score for a result."""
        score = 0.0
        
        if result.passed:
            score += 1000
        
        if result.phase1.passed:
            score += 500
        
        score += result.phase1.profit_pct * 10
        
        score -= result.phase1.max_drawdown_pct * 5
        score -= result.phase1.max_daily_loss_pct * 5
        
        score += result.phase1.win_rate * 2
        
        if result.phase2:
            if result.phase2.passed:
                score += 500
            score += result.phase2.profit_pct * 10
            score -= result.phase2.max_drawdown_pct * 5
        
        return score


class ReportGenerator:
    """Generates comprehensive reports for backtest results."""
    
    def __init__(self, output_dir: str = "november_ftmo_backtest"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def print_phase_result(self, phase: PhaseResult):
        """Print formatted phase results."""
        status = "PASSED" if phase.passed else "FAILED"
        status_icon = "✅" if phase.passed else "❌"
        
        print(f"\n{'='*70}")
        print(f"📊 PHASE {phase.phase} RESULTS (Starting Balance: ${phase.starting_balance:,.2f})")
        print(f"{'='*70}")
        print(f"   Ending Balance: ${phase.ending_balance:,.2f}")
        print(f"   Profit: ${phase.profit:,.2f} ({phase.profit_pct:.2f}%) - Target: ${phase.profit_target:,.2f} {'✅' if phase.profit >= phase.profit_target else '❌'}")
        print(f"   Max Daily Loss: ${phase.max_daily_loss:,.2f} ({phase.max_daily_loss_pct:.2f}%) - Limit: ${phase.daily_loss_limit:,.2f} (5%) {'✅' if phase.max_daily_loss <= phase.daily_loss_limit else '❌'}")
        print(f"   Max Drawdown: ${phase.max_drawdown:,.2f} ({phase.max_drawdown_pct:.2f}%) - Limit: ${phase.drawdown_limit:,.2f} (10%) {'✅' if phase.max_drawdown <= phase.drawdown_limit else '❌'}")
        print(f"   Trading Days: {phase.trading_days} - Minimum: {phase.min_trading_days} {'✅' if phase.trading_days >= phase.min_trading_days else '❌'}")
        print(f"   Win Rate: {phase.win_rate:.1f}%")
        print(f"   Total Trades: {phase.total_trades}")
        print(f"   STATUS: {status_icon} {status}")
        print(f"{'='*70}")
        
        if phase.failure_reasons:
            print("   Failure Reasons:")
            for reason in phase.failure_reasons:
                print(f"   ❌ {reason}")
    
    def print_challenge_result(self, result: ChallengeResult, iteration: int):
        """Print complete challenge result."""
        print(f"\n{'='*70}")
        print(f"🔄 ITERATION {iteration}: FTMO Challenge Results")
        print(f"   Starting Balance: ${FTMOBacktestRunner.INITIAL_BALANCE:,.2f}")
        print(f"{'='*70}")
        
        self.print_phase_result(result.phase1)
        
        if result.phase2:
            self.print_phase_result(result.phase2)
        
        if result.passed:
            print("\n" + "="*70)
            print("🎉🎉🎉 SUCCESS! Bot passed COMPLETE FTMO CHALLENGE! 🎉🎉🎉")
            print("="*70)
            print("✅ Phase 1: PASSED")
            print("✅ Phase 2: PASSED")
            print(f"💰 Total Profit: ${result.total_profit:,.2f} ({result.total_profit_pct:.2f}%)")
            print(f"💵 Final Balance: ${result.final_balance:,.2f}")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("❌ FTMO COMPLETE CHALLENGE NOT PASSED")
            if not result.phase1.passed:
                print("❌ Phase 1: FAILED")
            else:
                print("✅ Phase 1: PASSED")
                print("❌ Phase 2: FAILED")
            print("🔧 Optimizing parameters...")
            print("="*70)
    
    def save_iteration(self, result: ChallengeResult, iteration: int, params: OptimizableParams):
        """Save iteration results to files."""
        iter_dir = self.output_dir / f"iteration_{iteration}"
        iter_dir.mkdir(exist_ok=True)
        
        results_data = {
            'iteration': iteration,
            'passed': result.passed,
            'phase1': {
                'passed': result.phase1.passed,
                'profit': result.phase1.profit,
                'profit_pct': result.phase1.profit_pct,
                'max_daily_loss': result.phase1.max_daily_loss,
                'max_drawdown': result.phase1.max_drawdown,
                'trading_days': result.phase1.trading_days,
                'win_rate': result.phase1.win_rate,
                'total_trades': result.phase1.total_trades,
            },
            'params': params.to_dict(),
        }
        
        if result.phase2:
            results_data['phase2'] = {
                'passed': result.phase2.passed,
                'profit': result.phase2.profit,
                'profit_pct': result.phase2.profit_pct,
                'max_daily_loss': result.phase2.max_daily_loss,
                'max_drawdown': result.phase2.max_drawdown,
                'trading_days': result.phase2.trading_days,
                'win_rate': result.phase2.win_rate,
                'total_trades': result.phase2.total_trades,
            }
        
        with open(iter_dir / "backtest_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        with open(iter_dir / "ftmo_metrics.txt", 'w') as f:
            f.write(f"FTMO Challenge Backtest - Iteration {iteration}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Overall: {'PASSED' if result.passed else 'FAILED'}\n\n")
            f.write("Phase 1:\n")
            f.write(f"  Profit: ${result.phase1.profit:,.2f} ({result.phase1.profit_pct:.2f}%)\n")
            f.write(f"  Max Daily Loss: ${result.phase1.max_daily_loss:,.2f}\n")
            f.write(f"  Max Drawdown: ${result.phase1.max_drawdown:,.2f}\n")
            f.write(f"  Trading Days: {result.phase1.trading_days}\n")
            f.write(f"  Status: {'PASSED' if result.phase1.passed else 'FAILED'}\n\n")
            
            if result.phase2:
                f.write("Phase 2:\n")
                f.write(f"  Profit: ${result.phase2.profit:,.2f} ({result.phase2.profit_pct:.2f}%)\n")
                f.write(f"  Max Daily Loss: ${result.phase2.max_daily_loss:,.2f}\n")
                f.write(f"  Max Drawdown: ${result.phase2.max_drawdown:,.2f}\n")
                f.write(f"  Trading Days: {result.phase2.trading_days}\n")
                f.write(f"  Status: {'PASSED' if result.phase2.passed else 'FAILED'}\n")
    
    def save_winning_config(self, result: ChallengeResult, params: OptimizableParams, iteration: int):
        """Save winning configuration."""
        winner_dir = self.output_dir / "winning_configuration"
        winner_dir.mkdir(exist_ok=True)
        
        config_data = {
            'optimized_params': params.to_dict(),
            'iterations_to_pass': iteration,
            'final_balance': result.final_balance,
            'total_profit': result.total_profit,
            'total_profit_pct': result.total_profit_pct,
            'phase1_profit': result.phase1.profit,
            'phase2_profit': result.phase2.profit if result.phase2 else 0,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(winner_dir / "optimized_config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\n✅ Winning configuration saved to {winner_dir}/optimized_config.json")


def run_optimization_loop(max_iterations: int = 50, symbols: List[str] = None):
    """
    Main optimization loop.
    Runs backtests and optimizes until both FTMO phases pass.
    """
    print("\n" + "="*70)
    print("🚀 FTMO SELF-OPTIMIZING BACKTEST SYSTEM")
    print("="*70)
    print(f"Target: Pass BOTH Phase 1 (10% profit) AND Phase 2 (5% profit)")
    print(f"Starting Balance: $10,000")
    print(f"Max Iterations: {max_iterations}")
    print("="*70)
    
    runner = FTMOBacktestRunner(symbols=symbols)
    optimizer = ParameterOptimizer()
    reporter = ReportGenerator()
    
    current_params = OptimizableParams.from_ftmo_config()
    best_result = None
    best_score = -float('inf')
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"🔄 ITERATION {iteration}/{max_iterations}")
        print(f"{'='*70}")
        print(f"Current Parameters:")
        print(f"  - Risk per trade: {current_params.risk_per_trade_pct:.2f}%")
        print(f"  - Min confluence: {current_params.min_confluence_score}/7")
        print(f"  - TP multiples: {current_params.tp1_r_multiple}R / {current_params.tp2_r_multiple}R / {current_params.tp3_r_multiple}R")
        print(f"  - Max concurrent trades: {current_params.max_concurrent_trades}")
        
        print("\n📊 Running November 2024 backtest...")
        backtest_data = runner.run_november_backtest(current_params)
        
        print(f"   Found {len(backtest_data['trades'])} trades")
        
        print("🔍 Simulating FTMO Challenge...")
        result = runner.simulate_challenge(backtest_data, current_params)
        result.iteration = iteration
        
        reporter.print_challenge_result(result, iteration)
        reporter.save_iteration(result, iteration, current_params)
        
        score = optimizer.calculate_score(result)
        if score > best_score:
            best_score = score
            best_result = result
            print(f"   📈 New best score: {score:.2f}")
        
        if result.passed:
            print("\n🎉 OPTIMIZATION COMPLETE - BOTH PHASES PASSED!")
            reporter.save_winning_config(result, current_params, iteration)
            return result, current_params
        
        current_params = optimizer.optimize(current_params, result, iteration)
    
    print("\n⚠️ Maximum iterations reached without passing both phases.")
    print(f"Best score achieved: {best_score:.2f}")
    
    if best_result:
        print("\nBest result summary:")
        print(f"  Phase 1: {'PASSED' if best_result.phase1.passed else 'FAILED'}")
        if best_result.phase2:
            print(f"  Phase 2: {'PASSED' if best_result.phase2.passed else 'FAILED'}")
    
    return best_result, current_params


if __name__ == "__main__":
    test_symbols = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "EUR_GBP"]
    
    result, params = run_optimization_loop(max_iterations=20, symbols=test_symbols)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    if result and result.passed:
        print("✅ Successfully found configuration that passes both FTMO phases!")
    else:
        print("⚠️ Could not find configuration to pass both phases in allocated iterations")
    print("="*70)
