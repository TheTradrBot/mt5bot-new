"""
Output Manager for FTMO Optimization
=====================================
Centralized module for all optimization output files.

Output Files:
-------------
ftmo_analysis_output/{MODE}/
├── optimization.log           # Real-time trial logging (current run only)
├── best_trades_training.csv   # All trades from best trial - training period
├── best_trades_validation.csv # All trades from best trial - validation period  
├── best_trades_final.csv      # All trades from best trial - full period
├── monthly_stats.csv          # Monthly breakdown: trades, win-rate, profit
├── symbol_performance.csv     # Per-symbol: trades, win-rate, profit
├── optimization_report.csv    # Final report after optimization run
└── history/                   # Archived runs (each run in separate directory)
    ├── run_001/               # First optimization run
    │   ├── optimization.log
    │   ├── best_trades_training.csv
    │   ├── best_trades_validation.csv
    │   ├── best_trades_final.csv
    │   ├── monthly_stats.csv
    │   ├── symbol_performance.csv
    │   └── best_params.json   # Parameters used for this run (critical for replication!)
    ├── run_002/               # Second optimization run
    │   └── ...
    └── run_003/
        └── ...

Archiving Strategy:
------------------
- Each new optimization run creates a new run_XXX directory in history/
- Run numbering is sequential (run_001, run_002, run_003, ...)
- All files from previous run (logs, CSVs, params) archived together
- best_params.json saved per run for easy parameter recovery
- Current files always contain data from the active/latest run

Usage:
    from output_manager import OutputManager
    
    om = OutputManager()
    om.log_trial(trial_num=1, score=45.2, total_r=32.5, ...)
    om.save_best_trial_trades(training_trades, validation_trades, final_trades)
    om.save_best_params(params_dict)  # Save params for this run
    om.generate_final_report()
"""

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


# Base output directory - subdirectories created per optimization mode
BASE_OUTPUT_DIR = Path("ftmo_analysis_output")
BASE_OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class TrialResult:
    """Single trial result for logging."""
    trial_number: int
    timestamp: str
    score: float
    total_r: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    profit_usd: float
    max_drawdown_pct: float
    is_best: bool = False
    # Validation metrics (only for best trial)
    val_total_r: Optional[float] = None
    val_sharpe: Optional[float] = None
    val_win_rate: Optional[float] = None
    val_profit_usd: Optional[float] = None
    # Final metrics (only for best trial)
    final_total_r: Optional[float] = None
    final_sharpe: Optional[float] = None
    final_win_rate: Optional[float] = None
    final_profit_usd: Optional[float] = None


class OutputManager:
    """
    Centralized output manager for optimization results.
    
    Features:
    - Real-time trial logging to CSV (nohup compatible)
    - Best trial trade exports (training/validation/final)
    - Monthly statistics breakdown
    - Symbol performance analysis
    - Final optimization report
    - Separate directories for NSGA-II vs TPE runs
    """
    
    def __init__(self, output_dir: Path = None, optimization_mode: str = "NSGA"):
        """
        Initialize OutputManager.
        
        Args:
            output_dir: Custom output directory (optional)
            optimization_mode: "NSGA" or "TPE" - creates subdirectory in ftmo_analysis_output/
        """
        if output_dir is None:
            # Create mode-specific subdirectory: ftmo_analysis_output/NSGA/ or ftmo_analysis_output/TPE/
            self.output_dir = BASE_OUTPUT_DIR / optimization_mode
        else:
            self.output_dir = output_dir
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.optimization_mode = optimization_mode
        
        # File paths
        self.log_file = self.output_dir / "optimization.log"
        self.best_training_file = self.output_dir / "best_trades_training.csv"
        self.best_validation_file = self.output_dir / "best_trades_validation.csv"
        self.best_final_file = self.output_dir / "best_trades_final.csv"
        self.monthly_stats_file = self.output_dir / "monthly_stats.csv"
        self.symbol_perf_file = self.output_dir / "symbol_performance.csv"
        self.report_file = self.output_dir / "optimization_report.csv"
        
        # Track best trial
        self.best_score = float('-inf')
        self.best_trial_number = None
        self.trials_logged = 0
        self.best_params_dict = None  # Store best params for archiving
        
        # Initialize fresh log file (archiving happens at END of run, not start)
        self._init_log_file()
    
    def sync_best_from_study(self, study_best_value: float, study_best_trial_number: int = None):
        """
        Sync the best_score with Optuna study's best value.
        
        Call this after loading/resuming an Optuna study to prevent
        false 'NEW BEST' messages for trials that aren't actually better.
        
        Args:
            study_best_value: The current best value from Optuna study
            study_best_trial_number: Optional trial number of the best trial
        """
        if study_best_value is not None and study_best_value > self.best_score:
            self.best_score = study_best_value
            self.best_trial_number = study_best_trial_number
            # Write note to log file
            with open(self.log_file, 'a') as f:
                f.write(f"[Synced] Previous best from database: Score={study_best_value:.2f} "
                       f"(Trial #{study_best_trial_number})\n\n")
    
    def _get_next_run_number(self, history_dir: Path) -> int:
        """Get the next sequential run number by checking existing run directories."""
        if not history_dir.exists():
            return 1
        
        # Find all run_XXX directories
        run_dirs = [d for d in history_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        if not run_dirs:
            return 1
        
        # Extract numbers and find max
        run_numbers = []
        for run_dir in run_dirs:
            try:
                num = int(run_dir.name.split('_')[1])
                run_numbers.append(num)
            except (IndexError, ValueError):
                continue
        
        return max(run_numbers) + 1 if run_numbers else 1
    
    def _init_log_file(self):
        """
        Initialize a fresh log file for this run.
        Called at startup - does NOT archive (archiving happens at END of run).
        """
        # Create new log file with header
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"FTMO OPTIMIZATION LOG - {self.optimization_mode}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def archive_current_run(self):
        """
        Archive current run results to history/run_XXX/ directory.
        Call this at the END of optimization to save a copy of results.
        Files are COPIED (not moved) so TPE/ always has latest results.
        """
        import shutil
        
        history_dir = self.output_dir / "history"
        
        # Files to archive
        files_to_archive = [
            self.log_file,
            self.best_training_file,
            self.best_validation_file,
            self.best_final_file,
            self.monthly_stats_file,
            self.symbol_perf_file,
            self.output_dir / "best_params.json",
        ]
        
        existing_files = [f for f in files_to_archive if f.exists()]
        
        if not existing_files:
            print("⚠️  No files to archive")
            return None
        
        # Create history directory if needed
        history_dir.mkdir(exist_ok=True)
        
        # Get next run number
        run_number = self._get_next_run_number(history_dir)
        run_dir = history_dir / f"run_{run_number:03d}"
        run_dir.mkdir(exist_ok=True)
        
        # COPY files to run directory (not move!)
        for file_path in existing_files:
            target_path = run_dir / file_path.name
            shutil.copy2(file_path, target_path)
        
        print(f"📦 Archived run to: {run_dir.relative_to(self.output_dir.parent)}")
        return run_dir
    
    def log_trial(
        self,
        trial_number: int,
        score: float,
        total_r: float,
        sharpe_ratio: float,
        win_rate: float,
        profit_factor: float,
        total_trades: int,
        profit_usd: float,
        max_drawdown_pct: float = 0.0,
        val_metrics: Optional[Dict] = None,
        final_metrics: Optional[Dict] = None,
        ftmo_dd_pct: Optional[float] = None,
        ftmo_challenge_passed: Optional[bool] = None,
    ) -> bool:
        """
        Log a trial result to log file. Returns True if this is a new best.
        
        Args:
            trial_number: Optuna trial number
            score: Composite optimization score
            total_r: Total R (risk units profit)
            sharpe_ratio: Sharpe ratio
            win_rate: Win rate percentage
            profit_factor: Profit factor
            total_trades: Number of trades
            profit_usd: Profit in USD (based on $200K account)
            max_drawdown_pct: Maximum drawdown percentage
            val_metrics: Validation period metrics (dict with total_r, sharpe, win_rate, profit_usd)
            final_metrics: Final period metrics (dict with total_r, sharpe, win_rate, profit_usd)
        
        Returns:
            True if this trial is new best, False otherwise
        """
        is_best = score > self.best_score
        if is_best:
            self.best_score = score
            self.best_trial_number = trial_number
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write to log file in human-readable format
        with open(self.log_file, 'a') as f:
            if is_best:
                f.write("-" * 80 + "\n")
                f.write(f"🏆 NEW BEST - Trial #{trial_number} [{timestamp}]\n")
                f.write("-" * 80 + "\n")
            else:
                f.write(f"Trial #{trial_number} [{timestamp}]\n")
            
            f.write(f"  Score: {score:.2f} | R: {total_r:+.1f} | Sharpe: {sharpe_ratio:.3f}\n")
            f.write(f"  Win Rate: {win_rate:.1f}% | PF: {profit_factor:.2f} | Trades: {total_trades}\n")
            f.write(f"  Profit: ${profit_usd:,.2f} | Max DD: {max_drawdown_pct:.2f}%\n")

            if ftmo_dd_pct is not None:
                status = "PASS" if ftmo_challenge_passed else "FAIL"
                f.write(f"  FTMO DD: {ftmo_dd_pct:.1f}% | Challenge: {status}\n")
            
            if val_metrics:
                f.write(f"  [Validation] R: {val_metrics['total_r']:+.1f} | ")
                f.write(f"WR: {val_metrics['win_rate']:.1f}% | ${val_metrics['profit_usd']:,.2f}\n")
            
            if final_metrics:
                f.write(f"  [Final 2023-2025] R: {final_metrics['total_r']:+.1f} | ")
                f.write(f"WR: {final_metrics['win_rate']:.1f}% | ${final_metrics['profit_usd']:,.2f}\n")
            
            f.write("\n")
        
        self.trials_logged += 1
        
        # Print to stdout for nohup logging
        if is_best:
            print(f"🏆 NEW BEST Trial #{trial_number}: Score={score:.2f}, R={total_r:+.1f}, "
                  f"Sharpe={sharpe_ratio:.2f}, WR={win_rate:.1f}%, ${profit_usd:,.0f}")
            if val_metrics:
                print(f"   └─ Validation: R={val_metrics['total_r']:+.1f}, "
                      f"WR={val_metrics['win_rate']:.1f}%, ${val_metrics['profit_usd']:,.0f}")
            if final_metrics:
                print(f"   └─ Final: R={final_metrics['total_r']:+.1f}, "
                      f"WR={final_metrics['win_rate']:.1f}%, ${final_metrics['profit_usd']:,.0f}")
        else:
            print(f"   Trial #{trial_number}: Score={score:.2f}, R={total_r:+.1f}, "
                  f"WR={win_rate:.1f}%, ${profit_usd:,.0f}")
        
        return is_best
    
    def save_best_trial_trades(
        self,
        training_trades: List[Any],
        validation_trades: List[Any],
        final_trades: List[Any],
        risk_pct: float = 0.5,
        account_size: float = 200000.0,
    ):
        """
        Save all trades from best trial to separate CSV files.
        These will be archived to history/run_XXX/ when next optimization starts.
        
        Args:
            training_trades: List of Trade objects from training period
            validation_trades: List of Trade objects from validation period
            final_trades: List of Trade objects from full period
            risk_pct: Risk per trade percentage
            account_size: Account size in USD
        """
        # Save new CSVs (will be archived at next run)
        self._export_trades_csv(training_trades, self.best_training_file, risk_pct, account_size)
        self._export_trades_csv(validation_trades, self.best_validation_file, risk_pct, account_size)
        self._export_trades_csv(final_trades, self.best_final_file, risk_pct, account_size)
        
        print(f"\n📁 Best trial trades exported:")
        print(f"   Training:   {self.best_training_file.name} ({len(training_trades)} trades)")
        print(f"   Validation: {self.best_validation_file.name} ({len(validation_trades)} trades)")
        print(f"   Final:      {self.best_final_file.name} ({len(final_trades)} trades)")
    
    def save_best_params(self, params_dict: Dict[str, Any]):
        """
        Save best parameters to JSON file in output directory.
        This file will be archived with other run files when next optimization starts.
        
        Args:
            params_dict: Dictionary of optimized parameters
        """
        self.best_params_dict = params_dict
        params_file = self.output_dir / "best_params.json"
        
        # Add metadata
        params_with_meta = {
            "optimization_mode": self.optimization_mode,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "best_score": self.best_score,
            "parameters": params_dict
        }
        
        with open(params_file, 'w') as f:
            json.dump(params_with_meta, f, indent=2)
        
        print(f"💾 Best parameters saved to: {params_file.name}")
    
    def _export_trades_csv(
        self,
        trades: List[Any],
        filepath: Path,
        risk_pct: float,
        account_size: float,
    ):
        """Export trades to CSV file."""
        if not trades:
            # Create empty file with extended headers
            headers = [
                'trade_id', 'symbol', 'direction', 
                'entry_date', 'exit_date', 'trade_duration_hours',
                'entry_price', 'exit_price', 'stop_loss',
                'lot_size', 'risk_usd', 'risk_pct',
                'tp1_price', 'tp2_price', 'tp3_price',
                'tp1_hit', 'tp2_hit', 'tp3_hit',
                'tp1_close_pct', 'tp2_close_pct', 'tp3_close_pct',
                'exit_reason', 'partial_exits_json',
                'result_r', 'profit_usd', 'win',
                'max_favorable_excursion', 'max_adverse_excursion',
                'confluence_score', 'quality_factors'
            ]
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            return
        
        risk_per_trade = account_size * (risk_pct / 100)
        
        rows = []
        for i, t in enumerate(trades, 1):
            # Support both 'rr' (from strategy_core.Trade) and 'result_r' (legacy)
            result_r = getattr(t, 'rr', getattr(t, 'result_r', 0.0))
            profit_calc = result_r * risk_per_trade
            profit_usd = getattr(t, 'profit_usd', profit_calc)
            partial_exits = getattr(t, 'partial_exits', [])
            try:
                partial_json = json.dumps(partial_exits)
            except Exception:
                partial_json = '[]'
            rows.append({
                'trade_id': getattr(t, 'trade_id', i),
                'symbol': getattr(t, 'symbol', ''),
                'direction': getattr(t, 'direction', ''),
                'entry_date': str(getattr(t, 'entry_date', '')),
                'exit_date': str(getattr(t, 'exit_date', '')),
                'trade_duration_hours': getattr(t, 'trade_duration_hours', 0.0),
                'entry_price': getattr(t, 'entry_price', 0.0),
                'exit_price': getattr(t, 'exit_price', 0.0),
                'stop_loss': getattr(t, 'stop_loss', 0.0),
                'lot_size': getattr(t, 'lot_size', 0.0),
                'risk_usd': getattr(t, 'risk_usd', risk_per_trade),
                'risk_pct': getattr(t, 'risk_pct', risk_pct),
                'tp1_price': getattr(t, 'tp1', None),
                'tp2_price': getattr(t, 'tp2', None),
                'tp3_price': getattr(t, 'tp3', None),
                'tp1_hit': getattr(t, 'tp1_hit', False),
                'tp2_hit': getattr(t, 'tp2_hit', False),
                'tp3_hit': getattr(t, 'tp3_hit', False),
                'tp1_close_pct': getattr(t, 'tp1_close_pct', 0.0),
                'tp2_close_pct': getattr(t, 'tp2_close_pct', 0.0),
                'tp3_close_pct': getattr(t, 'tp3_close_pct', 0.0),
                'exit_reason': getattr(t, 'exit_reason', ''),
                'partial_exits_json': partial_json,
                'result_r': round(float(result_r), 4),
                'profit_usd': round(float(profit_usd), 2),
                'win': 1 if result_r > 0 else 0,
                'max_favorable_excursion': getattr(t, 'max_favorable_excursion', getattr(t, 'mfe_rr', 0.0)),
                'max_adverse_excursion': getattr(t, 'max_adverse_excursion', getattr(t, 'mae_rr', 0.0)),
                'confluence_score': getattr(t, 'confluence_score', 0),
                'quality_factors': getattr(t, 'quality_factors', 0),
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    def generate_monthly_stats(
        self,
        trades: List[Any],
        period_name: str,
        risk_pct: float = 0.5,
        account_size: float = 200000.0,
    ):
        """
        Generate monthly statistics breakdown.
        
        Creates/updates monthly_stats.csv with:
        - Period (training/validation/final)
        - Month
        - Number of trades
        - Win rate
        - Total profit USD
        """
        if not trades:
            return
        
        risk_per_trade = account_size * (risk_pct / 100)
        
        # Group trades by month
        monthly_data = {}
        for t in trades:
            if hasattr(t, 'entry_date') and t.entry_date:
                month_key = t.entry_date.strftime("%Y-%m") if hasattr(t.entry_date, 'strftime') else str(t.entry_date)[:7]
                if month_key not in monthly_data:
                    monthly_data[month_key] = {'trades': 0, 'wins': 0, 'profit_r': 0}
                
                # Support both 'rr' and 'result_r'
                result_r = getattr(t, 'rr', getattr(t, 'result_r', 0))
                
                monthly_data[month_key]['trades'] += 1
                if result_r > 0:
                    monthly_data[month_key]['wins'] += 1
                monthly_data[month_key]['profit_r'] += result_r
        
        # Build rows
        rows = []
        total_trades = 0
        total_wins = 0
        total_profit = 0
        
        for month, data in sorted(monthly_data.items()):
            win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
            profit_usd = data['profit_r'] * risk_per_trade
            
            rows.append({
                'period': period_name,
                'month': month,
                'trades': data['trades'],
                'wins': data['wins'],
                'win_rate': round(win_rate, 1),
                'profit_r': round(data['profit_r'], 2),
                'profit_usd': round(profit_usd, 2),
            })
            
            total_trades += data['trades']
            total_wins += data['wins']
            total_profit += profit_usd
        
        # Add total row
        total_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        rows.append({
            'period': period_name,
            'month': 'TOTAL',
            'trades': total_trades,
            'wins': total_wins,
            'win_rate': round(total_wr, 1),
            'profit_r': round(sum(m['profit_r'] for m in monthly_data.values()), 2),
            'profit_usd': round(total_profit, 2),
        })
        
        # Append to or create file
        df = pd.DataFrame(rows)
        if self.monthly_stats_file.exists():
            existing = pd.read_csv(self.monthly_stats_file)
            # Remove old data for this period
            existing = existing[existing['period'] != period_name]
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_csv(self.monthly_stats_file, index=False)
        print(f"📊 Monthly stats updated for {period_name}: {total_trades} trades, ${total_profit:,.0f}")
    
    def generate_symbol_performance(
        self,
        trades: List[Any],
        risk_pct: float = 0.5,
        account_size: float = 200000.0,
    ):
        """
        Generate symbol performance breakdown.
        
        Creates symbol_performance.csv with:
        - Symbol
        - Total trades
        - Win rate
        - Total profit USD
        - Average R per trade
        """
        if not trades:
            return
        
        risk_per_trade = account_size * (risk_pct / 100)
        
        # Group by symbol
        symbol_data = {}
        for t in trades:
            symbol = getattr(t, 'symbol', 'UNKNOWN')
            if symbol not in symbol_data:
                symbol_data[symbol] = {'trades': 0, 'wins': 0, 'profit_r': 0}
            
            # Support both 'rr' and 'result_r'
            result_r = getattr(t, 'rr', getattr(t, 'result_r', 0))
            
            symbol_data[symbol]['trades'] += 1
            if result_r > 0:
                symbol_data[symbol]['wins'] += 1
            symbol_data[symbol]['profit_r'] += result_r
        
        rows = []
        for symbol, data in sorted(symbol_data.items(), key=lambda x: x[1]['profit_r'], reverse=True):
            win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
            profit_usd = data['profit_r'] * risk_per_trade
            avg_r = data['profit_r'] / data['trades'] if data['trades'] > 0 else 0
            
            rows.append({
                'symbol': symbol,
                'trades': data['trades'],
                'wins': data['wins'],
                'win_rate': round(win_rate, 1),
                'profit_r': round(data['profit_r'], 2),
                'profit_usd': round(profit_usd, 2),
                'avg_r_per_trade': round(avg_r, 3),
            })
        
        # Add total row
        total_trades = sum(d['trades'] for d in symbol_data.values())
        total_wins = sum(d['wins'] for d in symbol_data.values())
        total_r = sum(d['profit_r'] for d in symbol_data.values())
        total_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        rows.append({
            'symbol': 'TOTAL',
            'trades': total_trades,
            'wins': total_wins,
            'win_rate': round(total_wr, 1),
            'profit_r': round(total_r, 2),
            'profit_usd': round(total_r * risk_per_trade, 2),
            'avg_r_per_trade': round(total_r / total_trades if total_trades > 0 else 0, 3),
        })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.symbol_perf_file, index=False)
        print(f"📊 Symbol performance saved: {len(symbol_data)} symbols analyzed")
    
    def generate_final_report(
        self,
        best_params: Dict[str, Any],
        training_metrics: Dict[str, float],
        validation_metrics: Dict[str, float],
        final_metrics: Dict[str, float],
        total_trials: int,
        optimization_time_hours: float = 0,
    ):
        """
        Generate final optimization report as CSV.
        
        Creates optimization_report.csv with complete summary.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build report rows (key-value format for AI parsing)
        rows = [
            {'section': 'META', 'metric': 'report_generated', 'value': timestamp},
            {'section': 'META', 'metric': 'total_trials', 'value': total_trials},
            {'section': 'META', 'metric': 'best_trial', 'value': self.best_trial_number},
            {'section': 'META', 'metric': 'best_score', 'value': round(self.best_score, 2)},
            {'section': 'META', 'metric': 'optimization_hours', 'value': round(optimization_time_hours, 2)},
        ]
        
        # Training metrics
        for key, val in training_metrics.items():
            rows.append({'section': 'TRAINING', 'metric': key, 'value': round(val, 4) if isinstance(val, float) else val})
        
        # Validation metrics
        for key, val in validation_metrics.items():
            rows.append({'section': 'VALIDATION', 'metric': key, 'value': round(val, 4) if isinstance(val, float) else val})
        
        # Final metrics
        for key, val in final_metrics.items():
            rows.append({'section': 'FINAL', 'metric': key, 'value': round(val, 4) if isinstance(val, float) else val})
        
        # Best parameters
        for key, val in best_params.items():
            rows.append({'section': 'PARAMS', 'metric': key, 'value': round(val, 4) if isinstance(val, float) else val})
        
        df = pd.DataFrame(rows)
        df.to_csv(self.report_file, index=False)
        
        print(f"\n{'='*60}")
        print("📋 OPTIMIZATION REPORT GENERATED")
        print(f"{'='*60}")
        print(f"File: {self.report_file}")
        print(f"Trials: {total_trials} | Best: #{self.best_trial_number} (Score: {self.best_score:.2f})")
        print(f"\nTraining:   R={training_metrics.get('total_r', 0):+.1f}, "
              f"WR={training_metrics.get('win_rate', 0):.1f}%, "
              f"${training_metrics.get('profit_usd', 0):,.0f}")
        print(f"Validation: R={validation_metrics.get('total_r', 0):+.1f}, "
              f"WR={validation_metrics.get('win_rate', 0):.1f}%, "
              f"${validation_metrics.get('profit_usd', 0):,.0f}")
        print(f"Final:      R={final_metrics.get('total_r', 0):+.1f}, "
              f"WR={final_metrics.get('win_rate', 0):.1f}%, "
              f"${final_metrics.get('profit_usd', 0):,.0f}")
        print(f"{'='*60}\n")
    
    def clear_output(self):
        """Clear all output files for fresh optimization run."""
        files_to_clear = [
            self.log_file,
            self.best_training_file,
            self.best_validation_file,
            self.best_final_file,
            self.monthly_stats_file,
            self.symbol_perf_file,
            self.report_file,
        ]
        
        for f in files_to_clear:
            if f.exists():
                f.unlink()
        
        self.best_score = float('-inf')
        self.best_trial_number = None
        self.trials_logged = 0
        self._init_log_file()
        
        print("🗑️  Output files cleared for fresh optimization run")


# Global instance for easy access
_output_manager: Optional[OutputManager] = None


def get_output_manager(optimization_mode: str = "NSGA") -> OutputManager:
    """
    Get the global OutputManager instance.
    
    Args:
        optimization_mode: "NSGA" or "TPE" - determines subdirectory
    """
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager(optimization_mode=optimization_mode)
    return _output_manager


def set_output_manager(optimization_mode: str = "NSGA"):
    """
    Explicitly set/reset the global OutputManager with specific mode.
    
    Args:
        optimization_mode: "NSGA" or "TPE"
    """
    global _output_manager
    _output_manager = OutputManager(optimization_mode=optimization_mode)


if __name__ == "__main__":
    # Demo/test
    om = OutputManager()
    print("Output Manager initialized")
    print(f"Output directory: {om.output_dir}")
    print(f"Log file: {om.log_file}")
