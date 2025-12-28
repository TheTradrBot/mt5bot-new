"""
Unified Optimization Configuration
===================================
Single source of truth for all optimization mode settings.

This consolidates the previously scattered configuration for:
- Single-objective vs Multi-objective optimization
- ADX Regime filtering (Trend/Range/Transition modes)
- Database paths and study names
- Feature toggles that affect optimization behavior

Usage:
    from params.optimization_config import get_optimization_config, OptimizationConfig
    
    config = get_optimization_config()
    if config.use_multi_objective:
        # Use NSGA-II sampler
    if config.use_adx_regime_filter:
        # Apply ADX-based regime detection
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import os

# Default paths
CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / "optimization_config.json"
DEFAULT_DB_PATH = "sqlite:///ftmo_optimization.db"
DEFAULT_STUDY_NAME = "ftmo_unified_study"


@dataclass
class OptimizationConfig:
    """
    Unified configuration for all optimization modes and features.
    
    Attributes:
        # Optimization Mode
        use_multi_objective: If True, use NSGA-II multi-objective optimization
                            If False, use TPE single-objective optimization
        
        # Strategy Mode Toggles
        use_adx_regime_filter: Enable ADX-based Trend/Range/Transition mode detection
        use_adx_slope_rising: Allow early trend entries on rising ADX
        
        # Database Configuration
        db_path: SQLite database path for Optuna studies
        study_name: Name of the Optuna study
        
        # Optimization Objectives (for multi-objective mode)
        objectives: List of objectives to optimize ['total_r', 'sharpe_ratio', 'win_rate']
        
        # Trial Configuration
        n_trials: Number of optimization trials to run
        n_startup_trials: Random trials before using sampler intelligence
        
        # Feature Toggles (passed to strategy)
        use_partial_exits: Enable partial profit taking at 1R
        use_atr_trailing: Enable ATR-based trailing stops
        use_volatility_sizing: Enable volatility-based position sizing boost
    """
    
    # =========================================================================
    # OPTIMIZATION MODE - Choose one optimization approach
    # =========================================================================
    use_multi_objective: bool = False  # False = TPE single-objective, True = NSGA-II multi-objective
    
    # =========================================================================
    # STRATEGY MODE TOGGLES - Enable/disable strategy features
    # =========================================================================
    use_adx_regime_filter: bool = False  # ADX-based Trend/Range/Transition detection
    use_adx_slope_rising: bool = False   # Early trend entry on rising ADX
    use_partial_exits: bool = True       # Partial profit at 1R
    use_atr_trailing: bool = True        # ATR trailing on runner position
    use_volatility_sizing: bool = False  # Volatility-based sizing boost
    
    # =========================================================================
    # DATABASE CONFIGURATION - Single unified database
    # =========================================================================
    db_path: str = DEFAULT_DB_PATH
    study_name: str = DEFAULT_STUDY_NAME
    
    # =========================================================================
    # MULTI-OBJECTIVE SETTINGS (only used when use_multi_objective=True)
    # =========================================================================
    objectives: List[str] = field(default_factory=lambda: ['total_r', 'sharpe_ratio', 'win_rate'])
    
    # =========================================================================
    # TRIAL CONFIGURATION
    # =========================================================================
    n_trials: int = 500          # Total trials to run
    n_startup_trials: int = 20   # Random trials before sampler kicks in
    timeout_hours: float = 48.0  # Max optimization time in hours
    
    # =========================================================================
    # DATE RANGES - Training and validation periods
    # =========================================================================
    train_start: str = "2024-01-01"
    train_end: str = "2024-09-30"
    validation_start: str = "2024-10-01"
    validation_end: str = "2024-12-31"
    
    # =========================================================================
    # DERIVED PROPERTIES
    # =========================================================================
    @property
    def optimization_mode_name(self) -> str:
        """Human-readable optimization mode name."""
        return "Multi-Objective NSGA-II" if self.use_multi_objective else "Single-Objective TPE"
    
    @property
    def regime_mode_name(self) -> str:
        """Human-readable regime mode name."""
        if self.use_adx_regime_filter:
            return "ADX Regime-Adaptive (Trend/Range/Transition)"
        return "Standard (No regime detection)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        d = asdict(self)
        # Add derived properties
        d['optimization_mode_name'] = self.optimization_mode_name
        d['regime_mode_name'] = self.regime_mode_name
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizationConfig":
        """Create config from dictionary."""
        # Filter out derived properties and unknown keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save config to JSON file."""
        path = path or CONFIG_FILE
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Optimization config saved to: {path}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "OptimizationConfig":
        """Load config from JSON file."""
        path = path or CONFIG_FILE
        if not path.exists():
            print(f"⚠️  Config file not found: {path}")
            print("   Using default configuration...")
            return cls()
        
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def print_summary(self) -> None:
        """Print a summary of the current configuration."""
        print("\n" + "="*60)
        print("📊 OPTIMIZATION CONFIGURATION")
        print("="*60)
        print(f"\n🎯 Optimization Mode: {self.optimization_mode_name}")
        print(f"📈 Regime Mode: {self.regime_mode_name}")
        print(f"\n📁 Database: {self.db_path}")
        print(f"📋 Study Name: {self.study_name}")
        print(f"\n⚙️  Feature Toggles:")
        print(f"   • ADX Regime Filter: {'✅ Enabled' if self.use_adx_regime_filter else '❌ Disabled'}")
        print(f"   • ADX Slope Rising:  {'✅ Enabled' if self.use_adx_slope_rising else '❌ Disabled'}")
        print(f"   • Partial Exits:     {'✅ Enabled' if self.use_partial_exits else '❌ Disabled'}")
        print(f"   • ATR Trailing:      {'✅ Enabled' if self.use_atr_trailing else '❌ Disabled'}")
        print(f"   • Volatility Sizing: {'✅ Enabled' if self.use_volatility_sizing else '❌ Disabled'}")
        print(f"\n🔬 Trial Configuration:")
        print(f"   • Total Trials: {self.n_trials}")
        print(f"   • Startup Trials: {self.n_startup_trials}")
        print(f"   • Timeout: {self.timeout_hours} hours")
        if self.use_multi_objective:
            print(f"   • Objectives: {', '.join(self.objectives)}")
        print(f"\n📅 Date Ranges:")
        print(f"   • Training:   {self.train_start} to {self.train_end}")
        print(f"   • Validation: {self.validation_start} to {self.validation_end}")
        print("="*60 + "\n")


def get_optimization_config(reload: bool = False) -> OptimizationConfig:
    """
    Get the current optimization configuration.
    
    Args:
        reload: If True, force reload from file even if cached
        
    Returns:
        OptimizationConfig instance
    """
    global _cached_config
    
    if reload or '_cached_config' not in globals() or _cached_config is None:
        _cached_config = OptimizationConfig.load()
    
    return _cached_config


def create_default_config() -> None:
    """Create a default configuration file if it doesn't exist."""
    if CONFIG_FILE.exists():
        print(f"⚠️  Config file already exists: {CONFIG_FILE}")
        return
    
    config = OptimizationConfig()
    config.save()
    print(f"✅ Created default config at: {CONFIG_FILE}")


# Module-level cache
_cached_config: Optional[OptimizationConfig] = None


if __name__ == "__main__":
    # If run directly, create default config and print summary
    print("🔧 Optimization Configuration Manager")
    print("-" * 40)
    
    if not CONFIG_FILE.exists():
        print("\nCreating default configuration...")
        create_default_config()
    
    config = get_optimization_config()
    config.print_summary()
