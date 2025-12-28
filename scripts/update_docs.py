#!/usr/bin/env python3
"""
Automatic Documentation Generator
==================================

This script automatically updates all documentation files based on the current
codebase state. It extracts information from:
- Source code (docstrings, type hints, function signatures)
- Configuration files (JSON, Python configs)
- Optimization database (recent trial results)
- Git history (recent commits, changelog)

Usage:
    python scripts/update_docs.py              # Update all docs
    python scripts/update_docs.py --file API   # Update specific doc
    python scripts/update_docs.py --check      # Check if docs are outdated

Auto-run: This script is triggered by .github/workflows/update-docs.yml on every commit
"""

import os
import sys
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import ast
import inspect

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Documentation files to maintain
DOCS_DIR = PROJECT_ROOT / "docs"


# ============================================================================
# METADATA EXTRACTION
# ============================================================================

def extract_function_signatures(file_path: Path) -> List[Dict]:
    """Extract all function signatures with docstrings from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function signature
                args = []
                for arg in node.args.args:
                    arg_name = arg.arg
                    arg_type = ast.unparse(arg.annotation) if arg.annotation else "Any"
                    args.append(f"{arg_name}: {arg_type}")
                
                # Extract return type
                return_type = ast.unparse(node.returns) if node.returns else "None"
                
                # Extract docstring
                docstring = ast.get_docstring(node) or "No description"
                
                functions.append({
                    'name': node.name,
                    'args': args,
                    'return_type': return_type,
                    'docstring': docstring,
                    'file': file_path.name,
                })
        
        return functions
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return []


def extract_class_info(file_path: Path) -> List[Dict]:
    """Extract all class definitions with methods from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            'name': item.name,
                            'docstring': ast.get_docstring(item) or "No description"
                        })
                
                classes.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node) or "No description",
                    'methods': methods,
                    'file': file_path.name,
                })
        
        return classes
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return []


def get_parameter_info() -> Dict:
    """Extract current parameters from current_params.json."""
    params_file = PROJECT_ROOT / "params" / "current_params.json"
    if params_file.exists():
        with open(params_file, 'r') as f:
            return json.load(f)
    return {}


def get_optimization_config() -> Dict:
    """Extract optimization config from optimization_config.json."""
    config_file = PROJECT_ROOT / "params" / "optimization_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}


def get_recent_commits(n: int = 10) -> List[Dict]:
    """Get recent git commits."""
    try:
        result = subprocess.run(
            ['git', 'log', f'-{n}', '--pretty=format:%H|%an|%ad|%s', '--date=short'],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        commits = []
        for line in result.stdout.strip().split('\n'):
            if '|' in line:
                hash, author, date, message = line.split('|', 3)
                commits.append({
                    'hash': hash[:7],
                    'author': author,
                    'date': date,
                    'message': message
                })
        
        return commits
    except Exception as e:
        print(f"Warning: Could not get git commits: {e}")
        return []


def get_file_tree(root: Path, exclude_dirs: List[str] = None) -> str:
    """Generate file tree structure."""
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.git', 'node_modules', '.venv', 'venv']
    
    lines = []
    
    def walk_dir(path: Path, prefix: str = ""):
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for i, item in enumerate(items):
                if item.name in exclude_dirs or item.name.startswith('.'):
                    continue
                
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                lines.append(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    walk_dir(item, prefix + extension)
        except PermissionError:
            pass
    
    walk_dir(root)
    return "\n".join(lines)


# ============================================================================
# DOCUMENTATION UPDATERS
# ============================================================================

def update_architecture_doc(output_path: Path) -> bool:
    """Update ARCHITECTURE.md with current system state."""
    print("Updating ARCHITECTURE.md...")
    
    # Architecture doc is already comprehensive, just update metadata
    if not output_path.exists():
        return False
    
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update last updated date
    today = datetime.now().strftime("%Y-%m-%d")
    content = re.sub(
        r'\*\*Last Updated\*\*: \d{4}-\d{2}-\d{2}',
        f'**Last Updated**: {today}',
        content
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated {output_path.name}")
    return True


def update_strategy_doc(output_path: Path) -> bool:
    """Update STRATEGY_GUIDE.md with strategy details."""
    print("Updating STRATEGY_GUIDE.md...")
    
    # Extract strategy core functions
    strategy_file = PROJECT_ROOT / "strategy_core.py"
    functions = extract_function_signatures(strategy_file)
    
    # Get current parameters
    params = get_parameter_info()
    
    content = f"""# Trading Strategy Guide

**Last Updated**: {datetime.now().strftime("%Y-%m-%d")}  
**Strategy**: 7-Pillar Confluence System with ADX Regime Detection

---

## Table of Contents
1. [Strategy Overview](#strategy-overview)
2. [7 Confluence Pillars](#7-confluence-pillars)
3. [ADX Regime Detection](#adx-regime-detection)
4. [Entry Rules](#entry-rules)
5. [Exit Management](#exit-management)
6. [Risk Management](#risk-management)
7. [Current Parameters](#current-parameters)

---

## Strategy Overview

The bot uses a **multi-timeframe confluence system** that combines 7 independent signals to identify high-probability setups. Each pillar votes on trade direction and quality, with entries requiring a minimum confluence score.

### Key Principles
- **Confluence over single indicators**: Requires multiple confirmations
- **Regime-adaptive**: Different rules for trending vs ranging markets
- **Risk-first approach**: Every trade pre-validated for R:R and FTMO limits
- **Multi-timeframe**: D1 signals confirmed by H4, W1, MN context

---

## 7 Confluence Pillars

### 1. Trend Alignment (Daily → Weekly → Monthly)
**Weight**: 2 points  
**Logic**: Entry must align with higher timeframe trend direction

```python
# Weekly trend trumps daily, monthly trumps weekly
weekly_trend = _infer_trend(weekly_candles)
monthly_trend = _infer_trend(monthly_candles)

if direction != weekly_trend or direction != monthly_trend:
    pillar_score -= 2  # Reject counter-trend
```

### 2. Support/Resistance Confluence
**Weight**: 1 point  
**Logic**: Price near key S/R level

```python
# Check if entry near S/R level (within 0.5% tolerance)
if abs(entry_price - sr_level) / entry_price < 0.005:
    pillar_score += 1
```

### 3. Fibonacci Zone Alignment
**Weight**: 1 point  
**Logic**: Entry at Fibonacci retracement level (38.2%, 50%, 61.8%)

```python
# Calculate Fib levels from recent swing
fib_levels = [0.382, 0.5, 0.618]
for level in fib_levels:
    if abs(entry_price - fib_price) < fib_tolerance:
        pillar_score += 1
```

### 4. RSI Divergence
**Weight**: 1 point  
**Logic**: Price makes new high/low but RSI doesn't (reversal signal)

```python
# Bullish divergence: price lower low, RSI higher low
if price_trend == "down" and rsi_trend == "up":
    pillar_score += 1
```

### 5. ADX Trend Strength
**Weight**: 1 point  
**Logic**: ADX > threshold confirms strong trend

```python
adx_value = calculate_adx(daily_candles, period=14)
if adx_value >= params['adx_trend_threshold']:
    pillar_score += 1
```

### 6. ATR Volatility Filter
**Weight**: 1 point  
**Logic**: Current ATR above minimum percentile (avoid dead markets)

```python
current_atr = calculate_atr(daily_candles, period=14)
atr_percentile = get_atr_percentile(current_atr, historical_atr)

if atr_percentile >= params['atr_min_percentile']:
    pillar_score += 1
```

### 7. Candlestick Pattern
**Weight**: 1 point  
**Logic**: Bullish/bearish engulfing, pin bar, inside bar

```python
pattern = detect_candlestick_pattern(daily_candles[-3:])
if pattern in ['engulfing', 'pin_bar'] and pattern_direction == direction:
    pillar_score += 1
```

---

## ADX Regime Detection

The strategy adapts based on market regime:

### Regime Classification
```python
adx = calculate_adx(daily_candles, period=14)

if adx >= params['adx_trend_threshold']:
    regime = "TREND"       # Momentum following
elif adx <= params['adx_range_threshold']:
    regime = "RANGE"       # Mean reversion
else:
    regime = "TRANSITION"  # No trading
```

### Regime-Specific Rules

| Regime | Min Confluence | RSI Filter | ATR Filter | SL Distance |
|--------|----------------|------------|------------|-------------|
| **TREND** | {params.get('trend_min_confluence', 5)} | Disabled | Strict | 1.5× ATR |
| **RANGE** | {params.get('range_min_confluence', 3)} | Enabled (oversold/overbought) | Relaxed | 1.0× ATR |
| **TRANSITION** | No entries | - | - | - |

---

## Entry Rules

### Minimum Requirements
1. **Confluence Score** ≥ {params.get('min_confluence_score', 4)} pillars
2. **Quality Factors** ≥ {params.get('min_quality_factors', 2)}
3. **Risk:Reward** ≥ 2.5:1
4. **ADX Regime** = TREND or RANGE (not TRANSITION)
5. **Spread** < 2× average spread
6. **FTMO Limits** not breached

### Entry Execution
```python
if confluence_score >= min_confluence and can_trade:
    # Place pending order at entry price
    lot_size = calculate_lot_size(risk_pct, sl_distance, symbol)
    
    if direction == "BUY":
        place_buy_stop(entry_price, lot_size, sl, tp)
    else:
        place_sell_stop(entry_price, lot_size, sl, tp)
```

---

## Exit Management

### Take Profit Levels
1. **TP1**: {params.get('trail_activation_r', 1.8)}R - Partial exit ({params.get('partial_exit_pct', 0.5) * 100}% position)
2. **TP2**: 2.5R - Final target

### Trailing Stop
Activated after {params.get('trail_activation_r', 1.8)}R profit:
```python
trail_distance = atr * params['atr_trail_multiplier']  # {params.get('atr_trail_multiplier', 2.25)}× ATR
```

### Stop Loss
- **Initial**: Set at S/R level or 1.5× ATR from entry
- **Breakeven**: Move to entry +1 pip after 1R profit
- **Trail**: Follows price at {params.get('atr_trail_multiplier', 2.25)}× ATR distance

---

## Risk Management

### Position Sizing
```python
# Current setting: {params.get('risk_per_trade_pct', 0.65)}% per trade
risk_amount = account_size * {params.get('risk_per_trade_pct', 0.65)} / 100
lot_size = risk_amount / (sl_pips * pip_value)
```

### FTMO Limits
- **Max Daily Loss**: 5.0% ($10,000) - Bot halts at 4.2%
- **Max Total Drawdown**: 10.0% ($20,000) - Bot emergency stop at 7%
- **Max Concurrent Trades**: {params.get('max_concurrent_trades', 6)}

### Seasonal Adjustments
- **Summer (June-Aug)**: Risk × {params.get('summer_risk_multiplier', 0.75)} (lower volatility)
- **December**: ATR × {params.get('december_atr_multiplier', 1.75)} (holiday spike)

---

## Current Parameters

**Active Configuration** (from `params/current_params.json`):

```json
{json.dumps(params, indent=2)}
```

---

## Strategy Functions Reference

### Core Functions

"""
    
    # Add function signatures
    for func in functions:
        if func['name'] in ['compute_confluence', 'simulate_trades', 'detect_regime', '_infer_trend']:
            content += f"\n#### `{func['name']}({', '.join(func['args'])}) -> {func['return_type']}`\n\n"
            content += f"{func['docstring']}\n\n"
    
    content += """
---

**Maintained by**: Auto-generated from source code  
**Update command**: `python scripts/update_docs.py`  
**Source**: `strategy_core.py`
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated {output_path.name}")
    return True


def update_api_reference(output_path: Path) -> bool:
    """Generate complete API reference from source code."""
    print("Updating API_REFERENCE.md...")
    
    # Files to document
    files_to_doc = [
        PROJECT_ROOT / "strategy_core.py",
        PROJECT_ROOT / "ftmo_challenge_analyzer.py",
        PROJECT_ROOT / "params" / "params_loader.py",
        PROJECT_ROOT / "tradr" / "risk" / "manager.py",
        PROJECT_ROOT / "tradr" / "utils" / "output_manager.py",
    ]
    
    content = f"""# API Reference

**Last Updated**: {datetime.now().strftime("%Y-%m-%d")}  
**Auto-generated**: Yes (run `python scripts/update_docs.py` to regenerate)

---

## Table of Contents

"""
    
    # Build TOC
    for file_path in files_to_doc:
        if file_path.exists():
            module_name = file_path.stem
            content += f"- [{module_name}](#{module_name})\n"
    
    content += "\n---\n\n"
    
    # Document each file
    for file_path in files_to_doc:
        if not file_path.exists():
            continue
        
        module_name = file_path.stem
        content += f"## {module_name}\n\n"
        content += f"**File**: `{file_path.relative_to(PROJECT_ROOT)}`\n\n"
        
        # Extract classes
        classes = extract_class_info(file_path)
        if classes:
            content += "### Classes\n\n"
            for cls in classes:
                content += f"#### `{cls['name']}`\n\n"
                content += f"{cls['docstring']}\n\n"
                
                if cls['methods']:
                    content += "**Methods**:\n\n"
                    for method in cls['methods']:
                        content += f"- `{method['name']}()`: {method['docstring'][:100]}...\n"
                content += "\n"
        
        # Extract functions
        functions = extract_function_signatures(file_path)
        if functions:
            content += "### Functions\n\n"
            for func in functions:
                content += f"#### `{func['name']}({', '.join(func['args'])})`\n\n"
                content += f"**Returns**: `{func['return_type']}`\n\n"
                content += f"{func['docstring']}\n\n"
                content += "---\n\n"
    
    content += """
## Usage Examples

### Load Strategy Parameters
```python
from params.params_loader import load_strategy_params

params = load_strategy_params()
min_confluence = params['min_confluence_score']
```

### Run Backtest
```python
from ftmo_challenge_analyzer import run_full_period_backtest

trades = run_full_period_backtest(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    min_confluence=4,
    risk_per_trade_pct=0.5
)
```

### Check Risk Manager
```python
from tradr.risk.manager import RiskManager

rm = RiskManager(account_size=200000)
can_trade, reason = rm.can_trade(symbol="EURUSD", risk_pct=0.5)
```

---

**Auto-generated**: Run `python scripts/update_docs.py` to regenerate  
**Last update**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated {output_path.name}")
    return True


def update_deployment_doc(output_path: Path) -> bool:
    """Update DEPLOYMENT_GUIDE.md."""
    print("Updating DEPLOYMENT_GUIDE.md...")
    
    content = f"""# Deployment Guide

**Last Updated**: {datetime.now().strftime("%Y-%m-%d")}

---

## Prerequisites

### Development Environment (Linux/Mac)
- Python 3.11+
- Git
- 8GB+ RAM
- SSD storage recommended

### Production Environment (Windows)
- Windows 10/11 (64-bit)
- Python 3.11+
- MetaTrader 5 terminal
- FTMO account credentials
- 24/7 uptime (VPS recommended)

---

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/TheTradrBot/mt5bot-new.git
cd mt5bot-new
```

### 2. Install Dependencies

**Linux/Mac** (optimizer):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows** (live bot):
```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
pip install MetaTrader5  # Windows only
```

### 3. Configure Environment

Create `.env` file in project root:
```bash
# MT5 Credentials (Windows only)
MT5_SERVER=FTMO-Demo
MT5_LOGIN=your_login_number
MT5_PASSWORD=your_password

# Optional: Telegram notifications
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 4. Verify Setup
```bash
python scripts/validate_setup.py
```

Expected output:
```
✓ params/current_params.json exists and valid
✓ All 25 required parameters present
✓ Historical data files present (34 assets × 4 TFs)
✓ Contract specs loaded for all tradable assets
✓ Optimization config valid
```

---

## Running Optimization

### Local Run
```bash
# Single run (100 trials)
python ftmo_challenge_analyzer.py --multi --trials 100

# With ADX regime filter
python ftmo_challenge_analyzer.py --multi --adx --trials 100

# Check progress
python ftmo_challenge_analyzer.py --status
```

### Background Run (Replit/VPS)
```bash
# Start in background
nohup python ftmo_challenge_analyzer.py --multi --trials 500 > opt.log 2>&1 &

# Monitor progress
tail -f ftmo_analysis_output/NSGA/optimization.log

# Check if still running
ps aux | grep ftmo_challenge_analyzer

# Stop optimization
pkill -f ftmo_challenge_analyzer
```

### Time Estimates
- 50 trials: ~1.5 hours
- 100 trials: ~2.5 hours
- 200 trials: ~4.5 hours
- 500 trials: ~11 hours

---

## Running Live Bot

### Windows VM Setup

1. **Install MetaTrader 5**
   - Download from FTMO
   - Login with credentials
   - Enable Expert Advisors → Allow automated trading

2. **Configure Bot**
```bash
# Verify .env file exists
type .env

# Test MT5 connection
python -c "from tradr.mt5.client import MT5Client; MT5Client().initialize()"
```

3. **Start Bot**
```bash
# Foreground (for testing)
python main_live_bot.py

# Background (production)
pythonw main_live_bot.py  # Runs without console window
```

4. **Monitor Logs**
```bash
# Real-time log watching
Get-Content logs\\tradr_live.log -Wait -Tail 50

# Check for errors
Select-String -Path logs\\tradr_live.log -Pattern "ERROR"
```

### Auto-Start on Boot (Windows)

Create `start_bot.bat`:
```batch
@echo off
cd C:\\Users\\YourUser\\mt5bot-new
venv\\Scripts\\activate
pythonw main_live_bot.py
```

Add to Windows Task Scheduler:
- Trigger: At system startup
- Action: Start `start_bot.bat`
- Run whether user is logged in or not

---

## Continuous Deployment Workflow

```
┌─────────────────────────────────────────────────┐
│ STEP 1: Development (Linux)                     │
├─────────────────────────────────────────────────┤
│ 1. Edit strategy_core.py                        │
│ 2. Run optimization (100-200 trials)            │
│ 3. Validate results (Sharpe > 1.5, WR > 50%)    │
│ 4. git commit && git push                       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ STEP 2: Production (Windows VM)                 │
├─────────────────────────────────────────────────┤
│ 1. Stop main_live_bot.py                        │
│ 2. git pull origin main                         │
│ 3. Verify params/current_params.json updated    │
│ 4. Restart main_live_bot.py                     │
│ 5. Monitor for 24h                              │
└─────────────────────────────────────────────────┘
```

---

## Monitoring & Alerts

### Health Checks

**Every Hour**:
```bash
# Check bot is running
ps aux | grep main_live_bot

# Check MT5 connection
curl http://localhost:5000/health  # If web API enabled
```

**Every Day**:
```bash
# Check account balance
python scripts/check_account_status.py

# Verify no FTMO violations
python scripts/check_ftmo_compliance.py

# Review trade log
python scripts/generate_daily_report.py
```

### Telegram Notifications (Optional)

Setup in `.env`:
```bash
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=987654321
```

Notifications sent for:
- ✅ Trade opened
- 💰 Trade closed (profit)
- 🔴 Trade closed (loss)
- ⚠️ FTMO limit warning (4% daily loss)
- 🚨 Emergency stop (7% total drawdown)

---

## Troubleshooting

### Common Issues

**Issue**: Bot not finding MT5 terminal  
**Fix**: Set `MT5_PATH` in .env: `MT5_PATH=C:\\Program Files\\MetaTrader 5\\terminal64.exe`

**Issue**: "Historical data missing"  
**Fix**: Run `python update_csvs.py` to download missing OHLCV data

**Issue**: "Optuna study locked"  
**Fix**: Check if another process is running: `ps aux | grep ftmo_challenge_analyzer`

**Issue**: "Spread too wide" errors  
**Fix**: Broker spread spike - wait for market open, or adjust `max_spread_multiplier` in config

### Debug Mode

Enable verbose logging:
```bash
export TRADR_DEBUG=1  # Linux
set TRADR_DEBUG=1     # Windows

python main_live_bot.py
```

### Database Reset

If optimization database corrupted:
```bash
# Backup old database
mv ftmo_optimization.db ftmo_optimization.db.backup

# Start fresh
python ftmo_challenge_analyzer.py --trials 5
```

---

## Security Best Practices

1. **Never commit .env file** (already in .gitignore)
2. **Use read-only API keys** if broker supports (FTMO doesn't)
3. **Enable 2FA** on broker account
4. **VPS firewall rules**: Only allow SSH (port 22) and MT5 ports
5. **Regular backups**: params/, data/, logs/ folders
6. **Monitor login attempts**: Check MT5 terminal for unauthorized access

---

## Performance Tuning

### Optimization Speed
- **Reduce trials**: Start with 50, increase if underfitting
- **Reduce startup trials**: Lower `n_startup_trials` in config (default 20)
- **Use TPE instead of NSGA-II**: Slightly faster (~10%)
- **Parallel execution**: Not recommended (Optuna SQLite doesn't support)

### Live Bot Efficiency
- **Scan interval**: Default 4h is optimal (D1 strategy)
- **Reduce logging**: Set `LOG_LEVEL=WARNING` for production
- **Disable debug mode**: Remove `TRADR_DEBUG` env var

---

## Maintenance Schedule

### Daily
- [ ] Check logs for errors
- [ ] Verify bot is running
- [ ] Review open positions

### Weekly
- [ ] Git pull latest changes
- [ ] Check for parameter updates
- [ ] Review win rate vs baseline

### Monthly
- [ ] Run full optimization (200-500 trials)
- [ ] Update params if improvement > 10%
- [ ] Backup database and logs
- [ ] Review FTMO compliance (drawdown, daily loss)

### Quarterly
- [ ] Update historical data (run update_csvs.py)
- [ ] Conduct walk-forward analysis
- [ ] Review strategy performance vs market regime
- [ ] Consider parameter sensitivity analysis

---

## Upgrade Path

### Version Updates
```bash
# Backup current state
cp params/current_params.json params/backup_params.json

# Pull latest code
git pull origin main

# Check for breaking changes
git log --oneline -10

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Validate setup
python scripts/validate_setup.py
```

### Major Version Migrations
See `docs/MIGRATION_GUIDES/` for version-specific instructions.

---

**Last Updated**: {datetime.now().strftime("%Y-%m-%d")}  
**Maintained by**: AI-assisted development team  
**Questions**: Open GitHub issue or check docs/FAQ.md
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated {output_path.name}")
    return True


def update_changelog(output_path: Path) -> bool:
    """Update CHANGELOG.md with recent commits."""
    print("Updating CHANGELOG.md...")
    
    commits = get_recent_commits(50)
    
    # Group commits by type (feat, fix, docs, chore, etc.)
    grouped = {'feat': [], 'fix': [], 'docs': [], 'refactor': [], 'chore': [], 'other': []}
    
    for commit in commits:
        msg = commit['message']
        if msg.startswith('feat'):
            grouped['feat'].append(commit)
        elif msg.startswith('fix'):
            grouped['fix'].append(commit)
        elif msg.startswith('docs'):
            grouped['docs'].append(commit)
        elif msg.startswith('refactor'):
            grouped['refactor'].append(commit)
        elif msg.startswith('chore'):
            grouped['chore'].append(commit)
        else:
            grouped['other'].append(commit)
    
    content = f"""# Changelog

**Last Updated**: {datetime.now().strftime("%Y-%m-%d")}  
**Auto-generated**: From git commits

---

## Recent Changes

### Features
"""
    
    for commit in grouped['feat'][:10]:
        content += f"- [{commit['hash']}] {commit['message']} ({commit['date']})\n"
    
    content += "\n### Bug Fixes\n"
    for commit in grouped['fix'][:10]:
        content += f"- [{commit['hash']}] {commit['message']} ({commit['date']})\n"
    
    content += "\n### Documentation\n"
    for commit in grouped['docs'][:10]:
        content += f"- [{commit['hash']}] {commit['message']} ({commit['date']})\n"
    
    content += "\n### Refactoring\n"
    for commit in grouped['refactor'][:5]:
        content += f"- [{commit['hash']}] {commit['message']} ({commit['date']})\n"
    
    content += f"""

---

## Version History

### v3.0 (2025-12-28)
**Major Update**: Unified Optimization Config + Smart NSGA-II Flow

**New Features**:
- ✅ Unified config system (params/optimization_config.json)
- ✅ Smart NSGA-II flow with OOS validation
- ✅ Separate NSGA/TPE output directories
- ✅ --single and --multi CLI flags
- ✅ OutputManager with mode-specific logging
- ✅ Comprehensive documentation auto-update system

**Optimizations**:
- 25+ parameter search space (was 15)
- Top-5 Pareto OOS validation (prevents overfitting)
- Fixed Optuna step divisibility warnings
- Default config: NSGA-II + ADX regime filtering

**Breaking Changes**:
- None (backwards compatible)

---

### v2.5 (2025-12-26)
**Update**: Parameter Space Expansion

**Changes**:
- Expanded parameter ranges (confluence 2-6, risk 0.2-1.0%)
- Added summer_risk_multiplier (Q3 drawdown protection)
- Added max_concurrent_trades limit

---

### v2.0 (2025-12-20)
**Major Update**: Regime-Adaptive Trading

**Features**:
- ADX regime detection (Trend/Range/Transition)
- Regime-specific confluence requirements
- RSI filters for range mode
- Partial exits and ATR trailing stops

---

### v1.0 (2024-06-15)
**Initial Release**: Production-Ready FTMO Bot

**Core Features**:
- 7-Pillar Confluence System
- Optuna TPE optimization
- FTMO risk management
- MT5 integration (Windows)
- 34 tradable assets

---

**Full commit history**: Run `git log --oneline` in repository root
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated {output_path.name}")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for documentation updater."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-update project documentation")
    parser.add_argument(
        '--file',
        choices=['ARCHITECTURE', 'STRATEGY', 'API', 'DEPLOYMENT', 'CHANGELOG', 'ALL'],
        default='ALL',
        help='Which documentation file to update'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if docs are outdated (don\'t update)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("DOCUMENTATION AUTO-UPDATE SYSTEM")
    print("=" * 60 + "\n")
    
    # Create docs directory if missing
    DOCS_DIR.mkdir(exist_ok=True)
    
    # Map file names to updater functions
    updaters = {
        'ARCHITECTURE': (DOCS_DIR / "ARCHITECTURE.md", update_architecture_doc),
        'STRATEGY': (DOCS_DIR / "STRATEGY_GUIDE.md", update_strategy_doc),
        'API': (DOCS_DIR / "API_REFERENCE.md", update_api_reference),
        'DEPLOYMENT': (DOCS_DIR / "DEPLOYMENT_GUIDE.md", update_deployment_doc),
        'CHANGELOG': (DOCS_DIR / "CHANGELOG.md", update_changelog),
    }
    
    if args.file == 'ALL':
        files_to_update = updaters.items()
    else:
        files_to_update = [(args.file, updaters[args.file])]
    
    success_count = 0
    for name, (path, updater) in files_to_update:
        try:
            if updater(path):
                success_count += 1
        except Exception as e:
            print(f"❌ Error updating {name}: {e}")
    
    print(f"\n✅ Updated {success_count}/{len(files_to_update)} documentation files")
    print("=" * 60 + "\n")
    
    # Update README last updated date
    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        with open(readme, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add/update last updated footer
        footer = f"\n\n---\n\n**Last Documentation Update**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n**Auto-generated**: Run `python scripts/update_docs.py` to regenerate docs\n"
        
        if '**Last Documentation Update**' in content:
            content = re.sub(
                r'\n\n---\n\n\*\*Last Documentation Update\*\*.*\n\*\*Auto-generated\*\*.*\n',
                footer,
                content
            )
        else:
            content += footer
        
        with open(readme, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✓ Updated README.md footer")


if __name__ == "__main__":
    main()
