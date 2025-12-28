# Deployment Guide

**Last Updated**: 2025-12-28

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
venv\Scripts\activate
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
Get-Content logs\tradr_live.log -Wait -Tail 50

# Check for errors
Select-String -Path logs\tradr_live.log -Pattern "ERROR"
```

### Auto-Start on Boot (Windows)

Create `start_bot.bat`:
```batch
@echo off
cd C:\Users\YourUser\mt5bot-new
venv\Scripts\activate
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
**Fix**: Set `MT5_PATH` in .env: `MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe`

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

**Last Updated**: 2025-12-28  
**Maintained by**: AI-assisted development team  
**Questions**: Open GitHub issue or check docs/FAQ.md
