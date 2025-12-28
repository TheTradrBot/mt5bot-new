# Scripts Directory

Utility scripts for MT5 FTMO Trading Bot maintenance and automation.

---

## Documentation Auto-Update System

### Overview
The bot includes an **automated documentation system** that keeps all docs synchronized with source code. Documentation is auto-generated from:
- Function signatures & docstrings
- Configuration files (JSON)
- Git commit history
- Parameter values

### Files

#### `update_docs.py` - Main Documentation Generator
**Purpose**: Auto-generate/update all documentation files

**Usage**:
```bash
# Update all documentation
python scripts/update_docs.py

# Update specific doc
python scripts/update_docs.py --file STRATEGY
python scripts/update_docs.py --file API
python scripts/update_docs.py --file DEPLOYMENT

# Check if docs are outdated (don't update)
python scripts/update_docs.py --check
```

**Generated Files**:
- `docs/ARCHITECTURE.md` (28KB) - System architecture, data flow, component details
- `docs/STRATEGY_GUIDE.md` (11KB) - Trading strategy with live parameters
- `docs/API_REFERENCE.md` (46KB) - Complete API documentation
- `docs/DEPLOYMENT_GUIDE.md` (8KB) - Setup, deployment, troubleshooting
- `docs/CHANGELOG.md` (2KB) - Version history from git commits

**Extraction Methods**:
- **AST Parsing**: Extracts function signatures, docstrings, type hints
- **JSON Loading**: Reads current parameters from `params/current_params.json`
- **Git Integration**: Fetches recent commits for changelog
- **File Tree**: Generates directory structure

#### `pre-commit-hook.sh` - Local Auto-Update Hook
**Purpose**: Automatically update docs before committing changes

**Installation**:
```bash
# One-time setup
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Behavior**:
- Triggers when committing changes to:
  - `strategy_core.py`
  - `ftmo_challenge_analyzer.py`
  - `main_live_bot.py`
  - `params/*.json`
  - `tradr/**/*.py`
- Runs `update_docs.py` automatically
- Stages updated docs in same commit
- **Never blocks commits** (warns on failure)

**Disable for specific commit**:
```bash
git commit --no-verify
```

---

## GitHub Actions Workflow

### `.github/workflows/update-docs.yml`
**Purpose**: Auto-update docs on every push to main branch

**Triggers**:
- Push to `main` branch with changes to source files
- Manual trigger via GitHub Actions UI

**Process**:
1. Checkout repository
2. Set up Python 3.11
3. Install dependencies
4. Run `update_docs.py`
5. Commit & push if docs changed
6. Uses `[skip ci]` to prevent infinite loops

**Status**: Check at `https://github.com/TheTradrBot/mt5bot-new/actions`

---

## Other Utility Scripts

### `monitor_optimization.sh`
Monitor live optimization progress:
```bash
./scripts/monitor_optimization.sh
```

### `validate_setup.py`
Pre-deployment validation:
```bash
python scripts/validate_setup.py
```

Checks:
- ✓ `params/current_params.json` exists
- ✓ All 25 parameters present
- ✓ Historical data files present
- ✓ MT5 connection (Windows)
- ✓ Contract specs loaded

---

## Maintenance

### Weekly Tasks
```bash
# Update all documentation
python scripts/update_docs.py

# Validate setup
python scripts/validate_setup.py

# Check optimization status
python ftmo_challenge_analyzer.py --status
```

### Monthly Tasks
```bash
# Update historical data
python update_csvs.py

# Re-generate docs with fresh data
python scripts/update_docs.py

# Backup parameter history
tar -czf params_backup_$(date +%Y%m%d).tar.gz params/history/
```

---

## Troubleshooting

**Issue**: `update_docs.py` fails with import errors  
**Fix**: Install dependencies: `pip install -r requirements.txt`

**Issue**: Pre-commit hook not executing  
**Fix**: Ensure hook is executable: `chmod +x .git/hooks/pre-commit`

**Issue**: GitHub Actions not triggering  
**Fix**: Check workflow file syntax, verify push to `main` branch

**Issue**: Documentation outdated despite auto-update  
**Fix**: Run manually: `python scripts/update_docs.py && git add docs/ && git commit -m "docs: Manual update"`

---

## Contributing

When adding new features:
1. **Add docstrings** to all functions/classes
2. **Use type hints** for parameters and returns
3. **Update config** if adding new parameters
4. **Run `update_docs.py`** before committing
5. **Verify docs** are accurate and complete

The auto-update system will propagate your docstrings to the documentation automatically.

---

**Last Updated**: 2025-12-28  
**Maintained by**: Auto-generated documentation system
