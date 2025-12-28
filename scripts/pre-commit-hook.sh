#!/bin/bash
# Git pre-commit hook: Auto-update documentation
#
# Installation:
#   cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit

# Check if documentation-critical files are being committed
DOCS_CRITICAL_FILES=(
    "strategy_core.py"
    "ftmo_challenge_analyzer.py"
    "main_live_bot.py"
    "params/optimization_config.json"
    "params/current_params.json"
    "tradr/risk/manager.py"
    "tradr/utils/output_manager.py"
)

# Check if any critical files are staged
SHOULD_UPDATE=false
for file in "${DOCS_CRITICAL_FILES[@]}"; do
    if git diff --cached --name-only | grep -q "^$file$"; then
        SHOULD_UPDATE=true
        break
    fi
done

if [ "$SHOULD_UPDATE" = true ]; then
    echo "📝 Documentation-critical files changed, updating docs..."
    
    # Run documentation updater
    python scripts/update_docs.py
    
    if [ $? -eq 0 ]; then
        # Stage updated docs
        git add docs/*.md README.md
        echo "✅ Documentation auto-updated and staged"
    else
        echo "❌ Warning: Documentation update failed"
        echo "   Run 'python scripts/update_docs.py' manually to fix"
        # Don't block commit, but warn user
    fi
else
    echo "⏭️  No documentation updates needed"
fi

exit 0
