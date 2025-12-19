#!/bin/bash
# Session Start Hook for AI Trading Bot
# This script runs when a Claude Code session starts (web or teleport)

set -e

echo "=== AI Trading Bot Session Initialization ==="

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Python 3 not found!"; exit 1; }

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt 2>/dev/null || pip install -q numpy pandas scikit-learn matplotlib

# Verify core imports work
echo "Verifying trading bot dependencies..."
python3 -c "import numpy; import pandas; import sklearn; print('Core dependencies OK')"

# Check if BTC.CSV data file exists
if [ -f "BTC.CSV" ]; then
    echo "Training data (BTC.CSV) found"
else
    echo "Warning: BTC.CSV training data not found - bot will use simulated data"
fi

# Create session logs directory if needed
mkdir -p session_logs 2>/dev/null || true

echo "=== Session Ready ==="
echo "Trading bot environment configured successfully."
echo "Run 'python3 trading_bot.py' to start the bot (requires display for GUI)."
