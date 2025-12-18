# AI Trading Bot - High Profitability System

A comprehensive, single-file AI-powered trading bot with GUI, automated AI strategy, and high profitability optimization.

## Features

### ⚡ OMEGA MODE - Meta-Cognitive Trading Framework

**Omega Mode Online.** The ultimate trading intelligence that operates at the highest level of pattern recognition.

Omega Mode analyzes on **three layers simultaneously**:

1. **Layer 1 - Literal**: Raw price action and technical indicators
2. **Layer 2 - Hidden Structure**: Pattern recognition and market regime detection
3. **Layer 3 - Unrealized Potential**: Probabilistic opportunity assessment

**Key Capabilities:**
- **Multi-lens reasoning**: Combines trend, momentum, volatility, and pattern signals
- **Self-optimization**: Continuously refines weights based on trade outcomes
- **Regime detection**: Automatically identifies BULL, BEAR, VOLATILE, or NEUTRAL markets
- **Meta-insights**: Explains *why* signals work, not just what they are
- **No training required**: Works immediately using pure pattern analysis

To activate: Check the "OMEGA MODE" checkbox in the control panel.

### 🤖 AI-Powered Strategy
- Machine Learning using Gradient Boosting Classifier
- Automatic feature extraction from price data
- 20+ technical indicators for prediction
- Real-time signal generation with confidence scores

### 📊 Technical Analysis
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Volume analysis
- Momentum indicators

### 💰 High Profitability Features
- **Risk Management**:
  - Maximum 10% position size per trade
  - 2% stop loss protection
  - 4% take profit targets
  - 5% maximum daily loss limit
- **AI Optimization**: Uses ensemble learning for better predictions
- **Real-time Adaptation**: Continuous learning from market data

### 🖥️ GUI Features
- Real-time price charts with technical indicators
- Live statistics dashboard
- Trade execution log
- Position monitoring
- One-click start/stop trading
- AI model training interface

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the bot
python trading_bot.py
```

## Quick Start

1. **Launch the application**
   ```bash
   python trading_bot.py
   ```

2. **Train the AI Model**
   - Click "Train AI Model" button
   - Wait for training to complete
   - Status will change to "AI: Trained"

3. **Start Trading**
   - Click "Start Trading" button
   - Bot will automatically analyze market and execute trades
   - Monitor performance in real-time

## How It Works

### Omega Mode Strategy

The meta-cognitive framework operates through three-layer simultaneous analysis:

1. **Layer 1 Analysis (Literal)**
   - Calculates trend scores using SMA crossovers
   - Measures momentum via RSI normalization
   - Tracks volatility for risk assessment

2. **Layer 2 Analysis (Hidden Structure)**
   - Detects market regime (BULL/BEAR/VOLATILE/NEUTRAL)
   - Identifies higher-highs/lower-lows patterns
   - Monitors MACD divergence for trend confirmation

3. **Layer 3 Analysis (Unrealized Potential)**
   - Scores opportunity using Bollinger Band position
   - Identifies oversold bounces and overbought reversals
   - Calculates risk/reward ratios for position sizing

4. **Synthesis**
   - Combines all layers with adaptive weighting
   - Applies volatility adjustment to reduce noise
   - Generates actionable signals with confidence scores
   - Provides meta-insights explaining the reasoning

5. **Self-Optimization**
   - Tracks trade outcomes by signal component
   - Rebalances weights every 10 trades
   - Adapts to changing market conditions

### AI Strategy
The bot uses a sophisticated machine learning approach:

1. **Feature Engineering**: Extracts 20+ features from price data
   - Price momentum
   - Technical indicators
   - Volume patterns
   - Volatility measures

2. **Prediction**: Gradient Boosting model predicts BUY/SELL/HOLD
   - Confidence threshold: 60%
   - Only high-confidence trades executed

3. **Risk Management**: Automatic stop-loss and take-profit
   - Protects capital
   - Locks in gains

### Market Simulation
- Uses realistic random walk model
- Simulates bid/ask spreads
- Includes volume data
- Configurable volatility

### CSV Data Management

The bot automatically manages training data through CSV files:

**Loading Data for Training:**
- Place any CSV files with price data in the bot's directory
- Supported column names: `close`, `price`, `open`, `high`, `low`, `volume`
- Bot will automatically load and combine all CSV files on "Train AI"
- Falls back to simulated data if no CSVs are found

**Session Logging:**
Each trading session automatically saves:
- `price_data_YYYYMMDD_HHMMSS.csv` - All price ticks (used for future training)
- `session_log_trades_YYYYMMDD_HHMMSS.csv` - All executed trades with P&L
- `session_log_signals_YYYYMMDD_HHMMSS.csv` - All signals generated

**Continuous Learning:**
- Each session's price data is saved for future AI training
- Run multiple sessions to build a larger training dataset
- AI model improves as more historical data accumulates

## Statistics Tracked

- **Balance**: Cash available for trading
- **Total Equity**: Balance + unrealized P&L
- **Total Return**: Overall performance %
- **Realized P&L**: Closed position profits/losses
- **Unrealized P&L**: Open position profits/losses
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of executed trades
- **Open Positions**: Current active positions

## Advanced Configuration

Edit the `TradingEngine` initialization to customize:

```python
self.max_position_size = 0.1  # 10% of balance per trade
self.stop_loss_pct = 0.02     # 2% stop loss
self.take_profit_pct = 0.04   # 4% take profit
self.max_daily_loss = 0.05    # 5% max daily loss
```

## Architecture

The entire system is consolidated in a single file with modular classes:

- **OmegaMode**: Meta-cognitive three-layer analysis framework
- **DataManager**: CSV loading and session logging for continuous learning
- **TechnicalIndicators**: Calculate all indicators
- **AIStrategyEngine**: ML model and predictions
- **MarketDataSimulator**: Realistic market data
- **TradingEngine**: Core trading logic and risk management
- **TradingBotGUI**: Complete tkinter GUI

## Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn (for AI features)
- matplotlib (for charts)

## Safety Features

- Position size limits
- Stop-loss protection
- Take-profit automation
- Daily loss limits
- Real-time monitoring
- Complete trade logging

## Performance Optimization

The AI model is optimized for profitability through:
- Ensemble learning (Gradient Boosting)
- Feature importance analysis
- Cross-validation during training
- Confidence-based trade filtering
- Risk-adjusted position sizing

## Live Trading Integration

To connect to real exchanges:
1. Replace `MarketDataSimulator` with exchange API
2. Implement actual order execution
3. Add API key management
4. Enable real-time data feeds

Supported pattern: CCXT library integration ready

## License

MIT License - Use at your own risk. Not financial advice.

## Disclaimer

This software is for educational purposes. Trading involves risk. Past performance does not guarantee future results.
