# AI Trading Bot - High Profitability System

A comprehensive, single-file AI-powered trading bot with GUI, automated AI strategy, and high profitability optimization.

## Features

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
