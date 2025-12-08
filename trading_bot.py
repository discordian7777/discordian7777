#!/usr/bin/env python3
"""
Consolidated AI Trading Bot
Features: GUI, Auto AI Strategy, High Profitability Optimization
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import datetime
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
import queue
import logging

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: scikit-learn not available. AI features will be limited.")

# Plotting imports
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Charts will be disabled.")


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Trade:
    """Represents a single trade"""
    timestamp: datetime.datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    profit: float = 0.0
    strategy: str = "AI"

    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side,
            'price': self.price,
            'quantity': self.quantity,
            'profit': self.profit,
            'strategy': self.strategy
        }


@dataclass
class Position:
    """Represents a current position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0

    def update_pnl(self):
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

class TechnicalIndicators:
    """Calculate technical indicators for trading signals"""

    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD Indicator"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2):
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


# ============================================================================
# AI STRATEGY ENGINE
# ============================================================================

class AIStrategyEngine:
    """AI-powered trading strategy using machine learning"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.is_trained = False
        self.feature_columns = []

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        if df.empty or len(df) < 50:
            return pd.DataFrame()

        features = pd.DataFrame()

        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Technical indicators
        features['sma_5'] = TechnicalIndicators.calculate_sma(df['close'], 5)
        features['sma_20'] = TechnicalIndicators.calculate_sma(df['close'], 20)
        features['sma_50'] = TechnicalIndicators.calculate_sma(df['close'], 50)
        features['ema_12'] = TechnicalIndicators.calculate_ema(df['close'], 12)
        features['ema_26'] = TechnicalIndicators.calculate_ema(df['close'], 26)
        features['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])

        macd, macd_signal, macd_hist = TechnicalIndicators.calculate_macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_hist'] = macd_hist

        upper_bb, middle_bb, lower_bb = TechnicalIndicators.calculate_bollinger_bands(df['close'])
        features['bb_upper'] = upper_bb
        features['bb_middle'] = middle_bb
        features['bb_lower'] = lower_bb
        features['bb_width'] = (upper_bb - lower_bb) / middle_bb

        # Volume features
        if 'volume' in df.columns:
            features['volume_sma'] = df['volume'].rolling(window=20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']

        # Price momentum
        features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features['momentum_10'] = df['close'] / df['close'].shift(10) - 1

        # Volatility
        features['volatility'] = df['close'].pct_change().rolling(window=20).std()

        # Price position relative to moving averages
        features['price_to_sma20'] = df['close'] / features['sma_20'] - 1
        features['price_to_sma50'] = df['close'] / features['sma_50'] - 1

        # Drop NaN values
        features = features.fillna(method='bfill').fillna(0)

        return features

    def create_labels(self, df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.02) -> pd.Series:
        """Create labels for supervised learning"""
        future_returns = df['close'].shift(-lookahead) / df['close'] - 1
        labels = pd.Series(0, index=df.index)  # 0 = HOLD
        labels[future_returns > threshold] = 1  # 1 = BUY
        labels[future_returns < -threshold] = -1  # -1 = SELL
        return labels

    def train(self, historical_data: pd.DataFrame):
        """Train the AI model"""
        if not ML_AVAILABLE:
            logging.warning("ML libraries not available. Cannot train AI model.")
            return False

        if len(historical_data) < 100:
            logging.warning("Not enough data to train AI model.")
            return False

        features = self.prepare_features(historical_data)
        labels = self.create_labels(historical_data)

        # Remove last few rows where we don't have labels
        valid_idx = labels != 0
        features = features[valid_idx]
        labels = labels[valid_idx]

        if len(features) < 50:
            logging.warning("Not enough valid samples to train.")
            return False

        self.feature_columns = features.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train ensemble model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        logging.info(f"AI Model trained. Train score: {train_score:.3f}, Test score: {test_score:.3f}")

        self.is_trained = True
        return True

    def predict(self, current_data: pd.DataFrame) -> Dict:
        """Predict trading signal"""
        if not self.is_trained or self.model is None:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Model not trained'}

        features = self.prepare_features(current_data)

        if features.empty:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}

        # Get last row features
        last_features = features.iloc[-1:][self.feature_columns]

        # Scale and predict
        last_features_scaled = self.scaler.transform(last_features)
        prediction = self.model.predict(last_features_scaled)[0]
        probabilities = self.model.predict_proba(last_features_scaled)[0]

        signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        signal = signal_map[prediction]
        confidence = max(probabilities)

        # Get feature importance for reasoning
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        reason = f"Top factors: {', '.join([f[0] for f in top_features])}"

        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'probabilities': probabilities.tolist()
        }


# ============================================================================
# MARKET DATA SIMULATOR
# ============================================================================

class MarketDataSimulator:
    """Simulates realistic market data for testing"""

    def __init__(self, initial_price: float = 100.0, volatility: float = 0.02):
        self.price = initial_price
        self.volatility = volatility
        self.trend = 0.0001  # Slight upward bias
        self.history = []

    def generate_tick(self) -> Dict:
        """Generate a single price tick"""
        # Add some trend and random walk
        change = np.random.normal(self.trend, self.volatility)
        self.price *= (1 + change)

        # Ensure price stays positive
        self.price = max(self.price, 1.0)

        tick = {
            'timestamp': datetime.datetime.now(),
            'open': self.price * 0.999,
            'high': self.price * 1.002,
            'low': self.price * 0.998,
            'close': self.price,
            'volume': np.random.randint(1000, 10000)
        }

        self.history.append(tick)

        # Keep last 1000 ticks
        if len(self.history) > 1000:
            self.history.pop(0)

        return tick

    def get_historical_data(self, periods: int = 200) -> pd.DataFrame:
        """Get historical data as DataFrame"""
        if not self.history:
            # Generate initial history
            for _ in range(periods):
                self.generate_tick()
                time.sleep(0.001)  # Small delay

        return pd.DataFrame(self.history[-periods:])


# ============================================================================
# TRADING ENGINE
# ============================================================================

class TradingEngine:
    """Core trading engine"""

    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.running = False

        # Risk management
        self.max_position_size = 0.1  # 10% of balance per trade
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        self.max_daily_loss = 0.05  # 5% max daily loss

        # Strategy
        self.ai_engine = AIStrategyEngine()
        self.market_simulator = MarketDataSimulator()

        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk management"""
        max_investment = self.balance * self.max_position_size
        quantity = max_investment / price
        return round(quantity, 4)

    def open_position(self, symbol: str, side: str, price: float, quantity: float, strategy: str = "AI"):
        """Open a new position"""
        cost = price * quantity

        if cost > self.balance:
            logging.warning(f"Insufficient balance to open position. Cost: {cost}, Balance: {self.balance}")
            return False

        if symbol in self.positions:
            logging.warning(f"Position already exists for {symbol}")
            return False

        self.balance -= cost
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity if side == 'BUY' else -quantity,
            entry_price=price,
            current_price=price
        )

        trade = Trade(
            timestamp=datetime.datetime.now(),
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            strategy=strategy
        )
        self.trades.append(trade)
        self.total_trades += 1

        logging.info(f"Opened {side} position: {quantity} {symbol} @ {price}")
        return True

    def close_position(self, symbol: str, price: float):
        """Close an existing position"""
        if symbol not in self.positions:
            logging.warning(f"No position exists for {symbol}")
            return False

        position = self.positions[symbol]
        proceeds = abs(position.quantity) * price
        profit = proceeds - (abs(position.quantity) * position.entry_price)

        self.balance += proceeds

        # Record trade
        side = 'SELL' if position.quantity > 0 else 'BUY'
        trade = Trade(
            timestamp=datetime.datetime.now(),
            symbol=symbol,
            side=side,
            price=price,
            quantity=abs(position.quantity),
            profit=profit
        )
        self.trades.append(trade)
        self.total_trades += 1

        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        logging.info(f"Closed {side} position: {abs(position.quantity)} {symbol} @ {price}, P&L: {profit:.2f}")

        del self.positions[symbol]
        return True

    def update_positions(self, symbol: str, current_price: float):
        """Update position with current price and check stop loss/take profit"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        position.current_price = current_price
        position.update_pnl()

        # Check stop loss and take profit
        pnl_pct = position.unrealized_pnl / (abs(position.quantity) * position.entry_price)

        if pnl_pct <= -self.stop_loss_pct:
            logging.info(f"Stop loss triggered for {symbol} at {pnl_pct*100:.2f}%")
            self.close_position(symbol, current_price)
        elif pnl_pct >= self.take_profit_pct:
            logging.info(f"Take profit triggered for {symbol} at {pnl_pct*100:.2f}%")
            self.close_position(symbol, current_price)

    def get_total_equity(self) -> float:
        """Calculate total equity (balance + unrealized P&L)"""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.balance + unrealized_pnl

    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        total_equity = self.get_total_equity()
        total_return = ((total_equity - self.initial_balance) / self.initial_balance) * 100

        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        realized_pnl = sum(trade.profit for trade in self.trades)
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        return {
            'balance': self.balance,
            'total_equity': total_equity,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'open_positions': len(self.positions)
        }


# ============================================================================
# GUI APPLICATION
# ============================================================================

class TradingBotGUI:
    """Main GUI application"""

    def __init__(self, root):
        self.root = root
        self.root.title("AI Trading Bot - High Profitability System")
        self.root.geometry("1400x900")

        # Trading engine
        self.engine = TradingEngine(initial_balance=10000.0)
        self.symbol = "BTC/USD"

        # GUI state
        self.is_running = False
        self.update_queue = queue.Queue()

        # Setup logging
        self.setup_logging()

        # Create GUI
        self.create_widgets()

        # Start GUI update loop
        self.update_gui()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # ===== Control Panel =====
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Trading", command=self.start_trading)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Trading", command=self.stop_trading, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=5)

        # Train AI button
        self.train_button = ttk.Button(control_frame, text="Train AI Model", command=self.train_ai)
        self.train_button.grid(row=0, column=2, padx=5)

        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Stopped", foreground="red")
        self.status_label.grid(row=0, column=3, padx=20)

        # AI Status
        self.ai_status_label = ttk.Label(control_frame, text="AI: Not Trained", foreground="orange")
        self.ai_status_label.grid(row=0, column=4, padx=5)

        # ===== Statistics Panel =====
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)

        self.stats_labels = {}
        stats_items = [
            ('Balance', 'balance'),
            ('Total Equity', 'equity'),
            ('Total Return', 'return'),
            ('Realized P&L', 'realized'),
            ('Unrealized P&L', 'unrealized'),
            ('Total Trades', 'trades'),
            ('Win Rate', 'winrate'),
            ('Open Positions', 'positions')
        ]

        for i, (label, key) in enumerate(stats_items):
            ttk.Label(stats_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="$0.00", font=('Arial', 10, 'bold'))
            self.stats_labels[key].grid(row=i, column=1, sticky=tk.W, padx=10, pady=2)

        # ===== Chart Panel =====
        if PLOTTING_AVAILABLE:
            chart_frame = ttk.LabelFrame(main_frame, text="Price Chart", padding="10")
            chart_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)

            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.ax1 = self.fig.add_subplot(211)
            self.ax2 = self.fig.add_subplot(212)

            self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ===== Log Panel =====
        log_frame = ttk.LabelFrame(main_frame, text="Trading Log", padding="10")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, width=50, height=20, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Update initial statistics
        self.update_statistics()

    def log_message(self, message: str):
        """Add message to log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def update_statistics(self):
        """Update statistics display"""
        stats = self.engine.get_statistics()

        self.stats_labels['balance'].config(text=f"${stats['balance']:.2f}")
        self.stats_labels['equity'].config(text=f"${stats['total_equity']:.2f}")

        return_color = "green" if stats['total_return'] >= 0 else "red"
        self.stats_labels['return'].config(
            text=f"{stats['total_return']:.2f}%",
            foreground=return_color
        )

        realized_color = "green" if stats['realized_pnl'] >= 0 else "red"
        self.stats_labels['realized'].config(
            text=f"${stats['realized_pnl']:.2f}",
            foreground=realized_color
        )

        unrealized_color = "green" if stats['unrealized_pnl'] >= 0 else "red"
        self.stats_labels['unrealized'].config(
            text=f"${stats['unrealized_pnl']:.2f}",
            foreground=unrealized_color
        )

        self.stats_labels['trades'].config(text=str(stats['total_trades']))
        self.stats_labels['winrate'].config(text=f"{stats['win_rate']:.1f}%")
        self.stats_labels['positions'].config(text=str(stats['open_positions']))

    def update_chart(self):
        """Update price chart"""
        if not PLOTTING_AVAILABLE:
            return

        df = self.engine.market_simulator.get_historical_data(periods=100)

        if df.empty:
            return

        self.ax1.clear()
        self.ax2.clear()

        # Price chart
        self.ax1.plot(df.index, df['close'], label='Price', color='blue', linewidth=1.5)

        # Add moving averages
        if len(df) >= 20:
            sma20 = TechnicalIndicators.calculate_sma(df['close'], 20)
            self.ax1.plot(df.index, sma20, label='SMA 20', color='orange', linewidth=1, alpha=0.7)

        if len(df) >= 50:
            sma50 = TechnicalIndicators.calculate_sma(df['close'], 50)
            self.ax1.plot(df.index, sma50, label='SMA 50', color='red', linewidth=1, alpha=0.7)

        # Mark trades
        for trade in self.engine.trades[-20:]:  # Last 20 trades
            try:
                idx = df[df['timestamp'] >= trade.timestamp].index[0]
                color = 'green' if trade.side == 'BUY' else 'red'
                marker = '^' if trade.side == 'BUY' else 'v'
                self.ax1.scatter(idx, trade.price, color=color, marker=marker, s=100, zorder=5)
            except (IndexError, KeyError):
                pass

        self.ax1.set_ylabel('Price')
        self.ax1.legend(loc='upper left', fontsize=8)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title(f'{self.symbol} - AI Trading Bot', fontsize=10)

        # Volume chart
        if 'volume' in df.columns:
            self.ax2.bar(df.index, df['volume'], color='gray', alpha=0.5)
            self.ax2.set_ylabel('Volume')
            self.ax2.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def train_ai(self):
        """Train AI model"""
        self.log_message("Training AI model...")
        self.train_button.config(state='disabled')

        def train_thread():
            # Generate historical data
            historical_df = self.engine.market_simulator.get_historical_data(periods=500)

            # Train model
            success = self.engine.ai_engine.train(historical_df)

            if success:
                self.update_queue.put(('ai_trained', True))
                self.update_queue.put(('log', 'AI model trained successfully!'))
            else:
                self.update_queue.put(('ai_trained', False))
                self.update_queue.put(('log', 'Failed to train AI model.'))

        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()

    def start_trading(self):
        """Start trading bot"""
        if not self.engine.ai_engine.is_trained:
            messagebox.showwarning("Warning", "Please train the AI model first!")
            return

        self.is_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_label.config(text="Status: Running", foreground="green")

        self.log_message("Trading bot started!")

        # Start trading thread
        thread = threading.Thread(target=self.trading_loop, daemon=True)
        thread.start()

    def stop_trading(self):
        """Stop trading bot"""
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Status: Stopped", foreground="red")

        self.log_message("Trading bot stopped!")

    def trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Generate new market tick
                tick = self.engine.market_simulator.generate_tick()
                current_price = tick['close']

                # Update positions
                if self.symbol in self.engine.positions:
                    self.engine.update_positions(self.symbol, current_price)

                # Get historical data for AI
                historical_df = self.engine.market_simulator.get_historical_data(periods=200)

                # Get AI prediction
                prediction = self.engine.ai_engine.predict(historical_df)
                signal = prediction['signal']
                confidence = prediction['confidence']

                # Trading logic
                if signal == 'BUY' and confidence > 0.6 and self.symbol not in self.engine.positions:
                    quantity = self.engine.calculate_position_size(current_price)
                    if self.engine.open_position(self.symbol, 'BUY', current_price, quantity):
                        msg = f"BUY {self.symbol} @ ${current_price:.2f} (Conf: {confidence:.2f})"
                        self.update_queue.put(('log', msg))

                elif signal == 'SELL' and self.symbol in self.engine.positions:
                    if self.engine.close_position(self.symbol, current_price):
                        msg = f"SELL {self.symbol} @ ${current_price:.2f} (Conf: {confidence:.2f})"
                        self.update_queue.put(('log', msg))

                # Update equity curve
                self.engine.equity_curve.append({
                    'timestamp': tick['timestamp'],
                    'equity': self.engine.get_total_equity()
                })

                # Queue update
                self.update_queue.put(('update', None))

                # Sleep
                time.sleep(1)  # 1 second per tick

            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                self.update_queue.put(('log', f"Error: {str(e)}"))

    def update_gui(self):
        """Update GUI from queue"""
        try:
            while True:
                msg_type, msg_data = self.update_queue.get_nowait()

                if msg_type == 'log':
                    self.log_message(msg_data)
                elif msg_type == 'update':
                    self.update_statistics()
                    self.update_chart()
                elif msg_type == 'ai_trained':
                    if msg_data:
                        self.ai_status_label.config(text="AI: Trained", foreground="green")
                    self.train_button.config(state='normal')

        except queue.Empty:
            pass

        # Schedule next update
        self.root.after(100, self.update_gui)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main entry point"""
    print("=" * 60)
    print("AI Trading Bot - High Profitability System")
    print("=" * 60)
    print(f"ML Available: {ML_AVAILABLE}")
    print(f"Plotting Available: {PLOTTING_AVAILABLE}")
    print("=" * 60)

    if not ML_AVAILABLE:
        print("\nWARNING: scikit-learn not installed.")
        print("Install with: pip install scikit-learn")
        print("AI features will be limited.\n")

    if not PLOTTING_AVAILABLE:
        print("\nWARNING: matplotlib not installed.")
        print("Install with: pip install matplotlib")
        print("Charts will be disabled.\n")

    # Create GUI
    root = tk.Tk()
    app = TradingBotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
