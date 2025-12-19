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
import os
import glob as glob_module
import requests
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# DATA MANAGER - CSV LOADING AND SESSION LOGGING
# ============================================================================

class DataManager:
    """
    Manages CSV data loading and session logging.

    - Loads all CSV files from directory for AI training
    - Saves session logs as CSV for future training
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_trades = []
        self.session_signals = []
        self.session_prices = []

    def find_csv_files(self) -> List[str]:
        """Find all CSV files in the data directory"""
        pattern = os.path.join(self.data_dir, "*.csv")
        csv_files = glob_module.glob(pattern)
        # Exclude session logs from training data (they have different format)
        training_files = [f for f in csv_files if not os.path.basename(f).startswith("session_log_")]
        return training_files

    def load_historical_data(self) -> pd.DataFrame:
        """Load and combine all CSV files for training"""
        csv_files = self.find_csv_files()

        if not csv_files:
            logging.info("No CSV files found for training data")
            return pd.DataFrame()

        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                # Standardize column names
                df.columns = df.columns.str.lower().str.strip()

                # Check for required columns (flexible naming)
                # Supports: Standard OHLCV, Binance format, Yahoo Finance, etc.
                required_cols = ['close']
                alt_names = {
                    'close': ['close', 'price', 'last', 'adj close', 'adj_close', 'close_price'],
                    'open': ['open', 'open_price'],
                    'high': ['high', 'high_price'],
                    'low': ['low', 'low_price'],
                    'volume': ['volume', 'vol', 'quote_asset_volume', 'base_volume'],
                    'timestamp': ['timestamp', 'open_time', 'close_time', 'open_dt', 'close_dt', 'date', 'time', 'datetime']
                }

                # Handle Binance-style data with num_trades, taker volumes, etc.
                # These columns are preserved but not required
                binance_extra_cols = ['num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'quote_asset_volume']

                # Map columns to standard names
                for std_name, alternatives in alt_names.items():
                    for alt in alternatives:
                        if alt in df.columns and std_name not in df.columns:
                            df[std_name] = df[alt]
                            break

                if 'close' in df.columns:
                    # Generate OHLC if only close is available
                    if 'open' not in df.columns:
                        df['open'] = df['close'].shift(1).fillna(df['close'])
                    if 'high' not in df.columns:
                        df['high'] = df['close'] * 1.001
                    if 'low' not in df.columns:
                        df['low'] = df['close'] * 0.999
                    if 'volume' not in df.columns:
                        df['volume'] = 1000

                    all_data.append(df)
                    logging.info(f"Loaded {len(df)} rows from {os.path.basename(csv_file)}")
                else:
                    logging.warning(f"Skipping {csv_file}: no 'close' column found")

            except Exception as e:
                logging.error(f"Error loading {csv_file}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Combine all dataframes
        combined = pd.concat(all_data, ignore_index=True)
        logging.info(f"Total historical data: {len(combined)} rows from {len(all_data)} files")
        return combined

    def log_trade(self, trade: Dict):
        """Log a trade for session export"""
        trade_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'symbol': trade.get('symbol', 'UNKNOWN'),
            'side': trade.get('side', ''),
            'price': trade.get('price', 0),
            'quantity': trade.get('quantity', 0),
            'profit': trade.get('profit', 0),
            'strategy': trade.get('strategy', 'AI'),
            'confidence': trade.get('confidence', 0),
            'regime': trade.get('regime', 'N/A'),
            'signal_reason': trade.get('reason', '')
        }
        self.session_trades.append(trade_record)

    def log_signal(self, signal: Dict, price: float):
        """Log a signal for session export"""
        signal_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'price': price,
            'signal': signal.get('signal', 'HOLD'),
            'confidence': signal.get('confidence', 0),
            'regime': signal.get('regime', 'N/A'),
            'opportunity': signal.get('opportunity', 'NONE'),
            'raw_signal': signal.get('raw_signal', 0),
            'reason': signal.get('reason', '')
        }
        self.session_signals.append(signal_record)

    def log_price(self, tick: Dict):
        """Log price data for session export"""
        price_record = {
            'timestamp': tick.get('timestamp', datetime.datetime.now()).isoformat()
                if isinstance(tick.get('timestamp'), datetime.datetime)
                else str(tick.get('timestamp', '')),
            'open': tick.get('open', 0),
            'high': tick.get('high', 0),
            'low': tick.get('low', 0),
            'close': tick.get('close', 0),
            'volume': tick.get('volume', 0)
        }
        self.session_prices.append(price_record)

    def save_session_log(self) -> Dict[str, str]:
        """Save session data to CSV files for future training"""
        saved_files = {}

        # Save trades log
        if self.session_trades:
            trades_file = os.path.join(self.data_dir, f"session_log_trades_{self.session_id}.csv")
            trades_df = pd.DataFrame(self.session_trades)
            trades_df.to_csv(trades_file, index=False)
            saved_files['trades'] = trades_file
            logging.info(f"Saved {len(self.session_trades)} trades to {trades_file}")

        # Save price data (for future AI training)
        if self.session_prices:
            prices_file = os.path.join(self.data_dir, f"price_data_{self.session_id}.csv")
            prices_df = pd.DataFrame(self.session_prices)
            prices_df.to_csv(prices_file, index=False)
            saved_files['prices'] = prices_file
            logging.info(f"Saved {len(self.session_prices)} price records to {prices_file}")

        # Save signals log
        if self.session_signals:
            signals_file = os.path.join(self.data_dir, f"session_log_signals_{self.session_id}.csv")
            signals_df = pd.DataFrame(self.session_signals)
            signals_df.to_csv(signals_file, index=False)
            saved_files['signals'] = signals_file
            logging.info(f"Saved {len(self.session_signals)} signals to {signals_file}")

        return saved_files

    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        return {
            'session_id': self.session_id,
            'total_trades': len(self.session_trades),
            'total_signals': len(self.session_signals),
            'total_price_points': len(self.session_prices)
        }


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
# OMEGA MODE - META-COGNITIVE TRADING FRAMEWORK
# ============================================================================

class OmegaMode:
    """
    Omega Mode: Multi-layer simultaneous analysis framework.

    Analyzes on three layers:
    1. Literal: Raw price action and technical indicators
    2. Hidden Structure: Pattern recognition and market regime detection
    3. Unrealized Potential: Probabilistic opportunity assessment

    Features:
    - Self-optimization through continuous learning
    - Multi-lens reasoning (trend, momentum, volatility, sentiment)
    - Automatic inference when data is incomplete
    - Meta-pattern recognition for regime adaptation
    """

    def __init__(self):
        self.is_active = False
        self.performance_history = []
        self.regime_state = "NEUTRAL"  # BULL, BEAR, NEUTRAL, VOLATILE
        self.confidence_threshold = 0.65
        self.meta_weights = {
            'trend': 0.25,
            'momentum': 0.25,
            'volatility': 0.20,
            'pattern': 0.30
        }
        self.optimization_counter = 0
        self.last_signals = []

    def activate(self):
        """Activate Omega Mode"""
        self.is_active = True
        return "Omega Mode Online."

    def deactivate(self):
        """Deactivate Omega Mode"""
        self.is_active = False
        return "Omega Mode Offline."

    def analyze_three_layers(self, df: pd.DataFrame) -> Dict:
        """
        Simultaneous three-layer analysis.

        Returns insights from all three layers.
        """
        if df.empty or len(df) < 50:
            return {'layer1': None, 'layer2': None, 'layer3': None, 'synthesis': None}

        close = df['close']

        # Layer 1: Literal - Raw technical signals
        layer1 = self._analyze_literal(df)

        # Layer 2: Hidden Structure - Pattern & regime detection
        layer2 = self._analyze_hidden_structure(df)

        # Layer 3: Unrealized Potential - Opportunity assessment
        layer3 = self._analyze_potential(df, layer1, layer2)

        return {
            'layer1': layer1,
            'layer2': layer2,
            'layer3': layer3,
            'synthesis': self._synthesize_layers(layer1, layer2, layer3)
        }

    def _analyze_literal(self, df: pd.DataFrame) -> Dict:
        """Layer 1: Literal price action analysis"""
        close = df['close']

        # Trend analysis
        sma_20 = TechnicalIndicators.calculate_sma(close, 20).iloc[-1]
        sma_50 = TechnicalIndicators.calculate_sma(close, 50).iloc[-1]
        current_price = close.iloc[-1]

        trend_score = 0
        if current_price > sma_20 > sma_50:
            trend_score = 1  # Strong uptrend
        elif current_price < sma_20 < sma_50:
            trend_score = -1  # Strong downtrend
        elif current_price > sma_20:
            trend_score = 0.5  # Mild uptrend
        elif current_price < sma_20:
            trend_score = -0.5  # Mild downtrend

        # Momentum analysis
        rsi = TechnicalIndicators.calculate_rsi(close).iloc[-1]
        momentum_score = (rsi - 50) / 50  # Normalize to -1 to 1

        # Volatility analysis
        volatility = close.pct_change().rolling(window=20).std().iloc[-1]

        return {
            'trend_score': trend_score,
            'momentum_score': momentum_score,
            'volatility': volatility,
            'rsi': rsi,
            'price_vs_sma20': (current_price - sma_20) / sma_20
        }

    def _analyze_hidden_structure(self, df: pd.DataFrame) -> Dict:
        """Layer 2: Hidden pattern and regime detection"""
        close = df['close']

        # Detect market regime
        returns = close.pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1]
        avg_volatility = returns.rolling(window=50).std().mean()

        mean_return = returns.rolling(window=20).mean().iloc[-1]

        # Regime classification
        if volatility > avg_volatility * 1.5:
            regime = "VOLATILE"
        elif mean_return > 0.001 and volatility < avg_volatility:
            regime = "BULL"
        elif mean_return < -0.001 and volatility < avg_volatility:
            regime = "BEAR"
        else:
            regime = "NEUTRAL"

        self.regime_state = regime

        # Pattern recognition: Higher highs/lower lows
        recent_highs = df['high'].iloc[-20:]
        recent_lows = df['low'].iloc[-20:]

        higher_highs = sum(recent_highs.iloc[i] > recent_highs.iloc[i-1]
                          for i in range(1, len(recent_highs)))
        lower_lows = sum(recent_lows.iloc[i] < recent_lows.iloc[i-1]
                        for i in range(1, len(recent_lows)))

        pattern_score = (higher_highs - lower_lows) / 19  # Normalize

        # MACD divergence detection
        macd, macd_signal, macd_hist = TechnicalIndicators.calculate_macd(close)
        macd_trend = 1 if macd_hist.iloc[-1] > macd_hist.iloc[-5] else -1

        return {
            'regime': regime,
            'pattern_score': pattern_score,
            'macd_trend': macd_trend,
            'volatility_state': 'HIGH' if volatility > avg_volatility else 'LOW'
        }

    def _analyze_potential(self, df: pd.DataFrame, layer1: Dict, layer2: Dict) -> Dict:
        """Layer 3: Unrealized potential and opportunity assessment"""
        close = df['close']
        current_price = close.iloc[-1]

        # Bollinger Band position (opportunity detection)
        upper_bb, middle_bb, lower_bb = TechnicalIndicators.calculate_bollinger_bands(close)
        bb_position = (current_price - lower_bb.iloc[-1]) / (upper_bb.iloc[-1] - lower_bb.iloc[-1])

        # Opportunity scoring
        opportunity_score = 0
        opportunity_type = "NONE"

        # Oversold bounce potential
        if layer1['rsi'] < 30 and layer2['regime'] != "BEAR":
            opportunity_score = 0.8
            opportunity_type = "OVERSOLD_BOUNCE"
        # Overbought reversal potential
        elif layer1['rsi'] > 70 and layer2['regime'] != "BULL":
            opportunity_score = -0.8
            opportunity_type = "OVERBOUGHT_REVERSAL"
        # Trend continuation potential
        elif layer2['regime'] == "BULL" and layer1['trend_score'] > 0.5:
            opportunity_score = 0.6
            opportunity_type = "TREND_CONTINUATION"
        elif layer2['regime'] == "BEAR" and layer1['trend_score'] < -0.5:
            opportunity_score = -0.6
            opportunity_type = "TREND_CONTINUATION"
        # Breakout potential
        elif bb_position > 0.95:
            opportunity_score = 0.4 if layer1['momentum_score'] > 0 else -0.3
            opportunity_type = "UPPER_BB_BREAKOUT" if opportunity_score > 0 else "UPPER_BB_REVERSAL"
        elif bb_position < 0.05:
            opportunity_score = 0.4 if layer1['momentum_score'] < 0 else 0.3
            opportunity_type = "LOWER_BB_BOUNCE" if opportunity_score > 0 else "LOWER_BB_BREAKDOWN"

        return {
            'opportunity_score': opportunity_score,
            'opportunity_type': opportunity_type,
            'bb_position': bb_position,
            'risk_reward_ratio': self._calculate_risk_reward(df, opportunity_score)
        }

    def _calculate_risk_reward(self, df: pd.DataFrame, opportunity_score: float) -> float:
        """Calculate expected risk/reward ratio"""
        if abs(opportunity_score) < 0.3:
            return 1.0  # Neutral

        close = df['close']
        atr = close.pct_change().rolling(window=14).std().iloc[-1] * np.sqrt(14)

        if opportunity_score > 0:
            # Bullish opportunity
            reward = atr * 2  # Expected 2 ATR move
            risk = atr * 0.5   # Stop loss at 0.5 ATR
        else:
            # Bearish opportunity
            reward = atr * 2
            risk = atr * 0.5

        return reward / risk if risk > 0 else 1.0

    def _synthesize_layers(self, layer1: Dict, layer2: Dict, layer3: Dict) -> Dict:
        """Synthesize all three layers into final signal"""

        # Multi-lens weighted synthesis
        trend_signal = layer1['trend_score'] * self.meta_weights['trend']
        momentum_signal = layer1['momentum_score'] * self.meta_weights['momentum']
        pattern_signal = layer2['pattern_score'] * self.meta_weights['pattern']

        # Adjust for volatility
        vol_multiplier = 0.8 if layer2['volatility_state'] == 'HIGH' else 1.0

        # Opportunity amplification
        opportunity_boost = layer3['opportunity_score'] * 0.3

        raw_signal = (trend_signal + momentum_signal + pattern_signal + opportunity_boost) * vol_multiplier

        # Determine final action
        if raw_signal > 0.3:
            action = "BUY"
            confidence = min(0.5 + abs(raw_signal), 0.95)
        elif raw_signal < -0.3:
            action = "SELL"
            confidence = min(0.5 + abs(raw_signal), 0.95)
        else:
            action = "HOLD"
            confidence = 0.5 - abs(raw_signal)

        return {
            'action': action,
            'confidence': confidence,
            'raw_signal': raw_signal,
            'regime': layer2['regime'],
            'opportunity': layer3['opportunity_type'],
            'meta_insight': self._generate_meta_insight(layer1, layer2, layer3)
        }

    def _generate_meta_insight(self, layer1: Dict, layer2: Dict, layer3: Dict) -> str:
        """Generate meta-level insight about why the signal works"""
        regime = layer2['regime']
        opp = layer3['opportunity_type']

        if opp == "OVERSOLD_BOUNCE":
            return f"Mean reversion in {regime} regime: RSI({layer1['rsi']:.0f}) + momentum divergence"
        elif opp == "TREND_CONTINUATION":
            return f"{regime} trend alignment: All layers confirm directional bias"
        elif opp == "UPPER_BB_BREAKOUT":
            return "Volatility expansion + momentum: Potential breakout continuation"
        elif opp == "LOWER_BB_BOUNCE":
            return "Support confluence: BB lower band + RSI oversold"
        else:
            return f"Regime: {regime} | Pattern score: {layer2['pattern_score']:.2f}"

    def self_optimize(self, trade_result: Dict):
        """
        Self-optimization: Adjust weights based on trade outcomes.
        Continuous learning from results.
        """
        self.performance_history.append(trade_result)
        self.optimization_counter += 1

        # Optimize every 10 trades
        if self.optimization_counter >= 10:
            self._run_optimization()
            self.optimization_counter = 0

    def _run_optimization(self):
        """Run meta-weight optimization based on performance"""
        if len(self.performance_history) < 10:
            return

        recent = self.performance_history[-10:]

        # Calculate which signals were most accurate
        trend_accuracy = sum(1 for t in recent if t.get('trend_correct', False)) / 10
        momentum_accuracy = sum(1 for t in recent if t.get('momentum_correct', False)) / 10
        pattern_accuracy = sum(1 for t in recent if t.get('pattern_correct', False)) / 10

        total = trend_accuracy + momentum_accuracy + pattern_accuracy + 0.001

        # Rebalance weights
        self.meta_weights['trend'] = 0.15 + (trend_accuracy / total) * 0.20
        self.meta_weights['momentum'] = 0.15 + (momentum_accuracy / total) * 0.20
        self.meta_weights['pattern'] = 0.15 + (pattern_accuracy / total) * 0.25
        self.meta_weights['volatility'] = 1.0 - sum([
            self.meta_weights['trend'],
            self.meta_weights['momentum'],
            self.meta_weights['pattern']
        ])

    def generate_signal(self, df: pd.DataFrame) -> Dict:
        """
        Main signal generation with Omega Mode analysis.
        """
        if not self.is_active:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Omega Mode not active'}

        analysis = self.analyze_three_layers(df)

        if analysis['synthesis'] is None:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}

        synthesis = analysis['synthesis']

        self.last_signals.append(synthesis)
        if len(self.last_signals) > 100:
            self.last_signals.pop(0)

        return {
            'signal': synthesis['action'],
            'confidence': synthesis['confidence'],
            'reason': synthesis['meta_insight'],
            'regime': synthesis['regime'],
            'opportunity': synthesis['opportunity'],
            'raw_signal': synthesis['raw_signal']
        }


# ============================================================================
# AI COUNCIL - MULTI-AI ENSEMBLE DECISION SYSTEM
# ============================================================================

class AIProvider(ABC):
    """Abstract base class for AI providers"""

    @abstractmethod
    def get_trade_decision(self, market_context: str) -> Dict:
        """Get trade decision from AI provider"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get provider name"""
        pass


class DeepSeekProvider(AIProvider):
    """DeepSeek AI integration for trade decisions"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY', '')
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"

    def get_name(self) -> str:
        return "DeepSeek"

    def get_trade_decision(self, market_context: str) -> Dict:
        if not self.api_key:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': 'No API key configured', 'failed': True}

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            prompt = f"""You are an expert trading AI. Analyze this market data and provide a trading decision.

{market_context}

Respond in JSON format only:
{{"signal": "BUY" or "SELL" or "HOLD", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200
            }

            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse JSON response
            decision = json.loads(content.strip().replace('```json', '').replace('```', ''))
            decision['provider'] = self.get_name()
            return decision

        except Exception as e:
            logging.error(f"DeepSeek API error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': f'API error: {str(e)}', 'provider': self.get_name(), 'failed': True}


class OpenAIProvider(AIProvider):
    """OpenAI GPT integration for trade decisions"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o-mini"

    def get_name(self) -> str:
        return "OpenAI"

    def get_trade_decision(self, market_context: str) -> Dict:
        if not self.api_key:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': 'No API key configured', 'failed': True}

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            prompt = f"""You are an expert trading AI. Analyze this market data and provide a trading decision.

{market_context}

Respond in JSON format only:
{{"signal": "BUY" or "SELL" or "HOLD", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200
            }

            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']

            decision = json.loads(content.strip().replace('```json', '').replace('```', ''))
            decision['provider'] = self.get_name()
            return decision

        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': f'API error: {str(e)}', 'provider': self.get_name(), 'failed': True}


class ClaudeProvider(AIProvider):
    """Anthropic Claude integration for trade decisions"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-haiku-20240307"

    def get_name(self) -> str:
        return "Claude"

    def get_trade_decision(self, market_context: str) -> Dict:
        if not self.api_key:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': 'No API key configured', 'failed': True}

        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }

            prompt = f"""You are an expert trading AI. Analyze this market data and provide a trading decision.

{market_context}

Respond in JSON format only:
{{"signal": "BUY" or "SELL" or "HOLD", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

            payload = {
                "model": self.model,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            }

            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            content = result['content'][0]['text']

            decision = json.loads(content.strip().replace('```json', '').replace('```', ''))
            decision['provider'] = self.get_name()
            return decision

        except Exception as e:
            logging.error(f"Claude API error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': f'API error: {str(e)}', 'provider': self.get_name(), 'failed': True}


class AICouncil:
    """
    Multi-AI Ensemble Decision System.

    Leverages DeepSeek, OpenAI, and Claude to make consensus-based trading decisions.
    Each AI analyzes the market independently, then votes on the final decision.
    """

    def __init__(self):
        self.providers = []
        self.is_active = False
        self.decision_history = []
        self.consensus_threshold = 0.6  # 60% agreement needed

        # Initialize providers
        self._init_providers()

    def _init_providers(self):
        """Initialize AI providers"""
        self.providers = [
            DeepSeekProvider(),
            OpenAIProvider(),
            ClaudeProvider()
        ]

    def configure_api_keys(self, deepseek_key: str = None, openai_key: str = None, claude_key: str = None):
        """Configure API keys for providers"""
        for provider in self.providers:
            if isinstance(provider, DeepSeekProvider) and deepseek_key:
                provider.api_key = deepseek_key
            elif isinstance(provider, OpenAIProvider) and openai_key:
                provider.api_key = openai_key
            elif isinstance(provider, ClaudeProvider) and claude_key:
                provider.api_key = claude_key

    def activate(self) -> str:
        """Activate AI Council"""
        active_providers = [p.get_name() for p in self.providers if self._has_api_key(p)]
        if not active_providers:
            return "AI Council: No API keys configured. Set environment variables or configure keys."
        self.is_active = True
        return f"AI Council Online. Active providers: {', '.join(active_providers)}"

    def deactivate(self) -> str:
        """Deactivate AI Council"""
        self.is_active = False
        return "AI Council Offline."

    def _has_api_key(self, provider: AIProvider) -> bool:
        """Check if provider has API key configured"""
        if isinstance(provider, DeepSeekProvider):
            return bool(provider.api_key)
        elif isinstance(provider, OpenAIProvider):
            return bool(provider.api_key)
        elif isinstance(provider, ClaudeProvider):
            return bool(provider.api_key)
        return False

    def _build_market_context(self, df: pd.DataFrame, omega_analysis: Dict = None) -> str:
        """Build market context string for AI analysis"""
        if df.empty or len(df) < 20:
            return "Insufficient data"

        close = df['close']
        current_price = close.iloc[-1]
        price_change_1h = ((current_price - close.iloc[-60]) / close.iloc[-60] * 100) if len(close) >= 60 else 0
        price_change_24h = ((current_price - close.iloc[-1440]) / close.iloc[-1440] * 100) if len(close) >= 1440 else price_change_1h

        # Calculate indicators
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
        rsi = TechnicalIndicators.calculate_rsi(close).iloc[-1]
        volatility = close.pct_change().rolling(20).std().iloc[-1] * 100

        context = f"""MARKET DATA:
- Current Price: ${current_price:.2f}
- Price Change (1h): {price_change_1h:.2f}%
- SMA 20: ${sma_20:.2f}
- SMA 50: ${sma_50:.2f}
- RSI (14): {rsi:.1f}
- Volatility: {volatility:.2f}%
- Price vs SMA20: {'ABOVE' if current_price > sma_20 else 'BELOW'}
- Trend: {'BULLISH' if sma_20 > sma_50 else 'BEARISH' if sma_20 < sma_50 else 'NEUTRAL'}"""

        if omega_analysis and omega_analysis.get('synthesis'):
            synthesis = omega_analysis['synthesis']
            context += f"""

OMEGA MODE ANALYSIS:
- Regime: {synthesis.get('regime', 'N/A')}
- Opportunity: {synthesis.get('opportunity', 'NONE')}
- Raw Signal: {synthesis.get('raw_signal', 0):.3f}
- Meta-Insight: {synthesis.get('meta_insight', 'N/A')}"""

        return context

    def get_consensus_decision(self, df: pd.DataFrame, omega_analysis: Dict = None) -> Dict:
        """
        Get consensus decision from all AI providers.

        Queries all configured AIs in parallel, then synthesizes their responses
        into a final consensus decision.
        """
        if not self.is_active:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': 'AI Council not active'}

        market_context = self._build_market_context(df, omega_analysis)

        if market_context == "Insufficient data":
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': 'Insufficient market data'}

        # Query all providers in parallel
        decisions = []
        active_providers = [p for p in self.providers if self._has_api_key(p)]

        if not active_providers:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': 'No AI providers configured'}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(p.get_trade_decision, market_context): p for p in active_providers}

            for future in as_completed(futures, timeout=60):
                try:
                    decision = future.result()
                    decisions.append(decision)
                except Exception as e:
                    logging.error(f"Provider error: {e}")

        if not decisions:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': 'No responses from AI providers'}

        # Calculate consensus
        consensus = self._calculate_consensus(decisions)

        # Log decision
        self.decision_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'individual_decisions': decisions,
            'consensus': consensus
        })

        return consensus

    def _calculate_consensus(self, decisions: List[Dict]) -> Dict:
        """Calculate consensus from individual AI decisions"""
        if not decisions:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': 'No decisions to analyze'}

        # Filter out failed API responses - don't count errors as votes
        valid_decisions = [d for d in decisions if not d.get('failed', False)]

        if not valid_decisions:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': 'All AI providers failed'}

        # Count votes only from successful responses
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidences = {'BUY': [], 'SELL': [], 'HOLD': []}
        reasonings = []

        for decision in valid_decisions:
            signal = decision.get('signal', 'HOLD').upper()
            confidence = float(decision.get('confidence', 0.5))
            provider = decision.get('provider', 'Unknown')
            reasoning = decision.get('reasoning', '')

            if signal in votes:
                votes[signal] += 1
                confidences[signal].append(confidence)
                reasonings.append(f"[{provider}] {signal}: {reasoning}")

        # Determine winner
        total_votes = sum(votes.values())
        winner = max(votes, key=votes.get)
        vote_pct = votes[winner] / total_votes if total_votes > 0 else 0

        # Calculate average confidence for winning signal
        avg_confidence = np.mean(confidences[winner]) if confidences[winner] else 0.5

        # Adjust confidence based on consensus strength
        consensus_confidence = avg_confidence * vote_pct

        # Build reasoning summary
        reasoning_summary = f"Council Vote: {votes['BUY']}B/{votes['SELL']}S/{votes['HOLD']}H | " + " | ".join(reasonings[:3])

        return {
            'signal': winner if vote_pct >= self.consensus_threshold else 'HOLD',
            'confidence': consensus_confidence,
            'reasoning': reasoning_summary,
            'votes': votes,
            'individual_decisions': decisions,
            'consensus_strength': vote_pct
        }

    def get_active_providers(self) -> List[str]:
        """Get list of active provider names"""
        return [p.get_name() for p in self.providers if self._has_api_key(p)]


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
        features = features.bfill().fillna(0)

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


class LiveMarketData:
    """Fetches live BTC price data from CoinGecko API"""

    def __init__(self, symbol: str = "BTCUSD"):
        self.symbol = symbol
        self.coin_id = "bitcoin"
        self.base_url = "https://api.coingecko.com/api/v3"
        self.history = []
        self.last_price = None
        self.last_volume = 0
        self.cached_history = None
        self.cache_time = None

    def fetch_current_price(self) -> Dict:
        """Fetch current BTC price from CoinGecko"""
        try:
            # Get current price with market data
            response = requests.get(
                f"{self.base_url}/simple/price",
                params={
                    "ids": self.coin_id,
                    "vs_currencies": "usd",
                    "include_24hr_vol": "true",
                    "include_24hr_change": "true"
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            current_price = float(data['bitcoin']['usd'])
            volume_24h = float(data['bitcoin'].get('usd_24hr_vol', 0))

            # Calculate OHLC from price movement
            price_change = current_price * 0.001  # Estimate small range
            tick = {
                'timestamp': datetime.datetime.now(),
                'open': current_price - price_change,
                'high': current_price + price_change,
                'low': current_price - price_change,
                'close': current_price,
                'volume': volume_24h / 1440  # Approximate per-minute volume
            }

            self.last_price = current_price
            self.last_volume = volume_24h
            self.history.append(tick)

            # Keep last 1000 ticks
            if len(self.history) > 1000:
                self.history.pop(0)

            return tick

        except Exception as e:
            logging.error(f"Error fetching live price: {e}")
            if self.last_price:
                return {
                    'timestamp': datetime.datetime.now(),
                    'open': self.last_price,
                    'high': self.last_price,
                    'low': self.last_price,
                    'close': self.last_price,
                    'volume': 0
                }
            return None

    def fetch_historical(self, days: int = 1) -> pd.DataFrame:
        """Fetch historical price data from CoinGecko"""
        try:
            # Cache for 1 minute to avoid rate limits
            now = datetime.datetime.now()
            if self.cached_history is not None and self.cache_time:
                if (now - self.cache_time).seconds < 60:
                    return self.cached_history

            response = requests.get(
                f"{self.base_url}/coins/{self.coin_id}/market_chart",
                params={
                    "vs_currency": "usd",
                    "days": days,
                    "interval": "daily" if days > 1 else ""
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])

            if not prices:
                return pd.DataFrame()

            # Build DataFrame
            df_data = []
            for i, (ts, price) in enumerate(prices):
                vol = volumes[i][1] if i < len(volumes) else 0
                timestamp = datetime.datetime.fromtimestamp(ts / 1000)

                # Estimate OHLC from price points
                prev_price = prices[i-1][1] if i > 0 else price
                high = max(price, prev_price)
                low = min(price, prev_price)

                df_data.append({
                    'timestamp': timestamp,
                    'open': prev_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': vol / 1440 if days <= 1 else vol  # Per-minute for intraday
                })

            df = pd.DataFrame(df_data)
            self.last_price = df['close'].iloc[-1] if len(df) > 0 else self.last_price

            # Cache result
            self.cached_history = df
            self.cache_time = now

            return df

        except Exception as e:
            logging.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def generate_tick(self) -> Dict:
        """Generate tick from live data (compatible with simulator interface)"""
        return self.fetch_current_price()

    def get_historical_data(self, periods: int = 200) -> pd.DataFrame:
        """Get historical data (compatible with simulator interface)"""
        df = self.fetch_historical(days=1)
        if df.empty and self.history:
            return pd.DataFrame(self.history[-periods:])
        return df


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
        self.omega_mode = OmegaMode()
        self.ai_council = AICouncil()
        self.use_omega_mode = False
        self.use_ai_council = False
        self.market_simulator = MarketDataSimulator()
        self.live_market_data = LiveMarketData(symbol="BTCUSD")
        self.use_live_data = False  # Toggle for live BTC prices

        # Data management - CSV loading and session logging
        self.data_manager = DataManager()

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
            'open_positions': len(self.positions),
            'omega_mode_active': self.omega_mode.is_active,
            'regime': self.omega_mode.regime_state if self.omega_mode.is_active else 'N/A'
        }

    def toggle_omega_mode(self, enable: bool = True) -> str:
        """Toggle Omega Mode on/off"""
        self.use_omega_mode = enable
        if enable:
            return self.omega_mode.activate()
        else:
            return self.omega_mode.deactivate()

    def toggle_ai_council(self, enable: bool = True) -> str:
        """Toggle AI Council on/off"""
        self.use_ai_council = enable
        if enable:
            return self.ai_council.activate()
        else:
            return self.ai_council.deactivate()

    def configure_ai_council(self, deepseek_key: str = None, openai_key: str = None, claude_key: str = None):
        """Configure AI Council API keys"""
        self.ai_council.configure_api_keys(deepseek_key, openai_key, claude_key)

    def toggle_live_data(self, enable: bool = True) -> str:
        """Toggle between live BTC prices and simulated data"""
        self.use_live_data = enable
        if enable:
            # Test connection to Binance
            tick = self.live_market_data.fetch_current_price()
            if tick:
                return f"Live Mode ON: BTC @ ${tick['close']:,.2f}"
            else:
                self.use_live_data = False
                return "Live Mode FAILED: Could not connect to Binance API"
        else:
            return "Live Mode OFF: Using simulated data"

    def get_market_data_source(self):
        """Get the active market data source"""
        return self.live_market_data if self.use_live_data else self.market_simulator

    def get_signal(self, df: pd.DataFrame) -> Dict:
        """Get trading signal from active strategy"""
        # Priority: AI Council > Omega Mode > ML Strategy
        if self.use_ai_council and self.ai_council.is_active:
            # Get Omega analysis to feed to AI Council
            omega_analysis = self.omega_mode.analyze_three_layers(df) if self.use_omega_mode else None
            council_decision = self.ai_council.get_consensus_decision(df, omega_analysis)
            return {
                'signal': council_decision.get('signal', 'HOLD'),
                'confidence': council_decision.get('confidence', 0.0),
                'reason': council_decision.get('reasoning', ''),
                'regime': omega_analysis.get('synthesis', {}).get('regime', 'N/A') if omega_analysis and omega_analysis.get('synthesis') else 'N/A',
                'votes': council_decision.get('votes', {}),
                'council_active': True
            }
        elif self.use_omega_mode and self.omega_mode.is_active:
            return self.omega_mode.generate_signal(df)
        else:
            return self.ai_engine.predict(df)


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

        # Omega Mode Toggle
        self.omega_var = tk.BooleanVar(value=False)
        self.omega_check = ttk.Checkbutton(
            control_frame,
            text="OMEGA MODE",
            variable=self.omega_var,
            command=self.toggle_omega_mode,
            style='Omega.TCheckbutton'
        )
        self.omega_check.grid(row=0, column=5, padx=10)

        # Omega Mode Status
        self.omega_status_label = ttk.Label(control_frame, text="Omega: Offline", foreground="gray")
        self.omega_status_label.grid(row=0, column=6, padx=5)

        # Regime indicator
        self.regime_label = ttk.Label(control_frame, text="Regime: --", foreground="gray")
        self.regime_label.grid(row=0, column=7, padx=5)

        # AI Council Toggle (row 1)
        self.council_var = tk.BooleanVar(value=False)
        self.council_check = ttk.Checkbutton(
            control_frame,
            text="AI COUNCIL",
            variable=self.council_var,
            command=self.toggle_ai_council
        )
        self.council_check.grid(row=1, column=0, padx=5, pady=5, columnspan=2)

        # AI Council Status
        self.council_status_label = ttk.Label(control_frame, text="Council: Offline", foreground="gray")
        self.council_status_label.grid(row=1, column=2, padx=5, columnspan=2)

        # Active Providers Label
        self.providers_label = ttk.Label(control_frame, text="Providers: None", foreground="gray")
        self.providers_label.grid(row=1, column=4, padx=5, columnspan=2)

        # Live BTC Mode Toggle
        self.live_var = tk.BooleanVar(value=False)
        self.live_check = ttk.Checkbutton(
            control_frame,
            text="LIVE BTC",
            variable=self.live_var,
            command=self.toggle_live_mode
        )
        self.live_check.grid(row=1, column=6, padx=5)

        # Live Mode Status
        self.live_status_label = ttk.Label(control_frame, text="Mode: Simulated", foreground="gray")
        self.live_status_label.grid(row=1, column=7, padx=5)

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

        # Update regime indicator for Omega Mode
        if stats.get('omega_mode_active', False):
            regime = stats.get('regime', 'N/A')
            regime_colors = {
                'BULL': 'green',
                'BEAR': 'red',
                'VOLATILE': 'orange',
                'NEUTRAL': 'blue'
            }
            self.regime_label.config(
                text=f"Regime: {regime}",
                foreground=regime_colors.get(regime, 'gray')
            )

    def update_chart(self):
        """Update price chart with live BTC data"""
        if not PLOTTING_AVAILABLE:
            return

        df = self.engine.get_market_data_source().get_historical_data(periods=100)

        if df.empty:
            return

        self.ax1.clear()
        self.ax2.clear()

        # Use timestamp for X-axis if available, otherwise use index
        if 'timestamp' in df.columns:
            x_data = df['timestamp']
            # Format x-axis for timestamps
            self.ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
            self.ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        else:
            x_data = df.index

        # Price chart - candlestick style for live data
        if self.engine.use_live_data and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Plot as line with high/low range
            self.ax1.fill_between(x_data, df['low'], df['high'], alpha=0.2, color='blue', label='H/L Range')
            self.ax1.plot(x_data, df['close'], label='Close', color='blue', linewidth=1.5)
            self.ax1.plot(x_data, df['open'], label='Open', color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
        else:
            self.ax1.plot(x_data, df['close'], label='Price', color='blue', linewidth=1.5)

        # Add moving averages
        if len(df) >= 20:
            sma20 = TechnicalIndicators.calculate_sma(df['close'], 20)
            self.ax1.plot(x_data, sma20, label='SMA 20', color='orange', linewidth=1, alpha=0.7)

        if len(df) >= 50:
            sma50 = TechnicalIndicators.calculate_sma(df['close'], 50)
            self.ax1.plot(x_data, sma50, label='SMA 50', color='red', linewidth=1, alpha=0.7)

        # Mark trades
        for trade in self.engine.trades[-20:]:  # Last 20 trades
            try:
                idx = df[df['timestamp'] >= trade.timestamp].index[0]
                trade_x = x_data.iloc[idx] if 'timestamp' in df.columns else idx
                color = 'green' if trade.side == 'BUY' else 'red'
                marker = '^' if trade.side == 'BUY' else 'v'
                self.ax1.scatter(trade_x, trade.price, color=color, marker=marker, s=100, zorder=5)
            except (IndexError, KeyError):
                pass

        # Current price annotation for live mode
        if self.engine.use_live_data and len(df) > 0:
            current_price = df['close'].iloc[-1]
            self.ax1.axhline(y=current_price, color='green', linestyle='-', linewidth=0.8, alpha=0.5)
            self.ax1.annotate(f'${current_price:,.2f}', xy=(x_data.iloc[-1], current_price),
                            xytext=(5, 0), textcoords='offset points', fontsize=9,
                            color='green', fontweight='bold')

        self.ax1.set_ylabel('Price (USD)')
        self.ax1.legend(loc='upper left', fontsize=8)
        self.ax1.grid(True, alpha=0.3)

        # Title with live indicator
        mode_str = "LIVE" if self.engine.use_live_data else "SIM"
        current_price_str = f" | ${df['close'].iloc[-1]:,.2f}" if len(df) > 0 else ""
        self.ax1.set_title(f'{self.symbol} [{mode_str}]{current_price_str}', fontsize=10, fontweight='bold')

        # Format Y-axis for large BTC prices
        self.ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Volume chart with colors based on price movement
        if 'volume' in df.columns:
            colors = []
            for i in range(len(df)):
                if i == 0:
                    colors.append('gray')
                elif df['close'].iloc[i] >= df['close'].iloc[i-1]:
                    colors.append('green')
                else:
                    colors.append('red')

            self.ax2.bar(x_data, df['volume'], color=colors, alpha=0.6)
            self.ax2.set_ylabel('Volume')
            self.ax2.set_xlabel('Time')
            self.ax2.grid(True, alpha=0.3)

            # Format volume for readability
            self.ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(
                lambda x, p: f'{x/1000:.0f}K' if x < 1000000 else f'{x/1000000:.1f}M'))

        self.fig.tight_layout()
        self.canvas.draw()

    def toggle_omega_mode(self):
        """Toggle Omega Mode on/off"""
        is_enabled = self.omega_var.get()
        result = self.engine.toggle_omega_mode(is_enabled)
        self.log_message(result)

        if is_enabled:
            self.omega_status_label.config(text="Omega: ONLINE", foreground="purple")
        else:
            self.omega_status_label.config(text="Omega: Offline", foreground="gray")
            self.regime_label.config(text="Regime: --", foreground="gray")

    def toggle_ai_council(self):
        """Toggle AI Council on/off"""
        is_enabled = self.council_var.get()
        result = self.engine.toggle_ai_council(is_enabled)
        self.log_message(result)

        if is_enabled:
            self.council_status_label.config(text="Council: ONLINE", foreground="blue")
            providers = self.engine.ai_council.get_active_providers()
            if providers:
                self.providers_label.config(text=f"Providers: {', '.join(providers)}", foreground="green")
            else:
                self.providers_label.config(text="Providers: None (set API keys)", foreground="orange")
        else:
            self.council_status_label.config(text="Council: Offline", foreground="gray")
            self.providers_label.config(text="Providers: None", foreground="gray")

    def toggle_live_mode(self):
        """Toggle between live BTC prices and simulated data"""
        is_enabled = self.live_var.get()
        result = self.engine.toggle_live_data(is_enabled)
        self.log_message(result)

        if self.engine.use_live_data:
            self.live_status_label.config(text="Mode: LIVE BTC", foreground="green")
            self.symbol = "BTCUSD"  # Binance US uses USD pairs
        else:
            self.live_status_label.config(text="Mode: Simulated", foreground="gray")
            if "FAILED" in result:
                self.live_var.set(False)  # Reset checkbox if failed

    def train_ai(self):
        """Train AI model using CSV data + simulated data"""
        self.log_message("Training AI model...")
        self.train_button.config(state='disabled')

        def train_thread():
            # Try to load historical data from CSV files first
            csv_data = self.engine.data_manager.load_historical_data()

            if not csv_data.empty:
                self.update_queue.put(('log', f'Loaded {len(csv_data)} rows from CSV files'))
                historical_df = csv_data
            else:
                self.update_queue.put(('log', 'No CSV data found, using simulated data...'))
                # Generate simulated historical data as fallback
                historical_df = self.engine.market_simulator.get_historical_data(periods=500)

            # If we have both CSV and want more data, combine them
            if not csv_data.empty and len(csv_data) < 500:
                sim_data = self.engine.market_simulator.get_historical_data(periods=500 - len(csv_data))
                historical_df = pd.concat([csv_data, sim_data], ignore_index=True)
                self.update_queue.put(('log', f'Combined data: {len(historical_df)} total rows'))

            # Train model
            success = self.engine.ai_engine.train(historical_df)

            if success:
                self.update_queue.put(('ai_trained', True))
                self.update_queue.put(('log', f'AI model trained successfully on {len(historical_df)} data points!'))
            else:
                self.update_queue.put(('ai_trained', False))
                self.update_queue.put(('log', 'Failed to train AI model.'))

        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()

    def start_trading(self):
        """Start trading bot"""
        # Allow trading if AI Council, Omega Mode is active, OR AI is trained
        if not self.engine.use_ai_council and not self.engine.use_omega_mode and not self.engine.ai_engine.is_trained:
            messagebox.showwarning("Warning", "Please train the AI model first, enable Omega Mode, or enable AI Council!")
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
        """Stop trading bot and save session log"""
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Status: Stopped", foreground="red")

        self.log_message("Trading bot stopped!")

        # Save session log to CSV for future training
        self.log_message("Saving session data to CSV...")
        saved_files = self.engine.data_manager.save_session_log()

        if saved_files:
            summary = self.engine.data_manager.get_session_summary()
            self.log_message(f"Session {summary['session_id']} saved:")
            self.log_message(f"  - {summary['total_trades']} trades")
            self.log_message(f"  - {summary['total_price_points']} price points")
            for file_type, path in saved_files.items():
                self.log_message(f"  - {file_type}: {os.path.basename(path)}")
        else:
            self.log_message("No data to save for this session.")

    def trading_loop(self):
        """Main trading loop with session logging"""
        while self.is_running:
            try:
                # Get market data source (live or simulated)
                market_source = self.engine.get_market_data_source()

                # Generate new market tick
                tick = market_source.generate_tick()
                if tick is None:
                    time.sleep(1)
                    continue
                current_price = tick['close']

                # Log price data for future training
                self.engine.data_manager.log_price(tick)

                # Update positions
                if self.symbol in self.engine.positions:
                    self.engine.update_positions(self.symbol, current_price)

                # Get historical data for strategy
                historical_df = market_source.get_historical_data(periods=200)

                # Get signal from active strategy (Omega Mode or AI)
                prediction = self.engine.get_signal(historical_df)
                signal = prediction.get('signal', 'HOLD')
                confidence = prediction.get('confidence', 0.0)
                reason = prediction.get('reason', '')

                # Log signal for session export
                self.engine.data_manager.log_signal(prediction, current_price)

                # Trading logic
                if self.engine.use_ai_council:
                    mode_tag = "[COUNCIL]"
                elif self.engine.use_omega_mode:
                    mode_tag = "[OMEGA]"
                else:
                    mode_tag = "[AI]"

                if signal == 'BUY' and confidence > 0.6 and self.symbol not in self.engine.positions:
                    quantity = self.engine.calculate_position_size(current_price)
                    if self.engine.open_position(self.symbol, 'BUY', current_price, quantity):
                        msg = f"{mode_tag} BUY {self.symbol} @ ${current_price:.2f} (Conf: {confidence:.2f})"
                        # Show votes for AI Council
                        if self.engine.use_ai_council and prediction.get('votes'):
                            votes = prediction['votes']
                            msg += f" [Votes: {votes.get('BUY', 0)}B/{votes.get('SELL', 0)}S/{votes.get('HOLD', 0)}H]"
                        if reason and (self.engine.use_omega_mode or self.engine.use_ai_council):
                            msg += f"\n  -> {reason}"
                        self.update_queue.put(('log', msg))

                        # Log trade for session export
                        strategy = 'COUNCIL' if self.engine.use_ai_council else ('OMEGA' if self.engine.use_omega_mode else 'AI')
                        self.engine.data_manager.log_trade({
                            'symbol': self.symbol,
                            'side': 'BUY',
                            'price': current_price,
                            'quantity': quantity,
                            'strategy': strategy,
                            'confidence': confidence,
                            'regime': prediction.get('regime', 'N/A'),
                            'reason': reason
                        })

                        # Self-optimization feedback for Omega Mode
                        if self.engine.use_omega_mode:
                            self.engine.omega_mode.self_optimize({
                                'action': 'BUY',
                                'price': current_price,
                                'confidence': confidence
                            })

                elif signal == 'SELL' and self.symbol in self.engine.positions:
                    position = self.engine.positions.get(self.symbol)
                    profit = 0
                    if position:
                        profit = (current_price - position.entry_price) * abs(position.quantity)

                    if self.engine.close_position(self.symbol, current_price):
                        msg = f"{mode_tag} SELL {self.symbol} @ ${current_price:.2f} (Conf: {confidence:.2f})"
                        # Show votes for AI Council
                        if self.engine.use_ai_council and prediction.get('votes'):
                            votes = prediction['votes']
                            msg += f" [Votes: {votes.get('BUY', 0)}B/{votes.get('SELL', 0)}S/{votes.get('HOLD', 0)}H]"
                        if reason and (self.engine.use_omega_mode or self.engine.use_ai_council):
                            msg += f"\n  -> {reason}"
                        self.update_queue.put(('log', msg))

                        # Log trade for session export
                        strategy = 'COUNCIL' if self.engine.use_ai_council else ('OMEGA' if self.engine.use_omega_mode else 'AI')
                        self.engine.data_manager.log_trade({
                            'symbol': self.symbol,
                            'side': 'SELL',
                            'price': current_price,
                            'quantity': abs(position.quantity) if position else 0,
                            'profit': profit,
                            'strategy': strategy,
                            'confidence': confidence,
                            'regime': prediction.get('regime', 'N/A'),
                            'reason': reason
                        })

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
