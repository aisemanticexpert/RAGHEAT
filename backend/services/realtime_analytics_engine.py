"""
Real-Time Analytics Engine
Advanced algorithms for processing streaming market data
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

@dataclass
class MarketSignal:
    signal_type: str
    symbol: str
    timestamp: str
    strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str
    metadata: Dict[str, Any]

@dataclass
class HeatAnalysis:
    symbol: str
    current_heat: float
    heat_trend: str  # 'rising', 'falling', 'stable'
    heat_velocity: float
    propagation_strength: float
    influenced_symbols: List[str]
    analysis_timestamp: str

class RealTimeAnalyticsEngine:
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        
        # Data buffers for each symbol
        self.price_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_window))
        self.volume_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_window))
        self.heat_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_window))
        self.volatility_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_window))
        
        # Market-wide buffers
        self.market_regime_buffer: deque = deque(maxlen=50)
        self.correlation_matrix_buffer: deque = deque(maxlen=20)
        
        # Signal tracking
        self.active_signals: List[MarketSignal] = []
        self.signal_history: deque = deque(maxlen=500)
        
        # Analytics state
        self.last_analysis_time = datetime.now()
        self.analysis_intervals = {
            'fast': 1.0,  # 1 second
            'medium': 5.0,  # 5 seconds
            'slow': 30.0   # 30 seconds
        }
        
        # Heat propagation network
        self.heat_network: Dict[str, Dict[str, float]] = {}
        self.heat_analysis_cache: Dict[str, HeatAnalysis] = {}
        
        # Technical indicators cache
        self.indicators_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Anomaly detection parameters
        self.anomaly_thresholds = {
            'price_zscore': 3.0,
            'volume_zscore': 2.5,
            'heat_zscore': 2.0,
            'correlation_break': 0.3
        }
        
    def update_data(self, symbol: str, price: float, volume: int, heat_score: float, volatility: float):
        """Update data buffers with new market data"""
        timestamp = datetime.now()
        
        self.price_buffers[symbol].append((timestamp, price))
        self.volume_buffers[symbol].append((timestamp, volume))
        self.heat_buffers[symbol].append((timestamp, heat_score))
        self.volatility_buffers[symbol].append((timestamp, volatility))
        
        # Update indicators
        self._update_technical_indicators(symbol)
        
        # Trigger analysis if needed
        if self._should_analyze('fast'):
            asyncio.create_task(self.fast_analysis())
    
    def _update_technical_indicators(self, symbol: str):
        """Update technical indicators for a symbol"""
        if len(self.price_buffers[symbol]) < 20:
            return
        
        prices = [p[1] for p in self.price_buffers[symbol]]
        volumes = [v[1] for v in self.volume_buffers[symbol]]
        
        # Moving averages
        self.indicators_cache[symbol]['sma_20'] = np.mean(prices[-20:])
        self.indicators_cache[symbol]['sma_50'] = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
        
        # Exponential moving average
        self.indicators_cache[symbol]['ema_12'] = self._calculate_ema(prices, 12)
        
        # RSI
        self.indicators_cache[symbol]['rsi'] = self._calculate_rsi(prices)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)
        self.indicators_cache[symbol]['bb_upper'] = bb_upper
        self.indicators_cache[symbol]['bb_middle'] = bb_middle
        self.indicators_cache[symbol]['bb_lower'] = bb_lower
        
        # Volume indicators
        self.indicators_cache[symbol]['volume_sma'] = np.mean(volumes[-20:])
        self.indicators_cache[symbol]['volume_ratio'] = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
        
        # Price momentum
        if len(prices) >= 10:
            self.indicators_cache[symbol]['momentum_10'] = (prices[-1] / prices[-10] - 1) * 100
        
        # Volatility indicators
        if len(prices) >= 20:
            returns = np.diff(prices[-20:]) / prices[-20:-1]
            self.indicators_cache[symbol]['historical_volatility'] = np.std(returns) * np.sqrt(252) * 100
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            mean_price = np.mean(prices)
            return mean_price, mean_price, mean_price
        
        recent_prices = prices[-period:]
        middle_band = np.mean(recent_prices)
        std_deviation = np.std(recent_prices)
        
        upper_band = middle_band + (std_deviation * std_dev)
        lower_band = middle_band - (std_deviation * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def _should_analyze(self, interval_type: str) -> bool:
        """Check if enough time has passed for analysis"""
        current_time = datetime.now()
        time_diff = (current_time - self.last_analysis_time).total_seconds()
        
        return time_diff >= self.analysis_intervals[interval_type]
    
    async def fast_analysis(self):
        """Fast analysis - run every second"""
        try:
            current_signals = []
            
            for symbol in self.price_buffers.keys():
                # Anomaly detection
                anomaly_signals = self._detect_anomalies(symbol)
                current_signals.extend(anomaly_signals)
                
                # Momentum signals
                momentum_signals = self._analyze_momentum(symbol)
                current_signals.extend(momentum_signals)
                
                # Volume analysis
                volume_signals = self._analyze_volume(symbol)
                current_signals.extend(volume_signals)
            
            # Update active signals
            self._update_signals(current_signals)
            
        except Exception as e:
            logging.error(f"Error in fast analysis: {e}")
    
    async def medium_analysis(self):
        """Medium analysis - run every 5 seconds"""
        try:
            current_signals = []
            
            for symbol in self.price_buffers.keys():
                # Technical indicator signals
                tech_signals = self._analyze_technical_indicators(symbol)
                current_signals.extend(tech_signals)
                
                # Heat propagation analysis
                heat_analysis = self._analyze_heat_propagation(symbol)
                if heat_analysis:
                    self.heat_analysis_cache[symbol] = heat_analysis
            
            # Market regime analysis
            regime_signals = self._analyze_market_regime()
            current_signals.extend(regime_signals)
            
            self._update_signals(current_signals)
            
        except Exception as e:
            logging.error(f"Error in medium analysis: {e}")
    
    async def slow_analysis(self):
        """Slow analysis - run every 30 seconds"""
        try:
            current_signals = []
            
            # Correlation analysis
            correlation_signals = self._analyze_correlations()
            current_signals.extend(correlation_signals)
            
            # Pattern recognition
            pattern_signals = self._detect_patterns()
            current_signals.extend(pattern_signals)
            
            # Market structure analysis
            structure_signals = self._analyze_market_structure()
            current_signals.extend(structure_signals)
            
            self._update_signals(current_signals)
            
        except Exception as e:
            logging.error(f"Error in slow analysis: {e}")
    
    def _detect_anomalies(self, symbol: str) -> List[MarketSignal]:
        """Detect price, volume, and heat anomalies"""
        signals = []
        
        if len(self.price_buffers[symbol]) < 20:
            return signals
        
        # Price anomaly detection
        prices = [p[1] for p in self.price_buffers[symbol]]
        recent_prices = prices[-20:]
        current_price = prices[-1]
        
        price_mean = np.mean(recent_prices[:-1])
        price_std = np.std(recent_prices[:-1])
        
        if price_std > 0:
            price_zscore = (current_price - price_mean) / price_std
            
            if abs(price_zscore) > self.anomaly_thresholds['price_zscore']:
                signals.append(MarketSignal(
                    signal_type="price_anomaly",
                    symbol=symbol,
                    timestamp=datetime.now().isoformat(),
                    strength=np.tanh(price_zscore / 3.0),
                    confidence=min(1.0, abs(price_zscore) / 5.0),
                    description=f"Price anomaly detected: {price_zscore:.2f}σ deviation",
                    metadata={
                        "zscore": price_zscore,
                        "current_price": current_price,
                        "mean_price": price_mean
                    }
                ))
        
        # Volume anomaly detection
        if len(self.volume_buffers[symbol]) >= 20:
            volumes = [v[1] for v in self.volume_buffers[symbol]]
            recent_volumes = volumes[-20:]
            current_volume = volumes[-1]
            
            volume_mean = np.mean(recent_volumes[:-1])
            volume_std = np.std(recent_volumes[:-1])
            
            if volume_std > 0:
                volume_zscore = (current_volume - volume_mean) / volume_std
                
                if abs(volume_zscore) > self.anomaly_thresholds['volume_zscore']:
                    signals.append(MarketSignal(
                        signal_type="volume_anomaly",
                        symbol=symbol,
                        timestamp=datetime.now().isoformat(),
                        strength=np.tanh(volume_zscore / 3.0),
                        confidence=min(1.0, abs(volume_zscore) / 4.0),
                        description=f"Volume anomaly detected: {volume_zscore:.2f}σ deviation",
                        metadata={
                            "zscore": volume_zscore,
                            "current_volume": current_volume,
                            "mean_volume": volume_mean
                        }
                    ))
        
        return signals
    
    def _analyze_momentum(self, symbol: str) -> List[MarketSignal]:
        """Analyze price momentum"""
        signals = []
        
        if symbol not in self.indicators_cache or 'momentum_10' not in self.indicators_cache[symbol]:
            return signals
        
        momentum = self.indicators_cache[symbol]['momentum_10']
        rsi = self.indicators_cache[symbol].get('rsi', 50.0)
        
        # Strong momentum signal
        if abs(momentum) > 5.0:  # >5% movement
            direction = 1.0 if momentum > 0 else -1.0
            strength = np.tanh(abs(momentum) / 10.0) * direction
            
            # Confirm with RSI
            rsi_confirmation = 0.5
            if (momentum > 0 and rsi > 60) or (momentum < 0 and rsi < 40):
                rsi_confirmation = 0.8
            
            signals.append(MarketSignal(
                signal_type="momentum",
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                strength=strength,
                confidence=min(1.0, abs(momentum) / 20.0 * rsi_confirmation),
                description=f"Strong momentum: {momentum:.2f}% (10-period)",
                metadata={
                    "momentum_10": momentum,
                    "rsi": rsi,
                    "direction": "bullish" if momentum > 0 else "bearish"
                }
            ))
        
        return signals
    
    def _analyze_volume(self, symbol: str) -> List[MarketSignal]:
        """Analyze volume patterns"""
        signals = []
        
        if symbol not in self.indicators_cache or 'volume_ratio' not in self.indicators_cache[symbol]:
            return signals
        
        volume_ratio = self.indicators_cache[symbol]['volume_ratio']
        
        # Unusual volume signal
        if volume_ratio > 2.0:  # Volume > 2x average
            signals.append(MarketSignal(
                signal_type="high_volume",
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                strength=np.tanh((volume_ratio - 1.0) / 2.0),
                confidence=min(1.0, (volume_ratio - 1.0) / 3.0),
                description=f"High volume: {volume_ratio:.2f}x average",
                metadata={
                    "volume_ratio": volume_ratio,
                    "interpretation": "increased_interest"
                }
            ))
        
        return signals
    
    def _analyze_technical_indicators(self, symbol: str) -> List[MarketSignal]:
        """Analyze technical indicators for signals"""
        signals = []
        
        if symbol not in self.indicators_cache:
            return signals
        
        indicators = self.indicators_cache[symbol]
        
        # RSI signals
        rsi = indicators.get('rsi', 50.0)
        if rsi > 80:
            signals.append(MarketSignal(
                signal_type="rsi_overbought",
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                strength=-0.8,
                confidence=(rsi - 70) / 30.0,
                description=f"RSI overbought: {rsi:.1f}",
                metadata={"rsi": rsi}
            ))
        elif rsi < 20:
            signals.append(MarketSignal(
                signal_type="rsi_oversold",
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                strength=0.8,
                confidence=(30 - rsi) / 30.0,
                description=f"RSI oversold: {rsi:.1f}",
                metadata={"rsi": rsi}
            ))
        
        # Bollinger Band signals
        if len(self.price_buffers[symbol]) > 0:
            current_price = self.price_buffers[symbol][-1][1]
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            
            if bb_upper and bb_lower:
                if current_price > bb_upper:
                    signals.append(MarketSignal(
                        signal_type="bollinger_breakout_upper",
                        symbol=symbol,
                        timestamp=datetime.now().isoformat(),
                        strength=0.6,
                        confidence=0.7,
                        description="Price above upper Bollinger Band",
                        metadata={
                            "price": current_price,
                            "bb_upper": bb_upper,
                            "bb_distance": (current_price - bb_upper) / bb_upper
                        }
                    ))
                elif current_price < bb_lower:
                    signals.append(MarketSignal(
                        signal_type="bollinger_breakout_lower",
                        symbol=symbol,
                        timestamp=datetime.now().isoformat(),
                        strength=-0.6,
                        confidence=0.7,
                        description="Price below lower Bollinger Band",
                        metadata={
                            "price": current_price,
                            "bb_lower": bb_lower,
                            "bb_distance": (bb_lower - current_price) / bb_lower
                        }
                    ))
        
        return signals
    
    def _analyze_heat_propagation(self, symbol: str) -> Optional[HeatAnalysis]:
        """Analyze heat propagation patterns"""
        if len(self.heat_buffers[symbol]) < 10:
            return None
        
        heat_values = [h[1] for h in self.heat_buffers[symbol]]
        current_heat = heat_values[-1]
        
        # Calculate heat trend
        if len(heat_values) >= 5:
            recent_heat = heat_values[-5:]
            heat_slope = np.polyfit(range(len(recent_heat)), recent_heat, 1)[0]
            
            if heat_slope > 1.0:
                trend = "rising"
            elif heat_slope < -1.0:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"
            heat_slope = 0.0
        
        # Calculate propagation strength
        propagation_strength = min(1.0, current_heat / 100.0)
        
        # Find influenced symbols (simplified)
        influenced_symbols = []
        for other_symbol in self.heat_buffers.keys():
            if other_symbol != symbol and len(self.heat_buffers[other_symbol]) > 0:
                other_heat = self.heat_buffers[other_symbol][-1][1]
                if other_heat > 60.0 and current_heat > 70.0:  # Both hot
                    influenced_symbols.append(other_symbol)
        
        return HeatAnalysis(
            symbol=symbol,
            current_heat=current_heat,
            heat_trend=trend,
            heat_velocity=heat_slope,
            propagation_strength=propagation_strength,
            influenced_symbols=influenced_symbols,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _analyze_market_regime(self) -> List[MarketSignal]:
        """Analyze overall market regime"""
        signals = []
        
        if len(self.market_regime_buffer) < 5:
            return signals
        
        # Check for regime changes
        recent_regimes = list(self.market_regime_buffer)[-5:]
        current_regime = recent_regimes[-1] if recent_regimes else None
        
        if current_regime and len(set(recent_regimes)) > 1:
            # Regime instability detected
            signals.append(MarketSignal(
                signal_type="regime_instability",
                symbol="MARKET",
                timestamp=datetime.now().isoformat(),
                strength=0.5,
                confidence=0.8,
                description=f"Market regime instability detected",
                metadata={
                    "current_regime": current_regime,
                    "recent_regimes": recent_regimes,
                    "regime_changes": len(set(recent_regimes))
                }
            ))
        
        return signals
    
    def _analyze_correlations(self) -> List[MarketSignal]:
        """Analyze correlation breakdowns"""
        signals = []
        
        # This is a simplified correlation analysis
        # In production, you'd maintain a full correlation matrix
        
        symbols = list(self.price_buffers.keys())
        if len(symbols) < 2:
            return signals
        
        # Check for correlation breakdowns between major pairs
        for i, symbol1 in enumerate(symbols[:5]):  # Check first 5 symbols
            for symbol2 in symbols[i+1:6]:
                if len(self.price_buffers[symbol1]) >= 20 and len(self.price_buffers[symbol2]) >= 20:
                    
                    prices1 = [p[1] for p in self.price_buffers[symbol1][-20:]]
                    prices2 = [p[1] for p in self.price_buffers[symbol2][-20:]]
                    
                    returns1 = np.diff(prices1) / prices1[:-1]
                    returns2 = np.diff(prices2) / prices2[:-1]
                    
                    if len(returns1) > 10 and len(returns2) > 10:
                        correlation = np.corrcoef(returns1, returns2)[0, 1]
                        
                        # Check for correlation breakdown (assuming normally correlated)
                        if not np.isnan(correlation) and abs(correlation) < 0.1:
                            signals.append(MarketSignal(
                                signal_type="correlation_breakdown",
                                symbol=f"{symbol1}-{symbol2}",
                                timestamp=datetime.now().isoformat(),
                                strength=-0.6,
                                confidence=0.7,
                                description=f"Correlation breakdown between {symbol1} and {symbol2}",
                                metadata={
                                    "symbol1": symbol1,
                                    "symbol2": symbol2,
                                    "correlation": correlation
                                }
                            ))
        
        return signals
    
    def _detect_patterns(self) -> List[MarketSignal]:
        """Detect chart patterns"""
        signals = []
        
        # Simplified pattern detection
        for symbol in list(self.price_buffers.keys())[:5]:  # Check first 5 symbols
            if len(self.price_buffers[symbol]) < 50:
                continue
            
            prices = [p[1] for p in self.price_buffers[symbol][-50:]]
            
            # Simple breakout pattern
            recent_high = max(prices[-20:-5])  # High from 20-5 periods ago
            current_price = prices[-1]
            
            if current_price > recent_high * 1.02:  # 2% breakout
                signals.append(MarketSignal(
                    signal_type="breakout_pattern",
                    symbol=symbol,
                    timestamp=datetime.now().isoformat(),
                    strength=0.7,
                    confidence=0.6,
                    description=f"Breakout pattern detected in {symbol}",
                    metadata={
                        "current_price": current_price,
                        "resistance_level": recent_high,
                        "breakout_strength": (current_price / recent_high - 1) * 100
                    }
                ))
        
        return signals
    
    def _analyze_market_structure(self) -> List[MarketSignal]:
        """Analyze overall market structure"""
        signals = []
        
        # Simplified market structure analysis
        if len(self.price_buffers) >= 5:
            # Calculate market-wide momentum
            total_momentum = 0
            active_symbols = 0
            
            for symbol, buffer in list(self.price_buffers.items())[:10]:
                if len(buffer) >= 20:
                    prices = [p[1] for p in buffer]
                    momentum = (prices[-1] / prices[-20] - 1) * 100
                    total_momentum += momentum
                    active_symbols += 1
            
            if active_symbols > 0:
                avg_momentum = total_momentum / active_symbols
                
                if abs(avg_momentum) > 3.0:  # Strong market-wide movement
                    signals.append(MarketSignal(
                        signal_type="market_momentum",
                        symbol="MARKET",
                        timestamp=datetime.now().isoformat(),
                        strength=np.tanh(avg_momentum / 10.0),
                        confidence=min(1.0, abs(avg_momentum) / 15.0),
                        description=f"Market-wide momentum: {avg_momentum:.2f}%",
                        metadata={
                            "avg_momentum": avg_momentum,
                            "active_symbols": active_symbols,
                            "direction": "bullish" if avg_momentum > 0 else "bearish"
                        }
                    ))
        
        return signals
    
    def _update_signals(self, new_signals: List[MarketSignal]):
        """Update active signals list"""
        current_time = datetime.now()
        
        # Remove expired signals (older than 30 seconds)
        self.active_signals = [
            signal for signal in self.active_signals
            if (current_time - datetime.fromisoformat(signal.timestamp.replace('Z', '+00:00').replace('+00:00', ''))).total_seconds() < 30
        ]
        
        # Add new signals
        for signal in new_signals:
            self.active_signals.append(signal)
            self.signal_history.append(signal)
        
        # Update analysis time
        self.last_analysis_time = current_time
    
    def get_active_signals(self) -> List[Dict]:
        """Get currently active signals"""
        return [asdict(signal) for signal in self.active_signals]
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of all signals"""
        if not self.active_signals:
            return {
                "total_signals": 0,
                "signal_types": {},
                "average_strength": 0.0,
                "average_confidence": 0.0
            }
        
        signal_types = defaultdict(int)
        total_strength = 0.0
        total_confidence = 0.0
        
        for signal in self.active_signals:
            signal_types[signal.signal_type] += 1
            total_strength += abs(signal.strength)
            total_confidence += signal.confidence
        
        return {
            "total_signals": len(self.active_signals),
            "signal_types": dict(signal_types),
            "average_strength": total_strength / len(self.active_signals),
            "average_confidence": total_confidence / len(self.active_signals),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_heat_analysis(self) -> Dict[str, Dict]:
        """Get heat propagation analysis"""
        return {
            symbol: asdict(analysis) 
            for symbol, analysis in self.heat_analysis_cache.items()
        }
    
    def get_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """Get technical indicators for a symbol"""
        return self.indicators_cache.get(symbol, {})
    
    def get_analytics_status(self) -> Dict[str, Any]:
        """Get current analytics engine status"""
        return {
            "tracked_symbols": len(self.price_buffers),
            "data_points_per_symbol": {
                symbol: len(buffer) for symbol, buffer in list(self.price_buffers.items())[:5]
            },
            "active_signals": len(self.active_signals),
            "signal_history_size": len(self.signal_history),
            "heat_analysis_cache_size": len(self.heat_analysis_cache),
            "last_analysis_time": self.last_analysis_time.isoformat(),
            "analysis_intervals": self.analysis_intervals
        }