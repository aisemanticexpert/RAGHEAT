"""
Advanced Technical Indicators Module
Implements MACD, Bollinger Bands, and Heat Equation calculations for dynamic weight assignment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class VolatilityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class TechnicalSignal:
    """Technical analysis signal with confidence and strength"""
    signal_type: str
    direction: TrendDirection
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime

@dataclass
class MACDResult:
    """MACD calculation results"""
    macd_line: float
    signal_line: float
    histogram: float
    trend_direction: TrendDirection
    signal_strength: float
    crossover_signal: Optional[TechnicalSignal]

@dataclass
class BollingerBandsResult:
    """Bollinger Bands calculation results"""
    upper_band: float
    middle_band: float  # SMA
    lower_band: float
    bandwidth: float
    bb_percent: float  # %B indicator
    squeeze_signal: Optional[TechnicalSignal]
    breakout_signal: Optional[TechnicalSignal]

@dataclass
class HeatEquationResult:
    """Heat equation calculation with time decay"""
    current_heat: float
    heat_velocity: float  # Rate of heat change
    heat_acceleration: float
    decay_factor: float
    propagation_strength: float
    temporal_validity: float  # How valid this heat is over time

class TechnicalIndicatorEngine:
    """
    Advanced technical indicator engine with dynamic weight calculation
    """
    
    def __init__(self, 
                 macd_fast_period: int = 12,
                 macd_slow_period: int = 26, 
                 macd_signal_period: int = 9,
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,
                 heat_decay_half_life: float = 300.0):  # 5 minutes
        
        self.macd_fast = macd_fast_period
        self.macd_slow = macd_slow_period
        self.macd_signal = macd_signal_period
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.heat_decay_half_life = heat_decay_half_life
        
        # Historical data storage for calculations
        self.price_history = {}  # symbol -> list of prices
        self.volume_history = {}  # symbol -> list of volumes
        self.heat_history = {}  # symbol -> list of heat values
        self.last_calculation = {}  # symbol -> timestamp
        
    def add_price_data(self, symbol: str, price: float, volume: int, timestamp: datetime = None):
        """Add new price data for a symbol"""
        if timestamp is None:
            timestamp = datetime.now()
            
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
            
        # Add new data
        self.price_history[symbol].append((timestamp, price))
        self.volume_history[symbol].append((timestamp, volume))
        
        # Keep only recent data (last 200 points for calculations)
        max_points = 200
        if len(self.price_history[symbol]) > max_points:
            self.price_history[symbol] = self.price_history[symbol][-max_points:]
            self.volume_history[symbol] = self.volume_history[symbol][-max_points:]
            
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return []
            
        multiplier = 2.0 / (period + 1)
        ema_values = []
        
        # Start with SMA for first value
        sma = sum(prices[:period]) / period
        ema_values.append(sma)
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
            
        return ema_values
    
    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return []
            
        sma_values = []
        for i in range(period - 1, len(prices)):
            sma = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(sma)
            
        return sma_values
    
    def calculate_macd(self, symbol: str) -> Optional[MACDResult]:
        """Calculate MACD with signal line and histogram"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < self.macd_slow:
            return None
            
        # Extract prices
        prices = [data[1] for data in self.price_history[symbol]]
        
        # Calculate EMAs
        fast_ema = self.calculate_ema(prices, self.macd_fast)
        slow_ema = self.calculate_ema(prices, self.macd_slow)
        
        if not fast_ema or not slow_ema:
            return None
            
        # Align EMAs (slow EMA starts later)
        start_idx = self.macd_slow - self.macd_fast
        fast_ema_aligned = fast_ema[start_idx:]
        
        # Calculate MACD line
        macd_line_values = [fast - slow for fast, slow in zip(fast_ema_aligned, slow_ema)]
        
        if len(macd_line_values) < self.macd_signal:
            return None
            
        # Calculate signal line (EMA of MACD)
        signal_line_values = self.calculate_ema(macd_line_values, self.macd_signal)
        
        if not signal_line_values:
            return None
            
        # Get latest values
        macd_line = macd_line_values[-1]
        signal_line = signal_line_values[-1]
        histogram = macd_line - signal_line
        
        # Determine trend direction and strength
        if macd_line > signal_line:
            trend_direction = TrendDirection.BULLISH
        elif macd_line < signal_line:
            trend_direction = TrendDirection.BEARISH
        else:
            trend_direction = TrendDirection.NEUTRAL
            
        # Calculate signal strength based on histogram magnitude
        signal_strength = min(abs(histogram) / (abs(macd_line) + 1e-8), 1.0)
        
        # Check for crossover signals
        crossover_signal = None
        if len(macd_line_values) > 1 and len(signal_line_values) > 1:
            prev_histogram = macd_line_values[-2] - signal_line_values[-2]
            
            # Bullish crossover
            if prev_histogram <= 0 and histogram > 0:
                crossover_signal = TechnicalSignal(
                    signal_type="macd_bullish_crossover",
                    direction=TrendDirection.BULLISH,
                    strength=signal_strength,
                    confidence=min(abs(histogram) * 10, 1.0),
                    timestamp=datetime.now()
                )
            # Bearish crossover
            elif prev_histogram >= 0 and histogram < 0:
                crossover_signal = TechnicalSignal(
                    signal_type="macd_bearish_crossover", 
                    direction=TrendDirection.BEARISH,
                    strength=signal_strength,
                    confidence=min(abs(histogram) * 10, 1.0),
                    timestamp=datetime.now()
                )
        
        return MACDResult(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            trend_direction=trend_direction,
            signal_strength=signal_strength,
            crossover_signal=crossover_signal
        )
    
    def calculate_bollinger_bands(self, symbol: str) -> Optional[BollingerBandsResult]:
        """Calculate Bollinger Bands with squeeze and breakout signals"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < self.bb_period:
            return None
            
        # Extract prices
        prices = [data[1] for data in self.price_history[symbol]]
        
        # Calculate SMA and standard deviation
        sma_values = self.calculate_sma(prices, self.bb_period)
        
        if not sma_values:
            return None
            
        # Calculate standard deviation for the same period
        middle_band = sma_values[-1]
        recent_prices = prices[-self.bb_period:]
        std_dev = np.std(recent_prices)
        
        # Calculate bands
        upper_band = middle_band + (self.bb_std_dev * std_dev)
        lower_band = middle_band - (self.bb_std_dev * std_dev)
        
        # Calculate bandwidth (volatility measure)
        bandwidth = (upper_band - lower_band) / middle_band
        
        # Calculate %B (position within bands)
        current_price = prices[-1]
        bb_percent = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        
        # Detect squeeze (low volatility)
        squeeze_signal = None
        if bandwidth < 0.1:  # Threshold for squeeze
            squeeze_signal = TechnicalSignal(
                signal_type="bollinger_squeeze",
                direction=TrendDirection.NEUTRAL,
                strength=1.0 - bandwidth * 10,
                confidence=0.8,
                timestamp=datetime.now()
            )
        
        # Detect breakout signals
        breakout_signal = None
        if bb_percent > 1.0:  # Price above upper band
            breakout_signal = TechnicalSignal(
                signal_type="bollinger_breakout_high",
                direction=TrendDirection.BULLISH,
                strength=min((bb_percent - 1.0) * 5, 1.0),
                confidence=0.7,
                timestamp=datetime.now()
            )
        elif bb_percent < 0.0:  # Price below lower band
            breakout_signal = TechnicalSignal(
                signal_type="bollinger_breakout_low", 
                direction=TrendDirection.BEARISH,
                strength=min(abs(bb_percent) * 5, 1.0),
                confidence=0.7,
                timestamp=datetime.now()
            )
        
        return BollingerBandsResult(
            upper_band=upper_band,
            middle_band=middle_band,
            lower_band=lower_band,
            bandwidth=bandwidth,
            bb_percent=bb_percent,
            squeeze_signal=squeeze_signal,
            breakout_signal=breakout_signal
        )
    
    def calculate_heat_equation(self, symbol: str, 
                              current_price: float,
                              volume: int,
                              volatility: float,
                              market_sentiment: float = 0.0) -> HeatEquationResult:
        """
        Calculate dynamic heat with time decay and propagation
        
        Heat equation: ∂H/∂t = α∇²H + β(price_momentum + volume_flow + volatility_burst + sentiment)
        
        Where:
        - α is thermal diffusivity (heat decay rate)
        - β is heat generation coefficient  
        - ∇²H represents heat diffusion to neighboring nodes
        """
        
        current_time = datetime.now()
        
        # Initialize heat history if needed
        if symbol not in self.heat_history:
            self.heat_history[symbol] = []
            
        # Calculate time-based decay factor
        if symbol in self.last_calculation:
            time_diff = (current_time - self.last_calculation[symbol]).total_seconds()
            # Exponential decay: e^(-λt) where λ = ln(2)/half_life
            decay_lambda = np.log(2) / self.heat_decay_half_life
            decay_factor = np.exp(-decay_lambda * time_diff)
        else:
            decay_factor = 1.0
            
        # Get previous heat value
        previous_heat = 0.0
        if self.heat_history[symbol]:
            previous_heat = self.heat_history[symbol][-1][1] * decay_factor
        
        # Calculate heat components
        
        # 1. Price momentum (rate of price change)
        price_momentum = 0.0
        if symbol in self.price_history and len(self.price_history[symbol]) >= 2:
            prev_price = self.price_history[symbol][-2][1]
            price_change_pct = (current_price - prev_price) / prev_price
            price_momentum = min(abs(price_change_pct) * 20, 1.0)  # Normalize to [0,1]
        
        # 2. Volume flow (volume relative to average)
        volume_flow = 0.0
        if symbol in self.volume_history and len(self.volume_history[symbol]) >= 10:
            recent_volumes = [v[1] for v in self.volume_history[symbol][-10:]]
            avg_volume = np.mean(recent_volumes)
            if avg_volume > 0:
                volume_ratio = volume / avg_volume
                volume_flow = min(volume_ratio / 5.0, 1.0)  # Normalize
        
        # 3. Volatility burst
        volatility_burst = min(volatility / 2.0, 1.0)  # Normalize volatility
        
        # 4. Market sentiment effect
        sentiment_effect = max(-0.5, min(market_sentiment, 0.5))  # Bounded sentiment
        
        # Heat generation (β term)
        heat_generation = 0.3 * price_momentum + 0.2 * volume_flow + 0.3 * volatility_burst + 0.2 * sentiment_effect
        
        # Calculate current heat with decay and generation
        current_heat = previous_heat * 0.8 + heat_generation * 0.2  # Blend previous and new heat
        current_heat = max(0.0, min(current_heat, 1.0))  # Bound to [0,1]
        
        # Calculate heat velocity (rate of change)
        heat_velocity = 0.0
        if self.heat_history[symbol] and len(self.heat_history[symbol]) >= 2:
            prev_heat_val = self.heat_history[symbol][-1][1]
            time_delta = (current_time - self.heat_history[symbol][-1][0]).total_seconds()
            if time_delta > 0:
                heat_velocity = (current_heat - prev_heat_val) / time_delta
        
        # Calculate heat acceleration
        heat_acceleration = 0.0
        if len(self.heat_history[symbol]) >= 2:
            # Use finite difference approximation
            heat_values = [h[1] for h in self.heat_history[symbol][-2:]] + [current_heat]
            if len(heat_values) >= 3:
                heat_acceleration = heat_values[-1] - 2*heat_values[-2] + heat_values[-3]
        
        # Propagation strength (how much this heat affects neighbors)
        propagation_strength = current_heat * (0.5 + 0.5 * volume_flow)
        
        # Temporal validity (how long this heat measurement stays valid)
        temporal_validity = max(0.1, 1.0 - (current_heat * 0.3))  # Higher heat = faster decay
        
        # Store heat value with timestamp
        self.heat_history[symbol].append((current_time, current_heat))
        
        # Keep only recent heat history
        if len(self.heat_history[symbol]) > 100:
            self.heat_history[symbol] = self.heat_history[symbol][-100:]
            
        self.last_calculation[symbol] = current_time
        
        return HeatEquationResult(
            current_heat=current_heat,
            heat_velocity=heat_velocity,
            heat_acceleration=heat_acceleration,
            decay_factor=decay_factor,
            propagation_strength=propagation_strength,
            temporal_validity=temporal_validity
        )
    
    def calculate_dynamic_weight(self, 
                               symbol: str,
                               macd_result: Optional[MACDResult],
                               bb_result: Optional[BollingerBandsResult],
                               heat_result: HeatEquationResult,
                               additional_factors: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate comprehensive dynamic weight considering multiple factors
        
        Weight factors:
        1. Technical momentum (MACD)
        2. Volatility state (Bollinger Bands) 
        3. Heat intensity and propagation
        4. Volume confirmation
        5. Time decay
        6. Market correlation
        """
        
        weights = {
            'technical_momentum': 0.0,
            'volatility_state': 0.0, 
            'heat_intensity': 0.0,
            'volume_confirmation': 0.0,
            'time_decay': 0.0,
            'market_correlation': 0.0,
            'composite_weight': 0.0
        }
        
        # 1. Technical momentum weight (MACD-based)
        if macd_result:
            momentum_weight = macd_result.signal_strength
            if macd_result.crossover_signal:
                momentum_weight *= 1.5  # Boost for crossover signals
            weights['technical_momentum'] = momentum_weight
        
        # 2. Volatility state weight (Bollinger Bands)
        if bb_result:
            volatility_weight = 0.0
            
            # High weight for breakouts
            if bb_result.breakout_signal:
                volatility_weight = bb_result.breakout_signal.strength * 0.8
            
            # Medium weight for squeezes (potential energy)
            elif bb_result.squeeze_signal:
                volatility_weight = bb_result.squeeze_signal.strength * 0.4
            
            # Base weight from bandwidth
            else:
                volatility_weight = min(bb_result.bandwidth * 2, 0.6)
                
            weights['volatility_state'] = volatility_weight
        
        # 3. Heat intensity weight
        heat_weight = heat_result.current_heat
        
        # Boost for high velocity/acceleration
        if abs(heat_result.heat_velocity) > 0.01:
            heat_weight *= 1.2
        if abs(heat_result.heat_acceleration) > 0.005:
            heat_weight *= 1.1
            
        weights['heat_intensity'] = min(heat_weight, 1.0)
        
        # 4. Volume confirmation weight
        volume_weight = 0.0
        if symbol in self.volume_history and len(self.volume_history[symbol]) >= 5:
            recent_volumes = [v[1] for v in self.volume_history[symbol][-5:]]
            current_volume = recent_volumes[-1]
            avg_volume = np.mean(recent_volumes[:-1])
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                volume_weight = min((volume_ratio - 1.0) / 4.0, 1.0) if volume_ratio > 1 else 0
                
        weights['volume_confirmation'] = max(0.0, volume_weight)
        
        # 5. Time decay weight
        time_decay_weight = heat_result.temporal_validity
        weights['time_decay'] = time_decay_weight
        
        # 6. Additional factors (market correlation, sector strength, etc.)
        correlation_weight = 0.0
        if additional_factors:
            correlation_weight = additional_factors.get('market_correlation', 0.0)
            correlation_weight += additional_factors.get('sector_strength', 0.0)
            correlation_weight /= 2.0  # Average
            
        weights['market_correlation'] = correlation_weight
        
        # Composite weight calculation (weighted average)
        factor_weights = {
            'technical_momentum': 0.25,
            'volatility_state': 0.20,
            'heat_intensity': 0.25,
            'volume_confirmation': 0.15,
            'time_decay': 0.10,
            'market_correlation': 0.05
        }
        
        composite_weight = sum(
            weights[factor] * factor_weights[factor] 
            for factor in factor_weights.keys()
        )
        
        weights['composite_weight'] = min(composite_weight, 1.0)
        
        return weights
    
    def get_node_color_heat_map(self, heat_value: float) -> Dict[str, str]:
        """
        Generate node color based on heat level for Neo4j visualization
        
        Color scheme:
        - Cold (0.0-0.2): Blue tones
        - Warm (0.2-0.4): Green tones  
        - Hot (0.4-0.7): Yellow/Orange tones
        - Extreme (0.7-1.0): Red tones
        """
        
        if heat_value <= 0.2:
            # Cold - Blue tones
            intensity = int(heat_value * 5 * 255)  # 0-255 scale
            return {
                'background_color': f'rgb({intensity//4}, {intensity//2}, {255})',
                'border_color': f'rgb(0, 0, {200})',
                'font_color': 'white' if heat_value < 0.1 else 'black',
                'heat_level': 'cold'
            }
        elif heat_value <= 0.4:
            # Warm - Green tones
            intensity = int((heat_value - 0.2) * 5 * 255)
            return {
                'background_color': f'rgb({intensity//2}, {255}, {intensity//4})', 
                'border_color': f'rgb(0, {200}, 0)',
                'font_color': 'black',
                'heat_level': 'warm'
            }
        elif heat_value <= 0.7:
            # Hot - Yellow/Orange tones
            intensity = int((heat_value - 0.4) * (255/0.3))
            red_val = min(255, 200 + intensity//2)
            green_val = min(255, 180 + intensity//3)
            return {
                'background_color': f'rgb({red_val}, {green_val}, 0)',
                'border_color': f'rgb({255}, {165}, 0)',
                'font_color': 'black',
                'heat_level': 'hot'
            }
        else:
            # Extreme - Red tones
            intensity = int((heat_value - 0.7) * (255/0.3))
            green_val = max(0, 100 - intensity)
            blue_val = max(0, 50 - intensity//2)
            return {
                'background_color': f'rgb({255}, {green_val}, {blue_val})',
                'border_color': f'rgb({200}, 0, 0)',
                'font_color': 'white',
                'heat_level': 'extreme'
            }