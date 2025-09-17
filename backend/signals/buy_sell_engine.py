"""
Buy/Sell Signal Engine for RAGHeat Dynamic Trading Recommendations
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

class SignalStrength(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TradingSignal:
    symbol: str
    signal: SignalStrength
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float]
    stop_loss: Optional[float]
    reasoning: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'signal': self.signal.value,
            'confidence': self.confidence,
            'price_target': self.price_target,
            'stop_loss': self.stop_loss,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }

class BuySellSignalEngine:
    """
    Advanced Buy/Sell Signal Engine using multi-factor analysis
    """
    
    def __init__(self):
        self.signal_weights = {
            'heat_score': 0.25,
            'macd_signal': 0.20,
            'bollinger_position': 0.15,
            'price_momentum': 0.15,
            'volume_confirmation': 0.10,
            'volatility_score': 0.10,
            'trend_strength': 0.05
        }
    
    def calculate_trading_signal(self, 
                               symbol: str,
                               current_price: float,
                               heat_score: float,
                               macd_result: Dict,
                               bollinger_result: Dict,
                               price_change_pct: float,
                               volume_ratio: float = 1.0,
                               volatility: float = 0.0) -> TradingSignal:
        """
        Calculate comprehensive trading signal based on multiple factors
        """
        
        # Initialize scoring system
        signal_score = 0.0
        reasoning = []
        confidence_factors = []
        
        # 1. Heat Score Analysis (0.25 weight)
        heat_signal, heat_reasoning = self._analyze_heat_score(heat_score)
        signal_score += heat_signal * self.signal_weights['heat_score']
        reasoning.extend(heat_reasoning)
        confidence_factors.append(abs(heat_signal))
        
        # 2. MACD Analysis (0.20 weight)
        macd_signal, macd_reasoning = self._analyze_macd(macd_result)
        signal_score += macd_signal * self.signal_weights['macd_signal']
        reasoning.extend(macd_reasoning)
        confidence_factors.append(abs(macd_signal))
        
        # 3. Bollinger Bands Analysis (0.15 weight)
        bb_signal, bb_reasoning = self._analyze_bollinger_bands(bollinger_result, current_price)
        signal_score += bb_signal * self.signal_weights['bollinger_position']
        reasoning.extend(bb_reasoning)
        confidence_factors.append(abs(bb_signal))
        
        # 4. Price Momentum Analysis (0.15 weight)
        momentum_signal, momentum_reasoning = self._analyze_price_momentum(price_change_pct)
        signal_score += momentum_signal * self.signal_weights['price_momentum']
        reasoning.extend(momentum_reasoning)
        confidence_factors.append(abs(momentum_signal))
        
        # 5. Volume Confirmation (0.10 weight)
        volume_signal, volume_reasoning = self._analyze_volume(volume_ratio, price_change_pct)
        signal_score += volume_signal * self.signal_weights['volume_confirmation']
        reasoning.extend(volume_reasoning)
        confidence_factors.append(abs(volume_signal))
        
        # 6. Volatility Score (0.10 weight)
        vol_signal, vol_reasoning = self._analyze_volatility(volatility, heat_score)
        signal_score += vol_signal * self.signal_weights['volatility_score']
        reasoning.extend(vol_reasoning)
        confidence_factors.append(abs(vol_signal))
        
        # Convert score to signal
        signal_strength = self._score_to_signal(signal_score)
        
        # Calculate confidence based on factor alignment
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        # Calculate price targets and stop loss
        price_target, stop_loss = self._calculate_targets(
            current_price, signal_strength, volatility, heat_score
        )
        
        return TradingSignal(
            symbol=symbol,
            signal=signal_strength,
            confidence=confidence,
            price_target=price_target,
            stop_loss=stop_loss,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def _analyze_heat_score(self, heat_score: float) -> Tuple[float, List[str]]:
        """Analyze heat score for trading signals"""
        reasoning = []
        
        if heat_score > 0.8:
            reasoning.append(f"🔥 EXTREME HEAT ({heat_score:.3f}) - Strong momentum")
            return 0.8, reasoning
        elif heat_score > 0.6:
            reasoning.append(f"🔥 HIGH HEAT ({heat_score:.3f}) - Good momentum")
            return 0.6, reasoning
        elif heat_score > 0.4:
            reasoning.append(f"🌡️ WARM ({heat_score:.3f}) - Building momentum")
            return 0.3, reasoning
        elif heat_score > 0.2:
            reasoning.append(f"❄️ COOL ({heat_score:.3f}) - Low momentum")
            return -0.2, reasoning
        else:
            reasoning.append(f"❄️ COLD ({heat_score:.3f}) - Very low momentum")
            return -0.4, reasoning
    
    def _analyze_macd(self, macd_result: Dict) -> Tuple[float, List[str]]:
        """Analyze MACD for trading signals"""
        reasoning = []
        
        if not macd_result:
            return 0.0, ["📊 MACD data unavailable"]
        
        macd_line = macd_result.get('macd_line', 0)
        signal_line = macd_result.get('signal_line', 0)
        histogram = macd_result.get('histogram', 0)
        
        # MACD above signal line = bullish
        if macd_line > signal_line and histogram > 0:
            if histogram > 0.5:
                reasoning.append("📈 MACD STRONG BULLISH - Line above signal, increasing histogram")
                return 0.7, reasoning
            else:
                reasoning.append("📈 MACD BULLISH - Line above signal")
                return 0.4, reasoning
        elif macd_line < signal_line and histogram < 0:
            if histogram < -0.5:
                reasoning.append("📉 MACD STRONG BEARISH - Line below signal, decreasing histogram")
                return -0.7, reasoning
            else:
                reasoning.append("📉 MACD BEARISH - Line below signal")
                return -0.4, reasoning
        else:
            reasoning.append("📊 MACD NEUTRAL - Mixed signals")
            return 0.0, reasoning
    
    def _analyze_bollinger_bands(self, bb_result: Dict, current_price: float) -> Tuple[float, List[str]]:
        """Analyze Bollinger Bands for trading signals"""
        reasoning = []
        
        if not bb_result:
            return 0.0, ["📊 Bollinger Bands data unavailable"]
        
        upper_band = bb_result.get('upper_band', current_price * 1.02)
        lower_band = bb_result.get('lower_band', current_price * 0.98)
        middle_band = bb_result.get('middle_band', current_price)
        
        # Calculate position within bands
        band_width = upper_band - lower_band
        if band_width > 0:
            position = (current_price - lower_band) / band_width
            
            if position > 0.8:
                reasoning.append(f"🔴 NEAR UPPER BAND ({position:.1%}) - Potential resistance")
                return -0.5, reasoning
            elif position < 0.2:
                reasoning.append(f"🟢 NEAR LOWER BAND ({position:.1%}) - Potential support")
                return 0.5, reasoning
            elif 0.4 <= position <= 0.6:
                reasoning.append(f"🟡 MID-BAND ({position:.1%}) - Neutral position")
                return 0.0, reasoning
            else:
                reasoning.append(f"📊 Band position: {position:.1%}")
                return 0.1 if position < 0.5 else -0.1, reasoning
        
        return 0.0, reasoning
    
    def _analyze_price_momentum(self, price_change_pct: float) -> Tuple[float, List[str]]:
        """Analyze price momentum"""
        reasoning = []
        
        if price_change_pct > 5.0:
            reasoning.append(f"🚀 STRONG UPWARD MOMENTUM (+{price_change_pct:.1f}%)")
            return 0.8, reasoning
        elif price_change_pct > 2.0:
            reasoning.append(f"📈 GOOD UPWARD MOMENTUM (+{price_change_pct:.1f}%)")
            return 0.5, reasoning
        elif price_change_pct > 0.5:
            reasoning.append(f"↗️ POSITIVE MOMENTUM (+{price_change_pct:.1f}%)")
            return 0.2, reasoning
        elif price_change_pct < -5.0:
            reasoning.append(f"📉 STRONG DOWNWARD MOMENTUM ({price_change_pct:.1f}%)")
            return -0.8, reasoning
        elif price_change_pct < -2.0:
            reasoning.append(f"📉 NEGATIVE MOMENTUM ({price_change_pct:.1f}%)")
            return -0.5, reasoning
        elif price_change_pct < -0.5:
            reasoning.append(f"↘️ SLIGHT DECLINE ({price_change_pct:.1f}%)")
            return -0.2, reasoning
        else:
            reasoning.append(f"➡️ SIDEWAYS ({price_change_pct:.1f}%)")
            return 0.0, reasoning
    
    def _analyze_volume(self, volume_ratio: float, price_change: float) -> Tuple[float, List[str]]:
        """Analyze volume confirmation"""
        reasoning = []
        
        if volume_ratio > 1.5 and price_change > 0:
            reasoning.append(f"📊 HIGH VOLUME BULLISH CONFIRMATION ({volume_ratio:.1f}x)")
            return 0.6, reasoning
        elif volume_ratio > 1.5 and price_change < 0:
            reasoning.append(f"📊 HIGH VOLUME BEARISH CONFIRMATION ({volume_ratio:.1f}x)")
            return -0.6, reasoning
        elif volume_ratio > 1.2:
            reasoning.append(f"📊 ELEVATED VOLUME ({volume_ratio:.1f}x)")
            return 0.2 if price_change > 0 else -0.2, reasoning
        elif volume_ratio < 0.8:
            reasoning.append(f"📊 LOW VOLUME ({volume_ratio:.1f}x) - Weak confirmation")
            return -0.1, reasoning
        else:
            reasoning.append(f"📊 NORMAL VOLUME ({volume_ratio:.1f}x)")
            return 0.0, reasoning
    
    def _analyze_volatility(self, volatility: float, heat_score: float) -> Tuple[float, List[str]]:
        """Analyze volatility for risk assessment"""
        reasoning = []
        
        if volatility > 0.8:
            reasoning.append(f"⚠️ VERY HIGH VOLATILITY ({volatility:.1%}) - High risk/reward")
            return -0.3 if heat_score < 0.3 else 0.2, reasoning
        elif volatility > 0.5:
            reasoning.append(f"⚠️ HIGH VOLATILITY ({volatility:.1%}) - Increased risk")
            return -0.2 if heat_score < 0.3 else 0.1, reasoning
        elif volatility > 0.3:
            reasoning.append(f"📊 MODERATE VOLATILITY ({volatility:.1%})")
            return 0.0, reasoning
        else:
            reasoning.append(f"📊 LOW VOLATILITY ({volatility:.1%}) - Stable")
            return 0.1, reasoning
    
    def _score_to_signal(self, score: float) -> SignalStrength:
        """Convert numerical score to signal strength"""
        if score >= 0.6:
            return SignalStrength.STRONG_BUY
        elif score >= 0.3:
            return SignalStrength.BUY
        elif score >= 0.1:
            return SignalStrength.WEAK_BUY
        elif score >= -0.1:
            return SignalStrength.HOLD
        elif score >= -0.3:
            return SignalStrength.WEAK_SELL
        elif score >= -0.6:
            return SignalStrength.SELL
        else:
            return SignalStrength.STRONG_SELL
    
    def _calculate_targets(self, current_price: float, signal: SignalStrength, 
                          volatility: float, heat_score: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate price targets and stop loss levels"""
        
        # Base target percentages
        if signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            target_pct = 0.08 + (volatility * 0.05) + (heat_score * 0.03)
            stop_pct = 0.05 + (volatility * 0.02)
            return current_price * (1 + target_pct), current_price * (1 - stop_pct)
        
        elif signal in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
            target_pct = 0.08 + (volatility * 0.05)
            stop_pct = 0.05 + (volatility * 0.02)
            return current_price * (1 - target_pct), current_price * (1 + stop_pct)
        
        else:
            # For HOLD, WEAK_BUY, WEAK_SELL - conservative targets
            target_pct = 0.03 + (volatility * 0.02)
            stop_pct = 0.03 + (volatility * 0.02)
            
            if 'BUY' in signal.value:
                return current_price * (1 + target_pct), current_price * (1 - stop_pct)
            elif 'SELL' in signal.value:
                return current_price * (1 - target_pct), current_price * (1 + stop_pct)
            else:
                return None, current_price * (1 - stop_pct)  # Only stop loss for HOLD