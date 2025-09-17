"""
HMM-based Trading Signal Generator
Integrates Hidden Markov Model regime detection with existing trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .hmm_model import HMMMarketModel, HMMPrediction
from .market_regime_detector import MarketRegimeDetector, RegimeDetectionResult
from .data_preprocessor import TimeSeriesPreprocessor

try:
    from ...signals.buy_sell_engine import BuySellSignalEngine, TradingSignal, SignalStrength
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from signals.buy_sell_engine import BuySellSignalEngine, TradingSignal, SignalStrength


@dataclass
class HMMSignal:
    """Enhanced trading signal with HMM regime detection"""
    symbol: str
    base_signal: TradingSignal
    regime_detection: RegimeDetectionResult
    hmm_prediction: HMMPrediction
    combined_signal: SignalStrength
    combined_confidence: float
    regime_adjusted_targets: Dict[str, float]
    position_sizing: Dict[str, float]
    risk_assessment: Dict[str, any]
    signal_components: Dict[str, float]
    regime_timing: Dict[str, any]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'base_signal': self.base_signal.to_dict(),
            'regime_detection': self.regime_detection.to_dict(),
            'hmm_prediction': self.hmm_prediction.to_dict(),
            'combined_signal': self.combined_signal.value,
            'combined_confidence': self.combined_confidence,
            'regime_adjusted_targets': self.regime_adjusted_targets,
            'position_sizing': self.position_sizing,
            'risk_assessment': self.risk_assessment,
            'signal_components': self.signal_components,
            'regime_timing': self.regime_timing,
            'timestamp': self.timestamp.isoformat()
        }


class HMMSignalGenerator:
    """
    Advanced signal generator combining HMM regime detection
    with traditional technical analysis signals
    """
    
    def __init__(self,
                 hmm_weight: float = 0.4,
                 technical_weight: float = 0.4,
                 regime_weight: float = 0.2,
                 n_regimes: int = 4,
                 regime_lookback: int = 252,
                 confidence_threshold: float = 0.3):
        """
        Initialize HMM signal generator
        
        Args:
            hmm_weight: Weight for HMM state predictions (0-1)
            technical_weight: Weight for technical analysis signals (0-1)
            regime_weight: Weight for regime-based adjustments (0-1)
            n_regimes: Number of market regimes to detect
            regime_lookback: Days of history for regime detection
            confidence_threshold: Minimum confidence for strong signals
        """
        total_weight = hmm_weight + technical_weight + regime_weight
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
            
        self.hmm_weight = hmm_weight
        self.technical_weight = technical_weight
        self.regime_weight = regime_weight
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector(
            n_regimes=n_regimes,
            lookback_period=regime_lookback
        )
        self.base_signal_engine = BuySellSignalEngine()
        self.preprocessor = TimeSeriesPreprocessor()
        
        # Cache for efficiency
        self._regime_cache = {}
        self._signal_history = {}
    
    def generate_signal(self,
                       symbol: str,
                       current_price: float,
                       historical_data: Optional[pd.DataFrame] = None,
                       heat_score: float = 0.5,
                       macd_result: Optional[Dict] = None,
                       bollinger_result: Optional[Dict] = None,
                       price_change_pct: float = 0.0,
                       volume_ratio: float = 1.0) -> HMMSignal:
        """
        Generate comprehensive trading signal with HMM regime analysis
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            historical_data: Historical OHLCV data
            heat_score: RAGHeat momentum score
            macd_result: MACD analysis results
            bollinger_result: Bollinger Bands analysis
            price_change_pct: Recent price change percentage
            volume_ratio: Volume ratio vs average
            
        Returns:
            HMMSignal with comprehensive analysis
        """
        try:
            # Step 1: Detect market regime
            regime_detection = self.regime_detector.detect_regime(
                symbol=symbol,
                data=historical_data
            )
            
            # Step 2: Get HMM prediction
            hmm_prediction = regime_detection  # Contains HMM prediction info
            
            # Step 3: Generate base technical signal
            regime_volatility = self._extract_regime_volatility(regime_detection)
            
            base_signal = self.base_signal_engine.calculate_trading_signal(
                symbol=symbol,
                current_price=current_price,
                heat_score=heat_score,
                macd_result=macd_result or {},
                bollinger_result=bollinger_result or {},
                price_change_pct=price_change_pct,
                volume_ratio=volume_ratio,
                volatility=regime_volatility
            )
            
            # Step 4: Create HMM prediction object for compatibility
            hmm_pred = self._create_hmm_prediction_from_regime(regime_detection)
            
            # Step 5: Combine signals
            combined_signal, combined_confidence, signal_components = self._combine_signals(
                base_signal, hmm_pred, regime_detection
            )
            
            # Step 6: Calculate regime-adjusted targets
            regime_targets = self._calculate_regime_adjusted_targets(
                current_price, combined_signal, regime_detection, base_signal
            )
            
            # Step 7: Calculate position sizing
            position_sizing = self._calculate_position_sizing(
                regime_detection, combined_signal, combined_confidence
            )
            
            # Step 8: Assess risk
            risk_assessment = self._assess_regime_risk(
                regime_detection, hmm_pred, base_signal
            )
            
            # Step 9: Analyze regime timing
            regime_timing = self._analyze_regime_timing(regime_detection)
            
            # Store in history
            self._update_signal_history(symbol, combined_signal, regime_detection)
            
            return HMMSignal(
                symbol=symbol,
                base_signal=base_signal,
                regime_detection=regime_detection,
                hmm_prediction=hmm_pred,
                combined_signal=combined_signal,
                combined_confidence=combined_confidence,
                regime_adjusted_targets=regime_targets,
                position_sizing=position_sizing,
                risk_assessment=risk_assessment,
                signal_components=signal_components,
                regime_timing=regime_timing,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error generating HMM signal for {symbol}: {e}")
            return self._generate_fallback_signal(symbol, current_price, heat_score)
    
    def _extract_regime_volatility(self, regime_detection: RegimeDetectionResult) -> float:
        """Extract volatility estimate from regime detection"""
        # Use regime-based volatility estimate
        regime_name = regime_detection.current_regime.lower()
        
        if 'crisis' in regime_name:
            return 0.06  # High volatility
        elif 'bear' in regime_name:
            return 0.04  # Medium-high volatility
        elif 'bull' in regime_name:
            return 0.02  # Medium volatility
        else:  # sideways
            return 0.015  # Low volatility
    
    def _create_hmm_prediction_from_regime(self, regime_detection: RegimeDetectionResult) -> HMMPrediction:
        """Create HMMPrediction object from regime detection for compatibility"""
        
        # Map regime detection to HMM prediction structure
        state_names = list(regime_detection.transition_probabilities.keys())
        current_state_idx = state_names.index(regime_detection.current_regime) if regime_detection.current_regime in state_names else 0
        
        state_probs = [regime_detection.transition_probabilities.get(name, 0.25) for name in state_names]
        
        return HMMPrediction(
            symbol=regime_detection.symbol,
            current_state=current_state_idx,
            state_probabilities=state_probs,
            predicted_state=current_state_idx,  # Simplified
            state_names=state_names,
            regime_characteristics={},  # Filled from regime detection
            signal_strength=self._regime_to_signal_strength(regime_detection),
            confidence=regime_detection.confidence,
            prediction_horizon=1,
            model_accuracy=regime_detection.confidence,
            timestamp=regime_detection.timestamp
        )
    
    def _regime_to_signal_strength(self, regime_detection: RegimeDetectionResult) -> float:
        """Convert regime detection to signal strength"""
        
        regime_name = regime_detection.current_regime.lower()
        regime_strength = regime_detection.regime_strength
        
        if 'bull' in regime_name:
            return 0.6 * regime_strength
        elif 'bear' in regime_name:
            return -0.6 * regime_strength
        elif 'crisis' in regime_name:
            return -0.8 * regime_strength
        else:  # sideways
            return 0.0
    
    def _combine_signals(self,
                        base_signal: TradingSignal,
                        hmm_prediction: HMMPrediction,
                        regime_detection: RegimeDetectionResult) -> Tuple[SignalStrength, float, Dict[str, float]]:
        """
        Combine base technical signal with HMM regime analysis
        
        Returns:
            Tuple of (combined_signal, combined_confidence, signal_components)
        """
        # Convert signals to numerical scores
        signal_mapping = {
            SignalStrength.STRONG_SELL: -1.0,
            SignalStrength.SELL: -0.67,
            SignalStrength.WEAK_SELL: -0.33,
            SignalStrength.HOLD: 0.0,
            SignalStrength.WEAK_BUY: 0.33,
            SignalStrength.BUY: 0.67,
            SignalStrength.STRONG_BUY: 1.0
        }
        
        base_score = signal_mapping[base_signal.signal]
        hmm_score = hmm_prediction.signal_strength
        
        # Regime-based adjustment
        regime_adjustment = self._calculate_regime_adjustment(regime_detection)
        
        # Confidence weighting
        base_confidence_weight = base_signal.confidence
        hmm_confidence_weight = hmm_prediction.confidence
        regime_confidence_weight = regime_detection.confidence
        
        # Weighted combination
        technical_component = base_score * self.technical_weight * base_confidence_weight
        hmm_component = hmm_score * self.hmm_weight * hmm_confidence_weight
        regime_component = regime_adjustment * self.regime_weight * regime_confidence_weight
        
        combined_score = technical_component + hmm_component + regime_component
        
        # Combined confidence
        combined_confidence = (
            base_signal.confidence * self.technical_weight +
            hmm_prediction.confidence * self.hmm_weight +
            regime_detection.confidence * self.regime_weight
        )
        
        # Risk-based adjustments
        if regime_detection.risk_level == "HIGH":
            combined_confidence *= 0.7  # Reduce confidence in high risk
            combined_score *= 0.8  # Dampen signals in high risk
        elif regime_detection.risk_level == "LOW":
            combined_confidence *= 1.1  # Boost confidence in low risk
            combined_score *= 1.05  # Amplify signals in low risk
        
        # Regime stability adjustment
        if regime_detection.regime_strength < 0.3:  # Unstable regime
            combined_score *= 0.7  # More conservative in unstable regimes
        
        # Clamp values
        combined_score = max(-1.0, min(1.0, combined_score))
        combined_confidence = max(0.0, min(1.0, combined_confidence))
        
        # Convert back to signal strength
        combined_signal = self._score_to_signal_strength(combined_score)
        
        signal_components = {
            'technical_component': technical_component,
            'hmm_component': hmm_component,
            'regime_component': regime_component,
            'base_score': base_score,
            'hmm_score': hmm_score,
            'regime_adjustment': regime_adjustment,
            'final_score': combined_score
        }
        
        return combined_signal, combined_confidence, signal_components
    
    def _calculate_regime_adjustment(self, regime_detection: RegimeDetectionResult) -> float:
        """Calculate regime-based signal adjustment"""
        
        regime_name = regime_detection.current_regime.lower()
        regime_duration = regime_detection.regime_duration
        regime_strength = regime_detection.regime_strength
        
        # Base adjustment from regime type
        if 'bull' in regime_name:
            base_adjustment = 0.4
        elif 'bear' in regime_name:
            base_adjustment = -0.4
        elif 'crisis' in regime_name:
            base_adjustment = -0.7
        else:  # sideways
            base_adjustment = 0.0
        
        # Adjust for regime maturity
        if regime_duration > 30:  # Mature regime
            maturity_factor = 0.7  # Reduce signal as regime ages
        elif regime_duration < 5:  # New regime
            maturity_factor = 1.2  # Boost signal for new regimes
        else:
            maturity_factor = 1.0
        
        # Adjust for regime strength
        strength_factor = 0.5 + regime_strength  # 0.5 to 1.5 range
        
        return base_adjustment * maturity_factor * strength_factor
    
    def _score_to_signal_strength(self, score: float) -> SignalStrength:
        """Convert numerical score back to SignalStrength enum"""
        if score >= 0.75:
            return SignalStrength.STRONG_BUY
        elif score >= 0.4:
            return SignalStrength.BUY
        elif score >= 0.15:
            return SignalStrength.WEAK_BUY
        elif score >= -0.15:
            return SignalStrength.HOLD
        elif score >= -0.4:
            return SignalStrength.WEAK_SELL
        elif score >= -0.75:
            return SignalStrength.SELL
        else:
            return SignalStrength.STRONG_SELL
    
    def _calculate_regime_adjusted_targets(self,
                                         current_price: float,
                                         signal: SignalStrength,
                                         regime_detection: RegimeDetectionResult,
                                         base_signal: TradingSignal) -> Dict[str, float]:
        """Calculate regime-adjusted price targets and stops"""
        
        regime_name = regime_detection.current_regime.lower()
        regime_duration = regime_detection.regime_duration
        regime_strength = regime_detection.regime_strength
        
        # Base volatility estimate from regime
        if 'crisis' in regime_name:
            vol_multiplier = 2.0
        elif 'bear' in regime_name:
            vol_multiplier = 1.5
        elif 'bull' in regime_name:
            vol_multiplier = 1.0
        else:  # sideways
            vol_multiplier = 0.7
        
        targets = {}
        
        if signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            # Bullish targets
            base_target_pct = 0.08 if signal == SignalStrength.STRONG_BUY else 0.05
            base_stop_pct = 0.04 if signal == SignalStrength.STRONG_BUY else 0.03
            
            # Adjust for regime
            if 'bull' in regime_name:
                target_multiplier = 1.2  # More aggressive in bull market
                stop_multiplier = 0.8    # Tighter stops
            elif 'bear' in regime_name:
                target_multiplier = 0.7  # Conservative in bear market
                stop_multiplier = 1.5    # Wider stops
            else:
                target_multiplier = 1.0
                stop_multiplier = 1.0
            
            target_pct = base_target_pct * vol_multiplier * target_multiplier
            stop_pct = base_stop_pct * vol_multiplier * stop_multiplier
            
            targets['price_target'] = current_price * (1 + target_pct)
            targets['stop_loss'] = current_price * (1 - stop_pct)
            targets['take_profit_1'] = current_price * (1 + target_pct * 0.4)
            targets['take_profit_2'] = current_price * (1 + target_pct * 0.7)
            
        elif signal in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
            # Bearish targets (for short positions)
            base_target_pct = 0.08 if signal == SignalStrength.STRONG_SELL else 0.05
            base_stop_pct = 0.04 if signal == SignalStrength.STRONG_SELL else 0.03
            
            if 'bear' in regime_name or 'crisis' in regime_name:
                target_multiplier = 1.3  # More aggressive in bear/crisis
                stop_multiplier = 0.8
            elif 'bull' in regime_name:
                target_multiplier = 0.6  # Conservative shorting in bull
                stop_multiplier = 1.5
            else:
                target_multiplier = 1.0
                stop_multiplier = 1.0
            
            target_pct = base_target_pct * vol_multiplier * target_multiplier
            stop_pct = base_stop_pct * vol_multiplier * stop_multiplier
            
            targets['price_target'] = current_price * (1 - target_pct)
            targets['stop_loss'] = current_price * (1 + stop_pct)
            targets['take_profit_1'] = current_price * (1 - target_pct * 0.4)
            targets['take_profit_2'] = current_price * (1 - target_pct * 0.7)
            
        else:
            # Neutral signals - conservative targets
            stop_pct = 0.03 * vol_multiplier
            targets['stop_loss'] = current_price * (1 - stop_pct)
        
        # Add regime-specific levels
        targets['regime_volatility_estimate'] = vol_multiplier * 0.02
        targets['regime_support'] = current_price * (1 - vol_multiplier * 0.05)
        targets['regime_resistance'] = current_price * (1 + vol_multiplier * 0.05)
        
        return targets
    
    def _calculate_position_sizing(self,
                                 regime_detection: RegimeDetectionResult,
                                 signal: SignalStrength,
                                 confidence: float) -> Dict[str, float]:
        """Calculate position sizing based on regime and signal"""
        
        # Base position size (as fraction of portfolio)
        base_size = 0.05  # 5% default
        
        # Adjust for regime risk
        if regime_detection.risk_level == "LOW":
            risk_multiplier = 1.5
        elif regime_detection.risk_level == "MEDIUM":
            risk_multiplier = 1.0
        else:  # HIGH
            risk_multiplier = 0.5
        
        # Adjust for signal strength
        signal_strength_map = {
            SignalStrength.STRONG_BUY: 1.5,
            SignalStrength.BUY: 1.2,
            SignalStrength.WEAK_BUY: 0.8,
            SignalStrength.HOLD: 0.0,
            SignalStrength.WEAK_SELL: 0.8,
            SignalStrength.SELL: 1.2,
            SignalStrength.STRONG_SELL: 1.5
        }
        
        signal_multiplier = signal_strength_map.get(signal, 1.0)
        
        # Adjust for confidence
        confidence_multiplier = 0.5 + confidence  # 0.5 to 1.5 range
        
        # Adjust for regime stability
        regime_stability = regime_detection.regime_strength
        stability_multiplier = 0.7 + (regime_stability * 0.6)  # 0.7 to 1.3 range
        
        # Calculate final position size
        position_size = (base_size * risk_multiplier * signal_multiplier * 
                        confidence_multiplier * stability_multiplier)
        
        # Clamp to reasonable range
        position_size = max(0.01, min(0.25, position_size))
        
        return {
            'recommended_position_size': position_size,
            'max_position_size': position_size * 1.5,
            'conservative_size': position_size * 0.7,
            'risk_multiplier': risk_multiplier,
            'signal_multiplier': signal_multiplier,
            'confidence_multiplier': confidence_multiplier,
            'stability_multiplier': stability_multiplier
        }
    
    def _assess_regime_risk(self,
                          regime_detection: RegimeDetectionResult,
                          hmm_prediction: HMMPrediction,
                          base_signal: TradingSignal) -> Dict[str, any]:
        """Comprehensive risk assessment"""
        
        risk_factors = {
            'regime_risk': regime_detection.risk_level,
            'regime_uncertainty': 1 - regime_detection.confidence,
            'regime_instability': 1 - regime_detection.regime_strength,
            'model_uncertainty': 1 - hmm_prediction.confidence,
            'signal_conflict': self._assess_signal_conflict(base_signal, hmm_prediction),
            'regime_transition_risk': self._assess_transition_risk(regime_detection)
        }
        
        # Overall risk score
        risk_weights = {
            'regime_risk': 0.3,
            'regime_uncertainty': 0.2,
            'regime_instability': 0.2,
            'model_uncertainty': 0.1,
            'signal_conflict': 0.1,
            'regime_transition_risk': 0.1
        }
        
        # Convert regime risk to numeric
        regime_risk_map = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8}
        numeric_regime_risk = regime_risk_map.get(risk_factors['regime_risk'], 0.5)
        
        overall_risk = (
            numeric_regime_risk * risk_weights['regime_risk'] +
            risk_factors['regime_uncertainty'] * risk_weights['regime_uncertainty'] +
            risk_factors['regime_instability'] * risk_weights['regime_instability'] +
            risk_factors['model_uncertainty'] * risk_weights['model_uncertainty'] +
            risk_factors['signal_conflict'] * risk_weights['signal_conflict'] +
            risk_factors['regime_transition_risk'] * risk_weights['regime_transition_risk']
        )
        
        return {
            'overall_risk_score': overall_risk,
            'risk_level': "LOW" if overall_risk < 0.3 else "MEDIUM" if overall_risk < 0.7 else "HIGH",
            'risk_factors': risk_factors,
            'recommended_hedge': overall_risk > 0.6,
            'max_leverage': 3.0 if overall_risk < 0.3 else 2.0 if overall_risk < 0.6 else 1.0
        }
    
    def _assess_signal_conflict(self, base_signal: TradingSignal, hmm_prediction: HMMPrediction) -> float:
        """Assess conflict between technical and HMM signals"""
        
        signal_mapping = {
            SignalStrength.STRONG_SELL: -1.0,
            SignalStrength.SELL: -0.67,
            SignalStrength.WEAK_SELL: -0.33,
            SignalStrength.HOLD: 0.0,
            SignalStrength.WEAK_BUY: 0.33,
            SignalStrength.BUY: 0.67,
            SignalStrength.STRONG_BUY: 1.0
        }
        
        base_score = signal_mapping[base_signal.signal]
        hmm_score = hmm_prediction.signal_strength
        
        # Calculate conflict as normalized difference
        conflict = abs(base_score - hmm_score) / 2.0  # Normalize to [0, 1]
        
        return conflict
    
    def _assess_transition_risk(self, regime_detection: RegimeDetectionResult) -> float:
        """Assess risk of regime transition"""
        
        # Risk is higher when:
        # 1. Regime has been active for a long time
        # 2. Transition probabilities are more evenly distributed
        
        duration_risk = min(1.0, regime_detection.regime_duration / 60)  # Risk increases after 60 days
        
        # Calculate entropy of transition probabilities
        transition_probs = list(regime_detection.transition_probabilities.values())
        if transition_probs:
            # Add small epsilon to avoid log(0)
            probs = np.array(transition_probs) + 1e-8
            probs = probs / probs.sum()  # Normalize
            entropy = -np.sum(probs * np.log(probs))
            max_entropy = np.log(len(probs))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 0.5
        
        transition_risk = (duration_risk + normalized_entropy) / 2
        
        return transition_risk
    
    def _analyze_regime_timing(self, regime_detection: RegimeDetectionResult) -> Dict[str, any]:
        """Analyze timing aspects of the current regime"""
        
        regime_duration = regime_detection.regime_duration
        regime_name = regime_detection.current_regime.lower()
        
        # Estimate typical regime durations
        typical_durations = {
            'bull': 180,    # 6 months
            'bear': 120,    # 4 months
            'sideways': 90, # 3 months
            'crisis': 45    # 1.5 months
        }
        
        # Find the most relevant duration
        relevant_duration = 120  # default
        for key, duration in typical_durations.items():
            if key in regime_name:
                relevant_duration = duration
                break
        
        # Calculate timing metrics
        regime_maturity = min(1.0, regime_duration / relevant_duration)
        
        if regime_maturity < 0.2:
            timing_phase = "Early"
            timing_confidence = 0.7  # Less confident in very new regimes
        elif regime_maturity < 0.8:
            timing_phase = "Established"
            timing_confidence = 1.0  # Most confident in established regimes
        else:
            timing_phase = "Mature"
            timing_confidence = 0.6  # Less confident in very mature regimes
        
        return {
            'regime_duration_days': regime_duration,
            'typical_duration_days': relevant_duration,
            'regime_maturity': regime_maturity,
            'timing_phase': timing_phase,
            'timing_confidence': timing_confidence,
            'estimated_remaining_days': max(0, relevant_duration - regime_duration),
            'regime_exhaustion_risk': regime_maturity
        }
    
    def _update_signal_history(self,
                             symbol: str,
                             signal: SignalStrength,
                             regime_detection: RegimeDetectionResult):
        """Update signal history for tracking"""
        
        if symbol not in self._signal_history:
            self._signal_history[symbol] = []
        
        self._signal_history[symbol].append({
            'signal': signal.value,
            'regime': regime_detection.current_regime,
            'confidence': regime_detection.confidence,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(days=90)
        self._signal_history[symbol] = [
            h for h in self._signal_history[symbol]
            if h['timestamp'] > cutoff
        ]
    
    def _generate_fallback_signal(self,
                                symbol: str,
                                current_price: float,
                                heat_score: float) -> HMMSignal:
        """Generate fallback signal when HMM analysis fails"""
        
        # Simple fallback based on heat score
        if heat_score > 0.7:
            signal_strength = SignalStrength.BUY
            confidence = 0.6
        elif heat_score > 0.3:
            signal_strength = SignalStrength.WEAK_BUY
            confidence = 0.4
        elif heat_score < 0.3:
            signal_strength = SignalStrength.WEAK_SELL
            confidence = 0.4
        else:
            signal_strength = SignalStrength.HOLD
            confidence = 0.3
        
        fallback_base_signal = TradingSignal(
            symbol=symbol,
            signal=signal_strength,
            confidence=confidence,
            price_target=current_price * 1.05 if 'BUY' in signal_strength.value else current_price * 0.95,
            stop_loss=current_price * 0.95 if 'BUY' in signal_strength.value else current_price * 1.05,
            reasoning=[f"Fallback signal based on heat score: {heat_score}"],
            timestamp=datetime.now()
        )
        
        # Create fallback regime detection
        fallback_regime = RegimeDetectionResult(
            symbol=symbol,
            current_regime="Unknown",
            regime_probability=0.5,
            regime_duration=1,
            regime_strength=0.5,
            transition_probabilities={"Unknown": 1.0},
            regime_forecast={"Unknown": 1.0},
            risk_level="MEDIUM",
            recommended_action="HOLD",
            confidence=0.3,
            timestamp=datetime.now()
        )
        
        fallback_hmm_pred = HMMPrediction(
            symbol=symbol,
            current_state=1,
            state_probabilities=[0.33, 0.34, 0.33],
            predicted_state=1,
            state_names=["Bear", "Sideways", "Bull"],
            regime_characteristics={},
            signal_strength=0.0,
            confidence=0.3,
            prediction_horizon=1,
            model_accuracy=0.0,
            timestamp=datetime.now()
        )
        
        return HMMSignal(
            symbol=symbol,
            base_signal=fallback_base_signal,
            regime_detection=fallback_regime,
            hmm_prediction=fallback_hmm_pred,
            combined_signal=signal_strength,
            combined_confidence=confidence,
            regime_adjusted_targets={'stop_loss': current_price * 0.95},
            position_sizing={'recommended_position_size': 0.02},
            risk_assessment={'overall_risk_score': 0.5, 'risk_level': 'MEDIUM'},
            signal_components={'fallback': True},
            regime_timing={'timing_phase': 'Unknown'},
            timestamp=datetime.now()
        )
    
    def get_signal_summary(self) -> Dict[str, any]:
        """Get summary of signal generation system"""
        
        return {
            "weights": {
                "hmm_weight": self.hmm_weight,
                "technical_weight": self.technical_weight,
                "regime_weight": self.regime_weight
            },
            "confidence_threshold": self.confidence_threshold,
            "regime_detector_summary": self.regime_detector.get_regime_summary(),
            "tracked_symbols": list(self._signal_history.keys()),
            "cache_size": len(self._regime_cache)
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self._regime_cache.clear()
        self._signal_history.clear()
        self.regime_detector.clear_cache()