"""
Unified Signal System integrating GARCH, HMM, and traditional technical analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .garch_signal_generator import GARCHSignalGenerator, GARCHSignal
from .hmm_signal_generator import HMMSignalGenerator, HMMSignal
from .data_preprocessor import TimeSeriesPreprocessor

try:
    from ...signals.buy_sell_engine import BuySellSignalEngine, TradingSignal, SignalStrength
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from signals.buy_sell_engine import BuySellSignalEngine, TradingSignal, SignalStrength


@dataclass
class UnifiedSignal:
    """Comprehensive trading signal combining multiple models"""
    symbol: str
    base_technical_signal: TradingSignal
    garch_signal: GARCHSignal
    hmm_signal: HMMSignal
    unified_signal: SignalStrength
    unified_confidence: float
    model_consensus: Dict[str, any]
    risk_adjusted_targets: Dict[str, float]
    portfolio_allocation: Dict[str, float]
    execution_strategy: Dict[str, any]
    model_diagnostics: Dict[str, any]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'base_technical_signal': self.base_technical_signal.to_dict(),
            'garch_signal': self.garch_signal.to_dict(),
            'hmm_signal': self.hmm_signal.to_dict(),
            'unified_signal': self.unified_signal.value,
            'unified_confidence': self.unified_confidence,
            'model_consensus': self.model_consensus,
            'risk_adjusted_targets': self.risk_adjusted_targets,
            'portfolio_allocation': self.portfolio_allocation,
            'execution_strategy': self.execution_strategy,
            'model_diagnostics': self.model_diagnostics,
            'timestamp': self.timestamp.isoformat()
        }


class UnifiedSignalSystem:
    """
    Unified Signal System that combines:
    - Traditional Technical Analysis
    - GARCH Volatility Models
    - Hidden Markov Model Regime Detection
    - Risk Management and Portfolio Optimization
    """
    
    def __init__(self,
                 technical_weight: float = 0.3,
                 garch_weight: float = 0.35,
                 hmm_weight: float = 0.35,
                 consensus_threshold: float = 0.6,
                 risk_management_mode: str = "aggressive",  # conservative, moderate, aggressive
                 enable_regime_overlay: bool = True):
        """
        Initialize unified signal system
        
        Args:
            technical_weight: Weight for technical analysis signals
            garch_weight: Weight for GARCH-based signals
            hmm_weight: Weight for HMM regime-based signals
            consensus_threshold: Minimum consensus required for strong signals
            risk_management_mode: Risk management approach
            enable_regime_overlay: Whether to use regime overlay for all signals
        """
        total_weight = technical_weight + garch_weight + hmm_weight
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
        
        self.technical_weight = technical_weight
        self.garch_weight = garch_weight
        self.hmm_weight = hmm_weight
        self.consensus_threshold = consensus_threshold
        self.risk_management_mode = risk_management_mode
        self.enable_regime_overlay = enable_regime_overlay
        
        # Initialize signal generators
        self.technical_engine = BuySellSignalEngine()
        self.garch_generator = GARCHSignalGenerator()
        self.hmm_generator = HMMSignalGenerator()
        self.preprocessor = TimeSeriesPreprocessor()
        
        # Model performance tracking
        self.model_performance = {
            'technical': {'accuracy': 0.6, 'recent_signals': []},
            'garch': {'accuracy': 0.65, 'recent_signals': []},
            'hmm': {'accuracy': 0.7, 'recent_signals': []}
        }
        
        # Risk management parameters
        self.risk_params = self._initialize_risk_parameters()
        
    def _initialize_risk_parameters(self) -> Dict[str, any]:
        """Initialize risk management parameters based on mode"""
        
        if self.risk_management_mode == "conservative":
            return {
                'max_position_size': 0.05,
                'stop_loss_multiplier': 0.8,
                'profit_target_multiplier': 1.2,
                'volatility_scaling': 0.7,
                'correlation_limit': 0.7,
                'drawdown_limit': 0.15
            }
        elif self.risk_management_mode == "moderate":
            return {
                'max_position_size': 0.08,
                'stop_loss_multiplier': 1.0,
                'profit_target_multiplier': 1.5,
                'volatility_scaling': 1.0,
                'correlation_limit': 0.8,
                'drawdown_limit': 0.20
            }
        else:  # aggressive
            return {
                'max_position_size': 0.12,
                'stop_loss_multiplier': 1.3,
                'profit_target_multiplier': 2.0,
                'volatility_scaling': 1.3,
                'correlation_limit': 0.9,
                'drawdown_limit': 0.25
            }
    
    def generate_unified_signal(self,
                              symbol: str,
                              current_price: float,
                              historical_data: Optional[pd.DataFrame] = None,
                              heat_score: float = 0.5,
                              macd_result: Optional[Dict] = None,
                              bollinger_result: Optional[Dict] = None,
                              price_change_pct: float = 0.0,
                              volume_ratio: float = 1.0,
                              market_context: Optional[Dict] = None) -> UnifiedSignal:
        """
        Generate unified trading signal combining all models
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            historical_data: Historical OHLCV data
            heat_score: RAGHeat momentum score
            macd_result: MACD analysis results
            bollinger_result: Bollinger Bands analysis
            price_change_pct: Recent price change percentage
            volume_ratio: Volume ratio vs average
            market_context: Additional market context data
            
        Returns:
            UnifiedSignal with comprehensive analysis
        """
        try:
            # Step 1: Generate base technical signal
            base_signal = self.technical_engine.calculate_trading_signal(
                symbol=symbol,
                current_price=current_price,
                heat_score=heat_score,
                macd_result=macd_result or {},
                bollinger_result=bollinger_result or {},
                price_change_pct=price_change_pct,
                volume_ratio=volume_ratio,
                volatility=0.02  # Default, will be overridden by GARCH
            )
            
            # Step 2: Generate GARCH signal
            garch_signal = self.garch_generator.generate_signal(
                symbol=symbol,
                current_price=current_price,
                historical_data=historical_data,
                heat_score=heat_score,
                macd_result=macd_result,
                bollinger_result=bollinger_result,
                price_change_pct=price_change_pct,
                volume_ratio=volume_ratio
            )
            
            # Step 3: Generate HMM regime signal
            hmm_signal = self.hmm_generator.generate_signal(
                symbol=symbol,
                current_price=current_price,
                historical_data=historical_data,
                heat_score=heat_score,
                macd_result=macd_result,
                bollinger_result=bollinger_result,
                price_change_pct=price_change_pct,
                volume_ratio=volume_ratio
            )
            
            # Step 4: Calculate model consensus
            model_consensus = self._calculate_model_consensus(
                base_signal, garch_signal, hmm_signal
            )
            
            # Step 5: Generate unified signal
            unified_signal, unified_confidence = self._combine_signals(
                base_signal, garch_signal, hmm_signal, model_consensus
            )
            
            # Step 6: Apply regime overlay if enabled
            if self.enable_regime_overlay:
                unified_signal, unified_confidence = self._apply_regime_overlay(
                    unified_signal, unified_confidence, hmm_signal
                )
            
            # Step 7: Calculate risk-adjusted targets
            risk_adjusted_targets = self._calculate_risk_adjusted_targets(
                current_price, unified_signal, garch_signal, hmm_signal
            )
            
            # Step 8: Calculate portfolio allocation
            portfolio_allocation = self._calculate_portfolio_allocation(
                unified_signal, unified_confidence, garch_signal, hmm_signal
            )
            
            # Step 9: Generate execution strategy
            execution_strategy = self._generate_execution_strategy(
                unified_signal, unified_confidence, model_consensus, current_price
            )
            
            # Step 10: Compile model diagnostics
            model_diagnostics = self._compile_model_diagnostics(
                base_signal, garch_signal, hmm_signal, model_consensus
            )
            
            # Update performance tracking
            self._update_performance_tracking(symbol, unified_signal)
            
            return UnifiedSignal(
                symbol=symbol,
                base_technical_signal=base_signal,
                garch_signal=garch_signal,
                hmm_signal=hmm_signal,
                unified_signal=unified_signal,
                unified_confidence=unified_confidence,
                model_consensus=model_consensus,
                risk_adjusted_targets=risk_adjusted_targets,
                portfolio_allocation=portfolio_allocation,
                execution_strategy=execution_strategy,
                model_diagnostics=model_diagnostics,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error generating unified signal for {symbol}: {e}")
            return self._generate_fallback_unified_signal(symbol, current_price, heat_score)
    
    def _calculate_model_consensus(self,
                                 base_signal: TradingSignal,
                                 garch_signal: GARCHSignal,
                                 hmm_signal: HMMSignal) -> Dict[str, any]:
        """Calculate consensus between different models"""
        
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
        
        technical_score = signal_mapping[base_signal.signal]
        garch_score = signal_mapping[garch_signal.combined_signal]
        hmm_score = signal_mapping[hmm_signal.combined_signal]
        
        # Calculate consensus metrics
        scores = [technical_score, garch_score, hmm_score]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Direction consensus (all positive, all negative, or mixed)
        positive_count = sum(1 for score in scores if score > 0.1)
        negative_count = sum(1 for score in scores if score < -0.1)
        neutral_count = len(scores) - positive_count - negative_count
        
        if positive_count >= 2:
            direction_consensus = "BULLISH"
        elif negative_count >= 2:
            direction_consensus = "BEARISH"
        else:
            direction_consensus = "MIXED"
        
        # Strength consensus
        strong_signals = sum(1 for score in scores if abs(score) > 0.5)
        consensus_strength = strong_signals / len(scores)
        
        # Overall consensus score
        consensus_score = 1.0 - (std_score / 2.0)  # Lower std = higher consensus
        
        return {
            'direction_consensus': direction_consensus,
            'consensus_strength': consensus_strength,
            'consensus_score': consensus_score,
            'mean_signal_score': mean_score,
            'signal_std': std_score,
            'model_scores': {
                'technical': technical_score,
                'garch': garch_score,
                'hmm': hmm_score
            },
            'signal_distribution': {
                'bullish_models': positive_count,
                'bearish_models': negative_count,
                'neutral_models': neutral_count
            }
        }
    
    def _combine_signals(self,
                        base_signal: TradingSignal,
                        garch_signal: GARCHSignal,
                        hmm_signal: HMMSignal,
                        consensus: Dict[str, any]) -> Tuple[SignalStrength, float]:
        """Combine signals from all models using weighted approach"""
        
        signal_mapping = {
            SignalStrength.STRONG_SELL: -1.0,
            SignalStrength.SELL: -0.67,
            SignalStrength.WEAK_SELL: -0.33,
            SignalStrength.HOLD: 0.0,
            SignalStrength.WEAK_BUY: 0.33,
            SignalStrength.BUY: 0.67,
            SignalStrength.STRONG_BUY: 1.0
        }
        
        # Get individual scores and confidences
        technical_score = signal_mapping[base_signal.signal]
        garch_score = signal_mapping[garch_signal.combined_signal]
        hmm_score = signal_mapping[hmm_signal.combined_signal]
        
        technical_confidence = base_signal.confidence
        garch_confidence = garch_signal.combined_confidence
        hmm_confidence = hmm_signal.combined_confidence
        
        # Adjust weights based on model performance
        performance_weights = self._calculate_performance_weights()
        
        adjusted_technical_weight = self.technical_weight * performance_weights['technical']
        adjusted_garch_weight = self.garch_weight * performance_weights['garch']
        adjusted_hmm_weight = self.hmm_weight * performance_weights['hmm']
        
        # Normalize weights
        total_weight = adjusted_technical_weight + adjusted_garch_weight + adjusted_hmm_weight
        if total_weight > 0:
            adjusted_technical_weight /= total_weight
            adjusted_garch_weight /= total_weight
            adjusted_hmm_weight /= total_weight
        
        # Calculate weighted signal
        weighted_score = (
            technical_score * adjusted_technical_weight * technical_confidence +
            garch_score * adjusted_garch_weight * garch_confidence +
            hmm_score * adjusted_hmm_weight * hmm_confidence
        )
        
        # Calculate weighted confidence
        weighted_confidence = (
            technical_confidence * adjusted_technical_weight +
            garch_confidence * adjusted_garch_weight +
            hmm_confidence * adjusted_hmm_weight
        )
        
        # Apply consensus adjustment
        consensus_adjustment = consensus['consensus_score']
        final_confidence = weighted_confidence * consensus_adjustment
        
        # Apply consensus threshold
        if consensus['consensus_score'] < self.consensus_threshold:
            # Reduce signal strength when consensus is low
            weighted_score *= consensus['consensus_score'] / self.consensus_threshold
        
        # Convert back to signal strength
        unified_signal = self._score_to_signal_strength(weighted_score)
        
        return unified_signal, final_confidence
    
    def _calculate_performance_weights(self) -> Dict[str, float]:
        """Calculate performance-based weights for models"""
        
        base_weights = {
            'technical': self.technical_weight,
            'garch': self.garch_weight,
            'hmm': self.hmm_weight
        }
        
        # Adjust based on recent performance
        performance_multipliers = {}
        for model, perf in self.model_performance.items():
            accuracy = perf['accuracy']
            if accuracy > 0.7:
                performance_multipliers[model] = 1.2
            elif accuracy > 0.6:
                performance_multipliers[model] = 1.0
            else:
                performance_multipliers[model] = 0.8
        
        # Apply multipliers
        adjusted_weights = {}
        for model in base_weights:
            adjusted_weights[model] = base_weights[model] * performance_multipliers.get(model, 1.0)
        
        return adjusted_weights
    
    def _apply_regime_overlay(self,
                            signal: SignalStrength,
                            confidence: float,
                            hmm_signal: HMMSignal) -> Tuple[SignalStrength, float]:
        """Apply regime overlay to adjust signals based on market regime"""
        
        current_regime = hmm_signal.regime_detection.current_regime.lower()
        regime_confidence = hmm_signal.regime_detection.confidence
        
        signal_mapping = {
            SignalStrength.STRONG_SELL: -1.0,
            SignalStrength.SELL: -0.67,
            SignalStrength.WEAK_SELL: -0.33,
            SignalStrength.HOLD: 0.0,
            SignalStrength.WEAK_BUY: 0.33,
            SignalStrength.BUY: 0.67,
            SignalStrength.STRONG_BUY: 1.0
        }
        
        signal_score = signal_mapping[signal]
        
        # Regime-based adjustments
        if 'bull' in current_regime:
            # Boost buy signals, dampen sell signals
            if signal_score > 0:
                signal_score *= 1.1
                confidence *= 1.05
            else:
                signal_score *= 0.8
                confidence *= 0.9
                
        elif 'bear' in current_regime or 'crisis' in current_regime:
            # Boost sell signals, dampen buy signals
            if signal_score < 0:
                signal_score *= 1.1
                confidence *= 1.05
            else:
                signal_score *= 0.8
                confidence *= 0.9
                
        else:  # sideways or unknown
            # Dampen all signals in sideways markets
            signal_score *= 0.9
            confidence *= 0.95
        
        # Apply regime confidence
        confidence *= regime_confidence
        
        # Clamp values
        signal_score = max(-1.0, min(1.0, signal_score))
        confidence = max(0.0, min(1.0, confidence))
        
        adjusted_signal = self._score_to_signal_strength(signal_score)
        
        return adjusted_signal, confidence
    
    def _score_to_signal_strength(self, score: float) -> SignalStrength:
        """Convert numerical score to SignalStrength enum"""
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
    
    def _calculate_risk_adjusted_targets(self,
                                       current_price: float,
                                       signal: SignalStrength,
                                       garch_signal: GARCHSignal,
                                       hmm_signal: HMMSignal) -> Dict[str, float]:
        """Calculate comprehensive risk-adjusted targets"""
        
        # Get volatility estimates from both models
        garch_vol = garch_signal.garch_prediction.predicted_volatility
        regime_vol = self._extract_regime_volatility(hmm_signal.regime_detection)
        
        # Use weighted average of volatility estimates
        combined_vol = 0.6 * garch_vol + 0.4 * regime_vol
        
        # Get risk parameters
        vol_multiplier = combined_vol * self.risk_params['volatility_scaling']
        stop_multiplier = self.risk_params['stop_loss_multiplier']
        target_multiplier = self.risk_params['profit_target_multiplier']
        
        targets = {}
        
        if signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            base_target = 0.08 if signal == SignalStrength.STRONG_BUY else 0.05
            base_stop = 0.04 if signal == SignalStrength.STRONG_BUY else 0.03
            
            targets['price_target'] = current_price * (1 + base_target * target_multiplier * vol_multiplier)
            targets['stop_loss'] = current_price * (1 - base_stop * stop_multiplier * vol_multiplier)
            targets['take_profit_25'] = current_price * (1 + base_target * target_multiplier * vol_multiplier * 0.25)
            targets['take_profit_50'] = current_price * (1 + base_target * target_multiplier * vol_multiplier * 0.5)
            targets['take_profit_75'] = current_price * (1 + base_target * target_multiplier * vol_multiplier * 0.75)
            
        elif signal in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
            base_target = 0.08 if signal == SignalStrength.STRONG_SELL else 0.05
            base_stop = 0.04 if signal == SignalStrength.STRONG_SELL else 0.03
            
            targets['price_target'] = current_price * (1 - base_target * target_multiplier * vol_multiplier)
            targets['stop_loss'] = current_price * (1 + base_stop * stop_multiplier * vol_multiplier)
            targets['take_profit_25'] = current_price * (1 - base_target * target_multiplier * vol_multiplier * 0.25)
            targets['take_profit_50'] = current_price * (1 - base_target * target_multiplier * vol_multiplier * 0.5)
            targets['take_profit_75'] = current_price * (1 - base_target * target_multiplier * vol_multiplier * 0.75)
            
        else:
            # Neutral signals
            targets['stop_loss'] = current_price * (1 - 0.03 * vol_multiplier)
            targets['resistance'] = current_price * (1 + 0.02 * vol_multiplier)
            targets['support'] = current_price * (1 - 0.02 * vol_multiplier)
        
        # Add model-specific targets
        targets['garch_vol_estimate'] = garch_vol
        targets['regime_vol_estimate'] = regime_vol
        targets['combined_vol_estimate'] = combined_vol
        
        return targets
    
    def _extract_regime_volatility(self, regime_detection) -> float:
        """Extract volatility estimate from regime detection"""
        regime_name = regime_detection.current_regime.lower()
        
        if 'crisis' in regime_name:
            return 0.06
        elif 'bear' in regime_name:
            return 0.04
        elif 'bull' in regime_name:
            return 0.025
        else:  # sideways
            return 0.018
    
    def _calculate_portfolio_allocation(self,
                                      signal: SignalStrength,
                                      confidence: float,
                                      garch_signal: GARCHSignal,
                                      hmm_signal: HMMSignal) -> Dict[str, float]:
        """Calculate optimal portfolio allocation"""
        
        # Base allocation from risk parameters
        max_position = self.risk_params['max_position_size']
        
        # Adjust for signal strength
        signal_strength_map = {
            SignalStrength.STRONG_BUY: 1.0,
            SignalStrength.BUY: 0.8,
            SignalStrength.WEAK_BUY: 0.5,
            SignalStrength.HOLD: 0.0,
            SignalStrength.WEAK_SELL: 0.5,
            SignalStrength.SELL: 0.8,
            SignalStrength.STRONG_SELL: 1.0
        }
        
        signal_multiplier = signal_strength_map.get(signal, 0.0)
        
        # Adjust for confidence
        confidence_multiplier = confidence
        
        # Adjust for volatility (from GARCH)
        vol_adjustment = 1.0 / (1.0 + garch_signal.garch_prediction.predicted_volatility * 10)
        
        # Adjust for regime risk
        regime_risk = hmm_signal.risk_assessment['overall_risk_score']
        regime_adjustment = 1.0 - regime_risk * 0.5
        
        # Calculate final allocation
        position_size = (max_position * signal_multiplier * confidence_multiplier * 
                        vol_adjustment * regime_adjustment)
        
        return {
            'recommended_allocation': min(position_size, max_position),
            'max_allocation': max_position,
            'conservative_allocation': position_size * 0.7,
            'aggressive_allocation': min(position_size * 1.3, max_position),
            'signal_multiplier': signal_multiplier,
            'confidence_multiplier': confidence_multiplier,
            'volatility_adjustment': vol_adjustment,
            'regime_adjustment': regime_adjustment
        }
    
    def _generate_execution_strategy(self,
                                   signal: SignalStrength,
                                   confidence: float,
                                   consensus: Dict[str, any],
                                   current_price: float) -> Dict[str, any]:
        """Generate execution strategy based on signal characteristics"""
        
        strategy = {
            'entry_method': 'MARKET',
            'entry_timing': 'IMMEDIATE',
            'position_sizing': 'FULL',
            'execution_phases': 1,
            'time_horizon': 'MEDIUM'
        }
        
        # Adjust based on confidence
        if confidence < 0.4:
            strategy['entry_method'] = 'LIMIT'
            strategy['position_sizing'] = 'SCALED'
            strategy['execution_phases'] = 3
            
        elif confidence > 0.8:
            strategy['entry_timing'] = 'IMMEDIATE'
            strategy['position_sizing'] = 'FULL'
            
        # Adjust based on consensus
        if consensus['consensus_score'] < 0.5:
            strategy['position_sizing'] = 'CONSERVATIVE'
            strategy['execution_phases'] = 2
            
        # Adjust based on signal strength
        if signal in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]:
            strategy['time_horizon'] = 'SHORT'
            strategy['entry_timing'] = 'IMMEDIATE'
            
        elif signal in [SignalStrength.WEAK_BUY, SignalStrength.WEAK_SELL]:
            strategy['time_horizon'] = 'LONG'
            strategy['entry_method'] = 'LIMIT'
            
        # Add specific execution parameters
        strategy['limit_price_offset'] = 0.001 if strategy['entry_method'] == 'LIMIT' else 0
        strategy['max_slippage'] = 0.005
        strategy['execution_window_hours'] = 4 if strategy['entry_timing'] == 'IMMEDIATE' else 24
        
        return strategy
    
    def _compile_model_diagnostics(self,
                                 base_signal: TradingSignal,
                                 garch_signal: GARCHSignal,
                                 hmm_signal: HMMSignal,
                                 consensus: Dict[str, any]) -> Dict[str, any]:
        """Compile comprehensive model diagnostics"""
        
        return {
            'model_performances': self.model_performance,
            'signal_agreement': consensus,
            'garch_diagnostics': {
                'model_accuracy': garch_signal.garch_prediction.model_accuracy,
                'volatility_regime': garch_signal.garch_prediction.volatility_regime,
                'predicted_volatility': garch_signal.garch_prediction.predicted_volatility
            },
            'hmm_diagnostics': {
                'current_regime': hmm_signal.regime_detection.current_regime,
                'regime_confidence': hmm_signal.regime_detection.confidence,
                'regime_duration': hmm_signal.regime_detection.regime_duration,
                'risk_level': hmm_signal.regime_detection.risk_level
            },
            'technical_diagnostics': {
                'signal_strength': base_signal.signal.value,
                'confidence': base_signal.confidence,
                'reasoning_count': len(base_signal.reasoning)
            },
            'system_health': {
                'data_quality': 'GOOD',  # Would be calculated from preprocessing
                'model_sync': 'SYNCHRONIZED',
                'last_update': datetime.now().isoformat()
            }
        }
    
    def _update_performance_tracking(self, symbol: str, signal: SignalStrength):
        """Update performance tracking for models"""
        
        # This would be enhanced with actual performance calculation
        # For now, just track recent signals
        
        for model in self.model_performance:
            self.model_performance[model]['recent_signals'].append({
                'symbol': symbol,
                'signal': signal.value,
                'timestamp': datetime.now()
            })
            
            # Keep only recent signals (last 100)
            if len(self.model_performance[model]['recent_signals']) > 100:
                self.model_performance[model]['recent_signals'] = (
                    self.model_performance[model]['recent_signals'][-100:]
                )
    
    def _generate_fallback_unified_signal(self,
                                        symbol: str,
                                        current_price: float,
                                        heat_score: float) -> UnifiedSignal:
        """Generate fallback unified signal when full analysis fails"""
        
        # Create simple fallback signals
        if heat_score > 0.7:
            signal_strength = SignalStrength.BUY
            confidence = 0.5
        elif heat_score < 0.3:
            signal_strength = SignalStrength.SELL
            confidence = 0.5
        else:
            signal_strength = SignalStrength.HOLD
            confidence = 0.3
        
        # Create minimal signal objects
        base_signal = TradingSignal(
            symbol=symbol,
            signal=signal_strength,
            confidence=confidence,
            price_target=current_price * 1.02,
            stop_loss=current_price * 0.98,
            reasoning=["Fallback signal"],
            timestamp=datetime.now()
        )
        
        # Create fallback GARCH and HMM signals (simplified)
        from .garch_signal_generator import GARCHSignal
        from .hmm_signal_generator import HMMSignal
        
        # This would use actual fallback methods from the respective generators
        garch_signal = self.garch_generator._generate_fallback_signal(symbol, current_price, heat_score)
        hmm_signal = self.hmm_generator._generate_fallback_signal(symbol, current_price, heat_score)
        
        return UnifiedSignal(
            symbol=symbol,
            base_technical_signal=base_signal,
            garch_signal=garch_signal,
            hmm_signal=hmm_signal,
            unified_signal=signal_strength,
            unified_confidence=confidence,
            model_consensus={'consensus_score': 0.3, 'direction_consensus': 'MIXED'},
            risk_adjusted_targets={'stop_loss': current_price * 0.98},
            portfolio_allocation={'recommended_allocation': 0.02},
            execution_strategy={'entry_method': 'LIMIT'},
            model_diagnostics={'system_health': 'DEGRADED'},
            timestamp=datetime.now()
        )
    
    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive system status"""
        
        return {
            'weights': {
                'technical': self.technical_weight,
                'garch': self.garch_weight,
                'hmm': self.hmm_weight
            },
            'risk_management_mode': self.risk_management_mode,
            'consensus_threshold': self.consensus_threshold,
            'regime_overlay_enabled': self.enable_regime_overlay,
            'model_performance': self.model_performance,
            'risk_parameters': self.risk_params,
            'system_uptime': datetime.now().isoformat()
        }
    
    def update_weights(self,
                      technical_weight: Optional[float] = None,
                      garch_weight: Optional[float] = None,
                      hmm_weight: Optional[float] = None):
        """Update model weights"""
        
        if technical_weight is not None:
            self.technical_weight = technical_weight
        if garch_weight is not None:
            self.garch_weight = garch_weight
        if hmm_weight is not None:
            self.hmm_weight = hmm_weight
        
        # Validate weights sum to 1
        total = self.technical_weight + self.garch_weight + self.hmm_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
    
    def clear_caches(self):
        """Clear all model caches"""
        self.garch_generator.clear_cache()
        self.hmm_generator.clear_cache()