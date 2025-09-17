"""
w
GARCH-based Buy/Sell Signal Generator
Integrates GARCH volatility predictions with existing trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .garch_model import GARCHModel, GARCHPrediction
from .data_preprocessor import TimeSeriesPreprocessor, PreprocessedData
try:
    from ...signals.buy_sell_engine import BuySellSignalEngine, TradingSignal, SignalStrength
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from signals.buy_sell_engine import BuySellSignalEngine, TradingSignal, SignalStrength


@dataclass
class GARCHSignal:
    """Enhanced trading signal with GARCH-based volatility prediction"""
    symbol: str
    base_signal: TradingSignal
    garch_prediction: GARCHPrediction
    combined_signal: SignalStrength
    combined_confidence: float
    volatility_adjusted_targets: Dict[str, float]
    risk_metrics: Dict[str, float]
    signal_components: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'base_signal': self.base_signal.to_dict(),
            'garch_prediction': self.garch_prediction.to_dict(),
            'combined_signal': self.combined_signal.value,
            'combined_confidence': self.combined_confidence,
            'volatility_adjusted_targets': self.volatility_adjusted_targets,
            'risk_metrics': self.risk_metrics,
            'signal_components': self.signal_components,
            'timestamp': self.timestamp.isoformat()
        }


class GARCHSignalGenerator:
    """
    Advanced signal generator combining GARCH volatility predictions
    with existing technical analysis signals
    """
    
    def __init__(self,
                 garch_weight: float = 0.4,
                 technical_weight: float = 0.6,
                 volatility_penalty_threshold: float = 0.35,
                 volatility_boost_threshold: float = 0.15,
                 risk_free_rate: float = 0.04):
        """
        Initialize GARCH signal generator
        
        Args:
            garch_weight: Weight for GARCH-based signals (0-1)
            technical_weight: Weight for technical analysis signals (0-1)
            volatility_penalty_threshold: High volatility threshold for signal penalty
            volatility_boost_threshold: Low volatility threshold for signal boost
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        if abs(garch_weight + technical_weight - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
            
        self.garch_weight = garch_weight
        self.technical_weight = technical_weight
        self.volatility_penalty_threshold = volatility_penalty_threshold
        self.volatility_boost_threshold = volatility_boost_threshold
        self.risk_free_rate = risk_free_rate
        
        # Initialize components
        self.garch_model = GARCHModel()
        self.preprocessor = TimeSeriesPreprocessor()
        self.base_signal_engine = BuySellSignalEngine()
        
        # Cache for efficiency
        self._model_cache = {}
        self._data_cache = {}
    
    def generate_signal(self,
                       symbol: str,
                       current_price: float,
                       historical_data: Optional[pd.DataFrame] = None,
                       heat_score: float = 0.5,
                       macd_result: Optional[Dict] = None,
                       bollinger_result: Optional[Dict] = None,
                       price_change_pct: float = 0.0,
                       volume_ratio: float = 1.0,
                       forecast_horizon: int = 1) -> GARCHSignal:
        """
        Generate comprehensive trading signal with GARCH predictions
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            historical_data: Historical OHLCV data
            heat_score: RAGHeat momentum score
            macd_result: MACD analysis results
            bollinger_result: Bollinger Bands analysis
            price_change_pct: Recent price change percentage
            volume_ratio: Volume ratio vs average
            forecast_horizon: Days ahead to forecast
            
        Returns:
            GARCHSignal with combined analysis
        """
        try:
            # Step 1: Preprocess data and get GARCH prediction
            if historical_data is None:
                # Fetch recent data for GARCH analysis
                end_date = datetime.now()
                start_date = end_date - timedelta(days=252)  # 1 year
                preprocessed_data = self.preprocessor.preprocess(
                    symbol, start_date=start_date, end_date=end_date
                )
            else:
                preprocessed_data = self.preprocessor.preprocess(
                    symbol, data=historical_data
                )
            
            # Step 2: Generate GARCH prediction
            garch_prediction = self._get_garch_prediction(
                symbol, preprocessed_data, forecast_horizon
            )
            
            # Step 3: Generate base technical signal
            base_signal = self.base_signal_engine.calculate_trading_signal(
                symbol=symbol,
                current_price=current_price,
                heat_score=heat_score,
                macd_result=macd_result or {},
                bollinger_result=bollinger_result or {},
                price_change_pct=price_change_pct,
                volume_ratio=volume_ratio,
                volatility=garch_prediction.predicted_volatility / 100  # Convert to decimal
            )
            
            # Step 4: Combine signals
            combined_signal, combined_confidence, signal_components = self._combine_signals(
                base_signal, garch_prediction, preprocessed_data
            )
            
            # Step 5: Calculate volatility-adjusted targets
            vol_adjusted_targets = self._calculate_volatility_adjusted_targets(
                current_price, combined_signal, garch_prediction, base_signal
            )
            
            # Step 6: Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                preprocessed_data, garch_prediction, combined_signal
            )
            
            return GARCHSignal(
                symbol=symbol,
                base_signal=base_signal,
                garch_prediction=garch_prediction,
                combined_signal=combined_signal,
                combined_confidence=combined_confidence,
                volatility_adjusted_targets=vol_adjusted_targets,
                risk_metrics=risk_metrics,
                signal_components=signal_components,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error generating GARCH signal for {symbol}: {e}")
            # Return fallback signal
            return self._generate_fallback_signal(symbol, current_price, heat_score)
    
    def _get_garch_prediction(self,
                            symbol: str,
                            data: PreprocessedData,
                            horizon: int) -> GARCHPrediction:
        """Get GARCH prediction with caching"""
        cache_key = f"{symbol}_{len(data.returns)}_{horizon}"
        
        if cache_key in self._model_cache:
            model, last_update = self._model_cache[cache_key]
            # Reuse if updated within last hour
            if (datetime.now() - last_update).seconds < 3600:
                return model.predict(data.log_returns, horizon=horizon)
        
        # Fit new model
        model = GARCHModel(
            model_type='GARCH',
            p=1, q=1,
            volatility_threshold_low=self.volatility_boost_threshold,
            volatility_threshold_high=self.volatility_penalty_threshold
        )
        
        model.fit(data.log_returns)
        prediction = model.predict(data.log_returns, horizon=horizon)
        
        # Cache the model
        self._model_cache[cache_key] = (model, datetime.now())
        
        return prediction
    
    def _combine_signals(self,
                        base_signal: TradingSignal,
                        garch_prediction: GARCHPrediction,
                        data: PreprocessedData) -> Tuple[SignalStrength, float, Dict[str, float]]:
        """
        Combine base technical signal with GARCH prediction
        
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
        garch_score = garch_prediction.signal_strength
        
        # Volatility regime adjustments
        vol_adjustment = self._calculate_volatility_adjustment(
            garch_prediction.volatility_regime,
            garch_prediction.predicted_volatility,
            data.volatility.mean() if not data.volatility.empty else 0.2
        )
        
        # Model confidence weighting
        garch_confidence_weight = min(1.0, garch_prediction.model_accuracy * 2)
        base_confidence_weight = base_signal.confidence
        
        # Weighted combination
        technical_component = base_score * self.technical_weight * base_confidence_weight
        garch_component = garch_score * self.garch_weight * garch_confidence_weight
        volatility_component = vol_adjustment * 0.1  # Small additional factor
        
        combined_score = technical_component + garch_component + volatility_component
        
        # Combined confidence
        combined_confidence = (
            base_signal.confidence * self.technical_weight +
            garch_prediction.model_accuracy * self.garch_weight
        )
        
        # Apply volatility penalty/boost
        if garch_prediction.predicted_volatility > self.volatility_penalty_threshold:
            combined_confidence *= 0.8  # Reduce confidence in high volatility
            combined_score *= 0.9  # Dampen signals in high volatility
        elif garch_prediction.predicted_volatility < self.volatility_boost_threshold:
            combined_confidence *= 1.1  # Boost confidence in low volatility
            combined_score *= 1.05  # Amplify signals in low volatility
        
        # Clamp values
        combined_score = max(-1.0, min(1.0, combined_score))
        combined_confidence = max(0.0, min(1.0, combined_confidence))
        
        # Convert back to signal strength
        combined_signal = self._score_to_signal_strength(combined_score)
        
        signal_components = {
            'technical_component': technical_component,
            'garch_component': garch_component,
            'volatility_component': volatility_component,
            'base_score': base_score,
            'garch_score': garch_score,
            'volatility_adjustment': vol_adjustment,
            'final_score': combined_score
        }
        
        return combined_signal, combined_confidence, signal_components
    
    def _calculate_volatility_adjustment(self,
                                       vol_regime: str,
                                       predicted_vol: float,
                                       historical_avg_vol: float) -> float:
        """Calculate volatility-based signal adjustment"""
        
        vol_ratio = predicted_vol / historical_avg_vol if historical_avg_vol > 0 else 1.0
        
        if vol_regime == 'low':
            # Low volatility - slight positive bias (mean reversion opportunity)
            return 0.1 * (1 - vol_ratio) if vol_ratio < 0.8 else 0.05
        
        elif vol_regime == 'high':
            # High volatility - negative bias (risk aversion)
            return -0.15 * (vol_ratio - 1) if vol_ratio > 1.5 else -0.1
        
        else:  # medium
            # Medium volatility - neutral to slightly positive
            return 0.02
    
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
    
    def _calculate_volatility_adjusted_targets(self,
                                             current_price: float,
                                             signal: SignalStrength,
                                             garch_pred: GARCHPrediction,
                                             base_signal: TradingSignal) -> Dict[str, float]:
        """Calculate volatility-adjusted price targets and stop losses"""
        
        vol_multiplier = max(0.5, min(2.0, garch_pred.predicted_volatility / 0.2))
        
        targets = {}
        
        if signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            # Bullish targets
            base_target_pct = 0.08 if signal == SignalStrength.STRONG_BUY else 0.05
            base_stop_pct = 0.04 if signal == SignalStrength.STRONG_BUY else 0.03
            
            # Adjust for volatility
            target_pct = base_target_pct * vol_multiplier
            stop_pct = base_stop_pct * vol_multiplier
            
            targets['price_target'] = current_price * (1 + target_pct)
            targets['stop_loss'] = current_price * (1 - stop_pct)
            targets['take_profit_1'] = current_price * (1 + target_pct * 0.5)
            targets['take_profit_2'] = current_price * (1 + target_pct * 0.75)
            
        elif signal in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
            # Bearish targets (for short positions)
            base_target_pct = 0.08 if signal == SignalStrength.STRONG_SELL else 0.05
            base_stop_pct = 0.04 if signal == SignalStrength.STRONG_SELL else 0.03
            
            target_pct = base_target_pct * vol_multiplier
            stop_pct = base_stop_pct * vol_multiplier
            
            targets['price_target'] = current_price * (1 - target_pct)
            targets['stop_loss'] = current_price * (1 + stop_pct)
            targets['take_profit_1'] = current_price * (1 - target_pct * 0.5)
            targets['take_profit_2'] = current_price * (1 - target_pct * 0.75)
            
        else:
            # Neutral/weak signals - conservative targets
            stop_pct = 0.03 * vol_multiplier
            targets['stop_loss'] = current_price * (1 - stop_pct)
            
            if 'BUY' in signal.value:
                targets['price_target'] = current_price * (1 + 0.02 * vol_multiplier)
            elif 'SELL' in signal.value:
                targets['price_target'] = current_price * (1 - 0.02 * vol_multiplier)
        
        # Add GARCH-specific confidence intervals
        if garch_pred.confidence_interval:
            targets['garch_lower_bound'] = current_price * (1 + garch_pred.confidence_interval[0] / 100)
            targets['garch_upper_bound'] = current_price * (1 + garch_pred.confidence_interval[1] / 100)
        
        return targets
    
    def _calculate_risk_metrics(self,
                              data: PreprocessedData,
                              garch_pred: GARCHPrediction,
                              signal: SignalStrength) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        returns = data.returns.dropna()
        
        metrics = {}
        
        if len(returns) > 0:
            # Basic risk metrics
            metrics['historical_volatility'] = returns.std() * np.sqrt(252)  # Annualized
            metrics['predicted_volatility'] = garch_pred.predicted_volatility * np.sqrt(252)
            metrics['volatility_ratio'] = metrics['predicted_volatility'] / metrics['historical_volatility']
            
            # Sharpe ratio (assuming daily risk-free rate)
            daily_rf = self.risk_free_rate / 252
            excess_returns = returns - daily_rf
            metrics['sharpe_ratio'] = excess_returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Downside metrics
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                metrics['downside_deviation'] = negative_returns.std()
                metrics['max_drawdown'] = self._calculate_max_drawdown(data.raw_prices)
                metrics['var_95'] = returns.quantile(0.05)  # 5% VaR
                metrics['cvar_95'] = negative_returns[negative_returns <= metrics['var_95']].mean()
            else:
                metrics['downside_deviation'] = 0
                metrics['max_drawdown'] = 0
                metrics['var_95'] = 0
                metrics['cvar_95'] = 0
            
            # Signal-specific risk assessment
            signal_risk_multiplier = {
                SignalStrength.STRONG_BUY: 1.2,
                SignalStrength.BUY: 1.0,
                SignalStrength.WEAK_BUY: 0.7,
                SignalStrength.HOLD: 0.3,
                SignalStrength.WEAK_SELL: 0.7,
                SignalStrength.SELL: 1.0,
                SignalStrength.STRONG_SELL: 1.2
            }
            
            base_position_risk = 0.02  # 2% base position risk
            metrics['recommended_position_size'] = base_position_risk / (
                metrics['predicted_volatility'] * signal_risk_multiplier.get(signal, 1.0)
            )
            
            # Kelly criterion for position sizing
            win_rate = (returns > 0).mean()
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
            
            if avg_loss > 0:
                win_loss_ratio = avg_win / avg_loss
                kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
                metrics['kelly_fraction'] = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            else:
                metrics['kelly_fraction'] = 0.1  # Default conservative sizing
        
        else:
            # Fallback values
            metrics = {
                'historical_volatility': 0.3,
                'predicted_volatility': garch_pred.predicted_volatility * np.sqrt(252),
                'volatility_ratio': 1.0,
                'sharpe_ratio': 0.0,
                'downside_deviation': 0.2,
                'max_drawdown': 0.1,
                'var_95': -0.03,
                'cvar_95': -0.05,
                'recommended_position_size': 0.01,
                'kelly_fraction': 0.05
            }
        
        return metrics
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return abs(drawdown.min())
    
    def _generate_fallback_signal(self,
                                symbol: str,
                                current_price: float,
                                heat_score: float) -> GARCHSignal:
        """Generate fallback signal when GARCH fails"""
        
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
        
        fallback_garch_pred = GARCHPrediction(
            symbol=symbol,
            predicted_volatility=0.25,  # Default moderate volatility
            predicted_returns=0.0,
            confidence_interval=(0.15, 0.35),
            signal_strength=0.0,
            volatility_regime='medium',
            prediction_horizon=1,
            model_accuracy=0.0,
            timestamp=datetime.now()
        )
        
        return GARCHSignal(
            symbol=symbol,
            base_signal=fallback_base_signal,
            garch_prediction=fallback_garch_pred,
            combined_signal=signal_strength,
            combined_confidence=confidence,
            volatility_adjusted_targets={'stop_loss': current_price * 0.95},
            risk_metrics={'recommended_position_size': 0.01},
            signal_components={'fallback': True},
            timestamp=datetime.now()
        )