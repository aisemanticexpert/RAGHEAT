"""
Market Regime Detection System using Hidden Markov Models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .hmm_model import HMMMarketModel, HMMPrediction, MarketRegime
from .data_preprocessor import TimeSeriesPreprocessor


@dataclass
class RegimeDetectionResult:
    """Result of market regime detection"""
    symbol: str
    current_regime: str
    regime_probability: float
    regime_duration: int  # Days in current regime
    regime_strength: float  # How characteristic the current state is
    transition_probabilities: Dict[str, float]
    regime_forecast: Dict[str, float]  # Next period regime probabilities
    risk_level: str  # LOW, MEDIUM, HIGH
    recommended_action: str  # BUY, SELL, HOLD, REDUCE
    confidence: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'current_regime': self.current_regime,
            'regime_probability': self.regime_probability,
            'regime_duration': self.regime_duration,
            'regime_strength': self.regime_strength,
            'transition_probabilities': self.transition_probabilities,
            'regime_forecast': self.regime_forecast,
            'risk_level': self.risk_level,
            'recommended_action': self.recommended_action,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class MarketRegimeDetector:
    """
    Advanced Market Regime Detection System
    
    Uses Hidden Markov Models to identify market regimes and provide
    trading recommendations based on regime characteristics and transitions.
    """
    
    def __init__(self,
                 n_regimes: int = 4,
                 lookback_period: int = 252,  # 1 year
                 min_regime_duration: int = 5,
                 regime_names: Optional[List[str]] = None,
                 refit_frequency: int = 30):
        """
        Initialize Market Regime Detector
        
        Args:
            n_regimes: Number of market regimes to detect
            lookback_period: Days of history to use for regime detection
            min_regime_duration: Minimum days to consider a regime change
            regime_names: Custom names for regimes
            refit_frequency: Days between model refitting
        """
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.min_regime_duration = min_regime_duration
        self.refit_frequency = refit_frequency
        
        # Initialize HMM model
        self.hmm_model = HMMMarketModel(
            n_states=n_regimes,
            regime_names=regime_names
        )
        
        # Initialize preprocessor
        self.preprocessor = TimeSeriesPreprocessor()
        
        # Cache for regime history
        self.regime_history = {}
        self.last_detection = {}
        
    def detect_regime(self,
                     symbol: str,
                     data: Optional[pd.DataFrame] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> RegimeDetectionResult:
        """
        Detect current market regime for a symbol
        
        Args:
            symbol: Stock symbol
            data: Optional pre-loaded data
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            RegimeDetectionResult object
        """
        try:
            # Prepare data
            if data is None:
                if not start_date:
                    start_date = datetime.now() - timedelta(days=self.lookback_period + 100)
                if not end_date:
                    end_date = datetime.now()
                
                processed_data = self.preprocessor.preprocess(
                    symbol, start_date=start_date, end_date=end_date
                )
            else:
                processed_data = self.preprocessor.preprocess(symbol, data=data)
            
            # Extract features for HMM
            volume = processed_data.volume if processed_data.volume is not None else None
            features = self.hmm_model.prepare_features(
                processed_data.raw_prices,
                volume=volume,
                include_technical=True
            )
            
            # Fit HMM model
            self.hmm_model.fit(features)
            
            # Get regime prediction
            hmm_prediction = self.hmm_model.predict(features)
            
            # Analyze regime characteristics
            regime_analysis = self._analyze_current_regime(
                symbol, features, hmm_prediction
            )
            
            # Calculate regime duration and strength
            regime_duration = self._calculate_regime_duration(
                symbol, hmm_prediction.current_state
            )
            
            regime_strength = self._calculate_regime_strength(
                features, hmm_prediction
            )
            
            # Generate transition probabilities
            transition_probs = self._get_transition_probabilities(hmm_prediction)
            
            # Generate regime forecast
            regime_forecast = self._generate_regime_forecast(hmm_prediction)
            
            # Assess risk level
            risk_level = self._assess_risk_level(hmm_prediction, regime_analysis)
            
            # Generate recommendation
            recommended_action = self._generate_recommendation(
                hmm_prediction, regime_analysis, risk_level
            )
            
            # Store in cache
            self.last_detection[symbol] = {
                'regime': hmm_prediction.current_state,
                'timestamp': datetime.now()
            }
            
            return RegimeDetectionResult(
                symbol=symbol,
                current_regime=hmm_prediction.state_names[hmm_prediction.current_state],
                regime_probability=hmm_prediction.state_probabilities[hmm_prediction.current_state],
                regime_duration=regime_duration,
                regime_strength=regime_strength,
                transition_probabilities=transition_probs,
                regime_forecast=regime_forecast,
                risk_level=risk_level,
                recommended_action=recommended_action,
                confidence=hmm_prediction.confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Regime detection error for {symbol}: {e}")
            return self._generate_fallback_result(symbol)
    
    def _analyze_current_regime(self,
                              symbol: str,
                              features: pd.DataFrame,
                              prediction: HMMPrediction) -> Dict[str, any]:
        """Analyze characteristics of the current regime"""
        
        current_state = prediction.current_state
        regime_char = prediction.regime_characteristics.get(current_state, {})
        
        # Recent performance metrics
        recent_features = features.tail(20)
        
        analysis = {
            'mean_return': regime_char.get('mean_return', 0.0),
            'volatility': regime_char.get('volatility', 0.02),
            'frequency': regime_char.get('frequency', 0.33),
            'avg_duration': regime_char.get('avg_duration', 10),
            'recent_return': recent_features['returns'].mean() if 'returns' in recent_features else 0.0,
            'recent_volatility': recent_features['returns'].std() if 'returns' in recent_features else 0.02,
            'trend_strength': self._calculate_trend_strength(recent_features),
            'momentum': self._calculate_momentum(recent_features),
            'mean_reversion_signal': self._calculate_mean_reversion(recent_features),
            'regime_stability': regime_char.get('avg_duration', 10) / 20  # Normalized stability
        }
        
        return analysis
    
    def _calculate_trend_strength(self, features: pd.DataFrame) -> float:
        """Calculate trend strength from features"""
        if 'returns' not in features.columns or len(features) < 5:
            return 0.0
        
        # Linear regression slope of cumulative returns
        cumulative_returns = (1 + features['returns']).cumprod()
        x = np.arange(len(cumulative_returns))
        
        if len(x) > 1:
            slope = np.polyfit(x, cumulative_returns.values, 1)[0]
            return np.tanh(slope * 100)  # Normalize to [-1, 1]
        
        return 0.0
    
    def _calculate_momentum(self, features: pd.DataFrame) -> float:
        """Calculate momentum from features"""
        if 'returns' not in features.columns or len(features) < 10:
            return 0.0
        
        # Simple momentum: recent vs older returns
        recent_return = features['returns'].tail(5).mean()
        older_return = features['returns'].head(5).mean()
        
        momentum = (recent_return - older_return) / (abs(older_return) + 0.001)
        return np.tanh(momentum * 10)  # Normalize
    
    def _calculate_mean_reversion(self, features: pd.DataFrame) -> float:
        """Calculate mean reversion signal"""
        if 'returns' not in features.columns or len(features) < 20:
            return 0.0
        
        # Distance from moving average
        if 'sma_ratio_5_20' in features.columns:
            sma_ratio = features['sma_ratio_5_20'].iloc[-1]
            # Mean reversion signal: stronger when far from mean
            deviation = abs(sma_ratio - 1.0)
            return min(1.0, deviation * 5)  # Normalize to [0, 1]
        
        return 0.0
    
    def _calculate_regime_duration(self, symbol: str, current_state: int) -> int:
        """Calculate how long we've been in the current regime"""
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        
        # Add current state to history
        self.regime_history[symbol].append({
            'state': current_state,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(days=100)
        self.regime_history[symbol] = [
            h for h in self.regime_history[symbol] 
            if h['timestamp'] > cutoff
        ]
        
        # Count consecutive days in current state
        duration = 0
        for i in range(len(self.regime_history[symbol]) - 1, -1, -1):
            if self.regime_history[symbol][i]['state'] == current_state:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_regime_strength(self,
                                 features: pd.DataFrame,
                                 prediction: HMMPrediction) -> float:
        """Calculate how characteristic the current state is"""
        
        current_prob = prediction.state_probabilities[prediction.current_state]
        
        # Higher strength if:
        # 1. High probability of current state
        # 2. Recent data fits regime characteristics well
        
        prob_strength = current_prob
        
        # Characteristic fit strength
        current_state = prediction.current_state
        regime_char = prediction.regime_characteristics.get(current_state, {})
        
        if 'returns' in features.columns and len(features) > 0:
            recent_return = features['returns'].tail(10).mean()
            recent_vol = features['returns'].tail(10).std()
            
            expected_return = regime_char.get('mean_return', 0.0)
            expected_vol = regime_char.get('volatility', 0.02)
            
            # Calculate how well recent data matches regime
            return_fit = 1 - abs(recent_return - expected_return) / (abs(expected_return) + 0.01)
            vol_fit = 1 - abs(recent_vol - expected_vol) / (expected_vol + 0.01)
            
            char_strength = (return_fit + vol_fit) / 2
        else:
            char_strength = 0.5
        
        # Combine probability and characteristic fit
        total_strength = (0.6 * prob_strength + 0.4 * char_strength)
        return max(0.0, min(1.0, total_strength))
    
    def _get_transition_probabilities(self, prediction: HMMPrediction) -> Dict[str, float]:
        """Get transition probabilities to other regimes"""
        
        transition_matrix = self.hmm_model.get_transition_matrix()
        if transition_matrix is None:
            # Equal probabilities as fallback
            equal_prob = 1.0 / self.n_regimes
            return {name: equal_prob for name in prediction.state_names}
        
        current_state = prediction.current_state
        transition_probs = transition_matrix[current_state]
        
        return {
            prediction.state_names[i]: float(prob) 
            for i, prob in enumerate(transition_probs)
        }
    
    def _generate_regime_forecast(self, prediction: HMMPrediction) -> Dict[str, float]:
        """Generate forecast of regime probabilities for next period"""
        
        # Simple forecast: current transition probabilities
        return self._get_transition_probabilities(prediction)
    
    def _assess_risk_level(self,
                          prediction: HMMPrediction,
                          regime_analysis: Dict[str, any]) -> str:
        """Assess overall risk level"""
        
        # Risk factors
        volatility_risk = regime_analysis['volatility']
        recent_volatility_risk = regime_analysis['recent_volatility']
        regime_uncertainty = 1 - prediction.confidence
        
        # Weighted risk score
        risk_score = (
            0.4 * min(1.0, volatility_risk / 0.05) +  # Normalize volatility
            0.3 * min(1.0, recent_volatility_risk / 0.05) +
            0.3 * regime_uncertainty
        )
        
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _generate_recommendation(self,
                               prediction: HMMPrediction,
                               regime_analysis: Dict[str, any],
                               risk_level: str) -> str:
        """Generate trading recommendation based on regime analysis"""
        
        current_regime_name = prediction.state_names[prediction.current_state]
        expected_return = regime_analysis['mean_return']
        trend_strength = regime_analysis['trend_strength']
        momentum = regime_analysis['momentum']
        regime_stability = regime_analysis['regime_stability']
        
        # Base recommendation from regime characteristics
        if "Bull" in current_regime_name or expected_return > 0.01:
            base_action = "BUY"
        elif "Bear" in current_regime_name or expected_return < -0.01:
            base_action = "SELL"
        elif "Crisis" in current_regime_name:
            base_action = "REDUCE"
        else:
            base_action = "HOLD"
        
        # Adjust for trend and momentum
        if trend_strength > 0.3 and momentum > 0.2:
            if base_action in ["HOLD", "SELL"]:
                base_action = "BUY"
        elif trend_strength < -0.3 and momentum < -0.2:
            if base_action in ["HOLD", "BUY"]:
                base_action = "SELL"
        
        # Adjust for risk level
        if risk_level == "HIGH":
            if base_action == "BUY":
                base_action = "HOLD"
            elif base_action == "SELL":
                base_action = "REDUCE"
        
        # Adjust for regime stability
        if regime_stability < 0.3:  # Unstable regime
            if base_action in ["BUY", "SELL"]:
                base_action = "HOLD"
        
        return base_action
    
    def _generate_fallback_result(self, symbol: str) -> RegimeDetectionResult:
        """Generate fallback result when regime detection fails"""
        
        return RegimeDetectionResult(
            symbol=symbol,
            current_regime="Unknown",
            regime_probability=0.33,
            regime_duration=1,
            regime_strength=0.0,
            transition_probabilities={f"State_{i}": 0.33 for i in range(3)},
            regime_forecast={f"State_{i}": 0.33 for i in range(3)},
            risk_level="MEDIUM",
            recommended_action="HOLD",
            confidence=0.0,
            timestamp=datetime.now()
        )
    
    def batch_detect_regimes(self,
                           symbols: List[str],
                           date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, RegimeDetectionResult]:
        """
        Detect regimes for multiple symbols
        
        Args:
            symbols: List of stock symbols
            date_range: Optional (start_date, end_date) tuple
            
        Returns:
            Dictionary mapping symbols to RegimeDetectionResult
        """
        results = {}
        
        start_date, end_date = date_range if date_range else (None, None)
        
        for symbol in symbols:
            try:
                result = self.detect_regime(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                results[symbol] = result
            except Exception as e:
                print(f"Failed to detect regime for {symbol}: {e}")
                results[symbol] = self._generate_fallback_result(symbol)
        
        return results
    
    def get_regime_summary(self) -> Dict[str, any]:
        """Get summary of the regime detection system"""
        
        return {
            "n_regimes": self.n_regimes,
            "lookback_period": self.lookback_period,
            "regime_names": self.hmm_model.regime_names,
            "model_accuracy": self.hmm_model.model_accuracy,
            "last_fit": self.hmm_model.last_fit_date.isoformat() if self.hmm_model.last_fit_date else None,
            "tracked_symbols": list(self.last_detection.keys()),
            "regime_characteristics": self.hmm_model.get_regime_summary()
        }
    
    def clear_cache(self):
        """Clear regime detection cache"""
        self.regime_history.clear()
        self.last_detection.clear()
    
    def update_regime_names(self, new_names: List[str]):
        """Update regime names"""
        if len(new_names) == self.n_regimes:
            self.hmm_model.regime_names = new_names
        else:
            raise ValueError(f"Must provide {self.n_regimes} regime names")