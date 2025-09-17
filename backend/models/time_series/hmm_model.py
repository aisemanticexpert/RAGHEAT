"""
Hidden Markov Model (HMM) Implementation for Market Regime Detection and Trading Signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

HMM_AVAILABLE = False
try:
    from hmmlearn import hmm
    from hmmlearn.hmm import GaussianHMM, MultinomialHMM
    HMM_AVAILABLE = True
    print("✅ HMMLearn package loaded successfully")
except ImportError as e:
    print(f"Warning: hmmlearn package not available: {e}. HMM functionality limited.")
    HMM_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class HMMPrediction:
    """HMM model prediction result"""
    symbol: str
    current_state: int
    state_probabilities: List[float]
    predicted_state: int
    state_names: List[str]
    regime_characteristics: Dict[str, any]
    signal_strength: float  # -1 to 1, where 1 = strong buy, -1 = strong sell
    confidence: float
    prediction_horizon: int
    model_accuracy: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'current_state': self.current_state,
            'state_probabilities': self.state_probabilities,
            'predicted_state': self.predicted_state,
            'state_names': self.state_names,
            'regime_characteristics': self.regime_characteristics,
            'signal_strength': self.signal_strength,
            'confidence': self.confidence,
            'prediction_horizon': self.prediction_horizon,
            'model_accuracy': self.model_accuracy,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class MarketRegime:
    """Market regime characteristics"""
    regime_id: int
    name: str
    mean_return: float
    volatility: float
    persistence: float  # Average time in this regime
    transition_probabilities: Dict[int, float]
    characteristics: Dict[str, any]


class HMMMarketModel:
    """
    Hidden Markov Model for Market Regime Detection and Prediction
    
    This model identifies hidden market states (regimes) such as:
    - Bull Market (high returns, moderate volatility)
    - Bear Market (negative returns, high volatility)  
    - Sideways Market (low returns, low volatility)
    - Crisis Market (very negative returns, very high volatility)
    """
    
    def __init__(self, 
                 n_states: int = 3,
                 covariance_type: str = "full",
                 n_iter: int = 100,
                 random_state: int = 42,
                 regime_names: Optional[List[str]] = None):
        """
        Initialize HMM model
        
        Args:
            n_states: Number of hidden states (market regimes)
            covariance_type: Type of covariance parameters ('full', 'diag', 'tied', 'spherical')
            n_iter: Maximum number of iterations for EM algorithm
            random_state: Random seed for reproducibility
            regime_names: Optional names for market regimes
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn package required for HMM functionality. Run: pip install hmmlearn")
            
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Default regime names
        if regime_names is None:
            if n_states == 3:
                self.regime_names = ["Bear Market", "Sideways Market", "Bull Market"]
            elif n_states == 4:
                self.regime_names = ["Crisis", "Bear Market", "Sideways Market", "Bull Market"]
            else:
                self.regime_names = [f"Regime_{i}" for i in range(n_states)]
        else:
            self.regime_names = regime_names
            
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.regime_characteristics = {}
        self.model_accuracy = 0.0
        self.last_fit_date = None
        
    def prepare_features(self, 
                        prices: pd.Series,
                        volume: Optional[pd.Series] = None,
                        include_technical: bool = True) -> pd.DataFrame:
        """
        Prepare features for HMM training
        
        Args:
            prices: Price series
            volume: Volume series (optional)
            include_technical: Whether to include technical indicators
            
        Returns:
            Feature DataFrame
        """
        features = pd.DataFrame(index=prices.index)
        
        # Basic return features
        returns = prices.pct_change().dropna()
        features['returns'] = returns
        features['log_returns'] = np.log(prices / prices.shift(1))
        features['abs_returns'] = np.abs(returns)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'mean_return_{window}'] = returns.rolling(window).mean()
            features[f'volatility_{window}'] = returns.rolling(window).std()
            features[f'skewness_{window}'] = returns.rolling(window).skew()
            features[f'kurtosis_{window}'] = returns.rolling(window).kurt()
        
        # Price momentum features
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = prices / prices.shift(period) - 1
            features[f'rsi_{period}'] = self._calculate_rsi(prices, period)
        
        # Technical indicators
        if include_technical:
            # Moving averages
            features['sma_ratio_5_20'] = prices.rolling(5).mean() / prices.rolling(20).mean()
            features['sma_ratio_10_50'] = prices.rolling(10).mean() / prices.rolling(50).mean()
            
            # Bollinger Bands
            sma_20 = prices.rolling(20).mean()
            std_20 = prices.rolling(20).std()
            features['bb_position'] = (prices - (sma_20 - 2*std_20)) / (4*std_20)
            features['bb_width'] = 4*std_20 / sma_20
            
            # MACD
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Volume features (if available)
        if volume is not None:
            features['volume_ma_ratio'] = volume / volume.rolling(20).mean()
            features['price_volume'] = returns * np.log(volume / volume.shift(1))
            
        # Market microstructure features
        features['high_low_ratio'] = (prices.rolling(5).max() - prices.rolling(5).min()) / prices
        features['close_to_high'] = prices / prices.rolling(5).max()
        features['close_to_low'] = prices / prices.rolling(5).min()
        
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        return features.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def fit(self, 
            features: pd.DataFrame,
            refit_frequency: int = 30) -> 'HMMMarketModel':
        """
        Fit HMM model to market data
        
        Args:
            features: Feature DataFrame
            refit_frequency: Days between model refitting
            
        Returns:
            Self for method chaining
        """
        if len(features) < 50:
            raise ValueError("Insufficient data: need at least 50 observations")
        
        # Check if we need to refit
        if (self.model is not None and 
            self.last_fit_date is not None and 
            (datetime.now() - self.last_fit_date).days < refit_frequency):
            return self
        
        try:
            # Prepare and scale features
            feature_array = features[self.feature_names].values
            feature_array_scaled = self.scaler.fit_transform(feature_array)
            
            # Initialize and fit HMM model
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
                verbose=False
            )
            
            self.model.fit(feature_array_scaled)
            self.last_fit_date = datetime.now()
            
            # Analyze regime characteristics
            self._analyze_regimes(features, feature_array_scaled)
            
            # Calculate model accuracy
            self._calculate_model_accuracy(features, feature_array_scaled)
            
        except Exception as e:
            print(f"HMM fitting error: {e}")
            # Create a simple fallback model
            self._create_fallback_model(features)
        
        return self
    
    def _analyze_regimes(self, features: pd.DataFrame, feature_array_scaled: np.ndarray):
        """Analyze characteristics of each regime"""
        if self.model is None:
            return
            
        try:
            # Get state sequence
            states = self.model.predict(feature_array_scaled)
            
            # Calculate regime characteristics
            self.regime_characteristics = {}
            
            for state in range(self.n_states):
                state_mask = states == state
                state_features = features[state_mask]
                
                if len(state_features) > 0:
                    regime_char = {
                        'mean_return': state_features['returns'].mean(),
                        'volatility': state_features['returns'].std(),
                        'frequency': state_mask.sum() / len(states),
                        'avg_duration': self._calculate_average_duration(states, state),
                        'skewness': state_features['returns'].skew() if len(state_features) > 2 else 0,
                        'kurtosis': state_features['returns'].kurt() if len(state_features) > 2 else 0,
                        'max_drawdown': self._calculate_regime_drawdown(state_features),
                        'volatility_regime': self._classify_volatility_regime(state_features['returns'].std())
                    }
                    
                    self.regime_characteristics[state] = regime_char
            
            # Assign regime names based on characteristics
            self._assign_regime_names()
            
        except Exception as e:
            print(f"Regime analysis error: {e}")
    
    def _calculate_average_duration(self, states: np.ndarray, target_state: int) -> float:
        """Calculate average duration in a specific state"""
        durations = []
        current_duration = 0
        
        for state in states:
            if state == target_state:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def _calculate_regime_drawdown(self, regime_features: pd.DataFrame) -> float:
        """Calculate maximum drawdown for a regime"""
        if 'returns' not in regime_features.columns or len(regime_features) < 2:
            return 0.0
        
        cumulative = (1 + regime_features['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility < 0.01:
            return 'low'
        elif volatility < 0.03:
            return 'medium'
        else:
            return 'high'
    
    def _assign_regime_names(self):
        """Assign meaningful names to regimes based on their characteristics"""
        if not self.regime_characteristics:
            return
        
        # Sort states by mean return
        sorted_states = sorted(self.regime_characteristics.keys(),
                             key=lambda x: self.regime_characteristics[x]['mean_return'])
        
        if self.n_states == 3:
            name_mapping = {
                sorted_states[0]: "Bear Market",
                sorted_states[1]: "Sideways Market", 
                sorted_states[2]: "Bull Market"
            }
        elif self.n_states == 4:
            # Check for crisis regime (very negative returns + high volatility)
            crisis_candidates = []
            for state in sorted_states:
                char = self.regime_characteristics[state]
                if char['mean_return'] < -0.02 and char['volatility'] > 0.04:
                    crisis_candidates.append(state)
            
            if crisis_candidates:
                crisis_state = min(crisis_candidates, 
                                 key=lambda x: self.regime_characteristics[x]['mean_return'])
                remaining_states = [s for s in sorted_states if s != crisis_state]
                
                name_mapping = {
                    crisis_state: "Crisis",
                    remaining_states[0]: "Bear Market",
                    remaining_states[1]: "Sideways Market",
                    remaining_states[2]: "Bull Market"
                }
            else:
                name_mapping = {
                    sorted_states[0]: "Deep Bear",
                    sorted_states[1]: "Bear Market",
                    sorted_states[2]: "Sideways Market",
                    sorted_states[3]: "Bull Market"
                }
        else:
            # Generic naming for other numbers of states
            name_mapping = {state: f"Regime_{i}" for i, state in enumerate(sorted_states)}
        
        # Update regime names
        self.regime_names = [name_mapping.get(i, f"State_{i}") for i in range(self.n_states)]
    
    def _calculate_model_accuracy(self, features: pd.DataFrame, feature_array_scaled: np.ndarray):
        """Calculate model accuracy using cross-validation approach"""
        if self.model is None:
            return
            
        try:
            # Use rolling window validation
            window_size = min(100, len(features) // 3)
            accuracies = []
            
            for i in range(window_size, len(features) - 10, 10):
                # Train on window
                train_data = feature_array_scaled[i-window_size:i]
                test_data = feature_array_scaled[i:i+10]
                
                # Temporary model for validation
                temp_model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=50,
                    random_state=self.random_state,
                    verbose=False
                )
                
                temp_model.fit(train_data)
                
                # Calculate log-likelihood as accuracy measure
                log_likelihood = temp_model.score(test_data)
                accuracies.append(log_likelihood)
            
            # Normalize accuracy to 0-1 range
            if accuracies:
                mean_accuracy = np.mean(accuracies)
                # Convert log-likelihood to probability-like score
                self.model_accuracy = max(0.0, min(1.0, (mean_accuracy + 10) / 20))
            else:
                self.model_accuracy = 0.5
                
        except Exception as e:
            print(f"Accuracy calculation error: {e}")
            self.model_accuracy = 0.5
    
    def _create_fallback_model(self, features: pd.DataFrame):
        """Create simple fallback model when HMM fails"""
        print("Creating fallback model...")
        
        # Use simple clustering as fallback
        returns = features['returns'].values.reshape(-1, 1)
        
        # Use KMeans clustering
        kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(returns)
        
        # Create simple transition matrix
        self.fallback_clusters = kmeans
        self.fallback_labels = cluster_labels
        self.model_accuracy = 0.3  # Lower accuracy for fallback
    
    def predict(self, 
                features: pd.DataFrame,
                horizon: int = 1) -> HMMPrediction:
        """
        Predict market regime and generate trading signal
        
        Args:
            features: Recent feature data
            horizon: Prediction horizon (currently supports 1)
            
        Returns:
            HMMPrediction object
        """
        if self.model is None and not hasattr(self, 'fallback_clusters'):
            raise ValueError("Model not fitted. Call fit() first.")
        
        symbol = features.index.name if hasattr(features.index, 'name') and features.index.name else 'UNKNOWN'
        
        try:
            # Use last few observations for prediction
            recent_features = features.tail(min(20, len(features)))
            feature_array = recent_features[self.feature_names].values
            feature_array_scaled = self.scaler.transform(feature_array)
            
            if self.model is not None:
                # HMM model prediction
                current_state = self.model.predict(feature_array_scaled)[-1]
                state_probs = self.model.predict_proba(feature_array_scaled)[-1]
                
                # Simple next-state prediction (most likely transition)
                transition_matrix = self.model.transmat_
                next_state_probs = transition_matrix[current_state]
                predicted_state = np.argmax(next_state_probs)
                
            else:
                # Fallback clustering prediction
                current_state = self.fallback_clusters.predict(
                    feature_array[-1].reshape(1, -1)
                )[0]
                state_probs = [0.33] * self.n_states  # Equal probabilities
                state_probs[current_state] = 0.4  # Slightly higher for current
                predicted_state = current_state
            
            # Generate trading signal based on regime
            signal_strength, confidence = self._generate_regime_signal(
                current_state, predicted_state, recent_features
            )
            
            return HMMPrediction(
                symbol=symbol,
                current_state=current_state,
                state_probabilities=state_probs.tolist(),
                predicted_state=predicted_state,
                state_names=self.regime_names,
                regime_characteristics=self.regime_characteristics,
                signal_strength=signal_strength,
                confidence=confidence,
                prediction_horizon=horizon,
                model_accuracy=self.model_accuracy,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"HMM prediction error: {e}")
            # Return neutral prediction
            return HMMPrediction(
                symbol=symbol,
                current_state=1,  # Assume neutral state
                state_probabilities=[0.33] * self.n_states,
                predicted_state=1,
                state_names=self.regime_names,
                regime_characteristics=self.regime_characteristics,
                signal_strength=0.0,
                confidence=0.0,
                prediction_horizon=horizon,
                model_accuracy=0.0,
                timestamp=datetime.now()
            )
    
    def _generate_regime_signal(self, 
                              current_state: int,
                              predicted_state: int,
                              recent_features: pd.DataFrame) -> Tuple[float, float]:
        """
        Generate trading signal based on market regime
        
        Returns:
            Tuple of (signal_strength, confidence)
        """
        if not self.regime_characteristics:
            return 0.0, 0.0
        
        current_regime = self.regime_characteristics.get(current_state, {})
        predicted_regime = self.regime_characteristics.get(predicted_state, {})
        
        # Base signal from current regime characteristics
        current_return = current_regime.get('mean_return', 0.0)
        current_vol = current_regime.get('volatility', 0.02)
        
        # Signal strength based on expected returns and volatility
        base_signal = np.tanh(current_return / max(current_vol, 0.01))  # Risk-adjusted return
        
        # Adjust for regime transition
        transition_signal = 0.0
        if predicted_state != current_state:
            predicted_return = predicted_regime.get('mean_return', 0.0)
            if predicted_return > current_return:
                transition_signal = 0.3  # Positive transition
            elif predicted_return < current_return:
                transition_signal = -0.3  # Negative transition
        
        # Momentum signal from recent performance
        momentum_signal = 0.0
        if 'returns' in recent_features.columns and len(recent_features) > 5:
            recent_momentum = recent_features['returns'].tail(5).mean()
            momentum_signal = np.tanh(recent_momentum / 0.01)  # Normalize momentum
        
        # Combine signals
        total_signal = (0.5 * base_signal + 
                       0.3 * transition_signal + 
                       0.2 * momentum_signal)
        
        # Calculate confidence based on regime persistence and model accuracy
        regime_persistence = current_regime.get('avg_duration', 1.0)
        confidence = min(1.0, self.model_accuracy * (1 + np.log(regime_persistence + 1) / 5))
        
        # Clamp signal to [-1, 1] range
        total_signal = max(-1.0, min(1.0, total_signal))
        
        return total_signal, confidence
    
    def get_regime_summary(self) -> Dict[str, any]:
        """Get summary of all market regimes"""
        if not self.regime_characteristics:
            return {"error": "No regime characteristics available"}
        
        summary = {
            "n_states": self.n_states,
            "regime_names": self.regime_names,
            "model_accuracy": self.model_accuracy,
            "last_fit": self.last_fit_date.isoformat() if self.last_fit_date else None,
            "regimes": {}
        }
        
        for state, char in self.regime_characteristics.items():
            regime_name = self.regime_names[state] if state < len(self.regime_names) else f"State_{state}"
            summary["regimes"][regime_name] = {
                "state_id": state,
                "mean_return": char['mean_return'],
                "volatility": char['volatility'],
                "frequency": char['frequency'],
                "avg_duration": char['avg_duration'],
                "volatility_regime": char.get('volatility_regime', 'unknown'),
                "max_drawdown": char.get('max_drawdown', 0.0)
            }
        
        return summary
    
    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Get state transition matrix"""
        if self.model is not None:
            return self.model.transmat_
        return None
    
    def update_model(self, new_features: pd.DataFrame) -> 'HMMMarketModel':
        """
        Update model with new data (online learning approach)
        
        Args:
            new_features: New feature data
            
        Returns:
            Self for method chaining
        """
        if self.model is None:
            return self.fit(new_features)
        
        # For simplicity, refit with extended data
        # In production, could implement more sophisticated online updates
        try:
            self.fit(new_features)
        except Exception as e:
            print(f"Model update error: {e}")
        
        return self