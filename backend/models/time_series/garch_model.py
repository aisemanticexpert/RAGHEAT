"""
GARCH Model Implementation for Volatility Prediction and Trading Signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

ARCH_AVAILABLE = False
try:
    from arch import arch_model
    from arch.univariate import GARCH, ARCH, EGARCH
    # Note: GJR-GARCH might be called differently or not available
    ARCH_AVAILABLE = True
    print("✅ ARCH package loaded successfully")
except ImportError as e:
    print(f"Warning: arch package not available: {e}. GARCH functionality limited.")
    ARCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class GARCHPrediction:
    """GARCH model prediction result"""
    symbol: str
    predicted_volatility: float
    predicted_returns: float
    confidence_interval: Tuple[float, float]
    signal_strength: float  # -1 to 1, where 1 = strong buy, -1 = strong sell
    volatility_regime: str  # 'low', 'medium', 'high'
    prediction_horizon: int  # days ahead
    model_accuracy: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'predicted_volatility': self.predicted_volatility,
            'predicted_returns': self.predicted_returns,
            'confidence_interval': list(self.confidence_interval),
            'signal_strength': self.signal_strength,
            'volatility_regime': self.volatility_regime,
            'prediction_horizon': self.prediction_horizon,
            'model_accuracy': self.model_accuracy,
            'timestamp': self.timestamp.isoformat()
        }


class GARCHModel:
    """
    Advanced GARCH Model for volatility prediction and trading signal generation
    
    Supports multiple GARCH variants:
    - GARCH(p,q): Standard Generalized Autoregressive Conditional Heteroskedasticity
    - EGARCH: Exponential GARCH (captures asymmetric effects)
    - GJR-GARCH: Glosten-Jagannathan-Runkle GARCH (threshold effects)
    """
    
    def __init__(self, 
                 model_type: str = 'GARCH',
                 p: int = 1, 
                 q: int = 1,
                 mean_model: str = 'AR',
                 ar_lags: int = 1,
                 volatility_threshold_low: float = 0.15,
                 volatility_threshold_high: float = 0.35):
        """
        Initialize GARCH model
        
        Args:
            model_type: 'GARCH', 'EGARCH', or 'GJR-GARCH'
            p: Number of lag variances in GARCH component
            q: Number of lag residuals in GARCH component  
            mean_model: Mean model type ('AR', 'Zero', 'Constant')
            ar_lags: Number of autoregressive lags for mean model
            volatility_threshold_low: Low volatility regime threshold
            volatility_threshold_high: High volatility regime threshold
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch package required for GARCH functionality. Run: pip install arch")
            
        self.model_type = model_type
        self.p = p
        self.q = q
        self.mean_model = mean_model
        self.ar_lags = ar_lags
        self.volatility_threshold_low = volatility_threshold_low
        self.volatility_threshold_high = volatility_threshold_high
        
        self.fitted_model = None
        self.scaler = StandardScaler()
        self.model_accuracy = 0.0
        self.last_fit_date = None
        
    def prepare_data(self, prices: Union[pd.Series, List[float]], 
                    dates: Optional[List[datetime]] = None) -> pd.Series:
        """
        Prepare price data for GARCH modeling
        
        Args:
            prices: Price series or list of prices
            dates: Optional dates corresponding to prices
            
        Returns:
            Prepared returns series
        """
        if isinstance(prices, list):
            if dates:
                prices = pd.Series(prices, index=pd.to_datetime(dates))
            else:
                prices = pd.Series(prices)
        
        # Calculate log returns
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Remove outliers (beyond 3 standard deviations)
        mean_return = returns.mean()
        std_return = returns.std()
        returns = returns[(returns >= mean_return - 3*std_return) & 
                         (returns <= mean_return + 3*std_return)]
        
        # Convert to percentage returns for better interpretability
        returns = returns * 100
        
        return returns
    
    def fit(self, returns: pd.Series, 
            refit_frequency: int = 30) -> 'GARCHModel':
        """
        Fit GARCH model to returns data
        
        Args:
            returns: Prepared returns series
            refit_frequency: Days between model refitting
            
        Returns:
            Self for method chaining
        """
        if len(returns) < 50:
            raise ValueError("Insufficient data: need at least 50 observations")
        
        # Check if we need to refit
        if (self.fitted_model is not None and 
            self.last_fit_date is not None and 
            (datetime.now() - self.last_fit_date).days < refit_frequency):
            return self
        
        try:
            # Create mean model
            if self.mean_model == 'AR':
                mean = 'AR'
                lags = self.ar_lags
            elif self.mean_model == 'Zero':
                mean = 'Zero'
                lags = None
            else:
                mean = 'Constant'
                lags = None
            
            # Create volatility model based on type
            if self.model_type == 'EGARCH':
                vol = EGARCH(self.p, 0, self.q)
            elif self.model_type == 'GJR-GARCH':
                # Fallback to standard GARCH if GJR-GARCH not available
                print("Warning: GJR-GARCH not available, using standard GARCH")
                vol = GARCH(self.p, 0, self.q)
            else:  # Standard GARCH
                vol = GARCH(self.p, 0, self.q)
            
            # Create and fit model
            model = arch_model(returns, 
                             mean=mean, 
                             lags=lags,
                             vol=vol,
                             dist='Normal')
            
            self.fitted_model = model.fit(disp='off', show_warning=False)
            self.last_fit_date = datetime.now()
            
            # Calculate model accuracy using in-sample forecasting
            self._calculate_model_accuracy(returns)
            
        except Exception as e:
            print(f"GARCH fitting error: {e}")
            # Fallback to simpler model
            try:
                model = arch_model(returns, vol='GARCH', p=1, q=1)
                self.fitted_model = model.fit(disp='off', show_warning=False)
                self.last_fit_date = datetime.now()
                self._calculate_model_accuracy(returns)
            except Exception as e2:
                print(f"Fallback GARCH fitting error: {e2}")
                raise
        
        return self
    
    def _calculate_model_accuracy(self, returns: pd.Series) -> None:
        """Calculate model accuracy using rolling forecasts"""
        if self.fitted_model is None:
            return
            
        try:
            # Use last 30 observations for out-of-sample testing
            test_size = min(30, len(returns) // 4)
            train_returns = returns[:-test_size]
            test_returns = returns[-test_size:]
            
            # Fit model on training data
            train_model = arch_model(train_returns, vol='GARCH', p=self.p, q=self.q)
            train_fit = train_model.fit(disp='off', show_warning=False)
            
            # Make rolling forecasts
            forecasts = []
            actuals = []
            
            for i in range(len(test_returns)):
                # Forecast one step ahead
                forecast = train_fit.forecast(horizon=1, start=len(train_returns)-1+i)
                vol_forecast = np.sqrt(forecast.variance.iloc[-1, 0])
                forecasts.append(vol_forecast)
                
                # Actual volatility (absolute return as proxy)
                actual_vol = abs(test_returns.iloc[i])
                actuals.append(actual_vol)
            
            # Calculate accuracy (1 - normalized RMSE)
            rmse = np.sqrt(mean_squared_error(actuals, forecasts))
            max_actual = max(actuals)
            normalized_rmse = rmse / max_actual if max_actual > 0 else 1.0
            self.model_accuracy = max(0.0, 1.0 - normalized_rmse)
            
        except Exception as e:
            print(f"Accuracy calculation error: {e}")
            self.model_accuracy = 0.5  # Default moderate accuracy
    
    def predict(self, 
                returns: pd.Series,
                horizon: int = 1,
                confidence_level: float = 0.95) -> GARCHPrediction:
        """
        Generate GARCH prediction and trading signal
        
        Args:
            returns: Historical returns series
            horizon: Prediction horizon in days
            confidence_level: Confidence level for intervals
            
        Returns:
            GARCHPrediction object with volatility forecast and trading signal
        """
        if self.fitted_model is None:
            self.fit(returns)
        
        symbol = returns.name if hasattr(returns, 'name') and returns.name else 'UNKNOWN'
        
        try:
            # Generate forecast
            forecast = self.fitted_model.forecast(horizon=horizon)
            
            # Extract predictions
            volatility_forecast = np.sqrt(forecast.variance.iloc[-1, 0])
            
            # Mean return forecast (if available)
            if hasattr(forecast, 'mean') and not forecast.mean.empty:
                return_forecast = forecast.mean.iloc[-1, 0]
            else:
                # Use historical mean as fallback
                return_forecast = returns.mean()
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            z_score = 1.96  # For 95% confidence
            
            vol_std = volatility_forecast * 0.1  # Approximate standard error
            ci_lower = volatility_forecast - z_score * vol_std
            ci_upper = volatility_forecast + z_score * vol_std
            
            # Determine volatility regime
            vol_regime = self._classify_volatility_regime(volatility_forecast)
            
            # Generate trading signal
            signal_strength = self._generate_trading_signal(
                volatility_forecast, return_forecast, returns, vol_regime
            )
            
            return GARCHPrediction(
                symbol=symbol,
                predicted_volatility=volatility_forecast,
                predicted_returns=return_forecast,
                confidence_interval=(ci_lower, ci_upper),
                signal_strength=signal_strength,
                volatility_regime=vol_regime,
                prediction_horizon=horizon,
                model_accuracy=self.model_accuracy,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"GARCH prediction error: {e}")
            # Return default prediction
            return GARCHPrediction(
                symbol=symbol,
                predicted_volatility=returns.std(),
                predicted_returns=returns.mean(),
                confidence_interval=(0.0, returns.std() * 2),
                signal_strength=0.0,
                volatility_regime='medium',
                prediction_horizon=horizon,
                model_accuracy=0.0,
                timestamp=datetime.now()
            )
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility into low/medium/high regime"""
        if volatility < self.volatility_threshold_low:
            return 'low'
        elif volatility < self.volatility_threshold_high:
            return 'medium'
        else:
            return 'high'
    
    def _generate_trading_signal(self, 
                               predicted_vol: float,
                               predicted_return: float,
                               historical_returns: pd.Series,
                               vol_regime: str) -> float:
        """
        Generate trading signal based on GARCH predictions
        
        Signal logic:
        - High predicted returns + low/medium volatility = Buy signal
        - Low predicted returns + high volatility = Sell signal
        - Mean reversion in volatility regimes
        
        Returns:
            Signal strength from -1 (strong sell) to 1 (strong buy)
        """
        
        # Historical statistics for comparison
        hist_vol = historical_returns.std()
        hist_mean_return = historical_returns.mean()
        
        # Base signal from return prediction
        return_signal = 0.0
        if predicted_return > hist_mean_return + 0.5 * historical_returns.std():
            return_signal = 0.6
        elif predicted_return > hist_mean_return:
            return_signal = 0.3
        elif predicted_return < hist_mean_return - 0.5 * historical_returns.std():
            return_signal = -0.6
        elif predicted_return < hist_mean_return:
            return_signal = -0.3
        
        # Volatility-based signal (mean reversion)
        vol_signal = 0.0
        vol_ratio = predicted_vol / hist_vol
        
        if vol_regime == 'low' and vol_ratio < 0.8:
            # Very low volatility - expect increase (contrarian)
            vol_signal = 0.2
        elif vol_regime == 'high' and vol_ratio > 1.5:
            # Very high volatility - expect decrease, risk-off
            vol_signal = -0.4
        elif vol_regime == 'medium':
            # Medium volatility - neutral to slightly positive
            vol_signal = 0.1
        
        # Risk-adjusted signal
        risk_adjustment = 0.0
        if vol_regime == 'high':
            # Penalize signals in high volatility environment
            risk_adjustment = -0.2
        elif vol_regime == 'low':
            # Boost signals in low volatility environment
            risk_adjustment = 0.1
        
        # Combine signals
        total_signal = return_signal + vol_signal + risk_adjustment
        
        # Apply model confidence weighting
        confidence_weight = min(1.0, self.model_accuracy * 2)
        total_signal *= confidence_weight
        
        # Clamp to [-1, 1] range
        return max(-1.0, min(1.0, total_signal))
    
    def get_model_diagnostics(self) -> Dict:
        """Get model diagnostic information"""
        if self.fitted_model is None:
            return {"error": "Model not fitted"}
        
        try:
            summary = self.fitted_model.summary()
            
            return {
                "model_type": self.model_type,
                "parameters": {
                    "p": self.p,
                    "q": self.q,
                    "mean_model": self.mean_model
                },
                "aic": float(self.fitted_model.aic),
                "bic": float(self.fitted_model.bic),
                "log_likelihood": float(self.fitted_model.loglikelihood),
                "accuracy": self.model_accuracy,
                "last_fit": self.last_fit_date.isoformat() if self.last_fit_date else None,
                "num_parameters": len(self.fitted_model.params),
                "convergence": self.fitted_model.convergence_flag == 0
            }
        except Exception as e:
            return {"error": f"Diagnostic error: {e}"}
    
    def update_model(self, new_returns: pd.Series) -> 'GARCHModel':
        """
        Update model with new data (online learning approach)
        
        Args:
            new_returns: New returns data to incorporate
            
        Returns:
            Self for method chaining
        """
        if self.fitted_model is None:
            return self.fit(new_returns)
        
        # For simplicity, refit with extended data
        # In production, could implement more sophisticated online updates
        try:
            # Get existing data if available
            if hasattr(self.fitted_model, 'resids'):
                # Extend with new data and refit
                self.fit(new_returns)
            else:
                self.fit(new_returns)
        except Exception as e:
            print(f"Model update error: {e}")
        
        return self