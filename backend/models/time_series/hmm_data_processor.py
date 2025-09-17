"""
HMM-specific Data Processing Utilities
Enhanced data preprocessing for Hidden Markov Model applications
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from .data_preprocessor import TimeSeriesPreprocessor, PreprocessedData
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA


@dataclass
class HMMPreprocessedData:
    """Container for HMM-specific preprocessed data"""
    symbol: str
    feature_matrix: np.ndarray
    feature_names: List[str]
    scaled_features: np.ndarray
    raw_prices: pd.Series
    returns: pd.Series
    regime_indicators: pd.DataFrame
    market_conditions: Dict[str, any]
    preprocessing_metadata: Dict[str, any]
    timestamp: datetime


class HMMDataProcessor:
    """
    Specialized data processor for Hidden Markov Model applications
    
    Focuses on creating features that help identify market regimes:
    - Volatility clustering indicators
    - Momentum and mean reversion signals
    - Market microstructure features
    - Macro-economic regime indicators
    """
    
    def __init__(self,
                 feature_selection_method: str = "kbest",
                 n_features: int = 20,
                 scaling_method: str = "robust",
                 lookback_periods: List[int] = None,
                 include_macro_features: bool = True):
        """
        Initialize HMM data processor
        
        Args:
            feature_selection_method: Method for feature selection ('kbest', 'pca', 'all')
            n_features: Number of features to select (if using selection)
            scaling_method: Scaling method ('standard', 'robust', 'minmax')
            lookback_periods: Periods for rolling calculations
            include_macro_features: Whether to include macro-economic features
        """
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.scaling_method = scaling_method
        self.include_macro_features = include_macro_features
        
        # Default lookback periods for different feature types
        if lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50, 100]
        else:
            self.lookback_periods = lookback_periods
        
        # Initialize base preprocessor
        self.base_preprocessor = TimeSeriesPreprocessor()
        
        # Initialize scalers
        self.scaler = self._get_scaler()
        self.feature_selector = None
        
        # Cache for feature importance
        self.feature_importance = {}
        
    def _get_scaler(self):
        """Get the appropriate scaler based on scaling method"""
        if self.scaling_method == "standard":
            return StandardScaler()
        elif self.scaling_method == "robust":
            return RobustScaler()
        elif self.scaling_method == "minmax":
            return MinMaxScaler()
        else:
            return StandardScaler()
    
    def process_for_hmm(self,
                       symbol: str,
                       data: Optional[pd.DataFrame] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       external_data: Optional[Dict[str, pd.Series]] = None) -> HMMPreprocessedData:
        """
        Process data specifically for HMM regime detection
        
        Args:
            symbol: Stock symbol
            data: Optional pre-loaded OHLCV data
            start_date: Start date for data fetching
            end_date: End date for data fetching
            external_data: Optional external data (VIX, rates, etc.)
            
        Returns:
            HMMPreprocessedData object
        """
        try:
            # Step 1: Basic preprocessing
            base_data = self.base_preprocessor.preprocess(
                symbol=symbol,
                data=data,
                start_date=start_date,
                end_date=end_date,
                add_features=True,
                clean_outliers=True
            )
            
            # Step 2: Create HMM-specific features
            hmm_features = self._create_hmm_features(base_data, external_data)
            
            # Step 3: Create regime indicators
            regime_indicators = self._create_regime_indicators(base_data, hmm_features)
            
            # Step 4: Feature selection
            selected_features, feature_names = self._select_features(
                hmm_features, base_data.returns
            )
            
            # Step 5: Scale features
            scaled_features = self._scale_features(selected_features)
            
            # Step 6: Analyze market conditions
            market_conditions = self._analyze_market_conditions(
                base_data, hmm_features, regime_indicators
            )
            
            # Step 7: Create metadata
            metadata = {
                'n_observations': len(selected_features),
                'n_features': selected_features.shape[1],
                'feature_selection_method': self.feature_selection_method,
                'scaling_method': self.scaling_method,
                'data_quality_score': self._calculate_data_quality(selected_features),
                'missing_data_pct': np.isnan(selected_features).mean(),
                'processing_timestamp': datetime.now()
            }
            
            return HMMPreprocessedData(
                symbol=symbol,
                feature_matrix=selected_features,
                feature_names=feature_names,
                scaled_features=scaled_features,
                raw_prices=base_data.raw_prices,
                returns=base_data.returns,
                regime_indicators=regime_indicators,
                market_conditions=market_conditions,
                preprocessing_metadata=metadata,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            raise RuntimeError(f"HMM data processing failed for {symbol}: {e}")
    
    def _create_hmm_features(self,
                           base_data: PreprocessedData,
                           external_data: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        """Create comprehensive features for HMM regime detection"""
        
        prices = base_data.raw_prices
        returns = base_data.returns
        volume = base_data.volume
        
        features = pd.DataFrame(index=prices.index)
        
        # 1. Return-based features
        features['returns'] = returns
        features['abs_returns'] = np.abs(returns)
        features['squared_returns'] = returns ** 2
        features['log_returns'] = np.log(prices / prices.shift(1))
        
        # 2. Volatility features (key for regime detection)
        for period in self.lookback_periods:
            if period <= len(returns):
                features[f'volatility_{period}'] = returns.rolling(period).std()
                features[f'realized_vol_{period}'] = np.sqrt(
                    (returns ** 2).rolling(period).sum()
                )
                features[f'vol_of_vol_{period}'] = (
                    returns.rolling(period).std().rolling(period//2).std()
                )
        
        # 3. Volatility regime indicators
        features['vol_regime_low'] = (features['volatility_20'] < features['volatility_20'].quantile(0.33)).astype(int)
        features['vol_regime_high'] = (features['volatility_20'] > features['volatility_20'].quantile(0.67)).astype(int)
        
        # 4. Return distribution features
        for period in [10, 20, 50]:
            if period <= len(returns):
                rolling_returns = returns.rolling(period)
                features[f'skewness_{period}'] = rolling_returns.skew()
                features[f'kurtosis_{period}'] = rolling_returns.kurt()
                features[f'return_range_{period}'] = (
                    rolling_returns.max() - rolling_returns.min()
                )
        
        # 5. Momentum features
        for period in self.lookback_periods:
            if period <= len(prices):
                features[f'momentum_{period}'] = prices / prices.shift(period) - 1
                features[f'roc_{period}'] = prices.pct_change(period)
        
        # 6. Mean reversion features
        for period in [10, 20, 50]:
            if period <= len(prices):
                sma = prices.rolling(period).mean()
                features[f'price_to_sma_{period}'] = prices / sma
                features[f'deviation_from_sma_{period}'] = (prices - sma) / sma
        
        # 7. Trend features
        for period in [5, 10, 20]:
            if period <= len(prices):
                features[f'trend_strength_{period}'] = self._calculate_trend_strength(
                    prices, period
                )
        
        # 8. Volatility clustering features
        features['vol_clustering'] = self._detect_volatility_clustering(returns)
        features['garch_proxy'] = self._calculate_garch_proxy(returns)
        
        # 9. Jump detection
        features['jump_indicator'] = self._detect_jumps(returns)
        features['jump_size'] = self._calculate_jump_size(returns)
        
        # 10. Market microstructure (if volume available)
        if volume is not None:
            features['volume_trend'] = volume / volume.rolling(20).mean()
            features['price_volume_trend'] = returns * np.log(volume / volume.shift(1))
            features['volume_volatility'] = (
                np.log(volume / volume.shift(1)).rolling(20).std()
            )
        
        # 11. Technical indicators optimized for regime detection
        features.update(self._create_regime_technical_indicators(prices, returns))
        
        # 12. External market data (if available)
        if external_data and self.include_macro_features:
            features.update(self._incorporate_external_data(external_data, features.index))
        
        # 13. Interaction features
        features.update(self._create_interaction_features(features))
        
        # Clean and validate features
        features = self._clean_features(features)
        
        return features
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate trend strength using linear regression slope"""
        
        def rolling_slope(window_prices):
            if len(window_prices) < period:
                return np.nan
            x = np.arange(len(window_prices))
            y = window_prices.values
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                return slope / window_prices.iloc[-1]  # Normalize by price
            return 0
        
        return prices.rolling(period).apply(rolling_slope, raw=False)
    
    def _detect_volatility_clustering(self, returns: pd.Series, threshold: float = 2.0) -> pd.Series:
        """Detect volatility clustering patterns"""
        
        # Calculate rolling volatility
        vol = returns.rolling(20).std()
        vol_mean = vol.rolling(100).mean()
        vol_std = vol.rolling(100).std()
        
        # Detect periods of high volatility clustering
        clustering = ((vol - vol_mean) / vol_std) > threshold
        
        return clustering.astype(int)
    
    def _calculate_garch_proxy(self, returns: pd.Series) -> pd.Series:
        """Calculate simple GARCH-like proxy for volatility"""
        
        # Simple GARCH(1,1) proxy: conditional variance
        alpha = 0.05
        beta = 0.9
        
        cond_var = pd.Series(index=returns.index, dtype=float)
        cond_var.iloc[0] = returns.var()
        
        for i in range(1, len(returns)):
            cond_var.iloc[i] = (
                alpha * returns.iloc[i-1]**2 + 
                beta * cond_var.iloc[i-1]
            )
        
        return np.sqrt(cond_var)
    
    def _detect_jumps(self, returns: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect price jumps using threshold method"""
        
        # Calculate rolling mean and std
        rolling_mean = returns.rolling(20).mean()
        rolling_std = returns.rolling(20).std()
        
        # Detect jumps
        standardized_returns = (returns - rolling_mean) / rolling_std
        jumps = (np.abs(standardized_returns) > threshold).astype(int)
        
        return jumps
    
    def _calculate_jump_size(self, returns: pd.Series) -> pd.Series:
        """Calculate size of detected jumps"""
        
        rolling_std = returns.rolling(20).std()
        jump_size = np.abs(returns) / rolling_std
        
        return jump_size
    
    def _create_regime_technical_indicators(self,
                                          prices: pd.Series,
                                          returns: pd.Series) -> Dict[str, pd.Series]:
        """Create technical indicators optimized for regime detection"""
        
        indicators = {}
        
        # Bollinger Band position (regime indicator)
        sma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        indicators['bb_position'] = (prices - bb_lower) / (bb_upper - bb_lower)
        indicators['bb_width'] = (bb_upper - bb_lower) / sma_20
        
        # RSI regime levels
        indicators['rsi'] = self._calculate_rsi(prices)
        indicators['rsi_regime'] = np.where(
            indicators['rsi'] > 70, 1,
            np.where(indicators['rsi'] < 30, -1, 0)
        )
        
        # MACD regime signals
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        indicators['macd_regime'] = np.where(macd > macd_signal, 1, -1)
        
        # Moving average regime
        sma_50 = prices.rolling(50).mean()
        sma_200 = prices.rolling(200).mean()
        indicators['ma_regime'] = np.where(
            (prices > sma_50) & (sma_50 > sma_200), 1,
            np.where((prices < sma_50) & (sma_50 < sma_200), -1, 0)
        )
        
        # Volatility-adjusted momentum
        for period in [10, 20]:
            momentum = prices / prices.shift(period) - 1
            volatility = returns.rolling(period).std()
            indicators[f'vol_adj_momentum_{period}'] = momentum / (volatility + 1e-8)
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _incorporate_external_data(self,
                                 external_data: Dict[str, pd.Series],
                                 index: pd.Index) -> Dict[str, pd.Series]:
        """Incorporate external market data"""
        
        external_features = {}
        
        for name, series in external_data.items():
            # Align with main data index
            aligned_series = series.reindex(index, method='ffill')
            
            if name.upper() in ['VIX', 'VOLATILITY']:
                # VIX features
                external_features[f'{name}_level'] = aligned_series
                external_features[f'{name}_regime'] = np.where(
                    aligned_series > aligned_series.quantile(0.75), 1,
                    np.where(aligned_series < aligned_series.quantile(0.25), -1, 0)
                )
                external_features[f'{name}_change'] = aligned_series.pct_change()
                
            elif name.upper() in ['TREASURY', 'RATE', 'YIELD']:
                # Interest rate features
                external_features[f'{name}_level'] = aligned_series
                external_features[f'{name}_change'] = aligned_series.diff()
                external_features[f'{name}_trend'] = aligned_series.rolling(20).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                
            elif name.upper() in ['USD', 'DOLLAR', 'DXY']:
                # Currency features
                external_features[f'{name}_strength'] = aligned_series
                external_features[f'{name}_volatility'] = aligned_series.pct_change().rolling(20).std()
                
        return external_features
    
    def _create_interaction_features(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create interaction features between key variables"""
        
        interactions = {}
        
        # Volatility-momentum interactions
        if 'volatility_20' in features.columns and 'momentum_20' in features.columns:
            interactions['vol_momentum_interaction'] = (
                features['volatility_20'] * features['momentum_20']
            )
        
        # Return-volume interactions (if volume available)
        vol_cols = [col for col in features.columns if 'volume' in col.lower()]
        if vol_cols and 'returns' in features.columns:
            interactions['return_volume_interaction'] = (
                features['returns'] * features[vol_cols[0]]
            )
        
        # Trend-volatility interactions
        trend_cols = [col for col in features.columns if 'trend' in col]
        vol_cols = [col for col in features.columns if 'volatility' in col]
        
        if trend_cols and vol_cols:
            interactions['trend_vol_interaction'] = (
                features[trend_cols[0]] * features[vol_cols[0]]
            )
        
        return interactions
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill missing values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        valid_cols = features.columns[features.isnull().mean() < missing_threshold]
        features = features[valid_cols]
        
        # Remove highly correlated features
        features = self._remove_highly_correlated_features(features)
        
        return features
    
    def _remove_highly_correlated_features(self,
                                         features: pd.DataFrame,
                                         threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features"""
        
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        return features.drop(columns=to_drop)
    
    def _select_features(self,
                        features: pd.DataFrame,
                        target: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Select most relevant features for HMM"""
        
        # Align features and target
        common_index = features.index.intersection(target.index)
        features_aligned = features.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Remove any remaining NaN values
        valid_mask = ~(features_aligned.isnull().any(axis=1) | target_aligned.isnull())
        features_clean = features_aligned[valid_mask]
        target_clean = target_aligned[valid_mask]
        
        if len(features_clean) == 0:
            raise ValueError("No valid features after cleaning")
        
        if self.feature_selection_method == "all":
            selected_features = features_clean.values
            feature_names = features_clean.columns.tolist()
            
        elif self.feature_selection_method == "kbest":
            # Use SelectKBest for feature selection
            n_features_actual = min(self.n_features, features_clean.shape[1])
            self.feature_selector = SelectKBest(f_regression, k=n_features_actual)
            
            selected_features = self.feature_selector.fit_transform(
                features_clean.values, np.abs(target_clean.values)  # Use absolute returns as target
            )
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            feature_names = [features_clean.columns[i] for i in selected_indices]
            
        elif self.feature_selection_method == "pca":
            # Use PCA for dimensionality reduction
            n_components = min(self.n_features, features_clean.shape[1], len(features_clean))
            self.feature_selector = PCA(n_components=n_components)
            
            selected_features = self.feature_selector.fit_transform(features_clean.values)
            feature_names = [f"PC_{i+1}" for i in range(selected_features.shape[1])]
            
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection_method}")
        
        # Store feature importance
        self._calculate_feature_importance(features_clean, target_clean)
        
        return selected_features, feature_names
    
    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using the specified method"""
        
        if len(features) == 0:
            return features
        
        try:
            scaled_features = self.scaler.fit_transform(features)
            return scaled_features
        except Exception as e:
            print(f"Feature scaling error: {e}")
            return features
    
    def _calculate_feature_importance(self,
                                    features: pd.DataFrame,
                                    target: pd.Series):
        """Calculate and store feature importance"""
        
        try:
            # Calculate correlation with absolute returns
            abs_target = np.abs(target)
            correlations = features.corrwith(abs_target).abs()
            
            # Store top features
            self.feature_importance = correlations.sort_values(ascending=False).to_dict()
            
        except Exception as e:
            print(f"Feature importance calculation error: {e}")
            self.feature_importance = {}
    
    def _create_regime_indicators(self,
                                base_data: PreprocessedData,
                                features: pd.DataFrame) -> pd.DataFrame:
        """Create binary regime indicators"""
        
        indicators = pd.DataFrame(index=features.index)
        
        # Volatility regime indicators
        if 'volatility_20' in features.columns:
            vol_20 = features['volatility_20']
            indicators['high_vol_regime'] = (vol_20 > vol_20.quantile(0.75)).astype(int)
            indicators['low_vol_regime'] = (vol_20 < vol_20.quantile(0.25)).astype(int)
        
        # Return regime indicators
        if 'returns' in features.columns:
            returns = features['returns']
            indicators['bull_regime'] = (returns.rolling(20).mean() > 0.01).astype(int)
            indicators['bear_regime'] = (returns.rolling(20).mean() < -0.01).astype(int)
        
        # Trend regime indicators
        trend_cols = [col for col in features.columns if 'trend_strength' in col]
        if trend_cols:
            trend = features[trend_cols[0]]
            indicators['uptrend_regime'] = (trend > 0).astype(int)
            indicators['downtrend_regime'] = (trend < 0).astype(int)
        
        # Technical regime indicators
        if 'rsi' in features.columns:
            rsi = features['rsi']
            indicators['overbought_regime'] = (rsi > 70).astype(int)
            indicators['oversold_regime'] = (rsi < 30).astype(int)
        
        return indicators
    
    def _analyze_market_conditions(self,
                                 base_data: PreprocessedData,
                                 features: pd.DataFrame,
                                 regime_indicators: pd.DataFrame) -> Dict[str, any]:
        """Analyze overall market conditions"""
        
        conditions = {}
        
        # Recent market performance
        recent_returns = base_data.returns.tail(20)
        conditions['recent_performance'] = {
            'mean_return': recent_returns.mean(),
            'volatility': recent_returns.std(),
            'max_drawdown': self._calculate_max_drawdown(base_data.raw_prices.tail(50)),
            'sharpe_ratio': recent_returns.mean() / recent_returns.std() if recent_returns.std() > 0 else 0
        }
        
        # Current regime probabilities
        if not regime_indicators.empty:
            recent_indicators = regime_indicators.tail(5)
            conditions['regime_probabilities'] = {
                col: recent_indicators[col].mean() 
                for col in regime_indicators.columns
            }
        
        # Volatility conditions
        if 'volatility_20' in features.columns:
            vol_20 = features['volatility_20'].dropna()
            conditions['volatility_conditions'] = {
                'current_volatility': vol_20.iloc[-1] if len(vol_20) > 0 else 0,
                'volatility_percentile': (vol_20.iloc[-1] <= vol_20).mean() if len(vol_20) > 0 else 0.5,
                'volatility_trend': vol_20.tail(10).mean() - vol_20.tail(20).mean()
            }
        
        # Market stress indicators
        stress_indicators = []
        if 'jump_indicator' in features.columns:
            stress_indicators.append(features['jump_indicator'].tail(10).sum())
        if 'vol_clustering' in features.columns:
            stress_indicators.append(features['vol_clustering'].tail(10).sum())
        
        conditions['market_stress'] = {
            'stress_score': np.mean(stress_indicators) if stress_indicators else 0,
            'stress_level': 'HIGH' if np.mean(stress_indicators) > 5 else 'MEDIUM' if np.mean(stress_indicators) > 2 else 'LOW'
        }
        
        return conditions
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return 0.0
        
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_data_quality(self, features: np.ndarray) -> float:
        """Calculate data quality score"""
        
        if len(features) == 0:
            return 0.0
        
        # Check for missing values
        missing_pct = np.isnan(features).mean()
        
        # Check for infinite values
        inf_pct = np.isinf(features).mean()
        
        # Check for constant features
        constant_features = 0
        for i in range(features.shape[1]):
            if np.std(features[:, i][~np.isnan(features[:, i])]) < 1e-8:
                constant_features += 1
        
        constant_pct = constant_features / features.shape[1] if features.shape[1] > 0 else 0
        
        # Calculate quality score
        quality_score = 1.0 - (missing_pct + inf_pct + constant_pct) / 3
        
        return max(0.0, min(1.0, quality_score))
    
    def get_feature_importance_report(self) -> Dict[str, any]:
        """Get feature importance report"""
        
        return {
            'top_features': dict(list(self.feature_importance.items())[:10]),
            'n_features_analyzed': len(self.feature_importance),
            'feature_selection_method': self.feature_selection_method,
            'scaling_method': self.scaling_method
        }
    
    def update_parameters(self,
                         feature_selection_method: Optional[str] = None,
                         n_features: Optional[int] = None,
                         scaling_method: Optional[str] = None):
        """Update processor parameters"""
        
        if feature_selection_method is not None:
            self.feature_selection_method = feature_selection_method
        
        if n_features is not None:
            self.n_features = n_features
        
        if scaling_method is not None:
            self.scaling_method = scaling_method
            self.scaler = self._get_scaler()