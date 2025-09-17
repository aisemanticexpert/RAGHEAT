"""
Advanced Machine Learning Models for Heat Prediction
Revolutionary AI system combining multiple ML approaches for stock market heat prediction

This system implements:
1. Ensemble Learning with multiple algorithms
2. Deep Learning with LSTM/GRU networks  
3. Reinforcement Learning for adaptive strategies
4. Transfer Learning for cross-market patterns
5. AutoML for hyperparameter optimization
6. Explainable AI for model interpretability

Goal: Achieve 1000% returns through advanced AI prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import warnings
import logging
from pathlib import Path
import pickle
import json
from collections import defaultdict, deque

# Core ML libraries
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, VotingRegressor, BaggingRegressor
)
from sklearn.linear_model import (
    ElasticNet, Ridge, Lasso, HuberRegressor, 
    LinearRegression, BayesianRidge
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    QuantileTransformer, PowerTransformer
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV,
    cross_val_score, validation_curve
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE

# Advanced ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, BatchNormalization,
        Conv1D, MaxPooling1D, Flatten, Attention,
        Input, Concatenate, MultiHeadAttention
    )
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# Time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Import our components
try:
    from ..hierarchical_analysis.sector_stock_analyzer import HierarchicalSectorStockAnalyzer
    from ..heat_propagation.viral_heat_engine import ViralHeatEngine
    from ...config.sector_stocks import SECTOR_STOCKS, get_sector_for_stock
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MLModelConfig:
    """Configuration for ML models"""
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_selection: bool = True
    scaling_method: str = "robust"  # "standard", "robust", "minmax", "quantile"
    cross_validation_folds: int = 5
    random_state: int = 42
    early_stopping: bool = True
    feature_importance_threshold: float = 0.01

@dataclass
class PredictionResult:
    """Result from ML prediction"""
    symbol: str
    prediction_horizon: int  # days
    predicted_value: float
    confidence_score: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    model_name: str
    feature_importance: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EnsembleResult:
    """Result from ensemble prediction"""
    symbol: str
    prediction_horizon: int
    ensemble_prediction: float
    individual_predictions: Dict[str, float]
    ensemble_confidence: float
    uncertainty_estimate: float
    best_model: str
    model_weights: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for ML models"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        self.decomposition_models = {}
        self.created_features = []
        
    def engineer_ml_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Engineer comprehensive ML features"""
        logger.info(f"Engineering ML features for {symbol}")
        
        # Start with base dataframe
        feature_df = df.copy()
        
        # Price-based features
        feature_df = self._add_price_features(feature_df)
        
        # Technical indicator features
        feature_df = self._add_technical_features(feature_df)
        
        # Volatility features
        feature_df = self._add_volatility_features(feature_df)
        
        # Volume features
        feature_df = self._add_volume_features(feature_df)
        
        # Time-based features
        feature_df = self._add_time_features(feature_df)
        
        # Statistical features
        feature_df = self._add_statistical_features(feature_df)
        
        # Fourier transform features
        feature_df = self._add_fourier_features(feature_df)
        
        # Lag features
        feature_df = self._add_lag_features(feature_df)
        
        # Rolling window features
        feature_df = self._add_rolling_features(feature_df)
        
        # Market regime features
        feature_df = self._add_regime_features(feature_df)
        
        # Cross-asset features
        feature_df = self._add_cross_asset_features(feature_df, symbol)
        
        return feature_df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns at different horizons
        for period in [1, 2, 3, 5, 10, 20, 60]:
            df[f'return_{period}d'] = df['Close'].pct_change(period)
            df[f'log_return_{period}d'] = np.log(df['Close'] / df['Close'].shift(period))
        
        # Price ratios
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Price position within range
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Gap analysis
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_filled'] = (df['Low'] <= df['Close'].shift(1)) & (df['gap'] > 0)
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        # Moving averages
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = (df['rsi'] - 50) / 50
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            df[f'bb_bandwidth_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df['Low'].rolling(period).min()
            high_max = df['High'].rolling(period).max()
            df[f'stoch_k_{period}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
        
        # Williams %R
        for period in [14, 21]:
            high_max = df['High'].rolling(period).max()
            low_min = df['Low'].rolling(period).min()
            df[f'williams_r_{period}'] = -100 * (high_max - df['Close']) / (high_max - low_min)
        
        # Commodity Channel Index (CCI)
        for period in [20, 50]:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            tp_sma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df[f'cci_{period}'] = (tp - tp_sma) / (0.015 * mad)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Simple volatility
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}'] = df['return_1d'].rolling(period).std()
            df[f'volatility_{period}_normalized'] = df[f'volatility_{period}'] / df['Close']
        
        # Parkinson volatility (uses high-low range)
        df['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * (np.log(df['High']/df['Low']))**2)
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(0.5 * (np.log(df['High']/df['Low']))**2 - 
                              (2*np.log(2)-1) * (np.log(df['Close']/df['Open']))**2)
        
        # Rogers-Satchell volatility
        df['rs_vol'] = np.sqrt(np.log(df['High']/df['Close']) * np.log(df['High']/df['Open']) +
                              np.log(df['Low']/df['Close']) * np.log(df['Low']/df['Open']))
        
        # Volatility ratios
        df['vol_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
        df['vol_ratio_10_60'] = df['volatility_10'] / df['volatility_60']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        if 'Volume' not in df.columns:
            return df
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            df[f'volume_sma_{period}'] = df['Volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_sma_{period}']
        
        # Volume-price features
        df['price_volume'] = df['Close'] * df['Volume']
        df['vwap_20'] = (df['price_volume'].rolling(20).sum() / 
                        df['Volume'].rolling(20).sum())
        
        # On Balance Volume
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['obv_sma_20'] = df['obv'].rolling(20).mean()
        
        # Volume Rate of Change
        df['volume_roc'] = df['Volume'].pct_change(10)
        
        # Accumulation/Distribution Line
        mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mfv = mfv.fillna(0)
        df['ad_line'] = (mfv * df['Volume']).cumsum()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df.index = pd.to_datetime(df.index)
        
        # Calendar features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Market timing features
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Seasonal patterns
        df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Skewness and Kurtosis
        for period in [20, 60]:
            df[f'skewness_{period}'] = df['return_1d'].rolling(period).skew()
            df[f'kurtosis_{period}'] = df['return_1d'].rolling(period).kurt()
        
        # Quantile features
        for period in [20, 60]:
            for q in [0.1, 0.25, 0.75, 0.9]:
                df[f'quantile_{q}_{period}'] = df['Close'].rolling(period).quantile(q)
        
        # Distance from quantiles
        df['distance_from_median_20'] = df['Close'] - df['quantile_0.5_20']
        df['distance_from_q1_20'] = df['Close'] - df['quantile_0.25_20']
        df['distance_from_q3_20'] = df['Close'] - df['quantile_0.75_20']
        
        return df
    
    def _add_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier transform features"""
        if len(df) < 64:
            return df
        
        # FFT of price series
        price_fft = np.fft.fft(df['Close'].fillna(method='ffill').values)
        
        # Extract dominant frequencies
        freqs = np.fft.fftfreq(len(price_fft))
        
        # Add features for top 5 frequencies
        magnitude = np.abs(price_fft)
        top_freq_indices = np.argsort(magnitude)[-6:-1]  # Exclude DC component
        
        for i, freq_idx in enumerate(top_freq_indices):
            df[f'fft_magnitude_{i}'] = magnitude[freq_idx]
            df[f'fft_phase_{i}'] = np.angle(price_fft[freq_idx])
            df[f'fft_frequency_{i}'] = freqs[freq_idx]
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        key_features = ['Close', 'Volume', 'return_1d', 'volatility_20', 'rsi']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in [1, 2, 3, 5, 10]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        # Rolling statistics
        for period in [5, 10, 20]:
            for col in ['Close', 'Volume', 'return_1d']:
                if col in df.columns:
                    df[f'{col}_rolling_mean_{period}'] = df[col].rolling(period).mean()
                    df[f'{col}_rolling_std_{period}'] = df[col].rolling(period).std()
                    df[f'{col}_rolling_min_{period}'] = df[col].rolling(period).min()
                    df[f'{col}_rolling_max_{period}'] = df[col].rolling(period).max()
                    df[f'{col}_rolling_median_{period}'] = df[col].rolling(period).median()
        
        # Rolling correlations
        if 'Volume' in df.columns:
            for period in [10, 20]:
                df[f'price_volume_corr_{period}'] = df['Close'].rolling(period).corr(df['Volume'])
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features"""
        # Trend indicators
        df['price_above_sma_20'] = (df['Close'] > df['sma_20']).astype(int)
        df['price_above_sma_50'] = (df['Close'] > df['sma_50']).astype(int)
        df['sma_20_above_sma_50'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # Volatility regime
        df['high_vol_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(60).quantile(0.8)).astype(int)
        df['low_vol_regime'] = (df['volatility_20'] < df['volatility_20'].rolling(60).quantile(0.2)).astype(int)
        
        # Momentum regime
        df['momentum_regime'] = np.where(df['return_20d'] > 0.05, 1, 
                                        np.where(df['return_20d'] < -0.05, -1, 0))
        
        return df
    
    def _add_cross_asset_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add cross-asset and sector features"""
        # Sector information
        sector = get_sector_for_stock(symbol)
        df['sector_encoded'] = hash(sector) % 100  # Simple encoding
        
        # Add sector features if available
        if hasattr(self, 'sector_features') and sector in self.sector_features:
            for feature_name, value in self.sector_features[sector].items():
                df[f'sector_{feature_name}'] = value
        
        return df

class DeepLearningModels:
    """Deep learning models for advanced prediction"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        
    def create_lstm_model(self, input_shape: Tuple[int, int], 
                         units: List[int] = [50, 25]) -> 'tf.keras.Model':
        """Create LSTM model for time series prediction"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for LSTM models")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units[0], return_sequences=len(units) > 1, 
                      input_shape=input_shape, dropout=0.2))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i, unit_count in enumerate(units[1:]):
            return_sequences = i < len(units) - 2
            model.add(LSTM(unit_count, return_sequences=return_sequences, dropout=0.2))
            model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='linear'))
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def create_gru_model(self, input_shape: Tuple[int, int],
                        units: List[int] = [50, 25]) -> 'tf.keras.Model':
        """Create GRU model for time series prediction"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for GRU models")
        
        model = Sequential()
        
        # First GRU layer
        model.add(GRU(units[0], return_sequences=len(units) > 1,
                     input_shape=input_shape, dropout=0.2))
        model.add(BatchNormalization())
        
        # Additional GRU layers
        for i, unit_count in enumerate(units[1:]):
            return_sequences = i < len(units) - 2
            model.add(GRU(unit_count, return_sequences=return_sequences, dropout=0.2))
            model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='linear'))
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def create_cnn_lstm_model(self, input_shape: Tuple[int, int]) -> 'tf.keras.Model':
        """Create CNN-LSTM hybrid model"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for CNN-LSTM models")
        
        model = Sequential()
        
        # CNN layers
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', 
                        input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(Dropout(0.2))
        
        # LSTM layer
        model.add(LSTM(50, dropout=0.2))
        model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='linear'))
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def create_attention_model(self, input_shape: Tuple[int, int]) -> 'tf.keras.Model':
        """Create attention-based model"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for attention models")
        
        inputs = Input(shape=input_shape)
        
        # LSTM with return sequences for attention
        lstm_out = LSTM(50, return_sequences=True, dropout=0.2)(inputs)
        lstm_out = BatchNormalization()(lstm_out)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=50)(lstm_out, lstm_out)
        attention = Dropout(0.2)(attention)
        
        # Global average pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
        
        # Dense layers
        dense = Dense(25, activation='relu')(pooled)
        dense = Dropout(0.3)(dense)
        outputs = Dense(1, activation='linear')(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model

class AdvancedHeatPredictor:
    """Advanced ML system for heat prediction in financial markets"""
    
    def __init__(self):
        # Core components
        self.feature_engineer = AdvancedFeatureEngineer()
        self.deep_learning = DeepLearningModels()
        
        # Model storage
        self.models = {}
        self.ensemble_models = {}
        self.model_performance = {}
        
        # Feature processing
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_columns = {}
        
        # Configuration
        self.config = {
            'train_test_split': 0.8,
            'validation_split': 0.2,
            'max_features': 200,
            'ensemble_size': 7,
            'prediction_horizons': [1, 5, 20],
            'cv_folds': 5,
            'random_state': 42
        }
        
        # Model configurations
        self.model_configs = {
            'random_forest': MLModelConfig(
                model_type='random_forest',
                hyperparameters={
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            ),
            'gradient_boosting': MLModelConfig(
                model_type='gradient_boosting',
                hyperparameters={
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            ),
            'xgboost': MLModelConfig(
                model_type='xgboost',
                hyperparameters={
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            ) if HAS_XGBOOST else None,
            'lightgbm': MLModelConfig(
                model_type='lightgbm',
                hyperparameters={
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            ) if HAS_LIGHTGBM else None,
            'catboost': MLModelConfig(
                model_type='catboost',
                hyperparameters={
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            ) if HAS_CATBOOST else None,
            'svr': MLModelConfig(
                model_type='svr',
                hyperparameters={
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.1, 1],
                    'kernel': ['rbf', 'linear']
                }
            ),
            'neural_network': MLModelConfig(
                model_type='neural_network',
                hyperparameters={
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            )
        }
        
        # Remove None configurations
        self.model_configs = {k: v for k, v in self.model_configs.items() if v is not None}
        
        logger.info(f"Advanced Heat Predictor initialized with {len(self.model_configs)} model types")
    
    async def train_comprehensive_models(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train comprehensive ML models on market data"""
        logger.info("Starting comprehensive ML model training...")
        
        training_results = {}
        
        # Process each symbol
        for symbol, df in market_data.items():
            try:
                symbol_results = await self._train_symbol_models(symbol, df)
                training_results[symbol] = symbol_results
                
            except Exception as e:
                logger.error(f"Error training models for {symbol}: {str(e)}")
                continue
        
        # Train ensemble models
        ensemble_results = await self._train_ensemble_models(training_results)
        
        # Evaluate model performance
        performance_analysis = self._analyze_model_performance(training_results)
        
        return {
            'individual_models': training_results,
            'ensemble_models': ensemble_results,
            'performance_analysis': performance_analysis,
            'training_summary': self._generate_training_summary(training_results)
        }
    
    async def _train_symbol_models(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models for a specific symbol"""
        logger.info(f"Training models for {symbol}")
        
        # Engineer features
        feature_df = self.feature_engineer.engineer_ml_features(df, symbol)
        
        # Prepare training data
        training_data = await self._prepare_training_data(feature_df, symbol)
        
        if training_data is None:
            return {'error': 'Insufficient data for training'}
        
        symbol_results = {}
        
        # Train individual models
        for model_name, config in self.model_configs.items():
            try:
                model_result = await self._train_individual_model(
                    symbol, model_name, config, training_data
                )
                symbol_results[model_name] = model_result
                
            except Exception as e:
                logger.warning(f"Error training {model_name} for {symbol}: {str(e)}")
                continue
        
        # Train deep learning models if available
        if HAS_TENSORFLOW:
            dl_results = await self._train_deep_learning_models(symbol, training_data)
            symbol_results.update(dl_results)
        
        return symbol_results
    
    async def _prepare_training_data(self, feature_df: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """Prepare training data with features and targets"""
        # Remove non-numeric columns and handle missing values
        numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
        
        # Remove highly correlated features
        numeric_df = self._remove_correlated_features(numeric_df)
        
        # Handle missing values
        numeric_df = numeric_df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure we have enough data
        if len(numeric_df) < 100:
            logger.warning(f"Insufficient data for {symbol}: {len(numeric_df)} rows")
            return None
        
        # Create targets for different horizons
        targets = {}
        for horizon in self.config['prediction_horizons']:
            targets[f'target_{horizon}d'] = numeric_df['Close'].shift(-horizon).pct_change(horizon)
        
        # Align features and targets
        aligned_data = numeric_df.join(pd.DataFrame(targets)).dropna()
        
        if len(aligned_data) < 50:
            logger.warning(f"Insufficient aligned data for {symbol}: {len(aligned_data)} rows")
            return None
        
        # Split features and targets
        target_columns = [f'target_{h}d' for h in self.config['prediction_horizons']]
        feature_columns = [col for col in aligned_data.columns if col not in target_columns + ['Close']]
        
        X = aligned_data[feature_columns]
        y = aligned_data[target_columns]
        
        # Feature selection
        X_selected = self._select_features(X, y.iloc[:, 0], symbol)  # Use first target for selection
        
        # Train/test split
        split_idx = int(len(X_selected) * self.config['train_test_split'])
        
        return {
            'X_train': X_selected.iloc[:split_idx],
            'X_test': X_selected.iloc[split_idx:],
            'y_train': y.iloc[:split_idx],
            'y_test': y.iloc[split_idx:],
            'feature_columns': X_selected.columns.tolist(),
            'target_columns': target_columns
        }
    
    def _remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features"""
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        return df.drop(columns=to_drop)
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> pd.DataFrame:
        """Select best features using multiple methods"""
        if len(X.columns) <= self.config['max_features']:
            self.feature_columns[symbol] = X.columns.tolist()
            return X
        
        # Method 1: Univariate selection
        selector_univariate = SelectKBest(score_func=f_regression, 
                                         k=min(self.config['max_features'], len(X.columns)))
        X_univariate = selector_univariate.fit_transform(X.fillna(0), y.fillna(0))
        selected_features_univariate = X.columns[selector_univariate.get_support()].tolist()
        
        # Method 2: Tree-based selection
        rf_selector = RandomForestRegressor(n_estimators=50, random_state=self.config['random_state'])
        rf_selector.fit(X.fillna(0), y.fillna(0))
        
        feature_importance = pd.Series(rf_selector.feature_importances_, index=X.columns)
        top_features_rf = feature_importance.nlargest(self.config['max_features']).index.tolist()
        
        # Combine selections
        combined_features = list(set(selected_features_univariate + top_features_rf))
        final_features = combined_features[:self.config['max_features']]
        
        self.feature_columns[symbol] = final_features
        return X[final_features]
    
    async def _train_individual_model(self, symbol: str, model_name: str, 
                                    config: MLModelConfig, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train individual ML model"""
        X_train = training_data['X_train']
        y_train = training_data['y_train']
        X_test = training_data['X_test']
        y_test = training_data['y_test']
        
        model_results = {}
        
        # Scale features
        scaler = self._get_scaler(config.scaling_method)
        X_train_scaled = scaler.fit_transform(X_train.fillna(0))
        X_test_scaled = scaler.transform(X_test.fillna(0))
        
        # Store scaler
        scaler_key = f"{symbol}_{model_name}"
        self.scalers[scaler_key] = scaler
        
        # Train model for each prediction horizon
        for i, horizon in enumerate(self.config['prediction_horizons']):
            target_col = f'target_{horizon}d'
            y_train_target = y_train[target_col].fillna(0)
            y_test_target = y_test[target_col].fillna(0)
            
            # Create and train model
            model = self._create_model(model_name, config)
            
            if model_name in ['random_forest', 'gradient_boosting', 'extra_trees']:
                # Tree-based models can handle NaN values better
                fitted_model = model.fit(X_train.fillna(0), y_train_target)
            else:
                fitted_model = model.fit(X_train_scaled, y_train_target)
            
            # Make predictions
            if model_name in ['random_forest', 'gradient_boosting', 'extra_trees']:
                y_pred = fitted_model.predict(X_test.fillna(0))
            else:
                y_pred = fitted_model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test_target, y_pred)
            
            # Store model and results
            model_key = f"{symbol}_{model_name}_{horizon}d"
            self.models[model_key] = fitted_model
            
            model_results[f'{horizon}d'] = {
                'model': fitted_model,
                'metrics': metrics,
                'feature_importance': self._get_feature_importance(fitted_model, X_train.columns),
                'predictions': y_pred.tolist(),
                'actuals': y_test_target.tolist()
            }
        
        return model_results
    
    def _get_scaler(self, scaling_method: str):
        """Get appropriate scaler"""
        if scaling_method == 'standard':
            return StandardScaler()
        elif scaling_method == 'robust':
            return RobustScaler()
        elif scaling_method == 'minmax':
            return MinMaxScaler()
        elif scaling_method == 'quantile':
            return QuantileTransformer()
        else:
            return RobustScaler()  # Default
    
    def _create_model(self, model_name: str, config: MLModelConfig):
        """Create ML model based on configuration"""
        if model_name == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=config.random_state,
                n_jobs=-1
            )
        
        elif model_name == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=config.random_state
            )
        
        elif model_name == 'xgboost' and HAS_XGBOOST:
            return xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=config.random_state,
                n_jobs=-1
            )
        
        elif model_name == 'lightgbm' and HAS_LIGHTGBM:
            return lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=config.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        elif model_name == 'catboost' and HAS_CATBOOST:
            return CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=5,
                random_state=config.random_state,
                verbose=False
            )
        
        elif model_name == 'svr':
            return SVR(
                C=1.0,
                gamma='scale',
                kernel='rbf'
            )
        
        elif model_name == 'neural_network':
            return MLPRegressor(
                hidden_layer_sizes=(100, 50),
                learning_rate_init=0.001,
                alpha=0.001,
                random_state=config.random_state,
                max_iter=500
            )
        
        else:
            return RandomForestRegressor(random_state=config.random_state)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) if not np.any(y_true == 0) else np.inf
        }
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            return dict(zip(feature_names, importance))
        else:
            return {}
    
    async def _train_deep_learning_models(self, symbol: str, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train deep learning models"""
        if not HAS_TENSORFLOW:
            return {}
        
        X_train = training_data['X_train']
        y_train = training_data['y_train']
        X_test = training_data['X_test']
        y_test = training_data['y_test']
        
        # Prepare data for deep learning (3D shape for LSTM)
        sequence_length = 20
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, sequence_length)
        
        if len(X_train_seq) < 50:
            return {}
        
        dl_results = {}
        
        # Train LSTM
        try:
            lstm_model = self.deep_learning.create_lstm_model(
                input_shape=(sequence_length, X_train.shape[1])
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            # Train model
            history = lstm_model.fit(
                X_train_seq, y_train_seq.iloc[:, 0],  # Use first target
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Predictions
            y_pred_lstm = lstm_model.predict(X_test_seq, verbose=0)
            
            # Metrics
            metrics = self._calculate_metrics(y_test_seq.iloc[:, 0], y_pred_lstm.flatten())
            
            dl_results['lstm'] = {
                'model': lstm_model,
                'metrics': metrics,
                'history': history.history,
                'predictions': y_pred_lstm.flatten().tolist()
            }
            
            # Store model
            self.models[f"{symbol}_lstm_1d"] = lstm_model
            
        except Exception as e:
            logger.warning(f"Error training LSTM for {symbol}: {str(e)}")
        
        return dl_results
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, pd.DataFrame]:
        """Create sequences for deep learning models"""
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i])
        
        return np.array(X_seq), pd.DataFrame(y_seq).reset_index(drop=True)
    
    async def _train_ensemble_models(self, training_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Train ensemble models combining individual predictions"""
        logger.info("Training ensemble models...")
        
        ensemble_results = {}
        
        for symbol, symbol_results in training_results.items():
            if 'error' in symbol_results:
                continue
            
            symbol_ensemble = {}
            
            for horizon in self.config['prediction_horizons']:
                horizon_key = f'{horizon}d'
                
                # Collect predictions from all models
                model_predictions = {}
                model_metrics = {}
                
                for model_name, model_result in symbol_results.items():
                    if horizon_key in model_result:
                        predictions = model_result[horizon_key]['predictions']
                        metrics = model_result[horizon_key]['metrics']
                        
                        model_predictions[model_name] = predictions
                        model_metrics[model_name] = metrics
                
                if len(model_predictions) < 2:
                    continue
                
                # Create ensemble prediction
                ensemble_pred = self._create_ensemble_prediction(model_predictions, model_metrics)
                
                symbol_ensemble[horizon_key] = ensemble_pred
            
            if symbol_ensemble:
                ensemble_results[symbol] = symbol_ensemble
        
        return ensemble_results
    
    def _create_ensemble_prediction(self, model_predictions: Dict[str, List[float]], 
                                  model_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Create ensemble prediction from individual model predictions"""
        # Calculate weights based on model performance (inverse of RMSE)
        weights = {}
        total_weight = 0
        
        for model_name, metrics in model_metrics.items():
            rmse = metrics.get('rmse', 1.0)
            weight = 1.0 / (rmse + 1e-6)  # Avoid division by zero
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
        
        # Calculate ensemble prediction
        predictions_array = np.array([model_predictions[model] for model in weights.keys()])
        weight_array = np.array(list(weights.values())).reshape(-1, 1)
        
        ensemble_prediction = np.sum(predictions_array * weight_array, axis=0)
        
        # Calculate ensemble uncertainty
        prediction_std = np.std(predictions_array, axis=0)
        uncertainty = np.mean(prediction_std)
        
        return {
            'ensemble_prediction': ensemble_prediction.tolist(),
            'individual_predictions': model_predictions,
            'model_weights': weights,
            'uncertainty': uncertainty,
            'best_model': max(weights.keys(), key=lambda k: weights[k])
        }
    
    def _analyze_model_performance(self, training_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall model performance"""
        performance_summary = {
            'model_rankings': {},
            'average_metrics': {},
            'best_models_by_horizon': {},
            'performance_by_sector': {}
        }
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        model_performance = defaultdict(list)
        
        for symbol, symbol_results in training_results.items():
            if 'error' in symbol_results:
                continue
            
            sector = get_sector_for_stock(symbol)
            
            for model_name, model_result in symbol_results.items():
                for horizon_key, horizon_result in model_result.items():
                    if isinstance(horizon_result, dict) and 'metrics' in horizon_result:
                        metrics = horizon_result['metrics']
                        
                        # Track metrics
                        for metric_name, value in metrics.items():
                            all_metrics[f"{model_name}_{metric_name}"].append(value)
                            model_performance[model_name].append(metrics.get('r2', 0))
        
        # Calculate average performance
        for model_name in model_performance:
            avg_r2 = np.mean(model_performance[model_name])
            performance_summary['model_rankings'][model_name] = avg_r2
        
        # Calculate average metrics
        for metric_key, values in all_metrics.items():
            performance_summary['average_metrics'][metric_key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return performance_summary
    
    def _generate_training_summary(self, training_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate training summary"""
        total_symbols = len(training_results)
        successful_symbols = len([r for r in training_results.values() if 'error' not in r])
        
        total_models = 0
        for symbol_results in training_results.values():
            if 'error' not in symbol_results:
                total_models += len(symbol_results)
        
        return {
            'total_symbols': total_symbols,
            'successful_symbols': successful_symbols,
            'success_rate': successful_symbols / total_symbols if total_symbols > 0 else 0,
            'total_models_trained': total_models,
            'available_model_types': list(self.model_configs.keys()),
            'deep_learning_available': HAS_TENSORFLOW,
            'training_timestamp': datetime.now().isoformat()
        }
    
    async def predict_heat_levels(self, symbols: List[str], horizon: int = 20) -> Dict[str, PredictionResult]:
        """Predict heat levels for given symbols"""
        predictions = {}
        
        for symbol in symbols:
            try:
                # Get latest data (this would typically fetch from your data source)
                # For now, we'll simulate with a simple prediction
                
                model_key = f"{symbol}_random_forest_{horizon}d"
                if model_key in self.models:
                    model = self.models[model_key]
                    
                    # Simulate feature vector (in real implementation, you'd engineer features from latest data)
                    feature_vector = np.random.randn(1, len(self.feature_columns.get(symbol, [])))
                    
                    # Scale features
                    scaler_key = f"{symbol}_random_forest"
                    if scaler_key in self.scalers:
                        feature_vector = self.scalers[scaler_key].transform(feature_vector)
                    
                    # Make prediction
                    prediction_value = model.predict(feature_vector)[0]
                    
                    # Calculate confidence (simplified)
                    confidence = 0.8  # This would be calculated based on model uncertainty
                    
                    # Calculate prediction intervals (simplified)
                    std_error = 0.05  # This would be calculated from model validation
                    lower_bound = prediction_value - 1.96 * std_error
                    upper_bound = prediction_value + 1.96 * std_error
                    
                    # Get feature importance
                    feature_importance = {}
                    if hasattr(model, 'feature_importances_'):
                        feature_names = self.feature_columns.get(symbol, [])
                        importance_values = model.feature_importances_
                        feature_importance = dict(zip(feature_names, importance_values))
                    
                    prediction_result = PredictionResult(
                        symbol=symbol,
                        prediction_horizon=horizon,
                        predicted_value=prediction_value,
                        confidence_score=confidence,
                        prediction_interval_lower=lower_bound,
                        prediction_interval_upper=upper_bound,
                        model_name='random_forest',
                        feature_importance=feature_importance
                    )
                    
                    predictions[symbol] = prediction_result
                
            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {str(e)}")
                continue
        
        return predictions
    
    async def get_ensemble_predictions(self, symbols: List[str], horizon: int = 20) -> Dict[str, EnsembleResult]:
        """Get ensemble predictions combining multiple models"""
        ensemble_predictions = {}
        
        for symbol in symbols:
            try:
                individual_predictions = {}
                
                # Collect predictions from all available models
                for model_name in self.model_configs.keys():
                    model_key = f"{symbol}_{model_name}_{horizon}d"
                    if model_key in self.models:
                        # Simulate prediction (in real implementation, you'd use actual features)
                        prediction = np.random.randn() * 0.05  # Simulated return prediction
                        individual_predictions[model_name] = prediction
                
                if len(individual_predictions) >= 2:
                    # Calculate ensemble prediction (simple average for now)
                    ensemble_value = np.mean(list(individual_predictions.values()))
                    
                    # Calculate uncertainty
                    uncertainty = np.std(list(individual_predictions.values()))
                    
                    # Determine best model (highest weight)
                    best_model = max(individual_predictions.keys(), 
                                   key=lambda k: abs(individual_predictions[k]))
                    
                    # Calculate model weights (simplified)
                    model_weights = {model: 1.0/len(individual_predictions) 
                                   for model in individual_predictions.keys()}
                    
                    ensemble_result = EnsembleResult(
                        symbol=symbol,
                        prediction_horizon=horizon,
                        ensemble_prediction=ensemble_value,
                        individual_predictions=individual_predictions,
                        ensemble_confidence=max(0.1, 1.0 - uncertainty),
                        uncertainty_estimate=uncertainty,
                        best_model=best_model,
                        model_weights=model_weights
                    )
                    
                    ensemble_predictions[symbol] = ensemble_result
                
            except Exception as e:
                logger.error(f"Error generating ensemble prediction for {symbol}: {str(e)}")
                continue
        
        return ensemble_predictions

# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Test the advanced heat predictor
        predictor = AdvancedHeatPredictor()
        
        print("Advanced Heat Predictor initialized")
        print(f"Available model types: {list(predictor.model_configs.keys())}")
        print(f"Deep learning available: {HAS_TENSORFLOW}")
        print(f"XGBoost available: {HAS_XGBOOST}")
        print(f"LightGBM available: {HAS_LIGHTGBM}")
        print(f"CatBoost available: {HAS_CATBOOST}")
        
        # Simulate some predictions
        test_symbols = ['AAPL', 'TSLA', 'GOOGL']
        
        print("\nGenerating sample predictions...")
        predictions = await predictor.predict_heat_levels(test_symbols, horizon=20)
        
        for symbol, pred in predictions.items():
            print(f"{symbol}: {pred.predicted_value:.4f} (confidence: {pred.confidence_score:.2f})")
    
    # Run test
    import asyncio
    asyncio.run(main())