"""
Time Series Data Preprocessing Utilities for GARCH and other models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass


@dataclass
class PreprocessedData:
    """Container for preprocessed time series data"""
    symbol: str
    raw_prices: pd.Series
    returns: pd.Series
    log_returns: pd.Series
    volatility: pd.Series
    volume: Optional[pd.Series]
    features: Dict[str, pd.Series]
    metadata: Dict[str, any]
    

class TimeSeriesPreprocessor:
    """
    Comprehensive time series data preprocessing for financial models
    """
    
    def __init__(self, 
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 3.0,
                 min_periods: int = 30,
                 frequency: str = 'D'):
        """
        Initialize preprocessor
        
        Args:
            outlier_method: 'iqr', 'zscore', or 'percentile'
            outlier_threshold: Threshold for outlier detection
            min_periods: Minimum periods required for processing
            frequency: Data frequency ('D', 'H', '5min', etc.)
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.min_periods = min_periods
        self.frequency = frequency
        
    def fetch_yahoo_data(self, 
                        symbol: str,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        period: str = '1y') -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data  
            period: Period string if dates not specified
            
        Returns:
            Raw OHLCV DataFrame
        """
        try:
            if start_date and end_date:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            else:
                data = yf.download(symbol, period=period, progress=False)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)  # Remove ticker level
                
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {symbol}: {e}")
    
    def calculate_returns(self, 
                         prices: pd.Series,
                         method: str = 'simple') -> pd.Series:
        """
        Calculate returns from price series
        
        Args:
            prices: Price series
            method: 'simple', 'log', or 'percentage'
            
        Returns:
            Returns series
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        elif method == 'percentage':
            returns = (prices / prices.shift(1) - 1) * 100
        else:  # simple
            returns = prices.pct_change()
        
        return returns.dropna()
    
    def detect_outliers(self, 
                       series: pd.Series,
                       method: Optional[str] = None) -> pd.Series:
        """
        Detect outliers in time series
        
        Args:
            series: Input series
            method: Override default outlier method
            
        Returns:
            Boolean series indicating outliers
        """
        method = method or self.outlier_method
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > self.outlier_threshold
            
        elif method == 'percentile':
            lower_percentile = (100 - 99.7) / 2  # 99.7% confidence
            upper_percentile = 100 - lower_percentile
            lower_bound = series.quantile(lower_percentile / 100)
            upper_bound = series.quantile(upper_percentile / 100)
            outliers = (series < lower_bound) | (series > upper_bound)
            
        else:
            raise ValueError(f"Unknown outlier method: {method}")
        
        return outliers
    
    def handle_outliers(self, 
                       series: pd.Series,
                       method: str = 'winsorize') -> pd.Series:
        """
        Handle outliers in series
        
        Args:
            series: Input series
            method: 'remove', 'winsorize', or 'interpolate'
            
        Returns:
            Cleaned series
        """
        outliers = self.detect_outliers(series)
        
        if method == 'remove':
            return series[~outliers]
        
        elif method == 'winsorize':
            # Replace outliers with 5th/95th percentiles
            lower_cap = series.quantile(0.05)
            upper_cap = series.quantile(0.95)
            series_clean = series.copy()
            series_clean[series < lower_cap] = lower_cap
            series_clean[series > upper_cap] = upper_cap
            return series_clean
        
        elif method == 'interpolate':
            series_clean = series.copy()
            series_clean[outliers] = np.nan
            return series_clean.interpolate()
        
        else:
            raise ValueError(f"Unknown outlier handling method: {method}")
    
    def calculate_volatility(self, 
                           returns: pd.Series,
                           window: int = 20,
                           method: str = 'rolling') -> pd.Series:
        """
        Calculate volatility measures
        
        Args:
            returns: Returns series
            window: Rolling window size
            method: 'rolling', 'ewm', or 'garch'
            
        Returns:
            Volatility series
        """
        if method == 'rolling':
            return returns.rolling(window=window).std()
        
        elif method == 'ewm':
            # Exponentially weighted moving average
            return returns.ewm(span=window).std()
        
        elif method == 'parkinson':
            # Parkinson volatility estimator (requires OHLC data)
            # This is a placeholder - would need high/low prices
            return returns.rolling(window=window).std()
        
        else:
            raise ValueError(f"Unknown volatility method: {method}")
    
    def add_technical_features(self, 
                             prices: pd.Series,
                             volume: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        Add technical analysis features
        
        Args:
            prices: Price series
            volume: Volume series (optional)
            
        Returns:
            Dictionary of feature series
        """
        features = {}
        
        # Moving averages
        features['sma_5'] = prices.rolling(5).mean()
        features['sma_20'] = prices.rolling(20).mean()
        features['sma_50'] = prices.rolling(50).mean()
        
        # Exponential moving averages
        features['ema_12'] = prices.ewm(span=12).mean()
        features['ema_26'] = prices.ewm(span=26).mean()
        
        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma_20 = features['sma_20']
        std_20 = prices.rolling(20).std()
        features['bb_upper'] = sma_20 + (2 * std_20)
        features['bb_lower'] = sma_20 - (2 * std_20)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (prices - features['bb_lower']) / features['bb_width']
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Price momentum
        features['momentum_5'] = prices / prices.shift(5) - 1
        features['momentum_10'] = prices / prices.shift(10) - 1
        features['momentum_20'] = prices / prices.shift(20) - 1
        
        # Volatility features
        returns = self.calculate_returns(prices)
        features['volatility_5'] = returns.rolling(5).std()
        features['volatility_20'] = returns.rolling(20).std()
        features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
        
        # Volume features (if available)
        if volume is not None:
            features['volume_sma'] = volume.rolling(20).mean()
            features['volume_ratio'] = volume / features['volume_sma']
            features['price_volume'] = prices * volume
            features['vwap'] = features['price_volume'].rolling(20).sum() / volume.rolling(20).sum()
        
        return features
    
    def resample_data(self, 
                     data: pd.DataFrame,
                     target_frequency: str) -> pd.DataFrame:
        """
        Resample data to different frequency
        
        Args:
            data: Input DataFrame with datetime index
            target_frequency: Target frequency ('D', 'H', '5min', etc.)
            
        Returns:
            Resampled DataFrame
        """
        # OHLCV aggregation rules
        agg_rules = {
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Apply rules only to columns that exist
        applicable_rules = {col: rule for col, rule in agg_rules.items() 
                          if col in data.columns}
        
        if not applicable_rules:
            # If no standard OHLCV columns, use last value
            return data.resample(target_frequency).last()
        
        return data.resample(target_frequency).agg(applicable_rules)
    
    def preprocess(self, 
                  symbol: str,
                  data: Optional[pd.DataFrame] = None,
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  add_features: bool = True,
                  clean_outliers: bool = True) -> PreprocessedData:
        """
        Comprehensive data preprocessing pipeline
        
        Args:
            symbol: Stock symbol
            data: Optional pre-loaded data
            start_date: Start date for fetching data
            end_date: End date for fetching data
            add_features: Whether to add technical features
            clean_outliers: Whether to clean outliers
            
        Returns:
            PreprocessedData object
        """
        # Fetch data if not provided
        if data is None:
            data = self.fetch_yahoo_data(symbol, start_date, end_date)
        
        if len(data) < self.min_periods:
            raise ValueError(f"Insufficient data: {len(data)} < {self.min_periods}")
        
        # Extract price series (prefer Close, fallback to first numeric column)
        if 'Close' in data.columns:
            prices = data['Close']
        elif 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            # Find first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric price columns found")
            prices = data[numeric_cols[0]]
        
        prices.name = symbol
        
        # Extract volume if available
        volume = data.get('Volume', None)
        
        # Clean outliers in prices
        if clean_outliers:
            try:
                prices = self.handle_outliers(prices, method='winsorize')
            except Exception as e:
                print(f"Warning: Could not clean outliers in prices: {e}")
        
        # Calculate returns
        simple_returns = self.calculate_returns(prices, method='simple')
        log_returns = self.calculate_returns(prices, method='log')
        
        # Clean outliers in returns
        if clean_outliers:
            try:
                simple_returns = self.handle_outliers(simple_returns, method='winsorize')
                log_returns = self.handle_outliers(log_returns, method='winsorize')
            except Exception as e:
                print(f"Warning: Could not clean outliers in returns: {e}")
        
        # Calculate volatility
        volatility = self.calculate_volatility(simple_returns)
        
        # Add technical features
        features = {}
        if add_features:
            features = self.add_technical_features(prices, volume)
        
        # Metadata
        metadata = {
            'symbol': symbol,
            'start_date': data.index[0],
            'end_date': data.index[-1],
            'total_periods': len(data),
            'frequency': self.frequency,
            'outliers_cleaned': clean_outliers,
            'features_added': add_features,
            'missing_values': int(prices.isna().sum()) if hasattr(prices, 'isna') else 0,
            'price_range': (prices.min(), prices.max()),
            'avg_daily_return': simple_returns.mean(),
            'avg_daily_volatility': volatility.mean() if not volatility.empty else 0,
            'preprocessing_timestamp': datetime.now()
        }
        
        return PreprocessedData(
            symbol=symbol,
            raw_prices=prices,
            returns=simple_returns,
            log_returns=log_returns,
            volatility=volatility,
            volume=volume,
            features=features,
            metadata=metadata
        )
    
    def validate_data_quality(self, data: PreprocessedData) -> Dict[str, any]:
        """
        Validate data quality and provide diagnostics
        
        Args:
            data: Preprocessed data object
            
        Returns:
            Quality assessment dictionary
        """
        quality_report = {
            'symbol': data.symbol,
            'total_observations': len(data.raw_prices),
            'date_range': (data.metadata['start_date'], data.metadata['end_date']),
            'issues': [],
            'warnings': [],
            'quality_score': 1.0  # Start with perfect score
        }
        
        # Check for missing values
        missing_prices = data.raw_prices.isna().sum()
        missing_returns = data.returns.isna().sum()
        
        if missing_prices > 0:
            quality_report['issues'].append(f"Missing prices: {missing_prices}")
            quality_report['quality_score'] -= 0.1
        
        if missing_returns > 0:
            quality_report['issues'].append(f"Missing returns: {missing_returns}")
            quality_report['quality_score'] -= 0.1
        
        # Check for extreme returns
        extreme_returns = (abs(data.returns) > 0.2).sum()  # >20% daily moves
        if extreme_returns > len(data.returns) * 0.05:  # >5% of observations
            quality_report['warnings'].append(f"High extreme returns: {extreme_returns}")
            quality_report['quality_score'] -= 0.05
        
        # Check data continuity
        date_gaps = pd.Series(data.raw_prices.index).diff().dt.days
        large_gaps = (date_gaps > 7).sum()  # Gaps > 1 week
        
        if large_gaps > 0:
            quality_report['warnings'].append(f"Date gaps detected: {large_gaps}")
            quality_report['quality_score'] -= 0.05
        
        # Check sufficient data length
        if len(data.raw_prices) < 50:
            quality_report['issues'].append(f"Insufficient data: {len(data.raw_prices)}")
            quality_report['quality_score'] -= 0.2
        
        # Overall assessment
        if quality_report['quality_score'] >= 0.8:
            quality_report['assessment'] = 'EXCELLENT'
        elif quality_report['quality_score'] >= 0.6:
            quality_report['assessment'] = 'GOOD'
        elif quality_report['quality_score'] >= 0.4:
            quality_report['assessment'] = 'FAIR'
        else:
            quality_report['assessment'] = 'POOR'
        
        return quality_report