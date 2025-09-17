"""
Valuation and Technical Analysis Tools for Portfolio Construction
==============================================================

Tools for quantitative valuation, technical analysis, and risk assessment.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from crewai_tools import BaseTool
from loguru import logger
import talib

class PriceVolumeAnalyzerTool(BaseTool):
    """Tool for analyzing price and volume patterns."""
    
    name: str = "price_volume_analyzer"
    description: str = "Analyze stock price movements, volume patterns, and trading activity"
    
    def _run(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Analyze price and volume data for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                return {'error': 'No historical data available', 'symbol': symbol}
            
            # Price analysis
            price_analysis = self._analyze_price_patterns(hist_data)
            
            # Volume analysis
            volume_analysis = self._analyze_volume_patterns(hist_data)
            
            # Price-volume relationship
            pv_relationship = self._analyze_price_volume_relationship(hist_data)
            
            # Recent performance
            recent_performance = self._calculate_recent_performance(hist_data)
            
            return {
                'symbol': symbol,
                'price_analysis': price_analysis,
                'volume_analysis': volume_analysis,
                'price_volume_relationship': pv_relationship,
                'recent_performance': recent_performance,
                'data_period': period,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing price/volume for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _analyze_price_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price movement patterns."""
        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']
        
        return {
            'current_price': float(close_prices.iloc[-1]),
            'price_range_52w': {
                'high': float(high_prices.max()),
                'low': float(low_prices.min())
            },
            'volatility': float(close_prices.pct_change().std() * np.sqrt(252)),  # Annualized
            'average_daily_return': float(close_prices.pct_change().mean()),
            'price_trend': self._determine_price_trend(close_prices),
            'support_resistance': self._find_support_resistance_levels(close_prices),
            'price_momentum': self._calculate_price_momentum(close_prices)
        }
    
    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume trading patterns."""
        volume = data['Volume']
        close_prices = data['Close']
        
        # Volume moving averages
        volume_ma_20 = volume.rolling(20).mean()
        volume_ma_50 = volume.rolling(50).mean()
        
        return {
            'average_volume': float(volume.mean()),
            'recent_volume': float(volume.iloc[-1]),
            'volume_trend': 'increasing' if volume.iloc[-10:].mean() > volume.iloc[-20:-10].mean() else 'decreasing',
            'volume_ratio_20d': float(volume.iloc[-1] / volume_ma_20.iloc[-1]) if not pd.isna(volume_ma_20.iloc[-1]) else 1.0,
            'volume_spike_days': self._identify_volume_spikes(volume),
            'on_balance_volume': self._calculate_obv(close_prices, volume)
        }
    
    def _analyze_price_volume_relationship(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price-volume relationship patterns."""
        price_changes = data['Close'].pct_change()
        volume = data['Volume']
        
        # Price-Volume correlation
        correlation = price_changes.corr(volume)
        
        # Volume-weighted average price (VWAP)
        vwap = (data['Close'] * data['Volume']).sum() / data['Volume'].sum()
        
        return {
            'price_volume_correlation': float(correlation) if not pd.isna(correlation) else 0,
            'vwap': float(vwap),
            'current_vs_vwap': float((data['Close'].iloc[-1] / vwap - 1) * 100),
            'accumulation_distribution': self._calculate_accumulation_distribution(data)
        }
    
    def _calculate_recent_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate recent performance metrics."""
        close_prices = data['Close']
        
        performance = {}
        periods = [1, 5, 10, 21, 63, 252]  # 1D, 1W, 2W, 1M, 3M, 1Y
        period_names = ['1d', '1w', '2w', '1m', '3m', '1y']
        
        for period, name in zip(periods, period_names):
            if len(close_prices) > period:
                performance[name] = float((close_prices.iloc[-1] / close_prices.iloc[-period-1] - 1) * 100)
        
        return performance
    
    def _determine_price_trend(self, prices: pd.Series) -> str:
        """Determine overall price trend."""
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()
        
        if len(prices) < 50:
            return 'insufficient_data'
        
        current_price = prices.iloc[-1]
        ma_20_current = ma_20.iloc[-1]
        ma_50_current = ma_50.iloc[-1]
        
        if current_price > ma_20_current > ma_50_current:
            return 'strong_uptrend'
        elif current_price > ma_20_current and ma_20_current < ma_50_current:
            return 'weak_uptrend'
        elif current_price < ma_20_current < ma_50_current:
            return 'strong_downtrend'
        elif current_price < ma_20_current and ma_20_current > ma_50_current:
            return 'weak_downtrend'
        else:
            return 'sideways'
    
    def _find_support_resistance_levels(self, prices: pd.Series) -> Dict[str, float]:
        """Find support and resistance levels."""
        recent_prices = prices.iloc[-60:]  # Last 60 days
        
        # Simple approach: use recent highs and lows
        resistance = float(recent_prices.max())
        support = float(recent_prices.min())
        
        return {
            'resistance': resistance,
            'support': support,
            'distance_to_resistance': float((resistance / prices.iloc[-1] - 1) * 100),
            'distance_to_support': float((prices.iloc[-1] / support - 1) * 100)
        }
    
    def _calculate_price_momentum(self, prices: pd.Series) -> float:
        """Calculate price momentum score."""
        if len(prices) < 10:
            return 0.0
        
        short_term_return = (prices.iloc[-1] / prices.iloc[-6] - 1)  # 5-day return
        medium_term_return = (prices.iloc[-1] / prices.iloc[-21] - 1)  # 20-day return
        
        momentum = (short_term_return * 0.6 + medium_term_return * 0.4) * 100
        return float(momentum)
    
    def _identify_volume_spikes(self, volume: pd.Series) -> int:
        """Identify days with volume spikes in recent period."""
        if len(volume) < 20:
            return 0
        
        volume_ma = volume.rolling(20).mean()
        recent_volume = volume.iloc[-10:]
        recent_ma = volume_ma.iloc[-10:]
        
        spikes = (recent_volume > recent_ma * 2).sum()
        return int(spikes)
    
    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> float:
        """Calculate On-Balance Volume."""
        price_changes = prices.diff()
        obv = 0
        
        for i in range(1, len(prices)):
            if price_changes.iloc[i] > 0:
                obv += volume.iloc[i]
            elif price_changes.iloc[i] < 0:
                obv -= volume.iloc[i]
        
        return float(obv)
    
    def _calculate_accumulation_distribution(self, data: pd.DataFrame) -> float:
        """Calculate Accumulation/Distribution Line."""
        high_low_close = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        high_low_close = high_low_close.fillna(0)
        
        ad_line = (high_low_close * data['Volume']).cumsum()
        return float(ad_line.iloc[-1])

class TechnicalIndicatorCalculatorTool(BaseTool):
    """Tool for calculating technical indicators."""
    
    name: str = "technical_indicator_calculator"
    description: str = "Calculate comprehensive technical indicators for stock analysis"
    
    def _run(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Calculate technical indicators for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                return {'error': 'No historical data available', 'symbol': symbol}
            
            # Calculate various technical indicators
            indicators = {
                'trend_indicators': self._calculate_trend_indicators(hist_data),
                'momentum_indicators': self._calculate_momentum_indicators(hist_data),
                'volatility_indicators': self._calculate_volatility_indicators(hist_data),
                'volume_indicators': self._calculate_volume_indicators(hist_data)
            }
            
            # Overall technical score
            technical_score = self._calculate_technical_score(indicators)
            
            return {
                'symbol': symbol,
                'technical_indicators': indicators,
                'technical_score': technical_score,
                'signal_summary': self._generate_signal_summary(indicators),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend-following indicators."""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        indicators = {}
        
        # Moving Averages
        if len(close) >= 20:
            indicators['sma_20'] = float(np.mean(close[-20:]))
        if len(close) >= 50:
            indicators['sma_50'] = float(np.mean(close[-50:]))
        if len(close) >= 200:
            indicators['sma_200'] = float(np.mean(close[-200:]))
        
        # Exponential Moving Averages
        if len(close) >= 12:
            indicators['ema_12'] = float(pd.Series(close).ewm(span=12).mean().iloc[-1])
        if len(close) >= 26:
            indicators['ema_26'] = float(pd.Series(close).ewm(span=26).mean().iloc[-1])
        
        # MACD
        if len(close) >= 26:
            ema_12 = pd.Series(close).ewm(span=12).mean()
            ema_26 = pd.Series(close).ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            indicators['macd'] = {
                'macd_line': float(macd_line.iloc[-1]),
                'signal_line': float(signal_line.iloc[-1]),
                'histogram': float(histogram.iloc[-1])
            }
        
        # Bollinger Bands
        if len(close) >= 20:
            sma_20 = pd.Series(close).rolling(20).mean()
            std_20 = pd.Series(close).rolling(20).std()
            
            indicators['bollinger_bands'] = {
                'upper_band': float(sma_20.iloc[-1] + 2 * std_20.iloc[-1]),
                'middle_band': float(sma_20.iloc[-1]),
                'lower_band': float(sma_20.iloc[-1] - 2 * std_20.iloc[-1]),
                'position': self._calculate_bb_position(close[-1], sma_20.iloc[-1], std_20.iloc[-1])
            }
        
        return indicators
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum indicators."""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        indicators = {}
        
        # RSI
        if len(close) >= 14:
            rsi = self._calculate_rsi(close, 14)
            indicators['rsi'] = float(rsi[-1])
        
        # Stochastic Oscillator
        if len(close) >= 14:
            k_percent, d_percent = self._calculate_stochastic(high, low, close, 14, 3)
            indicators['stochastic'] = {
                'k_percent': float(k_percent[-1]),
                'd_percent': float(d_percent[-1])
            }
        
        # Williams %R
        if len(close) >= 14:
            williams_r = self._calculate_williams_r(high, low, close, 14)
            indicators['williams_r'] = float(williams_r[-1])
        
        # Rate of Change
        if len(close) >= 10:
            roc = ((close[-1] / close[-10]) - 1) * 100
            indicators['rate_of_change'] = float(roc)
        
        return indicators
    
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility indicators."""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        indicators = {}
        
        # Average True Range (ATR)
        if len(close) >= 14:
            atr = self._calculate_atr(high, low, close, 14)
            indicators['atr'] = float(atr[-1])
        
        # Historical Volatility
        if len(close) >= 30:
            returns = pd.Series(close).pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            indicators['historical_volatility'] = float(volatility)
        
        return indicators
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators."""
        close = data['Close'].values
        volume = data['Volume'].values
        
        indicators = {}
        
        # Volume Moving Average
        if len(volume) >= 20:
            volume_ma_20 = np.mean(volume[-20:])
            indicators['volume_ma_20'] = float(volume_ma_20)
            indicators['volume_ratio'] = float(volume[-1] / volume_ma_20)
        
        # Money Flow Index (MFI)
        if len(close) >= 14:
            mfi = self._calculate_mfi(data, 14)
            indicators['money_flow_index'] = float(mfi[-1])
        
        return indicators
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(period).mean()
        avg_losses = pd.Series(losses).rolling(period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.values
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                             k_period: int, d_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator."""
        highest_high = pd.Series(high).rolling(k_period).max()
        lowest_low = pd.Series(low).rolling(k_period).min()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()
        
        return k_percent.values, d_percent.values
    
    def _calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Williams %R."""
        highest_high = pd.Series(high).rolling(period).max()
        lowest_low = pd.Series(low).rolling(period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r.values
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(true_range).rolling(period).mean()
        
        return atr.values
    
    def _calculate_mfi(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Money Flow Index."""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        price_change = typical_price.diff()
        positive_flow = money_flow.where(price_change > 0, 0).rolling(period).sum()
        negative_flow = money_flow.where(price_change < 0, 0).rolling(period).sum()
        
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi.values
    
    def _calculate_bb_position(self, current_price: float, sma: float, std: float) -> str:
        """Calculate position relative to Bollinger Bands."""
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        if current_price > upper_band:
            return 'above_upper'
        elif current_price < lower_band:
            return 'below_lower'
        elif current_price > sma:
            return 'upper_half'
        else:
            return 'lower_half'
    
    def _calculate_technical_score(self, indicators: Dict) -> float:
        """Calculate overall technical score (1-10)."""
        score = 5.0  # Base score
        
        # RSI scoring
        if 'rsi' in indicators.get('momentum_indicators', {}):
            rsi = indicators['momentum_indicators']['rsi']
            if 30 <= rsi <= 70:  # Neutral zone
                score += 0.5
            elif rsi < 30:  # Oversold (potentially bullish)
                score += 1.0
            elif rsi > 70:  # Overbought (potentially bearish)
                score -= 1.0
        
        # MACD scoring
        if 'macd' in indicators.get('trend_indicators', {}):
            macd_data = indicators['trend_indicators']['macd']
            if macd_data['histogram'] > 0:
                score += 0.5
        
        # Bollinger Bands scoring
        if 'bollinger_bands' in indicators.get('trend_indicators', {}):
            bb_position = indicators['trend_indicators']['bollinger_bands']['position']
            if bb_position == 'below_lower':
                score += 0.5  # Potentially oversold
            elif bb_position == 'above_upper':
                score -= 0.5  # Potentially overbought
        
        return min(10.0, max(1.0, score))
    
    def _generate_signal_summary(self, indicators: Dict) -> Dict[str, str]:
        """Generate signal summary from indicators."""
        signals = {}
        
        # RSI signal
        if 'rsi' in indicators.get('momentum_indicators', {}):
            rsi = indicators['momentum_indicators']['rsi']
            if rsi < 30:
                signals['rsi'] = 'oversold_bullish'
            elif rsi > 70:
                signals['rsi'] = 'overbought_bearish'
            else:
                signals['rsi'] = 'neutral'
        
        # MACD signal
        if 'macd' in indicators.get('trend_indicators', {}):
            histogram = indicators['trend_indicators']['macd']['histogram']
            signals['macd'] = 'bullish' if histogram > 0 else 'bearish'
        
        return signals

class VolatilityCalculatorTool(BaseTool):
    """Tool for calculating various volatility metrics."""
    
    name: str = "volatility_calculator"
    description: str = "Calculate volatility metrics including historical, implied, and GARCH models"
    
    def _run(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Calculate volatility metrics for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                return {'error': 'No historical data available', 'symbol': symbol}
            
            close_prices = hist_data['Close']
            returns = close_prices.pct_change().dropna()
            
            volatility_metrics = {
                'historical_volatility': self._calculate_historical_volatility(returns),
                'volatility_regimes': self._identify_volatility_regimes(returns),
                'volatility_forecast': self._forecast_volatility(returns),
                'risk_metrics': self._calculate_risk_metrics(returns, close_prices)
            }
            
            return {
                'symbol': symbol,
                'volatility_metrics': volatility_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _calculate_historical_volatility(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate various historical volatility measures."""
        return {
            'daily_volatility': float(returns.std()),
            'weekly_volatility': float(returns.std() * np.sqrt(5)),
            'monthly_volatility': float(returns.std() * np.sqrt(21)),
            'annual_volatility': float(returns.std() * np.sqrt(252)),
            'realized_volatility_30d': float(returns.tail(30).std() * np.sqrt(252)),
            'realized_volatility_60d': float(returns.tail(60).std() * np.sqrt(252)),
            'realized_volatility_90d': float(returns.tail(90).std() * np.sqrt(252))
        }
    
    def _identify_volatility_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """Identify high/low volatility regimes."""
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        vol_median = rolling_vol.median()
        
        current_vol = rolling_vol.iloc[-1]
        vol_percentile = (rolling_vol <= current_vol).mean() * 100
        
        if current_vol > vol_median * 1.5:
            regime = 'high_volatility'
        elif current_vol < vol_median * 0.7:
            regime = 'low_volatility'
        else:
            regime = 'normal_volatility'
        
        return {
            'current_regime': regime,
            'current_volatility': float(current_vol),
            'median_volatility': float(vol_median),
            'volatility_percentile': float(vol_percentile)
        }
    
    def _forecast_volatility(self, returns: pd.Series) -> Dict[str, float]:
        """Simple volatility forecasting."""
        # EWMA volatility forecast
        lambda_param = 0.94
        ewma_var = returns.var()
        
        for ret in returns:
            ewma_var = lambda_param * ewma_var + (1 - lambda_param) * ret**2
        
        ewma_vol = np.sqrt(ewma_var * 252)
        
        # Simple GARCH-like forecast
        long_term_vol = returns.std() * np.sqrt(252)
        
        return {
            'ewma_forecast': float(ewma_vol),
            'long_term_forecast': float(long_term_vol),
            'volatility_trend': 'increasing' if ewma_vol > long_term_vol else 'decreasing'
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series, prices: pd.Series) -> Dict[str, Any]:
        """Calculate risk metrics."""
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Maximum Drawdown
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        
        return {
            'var_95': float(var_95 * 100),  # 1-day VaR at 95% confidence
            'var_99': float(var_99 * 100),  # 1-day VaR at 99% confidence
            'max_drawdown': float(max_drawdown * 100),
            'downside_deviation': float(downside_deviation * np.sqrt(252) * 100),
            'upside_volatility': float(returns[returns > 0].std() * np.sqrt(252) * 100) if (returns > 0).any() else 0
        }

class SharpeRatioCalculatorTool(BaseTool):
    """Tool for calculating Sharpe ratio and risk-adjusted returns."""
    
    name: str = "sharpe_ratio_calculator"
    description: str = "Calculate Sharpe ratio and other risk-adjusted performance metrics"
    
    def _run(self, symbol: str, risk_free_rate: float = 0.05, period: str = "1y") -> Dict[str, Any]:
        """Calculate Sharpe ratio and risk-adjusted metrics."""
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                return {'error': 'No historical data available', 'symbol': symbol}
            
            close_prices = hist_data['Close']
            returns = close_prices.pct_change().dropna()
            
            # Calculate risk-adjusted metrics
            risk_adjusted_metrics = {
                'sharpe_ratio': self._calculate_sharpe_ratio(returns, risk_free_rate),
                'sortino_ratio': self._calculate_sortino_ratio(returns, risk_free_rate),
                'calmar_ratio': self._calculate_calmar_ratio(returns, close_prices),
                'information_ratio': self._calculate_information_ratio(returns),
                'treynor_ratio': self._calculate_treynor_ratio(symbol, returns, risk_free_rate)
            }
            
            return {
                'symbol': symbol,
                'risk_adjusted_metrics': risk_adjusted_metrics,
                'period_analyzed': period,
                'risk_free_rate_used': risk_free_rate,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return float(sharpe_ratio)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
        return float(sortino_ratio)
    
    def _calculate_calmar_ratio(self, returns: pd.Series, prices: pd.Series) -> float:
        """Calculate Calmar ratio (return/max drawdown)."""
        annual_return = (1 + returns.mean())**252 - 1
        
        # Calculate maximum drawdown
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        if max_drawdown == 0:
            return 0.0
        
        calmar_ratio = annual_return / max_drawdown
        return float(calmar_ratio)
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate Information ratio (active return / tracking error)."""
        # Using market return as benchmark (simplified with 0 for demonstration)
        benchmark_return = 0  # This would be actual benchmark returns in production
        active_returns = returns - benchmark_return
        
        if active_returns.std() == 0:
            return 0.0
        
        information_ratio = (active_returns.mean() / active_returns.std()) * np.sqrt(252)
        return float(information_ratio)
    
    def _calculate_treynor_ratio(self, symbol: str, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Treynor ratio (requires beta calculation)."""
        try:
            # Get market data (using SPY as market proxy)
            market = yf.Ticker("SPY")
            market_data = market.history(period="1y")
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Align dates
            aligned_returns = returns.align(market_returns, join='inner')
            stock_returns_aligned, market_returns_aligned = aligned_returns
            
            if len(stock_returns_aligned) < 30:  # Need sufficient data
                return 0.0
            
            # Calculate beta
            covariance = stock_returns_aligned.cov(market_returns_aligned)
            market_variance = market_returns_aligned.var()
            beta = covariance / market_variance if market_variance != 0 else 1.0
            
            # Calculate Treynor ratio
            excess_return = stock_returns_aligned.mean() * 252 - risk_free_rate
            treynor_ratio = excess_return / beta if beta != 0 else 0.0
            
            return float(treynor_ratio)
            
        except Exception as e:
            logger.warning(f"Could not calculate Treynor ratio for {symbol}: {e}")
            return 0.0

class CorrelationAnalyzerTool(BaseTool):
    """Tool for analyzing correlations with market and sectors."""
    
    name: str = "correlation_analyzer"
    description: str = "Analyze correlations with market indices, sectors, and other assets"
    
    def _run(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Analyze correlations for a stock."""
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            stock_data = ticker.history(period=period)
            
            if stock_data.empty:
                return {'error': 'No historical data available', 'symbol': symbol}
            
            stock_returns = stock_data['Close'].pct_change().dropna()
            
            # Get benchmark data
            benchmarks = self._get_benchmark_data(period)
            
            # Calculate correlations
            correlations = self._calculate_correlations(stock_returns, benchmarks)
            
            # Calculate rolling correlations
            rolling_correlations = self._calculate_rolling_correlations(stock_returns, benchmarks)
            
            return {
                'symbol': symbol,
                'correlations': correlations,
                'rolling_correlations': rolling_correlations,
                'correlation_analysis': self._analyze_correlations(correlations),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing correlations for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _get_benchmark_data(self, period: str) -> Dict[str, pd.Series]:
        """Get benchmark data for correlation analysis."""
        benchmarks = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'IWM': 'Russell 2000',
            'VTI': 'Total Market',
            'TLT': '20+ Year Treasury',
            'GLD': 'Gold',
            'VIX': 'Volatility Index'
        }
        
        benchmark_data = {}
        
        for symbol, name in benchmarks.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                if not data.empty:
                    benchmark_data[name] = data['Close'].pct_change().dropna()
            except:
                continue
        
        return benchmark_data
    
    def _calculate_correlations(self, stock_returns: pd.Series, benchmarks: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate static correlations."""
        correlations = {}
        
        for name, benchmark_returns in benchmarks.items():
            try:
                # Align the series
                aligned = stock_returns.align(benchmark_returns, join='inner')
                stock_aligned, benchmark_aligned = aligned
                
                if len(stock_aligned) > 30:  # Need sufficient data
                    correlation = stock_aligned.corr(benchmark_aligned)
                    correlations[name] = float(correlation) if not pd.isna(correlation) else 0.0
                    
            except Exception as e:
                logger.warning(f"Could not calculate correlation with {name}: {e}")
        
        return correlations
    
    def _calculate_rolling_correlations(self, stock_returns: pd.Series, benchmarks: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """Calculate rolling correlations."""
        rolling_correlations = {}
        
        # Focus on SPY for rolling correlation
        if 'S&P 500' in benchmarks:
            spy_returns = benchmarks['S&P 500']
            aligned = stock_returns.align(spy_returns, join='inner')
            stock_aligned, spy_aligned = aligned
            
            if len(stock_aligned) > 60:
                rolling_corr_30 = stock_aligned.rolling(30).corr(spy_aligned)
                rolling_corr_60 = stock_aligned.rolling(60).corr(spy_aligned)
                
                rolling_correlations['SPY'] = {
                    'rolling_30d': float(rolling_corr_30.iloc[-1]) if not pd.isna(rolling_corr_30.iloc[-1]) else 0.0,
                    'rolling_60d': float(rolling_corr_60.iloc[-1]) if not pd.isna(rolling_corr_60.iloc[-1]) else 0.0,
                    'correlation_trend': self._determine_correlation_trend(rolling_corr_30)
                }
        
        return rolling_correlations
    
    def _determine_correlation_trend(self, rolling_corr: pd.Series) -> str:
        """Determine if correlation is increasing or decreasing."""
        if len(rolling_corr.dropna()) < 10:
            return 'insufficient_data'
        
        recent_corr = rolling_corr.dropna().tail(10).mean()
        earlier_corr = rolling_corr.dropna().tail(20).head(10).mean()
        
        if recent_corr > earlier_corr + 0.1:
            return 'increasing'
        elif recent_corr < earlier_corr - 0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_correlations(self, correlations: Dict[str, float]) -> Dict[str, Any]:
        """Analyze correlation patterns."""
        analysis = {
            'diversification_benefit': 'high',  # Default
            'market_exposure': 'moderate',
            'defensive_characteristics': False
        }
        
        if 'S&P 500' in correlations:
            spy_corr = correlations['S&P 500']
            
            if spy_corr > 0.8:
                analysis['market_exposure'] = 'high'
                analysis['diversification_benefit'] = 'low'
            elif spy_corr < 0.3:
                analysis['market_exposure'] = 'low'
                analysis['diversification_benefit'] = 'high'
            
            if spy_corr < 0:
                analysis['defensive_characteristics'] = True
        
        return analysis

# Initialize tools
price_volume_analyzer = PriceVolumeAnalyzerTool()
technical_indicator_calculator = TechnicalIndicatorCalculatorTool()
volatility_calculator = VolatilityCalculatorTool()
sharpe_ratio_calculator = SharpeRatioCalculatorTool()
correlation_analyzer = CorrelationAnalyzerTool()