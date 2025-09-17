import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OptionAction(Enum):
    BUY_CALL = "BUY_CALL"
    BUY_PUT = "BUY_PUT" 
    SELL_CALL = "SELL_CALL"
    SELL_PUT = "SELL_PUT"
    HOLD = "HOLD"

class SignalStrength(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class OptionsSignal:
    symbol: str
    timestamp: datetime
    action: OptionAction
    strength: SignalStrength
    strike_price: float
    expiration: str
    probability: float
    heat_score: float
    pathrag_reasoning: str
    technical_indicators: Dict
    risk_reward: Dict
    market_context: Dict

class TechnicalIndicators:
    """Advanced technical indicators for options trading"""
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast=12, slow=26, signal=9) -> Dict:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1] if not macd_line.empty else 0,
            'signal': signal_line.iloc[-1] if not signal_line.empty else 0,
            'histogram': histogram.iloc[-1] if not histogram.empty else 0
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period=20, std_dev=2) -> Dict:
        """Bollinger Bands"""
        sma = TechnicalIndicators.calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        current_price = prices.iloc[-1] if not prices.empty else 0
        current_sma = sma.iloc[-1] if not sma.empty else 0
        current_upper = upper_band.iloc[-1] if not upper_band.empty else 0
        current_lower = lower_band.iloc[-1] if not lower_band.empty else 0
        
        return {
            'upper': current_upper,
            'middle': current_sma,
            'lower': current_lower,
            'price': current_price,
            'position': (current_price - current_lower) / (current_upper - current_lower) if current_upper != current_lower else 0.5
        }
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period=14) -> float:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period=14, d_period=3) -> Dict:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent.iloc[-1] if not k_percent.empty else 50,
            'd': d_percent.iloc[-1] if not d_percent.empty else 50
        }

class OptionsStrategy:
    """Advanced options trading strategy with PATHRAG reasoning"""
    
    def __init__(self, heat_threshold=0.6):
        self.heat_threshold = heat_threshold
        self.indicators = TechnicalIndicators()
        
    def get_market_data(self, symbol: str, period="5d", interval="5m") -> pd.DataFrame:
        """Fetch real-time market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_score(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical analysis score"""
        if data.empty:
            return {}
        
        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']
        volume = data['Volume']
        
        # Moving averages
        ma_5 = self.indicators.calculate_sma(close_prices, 5)
        ma_9 = self.indicators.calculate_sma(close_prices, 9)
        ma_20 = self.indicators.calculate_sma(close_prices, 20)
        
        # Current values
        current_price = close_prices.iloc[-1]
        current_ma5 = ma_5.iloc[-1] if not ma_5.empty else current_price
        current_ma9 = ma_9.iloc[-1] if not ma_9.empty else current_price
        current_ma20 = ma_20.iloc[-1] if not ma_20.empty else current_price
        
        # Technical indicators
        macd = self.indicators.calculate_macd(close_prices)
        bollinger = self.indicators.calculate_bollinger_bands(close_prices)
        rsi = self.indicators.calculate_rsi(close_prices)
        stoch = self.indicators.calculate_stochastic(high_prices, low_prices, close_prices)
        
        # Volume analysis
        avg_volume = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        volume_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1
        
        return {
            'price': current_price,
            'ma_5': current_ma5,
            'ma_9': current_ma9,
            'ma_20': current_ma20,
            'ma_alignment': self._analyze_ma_alignment(current_price, current_ma5, current_ma9, current_ma20),
            'macd': macd,
            'bollinger': bollinger,
            'rsi': rsi,
            'stochastic': stoch,
            'volume_ratio': volume_ratio,
            'trend_strength': self._calculate_trend_strength(data),
            'support_resistance': self._find_support_resistance(data)
        }
    
    def _analyze_ma_alignment(self, price: float, ma5: float, ma9: float, ma20: float) -> Dict:
        """Analyze moving average alignment for trend confirmation"""
        bullish_alignment = price > ma5 > ma9 > ma20
        bearish_alignment = price < ma5 < ma9 < ma20
        
        score = 0
        if bullish_alignment:
            score = 1
        elif bearish_alignment:
            score = -1
        else:
            # Partial alignment
            if price > ma5:
                score += 0.3
            if ma5 > ma9:
                score += 0.3
            if ma9 > ma20:
                score += 0.4
            if price < ma5:
                score -= 0.3
            if ma5 < ma9:
                score -= 0.3
            if ma9 < ma20:
                score -= 0.4
        
        return {
            'score': score,
            'bullish': bullish_alignment,
            'bearish': bearish_alignment,
            'trend': 'bullish' if score > 0.5 else 'bearish' if score < -0.5 else 'neutral'
        }
    
    def _calculate_trend_strength(self, data: pd.DataFrame, period=20) -> float:
        """Calculate trend strength using price momentum"""
        if len(data) < period:
            return 0
        
        prices = data['Close'].iloc[-period:]
        
        # Linear regression slope
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize slope by average price
        avg_price = prices.mean()
        normalized_slope = slope / avg_price if avg_price > 0 else 0
        
        return min(max(normalized_slope * 100, -5), 5)  # Clamp between -5 and 5
    
    def _find_support_resistance(self, data: pd.DataFrame, window=10) -> Dict:
        """Find recent support and resistance levels"""
        if len(data) < window * 2:
            current_price = data['Close'].iloc[-1] if not data.empty else 0
            return {'support': current_price * 0.98, 'resistance': current_price * 1.02}
        
        highs = data['High']
        lows = data['Low']
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(highs) - window):
            if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                resistance_levels.append(highs.iloc[i])
            if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                support_levels.append(lows.iloc[i])
        
        current_price = data['Close'].iloc[-1]
        
        # Find nearest support and resistance
        support = max([s for s in support_levels if s < current_price], default=current_price * 0.98)
        resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.02)
        
        return {'support': support, 'resistance': resistance}
    
    def generate_pathrag_reasoning(self, symbol: str, technical_data: Dict, heat_score: float) -> str:
        """Generate PATHRAG (Pattern Analysis Technical Heat Reasoning AI Generated) explanation"""
        
        ma_trend = technical_data.get('ma_alignment', {}).get('trend', 'neutral')
        macd_signal = technical_data.get('macd', {})
        bollinger = technical_data.get('bollinger', {})
        rsi = technical_data.get('rsi', 50)
        volume_ratio = technical_data.get('volume_ratio', 1)
        trend_strength = technical_data.get('trend_strength', 0)
        
        reasoning_parts = []
        
        # Heat analysis
        if heat_score > 0.8:
            reasoning_parts.append(f"🔥 HIGH HEAT ({heat_score:.1%}): Exceptional market attention detected")
        elif heat_score > 0.6:
            reasoning_parts.append(f"🌡️ ELEVATED HEAT ({heat_score:.1%}): Increased market focus")
        else:
            reasoning_parts.append(f"❄️ LOW HEAT ({heat_score:.1%}): Limited market attention")
        
        # Moving average analysis
        if ma_trend == 'bullish':
            reasoning_parts.append("📈 MA BULLISH: Price above all key moving averages (5>9>20)")
        elif ma_trend == 'bearish':
            reasoning_parts.append("📉 MA BEARISH: Price below all key moving averages (5<9<20)")
        else:
            reasoning_parts.append("➡️ MA NEUTRAL: Mixed moving average signals")
        
        # MACD analysis
        if macd_signal.get('macd', 0) > macd_signal.get('signal', 0):
            reasoning_parts.append("⚡ MACD BULLISH: MACD line above signal line")
        else:
            reasoning_parts.append("⚡ MACD BEARISH: MACD line below signal line")
        
        # Bollinger Bands
        bb_position = bollinger.get('position', 0.5)
        if bb_position > 0.8:
            reasoning_parts.append("🎯 BB OVERBOUGHT: Price near upper Bollinger Band")
        elif bb_position < 0.2:
            reasoning_parts.append("🎯 BB OVERSOLD: Price near lower Bollinger Band")
        
        # RSI analysis
        if rsi > 70:
            reasoning_parts.append(f"📊 RSI OVERBOUGHT ({rsi:.1f}): Potential pullback risk")
        elif rsi < 30:
            reasoning_parts.append(f"📊 RSI OVERSOLD ({rsi:.1f}): Potential bounce opportunity")
        else:
            reasoning_parts.append(f"📊 RSI NEUTRAL ({rsi:.1f}): Balanced momentum")
        
        # Volume confirmation
        if volume_ratio > 1.5:
            reasoning_parts.append(f"📢 HIGH VOLUME ({volume_ratio:.1f}x): Strong conviction move")
        elif volume_ratio < 0.7:
            reasoning_parts.append(f"🔇 LOW VOLUME ({volume_ratio:.1f}x): Weak conviction")
        
        # Trend strength
        if abs(trend_strength) > 2:
            direction = "UPTREND" if trend_strength > 0 else "DOWNTREND"
            reasoning_parts.append(f"🚀 STRONG {direction}: Momentum strength {abs(trend_strength):.1f}")
        
        return " | ".join(reasoning_parts)
    
    def calculate_option_probability(self, technical_data: Dict, heat_score: float, action: OptionAction, time_to_expiry: int) -> float:
        """Calculate probability of successful options trade"""
        
        # Base probability factors
        prob_factors = {
            'heat_score': min(heat_score * 0.4, 0.4),  # Max 40% from heat
            'technical_score': 0,
            'volume_confirmation': 0,
            'time_decay_risk': max(0.1, (time_to_expiry - 1) / 30 * 0.2),  # Time decay consideration
            'volatility_boost': 0
        }
        
        # Technical analysis contribution (max 35%)
        ma_score = technical_data.get('ma_alignment', {}).get('score', 0)
        rsi = technical_data.get('rsi', 50)
        macd = technical_data.get('macd', {})
        trend_strength = technical_data.get('trend_strength', 0)
        
        technical_score = 0
        
        # Direction alignment
        if action in [OptionAction.BUY_CALL, OptionAction.SELL_PUT]:
            # Bullish trades
            if ma_score > 0.5:
                technical_score += 0.15
            if macd.get('macd', 0) > macd.get('signal', 0):
                technical_score += 0.1
            if trend_strength > 1:
                technical_score += 0.1
        elif action in [OptionAction.BUY_PUT, OptionAction.SELL_CALL]:
            # Bearish trades
            if ma_score < -0.5:
                technical_score += 0.15
            if macd.get('macd', 0) < macd.get('signal', 0):
                technical_score += 0.1
            if trend_strength < -1:
                technical_score += 0.1
        
        prob_factors['technical_score'] = technical_score
        
        # Volume confirmation (max 15%)
        volume_ratio = technical_data.get('volume_ratio', 1)
        if volume_ratio > 1.2:
            prob_factors['volume_confirmation'] = min(0.15, (volume_ratio - 1) * 0.3)
        
        # Volatility consideration (max 10%)
        bollinger = technical_data.get('bollinger', {})
        bb_position = bollinger.get('position', 0.5)
        if abs(bb_position - 0.5) > 0.3:  # High volatility
            prob_factors['volatility_boost'] = 0.1
        
        total_probability = sum(prob_factors.values())
        
        # Ensure probability is between 0.1 and 0.9
        return min(max(total_probability, 0.1), 0.9)
    
    def determine_option_action(self, technical_data: Dict, heat_score: float) -> Tuple[OptionAction, SignalStrength]:
        """Determine the best options action based on analysis"""
        
        ma_alignment = technical_data.get('ma_alignment', {})
        macd = technical_data.get('macd', {})
        rsi = technical_data.get('rsi', 50)
        bollinger = technical_data.get('bollinger', {})
        trend_strength = technical_data.get('trend_strength', 0)
        volume_ratio = technical_data.get('volume_ratio', 1)
        
        # Calculate bullish and bearish signals
        bullish_signals = 0
        bearish_signals = 0
        signal_strength = 0
        
        # Moving average signals
        if ma_alignment.get('score', 0) > 0.7:
            bullish_signals += 2
            signal_strength += 2
        elif ma_alignment.get('score', 0) > 0.3:
            bullish_signals += 1
            signal_strength += 1
        elif ma_alignment.get('score', 0) < -0.7:
            bearish_signals += 2
            signal_strength += 2
        elif ma_alignment.get('score', 0) < -0.3:
            bearish_signals += 1
            signal_strength += 1
        
        # MACD signals
        macd_diff = macd.get('macd', 0) - macd.get('signal', 0)
        if macd_diff > 0:
            bullish_signals += 1
            if macd.get('histogram', 0) > 0:
                bullish_signals += 0.5
        else:
            bearish_signals += 1
            if macd.get('histogram', 0) < 0:
                bearish_signals += 0.5
        
        # RSI signals
        if rsi < 30:
            bullish_signals += 1.5  # Oversold bounce
        elif rsi > 70:
            bearish_signals += 1.5  # Overbought pullback
        
        # Bollinger Band signals
        bb_position = bollinger.get('position', 0.5)
        if bb_position < 0.2:
            bullish_signals += 1  # Near lower band
        elif bb_position > 0.8:
            bearish_signals += 1  # Near upper band
        
        # Trend strength
        if trend_strength > 2:
            bullish_signals += 1.5
        elif trend_strength < -2:
            bearish_signals += 1.5
        
        # Heat score consideration
        if heat_score > 0.8:
            signal_strength += 1
        elif heat_score > 0.6:
            signal_strength += 0.5
        
        # Volume confirmation
        if volume_ratio > 1.5:
            signal_strength += 1
        
        # Determine action and strength
        net_signal = bullish_signals - bearish_signals
        
        if net_signal > 3 and signal_strength > 3:
            return OptionAction.BUY_CALL, SignalStrength.STRONG_BUY
        elif net_signal > 2 and signal_strength > 2:
            return OptionAction.BUY_CALL, SignalStrength.BUY
        elif net_signal > 1:
            return OptionAction.BUY_CALL, SignalStrength.WEAK_BUY
        elif net_signal < -3 and signal_strength > 3:
            return OptionAction.BUY_PUT, SignalStrength.STRONG_SELL
        elif net_signal < -2 and signal_strength > 2:
            return OptionAction.BUY_PUT, SignalStrength.SELL
        elif net_signal < -1:
            return OptionAction.BUY_PUT, SignalStrength.WEAK_SELL
        else:
            return OptionAction.HOLD, SignalStrength.NEUTRAL
    
    def calculate_strike_and_expiry(self, symbol: str, current_price: float, action: OptionAction, technical_data: Dict) -> Tuple[float, str]:
        """Calculate optimal strike price and expiration"""
        
        # Support/Resistance levels
        support_resistance = technical_data.get('support_resistance', {})
        support = support_resistance.get('support', current_price * 0.98)
        resistance = support_resistance.get('resistance', current_price * 1.02)
        
        # ATM, ITM, OTM calculation
        if action == OptionAction.BUY_CALL:
            # For calls, slightly OTM for better risk/reward
            strike_price = current_price * 1.01  # 1% OTM
            if resistance > current_price * 1.02:
                strike_price = min(strike_price, resistance * 0.98)
        elif action == OptionAction.BUY_PUT:
            # For puts, slightly OTM
            strike_price = current_price * 0.99  # 1% OTM
            if support < current_price * 0.98:
                strike_price = max(strike_price, support * 1.02)
        else:
            strike_price = current_price
        
        # Round to nearest 0.5 for typical option strikes
        strike_price = round(strike_price * 2) / 2
        
        # Expiration based on signal strength and trend
        trend_strength = abs(technical_data.get('trend_strength', 0))
        
        if trend_strength > 3:
            days_to_expiry = 7  # Strong trend, short expiry
        elif trend_strength > 1:
            days_to_expiry = 14  # Medium trend
        else:
            days_to_expiry = 21  # Weak trend, more time
        
        # Calculate expiration date (next Friday for weekly options)
        today = datetime.now()
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0:  # If today is Friday
            days_until_friday = 7
        
        expiry_date = today + timedelta(days=days_until_friday + (days_to_expiry // 7) * 7)
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        
        return strike_price, expiry_str
    
    def generate_options_signal(self, symbol: str, heat_score: float = None) -> OptionsSignal:
        """Generate comprehensive options trading signal"""
        
        # Get market data
        data = self.get_market_data(symbol)
        if data.empty:
            # Return neutral signal if no data
            return OptionsSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                action=OptionAction.HOLD,
                strength=SignalStrength.NEUTRAL,
                strike_price=0,
                expiration="",
                probability=0.5,
                heat_score=heat_score or 0,
                pathrag_reasoning="No market data available",
                technical_indicators={},
                risk_reward={},
                market_context={}
            )
        
        # Calculate technical indicators
        technical_data = self.calculate_technical_score(data)
        
        # Use provided heat score or calculate default
        if heat_score is None:
            heat_score = min(technical_data.get('volume_ratio', 1) * 0.3, 0.8)
        
        # Determine action and strength
        action, strength = self.determine_option_action(technical_data, heat_score)
        
        # Calculate strike price and expiration
        current_price = technical_data.get('price', 0)
        strike_price, expiration = self.calculate_strike_and_expiry(symbol, current_price, action, technical_data)
        
        # Calculate probability
        time_to_expiry = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).days if expiration else 7
        probability = self.calculate_option_probability(technical_data, heat_score, action, time_to_expiry)
        
        # Generate PATHRAG reasoning
        pathrag_reasoning = self.generate_pathrag_reasoning(symbol, technical_data, heat_score)
        
        # Calculate risk/reward
        risk_reward = self._calculate_risk_reward(current_price, strike_price, action, technical_data)
        
        # Market context
        market_context = self._get_market_context(technical_data)
        
        return OptionsSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            action=action,
            strength=strength,
            strike_price=strike_price,
            expiration=expiration,
            probability=probability,
            heat_score=heat_score,
            pathrag_reasoning=pathrag_reasoning,
            technical_indicators=technical_data,
            risk_reward=risk_reward,
            market_context=market_context
        )
    
    def _calculate_risk_reward(self, current_price: float, strike_price: float, action: OptionAction, technical_data: Dict) -> Dict:
        """Calculate risk/reward ratio for the options trade"""
        
        support_resistance = technical_data.get('support_resistance', {})
        support = support_resistance.get('support', current_price * 0.98)
        resistance = support_resistance.get('resistance', current_price * 1.02)
        
        if action == OptionAction.BUY_CALL:
            target = resistance
            stop_loss = current_price * 0.95
            risk = abs(current_price - stop_loss)
            reward = abs(target - strike_price) if target > strike_price else current_price * 0.02
        elif action == OptionAction.BUY_PUT:
            target = support
            stop_loss = current_price * 1.05
            risk = abs(stop_loss - current_price)
            reward = abs(strike_price - target) if target < strike_price else current_price * 0.02
        else:
            return {'risk': 0, 'reward': 0, 'ratio': 0}
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return {
            'risk': risk,
            'reward': reward,
            'ratio': risk_reward_ratio,
            'target': target,
            'stop_loss': stop_loss
        }
    
    def _get_market_context(self, technical_data: Dict) -> Dict:
        """Get broader market context"""
        
        return {
            'trend': technical_data.get('ma_alignment', {}).get('trend', 'neutral'),
            'volatility': 'high' if technical_data.get('bollinger', {}).get('position', 0.5) in [x for x in [0.1, 0.9] if abs(x - 0.5) > 0.3] else 'normal',
            'momentum': 'strong' if abs(technical_data.get('trend_strength', 0)) > 2 else 'weak',
            'volume_profile': 'high' if technical_data.get('volume_ratio', 1) > 1.5 else 'normal'
        }