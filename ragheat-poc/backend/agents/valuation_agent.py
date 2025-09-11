from crewai import Agent, Task
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ValuationAgent:
    '''Agent responsible for technical analysis and valuation'''

    def __init__(self):
        self.agent = Agent(
            role='Valuation Analyst',
            goal='Analyze stock valuation and technical indicators',
            backstory='''You are a quantitative analyst specializing in 
            stock valuation and technical analysis. You use mathematical 
            models, price patterns, and technical indicators to identify 
            trading opportunities and assess if stocks are over or undervalued.''',
            verbose=True,
            allow_delegation=False
        )

    def analyze_valuation(self, symbol: str) -> Dict:
        '''Perform valuation analysis on a stock'''
        try:
            stock = yf.Ticker(symbol)

            # Get historical data
            hist = stock.history(period="3mo")

            if hist.empty:
                return {'symbol': symbol, 'error': 'No data available', 'valuation_score': 0.5}

            # Calculate technical indicators
            valuation = {
                'symbol': symbol,
                'current_price': float(hist['Close'].iloc[-1]) if not hist.empty else 0,
                'volatility': float(hist['Close'].pct_change().std() * np.sqrt(252)) if len(hist) > 1 else 0,
                'rsi': self._calculate_rsi(hist['Close']) if len(hist) > 14 else 50,
                'macd': self._calculate_macd(hist['Close']) if len(hist) > 26 else {},
                'price_change_30d': float(
                    (hist['Close'].iloc[-1] / hist['Close'].iloc[-30] - 1) * 100
                ) if len(hist) > 30 else 0,
                'volume_avg': float(hist['Volume'].mean()) if 'Volume' in hist else 0,
                'valuation_score': 0.5  # Will be calculated
            }

            # Calculate valuation score
            valuation['valuation_score'] = self._calculate_valuation_score(valuation)

            return valuation

        except Exception as e:
            logger.error(f"Error analyzing valuation for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e), 'valuation_score': 0.5}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        '''Calculate RSI indicator'''
        if len(prices) < period:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not rsi.empty else 50

    def _calculate_macd(self, prices: pd.Series) -> Dict:
        '''Calculate MACD indicator'''
        if len(prices) < 26:
            return {'macd': 0, 'signal': 0, 'histogram': 0}

        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        return {
            'macd': float(macd_line.iloc[-1]) if not macd_line.empty else 0,
            'signal': float(signal_line.iloc[-1]) if not signal_line.empty else 0,
            'histogram': float(macd_line.iloc[-1] - signal_line.iloc[-1]) 
                        if not macd_line.empty else 0
        }

    def _calculate_valuation_score(self, metrics: Dict) -> float:
        '''Calculate composite valuation score'''
        score = 0.5

        # RSI (30-70 is normal range)
        rsi = metrics.get('rsi', 50)
        if rsi < 30:
            score += 0.15  # Oversold
        elif rsi > 70:
            score -= 0.15  # Overbought

        # Price momentum
        price_change = metrics.get('price_change_30d', 0)
        if price_change > 10:
            score += 0.1
        elif price_change < -10:
            score -= 0.1

        # MACD signal
        macd_data = metrics.get('macd', {})
        if isinstance(macd_data, dict):
            histogram = macd_data.get('histogram', 0)
            if histogram > 0:
                score += 0.05
            else:
                score -= 0.05

        return max(0, min(1, score))