from crewai import Agent, Task
from typing import Dict, List, Optional
import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FundamentalAgent:
    '''Agent responsible for fundamental analysis of stocks'''

    def __init__(self):
        self.agent = Agent(
            role='Fundamental Analyst',
            goal='Analyze company fundamentals and financial health',
            backstory='''You are an experienced fundamental analyst with 
            20 years of experience analyzing financial statements, 
            evaluating company performance, and identifying value opportunities.
            You focus on metrics like P/E ratio, revenue growth, debt levels, 
            and cash flow.''',
            verbose=True,
            allow_delegation=False
        )

    def analyze_stock(self, symbol: str) -> Dict:
        '''Perform fundamental analysis on a stock'''
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            # Extract key fundamental metrics
            fundamentals = {
                'symbol': symbol,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                'recommendation': info.get('recommendationKey', 'hold')
            }

            # Calculate fundamental score (0-1)
            score = self._calculate_fundamental_score(fundamentals)
            fundamentals['fundamental_score'] = score

            return fundamentals

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e), 'fundamental_score': 0.5}

    def _calculate_fundamental_score(self, metrics: Dict) -> float:
        '''Calculate a composite fundamental score'''
        score = 0.5  # Base score

        # P/E Ratio (lower is better, typically)
        pe = metrics.get('pe_ratio', 0)
        if 0 < pe < 15:
            score += 0.1
        elif 15 <= pe < 25:
            score += 0.05
        elif pe > 40:
            score -= 0.1

        # ROE (higher is better)
        roe = metrics.get('roe', 0)
        if roe > 0.20:
            score += 0.1
        elif roe > 0.15:
            score += 0.05

        # Debt to Equity (lower is better)
        de = metrics.get('debt_to_equity', 0)
        if de < 0.5:
            score += 0.1
        elif de > 2.0:
            score -= 0.1

        # Revenue Growth (positive is good)
        growth = metrics.get('revenue_growth', 0)
        if growth > 0.15:
            score += 0.1
        elif growth > 0.05:
            score += 0.05
        elif growth < 0:
            score -= 0.1

        # Ensure score is between 0 and 1
        return max(0, min(1, score))