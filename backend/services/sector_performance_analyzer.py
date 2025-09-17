"""
Advanced Sector Performance Analyzer
Selects top 5 best/worst performing sectors for options trading analysis
"""

import asyncio
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import requests

from config.sector_stocks import SECTOR_STOCKS, PRIORITY_STOCKS

logger = logging.getLogger(__name__)

@dataclass
class SectorPerformance:
    sector: str
    sector_name: str
    performance_score: float
    avg_return: float
    volatility: float
    momentum: float
    volume_surge: float
    news_sentiment: float
    top_stocks: List[str]
    timestamp: datetime

@dataclass
class OptionsCandidate:
    symbol: str
    sector: str
    win_probability: float
    expected_move: float
    volatility_rank: float
    volume_ratio: float
    news_impact: float
    earnings_proximity: int  # Days to earnings
    option_flow_score: float
    recommendation: str  # "call", "put", "straddle"
    confidence: float

class SectorPerformanceAnalyzer:
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=8)
        
    async def analyze_sector_performance(self) -> List[SectorPerformance]:
        """Analyze all sectors and rank by performance"""
        logger.info("Starting comprehensive sector analysis...")
        
        sector_analyses = []
        
        # Analyze each sector in parallel
        tasks = []
        for sector, data in SECTOR_STOCKS.items():
            task = asyncio.create_task(
                self._analyze_single_sector(sector, data)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results and sort by performance
        for result in results:
            if isinstance(result, SectorPerformance):
                sector_analyses.append(result)
        
        # Sort by performance score (descending)
        sector_analyses.sort(key=lambda x: x.performance_score, reverse=True)
        
        logger.info(f"Analyzed {len(sector_analyses)} sectors")
        return sector_analyses
    
    async def _analyze_single_sector(self, sector: str, sector_data: Dict) -> Optional[SectorPerformance]:
        """Deep analysis of a single sector"""
        try:
            stocks = sector_data["all_stocks"][:8]  # Top 8 stocks per sector
            
            # Fetch data for all stocks in parallel
            loop = asyncio.get_event_loop()
            stock_data_tasks = [
                loop.run_in_executor(self.executor, self._get_stock_metrics, symbol)
                for symbol in stocks
            ]
            
            stock_metrics = await asyncio.gather(*stock_data_tasks, return_exceptions=True)
            
            # Filter valid metrics
            valid_metrics = [m for m in stock_metrics if isinstance(m, dict) and m is not None]
            
            if len(valid_metrics) < 3:  # Need at least 3 stocks for sector analysis
                logger.warning(f"Insufficient data for sector {sector}")
                return None
            
            # Calculate sector-wide metrics
            returns = [m['return_1d'] for m in valid_metrics]
            volatilities = [m['volatility'] for m in valid_metrics]
            volumes = [m['volume_ratio'] for m in valid_metrics]
            
            avg_return = np.mean(returns)
            avg_volatility = np.mean(volatilities)
            momentum = self._calculate_momentum(valid_metrics)
            volume_surge = np.mean(volumes)
            
            # Get news sentiment for sector
            news_sentiment = await self._get_sector_news_sentiment(sector)
            
            # Calculate comprehensive performance score
            performance_score = self._calculate_performance_score(
                avg_return, avg_volatility, momentum, volume_surge, news_sentiment
            )
            
            # Select top performing stocks in sector
            top_stocks = sorted(
                valid_metrics, 
                key=lambda x: x['return_1d'] + x['volume_ratio'] * 0.1,
                reverse=True
            )[:4]
            
            return SectorPerformance(
                sector=sector,
                sector_name=sector_data["sector_name"],
                performance_score=performance_score,
                avg_return=avg_return,
                volatility=avg_volatility,
                momentum=momentum,
                volume_surge=volume_surge,
                news_sentiment=news_sentiment,
                top_stocks=[s['symbol'] for s in top_stocks],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sector {sector}: {e}")
            return None
    
    def _get_stock_metrics(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive metrics for a single stock"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist = ticker.history(period="30d")
            if hist.empty:
                return None
                
            # Calculate metrics
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            return_1d = (current_price - prev_price) / prev_price * 100
            
            # Volatility (20-day)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Volume analysis
            avg_volume = hist['Volume'].tail(20).mean()
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price momentum indicators
            sma_5 = hist['Close'].tail(5).mean()
            sma_20 = hist['Close'].tail(20).mean()
            momentum = (sma_5 - sma_20) / sma_20 * 100 if sma_20 > 0 else 0
            
            return {
                'symbol': symbol,
                'return_1d': return_1d,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'momentum': momentum,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Error fetching metrics for {symbol}: {e}")
            return None
    
    def _calculate_momentum(self, stock_metrics: List[Dict]) -> float:
        """Calculate sector momentum score"""
        momentum_scores = [m['momentum'] for m in stock_metrics]
        return np.mean(momentum_scores)
    
    def _calculate_performance_score(self, avg_return: float, volatility: float, 
                                   momentum: float, volume_surge: float, 
                                   news_sentiment: float) -> float:
        """Calculate comprehensive sector performance score"""
        
        # Weights for different factors
        return_weight = 0.35
        momentum_weight = 0.25
        volume_weight = 0.20
        news_weight = 0.15
        volatility_weight = 0.05  # Lower is better for volatility
        
        # Normalize scores
        return_score = max(-10, min(10, avg_return))  # Cap at ±10%
        momentum_score = max(-5, min(5, momentum))    # Cap at ±5%
        volume_score = min(3, max(0.5, volume_surge))  # Volume ratio 0.5-3
        news_score = max(-1, min(1, news_sentiment))   # Sentiment -1 to 1
        vol_score = max(0, min(100, volatility))       # Volatility 0-100%
        
        # Calculate weighted score
        performance_score = (
            return_score * return_weight +
            momentum_score * momentum_weight +
            (volume_score - 1) * volume_weight * 5 +  # Bonus for high volume
            news_score * news_weight * 10 +
            (50 - vol_score) / 50 * volatility_weight * 2  # Penalty for high vol
        )
        
        return performance_score
    
    async def _get_sector_news_sentiment(self, sector: str) -> float:
        """Get news sentiment for sector (placeholder - integrate with Alpha Vantage)"""
        # TODO: Integrate with Alpha Vantage or Yahoo Finance news API
        # For now, return random sentiment
        import random
        return random.uniform(-0.3, 0.7)  # Slight positive bias
    
    async def get_top_sectors(self, count: int = 5) -> Tuple[List[SectorPerformance], List[SectorPerformance]]:
        """Get top performing and worst performing sectors"""
        all_sectors = await self.analyze_sector_performance()
        
        # Top performers (best 5)
        top_sectors = all_sectors[:count]
        
        # Worst performers (bottom 5, but could be negative performers)
        worst_sectors = all_sectors[-count:]
        
        # Identify extremely bad performers (negative performance score)
        extreme_bad = [s for s in all_sectors if s.performance_score < -2.0]
        if len(extreme_bad) > 0:
            worst_sectors = extreme_bad[:count]  # Prioritize extreme negatives
        
        logger.info(f"Selected {len(top_sectors)} top sectors and {len(worst_sectors)} worst sectors")
        return top_sectors, worst_sectors

# Global instance
sector_analyzer = SectorPerformanceAnalyzer()