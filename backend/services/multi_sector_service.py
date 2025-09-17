"""
Efficient multi-sector stock analysis service with parallel processing and caching
Handles 150+ stocks across 10 sectors without lag
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from config.sector_stocks import (
    SECTOR_STOCKS, FETCH_CONFIG, PRIORITY_STOCKS, 
    get_all_stocks, get_top_stocks_by_sector, get_sector_for_stock, get_sector_color
)

logger = logging.getLogger(__name__)

@dataclass
class StockData:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: float
    sector: str
    heat_score: float
    volatility: float
    timestamp: datetime

@dataclass 
class SectorSummary:
    sector: str
    sector_name: str
    color: str
    total_stocks: int
    top_performers: List[str]
    avg_change: float
    total_volume: int
    heat_index: float

class MultiSectorCache:
    """High-performance caching system for stock data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[dict]:
        async with self.lock:
            if key in self.cache:
                timestamp = self.cache_timestamps.get(key)
                if timestamp and (datetime.now() - timestamp).seconds < FETCH_CONFIG["cache_duration"]:
                    return self.cache[key]
            return None
    
    async def set(self, key: str, value: dict):
        async with self.lock:
            self.cache[key] = value
            self.cache_timestamps[key] = datetime.now()
    
    async def clear_expired(self):
        """Remove expired cache entries"""
        async with self.lock:
            current_time = datetime.now()
            expired_keys = [
                key for key, timestamp in self.cache_timestamps.items()
                if (current_time - timestamp).seconds >= FETCH_CONFIG["cache_duration"]
            ]
            for key in expired_keys:
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)

class MultiSectorService:
    """Efficient multi-sector stock analysis service"""
    
    def __init__(self):
        self.cache = MultiSectorCache()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.session = None
        self.last_update = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=FETCH_CONFIG["timeout"])
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _fetch_stock_data_sync(self, symbol: str) -> Optional[StockData]:
        """Synchronously fetch stock data for a single symbol"""
        try:
            # Increased delay to prevent rate limiting
            import time
            import random
            time.sleep(random.uniform(0.5, 1.0))  # 500ms-1s random delay between requests
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if hist.empty or not info:
                logger.warning(f"No data returned for {symbol}")
                return None
                
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            change = current_price - prev_price
            change_percent = (change / prev_price * 100) if prev_price != 0 else 0
            
            # Calculate volatility and heat score
            volatility = hist['Close'].pct_change().std() * 100
            volume_ratio = hist['Volume'].iloc[-1] / hist['Volume'].mean() if len(hist) > 1 else 1
            heat_score = min(max((abs(change_percent) * 0.3 + volatility * 0.4 + (volume_ratio - 1) * 0.3), 0), 1)
            
            return StockData(
                symbol=symbol,
                price=float(current_price),
                change=float(change),
                change_percent=float(change_percent),
                volume=int(hist['Volume'].iloc[-1]),
                market_cap=info.get('marketCap', 0),
                sector=get_sector_for_stock(symbol),
                heat_score=float(heat_score),
                volatility=float(volatility),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    async def fetch_stocks_batch(self, symbols: List[str]) -> List[StockData]:
        """Fetch stock data for a batch of symbols in parallel"""
        loop = asyncio.get_event_loop()
        tasks = []
        
        for symbol in symbols:
            # Check cache first
            cached_data = await self.cache.get(f"stock_{symbol}")
            if cached_data:
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=StockData(**cached_data))))
            else:
                task = loop.run_in_executor(self.executor, self._fetch_stock_data_sync, symbol)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cache successful results
        valid_results = []
        for result in results:
            if isinstance(result, StockData):
                valid_results.append(result)
                await self.cache.set(f"stock_{result.symbol}", {
                    'symbol': result.symbol,
                    'price': result.price,
                    'change': result.change,
                    'change_percent': result.change_percent,
                    'volume': result.volume,
                    'market_cap': result.market_cap,
                    'sector': result.sector,
                    'heat_score': result.heat_score,
                    'volatility': result.volatility,
                    'timestamp': result.timestamp.isoformat()
                })
        
        return valid_results

    async def get_all_sector_data(self) -> Dict[str, List[StockData]]:
        """Fetch data for all sectors efficiently"""
        all_stocks = get_all_stocks()
        
        # Process in smaller batches to avoid rate limiting but ensure we get real data
        batch_size = FETCH_CONFIG["batch_size"]
        all_data = {}
        
        # First, fetch priority stocks
        logger.info(f"Fetching {len(PRIORITY_STOCKS)} priority stocks")
        priority_data = await self.fetch_stocks_batch(PRIORITY_STOCKS)
        
        # Then fetch remaining stocks in batches (limit to top stocks per sector for now)
        top_stocks_per_sector = []
        for sector, data in SECTOR_STOCKS.items():
            top_stocks_per_sector.extend(data["top_stocks"][:4])  # Top 4 per sector
        
        # Remove duplicates and priority stocks
        remaining_stocks = list(set(top_stocks_per_sector) - set(PRIORITY_STOCKS))
        
        logger.info(f"Fetching {len(remaining_stocks)} remaining top sector stocks")
        for i in range(0, len(remaining_stocks), batch_size):
            batch = remaining_stocks[i:i + batch_size]
            batch_data = await self.fetch_stocks_batch(batch)
            priority_data.extend(batch_data)
        
        # Group by sector - only include if we have real data
        for stock_data in priority_data:
            if stock_data:  # Only include valid stock data
                sector = stock_data.sector
                if sector not in all_data:
                    all_data[sector] = []
                all_data[sector].append(stock_data)
        
        logger.info(f"Retrieved data for {len(priority_data)} stocks across {len(all_data)} sectors")
        return all_data

    async def get_sector_summaries(self) -> List[SectorSummary]:
        """Generate sector summaries with key metrics"""
        sector_data = await self.get_all_sector_data()
        summaries = []
        
        for sector, data in sector_data.items():
            if not data:
                continue
                
            # Calculate sector metrics
            changes = [stock.change_percent for stock in data]
            volumes = [stock.volume for stock in data]
            heat_scores = [stock.heat_score for stock in data]
            
            # Find top performers
            sorted_stocks = sorted(data, key=lambda x: x.change_percent, reverse=True)
            top_performers = [stock.symbol for stock in sorted_stocks[:3]]
            
            summary = SectorSummary(
                sector=sector,
                sector_name=SECTOR_STOCKS.get(sector, {}).get("sector_name", sector),
                color=get_sector_color(sector),
                total_stocks=len(data),
                top_performers=top_performers,
                avg_change=np.mean(changes) if changes else 0,
                total_volume=sum(volumes),
                heat_index=np.mean(heat_scores) if heat_scores else 0
            )
            summaries.append(summary)
        
        return sorted(summaries, key=lambda x: x.avg_change, reverse=True)

    async def get_top_stocks_by_sectors(self) -> Dict[str, List[StockData]]:
        """Get top 4 stocks from each sector"""
        all_sector_data = await self.get_all_sector_data()
        top_stocks = {}
        
        for sector, stocks in all_sector_data.items():
            # Sort by combined score (change + heat score)
            sorted_stocks = sorted(
                stocks, 
                key=lambda x: x.change_percent * 0.6 + x.heat_score * 40,
                reverse=True
            )
            top_stocks[sector] = sorted_stocks[:4]
        
        return top_stocks

    async def get_bubble_chart_data(self) -> List[Dict]:
        """Generate bubble chart data for visualization"""
        sector_data = await self.get_all_sector_data()
        bubble_data = []
        
        for sector, stocks in sector_data.items():
            sector_info = SECTOR_STOCKS.get(sector, {})
            
            for stock in stocks:
                bubble_data.append({
                    'symbol': stock.symbol,
                    'sector': sector,
                    'sector_name': sector_info.get('sector_name', sector),
                    'x': stock.change_percent,  # X-axis: Price change
                    'y': stock.volatility,      # Y-axis: Volatility  
                    'size': min(max(stock.heat_score * 100, 10), 100),  # Bubble size: Heat score
                    'color': sector_info.get('color', '#6b7280'),
                    'price': stock.price,
                    'volume': stock.volume,
                    'market_cap': stock.market_cap
                })
        
        return bubble_data

    async def get_tesla_analysis(self) -> Optional[StockData]:
        """Specific analysis for Tesla"""
        tesla_data = await self.fetch_stocks_batch(["TSLA"])
        return tesla_data[0] if tesla_data else None

# Global service instance
multi_sector_service = MultiSectorService()

async def get_efficient_sector_data():
    """Main function to get all sector data efficiently"""
    async with MultiSectorService() as service:
        return await service.get_all_sector_data()