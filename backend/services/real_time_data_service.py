"""
Real-time Data Service with fallback and mock data for development
Handles live market data with proper rate limiting and caching
"""

import asyncio
import logging
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import random
import time
from dataclasses import dataclass, asdict
import numpy as np

# Import professional data service
try:
    from professional_data_service import professional_data_service
    PROFESSIONAL_API_AVAILABLE = True
except ImportError:
    PROFESSIONAL_API_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class LiveStockData:
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
    rsi: float = 50.0
    bollinger_position: float = 0.5
    volume_surge: float = 1.0

class RealTimeDataService:
    """Service for fetching real-time market data with fallback options"""
    
    def __init__(self):
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds between API calls
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.use_mock_data = True  # Enable mock data for development
        
        # Mock market data for development
        self.mock_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
            "JPM", "BAC", "GS", "WFC", "MS", "C", "V", "MA",
            "JNJ", "PFE", "UNH", "ABBV", "MRK", "CVS", "LLY", "TMO",
            "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "KMI", "WMB"
        ]
        
        self.sectors = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "AMZN": "Consumer Discretionary", "NVDA": "Technology", "META": "Technology",
            "TSLA": "Consumer Discretionary", "NFLX": "Communication Services",
            "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
            "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
            "XOM": "Energy", "CVX": "Energy", "COP": "Energy"
        }

    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _generate_mock_data(self, symbol: str) -> LiveStockData:
        """Generate realistic mock market data for development"""
        # Base prices for different stocks
        base_prices = {
            "AAPL": 175, "MSFT": 330, "GOOGL": 130, "AMZN": 140, "NVDA": 450,
            "META": 300, "TSLA": 250, "NFLX": 400, "JPM": 150, "BAC": 30,
            "GS": 350, "JNJ": 160, "PFE": 30, "UNH": 500, "XOM": 110, "CVX": 150
        }
        
        base_price = base_prices.get(symbol, random.uniform(50, 200))
        
        # Simulate intraday movement
        change_percent = random.uniform(-3.0, 3.0)  # ±3% daily movement
        current_price = base_price * (1 + change_percent / 100)
        change = current_price - base_price
        
        # Generate realistic metrics
        volume = random.randint(1000000, 50000000)
        market_cap = current_price * random.uniform(100000000, 3000000000)
        volatility = random.uniform(15, 45)  # 15-45% annualized volatility
        
        # Heat score based on movement and volume
        volume_factor = min(volume / 10000000, 3.0)  # Volume surge factor
        heat_score = min(
            (abs(change_percent) * 0.4 + volatility * 0.3 + volume_factor * 0.3) / 10,
            1.0
        )
        
        return LiveStockData(
            symbol=symbol,
            price=round(current_price, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            volume=volume,
            market_cap=int(market_cap),
            sector=self.sectors.get(symbol, "Technology"),
            heat_score=round(heat_score, 3),
            volatility=round(volatility, 2),
            timestamp=datetime.now(),
            rsi=random.uniform(20, 80),
            bollinger_position=random.uniform(0, 1),
            volume_surge=random.uniform(0.5, 3.0)
        )

    async def get_stock_data(self, symbol: str, force_refresh: bool = False) -> Optional[LiveStockData]:
        """Get real-time stock data with caching and fallback"""
        
        # Check cache first
        cache_key = f"stock_{symbol}"
        if not force_refresh and cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_duration):
                return LiveStockData(**cached_data)
        
        # Use mock data in development mode or if API fails
        if self.use_mock_data:
            mock_data = self._generate_mock_data(symbol)
            self.cache[cache_key] = (asdict(mock_data), datetime.now())
            return mock_data
        
        # Try professional APIs first if available
        if PROFESSIONAL_API_AVAILABLE:
            try:
                professional_data = await professional_data_service.get_live_market_data(symbol)
                if professional_data:
                    # Convert professional data to LiveStockData format
                    stock_data = LiveStockData(
                        symbol=professional_data.symbol,
                        price=professional_data.current_price,
                        change=professional_data.change,
                        change_percent=professional_data.change_percent,
                        volume=professional_data.volume,
                        market_cap=professional_data.market_cap or 0,
                        sector=professional_data.sector,
                        heat_score=min(abs(professional_data.change_percent) / 10 + professional_data.volatility / 100, 1.0),
                        volatility=professional_data.volatility,
                        timestamp=professional_data.timestamp,
                        rsi=professional_data.rsi,
                        bollinger_position=professional_data.bb_position,
                        volume_surge=professional_data.volume_surge
                    )
                    
                    # Cache the result
                    self.cache[cache_key] = (asdict(stock_data), datetime.now())
                    logger.info(f"✅ Professional data for {symbol}: ${professional_data.current_price} ({professional_data.change_percent:+.2f}%)")
                    return stock_data
                    
            except Exception as e:
                logger.warning(f"Professional API failed for {symbol}: {e}, falling back to yfinance")
        
        # Fallback to yfinance API call with rate limiting
        try:
            self._rate_limit()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d", interval="1m")
            
            if hist.empty:
                logger.warning(f"No data for {symbol}, using mock data")
                return self._generate_mock_data(symbol)
            
            current_price = float(hist['Close'].iloc[-1])
            prev_close = float(info.get('regularMarketPreviousClose', current_price))
            
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
            
            # Calculate technical indicators
            closes = hist['Close'].values
            volatility = np.std(np.diff(closes) / closes[:-1]) * np.sqrt(252) * 100
            
            # RSI calculation (simplified)
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            rs = avg_gain / avg_loss if avg_loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))
            
            # Volume analysis
            volume = int(hist['Volume'].iloc[-1])
            avg_volume = int(hist['Volume'].mean())
            volume_surge = volume / avg_volume if avg_volume > 0 else 1
            
            # Heat score
            heat_score = min(
                (abs(change_percent) * 0.4 + volatility * 0.3 + (volume_surge - 1) * 0.3) / 10,
                1.0
            )
            
            stock_data = LiveStockData(
                symbol=symbol,
                price=round(current_price, 2),
                change=round(change, 2),
                change_percent=round(change_percent, 2),
                volume=volume,
                market_cap=info.get('marketCap', 0),
                sector=self.sectors.get(symbol, "Unknown"),
                heat_score=round(max(heat_score, 0), 3),
                volatility=round(volatility, 2),
                timestamp=datetime.now(),
                rsi=round(rsi, 2),
                bollinger_position=random.uniform(0, 1),  # Would need more data for real calculation
                volume_surge=round(volume_surge, 2)
            )
            
            # Cache the result
            self.cache[cache_key] = (asdict(stock_data), datetime.now())
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching real data for {symbol}: {e}")
            # Fallback to mock data
            mock_data = self._generate_mock_data(symbol)
            self.cache[cache_key] = (asdict(mock_data), datetime.now())
            return mock_data

    async def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, LiveStockData]:
        """Get data for multiple stocks with concurrent processing"""
        
        async def fetch_with_semaphore(symbol: str, semaphore: asyncio.Semaphore):
            async with semaphore:
                return await self.get_stock_data(symbol)
        
        # Limit concurrent requests to prevent rate limiting
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        tasks = [fetch_with_semaphore(symbol, semaphore) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        stock_data = {}
        for i, result in enumerate(results):
            if isinstance(result, LiveStockData):
                stock_data[symbols[i]] = result
            else:
                logger.error(f"Failed to fetch data for {symbols[i]}: {result}")
                # Generate mock data as fallback
                stock_data[symbols[i]] = self._generate_mock_data(symbols[i])
        
        return stock_data

    async def get_market_overview(self) -> Dict[str, Any]:
        """Get overall market overview with key metrics"""
        
        # Get data for major indices/stocks
        major_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]
        stock_data = await self.get_multiple_stocks(major_symbols)
        
        if not stock_data:
            return {"error": "No market data available"}
        
        # Calculate market metrics
        changes = [data.change_percent for data in stock_data.values()]
        heat_scores = [data.heat_score for data in stock_data.values()]
        volumes = [data.volume for data in stock_data.values()]
        
        market_direction = "UP" if np.mean(changes) > 0 else "DOWN"
        market_heat = np.mean(heat_scores)
        total_volume = sum(volumes)
        
        # Determine market status
        if market_heat > 0.8:
            status = "BLAZING_HOT"
        elif market_heat > 0.6:
            status = "HOT"
        elif market_heat > 0.4:
            status = "WARM"
        else:
            status = "COOL"
        
        return {
            "market_direction": market_direction,
            "market_heat_status": status,
            "average_change": round(np.mean(changes), 2),
            "market_heat_score": round(market_heat, 3),
            "total_volume": total_volume,
            "active_stocks": len(stock_data),
            "timestamp": datetime.now().isoformat(),
            "stocks": {symbol: asdict(data) for symbol, data in stock_data.items()}
        }

    def enable_live_data(self):
        """Enable live data fetching (disable mock mode)"""
        self.use_mock_data = False
        logger.info("Live data mode enabled")

    def enable_mock_data(self):
        """Enable mock data mode for development"""
        self.use_mock_data = True
        logger.info("Mock data mode enabled")

# Global service instance
real_time_service = RealTimeDataService()