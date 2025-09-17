"""
PROFESSIONAL Real-Time Market Data Service
100% Real Market Data - NO MOCK DATA
Uses yfinance and other professional APIs
"""

import yfinance as yf
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)

@dataclass
class RealStockData:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: float
    sector: str
    timestamp: datetime
    
    # Additional metrics
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    previous_close: float = 0.0

class ProfessionalRealDataService:
    """Professional-grade real market data service - NO MOCK DATA"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 60  # 1 minute cache for real-time
        self.last_fetch_time = {}
        self.min_fetch_interval = 1  # 1 second minimum between fetches per symbol
        
        # Sector mapping
        self.sector_mapping = {
            'AAPL': 'Technology',
            'MSFT': 'Technology', 
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'NVDA': 'Technology',
            'META': 'Technology',
            'TSLA': 'Consumer Discretionary',
            'JPM': 'Financials',
            'BAC': 'Financials',
            'JNJ': 'Healthcare',
            'PFE': 'Healthcare',
            'XOM': 'Energy',
            'CVX': 'Energy'
        }
        
        logger.info("🔥 Professional Real Data Service initialized - NO MOCK DATA")
    
    def _should_fetch(self, symbol: str) -> bool:
        """Check if we should fetch fresh data for symbol"""
        last_fetch = self.last_fetch_time.get(symbol, 0)
        return (time.time() - last_fetch) >= self.min_fetch_interval
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self.cache:
            return False
        
        cache_time = self.cache[symbol].get('timestamp')
        if not cache_time:
            return False
            
        age = (datetime.now() - cache_time).total_seconds()
        return age < self.cache_duration
    
    async def get_real_stock_data(self, symbol: str) -> Optional[RealStockData]:
        """Get real-time stock data using yfinance - NO MOCK DATA"""
        
        # Check cache first
        if self._is_cache_valid(symbol):
            cached_data = self.cache[symbol]['data']
            logger.info(f"📊 Using cached real data for {symbol}: ${cached_data.price:.2f}")
            return cached_data
        
        # Check rate limiting
        if not self._should_fetch(symbol):
            if symbol in self.cache:
                return self.cache[symbol]['data']
            
        try:
            logger.info(f"🔄 Fetching REAL market data for {symbol}")
            
            # Get real data from yfinance
            ticker = yf.Ticker(symbol)
            
            # Get current info
            info = ticker.info
            
            # Get recent price data (1 minute intervals for real-time)
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                logger.error(f"❌ No price data available for {symbol}")
                return None
            
            # Get latest data
            latest = hist.iloc[-1]
            current_price = float(latest['Close'])
            volume = int(latest['Volume'])
            open_price = float(latest['Open'])
            high_price = float(latest['High'])
            low_price = float(latest['Low'])
            
            # Calculate change from previous close
            if len(hist) > 1:
                previous_close = float(hist.iloc[-2]['Close'])
            else:
                previous_close = info.get('previousClose', current_price)
            
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
            
            # Get market cap from info
            market_cap = info.get('marketCap', 0)
            
            # Get sector
            sector = self.sector_mapping.get(symbol, info.get('sector', 'Unknown'))
            
            # Create real stock data
            stock_data = RealStockData(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=volume,
                market_cap=market_cap,
                sector=sector,
                timestamp=datetime.now(),
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                previous_close=previous_close
            )
            
            # Cache the data
            self.cache[symbol] = {
                'data': stock_data,
                'timestamp': datetime.now()
            }
            
            # Update fetch time
            self.last_fetch_time[symbol] = time.time()
            
            logger.info(f"✅ Real data for {symbol}: ${current_price:.2f} ({change_percent:+.2f}%)")
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching real data for {symbol}: {e}")
            return None
    
    async def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, RealStockData]:
        """Get real-time data for multiple stocks"""
        logger.info(f"🔄 Fetching real data for {len(symbols)} stocks: {symbols}")
        
        results = {}
        
        # Process stocks with rate limiting
        for symbol in symbols:
            stock_data = await self.get_real_stock_data(symbol)
            if stock_data:
                results[symbol] = stock_data
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        
        logger.info(f"✅ Successfully fetched real data for {len(results)}/{len(symbols)} stocks")
        return results
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get real market overview with actual stock prices"""
        
        # Major stocks to track
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"]
        
        # Get real data for all symbols
        stocks_data = await self.get_multiple_stocks(symbols)
        
        if not stocks_data:
            logger.error("❌ No real stock data available for market overview")
            return {}
        
        # Calculate real market metrics
        total_volume = sum(stock.volume for stock in stocks_data.values())
        average_change = sum(stock.change_percent for stock in stocks_data.values()) / len(stocks_data)
        total_market_cap = sum(stock.market_cap for stock in stocks_data.values() if stock.market_cap > 0)
        
        # Market direction based on real data
        positive_stocks = sum(1 for stock in stocks_data.values() if stock.change_percent > 0)
        market_direction = "UP" if positive_stocks > len(stocks_data) / 2 else "DOWN"
        
        # Market heat based on real volatility
        abs_changes = [abs(stock.change_percent) for stock in stocks_data.values()]
        market_heat_score = min(1.0, sum(abs_changes) / (len(abs_changes) * 5))  # Normalize to 0-1
        
        # Heat status
        if market_heat_score > 0.8:
            heat_status = "BLAZING_HOT"
        elif market_heat_score > 0.6:
            heat_status = "HOT"
        elif market_heat_score > 0.4:
            heat_status = "WARM"
        else:
            heat_status = "COOL"
        
        # Convert stocks to dict format
        stocks_dict = {}
        for symbol, stock_data in stocks_data.items():
            stocks_dict[symbol] = {
                "symbol": stock_data.symbol,
                "price": stock_data.price,
                "change": stock_data.change,
                "change_percent": stock_data.change_percent,
                "volume": stock_data.volume,
                "market_cap": stock_data.market_cap,
                "sector": stock_data.sector,
                "timestamp": stock_data.timestamp.isoformat(),
                "open": stock_data.open_price,
                "high": stock_data.high_price,
                "low": stock_data.low_price,
                "previous_close": stock_data.previous_close
            }
        
        return {
            "market_direction": market_direction,
            "market_heat_status": heat_status,
            "average_change": round(average_change, 2),
            "market_heat_score": round(market_heat_score, 3),
            "total_volume": total_volume,
            "total_market_cap": total_market_cap,
            "active_stocks": len(stocks_data),
            "timestamp": datetime.now().isoformat(),
            "stocks": stocks_dict,
            "data_source": "REAL_MARKET_DATA_YFINANCE"
        }

# Create global instance
professional_real_data_service = ProfessionalRealDataService()