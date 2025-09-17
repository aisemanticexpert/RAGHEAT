"""
MULTI-SOURCE PROFESSIONAL REAL-TIME DATA SERVICE
Fallback system: Finnhub -> Alpha Vantage -> yfinance
100% Real Market Data - NO MOCK DATA
"""

import os
import asyncio
import aiohttp
import requests
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import time
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class ProfessionalStockData:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: float
    sector: str
    timestamp: datetime
    data_source: str
    
    # Additional professional metrics
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    previous_close: float = 0.0

class MultiSourceProfessionalDataService:
    """Professional-grade multi-source real market data service with fallback"""
    
    def __init__(self):
        # API Keys from environment
        self.finnhub_key = os.getenv('FINNHUB_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        
        # API Endpoints
        self.finnhub_base = "https://finnhub.io/api/v1"
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
        
        # Caching and rate limiting
        self.cache = {}
        self.cache_duration = 30  # 30 seconds cache for real-time data
        self.last_fetch_time = {}
        
        # Rate limiting per API
        self.finnhub_min_interval = 1.0  # 1 second (60 calls/minute)
        self.alpha_vantage_min_interval = 12.0  # 12 seconds (5 calls/minute)
        self.yfinance_min_interval = 2.0  # 2 seconds for fallback
        
        # Sector mapping
        self.sector_mapping = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary', 'NVDA': 'Technology', 'META': 'Technology',
            'TSLA': 'Consumer Discretionary', 'JPM': 'Financials', 'BAC': 'Financials',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'XOM': 'Energy', 'CVX': 'Energy'
        }
        
        logger.info("🔥 Multi-Source Professional Data Service initialized")
        logger.info(f"✅ Finnhub API: {'ACTIVE' if self.finnhub_key else 'NO KEY'}")
        logger.info(f"✅ Alpha Vantage API: {'ACTIVE' if self.alpha_vantage_key else 'NO KEY'}")
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self.cache:
            return False
        
        cache_time = self.cache[symbol].get('timestamp')
        if not cache_time:
            return False
            
        age = (datetime.now() - cache_time).total_seconds()
        return age < self.cache_duration
    
    def _can_fetch_from_source(self, source: str, symbol: str) -> bool:
        """Check if we can fetch from specific source based on rate limiting"""
        key = f"{source}_{symbol}"
        last_fetch = self.last_fetch_time.get(key, 0)
        current_time = time.time()
        
        if source == 'finnhub':
            return (current_time - last_fetch) >= self.finnhub_min_interval
        elif source == 'alpha_vantage':
            return (current_time - last_fetch) >= self.alpha_vantage_min_interval
        elif source == 'yfinance':
            return (current_time - last_fetch) >= self.yfinance_min_interval
        
        return True
    
    def _update_fetch_time(self, source: str, symbol: str):
        """Update last fetch time for rate limiting"""
        key = f"{source}_{symbol}"
        self.last_fetch_time[key] = time.time()
    
    async def _fetch_from_finnhub(self, symbol: str) -> Optional[ProfessionalStockData]:
        """Fetch real-time data from Finnhub API"""
        if not self.finnhub_key or not self._can_fetch_from_source('finnhub', symbol):
            return None
        
        try:
            logger.info(f"🔄 Fetching {symbol} from Finnhub API")
            
            # Get real-time quote
            quote_url = f"{self.finnhub_base}/quote"
            params = {"symbol": symbol, "token": self.finnhub_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(quote_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"❌ Finnhub API error {response.status} for {symbol}")
                        return None
                    
                    quote_data = await response.json()
            
            # Get company profile for market cap and sector
            profile_url = f"{self.finnhub_base}/stock/profile2"
            async with aiohttp.ClientSession() as session:
                async with session.get(profile_url, params=params) as response:
                    profile_data = await response.json() if response.status == 200 else {}
            
            current_price = quote_data.get('c', 0)
            previous_close = quote_data.get('pc', 0)
            
            if current_price == 0:
                logger.error(f"❌ No price data from Finnhub for {symbol}")
                return None
            
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close > 0 else 0
            
            stock_data = ProfessionalStockData(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(quote_data.get('volume', 0)),
                market_cap=profile_data.get('marketCapitalization', 0) * 1000000,  # Convert to actual value
                sector=self.sector_mapping.get(symbol, profile_data.get('finnhubIndustry', 'Unknown')),
                timestamp=datetime.now(),
                data_source='FINNHUB_PROFESSIONAL',
                open_price=quote_data.get('o', 0),
                high_price=quote_data.get('h', 0),
                low_price=quote_data.get('l', 0),
                previous_close=previous_close
            )
            
            self._update_fetch_time('finnhub', symbol)
            logger.info(f"✅ Finnhub: {symbol} = ${current_price:.2f} ({change_percent:+.2f}%)")
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ Finnhub error for {symbol}: {e}")
            return None
    
    async def _fetch_from_alpha_vantage(self, symbol: str) -> Optional[ProfessionalStockData]:
        """Fetch real-time data from Alpha Vantage API"""
        if not self.alpha_vantage_key or not self._can_fetch_from_source('alpha_vantage', symbol):
            return None
        
        try:
            logger.info(f"🔄 Fetching {symbol} from Alpha Vantage API")
            
            # Get global quote (real-time data)
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.alpha_vantage_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.alpha_vantage_base, params=params) as response:
                    if response.status != 200:
                        logger.error(f"❌ Alpha Vantage API error {response.status} for {symbol}")
                        return None
                    
                    data = await response.json()
            
            quote = data.get("Global Quote", {})
            
            if not quote:
                logger.error(f"❌ No quote data from Alpha Vantage for {symbol}")
                return None
            
            current_price = float(quote.get("05. price", 0))
            previous_close = float(quote.get("08. previous close", 0))
            
            if current_price == 0:
                logger.error(f"❌ No price data from Alpha Vantage for {symbol}")
                return None
            
            change = float(quote.get("09. change", 0))
            change_percent = float(quote.get("10. change percent", "0%").replace("%", ""))
            
            stock_data = ProfessionalStockData(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(quote.get("06. volume", 0)),
                market_cap=0,  # Alpha Vantage doesn't provide market cap in global quote
                sector=self.sector_mapping.get(symbol, 'Unknown'),
                timestamp=datetime.now(),
                data_source='ALPHA_VANTAGE_PROFESSIONAL',
                open_price=float(quote.get("02. open", 0)),
                high_price=float(quote.get("03. high", 0)),
                low_price=float(quote.get("04. low", 0)),
                previous_close=previous_close
            )
            
            self._update_fetch_time('alpha_vantage', symbol)
            logger.info(f"✅ Alpha Vantage: {symbol} = ${current_price:.2f} ({change_percent:+.2f}%)")
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ Alpha Vantage error for {symbol}: {e}")
            return None
    
    async def _fetch_from_yfinance(self, symbol: str) -> Optional[ProfessionalStockData]:
        """Fetch data from yfinance as fallback"""
        if not self._can_fetch_from_source('yfinance', symbol):
            return None
        
        try:
            logger.info(f"🔄 Fetching {symbol} from yfinance (fallback)")
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d", interval="1m")
            
            if hist.empty:
                logger.error(f"❌ No yfinance data for {symbol}")
                return None
            
            latest = hist.iloc[-1]
            previous = hist.iloc[-2] if len(hist) > 1 else latest
            
            current_price = float(latest['Close'])
            previous_close = float(previous['Close'])
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close > 0 else 0
            
            # Get additional info
            info = ticker.info
            
            stock_data = ProfessionalStockData(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(latest['Volume']),
                market_cap=info.get('marketCap', 0),
                sector=self.sector_mapping.get(symbol, info.get('sector', 'Unknown')),
                timestamp=datetime.now(),
                data_source='YFINANCE_FALLBACK',
                open_price=float(latest['Open']),
                high_price=float(latest['High']),
                low_price=float(latest['Low']),
                previous_close=previous_close
            )
            
            self._update_fetch_time('yfinance', symbol)
            logger.info(f"✅ yfinance: {symbol} = ${current_price:.2f} ({change_percent:+.2f}%)")
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ yfinance error for {symbol}: {e}")
            return None
    
    async def get_real_stock_data(self, symbol: str) -> Optional[ProfessionalStockData]:
        """Get real stock data with fallback mechanism"""
        
        # Check cache first
        if self._is_cache_valid(symbol):
            cached_data = self.cache[symbol]['data']
            logger.info(f"📊 Cache hit for {symbol}: ${cached_data.price:.2f} from {cached_data.data_source}")
            return cached_data
        
        # Try data sources in order: Finnhub -> Alpha Vantage -> yfinance
        for fetch_func in [self._fetch_from_finnhub, self._fetch_from_alpha_vantage, self._fetch_from_yfinance]:
            try:
                stock_data = await fetch_func(symbol)
                if stock_data:
                    # Cache the successful result
                    self.cache[symbol] = {
                        'data': stock_data,
                        'timestamp': datetime.now()
                    }
                    return stock_data
            except Exception as e:
                logger.error(f"❌ Error in {fetch_func.__name__} for {symbol}: {e}")
                continue
        
        logger.error(f"❌ ALL DATA SOURCES FAILED for {symbol}")
        return None
    
    async def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, ProfessionalStockData]:
        """Get real-time data for multiple stocks with intelligent batching"""
        logger.info(f"🔄 Fetching professional data for {len(symbols)} stocks")
        
        results = {}
        
        # Process in small batches to avoid overwhelming APIs
        batch_size = 3
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.get_real_stock_data(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, ProfessionalStockData):
                    results[symbol] = result
                else:
                    logger.error(f"❌ Failed to get data for {symbol}")
            
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(1.0)
        
        logger.info(f"✅ Successfully fetched {len(results)}/{len(symbols)} stocks from professional APIs")
        return results
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get comprehensive market overview with real professional data"""
        
        # Major market stocks
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"]
        
        stocks_data = await self.get_multiple_stocks(symbols)
        
        if not stocks_data:
            logger.error("❌ No professional stock data available")
            return {
                "error": "No professional market data available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate real market metrics
        total_volume = sum(stock.volume for stock in stocks_data.values())
        average_change = sum(stock.change_percent for stock in stocks_data.values()) / len(stocks_data)
        total_market_cap = sum(stock.market_cap for stock in stocks_data.values() if stock.market_cap > 0)
        
        # Market direction
        positive_stocks = sum(1 for stock in stocks_data.values() if stock.change_percent > 0)
        market_direction = "UP" if positive_stocks > len(stocks_data) / 2 else "DOWN"
        
        # Market heat based on volatility
        abs_changes = [abs(stock.change_percent) for stock in stocks_data.values()]
        market_heat_score = min(1.0, sum(abs_changes) / (len(abs_changes) * 3))
        
        heat_status = "BLAZING_HOT" if market_heat_score > 0.8 else \
                     "HOT" if market_heat_score > 0.6 else \
                     "WARM" if market_heat_score > 0.4 else "COOL"
        
        # Data sources used
        data_sources = list(set(stock.data_source for stock in stocks_data.values()))
        
        # Convert to API format
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
                "data_source": stock_data.data_source,
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
            "data_sources_used": data_sources,
            "service": "MULTI_SOURCE_PROFESSIONAL_API"
        }

# Global instance
multi_source_professional_data_service = MultiSourceProfessionalDataService()