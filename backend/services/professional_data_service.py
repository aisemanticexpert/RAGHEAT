"""
Professional Live Market Data Service
Uses Finnhub and Alpha Vantage APIs for real-time market data
"""

import asyncio
import aiohttp
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class LiveMarketData:
    symbol: str
    current_price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open_price: float
    prev_close: float
    market_cap: Optional[int]
    rsi: float
    bb_position: float
    volume_surge: float
    volatility: float
    sector: str
    data_source: str
    timestamp: datetime

class ProfessionalDataService:
    """Professional live market data using Finnhub and Alpha Vantage APIs"""
    
    def __init__(self):
        # API Keys
        self.finnhub_api_key = "d31c90hr01qsprr0a8agd31c90hr01qsprr0a8b0"
        self.alpha_vantage_api_key = "ES7TJH918K5PPXQ8"
        
        # API endpoints
        self.finnhub_base = "https://finnhub.io/api/v1"
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
        
        # Caching
        self.cache = {}
        self.cache_duration = 60  # 1 minute cache for real-time data
        self.last_request_times = {}
        self.request_interval = 1.0  # 1 second between requests
        
        # Symbol to sector mapping
        self.symbol_sectors = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "NVDA": "Technology", "META": "Technology", "AMZN": "Consumer Discretionary",
            "TSLA": "Consumer Discretionary", "NFLX": "Communication Services",
            "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
            "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
            "XOM": "Energy", "CVX": "Energy", "COP": "Energy"
        }

    def rate_limit(self, api_name: str):
        """Rate limiting per API"""
        current_time = time.time()
        key = f"{api_name}_last_request"
        
        if key in self.last_request_times:
            time_since = current_time - self.last_request_times[key]
            if time_since < self.request_interval:
                sleep_time = self.request_interval - time_since
                time.sleep(sleep_time)
        
        self.last_request_times[key] = time.time()

    async def get_finnhub_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote from Finnhub"""
        try:
            self.rate_limit("finnhub")
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.finnhub_base}/quote"
                params = {
                    'symbol': symbol,
                    'token': self.finnhub_api_key
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check if we got valid data
                        if 'c' in data and data['c'] is not None and data['c'] > 0:
                            return {
                                'current_price': float(data['c']),  # Current price
                                'change': float(data['d']),         # Change
                                'change_percent': float(data['dp']),# Change percent
                                'high': float(data['h']),           # High
                                'low': float(data['l']),            # Low
                                'open': float(data['o']),           # Open
                                'prev_close': float(data['pc']),    # Previous close
                                'timestamp': datetime.now(),
                                'data_source': 'finnhub_realtime'
                            }
                        else:
                            logger.warning(f"Invalid Finnhub data for {symbol}: {data}")
                            return None
                    else:
                        logger.error(f"Finnhub API error for {symbol}: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching Finnhub data for {symbol}: {e}")
            return None

    async def get_alpha_vantage_intraday(self, symbol: str) -> Optional[Dict]:
        """Get intraday data from Alpha Vantage for technical analysis"""
        try:
            self.rate_limit("alpha_vantage")
            
            async with aiohttp.ClientSession() as session:
                url = self.alpha_vantage_base
                params = {
                    'function': 'TIME_SERIES_INTRADAY',
                    'symbol': symbol,
                    'interval': '5min',
                    'apikey': self.alpha_vantage_api_key,
                    'outputsize': 'compact'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API limit
                        if 'Note' in data:
                            logger.warning(f"Alpha Vantage rate limit hit for {symbol}")
                            return None
                        
                        # Check for valid time series data
                        time_series_key = 'Time Series (5min)'
                        if time_series_key not in data:
                            logger.warning(f"No time series data for {symbol}")
                            return None
                        
                        time_series = data[time_series_key]
                        if not time_series:
                            return None
                        
                        # Get recent data points for analysis
                        timestamps = sorted(time_series.keys(), reverse=True)
                        recent_data = []
                        
                        for ts in timestamps[:20]:  # Last 20 data points (100 minutes)
                            point = time_series[ts]
                            recent_data.append({
                                'timestamp': ts,
                                'open': float(point['1. open']),
                                'high': float(point['2. high']),
                                'low': float(point['3. low']),
                                'close': float(point['4. close']),
                                'volume': int(point['5. volume'])
                            })
                        
                        if len(recent_data) < 10:
                            logger.warning(f"Insufficient data points for {symbol}")
                            return None
                        
                        # Calculate technical indicators
                        closes = [d['close'] for d in recent_data]
                        volumes = [d['volume'] for d in recent_data]
                        
                        # RSI calculation (14 periods)
                        rsi = self.calculate_rsi(closes)
                        
                        # Bollinger Bands
                        bb_position = self.calculate_bollinger_position(closes)
                        
                        # Volume analysis
                        avg_volume = sum(volumes) / len(volumes)
                        current_volume = volumes[0]  # Most recent
                        volume_surge = current_volume / avg_volume if avg_volume > 0 else 1.0
                        
                        # Volatility (using standard deviation of returns)
                        returns = [(closes[i] / closes[i+1] - 1) for i in range(len(closes)-1)]
                        volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 25.0
                        
                        return {
                            'rsi': rsi,
                            'bb_position': bb_position,
                            'volume_surge': volume_surge,
                            'volatility': volatility,
                            'avg_volume': int(avg_volume),
                            'recent_closes': closes[:5],  # For additional analysis
                            'data_source': 'alpha_vantage_intraday'
                        }
                        
                    else:
                        logger.error(f"Alpha Vantage API error for {symbol}: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return None

    def calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI from price data"""
        if len(closes) < period + 1:
            return 50.0  # Neutral RSI if insufficient data
        
        deltas = [closes[i] - closes[i+1] for i in range(len(closes)-1)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return max(0, min(100, rsi))

    def calculate_bollinger_position(self, closes: List[float], period: int = 20) -> float:
        """Calculate position within Bollinger Bands"""
        if len(closes) < period:
            return 0.5  # Middle position if insufficient data
        
        recent_closes = closes[:period]
        sma = sum(recent_closes) / len(recent_closes)
        std = np.std(recent_closes)
        
        if std == 0:
            return 0.5
        
        current_price = closes[0]
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        # Position within bands (0 = lower band, 1 = upper band)
        position = (current_price - lower_band) / (upper_band - lower_band)
        return max(0, min(1, position))

    async def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """Get company profile from Finnhub for sector and market cap"""
        try:
            # Use cached sector info first
            if symbol in self.symbol_sectors:
                return {
                    'sector': self.symbol_sectors[symbol],
                    'marketCapitalization': None  # Will be filled by quote data
                }
            
            self.rate_limit("finnhub_profile")
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.finnhub_base}/stock/profile2"
                params = {
                    'symbol': symbol,
                    'token': self.finnhub_api_key
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and 'gics' in data:
                            return {
                                'sector': data.get('gics', 'Unknown'),
                                'marketCapitalization': data.get('marketCapitalization'),
                                'name': data.get('name')
                            }
                    
                    # Fallback to known sector
                    return {
                        'sector': self.symbol_sectors.get(symbol, 'Unknown'),
                        'marketCapitalization': None
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {e}")
            return {
                'sector': self.symbol_sectors.get(symbol, 'Unknown'),
                'marketCapitalization': None
            }

    async def get_live_market_data(self, symbol: str) -> Optional[LiveMarketData]:
        """Get comprehensive live market data combining multiple APIs"""
        
        # Check cache first
        cache_key = f"live_{symbol}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_duration):
                return cached_data
        
        try:
            # Get real-time quote from Finnhub
            quote_data = await self.get_finnhub_quote(symbol)
            if not quote_data:
                logger.warning(f"No quote data available for {symbol}")
                return None
            
            # Get technical analysis from Alpha Vantage
            technical_data = await self.get_alpha_vantage_intraday(symbol)
            
            # Get company profile
            profile_data = await self.get_company_profile(symbol)
            
            # Combine all data
            current_price = quote_data['current_price']
            change = quote_data['change']
            change_percent = quote_data['change_percent']
            
            # Use technical data if available, otherwise calculate from quote
            if technical_data:
                rsi = technical_data['rsi']
                bb_position = technical_data['bb_position']
                volume_surge = technical_data['volume_surge']
                volatility = technical_data['volatility']
                volume = technical_data['avg_volume']
            else:
                # Fallback calculations
                rsi = 50.0 + (change_percent * 2)  # Approximate
                bb_position = 0.5 + (change_percent / 10)  # Approximate
                volume_surge = 1.0 + abs(change_percent) / 5  # Higher volume on big moves
                volatility = max(15.0, min(50.0, 20.0 + abs(change_percent) * 2))
                volume = 1000000  # Default volume
            
            # Ensure values are within reasonable ranges
            rsi = max(0, min(100, rsi))
            bb_position = max(0, min(1, bb_position))
            volume_surge = max(0.5, min(5.0, volume_surge))
            
            sector = profile_data['sector'] if profile_data else self.symbol_sectors.get(symbol, 'Unknown')
            market_cap = profile_data.get('marketCapitalization') if profile_data else None
            
            live_data = LiveMarketData(
                symbol=symbol,
                current_price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(volume),
                high=quote_data['high'],
                low=quote_data['low'],
                open_price=quote_data['open'],
                prev_close=quote_data['prev_close'],
                market_cap=market_cap,
                rsi=round(rsi, 1),
                bb_position=round(bb_position, 3),
                volume_surge=round(volume_surge, 2),
                volatility=round(volatility, 1),
                sector=sector,
                data_source='professional_apis',
                timestamp=datetime.now()
            )
            
            # Cache the result
            self.cache[cache_key] = (live_data, datetime.now())
            
            logger.info(f"✅ Live data for {symbol}: ${current_price:.2f} ({change_percent:+.2f}%)")
            
            return live_data
            
        except Exception as e:
            logger.error(f"Error getting live market data for {symbol}: {e}")
            return None

    async def get_multiple_symbols(self, symbols: List[str]) -> Dict[str, LiveMarketData]:
        """Get live data for multiple symbols concurrently"""
        
        # Limit concurrency to respect API rate limits
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
        
        async def fetch_with_semaphore(symbol: str):
            async with semaphore:
                return await self.get_live_market_data(symbol)
        
        tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        live_data = {}
        for i, result in enumerate(results):
            if isinstance(result, LiveMarketData):
                live_data[symbols[i]] = result
            else:
                logger.error(f"Failed to get data for {symbols[i]}: {result}")
        
        logger.info(f"📊 Retrieved live data for {len(live_data)}/{len(symbols)} symbols")
        
        return live_data

    def get_api_status(self) -> Dict[str, Any]:
        """Get status of professional APIs"""
        return {
            "finnhub_key_active": bool(self.finnhub_api_key),
            "alpha_vantage_key_active": bool(self.alpha_vantage_api_key),
            "cache_size": len(self.cache),
            "cache_duration_seconds": self.cache_duration,
            "supported_symbols": len(self.symbol_sectors),
            "data_sources": ["finnhub_realtime", "alpha_vantage_intraday"],
            "last_update": datetime.now().isoformat()
        }

# Global professional data service instance
professional_data_service = ProfessionalDataService()