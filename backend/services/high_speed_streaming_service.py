"""
High-Speed Live Data Streaming Service
Uses multiple free APIs with WebSocket streaming for real-time data
"""

import asyncio
import aiohttp
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, asdict
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor
import websockets

logger = logging.getLogger(__name__)

@dataclass
class StreamingStockData:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    source: str
    high: float = 0.0
    low: float = 0.0
    open_price: float = 0.0
    market_cap: int = 0

class HighSpeedStreamingService:
    """High-speed streaming service using multiple free APIs"""
    
    def __init__(self):
        self.active_connections = set()
        self.streaming_task = None
        self.is_streaming = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # API endpoints
        self.apis = {
            'yahoo': self._fetch_yahoo_data,
            'iex': self._fetch_iex_data,
            'polygon': self._fetch_polygon_data,
            'financialmodelingprep': self._fetch_fmp_data
        }
        
        self.symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", 
            "JPM", "BAC", "JNJ", "PFE", "XOM", "CVX"
        ]

    async def _fetch_yahoo_data(self, symbol: str) -> Optional[StreamingStockData]:
        """Fast Yahoo Finance data using yfinance"""
        try:
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            
            # Run yfinance in thread pool to avoid blocking
            info = await loop.run_in_executor(self.executor, lambda: ticker.info)
            hist = await loop.run_in_executor(
                self.executor, 
                lambda: ticker.history(period="1d", interval="1m", timeout=5)
            )
            
            if hist.empty or len(hist) == 0:
                return None
                
            latest = hist.iloc[-1]
            current_price = float(latest['Close'])
            open_price = float(latest['Open'])
            high_price = float(latest['High'])
            low_price = float(latest['Low'])
            volume = int(latest['Volume'])
            
            # Calculate change
            prev_close = float(info.get('regularMarketPreviousClose', current_price))
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
            
            return StreamingStockData(
                symbol=symbol,
                price=round(current_price, 2),
                change=round(change, 2),
                change_percent=round(change_percent, 2),
                volume=volume,
                high=round(high_price, 2),
                low=round(low_price, 2),
                open_price=round(open_price, 2),
                market_cap=info.get('marketCap', 0),
                timestamp=datetime.now(),
                source='yahoo'
            )
            
        except Exception as e:
            logger.warning(f"Yahoo API failed for {symbol}: {e}")
            return None

    async def _fetch_iex_data(self, symbol: str) -> Optional[StreamingStockData]:
        """IEX Cloud free tier data"""
        try:
            # Using IEX Cloud sandbox (free tier)
            url = f"https://sandbox.iexapis.com/stable/stock/{symbol}/quote"
            params = {
                'token': 'Tsk_2d1b7b6f2e274fe7ae5b95efb0f7a69c'  # Sandbox token (free)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=3) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return StreamingStockData(
                            symbol=symbol,
                            price=float(data['latestPrice']),
                            change=float(data['change']),
                            change_percent=float(data['changePercent']) * 100,
                            volume=int(data['latestVolume']),
                            high=float(data['high'] or data['latestPrice']),
                            low=float(data['low'] or data['latestPrice']),
                            open_price=float(data['open'] or data['latestPrice']),
                            market_cap=int(data.get('marketCap', 0)),
                            timestamp=datetime.now(),
                            source='iex'
                        )
                        
        except Exception as e:
            logger.warning(f"IEX API failed for {symbol}: {e}")
            return None

    async def _fetch_polygon_data(self, symbol: str) -> Optional[StreamingStockData]:
        """Polygon.io free tier data"""
        try:
            # Using Polygon.io free tier
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            params = {
                'apikey': 'W8qYPXqFPu_HmQaQ4I6Srd_VrAJxCJc1',  # Free tier key
                'adjusted': 'true'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=3) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('results') and len(data['results']) > 0:
                            result = data['results'][0]
                            
                            current_price = float(result['c'])
                            prev_close = float(result['o'])
                            change = current_price - prev_close
                            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
                            
                            return StreamingStockData(
                                symbol=symbol,
                                price=current_price,
                                change=round(change, 2),
                                change_percent=round(change_percent, 2),
                                volume=int(result['v']),
                                high=float(result['h']),
                                low=float(result['l']),
                                open_price=float(result['o']),
                                timestamp=datetime.now(),
                                source='polygon'
                            )
                            
        except Exception as e:
            logger.warning(f"Polygon API failed for {symbol}: {e}")
            return None

    async def _fetch_fmp_data(self, symbol: str) -> Optional[StreamingStockData]:
        """Financial Modeling Prep free data"""
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
            params = {
                'apikey': 'h7pbOAqzGnJW0D8aq3LJr65gfuLsL3Ht'  # Free tier key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=3) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and len(data) > 0:
                            quote = data[0]
                            
                            return StreamingStockData(
                                symbol=symbol,
                                price=float(quote['price']),
                                change=float(quote['change']),
                                change_percent=float(quote['changesPercentage']),
                                volume=int(quote['volume']),
                                high=float(quote['dayHigh']),
                                low=float(quote['dayLow']),
                                open_price=float(quote['open']),
                                market_cap=int(quote.get('marketCap', 0)),
                                timestamp=datetime.now(),
                                source='fmp'
                            )
                            
        except Exception as e:
            logger.warning(f"FMP API failed for {symbol}: {e}")
            return None

    async def get_fastest_data(self, symbol: str) -> Optional[StreamingStockData]:
        """Get data from fastest responding API"""
        tasks = []
        
        # Try all APIs concurrently
        for api_name, api_func in self.apis.items():
            task = asyncio.create_task(api_func(symbol))
            tasks.append((api_name, task))
        
        # Return first successful result
        for api_name, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=5)
                if result:
                    logger.info(f"✅ Fast data for {symbol} from {api_name}: ${result.price} ({result.change_percent:+.2f}%)")
                    return result
            except asyncio.TimeoutError:
                logger.warning(f"{api_name} timeout for {symbol}")
            except Exception as e:
                logger.warning(f"{api_name} error for {symbol}: {e}")
        
        # Cancel remaining tasks
        for _, task in tasks:
            if not task.done():
                task.cancel()
        
        return None

    async def stream_data_generator(self) -> AsyncGenerator[Dict, None]:
        """Generate streaming data for all symbols"""
        while self.is_streaming:
            start_time = time.time()
            
            # Fetch data for all symbols concurrently
            tasks = [self.get_fastest_data(symbol) for symbol in self.symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            stock_data = {}
            for i, result in enumerate(results):
                if isinstance(result, StreamingStockData):
                    stock_data[self.symbols[i]] = asdict(result)
            
            if stock_data:
                stream_update = {
                    'type': 'market_update',
                    'timestamp': datetime.now().isoformat(),
                    'data': stock_data,
                    'count': len(stock_data),
                    'fetch_time_ms': int((time.time() - start_time) * 1000)
                }
                
                yield stream_update
                
            # Stream every 3 seconds for real-time feel
            await asyncio.sleep(3)

    async def add_websocket_connection(self, websocket):
        """Add WebSocket connection for streaming"""
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")

    async def remove_websocket_connection(self, websocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast_to_websockets(self, data: Dict):
        """Broadcast data to all connected WebSocket clients"""
        if not self.active_connections:
            return
            
        message = json.dumps(data, default=str)
        disconnected = set()
        
        for websocket in self.active_connections.copy():
            try:
                await websocket.send(message)
            except Exception as e:
                logger.warning(f"WebSocket send failed: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.active_connections -= disconnected

    async def start_streaming(self):
        """Start the streaming service"""
        if self.is_streaming:
            return
            
        self.is_streaming = True
        logger.info("🚀 Starting high-speed streaming service...")
        
        async for stream_data in self.stream_data_generator():
            # Broadcast to WebSocket clients
            await self.broadcast_to_websockets(stream_data)
            
            # Log streaming stats
            logger.info(f"📡 Streamed {stream_data['count']} stocks in {stream_data['fetch_time_ms']}ms")

    def stop_streaming(self):
        """Stop the streaming service"""
        self.is_streaming = False
        logger.info("⏹️ Streaming service stopped")

    async def get_streaming_status(self) -> Dict:
        """Get current streaming status"""
        return {
            'streaming': self.is_streaming,
            'active_connections': len(self.active_connections),
            'supported_apis': list(self.apis.keys()),
            'symbols': self.symbols,
            'update_interval_seconds': 3
        }

# Global streaming service instance
streaming_service = HighSpeedStreamingService()