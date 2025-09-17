"""
Mock Neo4j Service for Development
Simulates Neo4j functionality when the database is unavailable
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import random

from real_time_data_service import real_time_service, LiveStockData

logger = logging.getLogger(__name__)

class MockNeo4jService:
    """Mock service that simulates Neo4j graph operations"""
    
    def __init__(self):
        self.is_connected = True  # Always "connected" in mock mode
        self.mock_data = {
            "stocks": {},
            "sectors": {},
            "relationships": [],
            "market_sessions": []
        }
        
    async def connect(self):
        """Mock connection - always succeeds"""
        self.is_connected = True
        logger.info("Mock Neo4j service connected")
        
    async def close(self):
        """Mock close"""
        self.is_connected = False
        
    async def populate_live_data(self, symbols: List[str] = None) -> bool:
        """Mock data population with realistic market data"""
        try:
            if not symbols:
                symbols = [
                    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
                    "JPM", "BAC", "GS", "WFC", "JNJ", "PFE", "UNH", "ABBV"
                ]
            
            # Get live stock data
            stock_data = await real_time_service.get_multiple_stocks(symbols)
            
            # Store in mock database
            for symbol, data in stock_data.items():
                self.mock_data["stocks"][symbol] = {
                    "symbol": data.symbol,
                    "price": data.price,
                    "change": data.change,
                    "change_percent": data.change_percent,
                    "volume": data.volume,
                    "sector": data.sector,
                    "heat_score": data.heat_score,
                    "volatility": data.volatility,
                    "last_updated": data.timestamp.isoformat()
                }
            
            # Mock sector aggregation
            sectors = {}
            for symbol, data in stock_data.items():
                sector = data.sector
                if sector not in sectors:
                    sectors[sector] = {
                        "stocks": [],
                        "total_change": 0,
                        "total_volume": 0,
                        "heat_scores": []
                    }
                
                sectors[sector]["stocks"].append(symbol)
                sectors[sector]["total_change"] += data.change_percent
                sectors[sector]["total_volume"] += data.volume
                sectors[sector]["heat_scores"].append(data.heat_score)
            
            # Store sector data
            for sector, info in sectors.items():
                stock_count = len(info["stocks"])
                self.mock_data["sectors"][sector] = {
                    "name": sector,
                    "avg_change": info["total_change"] / stock_count,
                    "total_volume": info["total_volume"],
                    "avg_heat_score": sum(info["heat_scores"]) / stock_count,
                    "stock_count": stock_count,
                    "top_stocks": info["stocks"][:3]
                }
            
            # Mock relationships
            self.mock_data["relationships"] = []
            hot_stocks = [symbol for symbol, data in stock_data.items() if data.heat_score > 0.7]
            
            for i, stock1 in enumerate(hot_stocks):
                for stock2 in hot_stocks[i+1:]:
                    self.mock_data["relationships"].append({
                        "from": stock1,
                        "to": stock2,
                        "type": "HEAT_RELATED",
                        "strength": random.uniform(0.7, 1.0)
                    })
            
            # Mock market session
            market_overview = await real_time_service.get_market_overview()
            session = {
                "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M')}",
                "market_direction": market_overview.get("market_direction", "UP"),
                "market_heat_status": market_overview.get("market_heat_status", "HOT"),
                "timestamp": datetime.now().isoformat()
            }
            self.mock_data["market_sessions"].append(session)
            
            logger.info(f"Mock Neo4j populated with {len(stock_data)} stocks across {len(sectors)} sectors")
            return True
            
        except Exception as e:
            logger.error(f"Error in mock Neo4j population: {e}")
            return False
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get mock graph statistics"""
        return {
            "stock_nodes": len(self.mock_data["stocks"]),
            "sector_nodes": len(self.mock_data["sectors"]),
            "market_sessions": len(self.mock_data["market_sessions"]),
            "relationships": len(self.mock_data["relationships"]),
            "last_updated": datetime.now().isoformat(),
            "mode": "mock_database"
        }
    
    async def get_hot_stocks(self) -> List[Dict]:
        """Get stocks with high heat scores"""
        hot_stocks = []
        for symbol, data in self.mock_data["stocks"].items():
            if data.get("heat_score", 0) > 0.7:
                hot_stocks.append(data)
        
        return sorted(hot_stocks, key=lambda x: x["heat_score"], reverse=True)[:10]
    
    async def get_sector_performance(self) -> List[Dict]:
        """Get sector performance rankings"""
        sectors = list(self.mock_data["sectors"].values())
        return sorted(sectors, key=lambda x: x["avg_change"], reverse=True)
    
    async def get_correlations(self) -> List[Dict]:
        """Get stock correlations"""
        return self.mock_data["relationships"]
    
    async def start_live_population(self, interval_seconds: int = 60):
        """Start continuous mock data population"""
        logger.info(f"Starting mock Neo4j live population every {interval_seconds} seconds")
        
        import asyncio
        while True:
            try:
                await self.populate_live_data()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in mock continuous population: {e}")
                await asyncio.sleep(interval_seconds)

# Global mock service instance
mock_neo4j_service = MockNeo4jService()