"""
Neo4j Live Data Population Service
Populates Neo4j graph with real-time market data and relationships
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase
import json

from real_time_data_service import real_time_service, LiveStockData

logger = logging.getLogger(__name__)

class Neo4jLiveDataService:
    """Service for populating Neo4j with live market data"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.is_connected = False

    async def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            self.is_connected = True
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.is_connected = False

    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.is_connected = False

    def create_stock_node(self, tx, stock_data: LiveStockData):
        """Create or update a stock node with current data"""
        query = """
        MERGE (s:Stock {symbol: $symbol})
        SET s.price = $price,
            s.change = $change,
            s.change_percent = $change_percent,
            s.volume = $volume,
            s.market_cap = $market_cap,
            s.sector = $sector,
            s.heat_score = $heat_score,
            s.volatility = $volatility,
            s.rsi = $rsi,
            s.bollinger_position = $bollinger_position,
            s.volume_surge = $volume_surge,
            s.last_updated = datetime($timestamp)
        RETURN s
        """
        
        return tx.run(query, 
            symbol=stock_data.symbol,
            price=stock_data.price,
            change=stock_data.change,
            change_percent=stock_data.change_percent,
            volume=stock_data.volume,
            market_cap=stock_data.market_cap,
            sector=stock_data.sector,
            heat_score=stock_data.heat_score,
            volatility=stock_data.volatility,
            rsi=stock_data.rsi,
            bollinger_position=stock_data.bollinger_position,
            volume_surge=stock_data.volume_surge,
            timestamp=stock_data.timestamp.isoformat()
        )

    def create_sector_node(self, tx, sector_name: str, sector_data: Dict[str, Any]):
        """Create or update a sector node with aggregated data"""
        query = """
        MERGE (sect:Sector {name: $sector_name})
        SET sect.avg_change = $avg_change,
            sect.total_volume = $total_volume,
            sect.avg_heat_score = $avg_heat_score,
            sect.stock_count = $stock_count,
            sect.last_updated = datetime($timestamp)
        RETURN sect
        """
        
        return tx.run(query,
            sector_name=sector_name,
            avg_change=sector_data['avg_change'],
            total_volume=sector_data['total_volume'],
            avg_heat_score=sector_data['avg_heat_score'],
            stock_count=sector_data['stock_count'],
            timestamp=datetime.now().isoformat()
        )

    def create_stock_sector_relationship(self, tx, stock_symbol: str, sector_name: str):
        """Create relationship between stock and sector"""
        query = """
        MATCH (s:Stock {symbol: $stock_symbol})
        MATCH (sect:Sector {name: $sector_name})
        MERGE (s)-[:BELONGS_TO]->(sect)
        """
        
        return tx.run(query, stock_symbol=stock_symbol, sector_name=sector_name)

    def create_correlation_relationships(self, tx, stock_data: Dict[str, LiveStockData]):
        """Create correlation relationships between stocks"""
        symbols = list(stock_data.keys())
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                data1 = stock_data[symbol1]
                data2 = stock_data[symbol2]
                
                # Simple correlation based on price movement direction
                correlation_strength = 1.0 if (data1.change > 0) == (data2.change > 0) else 0.5
                
                # Only create relationships for strong correlations
                if correlation_strength > 0.7:
                    query = """
                    MATCH (s1:Stock {symbol: $symbol1})
                    MATCH (s2:Stock {symbol: $symbol2})
                    MERGE (s1)-[r:CORRELATED_WITH]-(s2)
                    SET r.strength = $strength,
                        r.last_updated = datetime($timestamp)
                    """
                    
                    tx.run(query,
                        symbol1=symbol1,
                        symbol2=symbol2,
                        strength=correlation_strength,
                        timestamp=datetime.now().isoformat()
                    )

    def create_heat_relationships(self, tx, stock_data: Dict[str, LiveStockData]):
        """Create heat-based relationships between hot stocks"""
        hot_stocks = {symbol: data for symbol, data in stock_data.items() if data.heat_score > 0.7}
        
        for symbol1, data1 in hot_stocks.items():
            for symbol2, data2 in hot_stocks.items():
                if symbol1 != symbol2:
                    # Create HEAT_RELATED relationship
                    combined_heat = (data1.heat_score + data2.heat_score) / 2
                    
                    query = """
                    MATCH (s1:Stock {symbol: $symbol1})
                    MATCH (s2:Stock {symbol: $symbol2})
                    MERGE (s1)-[r:HEAT_RELATED]->(s2)
                    SET r.combined_heat = $combined_heat,
                        r.last_updated = datetime($timestamp)
                    """
                    
                    tx.run(query,
                        symbol1=symbol1,
                        symbol2=symbol2,
                        combined_heat=combined_heat,
                        timestamp=datetime.now().isoformat()
                    )

    def create_market_session(self, tx, market_overview: Dict[str, Any]):
        """Create a market session node with current market state"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        query = """
        MERGE (ms:MarketSession {session_id: $session_id})
        SET ms.market_direction = $market_direction,
            ms.market_heat_status = $market_heat_status,
            ms.average_change = $average_change,
            ms.market_heat_score = $market_heat_score,
            ms.total_volume = $total_volume,
            ms.active_stocks = $active_stocks,
            ms.timestamp = datetime($timestamp)
        RETURN ms
        """
        
        return tx.run(query,
            session_id=session_id,
            market_direction=market_overview['market_direction'],
            market_heat_status=market_overview['market_heat_status'],
            average_change=market_overview['average_change'],
            market_heat_score=market_overview['market_heat_score'],
            total_volume=market_overview['total_volume'],
            active_stocks=market_overview['active_stocks'],
            timestamp=market_overview['timestamp']
        )

    async def populate_live_data(self, symbols: List[str] = None):
        """Main method to populate Neo4j with live market data"""
        if not self.is_connected:
            await self.connect()
            
        if not self.is_connected:
            logger.error("Cannot populate data - Neo4j not connected")
            return False

        try:
            # Default symbols if none provided
            if not symbols:
                symbols = [
                    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
                    "JPM", "BAC", "GS", "WFC", "JNJ", "PFE", "UNH", "ABBV",
                    "XOM", "CVX", "COP", "EOG"
                ]
            
            # Get live stock data
            logger.info(f"Fetching live data for {len(symbols)} stocks")
            stock_data = await real_time_service.get_multiple_stocks(symbols)
            
            # Get market overview
            market_overview = await real_time_service.get_market_overview()
            
            if not stock_data:
                logger.error("No stock data received")
                return False

            # Populate Neo4j in a transaction
            with self.driver.session() as session:
                # Create/update stock nodes
                for symbol, data in stock_data.items():
                    session.execute_write(self.create_stock_node, data)
                
                # Calculate sector data
                sectors = {}
                for symbol, data in stock_data.items():
                    sector = data.sector
                    if sector not in sectors:
                        sectors[sector] = {
                            'changes': [],
                            'volumes': [],
                            'heat_scores': [],
                            'stock_count': 0
                        }
                    
                    sectors[sector]['changes'].append(data.change_percent)
                    sectors[sector]['volumes'].append(data.volume)
                    sectors[sector]['heat_scores'].append(data.heat_score)
                    sectors[sector]['stock_count'] += 1

                # Create/update sector nodes
                for sector_name, sector_info in sectors.items():
                    sector_data = {
                        'avg_change': sum(sector_info['changes']) / len(sector_info['changes']),
                        'total_volume': sum(sector_info['volumes']),
                        'avg_heat_score': sum(sector_info['heat_scores']) / len(sector_info['heat_scores']),
                        'stock_count': sector_info['stock_count']
                    }
                    session.execute_write(self.create_sector_node, sector_name, sector_data)
                
                # Create stock-sector relationships
                for symbol, data in stock_data.items():
                    session.execute_write(self.create_stock_sector_relationship, symbol, data.sector)
                
                # Create correlation relationships
                session.execute_write(self.create_correlation_relationships, stock_data)
                
                # Create heat relationships
                session.execute_write(self.create_heat_relationships, stock_data)
                
                # Create market session
                session.execute_write(self.create_market_session, market_overview)

            logger.info(f"Successfully populated Neo4j with data for {len(stock_data)} stocks across {len(sectors)} sectors")
            return True
            
        except Exception as e:
            logger.error(f"Error populating Neo4j with live data: {e}")
            return False

    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the current graph"""
        if not self.is_connected:
            return {"error": "Neo4j not connected"}
        
        try:
            with self.driver.session() as session:
                # Count nodes and relationships
                stats_query = """
                MATCH (s:Stock) 
                WITH count(s) as stock_count
                MATCH (sect:Sector)
                WITH stock_count, count(sect) as sector_count
                MATCH (ms:MarketSession)
                WITH stock_count, sector_count, count(ms) as session_count
                MATCH ()-[r]->()
                RETURN stock_count, sector_count, session_count, count(r) as relationship_count
                """
                
                result = session.run(stats_query)
                record = result.single()
                
                if record:
                    return {
                        "stock_nodes": record["stock_count"],
                        "sector_nodes": record["sector_count"], 
                        "market_sessions": record["session_count"],
                        "relationships": record["relationship_count"],
                        "last_updated": datetime.now().isoformat()
                    }
                else:
                    return {"error": "No data found in graph"}
                    
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {"error": str(e)}

    async def start_live_population(self, interval_seconds: int = 60):
        """Start continuous live data population"""
        logger.info(f"Starting continuous Neo4j live data population every {interval_seconds} seconds")
        
        while True:
            try:
                await self.populate_live_data()
                logger.info(f"Live data populated, sleeping for {interval_seconds} seconds")
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in continuous population: {e}")
                await asyncio.sleep(interval_seconds)

# Global service instance
neo4j_live_service = Neo4jLiveDataService()