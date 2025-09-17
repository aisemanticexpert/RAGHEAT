"""
Streaming Neo4j Service - Integrates live streaming data with Neo4j
This service is optimized for Docker containerized environments
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from neo4j import GraphDatabase
import json

logger = logging.getLogger(__name__)

class StreamingNeo4jService:
    """
    Optimized Neo4j service for streaming live market data
    Designed specifically for Docker containerized deployment
    """
    
    def __init__(self, 
                 uri: str = "bolt://neo4j:7687",
                 user: str = "neo4j", 
                 password: str = "password"):
        
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.is_connected = False
        self._connection_task = None
        
    async def ensure_connected(self):
        """Ensure connection is established"""
        if not self.is_connected and self._connection_task is None:
            self._connection_task = asyncio.create_task(self.connect())
        
        if self._connection_task:
            await self._connection_task
            
        return self.is_connected

    async def connect(self):
        """Connect to Neo4j database with retry logic"""
        max_retries = 10
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                
                self.is_connected = True
                logger.info(f"✅ Connected to Neo4j at {self.uri}")
                
                # Initialize schema
                await self.initialize_schema()
                return True
                
            except Exception as e:
                logger.warning(f"Neo4j connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Failed to connect to Neo4j after all retries")
                    self.is_connected = False
                    return False

    async def initialize_schema(self):
        """Initialize Neo4j schema for streaming data"""
        if not self.is_connected:
            return

        constraints_and_indexes = [
            # Constraints
            "CREATE CONSTRAINT stock_symbol IF NOT EXISTS FOR (s:Stock) REQUIRE s.symbol IS UNIQUE",
            "CREATE CONSTRAINT sector_name IF NOT EXISTS FOR (sec:Sector) REQUIRE sec.name IS UNIQUE",
            "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (ms:MarketSession) REQUIRE ms.session_id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX stock_timestamp IF NOT EXISTS FOR (s:Stock) ON (s.last_updated)",
            "CREATE INDEX price_change IF NOT EXISTS FOR (s:Stock) ON (s.change_percent)",
            "CREATE INDEX market_session_timestamp IF NOT EXISTS FOR (ms:MarketSession) ON (ms.timestamp)",
        ]

        try:
            with self.driver.session() as session:
                for query in constraints_and_indexes:
                    try:
                        session.run(query)
                    except Exception as e:
                        # Constraint/index may already exist
                        logger.debug(f"Schema query skipped (likely exists): {query[:50]}...")
            
            logger.info("✅ Neo4j schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {e}")

    async def populate_streaming_data(self, streaming_data: Dict):
        """
        Populate Neo4j with live streaming data
        streaming_data format: {'AAPL': StreamingStockData, ...}
        """
        await self.ensure_connected()
        
        if not self.is_connected or not streaming_data:
            return False

        try:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create market session
            market_session_query = """
            MERGE (ms:MarketSession {session_id: $session_id})
            SET ms.timestamp = datetime($timestamp),
                ms.total_stocks = $total_stocks,
                ms.status = 'LIVE'
            RETURN ms.session_id as session_id
            """
            
            # Create/update stocks with streaming data
            stock_update_query = """
            UNWIND $stocks as stock_data
            MERGE (s:Stock {symbol: stock_data.symbol})
            SET s.price = stock_data.price,
                s.change = stock_data.change,
                s.change_percent = stock_data.change_percent,
                s.volume = stock_data.volume,
                s.high = stock_data.high,
                s.low = stock_data.low,
                s.open_price = stock_data.open_price,
                s.data_source = stock_data.source,
                s.last_updated = datetime($timestamp)
            
            MERGE (sec:Sector {name: COALESCE(stock_data.sector, 'Unknown')})
            MERGE (s)-[:BELONGS_TO]->(sec)
            
            WITH s, stock_data
            MATCH (ms:MarketSession {session_id: $session_id})
            MERGE (s)-[:TRADED_IN]->(ms)
            
            RETURN s.symbol as symbol, s.price as price
            """

            # Prepare stock data for Neo4j
            stocks_data = []
            for symbol, stock in streaming_data.items():
                stock_record = {
                    'symbol': symbol,
                    'price': float(stock.price),
                    'change': float(stock.change),
                    'change_percent': float(stock.change_percent),
                    'volume': int(stock.volume),
                    'high': float(getattr(stock, 'high', stock.price)),
                    'low': float(getattr(stock, 'low', stock.price)),
                    'open_price': float(getattr(stock, 'open_price', stock.price)),
                    'source': getattr(stock, 'source', 'yahoo'),
                    'sector': self._get_sector_for_symbol(symbol)
                }
                stocks_data.append(stock_record)

            # Execute queries
            with self.driver.session() as session:
                # Create market session
                session.run(
                    market_session_query,
                    session_id=session_id,
                    timestamp=datetime.now().isoformat(),
                    total_stocks=len(stocks_data)
                )
                
                # Update stocks
                result = session.run(
                    stock_update_query,
                    stocks=stocks_data,
                    session_id=session_id,
                    timestamp=datetime.now().isoformat()
                )
                
                updated_stocks = [record["symbol"] for record in result]
                
            logger.info(f"✅ Updated {len(updated_stocks)} stocks in Neo4j: {', '.join(updated_stocks[:5])}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to populate Neo4j with streaming data: {e}")
            return False

    def _get_sector_for_symbol(self, symbol: str) -> str:
        """Get sector for stock symbol"""
        sectors = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "NVDA": "Technology", "META": "Technology", "AMZN": "Consumer Discretionary",
            "TSLA": "Consumer Discretionary", "NFLX": "Communication Services",
            "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
            "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
            "XOM": "Energy", "CVX": "Energy", "COP": "Energy"
        }
        return sectors.get(symbol, "Technology")

    async def get_live_graph_stats(self) -> Dict:
        """Get current Neo4j graph statistics"""
        await self.ensure_connected()
        
        if not self.is_connected:
            return {"connected": False, "error": "Not connected to Neo4j"}

        try:
            stats_query = """
            MATCH (s:Stock)
            OPTIONAL MATCH (sec:Sector)
            OPTIONAL MATCH (ms:MarketSession)
            RETURN 
                count(DISTINCT s) as stock_nodes,
                count(DISTINCT sec) as sector_nodes,
                count(DISTINCT ms) as market_sessions,
                max(s.last_updated) as last_updated
            """
            
            with self.driver.session() as session:
                result = session.run(stats_query)
                record = result.single()
                
                if record:
                    return {
                        "connected": True,
                        "stock_nodes": record["stock_nodes"],
                        "sector_nodes": record["sector_nodes"],
                        "market_sessions": record["market_sessions"],
                        "last_updated": record["last_updated"],
                        "mode": "live_streaming",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {"connected": True, "error": "No data found"}
                    
        except Exception as e:
            logger.error(f"Failed to get Neo4j stats: {e}")
            return {"connected": False, "error": str(e)}

    async def create_live_visualization_queries(self) -> List[str]:
        """Generate Cypher queries for live data visualization"""
        return [
            # Top performing stocks
            """
            MATCH (s:Stock)-[:BELONGS_TO]->(sec:Sector)
            WHERE s.last_updated IS NOT NULL
            RETURN s.symbol, s.price, s.change_percent, sec.name as sector
            ORDER BY s.change_percent DESC
            LIMIT 10
            """,
            
            # Sector performance heatmap
            """
            MATCH (s:Stock)-[:BELONGS_TO]->(sec:Sector)
            WHERE s.last_updated IS NOT NULL
            WITH sec.name as sector, 
                 avg(s.change_percent) as avg_change,
                 count(s) as stock_count,
                 sum(s.volume) as total_volume
            RETURN sector, avg_change, stock_count, total_volume
            ORDER BY avg_change DESC
            """,
            
            # Recent market activity
            """
            MATCH (s:Stock)-[:TRADED_IN]->(ms:MarketSession)
            WHERE ms.timestamp > datetime() - duration('PT1H')
            RETURN ms.session_id, ms.timestamp, ms.total_stocks, s.symbol, s.price, s.change_percent
            ORDER BY ms.timestamp DESC, s.change_percent DESC
            LIMIT 20
            """
        ]

    async def execute_live_query(self, query: str) -> List[Dict]:
        """Execute a live query and return results"""
        await self.ensure_connected()
        
        if not self.is_connected:
            return []

        try:
            with self.driver.session() as session:
                result = session.run(query)
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Failed to execute live query: {e}")
            return []

    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.is_connected = False
            logger.info("Neo4j connection closed")

# Global service instance for Docker containers
# Note: Initialize this only within an async context
streaming_neo4j_service = None

def get_streaming_neo4j_service() -> StreamingNeo4jService:
    """Get or create the global streaming Neo4j service instance"""
    global streaming_neo4j_service
    if streaming_neo4j_service is None:
        streaming_neo4j_service = StreamingNeo4jService()
    return streaming_neo4j_service