"""
Professional Neo4j Streaming Service
High-performance data streaming service designed to handle billions of nodes
Implements industry best practices for knowledge graph population
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import aioredis
from neo4j import AsyncGraphDatabase
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketNode:
    """Professional market node structure"""
    symbol: str
    name: str
    sector: str
    market_cap: float
    price: float
    volume: int
    heat_level: float
    sentiment_score: float
    volatility: float
    correlation_matrix: Dict[str, float]
    technical_indicators: Dict[str, float]
    timestamp: datetime

@dataclass 
class MarketRelationship:
    """Professional market relationship structure"""
    source: str
    target: str
    relationship_type: str
    weight: float
    correlation: float
    strength: float
    timestamp: datetime

class ProfessionalNeo4jStreamingService:
    """
    Professional streaming service for Neo4j knowledge graph population
    Designed to handle billions of nodes with optimal performance
    """
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "ragheat123",
                 redis_url: str = "redis://localhost:6379"):
        
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user  
        self.neo4j_password = neo4j_password
        self.redis_url = redis_url
        
        self.driver = None
        self.redis = None
        
        # Professional market configuration
        self.sp500_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 
            'JNJ', 'V', 'WMT', 'JPM', 'MA', 'PG', 'UNH', 'HD', 'CVX', 'ABBV',
            'PFE', 'KO', 'BAC', 'PEP', 'COST', 'TMO', 'AVGO', 'XOM', 'MRK',
            'ADBE', 'NFLX', 'CRM', 'ACN', 'LIN', 'VZ', 'ORCL', 'NKE', 'T',
            'CSCO', 'AMD', 'DHR', 'ABT', 'TXN', 'BMY', 'QCOM', 'NEE', 'PM',
            'WFC', 'COP', 'UPS', 'RTX', 'LOW', 'SCHW', 'LMT', 'MDT', 'INTC'
        ]
        
        self.sector_mapping = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary', 'NVDA': 'Technology', 'TSLA': 'Consumer Discretionary',
            'META': 'Technology', 'JNJ': 'Healthcare', 'V': 'Financial Services',
            'WMT': 'Consumer Staples', 'JPM': 'Financial Services', 'MA': 'Financial Services',
            'PG': 'Consumer Staples', 'UNH': 'Healthcare', 'HD': 'Consumer Discretionary',
            'CVX': 'Energy', 'ABBV': 'Healthcare', 'PFE': 'Healthcare', 'KO': 'Consumer Staples',
            'BAC': 'Financial Services', 'PEP': 'Consumer Staples', 'COST': 'Consumer Staples'
        }
        
    async def initialize(self):
        """Initialize connections to Neo4j and Redis"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password),
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=100,     # Handle high load
                connection_acquisition_timeout=30 # 30 seconds timeout
            )
            
            self.redis = await aioredis.from_url(self.redis_url, decode_responses=True)
            
            # Test connections
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            
            await self.redis.ping()
            
            logger.info("✅ Professional Neo4j Streaming Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize service: {e}")
            raise

    async def create_indexes_and_constraints(self):
        """Create optimized indexes and constraints for billions of nodes"""
        
        index_queries = [
            # Node indexes for performance
            "CREATE INDEX stock_symbol IF NOT EXISTS FOR (s:Stock) ON (s.symbol)",
            "CREATE INDEX stock_sector IF NOT EXISTS FOR (s:Stock) ON (s.sector)",
            "CREATE INDEX stock_heat IF NOT EXISTS FOR (s:Stock) ON (s.heat_level)",
            "CREATE INDEX stock_timestamp IF NOT EXISTS FOR (s:Stock) ON (s.timestamp)",
            
            "CREATE INDEX sector_name IF NOT EXISTS FOR (sec:Sector) ON (sec.name)",
            
            # Constraints for data integrity
            "CREATE CONSTRAINT stock_symbol_unique IF NOT EXISTS FOR (s:Stock) ON (s.symbol)",
            "CREATE CONSTRAINT sector_name_unique IF NOT EXISTS FOR (sec:Sector) ON (sec.name)",
            
            # Composite indexes for complex queries
            "CREATE INDEX stock_composite IF NOT EXISTS FOR (s:Stock) ON (s.sector, s.heat_level, s.timestamp)",
            
            # Relationship indexes
            "CREATE INDEX rel_correlation IF NOT EXISTS FOR ()-[r:CORRELATES_WITH]-() ON (r.correlation)",
            "CREATE INDEX rel_belongs_to IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() ON (r.weight)",
        ]
        
        async with self.driver.session() as session:
            for query in index_queries:
                try:
                    await session.run(query)
                    logger.info(f"✅ Created: {query}")
                except Exception as e:
                    logger.warning(f"⚠️ Index/constraint may already exist: {query}")

    async def fetch_historical_data(self, days: int = 10) -> Dict[str, pd.DataFrame]:
        """Fetch 10 days of historical market data for ontology population"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        historical_data = {}
        
        logger.info(f"📊 Fetching {days} days of historical data for {len(self.sp500_symbols)} symbols...")
        
        # Batch download for efficiency
        tickers = yf.Tickers(' '.join(self.sp500_symbols[:20]))  # Limit for demo
        
        for symbol in self.sp500_symbols[:20]:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1h')
                
                if not data.empty:
                    # Calculate technical indicators
                    data['Returns'] = data['Close'].pct_change()
                    data['Volatility'] = data['Returns'].rolling(window=24).std()
                    data['RSI'] = self.calculate_rsi(data['Close'])
                    data['MACD'] = self.calculate_macd(data['Close'])
                    data['Heat_Level'] = self.calculate_heat_level(data)
                    
                    historical_data[symbol] = data
                    
            except Exception as e:
                logger.error(f"❌ Failed to fetch data for {symbol}: {e}")
                
        logger.info(f"✅ Successfully fetched historical data for {len(historical_data)} symbols")
        return historical_data
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd
    
    def calculate_heat_level(self, data: pd.DataFrame) -> pd.Series:
        """Calculate sophisticated heat level based on multiple factors"""
        
        # Volume spike factor
        volume_ma = data['Volume'].rolling(window=24).mean()
        volume_factor = (data['Volume'] / volume_ma).fillna(1)
        
        # Price movement factor
        price_change = abs(data['Close'].pct_change())
        
        # Volatility factor
        volatility_factor = data['Volatility'].fillna(0)
        
        # Combined heat level (0-100)
        heat = (volume_factor * 30 + price_change * 100 * 40 + volatility_factor * 100 * 30)
        heat = np.clip(heat, 0, 100)
        
        return heat

    async def populate_ontology_from_historical_data(self, historical_data: Dict[str, pd.DataFrame]):
        """Populate Neo4j ontology with 10 days of historical data"""
        
        logger.info("🏗️ Populating ontology with historical data...")
        
        async with self.driver.session() as session:
            
            # Create sectors first
            sectors = set(self.sector_mapping.values())
            for sector in sectors:
                query = """
                MERGE (s:Sector {name: $sector})
                SET s.created_at = datetime(),
                    s.node_count = 0,
                    s.avg_heat = 0.0
                """
                await session.run(query, sector=sector)
            
            # Process each stock's historical data
            batch_size = 1000  # Process in batches for performance
            batch_data = []
            
            for symbol, data in historical_data.items():
                sector = self.sector_mapping.get(symbol, 'Technology')
                
                # Process each time point
                for timestamp, row in data.iterrows():
                    
                    node_data = {
                        'symbol': symbol,
                        'sector': sector,
                        'timestamp': timestamp.isoformat(),
                        'price': float(row['Close']) if pd.notna(row['Close']) else 0.0,
                        'volume': int(row['Volume']) if pd.notna(row['Volume']) else 0,
                        'heat_level': float(row['Heat_Level']) if pd.notna(row['Heat_Level']) else 0.0,
                        'rsi': float(row['RSI']) if pd.notna(row['RSI']) else 50.0,
                        'macd': float(row['MACD']) if pd.notna(row['MACD']) else 0.0,
                        'volatility': float(row['Volatility']) if pd.notna(row['Volatility']) else 0.0
                    }
                    
                    batch_data.append(node_data)
                    
                    # Process batch when full
                    if len(batch_data) >= batch_size:
                        await self.process_historical_batch(session, batch_data)
                        batch_data = []
            
            # Process remaining data
            if batch_data:
                await self.process_historical_batch(session, batch_data)
                
        logger.info("✅ Historical data population completed")
    
    async def process_historical_batch(self, session, batch_data: List[Dict]):
        """Process a batch of historical data efficiently"""
        
        query = """
        UNWIND $batch as row
        MERGE (s:Stock {symbol: row.symbol, timestamp: row.timestamp})
        SET s.price = row.price,
            s.volume = row.volume,
            s.heat_level = row.heat_level,
            s.rsi = row.rsi,
            s.macd = row.macd,
            s.volatility = row.volatility,
            s.sector = row.sector,
            s.updated_at = datetime()
            
        WITH s, row
        MERGE (sec:Sector {name: row.sector})
        MERGE (s)-[:BELONGS_TO {weight: 1.0, timestamp: row.timestamp}]->(sec)
        """
        
        await session.run(query, batch=batch_data)
        logger.info(f"✅ Processed batch of {len(batch_data)} historical records")

    async def start_synthetic_streaming(self):
        """Start streaming synthetic data to populate knowledge graph continuously"""
        
        logger.info("🚀 Starting synthetic data streaming...")
        
        while True:
            try:
                # Generate synthetic market data
                synthetic_data = await self.generate_synthetic_market_data()
                
                # Stream to Neo4j
                await self.stream_to_neo4j(synthetic_data)
                
                # Cache in Redis for frontend
                await self.cache_in_redis(synthetic_data)
                
                # Wait for next update (every 5 seconds for demo)
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ Streaming error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def generate_synthetic_market_data(self) -> List[MarketNode]:
        """Generate realistic synthetic market data"""
        
        nodes = []
        current_time = datetime.now()
        
        for symbol in self.sp500_symbols[:20]:  # Limit for demo
            
            # Simulate realistic market behavior
            base_price = np.random.uniform(50, 500)
            price_change = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price = base_price * (1 + price_change)
            
            # Generate correlated heat level
            volume_spike = np.random.exponential(1.0)
            price_volatility = abs(price_change) * 100
            heat_level = min(100, max(0, (volume_spike * 20 + price_volatility * 50)))
            
            # Technical indicators
            rsi = np.random.uniform(20, 80)
            macd = np.random.normal(0, 0.1)
            
            # Correlation matrix with other stocks
            correlations = {}
            for other_symbol in self.sp500_symbols[:10]:
                if other_symbol != symbol:
                    correlations[other_symbol] = np.random.uniform(-0.5, 0.9)
            
            node = MarketNode(
                symbol=symbol,
                name=f"{symbol} Inc.",
                sector=self.sector_mapping.get(symbol, 'Technology'),
                market_cap=np.random.uniform(1e9, 1e12),  # 1B to 1T
                price=current_price,
                volume=int(np.random.uniform(1e6, 1e8)),   # 1M to 100M
                heat_level=heat_level,
                sentiment_score=np.random.uniform(-1, 1),
                volatility=abs(price_change),
                correlation_matrix=correlations,
                technical_indicators={
                    'rsi': rsi,
                    'macd': macd,
                    'bollinger_upper': current_price * 1.05,
                    'bollinger_lower': current_price * 0.95
                },
                timestamp=current_time
            )
            
            nodes.append(node)
            
        return nodes
    
    async def stream_to_neo4j(self, nodes: List[MarketNode]):
        """Stream data to Neo4j with optimized queries"""
        
        async with self.driver.session() as session:
            
            # Batch insert nodes
            node_batch = []
            relationship_batch = []
            
            for node in nodes:
                node_data = {
                    'symbol': node.symbol,
                    'name': node.name,
                    'sector': node.sector,
                    'market_cap': node.market_cap,
                    'price': node.price,
                    'volume': node.volume,
                    'heat_level': node.heat_level,
                    'sentiment_score': node.sentiment_score,
                    'volatility': node.volatility,
                    'rsi': node.technical_indicators['rsi'],
                    'macd': node.technical_indicators['macd'],
                    'timestamp': node.timestamp.isoformat()
                }
                node_batch.append(node_data)
                
                # Create correlation relationships
                for other_symbol, correlation in node.correlation_matrix.items():
                    if correlation > 0.5:  # Only significant correlations
                        relationship_batch.append({
                            'source': node.symbol,
                            'target': other_symbol,
                            'correlation': correlation,
                            'weight': correlation,
                            'timestamp': node.timestamp.isoformat()
                        })
            
            # Execute batch operations
            await self.execute_node_batch(session, node_batch)
            await self.execute_relationship_batch(session, relationship_batch)
    
    async def execute_node_batch(self, session, node_batch: List[Dict]):
        """Execute optimized node batch insertion"""
        
        query = """
        UNWIND $batch as row
        MERGE (s:Stock {symbol: row.symbol})
        SET s.name = row.name,
            s.sector = row.sector,
            s.market_cap = row.market_cap,
            s.price = row.price,
            s.volume = row.volume,
            s.heat_level = row.heat_level,
            s.sentiment_score = row.sentiment_score,
            s.volatility = row.volatility,
            s.rsi = row.rsi,
            s.macd = row.macd,
            s.timestamp = row.timestamp,
            s.updated_at = datetime()
            
        WITH s, row
        MERGE (sec:Sector {name: row.sector})
        MERGE (s)-[:BELONGS_TO {weight: 1.0, updated_at: datetime()}]->(sec)
        """
        
        await session.run(query, batch=node_batch)
        
    async def execute_relationship_batch(self, session, relationship_batch: List[Dict]):
        """Execute optimized relationship batch insertion"""
        
        if not relationship_batch:
            return
            
        query = """
        UNWIND $batch as row
        MATCH (s1:Stock {symbol: row.source})
        MATCH (s2:Stock {symbol: row.target})
        MERGE (s1)-[r:CORRELATES_WITH]-(s2)
        SET r.correlation = row.correlation,
            r.weight = row.weight,
            r.timestamp = row.timestamp,
            r.updated_at = datetime()
        """
        
        await session.run(query, batch=relationship_batch)
    
    async def cache_in_redis(self, nodes: List[MarketNode]):
        """Cache current state in Redis for fast frontend access"""
        
        # Cache individual node data
        for node in nodes:
            key = f"stock:{node.symbol}"
            data = {
                'symbol': node.symbol,
                'price': node.price,
                'heat_level': node.heat_level,
                'volume': node.volume,
                'sector': node.sector,
                'timestamp': node.timestamp.isoformat()
            }
            
            await self.redis.hset(key, mapping=data)
            await self.redis.expire(key, 3600)  # 1 hour TTL
            
        # Cache aggregated market status
        market_summary = {
            'total_stocks': len(nodes),
            'avg_heat': sum(n.heat_level for n in nodes) / len(nodes),
            'hot_stocks': len([n for n in nodes if n.heat_level > 70]),
            'last_update': datetime.now().isoformat()
        }
        
        await self.redis.hset("market:summary", mapping=market_summary)
        await self.redis.expire("market:summary", 300)  # 5 minutes TTL

    async def get_graph_for_frontend(self) -> Dict[str, Any]:
        """Get optimized graph data for frontend visualization"""
        
        async with self.driver.session() as session:
            
            # Query for nodes with latest data
            node_query = """
            MATCH (s:Stock)
            OPTIONAL MATCH (s)-[:BELONGS_TO]->(sec:Sector)
            RETURN s.symbol as symbol, s.name as name, s.sector as sector,
                   s.price as price, s.volume as volume, s.heat_level as heat_level,
                   s.rsi as rsi, s.macd as macd, s.timestamp as timestamp,
                   sec.name as sector_name
            ORDER BY s.heat_level DESC
            LIMIT 50
            """
            
            # Query for relationships
            rel_query = """
            MATCH (s1:Stock)-[r:CORRELATES_WITH]-(s2:Stock)
            WHERE r.correlation > 0.3
            RETURN s1.symbol as source, s2.symbol as target, 
                   r.correlation as correlation, r.weight as weight
            LIMIT 100
            """
            
            nodes_result = await session.run(node_query)
            relationships_result = await session.run(rel_query)
            
            # Process results
            nodes = []
            async for record in nodes_result:
                node = {
                    'id': record['symbol'],
                    'label': f"{record['symbol']} - {record['name'][:15]}...",
                    'type': record['sector'].lower().replace(' ', '_'),
                    'heat_level': record['heat_level'] or 0,
                    'price': record['price'] or 0,
                    'volume': record['volume'] or 0,
                    'rsi': record['rsi'] or 50,
                    'macd': record['macd'] or 0
                }
                nodes.append(node)
            
            relationships = []
            async for record in relationships_result:
                rel = {
                    'source': record['source'],
                    'target': record['target'],
                    'correlation': record['correlation'],
                    'weight': record['weight']
                }
                relationships.append(rel)
                
            return {
                'success': True,
                'data': {
                    'nodes': nodes,
                    'relationships': relationships,
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'total_nodes': len(nodes),
                        'total_edges': len(relationships)
                    }
                }
            }

    async def cleanup(self):
        """Cleanup connections"""
        if self.driver:
            await self.driver.close()
        if self.redis:
            await self.redis.close()
        logger.info("🧹 Service cleanup completed")

# Example usage
async def main():
    """Main function to demonstrate the service"""
    
    service = ProfessionalNeo4jStreamingService()
    
    try:
        await service.initialize()
        await service.create_indexes_and_constraints()
        
        # Fetch and populate historical data
        historical_data = await service.fetch_historical_data(10)
        await service.populate_ontology_from_historical_data(historical_data)
        
        # Start synthetic streaming (run in background)
        await service.start_synthetic_streaming()
        
    except KeyboardInterrupt:
        logger.info("🛑 Service stopped by user")
    except Exception as e:
        logger.error(f"❌ Service error: {e}")
    finally:
        await service.cleanup()

if __name__ == "__main__":
    asyncio.run(main())