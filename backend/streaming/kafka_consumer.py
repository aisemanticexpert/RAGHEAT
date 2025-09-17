"""
Kafka Consumer for Neo4j Ontology Population
Consumes market data from Kafka topics and populates Neo4j ontology
"""
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.redis_json_storage import RedisJSONStorage
from graph.advanced_neo4j_manager import AdvancedNeo4jManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaOntologyConsumer:
    def __init__(self):
        self.consumer = None
        self.redis_storage = RedisJSONStorage()
        self.neo4j_manager = AdvancedNeo4jManager()
        self.topics = [
            'ragheat.market.overview',
            'ragheat.market.heat', 
            'ragheat.stocks.performance',
            'ragheat.sectors.performance'
        ]
        
    def connect_services(self):
        """Connect to Kafka, Redis, and Neo4j"""
        try:
            # Connect to Kafka
            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=['localhost:9092'],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda m: m.decode('utf-8') if m else None,
                group_id='ragheat_ontology_group',
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            logger.info(f"✅ Connected to Kafka topics: {self.topics}")
            
            # Connect to Redis
            if not self.redis_storage.connect():
                raise Exception("Failed to connect to Redis")
                
            # Connect to Neo4j
            if not self.neo4j_manager.connect():
                raise Exception("Failed to connect to Neo4j")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to services: {e}")
            return False
    
    def process_market_overview(self, data: Dict[str, Any]):
        """Process market overview data and update ontology"""
        try:
            stocks = data.get('stocks', {})
            timestamp = data.get('kafka_timestamp', datetime.utcnow().isoformat())
            
            # Store in Redis with 5-second timeframe
            self.redis_storage.store_market_data('market_overview', data)
            
            # Update Neo4j ontology
            for symbol, stock_data in stocks.items():
                self.neo4j_manager.create_or_update_stock(
                    symbol=symbol,
                    price=stock_data.get('price', 0),
                    change_percent=stock_data.get('change_percent', 0),
                    volume=stock_data.get('volume', 0),
                    sector=stock_data.get('sector', 'Unknown'),
                    market_cap=stock_data.get('market_cap', 0),
                    heat_score=stock_data.get('heat_score', 0),
                    timestamp=timestamp
                )
            
            logger.info(f"📊 Processed market overview: {len(stocks)} stocks updated in ontology")
            
        except Exception as e:
            logger.error(f"❌ Error processing market overview: {e}")
    
    def process_heat_distribution(self, data: Dict[str, Any]):
        """Process heat distribution data and update ontology"""
        try:
            cells = data.get('cells', [])
            market_overview = data.get('market_overview', {})
            timestamp = data.get('kafka_timestamp', datetime.utcnow().isoformat())
            
            # Store in Redis
            self.redis_storage.store_market_data('heat_distribution', data)
            
            # Update market heat metrics in Neo4j
            if market_overview:
                heat_index = market_overview.get('market_heat_index', 0)
                total_volume = market_overview.get('total_volume', 0)
                gainers = market_overview.get('gainers', 0)
                losers = market_overview.get('losers', 0)
                
                # Create or update market heat node
                self.neo4j_manager.execute_query("""
                    MERGE (m:MarketHeat {date: date($timestamp)})
                    SET m.heat_index = $heat_index,
                        m.total_volume = $total_volume,
                        m.gainers = $gainers,
                        m.losers = $losers,
                        m.updated_at = datetime($timestamp)
                """, {
                    'timestamp': timestamp,
                    'heat_index': heat_index,
                    'total_volume': total_volume,
                    'gainers': gainers,
                    'losers': losers
                })
            
            # Process individual heat cells
            for cell in cells:
                symbol = cell.get('symbol')
                if symbol:
                    # Update stock heat relationships
                    self.neo4j_manager.execute_query("""
                        MATCH (s:Stock {symbol: $symbol})
                        MERGE (h:HeatSignal {symbol: $symbol, timestamp: datetime($timestamp)})
                        SET h.value = $value,
                            h.signal = $signal,
                            h.confidence = $confidence,
                            h.sector = $sector
                        MERGE (s)-[:HAS_HEAT_SIGNAL]->(h)
                    """, {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'value': cell.get('value', 0),
                        'signal': cell.get('signal', 'hold'),
                        'confidence': cell.get('confidence', 0),
                        'sector': cell.get('sector', 'Unknown')
                    })
            
            logger.info(f"🔥 Processed heat distribution: {len(cells)} heat signals updated")
            
        except Exception as e:
            logger.error(f"❌ Error processing heat distribution: {e}")
    
    def process_stock_performance(self, data: Dict[str, Any]):
        """Process Neo4j stock performance data"""
        try:
            stocks = data.get('data', [])
            timestamp = data.get('kafka_timestamp', datetime.utcnow().isoformat())
            
            # Store in Redis
            self.redis_storage.store_market_data('stock_performance', data)
            
            # Update stock performance relationships
            for stock in stocks:
                symbol = stock.get('symbol')
                if symbol:
                    # Create performance snapshot
                    self.neo4j_manager.execute_query("""
                        MATCH (s:Stock {symbol: $symbol})
                        MERGE (p:Performance {symbol: $symbol, timestamp: datetime($timestamp)})
                        SET p.price = $price,
                            p.change_percent = $change_percent,
                            p.volume = $volume,
                            p.sector = $sector
                        MERGE (s)-[:HAS_PERFORMANCE]->(p)
                    """, {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'price': stock.get('price', 0),
                        'change_percent': stock.get('change_percent', 0),
                        'volume': stock.get('volume', 0),
                        'sector': stock.get('sector', 'Unknown')
                    })
            
            logger.info(f"📈 Processed stock performance: {len(stocks)} performance records updated")
            
        except Exception as e:
            logger.error(f"❌ Error processing stock performance: {e}")
    
    def process_sector_performance(self, data: Dict[str, Any]):
        """Process sector performance data"""
        try:
            sectors = data.get('data', [])
            timestamp = data.get('kafka_timestamp', datetime.utcnow().isoformat())
            
            # Store in Redis
            self.redis_storage.store_market_data('sector_performance', data)
            
            # Update sector performance in ontology
            for sector_data in sectors:
                sector = sector_data.get('sector')
                if sector:
                    self.neo4j_manager.execute_query("""
                        MERGE (s:Sector {name: $sector})
                        MERGE (p:SectorPerformance {sector: $sector, timestamp: datetime($timestamp)})
                        SET p.avg_change = $avg_change,
                            p.stock_count = $stock_count,
                            p.total_volume = $total_volume
                        MERGE (s)-[:HAS_PERFORMANCE]->(p)
                    """, {
                        'sector': sector,
                        'timestamp': timestamp,
                        'avg_change': sector_data.get('avg_change', 0),
                        'stock_count': sector_data.get('stock_count', 0),
                        'total_volume': sector_data.get('total_volume', 0)
                    })
            
            logger.info(f"🏢 Processed sector performance: {len(sectors)} sectors updated")
            
        except Exception as e:
            logger.error(f"❌ Error processing sector performance: {e}")
    
    def consume_and_process(self):
        """Main consumer loop"""
        if not self.connect_services():
            logger.error("Cannot start consumer without service connections")
            return
            
        logger.info("🚀 Starting Kafka consumer for ontology population")
        
        try:
            for message in self.consumer:
                topic = message.topic
                data = message.value
                key = message.key
                
                logger.info(f"📨 Received message from {topic} (key: {key})")
                
                # Route to appropriate processor based on topic
                if topic == 'ragheat.market.overview':
                    self.process_market_overview(data)
                elif topic == 'ragheat.market.heat':
                    self.process_heat_distribution(data)
                elif topic == 'ragheat.stocks.performance':
                    self.process_stock_performance(data)
                elif topic == 'ragheat.sectors.performance':
                    self.process_sector_performance(data)
                else:
                    logger.warning(f"Unknown topic: {topic}")
                
        except KeyboardInterrupt:
            logger.info("⏹️  Consumer stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in consumer loop: {e}")
        finally:
            self.close()
    
    def close(self):
        """Close all connections"""
        if self.consumer:
            self.consumer.close()
        if self.neo4j_manager:
            self.neo4j_manager.close()
        logger.info("🔌 All connections closed")

if __name__ == "__main__":
    consumer = KafkaOntologyConsumer()
    consumer.consume_and_process()