"""
Real-Time Market Data Kafka Producer
Streams market data from working APIs to Kafka topics every 5 seconds
"""
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List
import requests
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataKafkaProducer:
    def __init__(self):
        self.producer = None
        self.api_base_url = "http://localhost:8000"
        self.topics = {
            'market_overview': 'ragheat.market.overview',
            'heat_distribution': 'ragheat.market.heat',
            'neo4j_stocks': 'ragheat.stocks.performance', 
            'neo4j_sectors': 'ragheat.sectors.performance'
        }
        self.last_fetch_time = {}
        
    def connect_kafka(self):
        """Connect to Kafka broker"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None,
                retries=5,
                retry_backoff_ms=1000
            )
            logger.info("✅ Connected to Kafka broker")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            return False
    
    def fetch_market_overview(self) -> Dict[str, Any]:
        """Fetch real-time market overview data"""
        try:
            response = requests.get(f"{self.api_base_url}/api/live-data/market-overview", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Add timestamp for 5-second timeframe
                data['kafka_timestamp'] = datetime.utcnow().isoformat()
                data['timeframe'] = '5s'
                logger.info(f"📊 Fetched market overview: {len(data.get('stocks', {}))} stocks")
                return data
        except Exception as e:
            logger.error(f"❌ Error fetching market overview: {e}")
        return {}
    
    def fetch_heat_distribution(self) -> Dict[str, Any]:
        """Fetch market heat distribution data"""
        try:
            response = requests.get(f"{self.api_base_url}/api/heat/distribution", timeout=5)
            if response.status_code == 200:
                data = response.json()
                data['kafka_timestamp'] = datetime.utcnow().isoformat()
                data['timeframe'] = '5s'
                logger.info(f"🔥 Fetched heat distribution: {data.get('market_overview', {}).get('market_heat_index', 0):.1f}% heat")
                return data
        except Exception as e:
            logger.error(f"❌ Error fetching heat distribution: {e}")
        return {}
    
    def fetch_neo4j_stocks(self) -> Dict[str, Any]:
        """Fetch Neo4j streaming stock data"""
        try:
            response = requests.get(f"{self.api_base_url}/api/streaming/neo4j/query/top-performers", timeout=5)
            if response.status_code == 200:
                data = response.json()
                data['kafka_timestamp'] = datetime.utcnow().isoformat()
                data['timeframe'] = '5s'
                logger.info(f"📈 Fetched Neo4j stocks: {len(data.get('data', []))} performers")
                return data
        except Exception as e:
            logger.error(f"❌ Error fetching Neo4j stocks: {e}")
        return {}
    
    def fetch_neo4j_sectors(self) -> Dict[str, Any]:
        """Fetch Neo4j sector performance data"""
        try:
            response = requests.get(f"{self.api_base_url}/api/streaming/neo4j/query/sector-performance", timeout=5)
            if response.status_code == 200:
                data = response.json()
                data['kafka_timestamp'] = datetime.utcnow().isoformat()
                data['timeframe'] = '5s'
                logger.info(f"🏢 Fetched Neo4j sectors: {len(data.get('data', []))} sectors")
                return data
        except Exception as e:
            logger.error(f"❌ Error fetching Neo4j sectors: {e}")
        return {}
    
    def send_to_kafka(self, topic: str, key: str, data: Dict[str, Any]):
        """Send data to Kafka topic"""
        if not data:
            return False
            
        try:
            future = self.producer.send(topic, key=key, value=data)
            record_metadata = future.get(timeout=10)
            logger.info(f"✅ Sent to {topic}: {len(str(data))} bytes")
            return True
        except KafkaError as e:
            logger.error(f"❌ Kafka send error for topic {topic}: {e}")
            return False
    
    def stream_market_data(self):
        """Main streaming loop - sends data every 5 seconds"""
        if not self.connect_kafka():
            logger.error("Cannot start streaming without Kafka connection")
            return
            
        logger.info("🚀 Starting real-time market data streaming (5-second intervals)")
        
        while True:
            try:
                current_time = datetime.utcnow()
                timestamp_key = current_time.strftime("%Y%m%d_%H%M%S")
                
                # Fetch and stream market overview
                market_data = self.fetch_market_overview()
                if market_data:
                    self.send_to_kafka(
                        self.topics['market_overview'], 
                        f"market_{timestamp_key}", 
                        market_data
                    )
                
                # Fetch and stream heat distribution
                heat_data = self.fetch_heat_distribution()
                if heat_data:
                    self.send_to_kafka(
                        self.topics['heat_distribution'],
                        f"heat_{timestamp_key}",
                        heat_data
                    )
                
                # Fetch and stream Neo4j stock data
                neo4j_stocks = self.fetch_neo4j_stocks()
                if neo4j_stocks:
                    self.send_to_kafka(
                        self.topics['neo4j_stocks'],
                        f"stocks_{timestamp_key}",
                        neo4j_stocks
                    )
                
                # Fetch and stream Neo4j sector data
                neo4j_sectors = self.fetch_neo4j_sectors()
                if neo4j_sectors:
                    self.send_to_kafka(
                        self.topics['neo4j_sectors'],
                        f"sectors_{timestamp_key}",
                        neo4j_sectors
                    )
                
                logger.info(f"📡 Streaming cycle completed at {current_time}")
                
                # Wait 5 seconds for next cycle
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("⏹️  Streaming stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in streaming loop: {e}")
                time.sleep(5)  # Wait before retry
    
    def close(self):
        """Close Kafka producer"""
        if self.producer:
            self.producer.close()
            logger.info("🔌 Kafka producer closed")

if __name__ == "__main__":
    producer = MarketDataKafkaProducer()
    try:
        producer.stream_market_data()
    finally:
        producer.close()