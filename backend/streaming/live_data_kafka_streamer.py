"""
Live Data Kafka Streamer
Streams synthetic and real market data through Kafka for real-time processing
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time
import random
import numpy as np
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import threading
from dataclasses import asdict

from services.synthetic_data_generator import SyntheticDataGenerator

class LiveDataKafkaStreamer:
    def __init__(self, 
                 kafka_servers: List[str] = ['localhost:9092'],
                 topics: Dict[str, str] = None):
        
        self.kafka_servers = kafka_servers
        self.topics = topics or {
            'market_data': 'ragheat-market-data',
            'heat_signals': 'ragheat-heat-signals',
            'correlations': 'ragheat-correlations',
            'regime_changes': 'ragheat-regime-changes',
            'analytics': 'ragheat-analytics'
        }
        
        # Initialize components
        self.data_generator = SyntheticDataGenerator()
        self.producer = None
        self.consumers = {}
        
        # Streaming state
        self.is_streaming = False
        self.stream_interval = 0.1  # 100ms between ticks (fast simulation)
        self.batch_size = 20  # Number of stocks per batch
        
        # Data directories
        self.data_dir = Path("data/synthetic_stocks")
        self.processed_dir = Path("data/processed")
        
        # Analytics state
        self.analytics_buffer = []
        self.last_analytics_time = time.time()
        
        # Performance monitoring
        self.message_count = 0
        self.start_time = time.time()
        self.errors = []
        
    def initialize_kafka_producer(self) -> bool:
        """Initialize Kafka producer with error handling"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas
                retries=3,
                batch_size=16384,
                linger_ms=10,  # Small delay to batch messages
                buffer_memory=33554432,
                compression_type='gzip'
            )
            logging.info("Kafka producer initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize Kafka producer: {e}")
            self.errors.append(f"Producer init error: {e}")
            return False
    
    def initialize_kafka_consumer(self, topic: str, group_id: str) -> Optional[KafkaConsumer]:
        """Initialize Kafka consumer for a specific topic"""
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.kafka_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            logging.info(f"Kafka consumer initialized for topic: {topic}")
            return consumer
            
        except Exception as e:
            logging.error(f"Failed to initialize Kafka consumer for {topic}: {e}")
            return None
    
    async def stream_market_data(self):
        """Main streaming loop for market data"""
        logging.info("Starting market data streaming...")
        
        while self.is_streaming:
            try:
                # Generate batch of market data
                batch_data = self.data_generator.generate_batch_data(1)
                
                for market_tick in batch_data:
                    # Send to main market data topic
                    await self.send_to_kafka(
                        self.topics['market_data'],
                        key=f"market_tick_{int(time.time())}",
                        value=market_tick
                    )
                    
                    # Process individual stock data
                    for symbol, stock_data in market_tick['stocks'].items():
                        
                        # Send heat signals
                        heat_signal = {
                            'symbol': symbol,
                            'timestamp': stock_data['timestamp'],
                            'heat_score': stock_data['heat_score'],
                            'volatility': stock_data['volatility'],
                            'regime': market_tick['market_regime']
                        }
                        
                        await self.send_to_kafka(
                            self.topics['heat_signals'],
                            key=symbol,
                            value=heat_signal
                        )
                        
                        # Send correlation data
                        correlation_data = {
                            'symbol': symbol,
                            'timestamp': stock_data['timestamp'],
                            'correlations': stock_data['correlation_signals'],
                            'sector': stock_data['sector']
                        }
                        
                        await self.send_to_kafka(
                            self.topics['correlations'],
                            key=f"{symbol}_correlations",
                            value=correlation_data
                        )
                    
                    # Check for regime changes
                    if hasattr(self.data_generator, '_last_regime'):
                        if self.data_generator._last_regime != market_tick['market_regime']:
                            regime_change = {
                                'timestamp': market_tick['timestamp'],
                                'old_regime': self.data_generator._last_regime,
                                'new_regime': market_tick['market_regime'],
                                'impact_analysis': self.analyze_regime_impact(market_tick)
                            }
                            
                            await self.send_to_kafka(
                                self.topics['regime_changes'],
                                key="regime_change",
                                value=regime_change
                            )
                    
                    self.data_generator._last_regime = market_tick['market_regime']
                    
                    # Buffer for analytics
                    self.analytics_buffer.append(market_tick)
                    
                    self.message_count += 1
                
                # Send analytics periodically
                if time.time() - self.last_analytics_time > 5.0:  # Every 5 seconds
                    await self.send_analytics()
                    self.last_analytics_time = time.time()
                
                # Realistic streaming delay
                await asyncio.sleep(self.stream_interval)
                
            except Exception as e:
                logging.error(f"Error in streaming loop: {e}")
                self.errors.append(f"Streaming error: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error
    
    async def send_to_kafka(self, topic: str, key: str, value: dict):
        """Send data to Kafka topic with error handling"""
        try:
            if self.producer:
                future = self.producer.send(topic, key=key, value=value)
                # Don't wait for each message to improve throughput
                # result = future.get(timeout=1)
            else:
                logging.warning("Producer not initialized, simulating send")
                # Simulate successful send for testing
                
        except KafkaError as e:
            logging.error(f"Kafka send error: {e}")
            self.errors.append(f"Kafka error: {e}")
        except Exception as e:
            logging.error(f"Unexpected send error: {e}")
            self.errors.append(f"Send error: {e}")
    
    def analyze_regime_impact(self, market_tick: Dict) -> Dict:
        """Analyze the impact of market regime changes"""
        stocks = market_tick['stocks']
        regime = market_tick['market_regime']
        
        # Calculate aggregate metrics
        total_volume = sum(stock['volume'] for stock in stocks.values())
        avg_volatility = np.mean([stock['volatility'] for stock in stocks.values()])
        avg_heat = np.mean([stock['heat_score'] for stock in stocks.values()])
        
        # Sector analysis
        sector_performance = {}
        for stock_data in stocks.values():
            sector = stock_data['sector']
            if sector not in sector_performance:
                sector_performance[sector] = {
                    'count': 0,
                    'total_change': 0.0,
                    'total_heat': 0.0
                }
            
            sector_performance[sector]['count'] += 1
            sector_performance[sector]['total_change'] += stock_data['change_percent']
            sector_performance[sector]['total_heat'] += stock_data['heat_score']
        
        # Calculate sector averages
        for sector in sector_performance:
            count = sector_performance[sector]['count']
            sector_performance[sector]['avg_change'] = sector_performance[sector]['total_change'] / count
            sector_performance[sector]['avg_heat'] = sector_performance[sector]['total_heat'] / count
        
        return {
            'regime': regime,
            'market_metrics': {
                'total_volume': total_volume,
                'avg_volatility': round(avg_volatility, 3),
                'avg_heat': round(avg_heat, 2),
                'active_stocks': len(stocks)
            },
            'sector_performance': sector_performance,
            'timestamp': market_tick['timestamp']
        }
    
    async def send_analytics(self):
        """Process and send analytics data"""
        if not self.analytics_buffer:
            return
        
        try:
            # Calculate analytics from buffer
            analytics = self.calculate_streaming_analytics()
            
            await self.send_to_kafka(
                self.topics['analytics'],
                key=f"analytics_{int(time.time())}",
                value=analytics
            )
            
            # Clear buffer
            self.analytics_buffer = []
            
        except Exception as e:
            logging.error(f"Analytics processing error: {e}")
    
    def calculate_streaming_analytics(self) -> Dict:
        """Calculate comprehensive analytics from buffered data"""
        if not self.analytics_buffer:
            return {}
        
        # Aggregate data from buffer
        all_stocks = []
        regimes = []
        
        for tick in self.analytics_buffer:
            regimes.append(tick['market_regime'])
            for stock_data in tick['stocks'].values():
                all_stocks.append(stock_data)
        
        # Market-wide metrics
        prices = [stock['price'] for stock in all_stocks]
        volumes = [stock['volume'] for stock in all_stocks]
        heats = [stock['heat_score'] for stock in all_stocks]
        volatilities = [stock['volatility'] for stock in all_stocks]
        changes = [stock['change_percent'] for stock in all_stocks]
        
        # Statistical analysis
        analytics = {
            'timestamp': datetime.utcnow().isoformat(),
            'buffer_size': len(self.analytics_buffer),
            'total_stocks_processed': len(all_stocks),
            
            'market_summary': {
                'avg_price': round(np.mean(prices), 2),
                'total_volume': sum(volumes),
                'avg_heat': round(np.mean(heats), 2),
                'max_heat': round(max(heats), 2),
                'avg_volatility': round(np.mean(volatilities), 3),
                'market_change': round(np.mean(changes), 3)
            },
            
            'regime_distribution': {
                regime: regimes.count(regime) / len(regimes) 
                for regime in set(regimes)
            },
            
            'heat_distribution': {
                'cold': sum(1 for h in heats if h < 30) / len(heats),
                'warm': sum(1 for h in heats if 30 <= h < 70) / len(heats),
                'hot': sum(1 for h in heats if h >= 70) / len(heats)
            },
            
            'performance_metrics': {
                'messages_per_second': round(self.message_count / (time.time() - self.start_time), 2),
                'total_messages': self.message_count,
                'uptime_seconds': round(time.time() - self.start_time, 1),
                'error_count': len(self.errors)
            }
        }
        
        return analytics
    
    async def start_streaming(self):
        """Start the streaming service"""
        logging.info("Starting Live Data Kafka Streamer...")
        
        # Initialize Kafka producer
        if not self.initialize_kafka_producer():
            logging.error("Failed to initialize Kafka producer")
            return False
        
        # Set streaming flag
        self.is_streaming = True
        self.start_time = time.time()
        
        # Start streaming task
        streaming_task = asyncio.create_task(self.stream_market_data())
        
        try:
            await streaming_task
        except KeyboardInterrupt:
            logging.info("Streaming stopped by user")
        except Exception as e:
            logging.error(f"Streaming error: {e}")
        finally:
            await self.stop_streaming()
        
        return True
    
    async def stop_streaming(self):
        """Stop the streaming service"""
        logging.info("Stopping streaming service...")
        self.is_streaming = False
        
        if self.producer:
            self.producer.flush()
            self.producer.close()
            
        for consumer in self.consumers.values():
            if consumer:
                consumer.close()
        
        # Final analytics
        if self.analytics_buffer:
            await self.send_analytics()
        
        logging.info(f"Streaming stopped. Processed {self.message_count} messages")
    
    def get_status(self) -> Dict:
        """Get current streaming status"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'is_streaming': self.is_streaming,
            'uptime_seconds': round(uptime, 1),
            'messages_processed': self.message_count,
            'messages_per_second': round(self.message_count / uptime if uptime > 0 else 0, 2),
            'current_regime': getattr(self.data_generator, 'current_regime', 'unknown'),
            'buffer_size': len(self.analytics_buffer),
            'error_count': len(self.errors),
            'recent_errors': self.errors[-5:] if self.errors else [],
            'topics': list(self.topics.values())
        }

# Consumer service for processing streamed data
class KafkaDataConsumer:
    def __init__(self, kafka_servers: List[str] = ['localhost:9092']):
        self.kafka_servers = kafka_servers
        self.consumers = {}
        self.is_consuming = False
        self.message_handlers = {}
        
    def add_message_handler(self, topic: str, handler_func):
        """Add a message handler for a specific topic"""
        self.message_handlers[topic] = handler_func
    
    async def start_consuming(self, topics: List[str]):
        """Start consuming messages from specified topics"""
        self.is_consuming = True
        
        for topic in topics:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.kafka_servers,
                group_id=f'ragheat_consumer_{topic}',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )
            
            self.consumers[topic] = consumer
            
            # Start consuming task for each topic
            asyncio.create_task(self.consume_topic(topic, consumer))
        
        logging.info(f"Started consuming from topics: {topics}")
    
    async def consume_topic(self, topic: str, consumer: KafkaConsumer):
        """Consume messages from a specific topic"""
        while self.is_consuming:
            try:
                message_batch = consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        # Process message
                        if topic in self.message_handlers:
                            handler = self.message_handlers[topic]
                            await handler(message.key, message.value)
                        else:
                            # Default processing
                            logging.info(f"Received message from {topic}: {message.key}")
                
                await asyncio.sleep(0.01)  # Small delay
                
            except Exception as e:
                logging.error(f"Error consuming from {topic}: {e}")
                await asyncio.sleep(1.0)

# Main execution
async def main():
    """Main function to run the streaming service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create streamer instance
    streamer = LiveDataKafkaStreamer()
    
    try:
        # Start streaming
        await streamer.start_streaming()
        
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await streamer.stop_streaming()

if __name__ == "__main__":
    asyncio.run(main())