"""
Revolutionary Kafka Streaming Engine for RAGHeat
Real-time financial data streaming with offline capabilities
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError
import threading
import queue
import sqlite3
import pickle
import time
from pathlib import Path

from models.physics.heat_equation_engine import PhysicsInformedHeatNetwork, create_sample_heat_network

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketEvent:
    """Real-time market event for streaming"""
    symbol: str
    event_type: str  # 'price_update', 'volume_spike', 'news', 'correlation_change'
    timestamp: datetime
    data: Dict[str, Any]
    heat_impact: float
    priority: int  # 1=low, 5=critical

@dataclass
class HeatPulse:
    """Heat pulse propagation event"""
    source_symbol: str
    target_symbols: List[str]
    intensity: float
    propagation_speed: float
    decay_rate: float
    timestamp: datetime

class RevolutionaryKafkaEngine:
    """
    Revolutionary Kafka streaming engine for real-time financial heat analysis
    Features:
    - Real-time market data streaming
    - Offline data storage and replay
    - Physics-based heat propagation
    - Advanced synthetic data generation
    - Fault-tolerant architecture
    """
    
    def __init__(self, 
                 kafka_servers: List[str] = ['localhost:9092'],
                 offline_db_path: str = 'data/ragheat_offline.db'):
        
        self.kafka_servers = kafka_servers
        self.offline_db_path = offline_db_path
        self.is_online = False
        self.offline_mode = False
        
        # Core components
        self.producer: Optional[KafkaProducer] = None
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.heat_network = PhysicsInformedHeatNetwork()
        
        # Topics for different data streams
        self.topics = {
            'market_events': 'ragheat-market-events',
            'heat_pulses': 'ragheat-heat-pulses', 
            'synthetic_data': 'ragheat-synthetic-data',
            'knowledge_graph': 'ragheat-knowledge-graph',
            'ml_predictions': 'ragheat-ml-predictions',
            'system_metrics': 'ragheat-system-metrics'
        }
        
        # Event queues for offline processing
        self.event_queue = queue.Queue(maxsize=10000)
        self.processing_threads = []
        self.running = False
        
        # Offline storage
        self.setup_offline_storage()
        
        # Synthetic data generators
        self.synthetic_generators = {}
        self.setup_synthetic_generators()
        
        logger.info("🚀 Revolutionary Kafka Engine initialized")

    def setup_offline_storage(self) -> None:
        """Setup SQLite database for offline storage"""
        
        Path(self.offline_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.offline_db_path)
        cursor = conn.cursor()
        
        # Create tables for different data types
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                heat_impact REAL,
                priority INTEGER,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS heat_pulses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_symbol TEXT NOT NULL,
                target_symbols TEXT NOT NULL,
                intensity REAL,
                propagation_speed REAL,
                decay_rate REAL,
                timestamp TEXT NOT NULL,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_graph_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                nodes_data TEXT NOT NULL,
                edges_data TEXT NOT NULL,
                physics_properties TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"💾 Offline storage initialized at {self.offline_db_path}")

    def setup_synthetic_generators(self) -> None:
        """Setup sophisticated synthetic data generators"""
        
        # Market regime generator
        self.synthetic_generators['market_regime'] = MarketRegimeGenerator()
        
        # News sentiment generator  
        self.synthetic_generators['news_sentiment'] = NewsSentimentGenerator()
        
        # Volatility clustering generator
        self.synthetic_generators['volatility_clustering'] = VolatilityClusteringGenerator()
        
        # Correlation dynamics generator
        self.synthetic_generators['correlation_dynamics'] = CorrelationDynamicsGenerator()
        
        logger.info("🎭 Synthetic data generators initialized")

    async def initialize_kafka(self) -> bool:
        """Initialize Kafka connection and topics"""
        
        try:
            # Test connection
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.kafka_servers,
                client_id='ragheat_admin'
            )
            
            # Create topics if they don't exist
            topics_to_create = []
            for topic_name, topic_id in self.topics.items():
                topics_to_create.append(
                    NewTopic(
                        name=topic_id,
                        num_partitions=6,  # High throughput
                        replication_factor=1
                    )
                )
            
            try:
                admin_client.create_topics(topics_to_create)
                logger.info("📡 Kafka topics created successfully")
            except TopicAlreadyExistsError:
                logger.info("📡 Kafka topics already exist")
                
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8'),
                key_serializer=str.encode,
                acks='all',  # Ensure data durability
                retries=3,
                batch_size=16384,
                linger_ms=10,  # Low latency
                compression_type='gzip'
            )
            
            self.is_online = True
            logger.info("✅ Kafka connection established")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Kafka connection failed: {e}. Switching to offline mode.")
            self.is_online = False
            self.offline_mode = True
            return False

    def start_streaming(self) -> None:
        """Start the revolutionary streaming engine"""
        
        logger.info("🌊 Starting revolutionary streaming engine")
        self.running = True
        
        # Start heat network
        self.initialize_heat_network()
        
        # Start processing threads
        self.start_processing_threads()
        
        # Start synthetic data generation
        self.start_synthetic_data_generation()
        
        # Start consumers
        self.start_consumers()
        
        logger.info("🚀 Revolutionary streaming engine active!")

    def initialize_heat_network(self) -> None:
        """Initialize the physics-based heat network"""
        
        # Create comprehensive financial heat network
        major_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B',
            'JNJ', 'V', 'WMT', 'JPM', 'MA', 'PG', 'UNH', 'HD', 'CVX', 'ABBV',
            'PFE', 'KO', 'BAC', 'PEP', 'COST', 'TMO', 'AVGO'
        ]
        
        sector_mapping = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'NVDA': 'Technology', 'META': 'Technology',
            'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer', 'COST': 'Consumer', 'WMT': 'Consumer',
            'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare', 'TMO': 'Healthcare',
            'V': 'Financial', 'JPM': 'Financial', 'MA': 'Financial', 'BAC': 'Financial', 'BRK-B': 'Financial',
            'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer', 'CVX': 'Energy', 'AVGO': 'Technology'
        }
        
        # Add nodes with realistic market data
        for symbol in major_symbols:
            market_data = self.generate_realistic_market_data(symbol, sector_mapping.get(symbol, 'Other'))
            self.heat_network.add_financial_node(symbol, market_data)
            
        # Add realistic correlations
        self.add_realistic_correlations(major_symbols)
        
        logger.info(f"🔥 Heat network initialized with {len(major_symbols)} instruments")

    def generate_realistic_market_data(self, symbol: str, sector: str) -> Dict:
        """Generate realistic market data for a symbol"""
        
        # Base parameters by sector
        sector_params = {
            'Technology': {'base_cap': 2e12, 'vol_range': (0.25, 0.4), 'vol_mult': 50e6},
            'Financial': {'base_cap': 400e9, 'vol_range': (0.2, 0.35), 'vol_mult': 30e6},
            'Healthcare': {'base_cap': 300e9, 'vol_range': (0.15, 0.3), 'vol_mult': 20e6},
            'Consumer': {'base_cap': 500e9, 'vol_range': (0.2, 0.35), 'vol_mult': 40e6},
            'Energy': {'base_cap': 200e9, 'vol_range': (0.3, 0.5), 'vol_mult': 25e6},
            'Other': {'base_cap': 100e9, 'vol_range': (0.25, 0.4), 'vol_mult': 15e6}
        }
        
        params = sector_params.get(sector, sector_params['Other'])
        
        # Generate realistic values
        market_cap = params['base_cap'] * np.random.uniform(0.5, 2.0)
        volatility = np.random.uniform(*params['vol_range'])
        volume = int(params['vol_mult'] * np.random.uniform(0.5, 2.0))
        price_change = np.random.normal(0, volatility/252**0.5)  # Daily returns
        sentiment = np.random.uniform(-0.5, 0.5)
        
        return {
            'symbol': symbol,
            'market_cap': market_cap,
            'volatility': volatility,
            'volume': volume,
            'price_change': price_change,
            'sentiment': sentiment,
            'sector': sector,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def add_realistic_correlations(self, symbols: List[str]) -> None:
        """Add realistic correlation network"""
        
        # Sector-based correlation matrix
        sector_correlations = {
            ('Technology', 'Technology'): (0.6, 0.8),
            ('Financial', 'Financial'): (0.7, 0.9),
            ('Healthcare', 'Healthcare'): (0.5, 0.7),
            ('Consumer', 'Consumer'): (0.4, 0.6),
            ('Technology', 'Consumer'): (0.3, 0.5),
            ('Financial', 'Consumer'): (0.2, 0.4),
            ('Healthcare', 'Consumer'): (0.1, 0.3)
        }
        
        # Add connections based on correlations
        for i, symbol1 in enumerate(symbols):
            sector1 = self.get_symbol_sector(symbol1)
            
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                sector2 = self.get_symbol_sector(symbol2)
                
                # Determine correlation range
                key = tuple(sorted([sector1, sector2]))
                if key in sector_correlations:
                    corr_range = sector_correlations[key]
                else:
                    corr_range = (0.0, 0.2)  # Low default correlation
                
                # Generate correlation with some randomness
                correlation = np.random.uniform(*corr_range)
                
                # Add some negative correlations
                if np.random.random() < 0.1:
                    correlation *= -1
                
                # Only add significant correlations
                if abs(correlation) > 0.15:
                    self.heat_network.add_heat_connection(symbol1, symbol2, correlation)

    def get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol (simplified)"""
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO']
        financial_stocks = ['V', 'JPM', 'MA', 'BAC', 'BRK-B']
        healthcare_stocks = ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO']
        consumer_stocks = ['AMZN', 'TSLA', 'HD', 'COST', 'WMT', 'PG', 'KO', 'PEP']
        
        if symbol in tech_stocks:
            return 'Technology'
        elif symbol in financial_stocks:
            return 'Financial'
        elif symbol in healthcare_stocks:
            return 'Healthcare'
        elif symbol in consumer_stocks:
            return 'Consumer'
        else:
            return 'Other'

    def start_processing_threads(self) -> None:
        """Start background processing threads"""
        
        # Heat simulation thread
        heat_thread = threading.Thread(
            target=self.heat_simulation_worker,
            daemon=True
        )
        heat_thread.start()
        self.processing_threads.append(heat_thread)
        
        # Market event processing thread
        event_thread = threading.Thread(
            target=self.event_processing_worker,
            daemon=True
        )
        event_thread.start()
        self.processing_threads.append(event_thread)
        
        # Knowledge graph update thread
        graph_thread = threading.Thread(
            target=self.graph_update_worker,
            daemon=True
        )
        graph_thread.start()
        self.processing_threads.append(graph_thread)
        
        logger.info("🧵 Processing threads started")

    def heat_simulation_worker(self) -> None:
        """Background worker for continuous heat simulation"""
        
        while self.running:
            try:
                # Run heat diffusion simulation
                evolution = self.heat_network.simulate_heat_diffusion(time_steps=50)
                
                # Create heat pulse events
                heat_pulses = self.generate_heat_pulses_from_evolution(evolution)
                
                # Stream heat pulses
                for pulse in heat_pulses:
                    self.stream_heat_pulse(pulse)
                
                # Wait before next simulation
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Heat simulation error: {e}")
                time.sleep(5)

    def event_processing_worker(self) -> None:
        """Background worker for processing market events"""
        
        while self.running:
            try:
                # Get event from queue (with timeout)
                event = self.event_queue.get(timeout=1)
                
                # Process the event
                self.process_market_event(event)
                
                # Mark as done
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Event processing error: {e}")

    def graph_update_worker(self) -> None:
        """Background worker for knowledge graph updates"""
        
        while self.running:
            try:
                # Get current network state
                viz_data = self.heat_network.get_network_visualization_data()
                
                # Stream knowledge graph update
                self.stream_knowledge_graph_update(viz_data)
                
                # Store offline snapshot
                self.store_offline_graph_snapshot(viz_data)
                
                # Wait before next update
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Graph update error: {e}")
                time.sleep(5)

    def start_synthetic_data_generation(self) -> None:
        """Start generating synthetic market data"""
        
        def synthetic_data_generator():
            while self.running:
                try:
                    # Generate market events
                    events = self.generate_synthetic_market_events()
                    
                    for event in events:
                        # Queue for processing
                        self.event_queue.put(event)
                        
                        # Stream immediately if online
                        self.stream_market_event(event)
                    
                    # Generate at realistic intervals
                    time.sleep(np.random.uniform(0.1, 0.5))
                    
                except Exception as e:
                    logger.error(f"❌ Synthetic data generation error: {e}")
                    time.sleep(1)
        
        # Start synthetic data thread
        synthetic_thread = threading.Thread(target=synthetic_data_generator, daemon=True)
        synthetic_thread.start()
        self.processing_threads.append(synthetic_thread)
        
        logger.info("🎭 Synthetic data generation started")

    def generate_synthetic_market_events(self) -> List[MarketEvent]:
        """Generate realistic synthetic market events"""
        
        events = []
        
        # Get random symbols
        symbols = list(self.heat_network.nodes.keys())
        if not symbols:
            return events
            
        # Generate different types of events
        event_types = [
            ('price_update', 0.7),
            ('volume_spike', 0.15), 
            ('news', 0.1),
            ('correlation_change', 0.05)
        ]
        
        # Generate 1-5 events
        num_events = np.random.poisson(2) + 1
        
        for _ in range(num_events):
            # Select event type
            event_type = np.random.choice(
                [et[0] for et in event_types],
                p=[et[1] for et in event_types]
            )
            
            # Select random symbol
            symbol = np.random.choice(symbols)
            
            # Generate event data
            event_data = self.generate_event_data(symbol, event_type)
            
            # Calculate heat impact
            heat_impact = self.calculate_heat_impact(event_type, event_data)
            
            # Determine priority
            priority = self.determine_event_priority(event_type, heat_impact)
            
            event = MarketEvent(
                symbol=symbol,
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                data=event_data,
                heat_impact=heat_impact,
                priority=priority
            )
            
            events.append(event)
            
        return events

    def generate_event_data(self, symbol: str, event_type: str) -> Dict:
        """Generate realistic event data"""
        
        if event_type == 'price_update':
            return {
                'price_change_pct': np.random.normal(0, 0.02),
                'volume': int(np.random.lognormal(15, 1)),
                'bid_ask_spread': np.random.uniform(0.001, 0.01)
            }
            
        elif event_type == 'volume_spike':
            return {
                'volume_multiplier': np.random.uniform(3, 10),
                'duration_minutes': np.random.randint(5, 60),
                'trigger': np.random.choice(['news', 'algorithmic', 'institutional'])
            }
            
        elif event_type == 'news':
            return {
                'sentiment_score': np.random.uniform(-1, 1),
                'relevance_score': np.random.uniform(0.5, 1.0),
                'source': np.random.choice(['bloomberg', 'reuters', 'cnbc', 'wsj']),
                'category': np.random.choice(['earnings', 'regulation', 'partnership', 'analyst'])
            }
            
        elif event_type == 'correlation_change':
            other_symbols = [s for s in self.heat_network.nodes.keys() if s != symbol]
            if other_symbols:
                target = np.random.choice(other_symbols)
                return {
                    'target_symbol': target,
                    'new_correlation': np.random.uniform(-0.8, 0.8),
                    'correlation_change': np.random.uniform(-0.3, 0.3)
                }
            else:
                return {}
                
        return {}

    def calculate_heat_impact(self, event_type: str, event_data: Dict) -> float:
        """Calculate heat impact of an event"""
        
        if event_type == 'price_update':
            price_change = abs(event_data.get('price_change_pct', 0))
            return min(1.0, price_change / 0.05)  # 5% change = max impact
            
        elif event_type == 'volume_spike':
            multiplier = event_data.get('volume_multiplier', 1)
            return min(1.0, (multiplier - 1) / 9)  # 10x volume = max impact
            
        elif event_type == 'news':
            sentiment = abs(event_data.get('sentiment_score', 0))
            relevance = event_data.get('relevance_score', 0.5)
            return sentiment * relevance
            
        elif event_type == 'correlation_change':
            change = abs(event_data.get('correlation_change', 0))
            return change / 0.3  # 30% correlation change = max impact
            
        return 0.1  # Default low impact

    def determine_event_priority(self, event_type: str, heat_impact: float) -> int:
        """Determine event priority (1=low, 5=critical)"""
        
        if heat_impact > 0.8:
            return 5  # Critical
        elif heat_impact > 0.6:
            return 4  # High
        elif heat_impact > 0.4:
            return 3  # Medium
        elif heat_impact > 0.2:
            return 2  # Low
        else:
            return 1  # Very low

    def stream_market_event(self, event: MarketEvent) -> None:
        """Stream market event to Kafka or store offline"""
        
        event_dict = asdict(event)
        
        if self.is_online and self.producer:
            try:
                self.producer.send(
                    self.topics['market_events'],
                    key=event.symbol,
                    value=event_dict
                )
                logger.debug(f"📡 Streamed event: {event.symbol} {event.event_type}")
                
            except Exception as e:
                logger.error(f"❌ Failed to stream event: {e}")
                self.store_offline_event(event)
        else:
            self.store_offline_event(event)

    def store_offline_event(self, event: MarketEvent) -> None:
        """Store event in offline database"""
        
        conn = sqlite3.connect(self.offline_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO market_events 
            (symbol, event_type, timestamp, data, heat_impact, priority)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            event.symbol,
            event.event_type,
            event.timestamp.isoformat(),
            json.dumps(event.data),
            event.heat_impact,
            event.priority
        ))
        
        conn.commit()
        conn.close()

    def process_market_event(self, event: MarketEvent) -> None:
        """Process market event and update heat network"""
        
        # Update node temperature based on event
        if event.symbol in self.heat_network.nodes:
            node = self.heat_network.nodes[event.symbol]
            
            # Apply heat impact
            temperature_increase = event.heat_impact * 0.1  # Scale appropriately
            new_temp = min(1.0, node.temperature + temperature_increase)
            node.temperature = new_temp
            
            logger.debug(f"🌡️ Updated {event.symbol} temperature: {new_temp:.3f}")

    def generate_heat_pulses_from_evolution(self, evolution: Dict) -> List[HeatPulse]:
        """Generate heat pulses from simulation evolution"""
        
        pulses = []
        
        if 'heat_flows' not in evolution or not evolution['heat_flows']:
            return pulses
            
        # Get latest heat flows
        latest_flows = evolution['heat_flows'][-1]
        
        for edge_id, flow_rate in latest_flows.items():
            if abs(flow_rate) > 0.1:  # Significant flow
                
                # Parse edge ID
                parts = edge_id.split('_')
                if len(parts) >= 2:
                    source = parts[0]
                    target = parts[1]
                    
                    pulse = HeatPulse(
                        source_symbol=source,
                        target_symbols=[target],
                        intensity=abs(flow_rate),
                        propagation_speed=np.random.uniform(0.5, 2.0),
                        decay_rate=np.random.uniform(0.1, 0.5),
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    pulses.append(pulse)
                    
        return pulses

    def stream_heat_pulse(self, pulse: HeatPulse) -> None:
        """Stream heat pulse event"""
        
        pulse_dict = asdict(pulse)
        
        if self.is_online and self.producer:
            try:
                self.producer.send(
                    self.topics['heat_pulses'],
                    key=pulse.source_symbol,
                    value=pulse_dict
                )
                
            except Exception as e:
                logger.error(f"❌ Failed to stream heat pulse: {e}")

    def stream_knowledge_graph_update(self, viz_data: Dict) -> None:
        """Stream knowledge graph update"""
        
        if self.is_online and self.producer:
            try:
                self.producer.send(
                    self.topics['knowledge_graph'],
                    key='graph_update',
                    value=viz_data
                )
                
            except Exception as e:
                logger.error(f"❌ Failed to stream graph update: {e}")

    def store_offline_graph_snapshot(self, viz_data: Dict) -> None:
        """Store graph snapshot offline"""
        
        conn = sqlite3.connect(self.offline_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO knowledge_graph_snapshots 
            (timestamp, nodes_data, edges_data, physics_properties)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(viz_data['nodes']),
            json.dumps(viz_data['edges']),
            json.dumps(viz_data['physics_properties'])
        ))
        
        conn.commit()
        conn.close()

    def start_consumers(self) -> None:
        """Start Kafka consumers for different topics"""
        
        if not self.is_online:
            return
            
        # Consumer configurations
        consumer_configs = {
            'bootstrap_servers': self.kafka_servers,
            'auto_offset_reset': 'latest',
            'enable_auto_commit': True,
            'group_id': 'ragheat_consumers',
            'value_deserializer': lambda x: json.loads(x.decode('utf-8'))
        }
        
        # Start consumers for each topic
        for topic_name, topic_id in self.topics.items():
            if topic_name in ['market_events', 'heat_pulses']:  # Only consume select topics
                consumer = KafkaConsumer(topic_id, **consumer_configs)
                self.consumers[topic_name] = consumer
                
                # Start consumer thread
                consumer_thread = threading.Thread(
                    target=self.consumer_worker,
                    args=(topic_name, consumer),
                    daemon=True
                )
                consumer_thread.start()
                self.processing_threads.append(consumer_thread)
                
        logger.info("👂 Kafka consumers started")

    def consumer_worker(self, topic_name: str, consumer: KafkaConsumer) -> None:
        """Worker for consuming Kafka messages"""
        
        while self.running:
            try:
                for message in consumer:
                    if not self.running:
                        break
                        
                    # Process consumed message
                    self.handle_consumed_message(topic_name, message.value)
                    
            except Exception as e:
                logger.error(f"❌ Consumer error for {topic_name}: {e}")
                time.sleep(5)

    def handle_consumed_message(self, topic_name: str, message_data: Dict) -> None:
        """Handle consumed message from Kafka"""
        
        logger.debug(f"📥 Consumed message from {topic_name}")
        
        # Process based on topic type
        if topic_name == 'market_events':
            # Handle market event
            pass
        elif topic_name == 'heat_pulses':
            # Handle heat pulse
            pass

    def get_latest_knowledge_graph(self) -> Dict:
        """Get latest knowledge graph data"""
        
        return self.heat_network.get_network_visualization_data()

    def get_offline_data(self, hours_back: int = 24) -> Dict:
        """Get offline data for the last N hours"""
        
        conn = sqlite3.connect(self.offline_db_path)
        
        # Get events from last N hours
        cutoff_time = datetime.now(timezone.utc) - pd.Timedelta(hours=hours_back)
        
        events_df = pd.read_sql_query('''
            SELECT * FROM market_events 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', conn, params=(cutoff_time.isoformat(),))
        
        # Get latest graph snapshot
        graph_df = pd.read_sql_query('''
            SELECT * FROM knowledge_graph_snapshots 
            ORDER BY timestamp DESC LIMIT 1
        ''', conn)
        
        conn.close()
        
        return {
            'events': events_df.to_dict('records'),
            'latest_graph': graph_df.to_dict('records')[0] if not graph_df.empty else None
        }

    def stop(self) -> None:
        """Stop the streaming engine"""
        
        logger.info("🛑 Stopping revolutionary streaming engine")
        self.running = False
        
        # Close producer
        if self.producer:
            self.producer.flush()
            self.producer.close()
            
        # Close consumers
        for consumer in self.consumers.values():
            consumer.close()
            
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=5)
            
        logger.info("✅ Revolutionary streaming engine stopped")

# Synthetic data generator classes
class MarketRegimeGenerator:
    """Generate different market regime scenarios"""
    
    def generate_regime_shift(self) -> Dict:
        regimes = ['bull_market', 'bear_market', 'high_volatility', 'low_volatility', 'crisis']
        return {
            'regime': np.random.choice(regimes),
            'intensity': np.random.uniform(0.5, 1.0),
            'duration': np.random.randint(30, 300)  # minutes
        }

class NewsSentimentGenerator:
    """Generate realistic news sentiment events"""
    
    def generate_news_event(self) -> Dict:
        categories = ['earnings', 'merger', 'regulatory', 'economic', 'geopolitical']
        return {
            'category': np.random.choice(categories),
            'sentiment': np.random.normal(0, 0.5),
            'impact_radius': np.random.randint(1, 5)  # How many stocks affected
        }

class VolatilityClusteringGenerator:
    """Generate volatility clustering patterns"""
    
    def __init__(self):
        self.current_vol_regime = 'normal'
        self.regime_duration = 0
        
    def generate_volatility_update(self) -> Dict:
        # Implement GARCH-like volatility clustering
        if self.regime_duration <= 0:
            self.current_vol_regime = np.random.choice(['low', 'normal', 'high'], p=[0.3, 0.5, 0.2])
            self.regime_duration = np.random.randint(50, 200)
            
        self.regime_duration -= 1
        
        vol_multipliers = {'low': 0.5, 'normal': 1.0, 'high': 2.0}
        
        return {
            'regime': self.current_vol_regime,
            'multiplier': vol_multipliers[self.current_vol_regime],
            'persistence': np.random.uniform(0.7, 0.95)
        }

class CorrelationDynamicsGenerator:
    """Generate dynamic correlation changes"""
    
    def generate_correlation_shift(self) -> Dict:
        return {
            'correlation_change': np.random.uniform(-0.3, 0.3),
            'affected_pairs': np.random.randint(3, 10),
            'persistence': np.random.uniform(0.1, 0.8)
        }

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize the revolutionary engine
        engine = RevolutionaryKafkaEngine()
        
        # Try to initialize Kafka (will fallback to offline if failed)
        await engine.initialize_kafka()
        
        # Start streaming
        engine.start_streaming()
        
        # Let it run for demonstration
        logger.info("🚀 Revolutionary engine running. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            engine.stop()
    
    # Run the example
    asyncio.run(main())