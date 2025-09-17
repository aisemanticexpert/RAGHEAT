"""
Advanced Neo4j Manager with Dynamic Heat Visualization and Temporal State Management
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from neo4j import GraphDatabase
import numpy as np

# Import our custom modules
from indicators.technical_indicators import (
    TechnicalIndicatorEngine, TechnicalSignal, MACDResult, 
    BollingerBandsResult, HeatEquationResult
)
from temporal.graph_state_manager import (
    TemporalGraphStateManager, GraphSnapshot, TemporalPoint
)
from signals.buy_sell_engine import (
    BuySellSignalEngine, TradingSignal, SignalStrength
)

logger = logging.getLogger(__name__)

class AdvancedNeo4jManager:
    """
    Advanced Neo4j manager with dynamic heat visualization,
    temporal state management, and technical indicator integration
    """
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", 
                 password: str = "password"):
        
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
        # Initialize subsystems
        self.technical_engine = TechnicalIndicatorEngine()
        self.temporal_manager = TemporalGraphStateManager()
        self.signal_engine = BuySellSignalEngine()
        
        # Performance tracking
        self.last_full_update = datetime.now()
        self.incremental_update_count = 0
        
        self._connect()
        
    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def update_stock_with_advanced_analytics(self, 
                                           symbol: str,
                                           price: float,
                                           volume: int,
                                           volatility: float,
                                           market_sentiment: float = 0.0) -> Dict[str, Any]:
        """
        Update stock with comprehensive analytics including technical indicators,
        heat calculations, and temporal state management
        """
        
        current_time = datetime.now()
        
        # Add price data to technical engine
        self.technical_engine.add_price_data(symbol, price, volume, current_time)
        
        # Calculate technical indicators
        macd_result = self.technical_engine.calculate_macd(symbol)
        bb_result = self.technical_engine.calculate_bollinger_bands(symbol)
        
        # Calculate advanced heat equation
        heat_result = self.technical_engine.calculate_heat_equation(
            symbol, price, volume, volatility, market_sentiment
        )
        
        # Calculate dynamic weights
        weights = self.technical_engine.calculate_dynamic_weight(
            symbol, macd_result, bb_result, heat_result,
            additional_factors={'market_correlation': market_sentiment}
        )
        
        # Get color mapping for visualization
        color_map = self.technical_engine.get_node_color_heat_map(heat_result.current_heat)
        
        # Calculate Buy/Sell Trading Signal
        trading_signal = self.signal_engine.calculate_trading_signal(
            symbol=symbol,
            current_price=price,
            heat_score=heat_result.current_heat,
            macd_result=self._serialize_macd_result(macd_result) if macd_result else {},
            bollinger_result=self._serialize_bb_result(bb_result) if bb_result else {},
            price_change_pct=weights.get('technical_momentum', 0.0) * 100,  # Convert to percentage
            volume_ratio=weights.get('volume_confirmation', 1.0),
            volatility=volatility
        )
        
        # Update color mapping based on trading signal
        color_map = self._enhance_color_with_trading_signal(color_map, trading_signal)
        
        # Update temporal state
        self.temporal_manager.update_node_evolution(
            f"STOCK_{symbol}",
            heat_result.current_heat,
            weights,
            activity_burst=weights.get('volume_confirmation', 0.0)
        )
        
        # Prepare comprehensive update data
        update_data = {
            # Basic data
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'volatility': volatility,
            'timestamp': current_time.isoformat(),
            
            # Technical indicators
            'macd': self._serialize_macd_result(macd_result) if macd_result else None,
            'bollinger_bands': self._serialize_bb_result(bb_result) if bb_result else None,
            
            # Heat analysis
            'heat_score': heat_result.current_heat,
            'heat_velocity': heat_result.heat_velocity,
            'heat_acceleration': heat_result.heat_acceleration,
            'propagation_strength': heat_result.propagation_strength,
            'temporal_validity': heat_result.temporal_validity,
            
            # Dynamic weights
            'weights': weights,
            
            # Visualization
            'color_mapping': color_map,
            
            # Trading Signal
            'trading_signal': trading_signal.to_dict(),
            
            # Signals
            'signals': []
        }
        
        # Add technical signals
        if macd_result and macd_result.crossover_signal:
            update_data['signals'].append(self._serialize_signal(macd_result.crossover_signal))
        
        if bb_result:
            if bb_result.squeeze_signal:
                update_data['signals'].append(self._serialize_signal(bb_result.squeeze_signal))
            if bb_result.breakout_signal:
                update_data['signals'].append(self._serialize_signal(bb_result.breakout_signal))
        
        # Update in Neo4j
        self._update_stock_in_neo4j(f"STOCK_{symbol}", update_data)
        
        # Calculate heat propagation to neighboring nodes
        propagated_heat = self.temporal_manager.calculate_heat_propagation(
            f"STOCK_{symbol}", 
            heat_result.current_heat
        )
        
        # Update propagated heat in related nodes
        self._update_heat_propagation(propagated_heat, f"STOCK_{symbol}")
        
        return update_data
    
    def _update_stock_in_neo4j(self, stock_id: str, update_data: Dict[str, Any]):
        """Update stock node in Neo4j with comprehensive data"""
        
        with self.driver.session() as session:
            # Main stock update
            session.run("""
                MATCH (s:Stock {id: $stock_id})
                SET s.current_price = $price,
                    s.volume = $volume,
                    s.volatility = $volatility,
                    s.heat_score = $heat_score,
                    s.heat_velocity = $heat_velocity,
                    s.heat_acceleration = $heat_acceleration,
                    s.propagation_strength = $propagation_strength,
                    s.temporal_validity = $temporal_validity,
                    s.last_updated = $timestamp,
                    s.composite_weight = $composite_weight,
                    s.technical_momentum = $technical_momentum,
                    s.volatility_state = $volatility_state,
                    s.heat_intensity = $heat_intensity,
                    s.volume_confirmation = $volume_confirmation,
                    s.time_decay = $time_decay,
                    s.market_correlation = $market_correlation,
                    s.background_color = $bg_color,
                    s.border_color = $border_color,
                    s.font_color = $font_color,
                    s.heat_level = $heat_level,
                    s.trading_signal = $trading_signal,
                    s.signal_confidence = $signal_confidence,
                    s.price_target = $price_target,
                    s.stop_loss = $stop_loss,
                    s.signal_reasoning = $signal_reasoning
            """,
            stock_id=stock_id,
            price=update_data['price'],
            volume=update_data['volume'],
            volatility=update_data['volatility'],
            heat_score=update_data['heat_score'],
            heat_velocity=update_data['heat_velocity'],
            heat_acceleration=update_data['heat_acceleration'],
            propagation_strength=update_data['propagation_strength'],
            temporal_validity=update_data['temporal_validity'],
            timestamp=update_data['timestamp'],
            composite_weight=update_data['weights']['composite_weight'],
            technical_momentum=update_data['weights']['technical_momentum'],
            volatility_state=update_data['weights']['volatility_state'],
            heat_intensity=update_data['weights']['heat_intensity'],
            volume_confirmation=update_data['weights']['volume_confirmation'],
            time_decay=update_data['weights']['time_decay'],
            market_correlation=update_data['weights']['market_correlation'],
            bg_color=update_data['color_mapping']['background_color'],
            border_color=update_data['color_mapping']['border_color'],
            font_color=update_data['color_mapping']['font_color'],
            heat_level=update_data['color_mapping']['heat_level'],
            trading_signal=update_data['trading_signal']['signal'],
            signal_confidence=update_data['trading_signal']['confidence'],
            price_target=update_data['trading_signal']['price_target'],
            stop_loss=update_data['trading_signal']['stop_loss'],
            signal_reasoning=str(update_data['trading_signal']['reasoning'])
            )
            
            # Store technical indicators if available
            if update_data['macd']:
                macd = update_data['macd']
                session.run("""
                    MATCH (s:Stock {id: $stock_id})
                    SET s.macd_line = $macd_line,
                        s.macd_signal = $signal_line,
                        s.macd_histogram = $histogram,
                        s.macd_trend = $trend_direction,
                        s.macd_strength = $signal_strength
                """,
                stock_id=stock_id,
                macd_line=macd['macd_line'],
                signal_line=macd['signal_line'],
                histogram=macd['histogram'],
                trend_direction=macd['trend_direction'],
                signal_strength=macd['signal_strength']
                )
            
            if update_data['bollinger_bands']:
                bb = update_data['bollinger_bands']
                session.run("""
                    MATCH (s:Stock {id: $stock_id})
                    SET s.bb_upper = $upper_band,
                        s.bb_middle = $middle_band,
                        s.bb_lower = $lower_band,
                        s.bb_bandwidth = $bandwidth,
                        s.bb_percent = $bb_percent
                """,
                stock_id=stock_id,
                upper_band=bb['upper_band'],
                middle_band=bb['middle_band'],
                lower_band=bb['lower_band'],
                bandwidth=bb['bandwidth'],
                bb_percent=bb['bb_percent']
                )
            
            # Store signals as separate nodes with relationships
            for signal in update_data['signals']:
                signal_id = f"SIGNAL_{stock_id}_{datetime.now().timestamp()}"
                session.run("""
                    CREATE (sig:Signal {
                        id: $signal_id,
                        type: $signal_type,
                        direction: $direction,
                        strength: $strength,
                        confidence: $confidence,
                        timestamp: $timestamp
                    })
                    WITH sig
                    MATCH (s:Stock {id: $stock_id})
                    CREATE (s)-[:HAS_SIGNAL]->(sig)
                """,
                signal_id=signal_id,
                stock_id=stock_id,
                signal_type=signal['signal_type'],
                direction=signal['direction'],
                strength=signal['strength'],
                confidence=signal['confidence'],
                timestamp=signal['timestamp']
                )
    
    def _update_heat_propagation(self, propagated_heat: Dict[str, float], source_node: str):
        """Update heat propagation to neighboring nodes"""
        
        with self.driver.session() as session:
            for node_id, heat_value in propagated_heat.items():
                if node_id != source_node and heat_value > 0.01:
                    
                    # Update the node's propagated heat
                    session.run("""
                        MATCH (n {id: $node_id})
                        SET n.propagated_heat = COALESCE(n.propagated_heat, 0.0) + $heat_value,
                            n.heat_sources = COALESCE(n.heat_sources, []) + [$source_node]
                    """,
                    node_id=node_id,
                    heat_value=heat_value,
                    source_node=source_node
                    )
                    
                    # Create/update heat flow relationship
                    session.run("""
                        MATCH (source {id: $source_node})
                        MATCH (target {id: $target_node})
                        MERGE (source)-[r:HEAT_FLOW]->(target)
                        SET r.intensity = $heat_value,
                            r.timestamp = $timestamp,
                            r.decay_rate = 0.1
                    """,
                    source_node=source_node,
                    target_node=node_id,
                    heat_value=heat_value,
                    timestamp=datetime.now().isoformat()
                    )
    
    def create_temporal_snapshot(self) -> GraphSnapshot:
        """Create a temporal snapshot of the current graph state"""
        
        with self.driver.session() as session:
            # Get all nodes with their properties
            nodes_result = session.run("""
                MATCH (n)
                RETURN n.id as id, labels(n) as labels, properties(n) as properties
            """)
            
            nodes = {}
            for record in nodes_result:
                nodes[record['id']] = {
                    'labels': record['labels'],
                    'properties': record['properties']
                }
            
            # Get all edges with their properties
            edges_result = session.run("""
                MATCH (source)-[r]->(target)
                RETURN id(r) as edge_id, source.id as source, target.id as target, 
                       type(r) as type, properties(r) as properties
            """)
            
            edges = {}
            for record in edges_result:
                edges[str(record['edge_id'])] = {
                    'source': record['source'],
                    'target': record['target'],
                    'type': record['type'],
                    'properties': record['properties']
                }
            
            # Calculate global metrics
            global_heat = self._calculate_global_heat(session)
            market_sentiment = self._calculate_market_sentiment(session)
        
        # Create snapshot using temporal manager
        snapshot = self.temporal_manager.create_snapshot(
            nodes=nodes,
            edges=edges,
            global_heat=global_heat,
            market_sentiment=market_sentiment
        )
        
        return snapshot
    
    def get_graph_with_heat_visualization(self, 
                                        time_point: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get graph data with heat-based visualization colors
        """
        
        target_time = time_point or datetime.now()
        
        # Get temporal snapshot
        snapshot = self.temporal_manager.get_graph_state_at_time(target_time)
        
        if not snapshot:
            # Fallback to current state
            snapshot = self.create_temporal_snapshot()
        
        # Prepare visualization data
        vis_nodes = []
        vis_edges = []
        
        # Process nodes with heat-based colors
        for node_id, node_data in snapshot.nodes.items():
            properties = node_data.get('properties', {})
            heat_score = properties.get('heat_score', 0.0)
            
            # Get color mapping
            if 'background_color' in properties:
                color_map = {
                    'background_color': properties['background_color'],
                    'border_color': properties.get('border_color', '#000000'),
                    'font_color': properties.get('font_color', '#000000'),
                    'heat_level': properties.get('heat_level', 'cold')
                }
            else:
                color_map = self.technical_engine.get_node_color_heat_map(heat_score)
            
            vis_node = {
                'id': node_id,
                'label': properties.get('name', node_id),
                'labels': node_data.get('labels', []),
                'heat_score': heat_score,
                'color': {
                    'background': color_map['background_color'],
                    'border': color_map['border_color']
                },
                'font': {
                    'color': color_map['font_color']
                },
                'size': max(20, min(60, 20 + heat_score * 40)),  # Size based on heat
                'heat_level': color_map['heat_level'],
                'properties': properties
            }
            
            vis_nodes.append(vis_node)
        
        # Process edges with dynamic styling
        for edge_id, edge_data in snapshot.edges.items():
            properties = edge_data.get('properties', {})
            strength = properties.get('strength', 1.0)
            edge_type = edge_data.get('type', 'UNKNOWN')
            
            # Color based on edge type and strength
            edge_color = self._get_edge_color(edge_type, strength)
            
            vis_edge = {
                'id': edge_id,
                'from': edge_data['source'],
                'to': edge_data['target'],
                'label': edge_type,
                'color': edge_color,
                'width': max(1, min(8, strength * 5)),  # Width based on strength
                'arrows': 'to' if edge_type in ['CONTAINS', 'BELONGS_TO', 'HEAT_FLOW'] else '',
                'properties': properties
            }
            
            vis_edges.append(vis_edge)
        
        return {
            'nodes': vis_nodes,
            'edges': vis_edges,
            'snapshot_info': {
                'snapshot_id': snapshot.snapshot_id,
                'timestamp': snapshot.temporal_point.event_time.isoformat(),
                'global_heat': snapshot.global_heat,
                'market_sentiment': snapshot.market_sentiment,
                'validity_score': snapshot.validity_score,
                'node_count': len(vis_nodes),
                'edge_count': len(vis_edges)
            }
        }
    
    def get_heat_analytics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive heat analytics for dashboard"""
        
        with self.driver.session() as session:
            # Top heated stocks
            hot_stocks = session.run("""
                MATCH (s:Stock)
                WHERE s.heat_score IS NOT NULL
                RETURN s.id as stock_id, s.name as name, s.heat_score as heat,
                       s.current_price as price, s.heat_level as level,
                       s.macd_trend as trend, s.composite_weight as weight
                ORDER BY s.heat_score DESC
                LIMIT 10
            """).data()
            
            # Heat distribution by sector
            sector_heat = session.run("""
                MATCH (sector:Sector)<-[:BELONGS_TO]-(company:Company)<-[:BELONGS_TO]-(stock:Stock)
                WHERE stock.heat_score IS NOT NULL
                RETURN sector.name as sector, 
                       AVG(stock.heat_score) as avg_heat,
                       MAX(stock.heat_score) as max_heat,
                       COUNT(stock) as stock_count
                ORDER BY avg_heat DESC
            """).data()
            
            # Recent signals
            recent_signals = session.run("""
                MATCH (s:Stock)-[:HAS_SIGNAL]->(sig:Signal)
                WHERE datetime(sig.timestamp) > datetime() - duration('PT1H')
                RETURN s.id as stock, sig.type as signal_type, 
                       sig.direction as direction, sig.strength as strength,
                       sig.confidence as confidence, sig.timestamp as timestamp
                ORDER BY datetime(sig.timestamp) DESC
                LIMIT 20
            """).data()
            
            # Heat flow analysis
            heat_flows = session.run("""
                MATCH (source)-[flow:HEAT_FLOW]->(target)
                WHERE datetime(flow.timestamp) > datetime() - duration('PT30M')
                RETURN source.id as source, target.id as target,
                       flow.intensity as intensity, flow.timestamp as timestamp,
                       labels(source)[0] as source_type, labels(target)[0] as target_type
                ORDER BY flow.intensity DESC
                LIMIT 15
            """).data()
            
            # Global metrics
            global_metrics = session.run("""
                MATCH (n)
                WHERE n.heat_score IS NOT NULL
                RETURN AVG(n.heat_score) as avg_global_heat,
                       MAX(n.heat_score) as max_heat,
                       COUNT(CASE WHEN n.heat_score > 0.7 THEN 1 END) as extreme_heat_count,
                       COUNT(CASE WHEN n.heat_score > 0.4 THEN 1 END) as high_heat_count
            """).single()
        
        # Detect anomalies
        anomalies = self.temporal_manager.detect_graph_anomalies()
        
        # Future predictions
        future_state = self.temporal_manager.predict_future_state(
            prediction_horizon=timedelta(minutes=15)
        )
        
        return {
            'hot_stocks': hot_stocks,
            'sector_heat_distribution': sector_heat,
            'recent_signals': recent_signals,
            'heat_flows': heat_flows,
            'global_metrics': {
                'average_heat': global_metrics['avg_global_heat'] or 0.0,
                'maximum_heat': global_metrics['max_heat'] or 0.0,
                'extreme_heat_nodes': global_metrics['extreme_heat_count'] or 0,
                'high_heat_nodes': global_metrics['high_heat_count'] or 0
            },
            'anomalies': anomalies,
            'predictions': {
                'snapshot_id': future_state.snapshot_id if future_state else None,
                'prediction_time': future_state.temporal_point.event_time.isoformat() if future_state else None,
                'confidence': future_state.validity_score if future_state else 0.0
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup_old_data(self, retention_hours: int = 24):
        """Clean up old temporal data to maintain performance"""
        
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        with self.driver.session() as session:
            # Remove old signals
            result = session.run("""
                MATCH (sig:Signal)
                WHERE datetime(sig.timestamp) < datetime($cutoff_time)
                DETACH DELETE sig
                RETURN count(sig) as deleted_count
            """, cutoff_time=cutoff_time.isoformat())
            
            deleted_signals = result.single()['deleted_count']
            
            # Remove old heat flow relationships
            result = session.run("""
                MATCH ()-[flow:HEAT_FLOW]->()
                WHERE datetime(flow.timestamp) < datetime($cutoff_time)
                DELETE flow
                RETURN count(flow) as deleted_count
            """, cutoff_time=cutoff_time.isoformat())
            
            deleted_flows = result.single()['deleted_count']
        
        logger.info(f"Cleaned up {deleted_signals} old signals and {deleted_flows} old heat flows")
    
    def _serialize_macd_result(self, macd: MACDResult) -> Dict[str, Any]:
        """Serialize MACD result for storage"""
        return {
            'macd_line': macd.macd_line,
            'signal_line': macd.signal_line,
            'histogram': macd.histogram,
            'trend_direction': macd.trend_direction.value,
            'signal_strength': macd.signal_strength
        }
    
    def _serialize_bb_result(self, bb: BollingerBandsResult) -> Dict[str, Any]:
        """Serialize Bollinger Bands result for storage"""
        return {
            'upper_band': bb.upper_band,
            'middle_band': bb.middle_band,
            'lower_band': bb.lower_band,
            'bandwidth': bb.bandwidth,
            'bb_percent': bb.bb_percent
        }
    
    def _serialize_signal(self, signal: TechnicalSignal) -> Dict[str, Any]:
        """Serialize technical signal for storage"""
        return {
            'signal_type': signal.signal_type,
            'direction': signal.direction.value,
            'strength': signal.strength,
            'confidence': signal.confidence,
            'timestamp': signal.timestamp.isoformat()
        }
    
    def _calculate_global_heat(self, session) -> float:
        """Calculate global heat level"""
        result = session.run("""
            MATCH (n)
            WHERE n.heat_score IS NOT NULL
            RETURN AVG(n.heat_score) as avg_heat
        """).single()
        
        return result['avg_heat'] or 0.0
    
    def _calculate_market_sentiment(self, session) -> float:
        """Calculate overall market sentiment"""
        result = session.run("""
            MATCH (s:Stock)
            WHERE s.macd_trend IS NOT NULL
            RETURN 
                COUNT(CASE WHEN s.macd_trend = 'bullish' THEN 1 END) as bullish_count,
                COUNT(CASE WHEN s.macd_trend = 'bearish' THEN 1 END) as bearish_count,
                COUNT(*) as total_count
        """).single()
        
        if result['total_count'] == 0:
            return 0.0
        
        bullish_ratio = result['bullish_count'] / result['total_count']
        bearish_ratio = result['bearish_count'] / result['total_count']
        
        return bullish_ratio - bearish_ratio  # Range: -1 to +1
    
    def _get_edge_color(self, edge_type: str, strength: float) -> str:
        """Get edge color based on type and strength"""
        
        base_colors = {
            'CONTAINS': '#2E7D32',      # Green
            'BELONGS_TO': '#1976D2',    # Blue  
            'COMPETES_WITH': '#D32F2F', # Red
            'HEAT_FLOW': '#FF6F00',     # Orange
            'HAS_SIGNAL': '#7B1FA2'     # Purple
        }
        
        base_color = base_colors.get(edge_type, '#424242')  # Default gray
        
        # Adjust opacity based on strength
        opacity = max(0.3, min(1.0, strength))
        
        # Convert hex to rgba
        hex_color = base_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'
    
    def _enhance_color_with_trading_signal(self, color_map: Dict, trading_signal: TradingSignal) -> Dict:
        """Enhance color mapping with trading signal indicators"""
        
        enhanced_color_map = color_map.copy()
        
        # Get signal type and confidence
        signal = trading_signal.signal
        confidence = trading_signal.confidence
        
        # Enhance colors based on trading signal
        if signal == SignalStrength.STRONG_BUY:
            enhanced_color_map.update({
                'background_color': '#00C853',  # Strong green
                'border_color': '#2E7D32',
                'font_color': 'white',
                'signal_indicator': 'STRONG_BUY',
                'signal_emoji': '🚀'
            })
        elif signal == SignalStrength.BUY:
            enhanced_color_map.update({
                'background_color': '#4CAF50',  # Medium green
                'border_color': '#388E3C',
                'font_color': 'white', 
                'signal_indicator': 'BUY',
                'signal_emoji': '📈'
            })
        elif signal == SignalStrength.WEAK_BUY:
            enhanced_color_map.update({
                'background_color': '#8BC34A',  # Light green
                'border_color': '#689F38',
                'font_color': 'black',
                'signal_indicator': 'WEAK_BUY', 
                'signal_emoji': '↗️'
            })
        elif signal == SignalStrength.HOLD:
            enhanced_color_map.update({
                'background_color': '#FFC107',  # Yellow
                'border_color': '#F57C00',
                'font_color': 'black',
                'signal_indicator': 'HOLD',
                'signal_emoji': '⏸️'
            })
        elif signal == SignalStrength.WEAK_SELL:
            enhanced_color_map.update({
                'background_color': '#FF9800',  # Light orange
                'border_color': '#F57C00',
                'font_color': 'black',
                'signal_indicator': 'WEAK_SELL',
                'signal_emoji': '↘️'
            })
        elif signal == SignalStrength.SELL:
            enhanced_color_map.update({
                'background_color': '#FF5722',  # Medium red
                'border_color': '#D84315',
                'font_color': 'white',
                'signal_indicator': 'SELL',
                'signal_emoji': '📉'
            })
        elif signal == SignalStrength.STRONG_SELL:
            enhanced_color_map.update({
                'background_color': '#D32F2F',  # Strong red
                'border_color': '#B71C1C',
                'font_color': 'white',
                'signal_indicator': 'STRONG_SELL',
                'signal_emoji': '🔻'
            })
        
        # Adjust border width based on confidence
        border_width = 2 + int(confidence * 3)  # 2-5px based on confidence
        enhanced_color_map['border_width'] = border_width
        
        return enhanced_color_map