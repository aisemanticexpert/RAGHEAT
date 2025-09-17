"""
Service for integrating GARCH model predictions with Neo4j graph database
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from neo4j import GraphDatabase

from ..models.time_series.garch_signal_generator import GARCHSignalGenerator, GARCHSignal
from ..models.time_series.garch_model import GARCHPrediction
from ..signals.buy_sell_engine import SignalStrength


class GARCHNeo4jService:
    """
    Service for storing and retrieving GARCH model predictions in Neo4j
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize Neo4j connection and GARCH signal generator
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.signal_generator = GARCHSignalGenerator()
        
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def create_garch_prediction_node(self, session, prediction: GARCHPrediction) -> str:
        """
        Create a GARCH prediction node in Neo4j
        
        Args:
            session: Neo4j session
            prediction: GARCH prediction object
            
        Returns:
            Created node ID
        """
        query = """
        CREATE (p:GARCHPrediction {
            symbol: $symbol,
            predicted_volatility: $predicted_volatility,
            predicted_returns: $predicted_returns,
            confidence_lower: $confidence_lower,
            confidence_upper: $confidence_upper,
            signal_strength: $signal_strength,
            volatility_regime: $volatility_regime,
            prediction_horizon: $prediction_horizon,
            model_accuracy: $model_accuracy,
            timestamp: datetime($timestamp),
            created_at: datetime()
        })
        RETURN id(p) as node_id
        """
        
        result = session.run(query, {
            'symbol': prediction.symbol,
            'predicted_volatility': prediction.predicted_volatility,
            'predicted_returns': prediction.predicted_returns,
            'confidence_lower': prediction.confidence_interval[0],
            'confidence_upper': prediction.confidence_interval[1],
            'signal_strength': prediction.signal_strength,
            'volatility_regime': prediction.volatility_regime,
            'prediction_horizon': prediction.prediction_horizon,
            'model_accuracy': prediction.model_accuracy,
            'timestamp': prediction.timestamp.isoformat()
        })
        
        return result.single()['node_id']
    
    def create_garch_signal_node(self, session, signal: GARCHSignal) -> str:
        """
        Create a comprehensive GARCH signal node in Neo4j
        
        Args:
            session: Neo4j session  
            signal: GARCH signal object
            
        Returns:
            Created node ID
        """
        query = """
        CREATE (s:GARCHSignal {
            symbol: $symbol,
            combined_signal: $combined_signal,
            combined_confidence: $combined_confidence,
            base_signal: $base_signal,
            base_confidence: $base_confidence,
            garch_signal_strength: $garch_signal_strength,
            garch_volatility: $garch_volatility,
            volatility_regime: $volatility_regime,
            price_target: $price_target,
            stop_loss: $stop_loss,
            take_profit_1: $take_profit_1,
            take_profit_2: $take_profit_2,
            recommended_position_size: $recommended_position_size,
            kelly_fraction: $kelly_fraction,
            sharpe_ratio: $sharpe_ratio,
            max_drawdown: $max_drawdown,
            var_95: $var_95,
            technical_component: $technical_component,
            garch_component: $garch_component,
            volatility_component: $volatility_component,
            timestamp: datetime($timestamp),
            created_at: datetime()
        })
        RETURN id(s) as node_id
        """
        
        # Extract values safely
        vol_targets = signal.volatility_adjusted_targets
        risk_metrics = signal.risk_metrics
        components = signal.signal_components
        
        result = session.run(query, {
            'symbol': signal.symbol,
            'combined_signal': signal.combined_signal.value,
            'combined_confidence': signal.combined_confidence,
            'base_signal': signal.base_signal.signal.value,
            'base_confidence': signal.base_signal.confidence,
            'garch_signal_strength': signal.garch_prediction.signal_strength,
            'garch_volatility': signal.garch_prediction.predicted_volatility,
            'volatility_regime': signal.garch_prediction.volatility_regime,
            'price_target': vol_targets.get('price_target'),
            'stop_loss': vol_targets.get('stop_loss'),
            'take_profit_1': vol_targets.get('take_profit_1'),
            'take_profit_2': vol_targets.get('take_profit_2'),
            'recommended_position_size': risk_metrics.get('recommended_position_size', 0.01),
            'kelly_fraction': risk_metrics.get('kelly_fraction', 0.05),
            'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': risk_metrics.get('max_drawdown', 0.0),
            'var_95': risk_metrics.get('var_95', 0.0),
            'technical_component': components.get('technical_component', 0.0),
            'garch_component': components.get('garch_component', 0.0),
            'volatility_component': components.get('volatility_component', 0.0),
            'timestamp': signal.timestamp.isoformat()
        })
        
        return result.single()['node_id']
    
    def link_signal_to_stock(self, session, signal_node_id: str, symbol: str):
        """
        Link GARCH signal to existing stock node
        
        Args:
            session: Neo4j session
            signal_node_id: ID of the signal node
            symbol: Stock symbol
        """
        query = """
        MATCH (s:GARCHSignal) WHERE id(s) = $signal_id
        MATCH (stock:Stock {symbol: $symbol})
        CREATE (stock)-[:HAS_GARCH_SIGNAL]->(s)
        """
        
        session.run(query, {
            'signal_id': signal_node_id,
            'symbol': symbol
        })
    
    def create_signal_relationships(self, session, signal_node_id: str, signal: GARCHSignal):
        """
        Create relationships between signal and other relevant nodes
        
        Args:
            session: Neo4j session
            signal_node_id: ID of the signal node
            signal: GARCH signal object
        """
        # Link to volatility regime
        regime_query = """
        MATCH (s:GARCHSignal) WHERE id(s) = $signal_id
        MERGE (regime:VolatilityRegime {name: $regime_name})
        CREATE (s)-[:IN_VOLATILITY_REGIME]->(regime)
        """
        
        session.run(regime_query, {
            'signal_id': signal_node_id,
            'regime_name': signal.garch_prediction.volatility_regime
        })
        
        # Link to signal strength category
        strength_query = """
        MATCH (s:GARCHSignal) WHERE id(s) = $signal_id
        MERGE (strength:SignalStrength {name: $strength_name})
        CREATE (s)-[:HAS_STRENGTH]->(strength)
        """
        
        session.run(strength_query, {
            'signal_id': signal_node_id,
            'strength_name': signal.combined_signal.value
        })
    
    def store_garch_signal(self, signal: GARCHSignal) -> Dict[str, str]:
        """
        Store complete GARCH signal in Neo4j with all relationships
        
        Args:
            signal: GARCH signal object
            
        Returns:
            Dictionary with created node IDs
        """
        with self.driver.session() as session:
            # Create main signal node
            signal_node_id = self.create_garch_signal_node(session, signal)
            
            # Create GARCH prediction node
            prediction_node_id = self.create_garch_prediction_node(session, signal.garch_prediction)
            
            # Link signal to prediction
            link_prediction_query = """
            MATCH (s:GARCHSignal) WHERE id(s) = $signal_id
            MATCH (p:GARCHPrediction) WHERE id(p) = $prediction_id
            CREATE (s)-[:BASED_ON_PREDICTION]->(p)
            """
            
            session.run(link_prediction_query, {
                'signal_id': signal_node_id,
                'prediction_id': prediction_node_id
            })
            
            # Create relationships
            self.create_signal_relationships(session, signal_node_id, signal)
            
            # Link to stock if exists
            try:
                self.link_signal_to_stock(session, signal_node_id, signal.symbol)
            except Exception:
                # Stock node might not exist yet
                pass
            
            return {
                'signal_node_id': signal_node_id,
                'prediction_node_id': prediction_node_id
            }
    
    def get_latest_garch_signals(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve latest GARCH signals from Neo4j
        
        Args:
            limit: Maximum number of signals to retrieve
            
        Returns:
            List of signal dictionaries
        """
        query = """
        MATCH (s:GARCHSignal)
        OPTIONAL MATCH (s)-[:BASED_ON_PREDICTION]->(p:GARCHPrediction)
        OPTIONAL MATCH (stock:Stock)-[:HAS_GARCH_SIGNAL]->(s)
        RETURN s, p, stock.symbol as stock_symbol
        ORDER BY s.created_at DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'limit': limit})
            
            signals = []
            for record in result:
                signal_props = dict(record['s'])
                prediction_props = dict(record['p']) if record['p'] else {}
                
                signals.append({
                    'signal': signal_props,
                    'prediction': prediction_props,
                    'stock_symbol': record['stock_symbol']
                })
            
            return signals
    
    def get_signals_by_symbol(self, symbol: str, limit: int = 5) -> List[Dict]:
        """
        Get GARCH signals for a specific symbol
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of signals
            
        Returns:
            List of signals for the symbol
        """
        query = """
        MATCH (s:GARCHSignal {symbol: $symbol})
        OPTIONAL MATCH (s)-[:BASED_ON_PREDICTION]->(p:GARCHPrediction)
        RETURN s, p
        ORDER BY s.created_at DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'symbol': symbol, 'limit': limit})
            
            signals = []
            for record in result:
                signals.append({
                    'signal': dict(record['s']),
                    'prediction': dict(record['p']) if record['p'] else {}
                })
            
            return signals
    
    def get_signals_by_strength(self, strength: str, limit: int = 20) -> List[Dict]:
        """
        Get signals by signal strength (BUY, SELL, etc.)
        
        Args:
            strength: Signal strength (STRONG_BUY, BUY, etc.)
            limit: Maximum number of signals
            
        Returns:
            List of signals with specified strength
        """
        query = """
        MATCH (s:GARCHSignal {combined_signal: $strength})
        OPTIONAL MATCH (s)-[:BASED_ON_PREDICTION]->(p:GARCHPrediction)
        RETURN s, p
        ORDER BY s.combined_confidence DESC, s.created_at DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'strength': strength, 'limit': limit})
            
            signals = []
            for record in result:
                signals.append({
                    'signal': dict(record['s']),
                    'prediction': dict(record['p']) if record['p'] else {}
                })
            
            return signals
    
    def get_volatility_regime_analysis(self) -> Dict[str, any]:
        """
        Analyze volatility regimes across all signals
        
        Returns:
            Volatility regime statistics
        """
        query = """
        MATCH (s:GARCHSignal)
        RETURN s.volatility_regime as regime, 
               COUNT(*) as count,
               AVG(s.garch_volatility) as avg_volatility,
               AVG(s.combined_confidence) as avg_confidence,
               COLLECT(s.combined_signal) as signals
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            
            regime_analysis = {}
            for record in result:
                regime = record['regime']
                if regime:
                    # Count signal types
                    signal_counts = {}
                    for signal in record['signals']:
                        signal_counts[signal] = signal_counts.get(signal, 0) + 1
                    
                    regime_analysis[regime] = {
                        'total_signals': record['count'],
                        'avg_volatility': record['avg_volatility'],
                        'avg_confidence': record['avg_confidence'],
                        'signal_distribution': signal_counts
                    }
            
            return regime_analysis
    
    def cleanup_old_signals(self, days_old: int = 30) -> int:
        """
        Remove old GARCH signals and predictions
        
        Args:
            days_old: Remove signals older than this many days
            
        Returns:
            Number of nodes deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        query = """
        MATCH (s:GARCHSignal)
        WHERE s.created_at < datetime($cutoff_date)
        OPTIONAL MATCH (s)-[:BASED_ON_PREDICTION]->(p:GARCHPrediction)
        DETACH DELETE s, p
        RETURN COUNT(s) as deleted_count
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'cutoff_date': cutoff_date.isoformat()})
            return result.single()['deleted_count']
    
    def get_model_performance_metrics(self) -> Dict[str, any]:
        """
        Calculate performance metrics for GARCH models
        
        Returns:
            Performance statistics
        """
        query = """
        MATCH (s:GARCHSignal)
        WHERE s.created_at >= datetime() - duration('P7D')  // Last 7 days
        RETURN 
            COUNT(*) as total_signals,
            AVG(s.combined_confidence) as avg_confidence,
            AVG(s.garch_component) as avg_garch_contribution,
            AVG(s.technical_component) as avg_technical_contribution,
            COLLECT(DISTINCT s.volatility_regime) as regimes,
            AVG(s.recommended_position_size) as avg_position_size,
            AVG(s.kelly_fraction) as avg_kelly_fraction
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            
            if record:
                return {
                    'total_signals_7d': record['total_signals'],
                    'avg_confidence': record['avg_confidence'],
                    'avg_garch_contribution': record['avg_garch_contribution'],
                    'avg_technical_contribution': record['avg_technical_contribution'],
                    'active_regimes': record['regimes'],
                    'avg_position_size': record['avg_position_size'],
                    'avg_kelly_fraction': record['avg_kelly_fraction'],
                    'last_updated': datetime.now().isoformat()
                }
            else:
                return {'message': 'No signals found in last 7 days'}
    
    async def batch_process_symbols(self, symbols: List[str], heat_scores: Dict[str, float] = None) -> Dict[str, any]:
        """
        Process multiple symbols and store results in Neo4j
        
        Args:
            symbols: List of stock symbols
            heat_scores: Optional heat scores for each symbol
            
        Returns:
            Processing results summary
        """
        results = {
            'processed': [],
            'errors': [],
            'stored_nodes': []
        }
        
        for symbol in symbols:
            try:
                heat_score = heat_scores.get(symbol, 0.5) if heat_scores else 0.5
                
                # Generate signal
                signal = self.signal_generator.generate_signal(
                    symbol=symbol,
                    current_price=100.0,  # Would need real price feed
                    heat_score=heat_score
                )
                
                # Store in Neo4j
                node_ids = self.store_garch_signal(signal)
                
                results['processed'].append(symbol)
                results['stored_nodes'].append({
                    'symbol': symbol,
                    'node_ids': node_ids
                })
                
            except Exception as e:
                results['errors'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return results