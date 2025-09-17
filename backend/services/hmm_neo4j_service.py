"""
Service for integrating HMM model predictions and regime detection with Neo4j graph database
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from neo4j import GraphDatabase

from ..models.time_series.hmm_signal_generator import HMMSignal
from ..models.time_series.market_regime_detector import RegimeDetectionResult
from ..models.time_series.unified_signal_system import UnifiedSignal
from ..models.time_series.hmm_model import HMMPrediction


class HMMNeo4jService:
    """
    Service for storing and retrieving HMM model predictions and regime detection in Neo4j
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize Neo4j connection and HMM services
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def create_market_regime_node(self, session, regime_detection: RegimeDetectionResult) -> str:
        """
        Create a market regime node in Neo4j
        
        Args:
            session: Neo4j session
            regime_detection: Regime detection result
            
        Returns:
            Created node ID
        """
        query = """
        CREATE (r:MarketRegime {
            symbol: $symbol,
            current_regime: $current_regime,
            regime_probability: $regime_probability,
            regime_duration: $regime_duration,
            regime_strength: $regime_strength,
            risk_level: $risk_level,
            recommended_action: $recommended_action,
            confidence: $confidence,
            timestamp: datetime($timestamp),
            created_at: datetime()
        })
        RETURN id(r) as node_id
        """
        
        result = session.run(query, {
            'symbol': regime_detection.symbol,
            'current_regime': regime_detection.current_regime,
            'regime_probability': regime_detection.regime_probability,
            'regime_duration': regime_detection.regime_duration,
            'regime_strength': regime_detection.regime_strength,
            'risk_level': regime_detection.risk_level,
            'recommended_action': regime_detection.recommended_action,
            'confidence': regime_detection.confidence,
            'timestamp': regime_detection.timestamp.isoformat()
        })
        
        return result.single()['node_id']
    
    def create_hmm_prediction_node(self, session, hmm_prediction: HMMPrediction) -> str:
        """
        Create an HMM prediction node in Neo4j
        
        Args:
            session: Neo4j session
            hmm_prediction: HMM prediction object
            
        Returns:
            Created node ID
        """
        query = """
        CREATE (h:HMMPrediction {
            symbol: $symbol,
            current_state: $current_state,
            predicted_state: $predicted_state,
            signal_strength: $signal_strength,
            confidence: $confidence,
            prediction_horizon: $prediction_horizon,
            model_accuracy: $model_accuracy,
            timestamp: datetime($timestamp),
            created_at: datetime()
        })
        RETURN id(h) as node_id
        """
        
        result = session.run(query, {
            'symbol': hmm_prediction.symbol,
            'current_state': hmm_prediction.current_state,
            'predicted_state': hmm_prediction.predicted_state,
            'signal_strength': hmm_prediction.signal_strength,
            'confidence': hmm_prediction.confidence,
            'prediction_horizon': hmm_prediction.prediction_horizon,
            'model_accuracy': hmm_prediction.model_accuracy,
            'timestamp': hmm_prediction.timestamp.isoformat()
        })
        
        return result.single()['node_id']
    
    def create_hmm_signal_node(self, session, hmm_signal: HMMSignal) -> str:
        """
        Create an HMM signal node in Neo4j
        
        Args:
            session: Neo4j session
            hmm_signal: HMM signal object
            
        Returns:
            Created node ID
        """
        query = """
        CREATE (s:HMMSignal {
            symbol: $symbol,
            combined_signal: $combined_signal,
            combined_confidence: $combined_confidence,
            current_regime: $current_regime,
            regime_risk_level: $regime_risk_level,
            recommended_position_size: $recommended_position_size,
            technical_component: $technical_component,
            hmm_component: $hmm_component,
            regime_component: $regime_component,
            overall_risk_score: $overall_risk_score,
            regime_timing_phase: $regime_timing_phase,
            timestamp: datetime($timestamp),
            created_at: datetime()
        })
        RETURN id(s) as node_id
        """
        
        # Extract values safely
        regime_detection = hmm_signal.regime_detection
        risk_assessment = hmm_signal.risk_assessment
        signal_components = hmm_signal.signal_components
        regime_timing = hmm_signal.regime_timing
        position_sizing = hmm_signal.position_sizing
        
        result = session.run(query, {
            'symbol': hmm_signal.symbol,
            'combined_signal': hmm_signal.combined_signal.value,
            'combined_confidence': hmm_signal.combined_confidence,
            'current_regime': regime_detection.current_regime,
            'regime_risk_level': regime_detection.risk_level,
            'recommended_position_size': position_sizing.get('recommended_position_size', 0.02),
            'technical_component': signal_components.get('technical_component', 0.0),
            'hmm_component': signal_components.get('hmm_component', 0.0),
            'regime_component': signal_components.get('regime_component', 0.0),
            'overall_risk_score': risk_assessment.get('overall_risk_score', 0.5),
            'regime_timing_phase': regime_timing.get('timing_phase', 'Unknown'),
            'timestamp': hmm_signal.timestamp.isoformat()
        })
        
        return result.single()['node_id']
    
    def create_unified_signal_node(self, session, unified_signal: UnifiedSignal) -> str:
        """
        Create a unified signal node in Neo4j
        
        Args:
            session: Neo4j session
            unified_signal: Unified signal object
            
        Returns:
            Created node ID
        """
        query = """
        CREATE (u:UnifiedSignal {
            symbol: $symbol,
            unified_signal: $unified_signal,
            unified_confidence: $unified_confidence,
            technical_weight: $technical_weight,
            garch_weight: $garch_weight,
            hmm_weight: $hmm_weight,
            consensus_score: $consensus_score,
            direction_consensus: $direction_consensus,
            recommended_allocation: $recommended_allocation,
            entry_method: $entry_method,
            time_horizon: $time_horizon,
            overall_risk_level: $overall_risk_level,
            timestamp: datetime($timestamp),
            created_at: datetime()
        })
        RETURN id(u) as node_id
        """
        
        # Extract values safely
        model_consensus = unified_signal.model_consensus
        portfolio_allocation = unified_signal.portfolio_allocation
        execution_strategy = unified_signal.execution_strategy
        model_diagnostics = unified_signal.model_diagnostics
        
        result = session.run(query, {
            'symbol': unified_signal.symbol,
            'unified_signal': unified_signal.unified_signal.value,
            'unified_confidence': unified_signal.unified_confidence,
            'technical_weight': 0.3,  # Default values - would be extracted from system config
            'garch_weight': 0.35,
            'hmm_weight': 0.35,
            'consensus_score': model_consensus.get('consensus_score', 0.5),
            'direction_consensus': model_consensus.get('direction_consensus', 'MIXED'),
            'recommended_allocation': portfolio_allocation.get('recommended_allocation', 0.02),
            'entry_method': execution_strategy.get('entry_method', 'MARKET'),
            'time_horizon': execution_strategy.get('time_horizon', 'MEDIUM'),
            'overall_risk_level': 'MEDIUM',  # Would be calculated from diagnostics
            'timestamp': unified_signal.timestamp.isoformat()
        })
        
        return result.single()['node_id']
    
    def create_regime_transition_edge(self, session, from_regime: str, to_regime: str, 
                                    probability: float, symbol: str):
        """
        Create regime transition edge between regime nodes
        
        Args:
            session: Neo4j session
            from_regime: Source regime
            to_regime: Target regime
            probability: Transition probability
            symbol: Stock symbol
        """
        query = """
        MATCH (r1:MarketRegime {current_regime: $from_regime, symbol: $symbol})
        MATCH (r2:MarketRegime {current_regime: $to_regime, symbol: $symbol})
        WHERE r1.created_at >= datetime() - duration('P7D')
        AND r2.created_at >= datetime() - duration('P7D')
        WITH r1, r2 ORDER BY r1.created_at DESC, r2.created_at DESC
        LIMIT 1
        MERGE (r1)-[t:TRANSITIONS_TO {
            probability: $probability,
            created_at: datetime()
        }]->(r2)
        """
        
        session.run(query, {
            'from_regime': from_regime,
            'to_regime': to_regime,
            'probability': probability,
            'symbol': symbol
        })
    
    def store_hmm_signal(self, hmm_signal: HMMSignal) -> Dict[str, str]:
        """
        Store complete HMM signal in Neo4j with all relationships
        
        Args:
            hmm_signal: HMM signal object
            
        Returns:
            Dictionary with created node IDs
        """
        with self.driver.session() as session:
            # Create main HMM signal node
            signal_node_id = self.create_hmm_signal_node(session, hmm_signal)
            
            # Create regime detection node
            regime_node_id = self.create_market_regime_node(session, hmm_signal.regime_detection)
            
            # Create HMM prediction node
            hmm_pred_node_id = self.create_hmm_prediction_node(session, hmm_signal.hmm_prediction)
            
            # Link signal to regime and prediction
            link_query = """
            MATCH (s:HMMSignal) WHERE id(s) = $signal_id
            MATCH (r:MarketRegime) WHERE id(r) = $regime_id
            MATCH (h:HMMPrediction) WHERE id(h) = $hmm_id
            CREATE (s)-[:BASED_ON_REGIME]->(r)
            CREATE (s)-[:USES_HMM_PREDICTION]->(h)
            """
            
            session.run(link_query, {
                'signal_id': signal_node_id,
                'regime_id': regime_node_id,
                'hmm_id': hmm_pred_node_id
            })
            
            # Create regime transitions if available
            transition_probs = hmm_signal.regime_detection.transition_probabilities
            current_regime = hmm_signal.regime_detection.current_regime
            
            for target_regime, probability in transition_probs.items():
                if target_regime != current_regime and probability > 0.1:
                    self.create_regime_transition_edge(
                        session, current_regime, target_regime, probability, hmm_signal.symbol
                    )
            
            # Link to stock if exists
            try:
                self.link_signal_to_stock(session, signal_node_id, hmm_signal.symbol)
            except Exception:
                pass  # Stock node might not exist
            
            return {
                'signal_node_id': signal_node_id,
                'regime_node_id': regime_node_id,
                'hmm_prediction_node_id': hmm_pred_node_id
            }
    
    def store_unified_signal(self, unified_signal: UnifiedSignal) -> Dict[str, str]:
        """
        Store unified signal in Neo4j with all component relationships
        
        Args:
            unified_signal: Unified signal object
            
        Returns:
            Dictionary with created node IDs
        """
        with self.driver.session() as session:
            # Create unified signal node
            unified_node_id = self.create_unified_signal_node(session, unified_signal)
            
            # Store component signals
            hmm_nodes = self.store_hmm_signal(unified_signal.hmm_signal)
            
            # Create GARCH signal node (simplified for this example)
            garch_node_id = self._create_simplified_garch_node(session, unified_signal.garch_signal)
            
            # Create technical signal node (simplified)
            tech_node_id = self._create_simplified_technical_node(session, unified_signal.base_technical_signal)
            
            # Link unified signal to component signals
            link_components_query = """
            MATCH (u:UnifiedSignal) WHERE id(u) = $unified_id
            MATCH (h:HMMSignal) WHERE id(h) = $hmm_id
            MATCH (g:GARCHSignal) WHERE id(g) = $garch_id
            MATCH (t:TechnicalSignal) WHERE id(t) = $tech_id
            CREATE (u)-[:COMBINES_HMM]->(h)
            CREATE (u)-[:COMBINES_GARCH]->(g)
            CREATE (u)-[:COMBINES_TECHNICAL]->(t)
            """
            
            session.run(link_components_query, {
                'unified_id': unified_node_id,
                'hmm_id': hmm_nodes['signal_node_id'],
                'garch_id': garch_node_id,
                'tech_id': tech_node_id
            })
            
            return {
                'unified_signal_id': unified_node_id,
                'hmm_signal_id': hmm_nodes['signal_node_id'],
                'garch_signal_id': garch_node_id,
                'technical_signal_id': tech_node_id
            }
    
    def _create_simplified_garch_node(self, session, garch_signal) -> str:
        """Create simplified GARCH signal node"""
        query = """
        CREATE (g:GARCHSignal {
            symbol: $symbol,
            signal: $signal,
            confidence: $confidence,
            predicted_volatility: $predicted_volatility,
            volatility_regime: $volatility_regime,
            timestamp: datetime($timestamp),
            created_at: datetime()
        })
        RETURN id(g) as node_id
        """
        
        result = session.run(query, {
            'symbol': garch_signal.symbol,
            'signal': garch_signal.combined_signal.value,
            'confidence': garch_signal.combined_confidence,
            'predicted_volatility': garch_signal.garch_prediction.predicted_volatility,
            'volatility_regime': garch_signal.garch_prediction.volatility_regime,
            'timestamp': garch_signal.timestamp.isoformat()
        })
        
        return result.single()['node_id']
    
    def _create_simplified_technical_node(self, session, technical_signal) -> str:
        """Create simplified technical signal node"""
        query = """
        CREATE (t:TechnicalSignal {
            symbol: $symbol,
            signal: $signal,
            confidence: $confidence,
            price_target: $price_target,
            stop_loss: $stop_loss,
            timestamp: datetime($timestamp),
            created_at: datetime()
        })
        RETURN id(t) as node_id
        """
        
        result = session.run(query, {
            'symbol': technical_signal.symbol,
            'signal': technical_signal.signal.value,
            'confidence': technical_signal.confidence,
            'price_target': technical_signal.price_target or 0,
            'stop_loss': technical_signal.stop_loss or 0,
            'timestamp': technical_signal.timestamp.isoformat()
        })
        
        return result.single()['node_id']
    
    def link_signal_to_stock(self, session, signal_node_id: str, symbol: str):
        """Link signal to existing stock node"""
        query = """
        MATCH (signal) WHERE id(signal) = $signal_id
        MATCH (stock:Stock {symbol: $symbol})
        MERGE (stock)-[:HAS_HMM_SIGNAL]->(signal)
        """
        
        session.run(query, {
            'signal_id': signal_node_id,
            'symbol': symbol
        })
    
    def get_latest_regime_detections(self, limit: int = 10) -> List[Dict]:
        """Retrieve latest regime detections from Neo4j"""
        query = """
        MATCH (r:MarketRegime)
        OPTIONAL MATCH (stock:Stock)-[:HAS_HMM_SIGNAL]->(s:HMMSignal)-[:BASED_ON_REGIME]->(r)
        RETURN r, stock.symbol as stock_symbol
        ORDER BY r.created_at DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'limit': limit})
            
            regimes = []
            for record in result:
                regime_props = dict(record['r'])
                regimes.append({
                    'regime': regime_props,
                    'stock_symbol': record['stock_symbol']
                })
            
            return regimes
    
    def get_regime_transitions(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get regime transitions for a symbol"""
        query = """
        MATCH (r1:MarketRegime {symbol: $symbol})-[t:TRANSITIONS_TO]->(r2:MarketRegime {symbol: $symbol})
        WHERE t.created_at >= datetime() - duration({days: $days})
        RETURN r1.current_regime as from_regime, 
               r2.current_regime as to_regime,
               t.probability as probability,
               t.created_at as transition_time
        ORDER BY t.created_at DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'symbol': symbol, 'days': days})
            
            transitions = []
            for record in result:
                transitions.append(dict(record))
            
            return transitions
    
    def get_regime_analysis(self, symbol: str, days: int = 60) -> Dict[str, any]:
        """Get comprehensive regime analysis for a symbol"""
        
        # Get recent regimes
        regime_query = """
        MATCH (r:MarketRegime {symbol: $symbol})
        WHERE r.created_at >= datetime() - duration({days: $days})
        RETURN r
        ORDER BY r.created_at DESC
        """
        
        # Get regime transitions
        transition_query = """
        MATCH (r1:MarketRegime {symbol: $symbol})-[t:TRANSITIONS_TO]->(r2:MarketRegime {symbol: $symbol})
        WHERE t.created_at >= datetime() - duration({days: $days})
        RETURN r1.current_regime as from_regime, 
               r2.current_regime as to_regime,
               AVG(t.probability) as avg_probability,
               COUNT(t) as transition_count
        """
        
        with self.driver.session() as session:
            # Get regimes
            regime_result = session.run(regime_query, {'symbol': symbol, 'days': days})
            regimes = [dict(record['r']) for record in regime_result]
            
            # Get transitions
            transition_result = session.run(transition_query, {'symbol': symbol, 'days': days})
            transitions = [dict(record) for record in transition_result]
            
            # Calculate regime statistics
            if regimes:
                regime_counts = {}
                total_duration = 0
                
                for regime in regimes:
                    regime_name = regime['current_regime']
                    duration = regime['regime_duration']
                    
                    if regime_name not in regime_counts:
                        regime_counts[regime_name] = {
                            'count': 0,
                            'total_duration': 0,
                            'avg_strength': 0,
                            'avg_confidence': 0
                        }
                    
                    regime_counts[regime_name]['count'] += 1
                    regime_counts[regime_name]['total_duration'] += duration
                    regime_counts[regime_name]['avg_strength'] += regime['regime_strength']
                    regime_counts[regime_name]['avg_confidence'] += regime['confidence']
                    total_duration += duration
                
                # Calculate averages
                for regime_name, stats in regime_counts.items():
                    count = stats['count']
                    stats['avg_duration'] = stats['total_duration'] / count
                    stats['avg_strength'] /= count
                    stats['avg_confidence'] /= count
                    stats['frequency'] = stats['total_duration'] / total_duration if total_duration > 0 else 0
            
            else:
                regime_counts = {}
            
            return {
                'symbol': symbol,
                'analysis_period_days': days,
                'regime_statistics': regime_counts,
                'regime_transitions': transitions,
                'total_regime_changes': len(transitions),
                'current_regime': regimes[0]['current_regime'] if regimes else 'Unknown',
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def get_signal_performance_metrics(self, days: int = 30) -> Dict[str, any]:
        """Calculate performance metrics for HMM signals"""
        
        query = """
        MATCH (s:HMMSignal)
        WHERE s.created_at >= datetime() - duration({days: $days})
        RETURN 
            COUNT(*) as total_signals,
            AVG(s.combined_confidence) as avg_confidence,
            COLLECT(DISTINCT s.current_regime) as active_regimes,
            AVG(s.recommended_position_size) as avg_position_size,
            s.regime_risk_level as risk_level,
            COUNT(s.regime_risk_level) as risk_count
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'days': days})
            record = result.single()
            
            if record:
                return {
                    'total_signals_' + str(days) + 'd': record['total_signals'],
                    'avg_confidence': record['avg_confidence'],
                    'active_regimes': record['active_regimes'],
                    'avg_position_size': record['avg_position_size'],
                    'performance_period_days': days,
                    'last_updated': datetime.now().isoformat()
                }
            else:
                return {'message': f'No signals found in last {days} days'}
    
    def cleanup_old_data(self, days_old: int = 60) -> int:
        """Remove old HMM data"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        query = """
        MATCH (n)
        WHERE n:HMMSignal OR n:MarketRegime OR n:HMMPrediction OR n:UnifiedSignal
        AND n.created_at < datetime($cutoff_date)
        DETACH DELETE n
        RETURN COUNT(n) as deleted_count
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'cutoff_date': cutoff_date.isoformat()})
            return result.single()['deleted_count']
    
    def get_model_consensus_analysis(self, symbol: str, days: int = 7) -> Dict[str, any]:
        """Analyze consensus between different models for a symbol"""
        
        query = """
        MATCH (u:UnifiedSignal {symbol: $symbol})
        WHERE u.created_at >= datetime() - duration({days: $days})
        RETURN 
            u.consensus_score as consensus_score,
            u.direction_consensus as direction_consensus,
            u.unified_signal as unified_signal,
            u.unified_confidence as unified_confidence,
            u.created_at as timestamp
        ORDER BY u.created_at DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'symbol': symbol, 'days': days})
            consensus_data = [dict(record) for record in result]
            
            if consensus_data:
                avg_consensus = np.mean([c['consensus_score'] for c in consensus_data])
                avg_confidence = np.mean([c['unified_confidence'] for c in consensus_data])
                
                # Count signal types
                signal_counts = {}
                for c in consensus_data:
                    signal = c['unified_signal']
                    signal_counts[signal] = signal_counts.get(signal, 0) + 1
                
                return {
                    'symbol': symbol,
                    'avg_consensus_score': avg_consensus,
                    'avg_confidence': avg_confidence,
                    'signal_distribution': signal_counts,
                    'total_signals': len(consensus_data),
                    'analysis_period_days': days,
                    'recent_signals': consensus_data[:5]  # Last 5 signals
                }
            else:
                return {
                    'symbol': symbol,
                    'message': f'No unified signals found in last {days} days'
                }
    
    async def batch_store_signals(self, signals: List[Union[HMMSignal, UnifiedSignal]]) -> Dict[str, any]:
        """Batch store multiple signals efficiently"""
        
        results = {
            'stored': [],
            'errors': [],
            'total_nodes_created': 0
        }
        
        for signal in signals:
            try:
                if isinstance(signal, HMMSignal):
                    node_ids = self.store_hmm_signal(signal)
                elif isinstance(signal, UnifiedSignal):
                    node_ids = self.store_unified_signal(signal)
                else:
                    continue
                
                results['stored'].append({
                    'symbol': signal.symbol,
                    'signal_type': type(signal).__name__,
                    'node_ids': node_ids
                })
                results['total_nodes_created'] += len(node_ids)
                
            except Exception as e:
                results['errors'].append({
                    'symbol': getattr(signal, 'symbol', 'unknown'),
                    'error': str(e)
                })
        
        return results