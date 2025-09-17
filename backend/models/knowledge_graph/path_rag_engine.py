"""
PathRAG Engine for Multi-Hop Reasoning Paths
Implements sophisticated path-based reasoning for financial knowledge graphs
"""

import asyncio
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from enum import Enum
import json
import heapq
from collections import defaultdict, deque
import logging

from .graph_rag_engine import GraphEntity, GraphRelationship, ReasoningType

logger = logging.getLogger(__name__)


class PathType(Enum):
    CAUSAL = "causal"  # A causes B causes C
    CORRELATIONAL = "correlational"  # A correlates with B correlates with C
    HIERARCHICAL = "hierarchical"  # Market -> Sector -> Stock
    TEMPORAL = "temporal"  # Time-based sequences
    COMPETITIVE = "competitive"  # Competitive relationships
    SUPPLY_CHAIN = "supply_chain"  # Supply chain dependencies


@dataclass
class PathHop:
    """A single hop in a reasoning path"""
    source_entity: GraphEntity
    target_entity: GraphEntity
    relationship: GraphRelationship
    hop_confidence: float
    hop_explanation: str
    temporal_context: Optional[Dict[str, Any]] = None


@dataclass
class ReasoningPathAnalysis:
    """Detailed analysis of a reasoning path"""
    path_id: str
    hops: List[PathHop]
    path_type: PathType
    reasoning_type: ReasoningType
    total_confidence: float
    path_strength: float  # Based on relationship weights
    novelty_score: float  # How unique this path is
    actionability_score: float  # How actionable the insights are
    risk_score: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    conclusion: str
    recommended_actions: List[str]
    time_sensitivity: str  # immediate, short_term, long_term
    path_length: int


@dataclass
class PathRAGQuery:
    """A query for multi-hop path reasoning"""
    query_text: str
    start_entity_ids: List[str]
    target_entity_ids: Optional[List[str]] = None
    max_hops: int = 4
    path_types: List[PathType] = None
    reasoning_types: List[ReasoningType] = None
    min_confidence: float = 0.3
    max_paths: int = 10


@dataclass
class PathRAGResult:
    """Result from PathRAG analysis"""
    query: PathRAGQuery
    discovered_paths: List[ReasoningPathAnalysis]
    path_intersections: Dict[str, List[str]]  # Where paths intersect
    consensus_insights: List[str]  # Insights supported by multiple paths
    conflicting_signals: List[str]  # Where paths disagree
    meta_analysis: Dict[str, Any]  # Analysis of the path set
    execution_time: float
    timestamp: datetime


class AdvancedPathRAG:
    """
    Advanced Path-based Retrieval-Augmented Generation
    """
    
    def __init__(self, graph_rag_engine):
        self.graph_rag = graph_rag_engine
        self.path_cache = {}  # Cache for expensive path computations
        self.path_patterns = self.initialize_path_patterns()
        
    def initialize_path_patterns(self) -> Dict[PathType, Dict]:
        """Initialize common path patterns for financial analysis"""
        return {
            PathType.CAUSAL: {
                "patterns": [
                    ["Indicator", "INFLUENCES", "Sector", "AFFECTS", "Stock"],
                    ["News", "IMPACTS", "Sentiment", "DRIVES", "Price"],
                    ["HeatSource", "HEATS", "Sector", "CONTAINS", "Stock"]
                ],
                "confidence_decay": 0.8,  # Confidence decay per hop
                "max_typical_length": 4
            },
            PathType.HIERARCHICAL: {
                "patterns": [
                    ["Market", "CONTAINS", "Sector", "CONTAINS", "Stock"],
                    ["Sector", "CONTAINS", "Stock", "CORRELATED_WITH", "Stock"]
                ],
                "confidence_decay": 0.9,
                "max_typical_length": 3
            },
            PathType.CORRELATIONAL: {
                "patterns": [
                    ["Stock", "CORRELATED_WITH", "Stock", "CORRELATED_WITH", "Stock"],
                    ["Sector", "CORRELATED_WITH", "Sector"]
                ],
                "confidence_decay": 0.7,
                "max_typical_length": 5
            },
            PathType.COMPETITIVE: {
                "patterns": [
                    ["Stock", "COMPETES_WITH", "Stock", "IN_SECTOR", "Sector"],
                    ["Sector", "ALTERNATIVE_TO", "Sector"]
                ],
                "confidence_decay": 0.8,
                "max_typical_length": 3
            }
        }
    
    async def find_multi_hop_paths(self, query: PathRAGQuery) -> List[ReasoningPathAnalysis]:
        """Find sophisticated multi-hop reasoning paths"""
        all_paths = []
        
        try:
            # For each starting entity, find paths
            for start_entity_id in query.start_entity_ids:
                if start_entity_id not in self.graph_rag.entities:
                    continue
                
                start_entity = self.graph_rag.entities[start_entity_id]
                
                # Use different algorithms based on query characteristics
                if query.target_entity_ids:
                    # Targeted search - find paths to specific entities
                    paths = await self.find_targeted_paths(
                        start_entity_id, query.target_entity_ids, query.max_hops
                    )
                else:
                    # Exploratory search - find interesting paths
                    paths = await self.find_exploratory_paths(
                        start_entity_id, query.max_hops, query.path_types
                    )
                
                # Filter and analyze paths
                for path in paths:
                    analyzed_path = await self.analyze_path(path, query)
                    if analyzed_path and analyzed_path.total_confidence >= query.min_confidence:
                        all_paths.append(analyzed_path)
            
            # Sort by relevance and confidence
            all_paths.sort(key=lambda p: (p.total_confidence * p.actionability_score), reverse=True)
            
            # Return top paths
            return all_paths[:query.max_paths]
            
        except Exception as e:
            logger.error(f"❌ Error finding multi-hop paths: {e}")
            return []
    
    async def find_targeted_paths(self, start_id: str, target_ids: List[str], max_hops: int) -> List[List[PathHop]]:
        """Find paths between specific entities using A* algorithm"""
        paths = []
        
        for target_id in target_ids:
            if target_id not in self.graph_rag.entities:
                continue
                
            # Use NetworkX shortest path with weights
            try:
                # Find multiple paths using different algorithms
                simple_paths = list(nx.all_simple_paths(
                    self.graph_rag.graph, start_id, target_id, cutoff=max_hops
                ))
                
                for path_nodes in simple_paths[:5]:  # Limit to 5 paths per target
                    path_hops = []
                    valid_path = True
                    
                    for i in range(len(path_nodes) - 1):
                        source_id, target_id_hop = path_nodes[i], path_nodes[i + 1]
                        
                        # Get edge data
                        edge_data = self.graph_rag.graph.get_edge_data(source_id, target_id_hop)
                        if not edge_data:
                            valid_path = False
                            break
                        
                        # Use first relationship (could be enhanced)
                        rel_type = list(edge_data.keys())[0]
                        rel_props = edge_data[rel_type]
                        
                        # Create hop
                        hop = PathHop(
                            source_entity=self.graph_rag.entities[source_id],
                            target_entity=self.graph_rag.entities[target_id_hop],
                            relationship=GraphRelationship(
                                source_id=source_id,
                                target_id=target_id_hop,
                                relationship_type=rel_type,
                                properties=rel_props,
                                weight=rel_props.get('weight', 0.5),
                                confidence=rel_props.get('confidence', 0.5)
                            ),
                            hop_confidence=rel_props.get('confidence', 0.5),
                            hop_explanation=f"{source_id} {rel_type} {target_id_hop}"
                        )
                        path_hops.append(hop)
                    
                    if valid_path and path_hops:
                        paths.append(path_hops)
                        
            except nx.NetworkXNoPath:
                continue  # No path exists
            except Exception as e:
                logger.warning(f"⚠️ Error finding path from {start_id} to {target_id}: {e}")
                continue
        
        return paths
    
    async def find_exploratory_paths(self, start_id: str, max_hops: int, preferred_types: List[PathType]) -> List[List[PathHop]]:
        """Find interesting exploratory paths using guided BFS"""
        paths = []
        visited_paths = set()
        
        # Priority queue: (negative_score, path_hops, current_entity_id)
        pq = [(0.0, [], start_id)]
        
        while pq and len(paths) < 20:  # Limit exploration
            neg_score, current_path, current_id = heapq.heappop(pq)
            
            if len(current_path) >= max_hops:
                continue
            
            # Path signature for duplicate detection
            path_signature = tuple(hop.relationship.relationship_type for hop in current_path)
            if path_signature in visited_paths:
                continue
            visited_paths.add(path_signature)
            
            # If we have a meaningful path, add it
            if len(current_path) >= 2:
                paths.append(current_path.copy())
            
            # Explore neighbors
            current_entity = self.graph_rag.entities.get(current_id)
            if not current_entity:
                continue
            
            for neighbor_id in self.graph_rag.graph.successors(current_id):
                if any(hop.target_entity.id == neighbor_id for hop in current_path):
                    continue  # Avoid cycles
                
                neighbor_entity = self.graph_rag.entities.get(neighbor_id)
                if not neighbor_entity:
                    continue
                
                # Get relationship
                edge_data = self.graph_rag.graph.get_edge_data(current_id, neighbor_id)
                if not edge_data:
                    continue
                
                for rel_type, rel_props in edge_data.items():
                    # Calculate path attractiveness score
                    attractiveness = self.calculate_path_attractiveness(
                        current_entity, neighbor_entity, rel_type, rel_props, current_path
                    )
                    
                    if attractiveness > 0.2:  # Minimum attractiveness threshold
                        new_hop = PathHop(
                            source_entity=current_entity,
                            target_entity=neighbor_entity,
                            relationship=GraphRelationship(
                                source_id=current_id,
                                target_id=neighbor_id,
                                relationship_type=rel_type,
                                properties=rel_props,
                                weight=rel_props.get('weight', 0.5),
                                confidence=rel_props.get('confidence', 0.5)
                            ),
                            hop_confidence=rel_props.get('confidence', 0.5),
                            hop_explanation=f"{current_entity.name} {rel_type} {neighbor_entity.name}"
                        )
                        
                        new_path = current_path + [new_hop]
                        heapq.heappush(pq, (-attractiveness, new_path, neighbor_id))
        
        return paths
    
    def calculate_path_attractiveness(self, source: GraphEntity, target: GraphEntity, 
                                    rel_type: str, rel_props: Dict, current_path: List[PathHop]) -> float:
        """Calculate how attractive a path extension is"""
        score = 0.0
        
        # Base relationship strength
        score += rel_props.get('weight', 0.5) * rel_props.get('confidence', 0.5)
        
        # Entity type diversity bonus
        path_entity_types = set(hop.target_entity.type for hop in current_path)
        if target.type not in path_entity_types:
            score += 0.2
        
        # Heat-related entities are more attractive
        if 'heat' in target.properties.get('type', '').lower() or 'Heat' in target.type:
            score += 0.3
        
        # High-value entities (stocks, sectors)
        if target.type in ['Stock', 'Sector']:
            score += 0.2
        
        # Temporal recency bonus
        last_updated = target.properties.get('last_updated')
        if last_updated:
            if isinstance(last_updated, datetime):
                hours_old = (datetime.now() - last_updated).total_seconds() / 3600
                recency_bonus = max(0, 0.2 * (1 - hours_old / 24))  # Decay over 24 hours
                score += recency_bonus
        
        # Path length penalty (prefer shorter meaningful paths)
        score -= len(current_path) * 0.05
        
        return max(0, score)
    
    async def analyze_path(self, path_hops: List[PathHop], query: PathRAGQuery) -> Optional[ReasoningPathAnalysis]:
        """Perform deep analysis of a reasoning path"""
        try:
            if not path_hops:
                return None
            
            # Generate unique path ID
            path_id = f"path_{hash(tuple(hop.relationship.relationship_type for hop in path_hops))}"
            
            # Determine path type
            path_type = self.classify_path_type(path_hops)
            reasoning_type = self.classify_reasoning_type(path_hops)
            
            # Calculate confidence metrics
            hop_confidences = [hop.hop_confidence for hop in path_hops]
            total_confidence = np.prod(hop_confidences) if hop_confidences else 0.0
            
            # Calculate path strength (based on relationship weights)
            path_weights = [hop.relationship.weight for hop in path_hops]
            path_strength = np.mean(path_weights) if path_weights else 0.0
            
            # Calculate novelty (how rare this path pattern is)
            novelty_score = self.calculate_novelty_score(path_hops)
            
            # Calculate actionability
            actionability_score = self.calculate_actionability_score(path_hops, query)
            
            # Calculate risk score
            risk_score = self.calculate_path_risk_score(path_hops)
            
            # Generate evidence
            supporting_evidence = []
            contradicting_evidence = []
            
            for hop in path_hops:
                evidence = f"{hop.source_entity.name} → {hop.target_entity.name} (conf: {hop.hop_confidence:.2f})"
                
                if hop.hop_confidence > 0.7:
                    supporting_evidence.append(evidence)
                elif hop.hop_confidence < 0.4:
                    contradicting_evidence.append(f"Low confidence: {evidence}")
            
            # Generate conclusion and actions
            conclusion = self.generate_path_conclusion(path_hops, path_type, reasoning_type)
            recommended_actions = self.generate_recommended_actions(path_hops, path_type, total_confidence)
            
            # Determine time sensitivity
            time_sensitivity = self.determine_time_sensitivity(path_hops, path_type)
            
            return ReasoningPathAnalysis(
                path_id=path_id,
                hops=path_hops,
                path_type=path_type,
                reasoning_type=reasoning_type,
                total_confidence=total_confidence,
                path_strength=path_strength,
                novelty_score=novelty_score,
                actionability_score=actionability_score,
                risk_score=risk_score,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                conclusion=conclusion,
                recommended_actions=recommended_actions,
                time_sensitivity=time_sensitivity,
                path_length=len(path_hops)
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing path: {e}")
            return None
    
    def classify_path_type(self, path_hops: List[PathHop]) -> PathType:
        """Classify the type of reasoning path"""
        rel_types = [hop.relationship.relationship_type for hop in path_hops]
        entity_types = [hop.source_entity.type for hop in path_hops] + [path_hops[-1].target_entity.type]
        
        # Check for hierarchical patterns
        if all(rt in ['CONTAINS', 'BELONGS_TO'] for rt in rel_types):
            return PathType.HIERARCHICAL
        
        # Check for causal patterns
        if any(rt in ['INFLUENCES', 'AFFECTS', 'DRIVES', 'CAUSES'] for rt in rel_types):
            return PathType.CAUSAL
        
        # Check for correlational patterns
        if any(rt in ['CORRELATED_WITH', 'SIMILAR_TO'] for rt in rel_types):
            return PathType.CORRELATIONAL
        
        # Check for competitive patterns
        if any(rt in ['COMPETES_WITH', 'ALTERNATIVE_TO'] for rt in rel_types):
            return PathType.COMPETITIVE
        
        # Default to causal for most financial relationships
        return PathType.CAUSAL
    
    def classify_reasoning_type(self, path_hops: List[PathHop]) -> ReasoningType:
        """Classify the reasoning type of the path"""
        entity_types = set()
        for hop in path_hops:
            entity_types.add(hop.source_entity.type)
            entity_types.add(hop.target_entity.type)
        
        # Heat diffusion reasoning
        if any('Heat' in et for et in entity_types):
            return ReasoningType.HEAT_DIFFUSION
        
        # Technical reasoning
        if 'Indicator' in entity_types:
            return ReasoningType.TECHNICAL
        
        # Sentiment reasoning
        if any('News' in et or 'Sentiment' in et for et in entity_types):
            return ReasoningType.SENTIMENT
        
        # Sector rotation
        if entity_types.count('Sector') > 1:
            return ReasoningType.SECTOR_ROTATION
        
        # Default to momentum
        return ReasoningType.MOMENTUM
    
    def calculate_novelty_score(self, path_hops: List[PathHop]) -> float:
        """Calculate how novel/unique this path is"""
        # Simple heuristic: paths with uncommon relationship types are more novel
        rel_types = [hop.relationship.relationship_type for hop in path_hops]
        common_rels = {'CONTAINS', 'BELONGS_TO', 'CORRELATED_WITH'}
        
        uncommon_count = sum(1 for rt in rel_types if rt not in common_rels)
        return min(1.0, uncommon_count / len(rel_types))
    
    def calculate_actionability_score(self, path_hops: List[PathHop], query: PathRAGQuery) -> float:
        """Calculate how actionable the insights from this path are"""
        score = 0.0
        
        # Paths ending in stocks are more actionable
        final_entity = path_hops[-1].target_entity
        if final_entity.type == 'Stock':
            score += 0.4
        
        # Shorter paths are generally more actionable
        if len(path_hops) <= 3:
            score += 0.3
        
        # Recent data is more actionable
        recent_entities = sum(1 for hop in path_hops 
                            if hop.target_entity.last_updated and 
                            (datetime.now() - hop.target_entity.last_updated).days < 1)
        
        score += (recent_entities / len(path_hops)) * 0.3
        
        return min(1.0, score)
    
    def calculate_path_risk_score(self, path_hops: List[PathHop]) -> float:
        """Calculate risk associated with this reasoning path"""
        risk_factors = 0
        
        # Low confidence hops increase risk
        low_conf_hops = sum(1 for hop in path_hops if hop.hop_confidence < 0.5)
        risk_factors += low_conf_hops * 0.2
        
        # Very long paths are riskier
        if len(path_hops) > 4:
            risk_factors += 0.3
        
        # Paths through volatile entities are riskier
        volatile_entities = sum(1 for hop in path_hops 
                              if hop.target_entity.properties.get('volatility', 0) > 0.3)
        risk_factors += (volatile_entities / len(path_hops)) * 0.2
        
        return min(1.0, risk_factors)
    
    def generate_path_conclusion(self, path_hops: List[PathHop], path_type: PathType, reasoning_type: ReasoningType) -> str:
        """Generate a natural language conclusion for the path"""
        start_entity = path_hops[0].source_entity.name
        end_entity = path_hops[-1].target_entity.name
        
        # Build relationship chain
        chain = []
        for hop in path_hops:
            chain.append(f"{hop.source_entity.name} {hop.relationship.relationship_type} {hop.target_entity.name}")
        
        conclusion = f"Analysis path from {start_entity} to {end_entity}: "
        
        if reasoning_type == ReasoningType.HEAT_DIFFUSION:
            conclusion += f"Heat diffusion indicates potential impact on {end_entity}"
        elif reasoning_type == ReasoningType.SECTOR_ROTATION:
            conclusion += f"Sector rotation pattern suggests movement affecting {end_entity}"
        elif path_type == PathType.CAUSAL:
            conclusion += f"Causal chain suggests {start_entity} influences {end_entity}"
        elif path_type == PathType.HIERARCHICAL:
            conclusion += f"Hierarchical relationship shows {start_entity} contains/affects {end_entity}"
        else:
            conclusion += f"{reasoning_type.value.title()} analysis suggests connection to {end_entity}"
        
        return conclusion
    
    def generate_recommended_actions(self, path_hops: List[PathHop], path_type: PathType, confidence: float) -> List[str]:
        """Generate actionable recommendations based on the path"""
        actions = []
        final_entity = path_hops[-1].target_entity
        
        if final_entity.type == 'Stock':
            if confidence > 0.7:
                actions.append(f"Consider buying {final_entity.name} based on strong reasoning path")
            elif confidence > 0.5:
                actions.append(f"Monitor {final_entity.name} - moderate confidence signal")
            else:
                actions.append(f"Watch {final_entity.name} but verify with additional analysis")
        
        elif final_entity.type == 'Sector':
            if confidence > 0.6:
                actions.append(f"Consider sector rotation into {final_entity.name}")
            else:
                actions.append(f"Monitor sector performance in {final_entity.name}")
        
        # Add risk management
        if confidence < 0.6:
            actions.append("Apply strict risk management due to path uncertainty")
        
        return actions
    
    def determine_time_sensitivity(self, path_hops: List[PathHop], path_type: PathType) -> str:
        """Determine time sensitivity of the path insights"""
        # Check for time-sensitive entities or relationships
        has_heat = any('Heat' in hop.target_entity.type for hop in path_hops)
        has_news = any('News' in hop.target_entity.type for hop in path_hops)
        
        if has_news or has_heat:
            return "immediate"  # Heat and news are time-sensitive
        elif path_type == PathType.TECHNICAL:
            return "short_term"  # Technical signals are shorter-term
        elif path_type == PathType.HIERARCHICAL:
            return "long_term"  # Structural relationships are longer-term
        else:
            return "short_term"  # Default to short-term
    
    async def execute_path_rag_query(self, query: PathRAGQuery) -> PathRAGResult:
        """Execute a complete PathRAG query"""
        start_time = datetime.now()
        
        try:
            # Find multi-hop paths
            discovered_paths = await self.find_multi_hop_paths(query)
            
            # Analyze path intersections
            path_intersections = self.find_path_intersections(discovered_paths)
            
            # Generate consensus insights
            consensus_insights = self.generate_consensus_insights(discovered_paths)
            
            # Find conflicting signals
            conflicting_signals = self.find_conflicting_signals(discovered_paths)
            
            # Meta analysis
            meta_analysis = self.perform_meta_analysis(discovered_paths)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return PathRAGResult(
                query=query,
                discovered_paths=discovered_paths,
                path_intersections=path_intersections,
                consensus_insights=consensus_insights,
                conflicting_signals=conflicting_signals,
                meta_analysis=meta_analysis,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ PathRAG query execution error: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return PathRAGResult(
                query=query,
                discovered_paths=[],
                path_intersections={},
                consensus_insights=[],
                conflicting_signals=[f"Query execution error: {e}"],
                meta_analysis={"error": str(e)},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    def find_path_intersections(self, paths: List[ReasoningPathAnalysis]) -> Dict[str, List[str]]:
        """Find where reasoning paths intersect (common entities)"""
        intersections = defaultdict(list)
        
        for i, path1 in enumerate(paths):
            path1_entities = set(hop.target_entity.id for hop in path1.hops)
            
            for j, path2 in enumerate(paths[i+1:], i+1):
                path2_entities = set(hop.target_entity.id for hop in path2.hops)
                common_entities = path1_entities & path2_entities
                
                if common_entities:
                    intersection_key = f"path_{i}_path_{j}"
                    intersections[intersection_key] = list(common_entities)
        
        return dict(intersections)
    
    def generate_consensus_insights(self, paths: List[ReasoningPathAnalysis]) -> List[str]:
        """Generate insights that are supported by multiple paths"""
        insights = []
        
        # Find entities that appear in multiple high-confidence paths
        entity_counts = defaultdict(int)
        high_conf_paths = [p for p in paths if p.total_confidence > 0.6]
        
        for path in high_conf_paths:
            for hop in path.hops:
                entity_counts[hop.target_entity.name] += 1
        
        # Generate consensus insights
        for entity_name, count in entity_counts.items():
            if count >= 2:
                insights.append(f"{entity_name} appears in {count} high-confidence reasoning paths")
        
        # Find common recommended actions
        action_counts = defaultdict(int)
        for path in high_conf_paths:
            for action in path.recommended_actions:
                action_counts[action] += 1
        
        for action, count in action_counts.items():
            if count >= 2:
                insights.append(f"Consensus recommendation: {action} (supported by {count} paths)")
        
        return insights[:5]  # Top 5 insights
    
    def find_conflicting_signals(self, paths: List[ReasoningPathAnalysis]) -> List[str]:
        """Find where paths provide conflicting signals"""
        conflicts = []
        
        # Check for opposite recommendations for same entity
        entity_recommendations = defaultdict(list)
        
        for path in paths:
            for action in path.recommended_actions:
                if "buy" in action.lower() or "sell" in action.lower():
                    # Extract entity name (simple heuristic)
                    for hop in path.hops:
                        if hop.target_entity.type == 'Stock':
                            entity_recommendations[hop.target_entity.name].append(action.lower())
        
        for entity, actions in entity_recommendations.items():
            buy_signals = sum(1 for a in actions if "buy" in a)
            sell_signals = sum(1 for a in actions if "sell" in a or "avoid" in a)
            
            if buy_signals > 0 and sell_signals > 0:
                conflicts.append(f"Conflicting signals for {entity}: {buy_signals} buy vs {sell_signals} sell/avoid")
        
        return conflicts
    
    def perform_meta_analysis(self, paths: List[ReasoningPathAnalysis]) -> Dict[str, Any]:
        """Perform meta-analysis of the discovered paths"""
        if not paths:
            return {"error": "No paths to analyze"}
        
        # Calculate statistics
        avg_confidence = np.mean([p.total_confidence for p in paths])
        avg_path_length = np.mean([p.path_length for p in paths])
        avg_actionability = np.mean([p.actionability_score for p in paths])
        
        # Path type distribution
        path_type_counts = defaultdict(int)
        for path in paths:
            path_type_counts[path.path_type.value] += 1
        
        # Reasoning type distribution
        reasoning_type_counts = defaultdict(int)
        for path in paths:
            reasoning_type_counts[path.reasoning_type.value] += 1
        
        # Time sensitivity distribution
        time_sensitivity_counts = defaultdict(int)
        for path in paths:
            time_sensitivity_counts[path.time_sensitivity] += 1
        
        return {
            "total_paths_found": len(paths),
            "average_confidence": round(avg_confidence, 3),
            "average_path_length": round(avg_path_length, 1),
            "average_actionability": round(avg_actionability, 3),
            "path_type_distribution": dict(path_type_counts),
            "reasoning_type_distribution": dict(reasoning_type_counts),
            "time_sensitivity_distribution": dict(time_sensitivity_counts),
            "high_confidence_paths": len([p for p in paths if p.total_confidence > 0.7]),
            "actionable_paths": len([p for p in paths if p.actionability_score > 0.6]),
            "risky_paths": len([p for p in paths if p.risk_score > 0.6])
        }


# Global PathRAG instance
path_rag_engine = None


def initialize_path_rag(graph_rag_engine):
    """Initialize PathRAG with a GraphRAG engine"""
    global path_rag_engine
    path_rag_engine = AdvancedPathRAG(graph_rag_engine)
    return path_rag_engine


async def analyze_stock_paths(symbol: str, max_hops: int = 4) -> PathRAGResult:
    """Analyze reasoning paths for a specific stock"""
    if not path_rag_engine:
        raise ValueError("PathRAG engine not initialized")
    
    query = PathRAGQuery(
        query_text=f"Analyze reasoning paths for {symbol}",
        start_entity_ids=[f"STOCK_{symbol}"],
        max_hops=max_hops,
        path_types=[PathType.CAUSAL, PathType.CORRELATIONAL, PathType.HIERARCHICAL],
        min_confidence=0.3,
        max_paths=10
    )
    
    return await path_rag_engine.execute_path_rag_query(query)


if __name__ == "__main__":
    # Test PathRAG (would need GraphRAG engine)
    print("🛤️ PathRAG Engine initialized")
    print("Multi-hop reasoning paths for financial knowledge graphs")