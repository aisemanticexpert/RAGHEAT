"""
Market Knowledge Graph Ontology and Reasoning Engine
Advanced semantic understanding system for financial markets

This system implements:
1. Comprehensive market ontology with entities and relationships
2. Semantic reasoning engine for market inference
3. Entity linking and relationship extraction
4. Temporal reasoning for market dynamics
5. Causal reasoning for market events
6. Multi-modal knowledge integration
7. Automated knowledge discovery

Goal: Achieve 1000% returns through intelligent semantic understanding
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import warnings
import logging
from pathlib import Path
import json
import pickle
from collections import defaultdict, deque
from enum import Enum
import networkx as nx
from abc import ABC, abstractmethod

# Graph and reasoning libraries
try:
    import rdflib
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef, BNode
    from rdflib.plugins.sparql import prepareQuery
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False

# NLP libraries for knowledge extraction
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

# Machine learning for pattern discovery
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Import our components
try:
    from ..hierarchical_analysis.sector_stock_analyzer import HierarchicalSectorStockAnalyzer
    from ..machine_learning.advanced_heat_predictor import AdvancedHeatPredictor
    from ..heat_propagation.viral_heat_engine import ViralHeatEngine
    from ...config.sector_stocks import SECTOR_STOCKS, get_sector_for_stock, get_all_stocks
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define market ontology namespaces
if HAS_RDFLIB:
    MARKET_NS = Namespace("http://ragheat.ai/ontology/market#")
    ENTITY_NS = Namespace("http://ragheat.ai/entities/")
    RELATIONSHIP_NS = Namespace("http://ragheat.ai/relationships/")
    EVENT_NS = Namespace("http://ragheat.ai/events/")
    TEMPORAL_NS = Namespace("http://ragheat.ai/temporal/")
else:
    # Fallback for when RDFLib is not available
    MARKET_NS = None
    ENTITY_NS = None
    RELATIONSHIP_NS = None
    EVENT_NS = None
    TEMPORAL_NS = None

class EntityType(Enum):
    """Types of entities in market ontology"""
    STOCK = "Stock"
    SECTOR = "Sector"
    INDUSTRY = "Industry"
    COMPANY = "Company"
    MARKET = "Market"
    ECONOMIC_INDICATOR = "EconomicIndicator"
    NEWS_EVENT = "NewsEvent"
    TRADING_SIGNAL = "TradingSignal"
    PATTERN = "Pattern"
    STRATEGY = "Strategy"

class RelationshipType(Enum):
    """Types of relationships in market ontology"""
    BELONGS_TO = "belongsTo"
    CORRELATED_WITH = "correlatedWith"
    INFLUENCES = "influences"
    DEPENDS_ON = "dependsOn"
    COMPETES_WITH = "competesWith"
    SUPPLIES = "supplies"
    PART_OF = "partOf"
    CAUSES = "causes"
    PREDICTS = "predicts"
    SIMILAR_TO = "similarTo"

class TemporalType(Enum):
    """Types of temporal relationships"""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    STARTS = "starts"
    FINISHES = "finishes"

@dataclass
class MarketEntity:
    """Represents an entity in the market knowledge graph"""
    entity_id: str
    entity_type: EntityType
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Semantic attributes
    synonyms: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketRelationship:
    """Represents a relationship between entities"""
    relationship_id: str
    source_entity: str
    target_entity: str
    relationship_type: RelationshipType
    weight: float = 1.0
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    temporal_info: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class KnowledgeRule:
    """Represents a reasoning rule in the knowledge graph"""
    rule_id: str
    name: str
    condition: str  # SPARQL-like condition
    conclusion: str  # What to infer
    confidence: float = 1.0
    priority: int = 1
    active: bool = True

@dataclass
class InferenceResult:
    """Result of reasoning inference"""
    inference_id: str
    rule_applied: str
    new_facts: List[Dict[str, Any]]
    confidence: float
    reasoning_chain: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class SemanticReasoner(ABC):
    """Abstract base class for semantic reasoning"""
    
    @abstractmethod
    def infer(self, knowledge_graph: 'MarketKnowledgeGraph') -> List[InferenceResult]:
        pass

class CorrelationReasoner(SemanticReasoner):
    """Reasoning based on correlation patterns"""
    
    def __init__(self, correlation_threshold: float = 0.7):
        self.correlation_threshold = correlation_threshold
    
    def infer(self, knowledge_graph: 'MarketKnowledgeGraph') -> List[InferenceResult]:
        """Infer new relationships based on correlation patterns"""
        inferences = []
        
        # Find stocks with high correlation
        correlation_patterns = knowledge_graph.find_correlation_patterns(self.correlation_threshold)
        
        for pattern in correlation_patterns:
            stock_a, stock_b, correlation = pattern
            
            # If both stocks are in same sector, infer stronger sector coherence
            sector_a = get_sector_for_stock(stock_a)
            sector_b = get_sector_for_stock(stock_b)
            
            if sector_a == sector_b and correlation > 0.8:
                inference = InferenceResult(
                    inference_id=f"corr_{stock_a}_{stock_b}_{datetime.now().timestamp()}",
                    rule_applied="high_correlation_same_sector",
                    new_facts=[{
                        'type': 'relationship',
                        'source': sector_a,
                        'target': 'coherent_sector',
                        'relationship': 'has_property',
                        'confidence': correlation
                    }],
                    confidence=correlation,
                    reasoning_chain=[
                        f"{stock_a} and {stock_b} have correlation {correlation:.3f}",
                        f"Both belong to sector {sector_a}",
                        f"High correlation implies sector coherence"
                    ]
                )
                inferences.append(inference)
        
        return inferences

class CausalReasoner(SemanticReasoner):
    """Reasoning based on causal relationships"""
    
    def __init__(self):
        self.causal_patterns = {
            'supply_chain': {
                'condition': 'supplies',
                'effect': 'influences_performance',
                'lag': timedelta(days=1)
            },
            'sector_momentum': {
                'condition': 'belongs_to_sector',
                'effect': 'momentum_spillover',
                'lag': timedelta(days=2)
            }
        }
    
    def infer(self, knowledge_graph: 'MarketKnowledgeGraph') -> List[InferenceResult]:
        """Infer causal relationships"""
        inferences = []
        
        # Check supply chain causality
        supply_relationships = knowledge_graph.get_relationships_by_type(RelationshipType.SUPPLIES)
        
        for rel in supply_relationships:
            # If supplier performance changes, expect effect on customer
            inference = InferenceResult(
                inference_id=f"causal_{rel.source_entity}_{rel.target_entity}_{datetime.now().timestamp()}",
                rule_applied="supply_chain_causality",
                new_facts=[{
                    'type': 'causal_relationship',
                    'cause': rel.source_entity,
                    'effect': rel.target_entity,
                    'mechanism': 'supply_chain_dependency',
                    'expected_lag': '1_day'
                }],
                confidence=0.7,
                reasoning_chain=[
                    f"{rel.source_entity} supplies to {rel.target_entity}",
                    "Supply chain relationships create causal dependencies",
                    "Performance changes propagate with ~1 day lag"
                ]
            )
            inferences.append(inference)
        
        return inferences

class TemporalReasoner(SemanticReasoner):
    """Reasoning about temporal patterns and sequences"""
    
    def __init__(self):
        self.temporal_patterns = {}
    
    def infer(self, knowledge_graph: 'MarketKnowledgeGraph') -> List[InferenceResult]:
        """Infer temporal relationships and patterns"""
        inferences = []
        
        # Find recurring temporal patterns
        temporal_events = knowledge_graph.get_temporal_events()
        
        for event_type, events in temporal_events.items():
            if len(events) >= 3:  # Need minimum events for pattern
                # Look for seasonal patterns
                monthly_distribution = defaultdict(int)
                for event in events:
                    month = event['timestamp'].month
                    monthly_distribution[month] += 1
                
                # Find peak months
                max_count = max(monthly_distribution.values())
                peak_months = [month for month, count in monthly_distribution.items() 
                             if count == max_count]
                
                if max_count >= 2:  # Significant pattern
                    inference = InferenceResult(
                        inference_id=f"temporal_{event_type}_{datetime.now().timestamp()}",
                        rule_applied="seasonal_pattern_detection",
                        new_facts=[{
                            'type': 'temporal_pattern',
                            'event_type': event_type,
                            'peak_months': peak_months,
                            'frequency': max_count,
                            'pattern_type': 'seasonal'
                        }],
                        confidence=min(max_count / len(events), 1.0),
                        reasoning_chain=[
                            f"Found {len(events)} instances of {event_type}",
                            f"Peak activity in months: {peak_months}",
                            "Seasonal pattern detected"
                        ]
                    )
                    inferences.append(inference)
        
        return inferences

class MarketKnowledgeGraph:
    """Comprehensive market knowledge graph with reasoning capabilities"""
    
    def __init__(self):
        # Core graph storage
        self.entities: Dict[str, MarketEntity] = {}
        self.relationships: Dict[str, MarketRelationship] = {}
        self.networkx_graph = nx.MultiDiGraph()
        
        # RDF graph for semantic reasoning
        if HAS_RDFLIB:
            self.rdf_graph = Graph()
            self._initialize_ontology()
        
        # Reasoning components
        self.reasoners: List[SemanticReasoner] = [
            CorrelationReasoner(),
            CausalReasoner(),
            TemporalReasoner()
        ]
        
        self.knowledge_rules: Dict[str, KnowledgeRule] = {}
        self.inference_cache: Dict[str, List[InferenceResult]] = {}
        
        # Embeddings for semantic similarity
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.relationship_embeddings: Dict[str, np.ndarray] = {}
        
        # Temporal storage
        self.temporal_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Performance tracking
        self.reasoning_stats = {
            'total_inferences': 0,
            'successful_predictions': 0,
            'reasoning_time': 0.0
        }
        
        logger.info("Market Knowledge Graph initialized")
    
    def _initialize_ontology(self):
        """Initialize basic market ontology in RDF"""
        if not HAS_RDFLIB:
            return
        
        # Define basic classes
        self.rdf_graph.add((MARKET_NS.Stock, RDF.type, OWL.Class))
        self.rdf_graph.add((MARKET_NS.Sector, RDF.type, OWL.Class))
        self.rdf_graph.add((MARKET_NS.Company, RDF.type, OWL.Class))
        self.rdf_graph.add((MARKET_NS.Market, RDF.type, OWL.Class))
        
        # Define properties
        self.rdf_graph.add((MARKET_NS.belongsTo, RDF.type, OWL.ObjectProperty))
        self.rdf_graph.add((MARKET_NS.correlatedWith, RDF.type, OWL.ObjectProperty))
        self.rdf_graph.add((MARKET_NS.influences, RDF.type, OWL.ObjectProperty))
        
        # Define hierarchies
        self.rdf_graph.add((MARKET_NS.Stock, RDFS.subClassOf, MARKET_NS.FinancialInstrument))
        self.rdf_graph.add((MARKET_NS.Sector, RDFS.subClassOf, MARKET_NS.MarketSegment))
    
    def add_entity(self, entity: MarketEntity) -> bool:
        """Add entity to knowledge graph"""
        try:
            self.entities[entity.entity_id] = entity
            
            # Add to NetworkX graph
            self.networkx_graph.add_node(
                entity.entity_id,
                type=entity.entity_type.value,
                name=entity.name,
                **entity.attributes
            )
            
            # Add to RDF graph
            if HAS_RDFLIB:
                entity_uri = ENTITY_NS[entity.entity_id]
                entity_class = MARKET_NS[entity.entity_type.value]
                
                self.rdf_graph.add((entity_uri, RDF.type, entity_class))
                self.rdf_graph.add((entity_uri, RDFS.label, Literal(entity.name)))
                
                # Add attributes as properties
                for attr_name, attr_value in entity.attributes.items():
                    prop_uri = MARKET_NS[attr_name]
                    self.rdf_graph.add((entity_uri, prop_uri, Literal(attr_value)))
            
            logger.debug(f"Added entity: {entity.entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding entity {entity.entity_id}: {str(e)}")
            return False
    
    def add_relationship(self, relationship: MarketRelationship) -> bool:
        """Add relationship to knowledge graph"""
        try:
            self.relationships[relationship.relationship_id] = relationship
            
            # Add to NetworkX graph
            self.networkx_graph.add_edge(
                relationship.source_entity,
                relationship.target_entity,
                key=relationship.relationship_id,
                type=relationship.relationship_type.value,
                weight=relationship.weight,
                confidence=relationship.confidence,
                **relationship.properties
            )
            
            # Add to RDF graph
            if HAS_RDFLIB:
                source_uri = ENTITY_NS[relationship.source_entity]
                target_uri = ENTITY_NS[relationship.target_entity]
                relation_uri = MARKET_NS[relationship.relationship_type.value]
                
                self.rdf_graph.add((source_uri, relation_uri, target_uri))
            
            logger.debug(f"Added relationship: {relationship.relationship_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding relationship {relationship.relationship_id}: {str(e)}")
            return False
    
    def populate_market_entities(self):
        """Populate knowledge graph with market entities"""
        logger.info("Populating market entities...")
        
        # Add market entity
        market_entity = MarketEntity(
            entity_id="GLOBAL_MARKET",
            entity_type=EntityType.MARKET,
            name="Global Stock Market",
            attributes={'description': 'Global financial market system'}
        )
        self.add_entity(market_entity)
        
        # Add sector entities
        for sector_key, sector_data in SECTOR_STOCKS.items():
            sector_entity = MarketEntity(
                entity_id=sector_key,
                entity_type=EntityType.SECTOR,
                name=sector_data['sector_name'],
                attributes={
                    'color': sector_data['color'],
                    'stock_count': len(sector_data['all_stocks'])
                },
                categories=['financial_sector']
            )
            self.add_entity(sector_entity)
            
            # Add relationship to market
            market_relationship = MarketRelationship(
                relationship_id=f"market_{sector_key}",
                source_entity="GLOBAL_MARKET",
                target_entity=sector_key,
                relationship_type=RelationshipType.PART_OF,
                weight=1.0
            )
            self.add_relationship(market_relationship)
        
        # Add stock entities
        all_stocks = get_all_stocks()
        for stock in all_stocks:
            sector = get_sector_for_stock(stock)
            
            stock_entity = MarketEntity(
                entity_id=stock,
                entity_type=EntityType.STOCK,
                name=stock,
                attributes={
                    'sector': sector,
                    'exchange': 'NASDAQ'  # Simplified
                },
                categories=['equity', 'tradeable']
            )
            self.add_entity(stock_entity)
            
            # Add relationship to sector
            sector_relationship = MarketRelationship(
                relationship_id=f"{stock}_{sector}",
                source_entity=stock,
                target_entity=sector,
                relationship_type=RelationshipType.BELONGS_TO,
                weight=1.0
            )
            self.add_relationship(sector_relationship)
        
        logger.info(f"Populated {len(self.entities)} entities and {len(self.relationships)} relationships")
    
    def add_correlation_relationships(self, correlation_matrix: pd.DataFrame, threshold: float = 0.3):
        """Add correlation relationships from correlation matrix"""
        for i, stock_a in enumerate(correlation_matrix.index):
            for j, stock_b in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    correlation = correlation_matrix.iloc[i, j]
                    
                    if abs(correlation) > threshold:
                        relationship = MarketRelationship(
                            relationship_id=f"corr_{stock_a}_{stock_b}",
                            source_entity=stock_a,
                            target_entity=stock_b,
                            relationship_type=RelationshipType.CORRELATED_WITH,
                            weight=abs(correlation),
                            confidence=min(abs(correlation), 1.0),
                            properties={
                                'correlation_value': correlation,
                                'correlation_type': 'positive' if correlation > 0 else 'negative'
                            }
                        )
                        self.add_relationship(relationship)
    
    def add_supply_chain_relationships(self):
        """Add known supply chain relationships"""
        supply_chains = {
            'AAPL': ['NVDA', 'QCOM', 'AVGO'],  # Apple suppliers
            'TSLA': ['NVDA', 'AMD'],           # Tesla AI chips
            'AMZN': ['UPS', 'FDX'],            # Amazon logistics
            'GOOGL': ['NVDA', 'AMD']           # Google cloud infrastructure
        }
        
        for customer, suppliers in supply_chains.items():
            for supplier in suppliers:
                if customer in self.entities and supplier in self.entities:
                    relationship = MarketRelationship(
                        relationship_id=f"supply_{supplier}_{customer}",
                        source_entity=supplier,
                        target_entity=customer,
                        relationship_type=RelationshipType.SUPPLIES,
                        weight=0.8,
                        confidence=0.9,
                        properties={'relationship_strength': 'strong'}
                    )
                    self.add_relationship(relationship)
    
    def add_competitive_relationships(self):
        """Add competitive relationships within sectors"""
        competitive_groups = {
            'Technology': [
                ['AAPL', 'GOOGL', 'MSFT'],      # Big Tech
                ['NVDA', 'AMD', 'INTC'],        # Semiconductors
                ['META', 'GOOGL', 'NFLX']       # Digital platforms
            ],
            'Healthcare': [
                ['JNJ', 'PFE', 'MRK'],          # Pharmaceuticals
                ['UNH', 'ANTM', 'CVS']          # Health insurance
            ],
            'Financial': [
                ['JPM', 'BAC', 'WFC', 'C'],     # Banks
                ['BLK', 'MS', 'GS']             # Investment banks
            ]
        }
        
        for sector, groups in competitive_groups.items():
            for group in groups:
                for i, company_a in enumerate(group):
                    for j, company_b in enumerate(group):
                        if i < j and company_a in self.entities and company_b in self.entities:
                            relationship = MarketRelationship(
                                relationship_id=f"compete_{company_a}_{company_b}",
                                source_entity=company_a,
                                target_entity=company_b,
                                relationship_type=RelationshipType.COMPETES_WITH,
                                weight=0.7,
                                confidence=0.8,
                                properties={'competition_intensity': 'high'}
                            )
                            self.add_relationship(relationship)
    
    def find_correlation_patterns(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find high correlation patterns"""
        patterns = []
        
        corr_relationships = self.get_relationships_by_type(RelationshipType.CORRELATED_WITH)
        
        for rel in corr_relationships:
            if rel.weight > threshold:
                correlation_value = rel.properties.get('correlation_value', rel.weight)
                patterns.append((rel.source_entity, rel.target_entity, correlation_value))
        
        return patterns
    
    def get_relationships_by_type(self, relationship_type: RelationshipType) -> List[MarketRelationship]:
        """Get all relationships of a specific type"""
        return [rel for rel in self.relationships.values() 
                if rel.relationship_type == relationship_type]
    
    def get_entity_neighbors(self, entity_id: str, relationship_types: Optional[List[RelationshipType]] = None) -> List[str]:
        """Get neighboring entities"""
        neighbors = []
        
        for rel in self.relationships.values():
            if relationship_types and rel.relationship_type not in relationship_types:
                continue
                
            if rel.source_entity == entity_id:
                neighbors.append(rel.target_entity)
            elif rel.target_entity == entity_id:
                neighbors.append(rel.source_entity)
        
        return list(set(neighbors))
    
    def calculate_entity_centrality(self) -> Dict[str, float]:
        """Calculate centrality measures for entities"""
        centrality_measures = {}
        
        # PageRank centrality
        pagerank = nx.pagerank(self.networkx_graph)
        
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(self.networkx_graph)
        
        # Combine measures
        for entity_id in self.entities:
            centrality_measures[entity_id] = {
                'pagerank': pagerank.get(entity_id, 0.0),
                'betweenness': betweenness.get(entity_id, 0.0),
                'degree': self.networkx_graph.degree(entity_id) if entity_id in self.networkx_graph else 0
            }
        
        return centrality_measures
    
    def perform_reasoning(self) -> List[InferenceResult]:
        """Perform comprehensive reasoning using all reasoners"""
        logger.info("Performing knowledge graph reasoning...")
        
        all_inferences = []
        
        for reasoner in self.reasoners:
            try:
                reasoner_inferences = reasoner.infer(self)
                all_inferences.extend(reasoner_inferences)
                logger.debug(f"{reasoner.__class__.__name__} generated {len(reasoner_inferences)} inferences")
                
            except Exception as e:
                logger.error(f"Error in {reasoner.__class__.__name__}: {str(e)}")
                continue
        
        # Store inferences
        cache_key = datetime.now().strftime("%Y-%m-%d")
        self.inference_cache[cache_key] = all_inferences
        
        # Update statistics
        self.reasoning_stats['total_inferences'] += len(all_inferences)
        
        logger.info(f"Generated {len(all_inferences)} total inferences")
        return all_inferences
    
    def query_sparql(self, sparql_query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL query on RDF graph"""
        if not HAS_RDFLIB:
            logger.warning("RDFLib not available for SPARQL queries")
            return []
        
        try:
            results = []
            query_result = self.rdf_graph.query(sparql_query)
            
            for row in query_result:
                result_dict = {}
                for i, var in enumerate(query_result.vars):
                    result_dict[str(var)] = str(row[i])
                results.append(result_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"SPARQL query error: {str(e)}")
            return []
    
    def find_similar_entities(self, entity_id: str, similarity_threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find entities similar to given entity"""
        if entity_id not in self.entity_embeddings:
            return []
        
        target_embedding = self.entity_embeddings[entity_id]
        similarities = []
        
        for other_id, other_embedding in self.entity_embeddings.items():
            if other_id != entity_id:
                similarity = cosine_similarity(
                    target_embedding.reshape(1, -1),
                    other_embedding.reshape(1, -1)
                )[0, 0]
                
                if similarity > similarity_threshold:
                    similarities.append((other_id, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def get_temporal_events(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get temporal events for pattern analysis"""
        return dict(self.temporal_events)
    
    def add_temporal_event(self, event_type: str, event_data: Dict[str, Any]):
        """Add temporal event"""
        event_data['timestamp'] = event_data.get('timestamp', datetime.now())
        self.temporal_events[event_type].append(event_data)
    
    def predict_entity_properties(self, entity_id: str) -> Dict[str, Any]:
        """Predict properties of entity based on similar entities"""
        similar_entities = self.find_similar_entities(entity_id)
        
        if not similar_entities:
            return {}
        
        # Aggregate properties from similar entities
        property_predictions = defaultdict(list)
        
        for similar_id, similarity in similar_entities:
            if similar_id in self.entities:
                similar_entity = self.entities[similar_id]
                for prop_name, prop_value in similar_entity.properties.items():
                    if isinstance(prop_value, (int, float)):
                        property_predictions[prop_name].append(prop_value * similarity)
        
        # Calculate weighted averages
        predictions = {}
        for prop_name, values in property_predictions.items():
            if values:
                predictions[f"predicted_{prop_name}"] = np.mean(values)
                predictions[f"confidence_{prop_name}"] = min(len(values) / 5, 1.0)
        
        return predictions
    
    def explain_relationship(self, source_entity: str, target_entity: str) -> Dict[str, Any]:
        """Explain why two entities are related"""
        explanation = {
            'direct_relationships': [],
            'indirect_paths': [],
            'shared_properties': [],
            'reasoning_chain': []
        }
        
        # Find direct relationships
        for rel in self.relationships.values():
            if ((rel.source_entity == source_entity and rel.target_entity == target_entity) or
                (rel.source_entity == target_entity and rel.target_entity == source_entity)):
                explanation['direct_relationships'].append({
                    'type': rel.relationship_type.value,
                    'weight': rel.weight,
                    'confidence': rel.confidence
                })
        
        # Find indirect paths
        try:
            if source_entity in self.networkx_graph and target_entity in self.networkx_graph:
                try:
                    shortest_path = nx.shortest_path(self.networkx_graph, source_entity, target_entity)
                    if len(shortest_path) <= 4:  # Limit path length
                        explanation['indirect_paths'].append(shortest_path)
                except nx.NetworkXNoPath:
                    pass
        except Exception:
            pass
        
        # Find shared properties
        if source_entity in self.entities and target_entity in self.entities:
            source_props = set(self.entities[source_entity].categories)
            target_props = set(self.entities[target_entity].categories)
            shared = source_props.intersection(target_props)
            explanation['shared_properties'] = list(shared)
        
        # Generate reasoning chain
        if explanation['direct_relationships']:
            explanation['reasoning_chain'].append(f"Direct relationship exists: {explanation['direct_relationships'][0]['type']}")
        
        if explanation['shared_properties']:
            explanation['reasoning_chain'].append(f"Share common properties: {', '.join(explanation['shared_properties'])}")
        
        if explanation['indirect_paths']:
            path_length = len(explanation['indirect_paths'][0]) - 1
            explanation['reasoning_chain'].append(f"Connected through {path_length}-step path")
        
        return explanation
    
    def generate_investment_insights(self, entity_id: str) -> Dict[str, Any]:
        """Generate investment insights for an entity"""
        insights = {
            'entity_id': entity_id,
            'centrality_score': 0.0,
            'influence_score': 0.0,
            'risk_factors': [],
            'opportunity_factors': [],
            'recommendations': []
        }
        
        if entity_id not in self.entities:
            return insights
        
        # Calculate centrality
        centrality_measures = self.calculate_entity_centrality()
        if entity_id in centrality_measures:
            insights['centrality_score'] = centrality_measures[entity_id]['pagerank']
        
        # Analyze relationships
        influential_relationships = 0
        risky_relationships = 0
        
        for rel in self.relationships.values():
            if rel.source_entity == entity_id or rel.target_entity == entity_id:
                if rel.relationship_type == RelationshipType.INFLUENCES:
                    influential_relationships += 1
                elif rel.relationship_type in [RelationshipType.DEPENDS_ON, RelationshipType.COMPETES_WITH]:
                    risky_relationships += 1
        
        insights['influence_score'] = influential_relationships / max(len(self.get_entity_neighbors(entity_id)), 1)
        
        # Generate risk factors
        if risky_relationships > 3:
            insights['risk_factors'].append("High dependency on other entities")
        
        competitors = self.get_entity_neighbors(entity_id, [RelationshipType.COMPETES_WITH])
        if len(competitors) > 5:
            insights['risk_factors'].append("Intense competitive environment")
        
        # Generate opportunity factors
        suppliers = self.get_entity_neighbors(entity_id, [RelationshipType.SUPPLIES])
        if len(suppliers) > 3:
            insights['opportunity_factors'].append("Strong supplier network")
        
        if insights['centrality_score'] > 0.01:  # High centrality
            insights['opportunity_factors'].append("High network influence")
        
        # Generate recommendations
        if insights['influence_score'] > 0.5:
            insights['recommendations'].append("Consider for momentum strategies")
        
        if len(insights['risk_factors']) > len(insights['opportunity_factors']):
            insights['recommendations'].append("Exercise caution - high risk profile")
        else:
            insights['recommendations'].append("Positive risk-reward profile")
        
        return insights

class MarketOntologyEngine:
    """Main engine for market ontology and reasoning"""
    
    def __init__(self):
        self.knowledge_graph = MarketKnowledgeGraph()
        self.initialized = False
        
        # External integrations
        self.hierarchical_analyzer = None
        self.heat_predictor = None
        self.viral_engine = None
        
        logger.info("Market Ontology Engine initialized")
    
    async def initialize(self):
        """Initialize the ontology engine with market data"""
        logger.info("Initializing Market Ontology Engine...")
        
        # Populate basic market structure
        self.knowledge_graph.populate_market_entities()
        
        # Add domain-specific relationships
        self.knowledge_graph.add_supply_chain_relationships()
        self.knowledge_graph.add_competitive_relationships()
        
        # Initialize external components
        self.hierarchical_analyzer = HierarchicalSectorStockAnalyzer()
        self.heat_predictor = AdvancedHeatPredictor()
        self.viral_engine = ViralHeatEngine()
        
        self.initialized = True
        logger.info("Market Ontology Engine initialization complete")
    
    async def update_with_market_data(self, market_data: Dict[str, pd.DataFrame]):
        """Update knowledge graph with latest market data"""
        if not self.initialized:
            await self.initialize()
        
        # Calculate correlations and add to knowledge graph
        if len(market_data) > 1:
            returns_data = {}
            for symbol, df in market_data.items():
                if 'Close' in df.columns and len(df) > 20:
                    returns_data[symbol] = df['Close'].pct_change().dropna()
            
            if len(returns_data) > 1:
                aligned_data = pd.DataFrame(returns_data).dropna()
                correlation_matrix = aligned_data.corr()
                self.knowledge_graph.add_correlation_relationships(correlation_matrix)
        
        # Add temporal events
        for symbol, df in market_data.items():
            if 'Close' in df.columns and len(df) > 0:
                latest_price = df['Close'].iloc[-1]
                latest_date = df.index[-1] if hasattr(df.index, 'date') else datetime.now()
                
                self.knowledge_graph.add_temporal_event('price_update', {
                    'symbol': symbol,
                    'price': latest_price,
                    'timestamp': latest_date
                })
        
        logger.info("Knowledge graph updated with market data")
    
    async def perform_comprehensive_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Perform comprehensive analysis combining all reasoning capabilities"""
        logger.info("Performing comprehensive ontological analysis...")
        
        # Perform reasoning
        inferences = self.knowledge_graph.perform_reasoning()
        
        # Generate insights for each symbol
        symbol_insights = {}
        for symbol in symbols:
            insights = self.knowledge_graph.generate_investment_insights(symbol)
            symbol_insights[symbol] = insights
        
        # Find network patterns
        centrality_measures = self.knowledge_graph.calculate_entity_centrality()
        
        # Identify key influencers
        key_influencers = sorted(
            centrality_measures.items(),
            key=lambda x: x[1]['pagerank'],
            reverse=True
        )[:10]
        
        # Generate relationship explanations for top pairs
        relationship_explanations = {}
        for i, symbol1 in enumerate(symbols[:5]):
            for symbol2 in symbols[i+1:i+3]:  # Limit combinations
                explanation = self.knowledge_graph.explain_relationship(symbol1, symbol2)
                if explanation['direct_relationships'] or explanation['indirect_paths']:
                    relationship_explanations[f"{symbol1}_{symbol2}"] = explanation
        
        return {
            'inferences': inferences,
            'symbol_insights': symbol_insights,
            'key_influencers': key_influencers,
            'relationship_explanations': relationship_explanations,
            'network_statistics': {
                'total_entities': len(self.knowledge_graph.entities),
                'total_relationships': len(self.knowledge_graph.relationships),
                'reasoning_stats': self.knowledge_graph.reasoning_stats
            },
            'timestamp': datetime.now()
        }
    
    def query_knowledge(self, query_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge graph with different query types"""
        if query_type == "find_similar":
            entity_id = parameters.get('entity_id')
            threshold = parameters.get('threshold', 0.8)
            similar = self.knowledge_graph.find_similar_entities(entity_id, threshold)
            return {'similar_entities': similar}
        
        elif query_type == "explain_relationship":
            source = parameters.get('source')
            target = parameters.get('target')
            explanation = self.knowledge_graph.explain_relationship(source, target)
            return {'explanation': explanation}
        
        elif query_type == "predict_properties":
            entity_id = parameters.get('entity_id')
            predictions = self.knowledge_graph.predict_entity_properties(entity_id)
            return {'predictions': predictions}
        
        elif query_type == "sparql":
            sparql_query = parameters.get('query')
            results = self.knowledge_graph.query_sparql(sparql_query)
            return {'sparql_results': results}
        
        else:
            return {'error': f'Unknown query type: {query_type}'}
    
    def export_knowledge_graph(self, format: str = 'json') -> str:
        """Export knowledge graph in various formats"""
        if format == 'json':
            export_data = {
                'entities': {
                    eid: {
                        'type': entity.entity_type.value,
                        'name': entity.name,
                        'attributes': entity.attributes,
                        'categories': entity.categories
                    }
                    for eid, entity in self.knowledge_graph.entities.items()
                },
                'relationships': {
                    rid: {
                        'source': rel.source_entity,
                        'target': rel.target_entity,
                        'type': rel.relationship_type.value,
                        'weight': rel.weight,
                        'confidence': rel.confidence
                    }
                    for rid, rel in self.knowledge_graph.relationships.items()
                }
            }
            return json.dumps(export_data, indent=2)
        
        elif format == 'rdf' and HAS_RDFLIB:
            return self.knowledge_graph.rdf_graph.serialize(format='turtle')
        
        else:
            return "Unsupported format"

# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Test the market ontology engine
        engine = MarketOntologyEngine()
        
        print("Initializing Market Ontology Engine...")
        await engine.initialize()
        
        print(f"Knowledge Graph Statistics:")
        print(f"- Entities: {len(engine.knowledge_graph.entities)}")
        print(f"- Relationships: {len(engine.knowledge_graph.relationships)}")
        
        # Test reasoning
        print("\nPerforming reasoning...")
        test_symbols = ['AAPL', 'TSLA', 'GOOGL', 'NVDA', 'META']
        
        analysis = await engine.perform_comprehensive_analysis(test_symbols)
        
        print(f"\nAnalysis Results:")
        print(f"- Generated {len(analysis['inferences'])} inferences")
        print(f"- Analyzed {len(analysis['symbol_insights'])} symbols")
        print(f"- Found {len(analysis['key_influencers'])} key influencers")
        
        # Display top influencers
        print("\nTop 5 Network Influencers:")
        for i, (entity, metrics) in enumerate(analysis['key_influencers'][:5]):
            print(f"{i+1}. {entity}: PageRank = {metrics['pagerank']:.4f}")
        
        # Test specific queries
        print("\nTesting knowledge queries...")
        
        # Find similar entities
        similar_query = engine.query_knowledge('find_similar', {'entity_id': 'AAPL', 'threshold': 0.1})
        print(f"Entities similar to AAPL: {len(similar_query.get('similar_entities', []))}")
        
        # Explain relationship
        rel_query = engine.query_knowledge('explain_relationship', {'source': 'AAPL', 'target': 'Technology'})
        explanation = rel_query.get('explanation', {})
        print(f"AAPL-Technology relationship: {len(explanation.get('direct_relationships', []))} direct connections")
        
        print("\nMarket Ontology Engine test completed successfully!")
    
    # Run test
    import asyncio
    asyncio.run(main())