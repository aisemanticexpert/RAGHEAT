"""
Advanced GraphRAG Engine for Knowledge Graph Reasoning
Implements retrieval-augmented generation with graph context for intelligent stock analysis
"""

import asyncio
import json
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import openai
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    SENTIMENT = "sentiment"
    HEAT_DIFFUSION = "heat_diffusion"
    CORRELATION = "correlation"
    SECTOR_ROTATION = "sector_rotation"


@dataclass
class GraphEntity:
    """Entity in the knowledge graph"""
    id: str
    type: str  # Market, Sector, Stock, Indicator, News, etc.
    name: str
    properties: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    last_updated: Optional[datetime] = None


@dataclass
class GraphRelationship:
    """Relationship in the knowledge graph"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]
    weight: float
    confidence: float
    last_updated: Optional[datetime] = None


@dataclass
class ReasoningPath:
    """A reasoning path through the knowledge graph"""
    entities: List[GraphEntity]
    relationships: List[GraphRelationship]
    reasoning_type: ReasoningType
    confidence: float
    evidence: List[str]
    conclusion: str
    path_length: int


@dataclass 
class GraphRAGResult:
    """Result from GraphRAG query"""
    query: str
    primary_entity: GraphEntity
    reasoning_paths: List[ReasoningPath]
    contextual_entities: List[GraphEntity]
    llm_analysis: str
    confidence: float
    supporting_evidence: List[str]
    risk_factors: List[str]
    timestamp: datetime


class AdvancedGraphRAG:
    """
    Advanced Graph Retrieval-Augmented Generation Engine
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.graph = nx.MultiDiGraph()
        self.entities = {}  # id -> GraphEntity
        self.relationships = {}  # (source, target, type) -> GraphRelationship
        self.entity_embeddings = {}  # id -> embedding
        self.openai_client = None
        
        if openai_api_key:
            openai.api_key = openai_api_key
            self.openai_client = openai
        
        # Initialize with domain knowledge
        self.initialize_financial_ontology()
        
        # Reasoning templates
        self.reasoning_templates = {
            ReasoningType.FUNDAMENTAL: {
                "context": "fundamental analysis indicators",
                "keywords": ["earnings", "revenue", "profit", "debt", "growth", "valuation"],
                "weight": 0.3
            },
            ReasoningType.TECHNICAL: {
                "context": "technical analysis patterns",
                "keywords": ["support", "resistance", "trend", "volume", "momentum", "breakout"],
                "weight": 0.25
            },
            ReasoningType.MOMENTUM: {
                "context": "price and volume momentum",
                "keywords": ["momentum", "acceleration", "velocity", "trend strength"],
                "weight": 0.2
            },
            ReasoningType.SENTIMENT: {
                "context": "market sentiment and news",
                "keywords": ["sentiment", "news", "analyst", "upgrade", "downgrade"],
                "weight": 0.1
            },
            ReasoningType.HEAT_DIFFUSION: {
                "context": "heat diffusion and sector contagion",
                "keywords": ["heat", "diffusion", "contagion", "spillover", "correlation"],
                "weight": 0.15
            }
        }
    
    def initialize_financial_ontology(self):
        """Initialize the graph with financial domain knowledge"""
        # Market entity
        market_entity = GraphEntity(
            id="US_MARKET",
            type="Market",
            name="US Stock Market",
            properties={
                "timezone": "America/New_York",
                "trading_hours": "9:30-16:00",
                "indices": ["SPY", "QQQ", "DIA"]
            }
        )
        self.add_entity(market_entity)
        
        # Sector entities with characteristics
        sectors_data = {
            "TECHNOLOGY": {
                "name": "Technology Sector",
                "beta": 1.2,
                "volatility": 0.25,
                "growth_oriented": True,
                "interest_rate_sensitive": True
            },
            "HEALTHCARE": {
                "name": "Healthcare Sector", 
                "beta": 0.8,
                "volatility": 0.18,
                "defensive": True,
                "regulatory_sensitive": True
            },
            "FINANCIAL": {
                "name": "Financial Sector",
                "beta": 1.1,
                "volatility": 0.22,
                "interest_rate_sensitive": True,
                "economic_cyclical": True
            },
            "ENERGY": {
                "name": "Energy Sector",
                "beta": 1.0,
                "volatility": 0.35,
                "commodity_dependent": True,
                "geopolitical_sensitive": True
            }
        }
        
        for sector_id, props in sectors_data.items():
            sector_entity = GraphEntity(
                id=sector_id,
                type="Sector",
                name=props["name"],
                properties=props
            )
            self.add_entity(sector_entity)
            
            # Market contains sectors
            market_sector_rel = GraphRelationship(
                source_id="US_MARKET",
                target_id=sector_id,
                relationship_type="CONTAINS",
                properties={"hierarchical": True},
                weight=1.0,
                confidence=1.0
            )
            self.add_relationship(market_sector_rel)
        
        # Economic indicators
        indicators = [
            "FED_RATE", "INFLATION_RATE", "GDP_GROWTH", "UNEMPLOYMENT",
            "VIX", "DXY", "OIL_PRICE", "GOLD_PRICE"
        ]
        
        for indicator in indicators:
            indicator_entity = GraphEntity(
                id=indicator,
                type="Indicator",
                name=indicator.replace("_", " ").title(),
                properties={"category": "economic", "frequency": "daily"}
            )
            self.add_entity(indicator_entity)
    
    def add_entity(self, entity: GraphEntity):
        """Add entity to the knowledge graph"""
        self.entities[entity.id] = entity
        # Separate entity properties from networkx node attributes to avoid conflicts
        node_attrs = {'entity_type': entity.type, 'entity_name': entity.name}
        node_attrs.update({k: v for k, v in entity.properties.items() if k not in ['type', 'name']})
        self.graph.add_node(entity.id, **node_attrs)
        entity.last_updated = datetime.now()
    
    def add_relationship(self, relationship: GraphRelationship):
        """Add relationship to the knowledge graph"""
        key = (relationship.source_id, relationship.target_id, relationship.relationship_type)
        self.relationships[key] = relationship
        
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            key=relationship.relationship_type,
            weight=relationship.weight,
            confidence=relationship.confidence,
            **relationship.properties
        )
        relationship.last_updated = datetime.now()
    
    def update_from_market_data(self, market_data: Dict[str, Any]):
        """Update graph with real-time market data"""
        try:
            # Update stock entities
            if 'stocks' in market_data:
                for symbol, stock_data in market_data['stocks'].items():
                    self.update_stock_entity(symbol, stock_data)
            
            # Update sector entities
            if 'sectors' in market_data:
                for sector_name, sector_data in market_data['sectors'].items():
                    sector_id = sector_name.upper()
                    self.update_sector_entity(sector_id, sector_data)
            
            # Update market-level entity
            if 'market_overview' in market_data:
                self.update_market_entity(market_data['market_overview'])
                
            # Add heat diffusion relationships
            if 'heated_sectors' in market_data:
                self.update_heat_relationships(market_data['heated_sectors'])
                
        except Exception as e:
            logger.error(f"❌ Error updating graph from market data: {e}")
    
    def update_stock_entity(self, symbol: str, stock_data: Dict[str, Any]):
        """Update or create stock entity"""
        entity_id = f"STOCK_{symbol}"
        
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            entity.properties.update({
                "price": stock_data.get('price', 0),
                "change_percent": stock_data.get('change_percent', 0),
                "volume": stock_data.get('volume', 0),
                "heat_level": stock_data.get('heat_level', 0),
                "last_updated": datetime.now()
            })
        else:
            entity = GraphEntity(
                id=entity_id,
                type="Stock",
                name=symbol,
                properties={
                    "symbol": symbol,
                    "sector": stock_data.get('sector', 'unknown'),
                    "price": stock_data.get('price', 0),
                    "change_percent": stock_data.get('change_percent', 0),
                    "volume": stock_data.get('volume', 0),
                    "heat_level": stock_data.get('heat_level', 0),
                    "beta": stock_data.get('beta', 1.0),
                    "fundamental_score": stock_data.get('fundamental_score', 0)
                }
            )
            self.add_entity(entity)
            
            # Add sector relationship
            sector_id = stock_data.get('sector', '').upper()
            if sector_id and sector_id in self.entities:
                rel = GraphRelationship(
                    source_id=sector_id,
                    target_id=entity_id,
                    relationship_type="CONTAINS",
                    properties={"beta": stock_data.get('beta', 1.0)},
                    weight=1.0,
                    confidence=1.0
                )
                self.add_relationship(rel)
    
    def update_heat_relationships(self, heated_sectors: List[Dict]):
        """Update heat diffusion relationships"""
        for heated in heated_sectors:
            sector_id = heated['sector'].upper()
            heat_level = heated['heat_level']
            
            if sector_id in self.entities and abs(heat_level) > 0.3:
                # Create heat source entity
                heat_entity_id = f"HEAT_SOURCE_{sector_id}"
                
                heat_entity = GraphEntity(
                    id=heat_entity_id,
                    type="HeatSource",
                    name=f"Heat in {heated['sector']}",
                    properties={
                        "heat_level": heat_level,
                        "reason": heated.get('reason', ''),
                        "intensity": "high" if abs(heat_level) > 0.7 else "medium"
                    }
                )
                self.add_entity(heat_entity)
                
                # Heat affects sector
                heat_rel = GraphRelationship(
                    source_id=heat_entity_id,
                    target_id=sector_id,
                    relationship_type="HEATS",
                    properties={"intensity": abs(heat_level)},
                    weight=abs(heat_level),
                    confidence=0.8
                )
                self.add_relationship(heat_rel)
    
    def find_reasoning_paths(self, start_entity_id: str, max_hops: int = 3) -> List[ReasoningPath]:
        """Find reasoning paths from a starting entity"""
        reasoning_paths = []
        
        try:
            # BFS to find paths within max_hops
            visited = set()
            queue = deque([(start_entity_id, [])])  # (entity_id, path_so_far)
            
            while queue and len(reasoning_paths) < 10:  # Limit to top 10 paths
                current_id, path = queue.popleft()
                
                if len(path) >= max_hops:
                    continue
                
                if current_id in visited:
                    continue
                    
                visited.add(current_id)
                current_entity = self.entities.get(current_id)
                
                if not current_entity:
                    continue
                
                # Get neighbors
                for neighbor_id in self.graph.successors(current_id):
                    if neighbor_id not in visited:
                        # Get relationship details
                        edge_data = self.graph.get_edge_data(current_id, neighbor_id)
                        
                        for rel_type, rel_props in edge_data.items():
                            new_path = path + [(current_id, neighbor_id, rel_type, rel_props)]
                            
                            # Create reasoning path if meaningful
                            if len(new_path) >= 2:  # At least 2 hops for reasoning
                                reasoning_path = self.create_reasoning_path(new_path, start_entity_id)
                                if reasoning_path:
                                    reasoning_paths.append(reasoning_path)
                            
                            # Continue BFS
                            queue.append((neighbor_id, new_path))
            
            # Sort by confidence and return top paths
            reasoning_paths.sort(key=lambda p: p.confidence, reverse=True)
            return reasoning_paths[:5]  # Top 5 paths
            
        except Exception as e:
            logger.error(f"❌ Error finding reasoning paths: {e}")
            return []
    
    def create_reasoning_path(self, path_edges: List[Tuple], start_id: str) -> Optional[ReasoningPath]:
        """Create a structured reasoning path from graph edges"""
        try:
            entities = [self.entities.get(start_id)]
            relationships = []
            evidence = []
            
            for source_id, target_id, rel_type, rel_props in path_edges:
                target_entity = self.entities.get(target_id)
                if not target_entity:
                    continue
                    
                entities.append(target_entity)
                
                # Create relationship object
                relationship = GraphRelationship(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=rel_type,
                    properties=rel_props,
                    weight=rel_props.get('weight', 0.5),
                    confidence=rel_props.get('confidence', 0.5)
                )
                relationships.append(relationship)
                
                # Generate evidence
                evidence.append(f"{source_id} {rel_type} {target_id} (confidence: {relationship.confidence:.2f})")
            
            # Determine reasoning type based on entities and relationships
            reasoning_type = self.determine_reasoning_type(entities, relationships)
            
            # Calculate path confidence
            confidences = [r.confidence for r in relationships]
            path_confidence = np.mean(confidences) if confidences else 0.5
            
            # Generate conclusion
            conclusion = self.generate_path_conclusion(entities, relationships, reasoning_type)
            
            return ReasoningPath(
                entities=entities,
                relationships=relationships,
                reasoning_type=reasoning_type,
                confidence=path_confidence,
                evidence=evidence,
                conclusion=conclusion,
                path_length=len(relationships)
            )
            
        except Exception as e:
            logger.error(f"❌ Error creating reasoning path: {e}")
            return None
    
    def determine_reasoning_type(self, entities: List[GraphEntity], relationships: List[GraphRelationship]) -> ReasoningType:
        """Determine the type of reasoning based on path content"""
        entity_types = [e.type for e in entities if e]
        rel_types = [r.relationship_type for r in relationships]
        
        # Heat diffusion reasoning
        if any("Heat" in et for et in entity_types) or any("HEAT" in rt for rt in rel_types):
            return ReasoningType.HEAT_DIFFUSION
        
        # Technical reasoning
        if "Indicator" in entity_types:
            return ReasoningType.TECHNICAL
        
        # Sector rotation reasoning
        if entity_types.count("Sector") > 1:
            return ReasoningType.SECTOR_ROTATION
        
        # Correlation reasoning
        if any("CORRELATED" in rt for rt in rel_types):
            return ReasoningType.CORRELATION
        
        # Default to momentum
        return ReasoningType.MOMENTUM
    
    def generate_path_conclusion(self, entities: List[GraphEntity], relationships: List[GraphRelationship], reasoning_type: ReasoningType) -> str:
        """Generate a conclusion for the reasoning path"""
        try:
            if not entities or not relationships:
                return "Insufficient data for conclusion"
            
            start_entity = entities[0]
            end_entity = entities[-1]
            
            if reasoning_type == ReasoningType.HEAT_DIFFUSION:
                return f"Heat diffusion from {start_entity.name} indicates potential impact on {end_entity.name}"
            elif reasoning_type == ReasoningType.SECTOR_ROTATION:
                return f"Sector rotation pattern suggests movement from {start_entity.name} to {end_entity.name}"
            elif reasoning_type == ReasoningType.CORRELATION:
                return f"Strong correlation between {start_entity.name} and {end_entity.name} suggests similar price movement"
            else:
                return f"Analysis path from {start_entity.name} to {end_entity.name} shows {reasoning_type.value} signals"
        
        except Exception as e:
            return f"Error generating conclusion: {e}"
    
    async def query_with_llm(self, query: str, context_entities: List[GraphEntity], reasoning_paths: List[ReasoningPath]) -> str:
        """Use LLM to generate analysis based on graph context"""
        if not self.openai_client:
            return "LLM analysis unavailable - no OpenAI API key provided"
        
        try:
            # Prepare context
            context_text = "Financial Knowledge Graph Context:\n\n"
            
            # Add entity context
            context_text += "Relevant Entities:\n"
            for entity in context_entities[:5]:  # Limit to top 5
                props_text = ", ".join([f"{k}: {v}" for k, v in entity.properties.items() if k not in ['last_updated']])
                context_text += f"- {entity.name} ({entity.type}): {props_text}\n"
            
            # Add reasoning paths
            context_text += "\nReasoning Paths:\n"
            for i, path in enumerate(reasoning_paths[:3], 1):  # Top 3 paths
                context_text += f"{i}. {path.reasoning_type.value.title()} Path (Confidence: {path.confidence:.2f}):\n"
                context_text += f"   {path.conclusion}\n"
                context_text += f"   Evidence: {'; '.join(path.evidence[:2])}\n\n"
            
            # Create prompt
            prompt = f"""
            You are a financial analyst with access to a comprehensive knowledge graph of market relationships.
            
            {context_text}
            
            User Query: {query}
            
            Based on the knowledge graph context and reasoning paths above, provide a comprehensive analysis that includes:
            1. Direct answer to the user's question
            2. Supporting evidence from the graph relationships
            3. Risk factors to consider
            4. Confidence level in your analysis
            5. Specific actionable insights
            
            Keep the analysis professional, data-driven, and focused on the relationships shown in the knowledge graph.
            """
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst specializing in knowledge graph-based market analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ LLM query error: {e}")
            return f"LLM analysis error: {e}"
    
    async def query(self, query: str, entity_id: Optional[str] = None) -> GraphRAGResult:
        """Main GraphRAG query method"""
        try:
            # Find or infer primary entity
            if entity_id and entity_id in self.entities:
                primary_entity = self.entities[entity_id]
            else:
                # Try to extract entity from query
                primary_entity = self.extract_entity_from_query(query)
                if not primary_entity:
                    # Use market as default
                    primary_entity = self.entities.get("US_MARKET")
            
            if not primary_entity:
                raise ValueError("Could not identify primary entity for query")
            
            # Find reasoning paths
            reasoning_paths = self.find_reasoning_paths(primary_entity.id)
            
            # Get contextual entities (entities in reasoning paths)
            contextual_entities = []
            for path in reasoning_paths:
                contextual_entities.extend(path.entities)
            
            # Remove duplicates and limit
            seen_ids = set()
            unique_entities = []
            for entity in contextual_entities:
                if entity and entity.id not in seen_ids:
                    unique_entities.append(entity)
                    seen_ids.add(entity.id)
            
            contextual_entities = unique_entities[:10]  # Limit context
            
            # Generate LLM analysis
            llm_analysis = await self.query_with_llm(query, contextual_entities, reasoning_paths)
            
            # Extract supporting evidence and risks
            supporting_evidence = []
            risk_factors = []
            
            for path in reasoning_paths:
                supporting_evidence.extend(path.evidence)
                if path.confidence < 0.6:
                    risk_factors.append(f"Low confidence in {path.reasoning_type.value} path")
            
            # Calculate overall confidence
            if reasoning_paths:
                overall_confidence = np.mean([p.confidence for p in reasoning_paths])
            else:
                overall_confidence = 0.5
            
            return GraphRAGResult(
                query=query,
                primary_entity=primary_entity,
                reasoning_paths=reasoning_paths,
                contextual_entities=contextual_entities,
                llm_analysis=llm_analysis,
                confidence=overall_confidence,
                supporting_evidence=supporting_evidence[:5],  # Top 5
                risk_factors=risk_factors[:3],  # Top 3
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ GraphRAG query error: {e}")
            # Return error result
            return GraphRAGResult(
                query=query,
                primary_entity=self.entities.get("US_MARKET"),
                reasoning_paths=[],
                contextual_entities=[],
                llm_analysis=f"Analysis error: {e}",
                confidence=0.0,
                supporting_evidence=[],
                risk_factors=[f"Query processing error: {e}"],
                timestamp=datetime.now()
            )
    
    def extract_entity_from_query(self, query: str) -> Optional[GraphEntity]:
        """Extract relevant entity from query text"""
        query_upper = query.upper()
        
        # Look for stock symbols (simple heuristic)
        for entity_id, entity in self.entities.items():
            if entity.type == "Stock" and entity.name in query_upper:
                return entity
        
        # Look for sector names
        sector_keywords = {
            "TECH": "TECHNOLOGY",
            "HEALTHCARE": "HEALTHCARE", 
            "FINANCIAL": "FINANCIAL",
            "ENERGY": "ENERGY"
        }
        
        for keyword, sector_id in sector_keywords.items():
            if keyword in query_upper and sector_id in self.entities:
                return self.entities[sector_id]
        
        return None
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": {
                entity_type: sum(1 for e in self.entities.values() if e.type == entity_type)
                for entity_type in set(e.type for e in self.entities.values())
            },
            "relationship_types": {
                rel_type: sum(1 for r in self.relationships.values() if r.relationship_type == rel_type)  
                for rel_type in set(r.relationship_type for r in self.relationships.values())
            },
            "graph_connectivity": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "connected_components": nx.number_weakly_connected_components(self.graph)
            }
        }


# Global GraphRAG instance
graph_rag_engine = AdvancedGraphRAG()


# Convenience functions
def update_graph_from_market_data(market_data: Dict[str, Any]):
    """Update the knowledge graph with market data"""
    graph_rag_engine.update_from_market_data(market_data)


async def analyze_stock_with_graph_rag(symbol: str, query: Optional[str] = None) -> GraphRAGResult:
    """Analyze a stock using GraphRAG"""
    if not query:
        query = f"Should I buy {symbol}? Provide detailed analysis based on current market conditions."
    
    entity_id = f"STOCK_{symbol}"
    return await graph_rag_engine.query(query, entity_id)


def get_graph_rag_stats() -> Dict[str, Any]:
    """Get GraphRAG engine statistics"""
    return graph_rag_engine.get_graph_statistics()


if __name__ == "__main__":
    # Test GraphRAG engine
    async def test_graph_rag():
        print("🧠 Testing Advanced GraphRAG Engine")
        print("=" * 50)
        
        # Test query
        result = await graph_rag_engine.query("What's the outlook for technology stocks?")
        
        print(f"Query: {result.query}")
        print(f"Primary Entity: {result.primary_entity.name}")
        print(f"Reasoning Paths: {len(result.reasoning_paths)}")
        print(f"Overall Confidence: {result.confidence:.2f}")
        print(f"\nLLM Analysis Preview:")
        print(result.llm_analysis[:200] + "..." if len(result.llm_analysis) > 200 else result.llm_analysis)
        
        # Show graph stats
        stats = graph_rag_engine.get_graph_statistics()
        print(f"\nGraph Statistics:")
        print(f"Entities: {stats['total_entities']}")
        print(f"Relationships: {stats['total_relationships']}")
        print(f"Entity Types: {stats['entity_types']}")
    
    # Run test
    asyncio.run(test_graph_rag())