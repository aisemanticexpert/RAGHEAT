"""
Knowledge Graph Tools for Portfolio Construction
=============================================

Tools for building, querying, and managing financial knowledge graphs.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crewai_tools import BaseTool
from loguru import logger

try:
    from neo4j import GraphDatabase
    import networkx as nx
    from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal
    from SPARQLWrapper import SPARQLWrapper, JSON
except ImportError:
    logger.warning("Some graph libraries not available")

class GraphConstructorTool(BaseTool):
    """Tool for constructing financial knowledge graphs."""
    
    name: str = "graph_constructor"
    description: str = "Construct and update financial knowledge graphs with entities and relationships"
    
    def __init__(self):
        super().__init__()
        self.graph = nx.DiGraph()  # NetworkX graph for in-memory operations
        self.rdf_graph = None
        try:
            from rdflib import Graph
            self.rdf_graph = Graph()
        except ImportError:
            pass
    
    def _run(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Construct knowledge graph from entities and relationships."""
        try:
            # Add entities to graph
            entity_count = self._add_entities(entities)
            
            # Add relationships to graph  
            relationship_count = self._add_relationships(relationships)
            
            # Calculate graph statistics
            graph_stats = self._calculate_graph_statistics()
            
            # Extract key insights
            insights = self._extract_graph_insights()
            
            return {
                'graph_construction': {
                    'entities_added': entity_count,
                    'relationships_added': relationship_count,
                    'graph_statistics': graph_stats,
                    'key_insights': insights
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error constructing knowledge graph: {e}")
            return {'error': str(e)}
    
    def _add_entities(self, entities: List[Dict[str, Any]]) -> int:
        """Add entities to the knowledge graph."""
        count = 0
        
        for entity in entities:
            entity_id = entity.get('id') or entity.get('symbol') or entity.get('name')
            entity_type = entity.get('type', 'unknown')
            
            if entity_id:
                # Add to NetworkX graph
                self.graph.add_node(
                    entity_id,
                    type=entity_type,
                    **{k: v for k, v in entity.items() if k not in ['id', 'type']}
                )
                
                # Add to RDF graph if available
                if self.rdf_graph:
                    self._add_rdf_entity(entity_id, entity_type, entity)
                
                count += 1
        
        return count
    
    def _add_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """Add relationships to the knowledge graph."""
        count = 0
        
        for rel in relationships:
            source = rel.get('source')
            target = rel.get('target')
            relation_type = rel.get('type', 'related_to')
            weight = rel.get('weight', 1.0)
            
            if source and target and source != target:
                # Add to NetworkX graph
                self.graph.add_edge(
                    source, target,
                    type=relation_type,
                    weight=weight,
                    **{k: v for k, v in rel.items() if k not in ['source', 'target', 'type', 'weight']}
                )
                
                # Add to RDF graph if available
                if self.rdf_graph:
                    self._add_rdf_relationship(source, target, relation_type)
                
                count += 1
        
        return count
    
    def _add_rdf_entity(self, entity_id: str, entity_type: str, entity_data: Dict):
        """Add entity to RDF graph."""
        try:
            from rdflib import URIRef, Literal, RDF
            
            # Define namespaces
            FINANCE = Namespace("http://finance.example.com/")
            
            entity_uri = FINANCE[entity_id]
            type_uri = FINANCE[entity_type]
            
            # Add type
            self.rdf_graph.add((entity_uri, RDF.type, type_uri))
            
            # Add properties
            for key, value in entity_data.items():
                if key not in ['id', 'type'] and value is not None:
                    property_uri = FINANCE[key]
                    if isinstance(value, str):
                        self.rdf_graph.add((entity_uri, property_uri, Literal(value)))
                    else:
                        self.rdf_graph.add((entity_uri, property_uri, Literal(str(value))))
        except Exception as e:
            logger.warning(f"Error adding RDF entity: {e}")
    
    def _add_rdf_relationship(self, source: str, target: str, relation_type: str):
        """Add relationship to RDF graph."""
        try:
            from rdflib import URIRef
            
            FINANCE = Namespace("http://finance.example.com/")
            
            source_uri = FINANCE[source]
            target_uri = FINANCE[target]
            relation_uri = FINANCE[relation_type]
            
            self.rdf_graph.add((source_uri, relation_uri, target_uri))
        except Exception as e:
            logger.warning(f"Error adding RDF relationship: {e}")
    
    def _calculate_graph_statistics(self) -> Dict[str, Any]:
        """Calculate basic graph statistics."""
        try:
            stats = {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'is_directed': self.graph.is_directed(),
                'is_connected': nx.is_weakly_connected(self.graph) if self.graph.is_directed() else nx.is_connected(self.graph),
                'number_of_components': nx.number_weakly_connected_components(self.graph) if self.graph.is_directed() else nx.number_connected_components(self.graph)
            }
            
            if self.graph.number_of_nodes() > 0:
                stats['average_degree'] = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
                
                # Calculate centrality measures for top nodes
                if self.graph.number_of_nodes() > 1:
                    degree_centrality = nx.degree_centrality(self.graph)
                    betweenness_centrality = nx.betweenness_centrality(self.graph)
                    
                    stats['most_connected_nodes'] = sorted(
                        degree_centrality.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                    
                    stats['most_important_nodes'] = sorted(
                        betweenness_centrality.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating graph statistics: {e}")
            return {'error': str(e)}
    
    def _extract_graph_insights(self) -> Dict[str, Any]:
        """Extract key insights from the knowledge graph."""
        insights = {}
        
        try:
            # Node type distribution
            node_types = {}
            for node in self.graph.nodes(data=True):
                node_type = node[1].get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            insights['entity_type_distribution'] = node_types
            
            # Relationship type distribution
            edge_types = {}
            for edge in self.graph.edges(data=True):
                edge_type = edge[2].get('type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            insights['relationship_type_distribution'] = edge_types
            
            # Find clusters or communities
            if self.graph.number_of_nodes() > 3:
                try:
                    # Convert to undirected for community detection
                    undirected_graph = self.graph.to_undirected()
                    communities = list(nx.connected_components(undirected_graph))
                    insights['communities'] = {
                        'number_of_communities': len(communities),
                        'largest_community_size': max(len(c) for c in communities) if communities else 0
                    }
                except:
                    pass
            
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting graph insights: {e}")
            return {'error': str(e)}

class SPARQLQueryEngineTool(BaseTool):
    """Tool for executing SPARQL queries on RDF knowledge graphs."""
    
    name: str = "sparql_query_engine"
    description: str = "Execute SPARQL queries to extract insights from RDF knowledge graphs"
    
    def __init__(self):
        super().__init__()
        self.rdf_graph = None
        try:
            from rdflib import Graph
            self.rdf_graph = Graph()
        except ImportError:
            logger.warning("RDFLib not available")
    
    def _run(self, query: str, graph_data: Optional[str] = None) -> Dict[str, Any]:
        """Execute SPARQL query on the knowledge graph."""
        try:
            if not self.rdf_graph:
                return {'error': 'RDF graph not available'}
            
            # Load graph data if provided
            if graph_data:
                self.rdf_graph.parse(data=graph_data, format='turtle')
            
            # Execute SPARQL query
            results = self.rdf_graph.query(query)
            
            # Convert results to serializable format
            query_results = []
            for row in results:
                result_row = {}
                for i, var in enumerate(results.vars):
                    value = row[i]
                    result_row[str(var)] = str(value) if value else None
                query_results.append(result_row)
            
            return {
                'query': query,
                'results': query_results,
                'result_count': len(query_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            return {'error': str(e), 'query': query}
    
    def get_sample_queries(self) -> Dict[str, str]:
        """Get sample SPARQL queries for financial knowledge graphs."""
        return {
            'all_stocks_in_sector': """
                PREFIX finance: <http://finance.example.com/>
                SELECT ?stock ?sector
                WHERE {
                    ?stock finance:type "stock" .
                    ?stock finance:sector ?sector .
                }
            """,
            'correlated_stocks': """
                PREFIX finance: <http://finance.example.com/>
                SELECT ?stock1 ?stock2 ?correlation
                WHERE {
                    ?stock1 finance:correlates_with ?stock2 .
                    ?stock1 finance:correlation_value ?correlation .
                    FILTER(?correlation > 0.7)
                }
            """,
            'stocks_affected_by_event': """
                PREFIX finance: <http://finance.example.com/>
                SELECT ?stock ?event ?impact
                WHERE {
                    ?event finance:affects ?stock .
                    ?event finance:impact_score ?impact .
                }
                ORDER BY DESC(?impact)
            """
        }

class Neo4jInterfaceTool(BaseTool):
    """Tool for interfacing with Neo4j graph database."""
    
    name: str = "neo4j_interface"
    description: str = "Interface with Neo4j database for storing and querying knowledge graphs"
    
    def __init__(self):
        super().__init__()
        self.driver = None
        self._connect_to_neo4j()
    
    def _connect_to_neo4j(self):
        """Connect to Neo4j database."""
        try:
            from ..config.settings import settings
            
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            logger.info("Connected to Neo4j database")
            
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
            self.driver = None
    
    def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Neo4j operations."""
        try:
            if not self.driver:
                return {'error': 'Neo4j connection not available'}
            
            if operation == 'create_nodes':
                return self._create_nodes(kwargs.get('nodes', []))
            elif operation == 'create_relationships':
                return self._create_relationships(kwargs.get('relationships', []))
            elif operation == 'query':
                return self._execute_cypher_query(kwargs.get('query', ''))
            elif operation == 'get_graph_stats':
                return self._get_graph_statistics()
            else:
                return {'error': f'Unknown operation: {operation}'}
                
        except Exception as e:
            logger.error(f"Error in Neo4j operation: {e}")
            return {'error': str(e)}
    
    def _create_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create nodes in Neo4j."""
        try:
            with self.driver.session() as session:
                created_count = 0
                
                for node in nodes:
                    node_id = node.get('id')
                    labels = node.get('labels', ['Entity'])
                    properties = {k: v for k, v in node.items() if k not in ['id', 'labels']}
                    
                    if node_id:
                        # Create Cypher query
                        labels_str = ':'.join(labels)
                        query = f"""
                        MERGE (n:{labels_str} {{id: $id}})
                        SET n += $properties
                        """
                        
                        session.run(query, id=node_id, properties=properties)
                        created_count += 1
                
                return {
                    'nodes_created': created_count,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error creating nodes: {e}")
            return {'error': str(e)}
    
    def _create_relationships(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create relationships in Neo4j."""
        try:
            with self.driver.session() as session:
                created_count = 0
                
                for rel in relationships:
                    source_id = rel.get('source')
                    target_id = rel.get('target')
                    rel_type = rel.get('type', 'RELATED_TO')
                    properties = {k: v for k, v in rel.items() if k not in ['source', 'target', 'type']}
                    
                    if source_id and target_id:
                        query = f"""
                        MATCH (a {{id: $source_id}})
                        MATCH (b {{id: $target_id}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r += $properties
                        """
                        
                        session.run(query, source_id=source_id, target_id=target_id, properties=properties)
                        created_count += 1
                
                return {
                    'relationships_created': created_count,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error creating relationships: {e}")
            return {'error': str(e)}
    
    def _execute_cypher_query(self, query: str) -> Dict[str, Any]:
        """Execute Cypher query."""
        try:
            with self.driver.session() as session:
                result = session.run(query)
                
                records = []
                for record in result:
                    records.append(dict(record))
                
                return {
                    'query': query,
                    'results': records,
                    'result_count': len(records),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return {'error': str(e), 'query': query}
    
    def _get_graph_statistics(self) -> Dict[str, Any]:
        """Get Neo4j graph statistics."""
        try:
            with self.driver.session() as session:
                # Count nodes
                node_count_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = node_count_result.single()['count']
                
                # Count relationships
                rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_count_result.single()['count']
                
                # Get node labels
                labels_result = session.run("CALL db.labels()")
                labels = [record['label'] for record in labels_result]
                
                # Get relationship types
                rel_types_result = session.run("CALL db.relationshipTypes()")
                rel_types = [record['relationshipType'] for record in rel_types_result]
                
                return {
                    'total_nodes': node_count,
                    'total_relationships': rel_count,
                    'node_labels': labels,
                    'relationship_types': rel_types,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {'error': str(e)}
    
    def __del__(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

class OntologyMapperTool(BaseTool):
    """Tool for mapping financial entities to ontology concepts."""
    
    name: str = "ontology_mapper"
    description: str = "Map financial entities and relationships to structured ontology concepts"
    
    def __init__(self):
        super().__init__()
        self.financial_ontology = self._load_financial_ontology()
    
    def _run(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map entities to ontological concepts."""
        try:
            mapped_entities = []
            
            for entity in entities:
                mapped_entity = self._map_entity_to_ontology(entity)
                mapped_entities.append(mapped_entity)
            
            # Generate ontological relationships
            inferred_relationships = self._infer_relationships(mapped_entities)
            
            return {
                'mapped_entities': mapped_entities,
                'inferred_relationships': inferred_relationships,
                'ontology_coverage': self._calculate_ontology_coverage(mapped_entities),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in ontology mapping: {e}")
            return {'error': str(e)}
    
    def _load_financial_ontology(self) -> Dict[str, Any]:
        """Load financial domain ontology."""
        # Simplified ontology structure
        return {
            'classes': {
                'FinancialInstrument': {
                    'subclasses': ['Stock', 'Bond', 'Option', 'ETF'],
                    'properties': ['ticker', 'price', 'volume', 'market_cap']
                },
                'Company': {
                    'subclasses': ['PublicCompany', 'PrivateCompany'],
                    'properties': ['name', 'sector', 'industry', 'employees', 'revenue']
                },
                'Sector': {
                    'subclasses': ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer'],
                    'properties': ['name', 'description']
                },
                'Event': {
                    'subclasses': ['EarningsAnnouncement', 'FedDecision', 'NewsEvent'],
                    'properties': ['date', 'impact', 'description']
                },
                'Indicator': {
                    'subclasses': ['EconomicIndicator', 'TechnicalIndicator'],
                    'properties': ['value', 'date', 'significance']
                }
            },
            'relationships': {
                'belongsToSector': {'domain': 'Company', 'range': 'Sector'},
                'correlatesWith': {'domain': 'FinancialInstrument', 'range': 'FinancialInstrument'},
                'affectedBy': {'domain': 'FinancialInstrument', 'range': 'Event'},
                'influences': {'domain': 'Event', 'range': 'FinancialInstrument'},
                'issuedBy': {'domain': 'FinancialInstrument', 'range': 'Company'}
            }
        }
    
    def _map_entity_to_ontology(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Map an entity to ontological concepts."""
        entity_type = entity.get('type', '').lower()
        
        # Determine ontological class
        ontological_class = self._determine_ontological_class(entity_type, entity)
        
        # Extract relevant properties
        mapped_properties = self._extract_ontological_properties(entity, ontological_class)
        
        return {
            'original_entity': entity,
            'ontological_class': ontological_class,
            'mapped_properties': mapped_properties,
            'confidence': self._calculate_mapping_confidence(entity_type, ontological_class)
        }
    
    def _determine_ontological_class(self, entity_type: str, entity: Dict[str, Any]) -> str:
        """Determine the appropriate ontological class."""
        # Simple rule-based mapping
        if entity_type in ['stock', 'equity']:
            return 'Stock'
        elif entity_type in ['company', 'corporation']:
            return 'Company'
        elif entity_type in ['sector', 'industry']:
            return 'Sector'
        elif entity_type in ['event', 'news']:
            return 'Event'
        elif entity_type in ['indicator', 'metric']:
            return 'Indicator'
        elif 'ticker' in entity or 'symbol' in entity:
            return 'Stock'
        elif 'sector' in entity:
            return 'Company'
        else:
            return 'Entity'  # Default class
    
    def _extract_ontological_properties(self, entity: Dict[str, Any], ontological_class: str) -> Dict[str, Any]:
        """Extract properties relevant to the ontological class."""
        class_info = self.financial_ontology['classes'].get(ontological_class, {})
        relevant_properties = class_info.get('properties', [])
        
        mapped_props = {}
        for prop in relevant_properties:
            if prop in entity:
                mapped_props[prop] = entity[prop]
            # Handle variations
            elif prop == 'ticker' and 'symbol' in entity:
                mapped_props[prop] = entity['symbol']
            elif prop == 'price' and 'current_price' in entity:
                mapped_props[prop] = entity['current_price']
        
        return mapped_props
    
    def _calculate_mapping_confidence(self, entity_type: str, ontological_class: str) -> float:
        """Calculate confidence in the ontological mapping."""
        # Simple confidence scoring
        direct_mappings = {
            'stock': 'Stock',
            'company': 'Company',
            'sector': 'Sector',
            'event': 'Event'
        }
        
        if entity_type in direct_mappings and direct_mappings[entity_type] == ontological_class:
            return 0.9
        elif ontological_class != 'Entity':
            return 0.7
        else:
            return 0.4
    
    def _infer_relationships(self, mapped_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Infer relationships between mapped entities."""
        relationships = []
        
        # Simple relationship inference
        for i, entity1 in enumerate(mapped_entities):
            for j, entity2 in enumerate(mapped_entities):
                if i != j:
                    relationship = self._infer_relationship_between_entities(entity1, entity2)
                    if relationship:
                        relationships.append(relationship)
        
        return relationships
    
    def _infer_relationship_between_entities(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Infer relationship between two entities."""
        class1 = entity1['ontological_class']
        class2 = entity2['ontological_class']
        
        # Define inference rules
        if class1 == 'Company' and class2 == 'Sector':
            # Check if company has sector information
            if entity2['original_entity'].get('name') in str(entity1['mapped_properties'].get('sector', '')):
                return {
                    'source': entity1['original_entity'].get('id'),
                    'target': entity2['original_entity'].get('id'),
                    'type': 'belongsToSector',
                    'confidence': 0.8
                }
        
        elif class1 == 'Stock' and class2 == 'Company':
            # Stock issued by company
            return {
                'source': entity1['original_entity'].get('id'),
                'target': entity2['original_entity'].get('id'),
                'type': 'issuedBy',
                'confidence': 0.9
            }
        
        return None
    
    def _calculate_ontology_coverage(self, mapped_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate how well entities cover the ontology."""
        class_counts = {}
        total_entities = len(mapped_entities)
        
        for entity in mapped_entities:
            ontological_class = entity['ontological_class']
            class_counts[ontological_class] = class_counts.get(ontological_class, 0) + 1
        
        return {
            'total_entities': total_entities,
            'class_distribution': class_counts,
            'coverage_percentage': len(class_counts) / len(self.financial_ontology['classes']) * 100
        }

class TripleExtractorTool(BaseTool):
    """Tool for extracting RDF triples from text and structured data."""
    
    name: str = "triple_extractor"
    description: str = "Extract RDF triples (subject-predicate-object) from financial text and data"
    
    def _run(self, text: str = "", structured_data: List[Dict] = None) -> Dict[str, Any]:
        """Extract triples from text or structured data."""
        try:
            triples = []
            
            if text:
                text_triples = self._extract_triples_from_text(text)
                triples.extend(text_triples)
            
            if structured_data:
                structured_triples = self._extract_triples_from_structured_data(structured_data)
                triples.extend(structured_triples)
            
            # Validate and clean triples
            validated_triples = self._validate_triples(triples)
            
            return {
                'extracted_triples': validated_triples,
                'triple_count': len(validated_triples),
                'extraction_confidence': self._calculate_extraction_confidence(validated_triples),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error extracting triples: {e}")
            return {'error': str(e)}
    
    def _extract_triples_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract triples from natural language text."""
        triples = []
        
        # Simple pattern-based extraction (would use NLP in production)
        financial_patterns = [
            (r'(\w+)\s+belongs to\s+(\w+)\s+sector', 'belongsToSector'),
            (r'(\w+)\s+correlates with\s+(\w+)', 'correlatesWith'),
            (r'(\w+)\s+is affected by\s+(.+)', 'affectedBy'),
            (r'(\w+)\s+reports\s+(.+)', 'reports'),
            (r'(\w+)\s+has\s+(.+)', 'hasProperty')
        ]
        
        import re
        
        for pattern, predicate in financial_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                object_val = match.group(2).strip()
                
                triple = {
                    'subject': subject,
                    'predicate': predicate,
                    'object': object_val,
                    'confidence': 0.6,  # Pattern-based extraction has moderate confidence
                    'source': 'text_extraction'
                }
                triples.append(triple)
        
        return triples
    
    def _extract_triples_from_structured_data(self, structured_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract triples from structured data."""
        triples = []
        
        for item in structured_data:
            if isinstance(item, dict):
                # Extract triples from key-value pairs
                subject = item.get('id') or item.get('symbol') or item.get('name')
                
                if subject:
                    for key, value in item.items():
                        if key not in ['id', 'symbol', 'name'] and value is not None:
                            predicate = f"has{key.capitalize()}"
                            
                            triple = {
                                'subject': str(subject),
                                'predicate': predicate,
                                'object': str(value),
                                'confidence': 0.9,  # High confidence for structured data
                                'source': 'structured_data'
                            }
                            triples.append(triple)
        
        return triples
    
    def _validate_triples(self, triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean extracted triples."""
        validated_triples = []
        
        for triple in triples:
            # Check that all required fields are present
            if all(field in triple for field in ['subject', 'predicate', 'object']):
                # Clean the triple
                cleaned_triple = {
                    'subject': str(triple['subject']).strip(),
                    'predicate': str(triple['predicate']).strip(),
                    'object': str(triple['object']).strip(),
                    'confidence': float(triple.get('confidence', 0.5)),
                    'source': triple.get('source', 'unknown')
                }
                
                # Additional validation
                if (len(cleaned_triple['subject']) > 0 and 
                    len(cleaned_triple['predicate']) > 0 and 
                    len(cleaned_triple['object']) > 0):
                    validated_triples.append(cleaned_triple)
        
        return validated_triples
    
    def _calculate_extraction_confidence(self, triples: List[Dict[str, Any]]) -> float:
        """Calculate overall extraction confidence."""
        if not triples:
            return 0.0
        
        total_confidence = sum(triple.get('confidence', 0.0) for triple in triples)
        return total_confidence / len(triples)

# Initialize tools
graph_constructor = GraphConstructorTool()
sparql_query_engine = SPARQLQueryEngineTool()
neo4j_interface = Neo4jInterfaceTool()
ontology_mapper = OntologyMapperTool()
triple_extractor = TripleExtractorTool()