"""
Neo4j Knowledge Graph Implementation for RAGHeat System
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ontology.financial_ontology import (
        FinancialOntology, EntityType, RelationType, 
        OntologyEntity, OntologyRelation
    )
except ImportError:
    # Handle relative imports when run from different directories
    import sys
    import os
    backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, backend_path)
    from ontology.financial_ontology import (
        FinancialOntology, EntityType, RelationType, 
        OntologyEntity, OntologyRelation
    )

logger = logging.getLogger(__name__)

class Neo4jKnowledgeGraph:
    """
    Neo4j-backed knowledge graph for financial entities and relationships
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.ontology = FinancialOntology()
        self._connect()
        
    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def initialize_schema(self):
        """Initialize the Neo4j schema based on our ontology"""
        logger.info("Initializing Neo4j schema...")
        
        with self.driver.session() as session:
            # Create constraints and indexes
            schema_queries = self.ontology.get_schema_cypher()
            
            for query in schema_queries:
                try:
                    session.run(query.strip())
                    logger.debug(f"Executed: {query.strip()}")
                except ClientError as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Schema query failed: {e}")
        
        logger.info("Neo4j schema initialization complete")
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
    
    def create_entity(self, entity: OntologyEntity) -> bool:
        """Create a new entity in the knowledge graph"""
        if not self.ontology.validate_entity(entity):
            logger.error(f"Entity validation failed: {entity}")
            return False
        
        cypher = f"""
        MERGE (e:{entity.type.value} {{id: $entity_id}})
        SET e.name = $name,
            e.created_at = $created_at,
            e.updated_at = $updated_at,
            e.properties = $properties
        RETURN e
        """
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher, 
                    entity_id=entity.id,
                    name=entity.name,
                    created_at=entity.created_at.isoformat(),
                    updated_at=entity.updated_at.isoformat(),
                    properties=json.dumps(entity.properties)
                )
                logger.info(f"Created entity: {entity.id} ({entity.type.value})")
                return True
            except Exception as e:
                logger.error(f"Failed to create entity {entity.id}: {e}")
                return False
    
    def create_relationship(self, relation: OntologyRelation,
                          source_entity: OntologyEntity,
                          target_entity: OntologyEntity) -> bool:
        """Create a relationship between two entities"""
        if not self.ontology.validate_relation(relation, source_entity, target_entity):
            logger.error(f"Relationship validation failed: {relation}")
            return False
        
        cypher = f"""
        MATCH (s:{source_entity.type.value} {{id: $source_id}})
        MATCH (t:{target_entity.type.value} {{id: $target_id}})
        MERGE (s)-[r:{relation.relation_type.value}]->(t)
        SET r.id = $relation_id,
            r.strength = $strength,
            r.created_at = $created_at,
            r.properties = $properties
        RETURN r
        """
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher,
                    source_id=relation.source_id,
                    target_id=relation.target_id,
                    relation_id=relation.id,
                    strength=relation.strength,
                    created_at=relation.created_at.isoformat(),
                    properties=json.dumps(relation.properties)
                )
                logger.info(f"Created relationship: {relation.source_id} -[{relation.relation_type.value}]-> {relation.target_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to create relationship {relation.id}: {e}")
                return False
    
    def update_entity_properties(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Update properties of an existing entity"""
        cypher = """
        MATCH (e {id: $entity_id})
        SET e.properties = $properties,
            e.updated_at = $updated_at
        RETURN e
        """
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher,
                    entity_id=entity_id,
                    properties=json.dumps(properties),
                    updated_at=datetime.now().isoformat()
                )
                
                if result.single():
                    logger.info(f"Updated entity properties: {entity_id}")
                    return True
                else:
                    logger.warning(f"Entity not found: {entity_id}")
                    return False
            except Exception as e:
                logger.error(f"Failed to update entity {entity_id}: {e}")
                return False
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an entity by ID"""
        cypher = "MATCH (e {id: $entity_id}) RETURN e"
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher, entity_id=entity_id)
                record = result.single()
                
                if record:
                    node = record["e"]
                    return {
                        "id": node["id"],
                        "name": node.get("name"),
                        "labels": list(node.labels),
                        "properties": json.loads(node.get("properties", "{}")),
                        "created_at": node.get("created_at"),
                        "updated_at": node.get("updated_at")
                    }
                return None
            except Exception as e:
                logger.error(f"Failed to get entity {entity_id}: {e}")
                return None
    
    def get_entities_by_type(self, entity_type: EntityType, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all entities of a specific type"""
        cypher = f"MATCH (e:{entity_type.value}) RETURN e LIMIT $limit"
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher, limit=limit)
                entities = []
                
                for record in result:
                    node = record["e"]
                    entities.append({
                        "id": node["id"],
                        "name": node.get("name"),
                        "properties": json.loads(node.get("properties", "{}")),
                        "created_at": node.get("created_at"),
                        "updated_at": node.get("updated_at")
                    })
                
                return entities
            except Exception as e:
                logger.error(f"Failed to get entities of type {entity_type.value}: {e}")
                return []
    
    def get_neighbors(self, entity_id: str, relation_types: Optional[List[RelationType]] = None) -> List[Dict[str, Any]]:
        """Get neighboring entities connected by specific relationship types"""
        if relation_types:
            rel_filter = "|".join([rt.value for rt in relation_types])
            cypher = f"""
            MATCH (e {{id: $entity_id}})-[r:{rel_filter}]-(neighbor)
            RETURN neighbor, r, type(r) as relationship_type
            """
        else:
            cypher = """
            MATCH (e {id: $entity_id})-[r]-(neighbor)
            RETURN neighbor, r, type(r) as relationship_type
            """
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher, entity_id=entity_id)
                neighbors = []
                
                for record in result:
                    neighbor = record["neighbor"]
                    relationship = record["r"]
                    
                    neighbors.append({
                        "entity": {
                            "id": neighbor["id"],
                            "name": neighbor.get("name"),
                            "labels": list(neighbor.labels),
                            "properties": json.loads(neighbor.get("properties", "{}"))
                        },
                        "relationship": {
                            "type": record["relationship_type"],
                            "strength": relationship.get("strength"),
                            "properties": json.loads(relationship.get("properties", "{}"))
                        }
                    })
                
                return neighbors
            except Exception as e:
                logger.error(f"Failed to get neighbors for {entity_id}: {e}")
                return []
    
    def find_shortest_path(self, source_id: str, target_id: str, max_hops: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Find the shortest path between two entities"""
        cypher = """
        MATCH path = shortestPath((source {id: $source_id})-[*..{max_hops}]-(target {id: $target_id}))
        RETURN path
        """.format(max_hops=max_hops)
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher, source_id=source_id, target_id=target_id)
                record = result.single()
                
                if record:
                    path = record["path"]
                    path_info = []
                    
                    for i, node in enumerate(path.nodes):
                        path_info.append({
                            "step": i,
                            "entity": {
                                "id": node["id"],
                                "name": node.get("name"),
                                "labels": list(node.labels)
                            }
                        })
                        
                        if i < len(path.relationships):
                            rel = path.relationships[i]
                            path_info[-1]["outgoing_relationship"] = {
                                "type": rel.type,
                                "strength": rel.get("strength")
                            }
                    
                    return path_info
                return None
            except Exception as e:
                logger.error(f"Failed to find path between {source_id} and {target_id}: {e}")
                return None
    
    def calculate_centrality(self, entity_type: Optional[EntityType] = None) -> Dict[str, float]:
        """Calculate degree centrality for entities"""
        if entity_type:
            cypher = f"""
            MATCH (n:{entity_type.value})
            RETURN n.id as entity_id, size((n)-[]-()) as degree
            ORDER BY degree DESC
            """
        else:
            cypher = """
            MATCH (n)
            RETURN n.id as entity_id, size((n)-[]-()) as degree
            ORDER BY degree DESC
            """
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher)
                centrality_scores = {}
                
                for record in result:
                    centrality_scores[record["entity_id"]] = record["degree"]
                
                return centrality_scores
            except Exception as e:
                logger.error(f"Failed to calculate centrality: {e}")
                return {}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get overall graph statistics"""
        stats_queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "node_types": "MATCH (n) RETURN labels(n) as labels, count(n) as count",
            "relationship_types": "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
        }
        
        stats = {}
        
        with self.driver.session() as session:
            try:
                # Get basic counts
                for stat_name, query in stats_queries.items():
                    if stat_name in ["total_nodes", "total_relationships"]:
                        result = session.run(query)
                        stats[stat_name] = result.single()["count"]
                    else:
                        result = session.run(query)
                        stats[stat_name] = [dict(record) for record in result]
                
                return stats
            except Exception as e:
                logger.error(f"Failed to get graph statistics: {e}")
                return {}
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export the entire graph for visualization"""
        nodes_query = """
        MATCH (n)
        RETURN n.id as id, labels(n) as labels, n.name as name, n.properties as properties
        """
        
        edges_query = """
        MATCH (source)-[r]->(target)
        RETURN source.id as source, target.id as target, type(r) as type, 
               r.strength as strength, r.properties as properties
        """
        
        with self.driver.session() as session:
            try:
                # Get nodes
                nodes_result = session.run(nodes_query)
                nodes = []
                for record in nodes_result:
                    nodes.append({
                        "id": record["id"],
                        "labels": record["labels"],
                        "name": record["name"],
                        "properties": json.loads(record["properties"] or "{}")
                    })
                
                # Get edges
                edges_result = session.run(edges_query)
                edges = []
                for record in edges_result:
                    edges.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"],
                        "strength": record["strength"],
                        "properties": json.loads(record["properties"] or "{}")
                    })
                
                return {
                    "nodes": nodes,
                    "edges": edges,
                    "metadata": {
                        "node_count": len(nodes),
                        "edge_count": len(edges),
                        "exported_at": datetime.now().isoformat()
                    }
                }
            except Exception as e:
                logger.error(f"Failed to export graph data: {e}")
                return {"nodes": [], "edges": [], "metadata": {}}