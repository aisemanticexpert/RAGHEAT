"""
Financial Ontology for RAGHeat System
Defines the conceptual model for financial entities and relationships
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class EntityType(Enum):
    """Financial entity types in our ontology"""
    MARKET = "Market"
    SECTOR = "Sector" 
    INDUSTRY = "Industry"
    STOCK = "Stock"
    COMPANY = "Company"
    FINANCIAL_METRIC = "FinancialMetric"
    NEWS_EVENT = "NewsEvent"
    ANALYST_RATING = "AnalystRating"
    HEAT_SOURCE = "HeatSource"

class RelationType(Enum):
    """Relationship types between entities"""
    CONTAINS = "CONTAINS"
    BELONGS_TO = "BELONGS_TO"
    INFLUENCES = "INFLUENCES"
    COMPETES_WITH = "COMPETES_WITH"
    CORRELATES_WITH = "CORRELATES_WITH"
    HAS_METRIC = "HAS_METRIC"
    RATED_BY = "RATED_BY"
    AFFECTED_BY = "AFFECTED_BY"
    GENERATES_HEAT = "GENERATES_HEAT"
    PROPAGATES_TO = "PROPAGATES_TO"

@dataclass
class OntologyEntity:
    """Base class for all ontology entities"""
    id: str
    type: EntityType
    name: str
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class OntologyRelation:
    """Represents relationships between entities"""
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any]
    strength: float  # 0.0 to 1.0
    created_at: datetime

class FinancialOntology:
    """
    Financial Ontology defining our knowledge structure
    
    Hierarchy:
    Market (Root)
    ├── Technology Sector
    │   ├── Software Industry
    │   │   ├── AAPL Company
    │   │   ├── GOOGL Company  
    │   │   └── MSFT Company
    │   └── Hardware Industry
    │       ├── NVDA Company
    │       └── ...
    ├── Healthcare Sector
    │   ├── Pharmaceuticals Industry
    │   │   ├── JNJ Company
    │   │   └── PFE Company
    │   └── Insurance Industry
    │       └── UNH Company
    └── ...
    """
    
    def __init__(self):
        self.entity_definitions = self._define_entities()
        self.relation_definitions = self._define_relations()
        
    def _define_entities(self) -> Dict[EntityType, Dict]:
        """Define the properties and constraints for each entity type"""
        return {
            EntityType.MARKET: {
                "required_properties": ["name", "region", "currency"],
                "optional_properties": ["total_market_cap", "trading_hours", "timezone"],
                "constraints": {
                    "name": str,
                    "region": str,
                    "currency": str
                }
            },
            
            EntityType.SECTOR: {
                "required_properties": ["name", "code", "description"],
                "optional_properties": ["market_cap_percentage", "volatility_index", "beta"],
                "constraints": {
                    "name": str,
                    "code": str,
                    "description": str
                }
            },
            
            EntityType.INDUSTRY: {
                "required_properties": ["name", "sector", "description"],
                "optional_properties": ["companies_count", "avg_market_cap", "growth_rate"],
                "constraints": {
                    "name": str,
                    "sector": str,
                    "description": str
                }
            },
            
            EntityType.STOCK: {
                "required_properties": ["symbol", "name", "exchange"],
                "optional_properties": ["isin", "cusip", "listing_date", "delisting_date"],
                "constraints": {
                    "symbol": str,
                    "name": str,
                    "exchange": str
                }
            },
            
            EntityType.COMPANY: {
                "required_properties": ["name", "sector", "industry", "headquarters"],
                "optional_properties": [
                    "founded_year", "employees_count", "website", "ceo", 
                    "description", "business_model"
                ],
                "constraints": {
                    "name": str,
                    "sector": str,
                    "industry": str,
                    "headquarters": str
                }
            },
            
            EntityType.FINANCIAL_METRIC: {
                "required_properties": ["name", "value", "unit", "period"],
                "optional_properties": ["previous_value", "change_pct", "peer_comparison"],
                "constraints": {
                    "name": str,
                    "value": float,
                    "unit": str,
                    "period": str
                }
            },
            
            EntityType.NEWS_EVENT: {
                "required_properties": ["headline", "source", "published_at", "impact_score"],
                "optional_properties": ["summary", "sentiment", "relevance", "url"],
                "constraints": {
                    "headline": str,
                    "source": str,
                    "published_at": datetime,
                    "impact_score": float
                }
            },
            
            EntityType.ANALYST_RATING: {
                "required_properties": ["analyst_firm", "rating", "target_price", "date"],
                "optional_properties": ["previous_rating", "reasoning", "time_horizon"],
                "constraints": {
                    "analyst_firm": str,
                    "rating": str,
                    "target_price": float,
                    "date": datetime
                }
            },
            
            EntityType.HEAT_SOURCE: {
                "required_properties": ["source_type", "intensity", "timestamp"],
                "optional_properties": ["duration", "decay_rate", "propagation_factor"],
                "constraints": {
                    "source_type": str,
                    "intensity": float,
                    "timestamp": datetime
                }
            }
        }
    
    def _define_relations(self) -> Dict[RelationType, Dict]:
        """Define valid relationships and their properties"""
        return {
            RelationType.CONTAINS: {
                "valid_pairs": [
                    (EntityType.MARKET, EntityType.SECTOR),
                    (EntityType.SECTOR, EntityType.INDUSTRY),
                    (EntityType.INDUSTRY, EntityType.COMPANY)
                ],
                "properties": ["containment_percentage", "primary_flag"],
                "bidirectional": False
            },
            
            RelationType.BELONGS_TO: {
                "valid_pairs": [
                    (EntityType.SECTOR, EntityType.MARKET),
                    (EntityType.INDUSTRY, EntityType.SECTOR),
                    (EntityType.COMPANY, EntityType.INDUSTRY),
                    (EntityType.STOCK, EntityType.COMPANY)
                ],
                "properties": ["membership_strength", "primary_flag"],
                "bidirectional": False
            },
            
            RelationType.INFLUENCES: {
                "valid_pairs": [
                    (EntityType.COMPANY, EntityType.COMPANY),
                    (EntityType.SECTOR, EntityType.SECTOR),
                    (EntityType.NEWS_EVENT, EntityType.STOCK),
                    (EntityType.ANALYST_RATING, EntityType.STOCK)
                ],
                "properties": ["influence_strength", "influence_type", "time_lag"],
                "bidirectional": False
            },
            
            RelationType.COMPETES_WITH: {
                "valid_pairs": [
                    (EntityType.COMPANY, EntityType.COMPANY),
                    (EntityType.STOCK, EntityType.STOCK)
                ],
                "properties": ["competition_intensity", "market_overlap"],
                "bidirectional": True
            },
            
            RelationType.CORRELATES_WITH: {
                "valid_pairs": [
                    (EntityType.STOCK, EntityType.STOCK),
                    (EntityType.SECTOR, EntityType.SECTOR)
                ],
                "properties": ["correlation_coefficient", "time_period", "significance"],
                "bidirectional": True
            },
            
            RelationType.HAS_METRIC: {
                "valid_pairs": [
                    (EntityType.STOCK, EntityType.FINANCIAL_METRIC),
                    (EntityType.COMPANY, EntityType.FINANCIAL_METRIC),
                    (EntityType.SECTOR, EntityType.FINANCIAL_METRIC)
                ],
                "properties": ["metric_type", "reporting_frequency"],
                "bidirectional": False
            },
            
            RelationType.RATED_BY: {
                "valid_pairs": [
                    (EntityType.STOCK, EntityType.ANALYST_RATING)
                ],
                "properties": ["rating_date", "confidence_level"],
                "bidirectional": False
            },
            
            RelationType.AFFECTED_BY: {
                "valid_pairs": [
                    (EntityType.STOCK, EntityType.NEWS_EVENT),
                    (EntityType.COMPANY, EntityType.NEWS_EVENT),
                    (EntityType.SECTOR, EntityType.NEWS_EVENT)
                ],
                "properties": ["impact_magnitude", "impact_direction", "time_delay"],
                "bidirectional": False
            },
            
            RelationType.GENERATES_HEAT: {
                "valid_pairs": [
                    (EntityType.NEWS_EVENT, EntityType.HEAT_SOURCE),
                    (EntityType.ANALYST_RATING, EntityType.HEAT_SOURCE),
                    (EntityType.FINANCIAL_METRIC, EntityType.HEAT_SOURCE)
                ],
                "properties": ["heat_intensity", "generation_mechanism"],
                "bidirectional": False
            },
            
            RelationType.PROPAGATES_TO: {
                "valid_pairs": [
                    (EntityType.HEAT_SOURCE, EntityType.STOCK),
                    (EntityType.STOCK, EntityType.STOCK),
                    (EntityType.STOCK, EntityType.SECTOR),
                    (EntityType.SECTOR, EntityType.MARKET)
                ],
                "properties": ["propagation_strength", "decay_factor", "path_length"],
                "bidirectional": False
            }
        }
    
    def validate_entity(self, entity: OntologyEntity) -> bool:
        """Validate that an entity conforms to the ontology"""
        if entity.type not in self.entity_definitions:
            return False
            
        definition = self.entity_definitions[entity.type]
        
        # Check required properties
        for required_prop in definition["required_properties"]:
            if required_prop not in entity.properties:
                return False
                
        # Check property types
        constraints = definition.get("constraints", {})
        for prop, expected_type in constraints.items():
            if prop in entity.properties:
                if not isinstance(entity.properties[prop], expected_type):
                    return False
                    
        return True
    
    def validate_relation(self, relation: OntologyRelation, 
                         source_entity: OntologyEntity, 
                         target_entity: OntologyEntity) -> bool:
        """Validate that a relationship conforms to the ontology"""
        if relation.relation_type not in self.relation_definitions:
            return False
            
        definition = self.relation_definitions[relation.relation_type]
        
        # Check if the entity pair is valid for this relation type
        entity_pair = (source_entity.type, target_entity.type)
        if entity_pair not in definition["valid_pairs"]:
            return False
            
        # Check strength is in valid range
        if not 0.0 <= relation.strength <= 1.0:
            return False
            
        return True
    
    def get_schema_cypher(self) -> List[str]:
        """Generate Cypher queries to create the ontology schema in Neo4j"""
        queries = []
        
        # Create constraints for unique entity IDs
        for entity_type in EntityType:
            queries.append(f"""
                CREATE CONSTRAINT {entity_type.value.lower()}_id_unique 
                IF NOT EXISTS FOR (n:{entity_type.value}) 
                REQUIRE n.id IS UNIQUE
            """)
            
        # Create indexes for better performance
        performance_indexes = [
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX stock_symbol_index IF NOT EXISTS FOR (n:Stock) ON (n.symbol)",
            "CREATE INDEX company_sector_index IF NOT EXISTS FOR (n:Company) ON (n.sector)",
            "CREATE INDEX heat_timestamp_index IF NOT EXISTS FOR (n:HeatSource) ON (n.timestamp)"
        ]
        
        queries.extend(performance_indexes)
        
        return queries