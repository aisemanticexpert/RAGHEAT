#!/usr/bin/env python3
"""
Script to populate Neo4j with financial knowledge graph data
"""

import sys
import os
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.neo4j_knowledge_graph import Neo4jKnowledgeGraph
from ontology.financial_ontology import (
    EntityType, RelationType, OntologyEntity, OntologyRelation
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_financial_entities(kg: Neo4jKnowledgeGraph):
    """Create the core financial entities"""
    
    # Create Market entity
    market = OntologyEntity(
        id="US_MARKET",
        type=EntityType.MARKET,
        name="US Stock Market",
        properties={
            "region": "North America",
            "currency": "USD",
            "total_market_cap": 45000000000000,  # ~45 trillion
            "trading_hours": "9:30 AM - 4:00 PM EST",
            "timezone": "America/New_York"
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    kg.create_entity(market)
    
    # Define sectors with their properties
    sectors_data = {
        "TECHNOLOGY": {
            "name": "Technology",
            "description": "Companies involved in software, hardware, and technology services",
            "market_cap_percentage": 28.5,
            "volatility_index": 1.2,
            "beta": 1.15
        },
        "HEALTHCARE": {
            "name": "Healthcare", 
            "description": "Pharmaceutical, biotechnology, and health services companies",
            "market_cap_percentage": 13.2,
            "volatility_index": 0.9,
            "beta": 0.85
        },
        "FINANCE": {
            "name": "Finance",
            "description": "Banks, investment firms, and financial services companies",
            "market_cap_percentage": 11.8,
            "volatility_index": 1.4,
            "beta": 1.25
        },
        "ENERGY": {
            "name": "Energy",
            "description": "Oil, gas, and renewable energy companies",
            "market_cap_percentage": 4.2,
            "volatility_index": 1.8,
            "beta": 1.45
        },
        "CONSUMER_GOODS": {
            "name": "Consumer Goods",
            "description": "Retail, e-commerce, and consumer product companies",
            "market_cap_percentage": 14.3,
            "volatility_index": 1.1,
            "beta": 1.05
        }
    }
    
    # Create sector entities
    for sector_code, sector_info in sectors_data.items():
        sector = OntologyEntity(
            id=f"SECTOR_{sector_code}",
            type=EntityType.SECTOR,
            name=sector_info["name"],
            properties={
                "code": sector_code,
                "description": sector_info["description"],
                "market_cap_percentage": sector_info["market_cap_percentage"],
                "volatility_index": sector_info["volatility_index"],
                "beta": sector_info["beta"]
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        kg.create_entity(sector)
        
        # Create relationship: Market CONTAINS Sector
        market_sector_rel = OntologyRelation(
            id=f"MARKET_CONTAINS_{sector_code}",
            source_id="US_MARKET",
            target_id=f"SECTOR_{sector_code}",
            relation_type=RelationType.CONTAINS,
            properties={
                "containment_percentage": sector_info["market_cap_percentage"],
                "primary_flag": True
            },
            strength=sector_info["market_cap_percentage"] / 100,
            created_at=datetime.now()
        )
        kg.create_relationship(market_sector_rel, market, sector)
    
    # Define companies with their data
    companies_data = {
        "AAPL": {
            "name": "Apple Inc.",
            "sector": "TECHNOLOGY",
            "headquarters": "Cupertino, CA",
            "founded_year": 1976,
            "employees_count": 164000,
            "website": "https://www.apple.com",
            "ceo": "Tim Cook",
            "description": "Technology company focusing on consumer electronics, software, and services",
            "business_model": "Hardware and Services"
        },
        "GOOGL": {
            "name": "Alphabet Inc.",
            "sector": "TECHNOLOGY", 
            "headquarters": "Mountain View, CA",
            "founded_year": 1998,
            "employees_count": 190000,
            "website": "https://www.alphabet.com",
            "ceo": "Sundar Pichai",
            "description": "Technology conglomerate specializing in Internet services and AI",
            "business_model": "Advertising and Cloud Services"
        },
        "MSFT": {
            "name": "Microsoft Corporation",
            "sector": "TECHNOLOGY",
            "headquarters": "Redmond, WA", 
            "founded_year": 1975,
            "employees_count": 228000,
            "website": "https://www.microsoft.com",
            "ceo": "Satya Nadella",
            "description": "Technology company providing software, cloud services, and productivity tools",
            "business_model": "Software and Cloud Services"
        },
        "META": {
            "name": "Meta Platforms Inc.",
            "sector": "TECHNOLOGY",
            "headquarters": "Menlo Park, CA",
            "founded_year": 2004,
            "employees_count": 87000,
            "website": "https://www.meta.com",
            "ceo": "Mark Zuckerberg",
            "description": "Social media and virtual reality technology company",
            "business_model": "Social Media Advertising"
        },
        "NVDA": {
            "name": "NVIDIA Corporation",
            "sector": "TECHNOLOGY",
            "headquarters": "Santa Clara, CA",
            "founded_year": 1993,
            "employees_count": 29600,
            "website": "https://www.nvidia.com",
            "ceo": "Jensen Huang",
            "description": "Graphics processing and AI computing technology company",
            "business_model": "Semiconductor Hardware"
        },
        "JNJ": {
            "name": "Johnson & Johnson",
            "sector": "HEALTHCARE",
            "headquarters": "New Brunswick, NJ",
            "founded_year": 1886,
            "employees_count": 152000,
            "website": "https://www.jnj.com",
            "ceo": "Joaquin Duato",
            "description": "Multinational healthcare and pharmaceutical company",
            "business_model": "Healthcare Products and Pharmaceuticals"
        },
        "PFE": {
            "name": "Pfizer Inc.",
            "sector": "HEALTHCARE",
            "headquarters": "New York, NY",
            "founded_year": 1849,
            "employees_count": 83000,
            "website": "https://www.pfizer.com",
            "ceo": "Albert Bourla",
            "description": "Pharmaceutical and biotechnology company",
            "business_model": "Pharmaceutical Research and Development"
        },
        "UNH": {
            "name": "UnitedHealth Group",
            "sector": "HEALTHCARE",
            "headquarters": "Minnetonka, MN",
            "founded_year": 1977,
            "employees_count": 440000,
            "website": "https://www.unitedhealthgroup.com",
            "ceo": "Andrew Witty",
            "description": "Healthcare and health insurance company",
            "business_model": "Health Insurance and Services"
        },
        "JPM": {
            "name": "JPMorgan Chase & Co.",
            "sector": "FINANCE",
            "headquarters": "New York, NY",
            "founded_year": 1799,
            "employees_count": 296000,
            "website": "https://www.jpmorganchase.com",
            "ceo": "Jamie Dimon",
            "description": "Multinational investment bank and financial services company",
            "business_model": "Banking and Financial Services"
        },
        "BAC": {
            "name": "Bank of America Corporation",
            "sector": "FINANCE",
            "headquarters": "Charlotte, NC",
            "founded_year": 1904,
            "employees_count": 216000,
            "website": "https://www.bankofamerica.com",
            "ceo": "Brian Moynihan",
            "description": "Multinational investment bank and financial services company",
            "business_model": "Consumer and Commercial Banking"
        },
        "GS": {
            "name": "The Goldman Sachs Group Inc.",
            "sector": "FINANCE",
            "headquarters": "New York, NY",
            "founded_year": 1869,
            "employees_count": 49100,
            "website": "https://www.goldmansachs.com",
            "ceo": "David Solomon",
            "description": "Multinational investment bank and financial services company",
            "business_model": "Investment Banking and Trading"
        },
        "XOM": {
            "name": "Exxon Mobil Corporation",
            "sector": "ENERGY",
            "headquarters": "Irving, TX",
            "founded_year": 1870,
            "employees_count": 63000,
            "website": "https://www.exxonmobil.com",
            "ceo": "Darren Woods",
            "description": "Multinational oil and gas corporation",
            "business_model": "Oil and Gas Exploration and Refining"
        },
        "CVX": {
            "name": "Chevron Corporation",
            "sector": "ENERGY",
            "headquarters": "San Ramon, CA",
            "founded_year": 1879,
            "employees_count": 47600,
            "website": "https://www.chevron.com",
            "ceo": "Mike Wirth",
            "description": "Multinational energy corporation",
            "business_model": "Oil and Gas Production and Refining"
        },
        "AMZN": {
            "name": "Amazon.com Inc.",
            "sector": "CONSUMER_GOODS",
            "headquarters": "Seattle, WA",
            "founded_year": 1994,
            "employees_count": 1540000,
            "website": "https://www.amazon.com",
            "ceo": "Andy Jassy",
            "description": "E-commerce, cloud computing, and digital services company",
            "business_model": "E-commerce and Cloud Services"
        },
        "WMT": {
            "name": "Walmart Inc.",
            "sector": "CONSUMER_GOODS",
            "headquarters": "Bentonville, AR",
            "founded_year": 1962,
            "employees_count": 2100000,
            "website": "https://www.walmart.com",
            "ceo": "Doug McMillon",
            "description": "Multinational retail corporation",
            "business_model": "Retail and E-commerce"
        }
    }
    
    # Create company and stock entities
    for symbol, company_info in companies_data.items():
        # Create company entity
        company = OntologyEntity(
            id=f"COMPANY_{symbol}",
            type=EntityType.COMPANY,
            name=company_info["name"],
            properties={
                "sector": company_info["sector"],
                "headquarters": company_info["headquarters"],
                "founded_year": company_info["founded_year"],
                "employees_count": company_info["employees_count"],
                "website": company_info["website"],
                "ceo": company_info["ceo"],
                "description": company_info["description"],
                "business_model": company_info["business_model"]
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        kg.create_entity(company)
        
        # Create stock entity
        stock = OntologyEntity(
            id=f"STOCK_{symbol}",
            type=EntityType.STOCK,
            name=f"{symbol} Stock",
            properties={
                "symbol": symbol,
                "exchange": "NASDAQ" if symbol in ["AAPL", "GOOGL", "MSFT", "META", "NVDA", "AMZN"] else "NYSE",
                "isin": f"US{symbol}001234",  # Mock ISIN
                "listing_date": "1980-12-12"  # Mock listing date
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        kg.create_entity(stock)
        
        # Create relationships
        # Sector CONTAINS Company
        sector_company_rel = OntologyRelation(
            id=f"SECTOR_CONTAINS_{symbol}",
            source_id=f"SECTOR_{company_info['sector']}",
            target_id=f"COMPANY_{symbol}",
            relation_type=RelationType.CONTAINS,
            properties={"primary_flag": True},
            strength=1.0,
            created_at=datetime.now()
        )
        
        # Stock BELONGS_TO Company
        stock_company_rel = OntologyRelation(
            id=f"STOCK_BELONGS_{symbol}",
            source_id=f"STOCK_{symbol}",
            target_id=f"COMPANY_{symbol}",
            relation_type=RelationType.BELONGS_TO,
            properties={"ownership_type": "public_stock"},
            strength=1.0,
            created_at=datetime.now()
        )
        
        # Execute relationships (we need dummy entities for validation)
        sector_entity = OntologyEntity("", EntityType.SECTOR, "", {}, datetime.now(), datetime.now())
        company_entity = company
        stock_entity = stock
        
        # Create the actual relationships in Neo4j
        kg.create_relationship(sector_company_rel, sector_entity, company_entity)
        kg.create_relationship(stock_company_rel, stock_entity, company_entity)

def create_competitive_relationships(kg: Neo4jKnowledgeGraph):
    """Create competitive relationships between similar companies"""
    
    # Technology sector competitions
    tech_competitions = [
        ("AAPL", "GOOGL", 0.7, "Mobile ecosystems and cloud services"),
        ("AAPL", "MSFT", 0.6, "Cloud services and productivity software"),
        ("GOOGL", "MSFT", 0.8, "Cloud computing and enterprise services"),
        ("META", "GOOGL", 0.9, "Digital advertising market"),
    ]
    
    # Healthcare competitions
    healthcare_competitions = [
        ("JNJ", "PFE", 0.8, "Pharmaceutical development and vaccines"),
        ("JNJ", "UNH", 0.4, "Healthcare services overlap"),
        ("PFE", "UNH", 0.3, "Healthcare insurance coverage of drugs")
    ]
    
    # Finance competitions
    finance_competitions = [
        ("JPM", "BAC", 0.9, "Commercial and investment banking"),
        ("JPM", "GS", 0.7, "Investment banking services"),
        ("BAC", "GS", 0.6, "Wealth management and trading")
    ]
    
    # Energy competitions
    energy_competitions = [
        ("XOM", "CVX", 0.95, "Oil and gas exploration and refining")
    ]
    
    # Consumer goods competitions
    consumer_competitions = [
        ("AMZN", "WMT", 0.8, "Retail and e-commerce")
    ]
    
    all_competitions = (tech_competitions + healthcare_competitions + 
                       finance_competitions + energy_competitions + consumer_competitions)
    
    for comp1, comp2, strength, description in all_competitions:
        # Create bidirectional competition relationship
        comp_rel = OntologyRelation(
            id=f"COMPETE_{comp1}_{comp2}",
            source_id=f"COMPANY_{comp1}",
            target_id=f"COMPANY_{comp2}",
            relation_type=RelationType.COMPETES_WITH,
            properties={
                "competition_intensity": strength,
                "competition_area": description,
                "market_overlap": strength * 0.8
            },
            strength=strength,
            created_at=datetime.now()
        )
        
        # For validation, create dummy entities
        dummy_entity = OntologyEntity("", EntityType.COMPANY, "", {}, datetime.now(), datetime.now())
        kg.create_relationship(comp_rel, dummy_entity, dummy_entity)

def populate_neo4j():
    """Main function to populate Neo4j with financial knowledge graph"""
    logger.info("Starting Neo4j population...")
    
    # Connect to Neo4j
    kg = Neo4jKnowledgeGraph()
    
    try:
        # Initialize schema
        logger.info("Initializing schema...")
        kg.initialize_schema()
        
        # Clear existing data (optional - comment out if you want to preserve data)
        logger.info("Clearing existing data...")
        kg.clear_database()
        
        # Create entities and relationships
        logger.info("Creating financial entities...")
        create_financial_entities(kg)
        
        logger.info("Creating competitive relationships...")
        create_competitive_relationships(kg)
        
        # Get statistics
        stats = kg.get_graph_statistics()
        logger.info(f"Population complete! Graph statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Error populating Neo4j: {e}")
    finally:
        kg.close()

if __name__ == "__main__":
    populate_neo4j()