#!/usr/bin/env python3
"""
Simple script to populate Neo4j with financial data without ontology validation
"""

import logging
from neo4j import GraphDatabase
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleNeo4jPopulator:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def clear_database(self):
        """Clear all data"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    
    def create_constraints(self):
        """Create basic constraints"""
        constraints = [
            "CREATE CONSTRAINT market_id_unique IF NOT EXISTS FOR (n:Market) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT sector_id_unique IF NOT EXISTS FOR (n:Sector) REQUIRE n.id IS UNIQUE", 
            "CREATE CONSTRAINT company_id_unique IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT stock_id_unique IF NOT EXISTS FOR (n:Stock) REQUIRE n.id IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint.split()[2]}")
                except Exception as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Constraint creation failed: {e}")
    
    def populate_financial_data(self):
        """Populate the financial knowledge graph"""
        
        with self.driver.session() as session:
            # Create Market node
            session.run("""
                MERGE (m:Market {id: 'US_MARKET'})
                SET m.name = 'US Stock Market',
                    m.region = 'North America',
                    m.currency = 'USD',
                    m.created_at = $timestamp
            """, timestamp=datetime.now().isoformat())
            
            # Define sectors
            sectors = [
                {'id': 'SECTOR_TECHNOLOGY', 'name': 'Technology', 'market_cap_pct': 28.5},
                {'id': 'SECTOR_HEALTHCARE', 'name': 'Healthcare', 'market_cap_pct': 13.2},
                {'id': 'SECTOR_FINANCE', 'name': 'Finance', 'market_cap_pct': 11.8},
                {'id': 'SECTOR_ENERGY', 'name': 'Energy', 'market_cap_pct': 4.2},
                {'id': 'SECTOR_CONSUMER_GOODS', 'name': 'Consumer Goods', 'market_cap_pct': 14.3}
            ]
            
            # Create sector nodes and relationships
            for sector in sectors:
                session.run("""
                    MERGE (s:Sector {id: $sector_id})
                    SET s.name = $name,
                        s.market_cap_percentage = $market_cap_pct,
                        s.created_at = $timestamp
                """, 
                sector_id=sector['id'], 
                name=sector['name'],
                market_cap_pct=sector['market_cap_pct'],
                timestamp=datetime.now().isoformat())
                
                # Create Market -> Sector relationship
                session.run("""
                    MATCH (m:Market {id: 'US_MARKET'})
                    MATCH (s:Sector {id: $sector_id})
                    MERGE (m)-[r:CONTAINS]->(s)
                    SET r.strength = $strength
                """, 
                sector_id=sector['id'],
                strength=sector['market_cap_pct']/100)
            
            # Define companies
            companies = [
                {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'SECTOR_TECHNOLOGY', 'exchange': 'NASDAQ'},
                {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': 'SECTOR_TECHNOLOGY', 'exchange': 'NASDAQ'},
                {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'sector': 'SECTOR_TECHNOLOGY', 'exchange': 'NASDAQ'},
                {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'sector': 'SECTOR_TECHNOLOGY', 'exchange': 'NASDAQ'},
                {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'sector': 'SECTOR_TECHNOLOGY', 'exchange': 'NASDAQ'},
                
                {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'sector': 'SECTOR_HEALTHCARE', 'exchange': 'NYSE'},
                {'symbol': 'PFE', 'name': 'Pfizer Inc.', 'sector': 'SECTOR_HEALTHCARE', 'exchange': 'NYSE'},
                {'symbol': 'UNH', 'name': 'UnitedHealth Group', 'sector': 'SECTOR_HEALTHCARE', 'exchange': 'NYSE'},
                
                {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.', 'sector': 'SECTOR_FINANCE', 'exchange': 'NYSE'},
                {'symbol': 'BAC', 'name': 'Bank of America Corporation', 'sector': 'SECTOR_FINANCE', 'exchange': 'NYSE'},
                {'symbol': 'GS', 'name': 'The Goldman Sachs Group Inc.', 'sector': 'SECTOR_FINANCE', 'exchange': 'NYSE'},
                
                {'symbol': 'XOM', 'name': 'Exxon Mobil Corporation', 'sector': 'SECTOR_ENERGY', 'exchange': 'NYSE'},
                {'symbol': 'CVX', 'name': 'Chevron Corporation', 'sector': 'SECTOR_ENERGY', 'exchange': 'NYSE'},
                
                {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'sector': 'SECTOR_CONSUMER_GOODS', 'exchange': 'NASDAQ'},
                {'symbol': 'WMT', 'name': 'Walmart Inc.', 'sector': 'SECTOR_CONSUMER_GOODS', 'exchange': 'NYSE'}
            ]
            
            # Create company and stock nodes
            for company in companies:
                # Create Company node
                session.run("""
                    MERGE (c:Company {id: $company_id})
                    SET c.name = $name,
                        c.symbol = $symbol,
                        c.created_at = $timestamp
                """, 
                company_id=f"COMPANY_{company['symbol']}",
                name=company['name'],
                symbol=company['symbol'],
                timestamp=datetime.now().isoformat())
                
                # Create Stock node
                session.run("""
                    MERGE (s:Stock {id: $stock_id})
                    SET s.symbol = $symbol,
                        s.name = $stock_name,
                        s.exchange = $exchange,
                        s.created_at = $timestamp
                """, 
                stock_id=f"STOCK_{company['symbol']}",
                symbol=company['symbol'],
                stock_name=f"{company['symbol']} Stock",
                exchange=company['exchange'],
                timestamp=datetime.now().isoformat())
                
                # Create Sector -> Company relationship
                session.run("""
                    MATCH (sector:Sector {id: $sector_id})
                    MATCH (company:Company {id: $company_id})
                    MERGE (sector)-[r:CONTAINS]->(company)
                    SET r.strength = 1.0
                """, 
                sector_id=company['sector'],
                company_id=f"COMPANY_{company['symbol']}")
                
                # Create Stock -> Company relationship
                session.run("""
                    MATCH (stock:Stock {id: $stock_id})
                    MATCH (company:Company {id: $company_id})
                    MERGE (stock)-[r:BELONGS_TO]->(company)
                    SET r.strength = 1.0
                """, 
                stock_id=f"STOCK_{company['symbol']}",
                company_id=f"COMPANY_{company['symbol']}")
            
            # Create some competitive relationships
            competitions = [
                ('AAPL', 'GOOGL', 0.7),
                ('AAPL', 'MSFT', 0.6), 
                ('GOOGL', 'MSFT', 0.8),
                ('META', 'GOOGL', 0.9),
                ('JNJ', 'PFE', 0.8),
                ('JPM', 'BAC', 0.9),
                ('JPM', 'GS', 0.7),
                ('XOM', 'CVX', 0.95),
                ('AMZN', 'WMT', 0.8)
            ]
            
            for comp1, comp2, strength in competitions:
                session.run("""
                    MATCH (c1:Company {id: $comp1_id})
                    MATCH (c2:Company {id: $comp2_id})
                    MERGE (c1)-[r:COMPETES_WITH]-(c2)
                    SET r.strength = $strength
                """, 
                comp1_id=f"COMPANY_{comp1}",
                comp2_id=f"COMPANY_{comp2}",
                strength=strength)
            
            logger.info("Financial data population completed")
    
    def get_statistics(self):
        """Get graph statistics"""
        with self.driver.session() as session:
            # Count nodes by type
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
            """).data()
            
            # Count relationships by type
            rel_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
            """).data()
            
            total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            return {
                'total_nodes': total_nodes,
                'total_relationships': total_rels,
                'node_types': node_counts,
                'relationship_types': rel_counts
            }

def main():
    populator = SimpleNeo4jPopulator()
    
    try:
        logger.info("Starting simple Neo4j population...")
        
        # Create constraints
        populator.create_constraints()
        
        # Clear existing data  
        populator.clear_database()
        
        # Populate data
        populator.populate_financial_data()
        
        # Get statistics
        stats = populator.get_statistics()
        logger.info(f"Population complete! Statistics: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Population failed: {e}")
        return False
    finally:
        populator.close()

if __name__ == "__main__":
    main()