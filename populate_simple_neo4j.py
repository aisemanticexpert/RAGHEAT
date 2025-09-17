#!/usr/bin/env python3
"""
Simple Neo4j population script for financial knowledge graph
"""

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Connect to Neo4j
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    try:
        with driver.session() as session:
            # Clear existing data
            print("Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create Market
            print("Creating market...")
            session.run("""
                CREATE (m:Market {
                    id: 'US_MARKET', 
                    name: 'US Stock Market', 
                    region: 'North America',
                    currency: 'USD'
                })
            """)
            
            # Create Sectors
            print("Creating sectors...")
            session.run("""
                CREATE (tech:Sector {id: 'TECH', name: 'Technology', code: 'TECHNOLOGY'}),
                       (health:Sector {id: 'HEALTH', name: 'Healthcare', code: 'HEALTHCARE'}),
                       (finance:Sector {id: 'FINANCE', name: 'Finance', code: 'FINANCE'}),
                       (energy:Sector {id: 'ENERGY', name: 'Energy', code: 'ENERGY'}),
                       (consumer:Sector {id: 'CONSUMER', name: 'Consumer Goods', code: 'CONSUMER_GOODS'})
            """)
            
            # Create Companies
            print("Creating companies...")
            session.run("""
                CREATE (aapl_company:Company {id: 'AAPL_CO', name: 'Apple Inc.', sector: 'TECHNOLOGY', ticker: 'AAPL'}),
                       (googl_company:Company {id: 'GOOGL_CO', name: 'Alphabet Inc.', sector: 'TECHNOLOGY', ticker: 'GOOGL'}),
                       (msft_company:Company {id: 'MSFT_CO', name: 'Microsoft Corporation', sector: 'TECHNOLOGY', ticker: 'MSFT'}),
                       (meta_company:Company {id: 'META_CO', name: 'Meta Platforms Inc.', sector: 'TECHNOLOGY', ticker: 'META'}),
                       (nvda_company:Company {id: 'NVDA_CO', name: 'NVIDIA Corporation', sector: 'TECHNOLOGY', ticker: 'NVDA'}),
                       (jnj_company:Company {id: 'JNJ_CO', name: 'Johnson & Johnson', sector: 'HEALTHCARE', ticker: 'JNJ'}),
                       (jpm_company:Company {id: 'JPM_CO', name: 'JPMorgan Chase & Co.', sector: 'FINANCE', ticker: 'JPM'}),
                       (xom_company:Company {id: 'XOM_CO', name: 'Exxon Mobil Corporation', sector: 'ENERGY', ticker: 'XOM'}),
                       (amzn_company:Company {id: 'AMZN_CO', name: 'Amazon.com Inc.', sector: 'CONSUMER_GOODS', ticker: 'AMZN'})
            """)
            
            # Create Stocks
            print("Creating stocks...")
            session.run("""
                CREATE (aapl_stock:Stock {id: 'AAPL_ST', symbol: 'AAPL', exchange: 'NASDAQ'}),
                       (googl_stock:Stock {id: 'GOOGL_ST', symbol: 'GOOGL', exchange: 'NASDAQ'}),
                       (msft_stock:Stock {id: 'MSFT_ST', symbol: 'MSFT', exchange: 'NASDAQ'}),
                       (meta_stock:Stock {id: 'META_ST', symbol: 'META', exchange: 'NASDAQ'}),
                       (nvda_stock:Stock {id: 'NVDA_ST', symbol: 'NVDA', exchange: 'NASDAQ'}),
                       (jnj_stock:Stock {id: 'JNJ_ST', symbol: 'JNJ', exchange: 'NYSE'}),
                       (jpm_stock:Stock {id: 'JPM_ST', symbol: 'JPM', exchange: 'NYSE'}),
                       (xom_stock:Stock {id: 'XOM_ST', symbol: 'XOM', exchange: 'NYSE'}),
                       (amzn_stock:Stock {id: 'AMZN_ST', symbol: 'AMZN', exchange: 'NASDAQ'})
            """)
            
            # Create Market-Sector relationships
            print("Creating market-sector relationships...")
            session.run("""
                MATCH (m:Market {id: 'US_MARKET'})
                MATCH (tech:Sector {id: 'TECH'})
                MATCH (health:Sector {id: 'HEALTH'})
                MATCH (finance:Sector {id: 'FINANCE'})
                MATCH (energy:Sector {id: 'ENERGY'})
                MATCH (consumer:Sector {id: 'CONSUMER'})
                CREATE (m)-[:CONTAINS]->(tech),
                       (m)-[:CONTAINS]->(health),
                       (m)-[:CONTAINS]->(finance),
                       (m)-[:CONTAINS]->(energy),
                       (m)-[:CONTAINS]->(consumer)
            """)
            
            # Create Sector-Company relationships
            print("Creating sector-company relationships...")
            session.run("""
                MATCH (tech:Sector {id: 'TECH'})
                MATCH (aapl_company:Company {id: 'AAPL_CO'})
                MATCH (googl_company:Company {id: 'GOOGL_CO'})
                MATCH (msft_company:Company {id: 'MSFT_CO'})
                MATCH (meta_company:Company {id: 'META_CO'})
                MATCH (nvda_company:Company {id: 'NVDA_CO'})
                CREATE (tech)-[:CONTAINS]->(aapl_company),
                       (tech)-[:CONTAINS]->(googl_company),
                       (tech)-[:CONTAINS]->(msft_company),
                       (tech)-[:CONTAINS]->(meta_company),
                       (tech)-[:CONTAINS]->(nvda_company)
            """)
            
            session.run("""
                MATCH (health:Sector {id: 'HEALTH'})
                MATCH (jnj_company:Company {id: 'JNJ_CO'})
                CREATE (health)-[:CONTAINS]->(jnj_company)
            """)
            
            session.run("""
                MATCH (finance:Sector {id: 'FINANCE'})
                MATCH (jpm_company:Company {id: 'JPM_CO'})
                CREATE (finance)-[:CONTAINS]->(jpm_company)
            """)
            
            session.run("""
                MATCH (energy:Sector {id: 'ENERGY'})
                MATCH (xom_company:Company {id: 'XOM_CO'})
                CREATE (energy)-[:CONTAINS]->(xom_company)
            """)
            
            session.run("""
                MATCH (consumer:Sector {id: 'CONSUMER'})
                MATCH (amzn_company:Company {id: 'AMZN_CO'})
                CREATE (consumer)-[:CONTAINS]->(amzn_company)
            """)
            
            # Create Stock-Company relationships
            print("Creating stock-company relationships...")
            session.run("""
                MATCH (aapl_stock:Stock {id: 'AAPL_ST'})
                MATCH (aapl_company:Company {id: 'AAPL_CO'})
                MATCH (googl_stock:Stock {id: 'GOOGL_ST'})
                MATCH (googl_company:Company {id: 'GOOGL_CO'})
                MATCH (msft_stock:Stock {id: 'MSFT_ST'})
                MATCH (msft_company:Company {id: 'MSFT_CO'})
                MATCH (meta_stock:Stock {id: 'META_ST'})
                MATCH (meta_company:Company {id: 'META_CO'})
                MATCH (nvda_stock:Stock {id: 'NVDA_ST'})
                MATCH (nvda_company:Company {id: 'NVDA_CO'})
                CREATE (aapl_stock)-[:BELONGS_TO]->(aapl_company),
                       (googl_stock)-[:BELONGS_TO]->(googl_company),
                       (msft_stock)-[:BELONGS_TO]->(msft_company),
                       (meta_stock)-[:BELONGS_TO]->(meta_company),
                       (nvda_stock)-[:BELONGS_TO]->(nvda_company)
            """)
            
            session.run("""
                MATCH (jnj_stock:Stock {id: 'JNJ_ST'})
                MATCH (jnj_company:Company {id: 'JNJ_CO'})
                MATCH (jpm_stock:Stock {id: 'JPM_ST'})
                MATCH (jpm_company:Company {id: 'JPM_CO'})
                MATCH (xom_stock:Stock {id: 'XOM_ST'})
                MATCH (xom_company:Company {id: 'XOM_CO'})
                MATCH (amzn_stock:Stock {id: 'AMZN_ST'})
                MATCH (amzn_company:Company {id: 'AMZN_CO'})
                CREATE (jnj_stock)-[:BELONGS_TO]->(jnj_company),
                       (jpm_stock)-[:BELONGS_TO]->(jpm_company),
                       (xom_stock)-[:BELONGS_TO]->(xom_company),
                       (amzn_stock)-[:BELONGS_TO]->(amzn_company)
            """)
            
            # Create competitive relationships
            print("Creating competitive relationships...")
            session.run("""
                MATCH (aapl_company:Company {id: 'AAPL_CO'})
                MATCH (googl_company:Company {id: 'GOOGL_CO'})
                MATCH (msft_company:Company {id: 'MSFT_CO'})
                MATCH (meta_company:Company {id: 'META_CO'})
                CREATE (aapl_company)-[:COMPETES_WITH {strength: 0.8}]->(googl_company),
                       (aapl_company)-[:COMPETES_WITH {strength: 0.7}]->(msft_company),
                       (googl_company)-[:COMPETES_WITH {strength: 0.9}]->(meta_company),
                       (msft_company)-[:COMPETES_WITH {strength: 0.6}]->(googl_company)
            """)
            
            print("✓ Neo4j populated with financial knowledge graph")
            
            # Verify data
            result = session.run('MATCH (n) RETURN count(n) as total_nodes')
            total_nodes = result.single()['total_nodes']
            
            result = session.run('MATCH ()-[r]-() RETURN count(r) as total_relationships')
            total_relationships = result.single()['total_relationships']
            
            print(f'✓ Created {total_nodes} nodes and {total_relationships} relationships')
            
            # Show node types
            result = session.run('MATCH (n) RETURN distinct labels(n) as node_types')
            node_types = [record['node_types'] for record in result]
            print(f'✓ Node types: {node_types}')
            
    except Exception as e:
        print(f"Error: {e}")
        return False
        
    finally:
        driver.close()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Knowledge graph populated successfully!")
        print("You can now view it in Neo4j Browser at: http://localhost:7474")
        print("Try this query to see the full graph: MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 50")
    else:
        print("❌ Failed to populate knowledge graph")