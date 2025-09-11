#!/usr/bin/env python3
"""
Neo4j Visualization Fix - Check and fix dynamic graph visualization in Neo4j browser
"""

import os
import sys
from datetime import datetime
from graph.advanced_neo4j_manager import AdvancedNeo4jManager

def setup_neo4j_visualization():
    """Set up proper Neo4j visualization with heat-based colors."""
    
    # Get credentials from environment
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'neo4j123')
    
    print("=== Neo4j Visualization Fix ===")
    print(f"Connecting to Neo4j at {neo4j_uri}...")
    
    try:
        manager = AdvancedNeo4jManager(neo4j_uri, 'neo4j', neo4j_password)
        print("✓ Connected to Neo4j successfully")
        
        # Check current data structure
        print("\n1. Checking current stock nodes with heat properties...")
        query = '''
        MATCH (n:Stock) 
        RETURN n.symbol, n.heat_score, n.background_color, n.heat_level, n.current_price, n.last_updated
        ORDER BY n.heat_score DESC NULLS LAST
        LIMIT 10
        '''
        
        result = manager.run_query(query)
        print(f"Found {len(result)} stock nodes:")
        for record in result:
            print(f"  {record['n.symbol']}: Heat={record['n.heat_score']}, Color={record['n.background_color']}, Level={record['n.heat_level']}")
        
        # Create visualization style configuration
        print("\n2. Setting up Neo4j browser style configuration...")
        
        # Create a visualization style query for Neo4j browser
        style_config = '''
        // Neo4j Browser Style Configuration for RAGHeat Visualization
        // Copy and paste this into Neo4j Browser Style tab
        
        node {
          diameter: 50px;
          color: #A5ABB6;
          border-color: #9AA1AC;
          border-width: 2px;
          text-color-internal: #FFFFFF;
          font-size: 10px;
        }
        
        node.Stock {
          diameter: 60px;
          caption: {symbol};
        }
        
        // Heat-based coloring - these will be applied based on heat_level property
        node.Stock[heat_level = "cold"] {
          color: #87CEEB;
          border-color: #4682B4;
        }
        
        node.Stock[heat_level = "cool"] {
          color: #98FB98;
          border-color: #32CD32;
        }
        
        node.Stock[heat_level = "warm"] {
          color: #FFD700;
          border-color: #FFA500;
        }
        
        node.Stock[heat_level = "hot"] {
          color: #FF6347;
          border-color: #FF4500;
        }
        
        node.Stock[heat_level = "extreme"] {
          color: #DC143C;
          border-color: #8B0000;
          diameter: 70px;
        }
        
        relationship {
          color: #A5ABB6;
          shaft-width: 1px;
          font-size: 8px;
          padding: 3px;
          text-color-external: #000000;
          text-color-internal: #FFFFFF;
          caption: '<type>';
        }
        
        relationship.CORRELATES {
          color: #4A90E2;
          shaft-width: 2px;
        }
        
        relationship.SECTOR_OF {
          color: #50E3C2;
          shaft-width: 3px;
        }
        '''
        
        # Save style configuration to file
        style_file = "/Users/rajeshgupta/PycharmProjects/ragheat-poc/ragheat-poc/backend/neo4j_browser_style.grass"
        with open(style_file, 'w') as f:
            f.write(style_config)
        print(f"✓ Style configuration saved to: {style_file}")
        
        # Update all nodes with proper heat levels if missing
        print("\n3. Ensuring all stock nodes have proper heat level properties...")
        update_query = '''
        MATCH (s:Stock)
        WHERE s.heat_score IS NOT NULL AND s.heat_level IS NULL
        SET s.heat_level = CASE
            WHEN s.heat_score <= 0.2 THEN 'cold'
            WHEN s.heat_score <= 0.4 THEN 'cool'
            WHEN s.heat_score <= 0.6 THEN 'warm'
            WHEN s.heat_score <= 0.8 THEN 'hot'
            ELSE 'extreme'
        END
        RETURN count(s) as updated_count
        '''
        
        update_result = manager.run_query(update_query)
        if update_result:
            print(f"✓ Updated {update_result[0]['updated_count']} nodes with heat levels")
        
        # Create comprehensive visualization query
        print("\n4. Creating visualization queries for Neo4j browser...")
        
        viz_queries = [
            {
                "name": "Heat Map Overview",
                "query": '''
                MATCH (s:Stock)
                WHERE s.heat_score IS NOT NULL
                RETURN s
                ORDER BY s.heat_score DESC
                LIMIT 20
                ''',
                "description": "Shows top 20 heated stocks with heat-based coloring"
            },
            {
                "name": "Full Graph with Heat",
                "query": '''
                MATCH (s:Stock)-[r]-(connected)
                WHERE s.heat_score IS NOT NULL
                RETURN s, r, connected
                LIMIT 100
                ''',
                "description": "Shows interconnected stocks with relationships and heat visualization"
            },
            {
                "name": "Extreme Heat Nodes",
                "query": '''
                MATCH (s:Stock)
                WHERE s.heat_level = 'extreme' OR s.heat_score > 0.7
                OPTIONAL MATCH (s)-[r]-(connected)
                RETURN s, r, connected
                ''',
                "description": "Focus on extremely heated stocks and their connections"
            },
            {
                "name": "Live Heat Updates",
                "query": '''
                MATCH (s:Stock)
                WHERE datetime(s.last_updated) > datetime() - duration('PT5M')
                OPTIONAL MATCH (s)-[r:CORRELATES]-(other)
                RETURN s, r, other
                ORDER BY s.heat_score DESC
                ''',
                "description": "Shows recently updated stocks with current heat levels"
            }
        ]
        
        # Save visualization queries
        queries_file = "/Users/rajeshgupta/PycharmProjects/ragheat-poc/ragheat-poc/backend/neo4j_visualization_queries.cypher"
        with open(queries_file, 'w') as f:
            f.write("// RAGHeat Neo4j Visualization Queries\n")
            f.write("// Copy and paste these into Neo4j Browser\n\n")
            for i, viz in enumerate(viz_queries, 1):
                f.write(f"// {i}. {viz['name']}\n")
                f.write(f"// {viz['description']}\n")
                f.write(viz['query'])
                f.write("\n\n" + "="*60 + "\n\n")
        
        print(f"✓ Visualization queries saved to: {queries_file}")
        
        # Test the visualization queries
        print("\n5. Testing visualization queries...")
        for viz in viz_queries[:2]:  # Test first two queries
            print(f"  Testing: {viz['name']}")
            result = manager.run_query(viz['query'])
            print(f"    ✓ Returns {len(result)} records")
        
        # Check for real-time updates
        print("\n6. Checking real-time heat updates...")
        recent_query = '''
        MATCH (s:Stock)
        WHERE datetime(s.last_updated) > datetime() - duration('PT10M')
        RETURN s.symbol, s.heat_score, s.heat_level, s.last_updated
        ORDER BY s.heat_score DESC
        LIMIT 5
        '''
        
        recent_result = manager.run_query(recent_query)
        if recent_result:
            print("Recently updated stocks (last 10 minutes):")
            for record in recent_result:
                print(f"  {record['s.symbol']}: Heat={record['s.heat_score']:.4f}, Level={record['s.heat_level']}, Updated={record['s.last_updated']}")
        else:
            print("⚠ No recent updates found - check if live data is updating")
        
        manager.close()
        
        # Provide instructions
        print("\n" + "="*60)
        print("INSTRUCTIONS FOR NEO4J BROWSER VISUALIZATION:")
        print("="*60)
        print()
        print("1. Open Neo4j Browser at: http://localhost:7474")
        print("2. Login with username: neo4j, password: neo4j123")
        print()
        print("3. Import the style configuration:")
        print("   - Click the gear icon (⚙) in the bottom left")
        print("   - Go to 'Browser Settings' tab")
        print("   - Scroll to 'Graph Stylesheet (*.grass)'")
        print(f"   - Copy content from: {style_file}")
        print("   - Paste into the stylesheet box and click 'Apply'")
        print()
        print("4. Run visualization queries:")
        print(f"   - Open: {queries_file}")
        print("   - Copy and paste queries into Neo4j Browser")
        print("   - Start with 'Heat Map Overview' query")
        print()
        print("5. Expected behavior:")
        print("   - Cold stocks (heat ≤ 0.2): Light blue")
        print("   - Cool stocks (0.2 < heat ≤ 0.4): Light green")
        print("   - Warm stocks (0.4 < heat ≤ 0.6): Yellow/Gold")
        print("   - Hot stocks (0.6 < heat ≤ 0.8): Orange/Red")
        print("   - Extreme stocks (heat > 0.8): Dark red (larger size)")
        print()
        print("6. Real-time updates:")
        print("   - Re-run queries to see latest heat values")
        print("   - Heat updates every ~60 seconds from live API")
        print("   - Node colors should change based on current heat_level")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = setup_neo4j_visualization()
    if success:
        print("\n✓ Neo4j visualization setup completed successfully!")
        print("Open Neo4j Browser and follow the instructions above.")
    else:
        print("\n✗ Failed to setup Neo4j visualization")
        sys.exit(1)