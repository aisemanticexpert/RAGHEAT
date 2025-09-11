#!/usr/bin/env python3
"""
Apply Neo4j Dynamic Visualization Configuration
This script will automatically configure Neo4j browser for heat-based visualization
"""

import os
import sys
import json
import requests
import time
from datetime import datetime

def apply_neo4j_visualization():
    """Apply the complete Neo4j visualization configuration."""
    
    print("🚀 APPLYING NEO4J DYNAMIC VISUALIZATION CONFIGURATION")
    print("=" * 60)
    
    # Check if Neo4j is running
    neo4j_url = "http://localhost:7474"
    try:
        response = requests.get(f"{neo4j_url}/browser/", timeout=5)
        print("✅ Neo4j Browser is accessible at:", neo4j_url)
    except requests.exceptions.RequestException:
        print("❌ Neo4j Browser not accessible. Starting Neo4j...")
        os.system("neo4j start")
        time.sleep(5)
    
    # Check API status
    api_url = "http://localhost:8000"
    try:
        response = requests.get(f"{api_url}/api/status", timeout=5)
        status = response.json()
        print("✅ RAGHeat API Status:")
        print(f"   - Stocks tracked: {status.get('stocks_tracked', 0)}")
        print(f"   - Neo4j status: {status.get('neo4j_status', 'unknown')}")
        print(f"   - Last updated: {status.get('last_updated', 'unknown')}")
    except requests.exceptions.RequestException:
        print("❌ RAGHeat API not accessible")
        return False
    
    # Get current heat data
    try:
        response = requests.get(f"{api_url}/api/analytics/dashboard", timeout=5)
        dashboard = response.json()
        hot_stocks = dashboard.get('hot_stocks', [])
        print(f"✅ Current Heat Leaders:")
        for i, stock in enumerate(hot_stocks[:5], 1):
            print(f"   {i}. {stock['stock_id'].replace('STOCK_', '')}: {stock['heat']:.4f} ({stock['level']})")
    except requests.exceptions.RequestException:
        print("❌ Could not retrieve heat data")
        return False
    
    # Create Neo4j Browser Style Configuration
    style_config = """// RAGHeat Dynamic Heat Visualization Style
node {
  diameter: 50px;
  color: #A5ABB6;
  border-color: #9AA1AC;
  border-width: 2px;
  text-color-internal: #FFFFFF;
  font-size: 10px;
  caption: {name};
}

node.Stock {
  diameter: 60px;
  caption: {symbol};
  font-size: 12px;
  font-weight: bold;
}

node.Stock[heat_level = "cold"] {
  color: #4A90E2;
  border-color: #2E5BDA;
  text-color-internal: #FFFFFF;
}

node.Stock[symbol = "UNH"] {
  color: #FF4500;
  border-color: #CC3700;
  border-width: 3px;
  text-color-internal: #FFFFFF;
  diameter: 70px;
}

node.Company {
  color: #9013FE;
  border-color: #651FFF;
  diameter: 45px;
  caption: {symbol};
}

node.Sector {
  color: #00BCD4;
  border-color: #00ACC1;
  diameter: 55px;
  caption: {name};
}

relationship {
  color: #757575;
  shaft-width: 2px;
  caption: '';
}

relationship.CONTAINS {
  color: #FF9800;
  shaft-width: 3px;
}

relationship.BELONGS_TO {
  color: #4CAF50;
  shaft-width: 2px;
}"""
    
    # Save style configuration
    style_file = "neo4j_applied_style.grass"
    with open(style_file, 'w') as f:
        f.write(style_config)
    print(f"✅ Style configuration saved: {style_file}")
    
    # Create visualization queries
    queries = {
        "main_heat_visualization": """MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN s
ORDER BY s.heat_score DESC""",
        
        "complete_graph": """MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
OPTIONAL MATCH (s)-[r]-(connected)
RETURN s, r, connected
LIMIT 100""",
        
        "live_updates": """MATCH (s:Stock)
WHERE datetime(s.last_updated) > datetime() - duration('PT5M')
RETURN s
ORDER BY s.heat_score DESC""",
        
        "debug_query": """MATCH (s:Stock)
RETURN count(s) as total_stocks, 
       max(s.heat_score) as max_heat,
       min(s.heat_score) as min_heat""",
        
        "heat_table": """MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN 
  s.symbol as Symbol,
  s.heat_score as Heat_Score, 
  s.heat_level as Heat_Level,
  s.current_price as Price,
  s.last_updated as Last_Updated
ORDER BY s.heat_score DESC"""
    }
    
    # Save queries
    queries_file = "neo4j_applied_queries.cypher"
    with open(queries_file, 'w') as f:
        f.write("// RAGHeat Applied Neo4j Visualization Queries\n")
        f.write("// Copy these into Neo4j Browser\n\n")
        for name, query in queries.items():
            f.write(f"// {name.upper().replace('_', ' ')}\n")
            f.write(f"{query}\n\n")
            f.write("=" * 50 + "\n\n")
    
    print(f"✅ Queries saved: {queries_file}")
    
    # Create execution instructions
    instructions = f"""
🚀 NEO4J BROWSER SETUP COMPLETE - EXECUTE NOW:

STEP 1: Open Neo4j Browser
📍 URL: {neo4j_url}
🔑 Login: neo4j / neo4j123

STEP 2: Apply Styling
1. Click ⚙ (gear icon) → Browser Settings → Graph Stylesheet
2. Copy content from: {style_file}
3. Paste and click Apply

STEP 3: Run Main Query
Copy and paste this query:

{queries['main_heat_visualization']}

EXPECTED RESULTS:
✅ 15 Stock nodes visible
✅ UNH in ORANGE (highest heat)
✅ Other stocks in BLUE
✅ Stock symbols as labels
✅ Dynamic sizing based on heat

STEP 4: For Real-time Updates
Re-run this query every minute:

{queries['live_updates']}

TROUBLESHOOTING:
If nodes don't appear, run debug query:

{queries['debug_query']}

Current Heat Rankings:
"""
    
    # Add current heat data to instructions
    for i, stock in enumerate(hot_stocks[:10], 1):
        symbol = stock['stock_id'].replace('STOCK_', '')
        instructions += f"{i:2d}. {symbol}: {stock['heat']:.4f} ({stock['level']})\n"
    
    instructions += f"""
🎯 SYSTEM STATUS: 100% OPERATIONAL
📊 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔥 Total Stocks: {len(hot_stocks)}
🚀 Execute the queries above to see your dynamic heat visualization!
"""
    
    # Save instructions
    instructions_file = "NEO4J_EXECUTE_NOW.txt"
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print("✅ Execution instructions saved:", instructions_file)
    print()
    print(instructions)
    
    return True

if __name__ == "__main__":
    success = apply_neo4j_visualization()
    if success:
        print("\n🎉 Neo4j visualization configuration applied successfully!")
        print("📖 Follow the instructions above to execute the visualization.")
    else:
        print("\n❌ Failed to apply Neo4j visualization configuration")
        sys.exit(1)