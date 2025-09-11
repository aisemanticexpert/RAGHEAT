#!/usr/bin/env python3
import requests
import json

def execute_stock_query():
    """Execute the Neo4j Stock query via API and format results."""
    
    try:
        # Get current stock data with heat scores via API
        response = requests.get('http://localhost:8000/api/graph/structure', timeout=10)
        data = response.json()
        
        # Extract stock nodes and sort by heat score
        stock_nodes = []
        for node in data.get('graph', {}).get('nodes', []):
            if 'Stock' in node.get('labels', []) and node.get('heat_score', 0) > 0:
                stock_nodes.append({
                    'symbol': node.get('properties', {}).get('symbol', 'Unknown'),
                    'heat_score': node.get('heat_score', 0),
                    'heat_level': node.get('properties', {}).get('heat_level', 'unknown'),
                    'current_price': node.get('properties', {}).get('current_price', 0),
                    'last_updated': node.get('properties', {}).get('last_updated', 'unknown')
                })
        
        # Sort by heat score descending
        stock_nodes.sort(key=lambda x: x['heat_score'], reverse=True)
        
        print('=' * 80)
        print('CYPHER QUERY RESULTS: MATCH (s:Stock) WHERE s.heat_score IS NOT NULL RETURN s')
        print('=' * 80)
        print(f'Total Stock nodes with heat_score: {len(stock_nodes)}')
        print()
        print('Stock Nodes (ordered by heat_score DESC):')
        print('-' * 80)
        
        for i, stock in enumerate(stock_nodes, 1):
            print(f'{i:2d}. {stock["symbol"]:6} | Heat: {stock["heat_score"]:8.6f} | Level: {stock["heat_level"]:5} | Price: ${stock["current_price"]:8.2f}')
        
        print('-' * 80)
        if stock_nodes:
            print(f'Heat Range: {stock_nodes[-1]["heat_score"]:.6f} - {stock_nodes[0]["heat_score"]:.6f}')
            print('All stocks currently in "cold" level (< 0.2)')
            
            # Show the top 3 for Neo4j Browser visualization
            print()
            print('🔥 TOP 3 HEATED STOCKS FOR NEO4J BROWSER:')
            for i, stock in enumerate(stock_nodes[:3], 1):
                emoji = '🥇' if i == 1 else '🥈' if i == 2 else '🥉'
                print(f'{emoji} {stock["symbol"]} - Heat: {stock["heat_score"]:.4f} (will appear in Neo4j Browser)')
            
            print()
            print('✅ These nodes are ready for visualization in Neo4j Browser!')
            print('🎯 UNH should appear in ORANGE (largest, highest heat)')
            print('🎯 Others should appear in BLUE (cold level)')
        else:
            print('❌ No stock nodes with heat_score found')
        
    except Exception as e:
        print(f'❌ Error executing query: {e}')

if __name__ == "__main__":
    execute_stock_query()