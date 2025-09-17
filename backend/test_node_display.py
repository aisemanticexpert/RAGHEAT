#!/usr/bin/env python3
"""
Test Node Display for Different Node Types in RAGHeat
"""

import requests
import json

def test_node_display():
    """Test that different node types display correctly"""
    
    print("🔍 TESTING NODE DISPLAY IN RAGHEAT VISUALIZATION")
    print("=" * 60)
    
    try:
        # Get graph structure
        response = requests.get('http://localhost:8000/api/graph/structure', timeout=10)
        data = response.json()
        
        # Group nodes by type
        node_types = {
            'Market': [],
            'Sector': [],
            'Company': [], 
            'Stock': []
        }
        
        for node in data.get('graph', {}).get('nodes', []):
            labels = node.get('labels', [])
            for label in labels:
                if label in node_types:
                    node_types[label].append(node)
                    break
        
        # Display each node type
        for node_type, nodes in node_types.items():
            if nodes:
                print(f"\n🏷️  {node_type.upper()} NODES ({len(nodes)} nodes):")
                print("-" * 70)
                
                for node in nodes[:5]:  # Show first 5 nodes of each type
                    props = node.get('properties', {})
                    
                    if node_type == 'Market':
                        display_name = props.get('name', 'Unknown Market')
                        print(f"   📊 {node['id']:20} → '{display_name}'")
                        
                    elif node_type == 'Sector':
                        display_name = props.get('name', 'Unknown Sector')
                        print(f"   🏢 {node['id']:20} → '{display_name}'")
                        
                    elif node_type == 'Company':
                        display_name = props.get('symbol', props.get('name', 'Unknown'))
                        company_name = props.get('name', 'Unknown Company')
                        print(f"   🏭 {node['id']:20} → '{display_name}' ({company_name})")
                        
                    elif node_type == 'Stock':
                        display_name = props.get('symbol', 'Unknown Symbol')
                        signal = props.get('trading_signal', 'NO_SIGNAL')
                        price = props.get('current_price', 0)
                        print(f"   📈 {node['id']:20} → '{display_name}' | {signal} | ${price:.2f}")
        
        print("\n" + "=" * 60)
        print("📋 NEO4J BROWSER STYLING GUIDE:")
        print("-" * 60)
        print("🎯 Each node type should display:")
        print("   📊 Market nodes: Market name (e.g., 'US Stock Market')")
        print("   🏢 Sector nodes: Sector name (e.g., 'Technology', 'Healthcare')")
        print("   🏭 Company nodes: Stock symbol (e.g., 'AAPL', 'GOOGL')")  
        print("   📈 Stock nodes: Stock symbol (e.g., 'AAPL', 'MSFT')")
        print()
        print("🔧 If nodes show the same name, copy the updated style:")
        print("   📁 File: neo4j_buy_sell_style.grass")
        print("   🎨 Apply to Neo4j Browser Style tab")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing node display: {e}")
        return False

if __name__ == "__main__":
    success = test_node_display()
    if success:
        print("\n✅ Node display test completed!")
    else:
        print("\n❌ Node display test failed")