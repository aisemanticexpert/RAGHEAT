"""
Graph API Routes - Endpoints that match frontend expectations
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import random
import json
from datetime import datetime
import pytz

router = APIRouter(prefix="/api/graph", tags=["Graph"])

# Reuse the same data generation from simple_enhanced_graph
from api.simple_enhanced_graph import generate_mock_graph_data, is_market_open

def format_for_frontend(graph_data):
    """Format graph data to match frontend expectations"""
    
    # Convert nodes to frontend format
    nodes = []
    for node in graph_data.nodes:
        # Better node sizing logic
        if 'sector' in node.type.lower():
            # Sectors are larger and more prominent
            size = max(80, min(120, 80 + (node.heat_level / 100) * 40))
        else:
            # Stock nodes sized by heat and volume
            base_size = 40
            heat_factor = (node.heat_level / 100) * 30
            volume_factor = min(20, (node.volume / 50000000) * 20)  # Volume-based sizing
            size = max(30, min(70, base_size + heat_factor + volume_factor))
        
        frontend_node = {
            "id": node.id,
            "label": node.label,
            "type": node.type.lower().replace(" ", "_"),
            "heat_level": node.heat_level,
            "price": node.price,
            "change": node.change,
            "volume": node.volume,
            "size": size,
            "color": get_node_color(node.heat_level, node.type),
            "border_color": "#FFFFFF",
            "border_width": 1,
            "opacity": max(0.7, min(1.0, 0.5 + (node.heat_level / 100) * 0.5))  # Heat-based opacity
        }
        nodes.append(frontend_node)
    
    # Convert edges to relationships format
    relationships = []
    for edge in graph_data.edges:
        # Better edge styling based on type and strength
        if edge.label == "belongs_to":
            edge_color = "#94A3B8"  # Gray for sector relationships
            edge_width = max(0.5, min(2, 0.5 + edge.weight * 1.5))  # Thinner edges
            edge_style = "solid"
        elif edge.label == "correlated":
            # Color based on correlation strength
            if edge.correlation > 0.7:
                edge_color = "#EF4444"  # Strong correlation - red
            elif edge.correlation > 0.5:
                edge_color = "#F97316"  # Medium correlation - orange  
            else:
                edge_color = "#FCD34D"  # Weak correlation - yellow
            edge_width = max(0.5, min(2, 0.5 + edge.correlation * 1.5))  # Thinner edges
            edge_style = "dashed"
        else:
            edge_color = "#6B7280"
            edge_width = 1  # Thinner default
            edge_style = "solid"
        
        relationship = {
            "source": edge.source,
            "target": edge.target,
            "type": edge.label,
            "weight": edge.weight,
            "correlation": edge.correlation,
            "color": edge_color,
            "width": edge_width,
            "style": edge_style,
            "opacity": max(0.5, min(1.0, 0.3 + edge.weight * 0.7))
        }
        relationships.append(relationship)
    
    return {
        "success": True,
        "data": {
            "nodes": nodes,
            "relationships": relationships,
            "metadata": {
                "timestamp": graph_data.timestamp,
                "market_status": graph_data.market_status,
                "total_nodes": len(nodes),
                "total_edges": len(relationships)
            }
        }
    }

def get_node_color(heat_level, node_type):
    """Get node color based on heat level and type"""
    
    # Check if it's a sector node - use distinct colors
    if 'sector' in node_type.lower():
        sector_colors = {
            'technology': '#6366F1',           # Indigo
            'financial_services': '#059669',   # Emerald  
            'healthcare': '#DC2626',           # Red
            'consumer_discretionary': '#EC4899', # Pink
            'consumer_staples': '#10B981',     # Green
            'sector': '#4338CA'                # Deep blue
        }
        return sector_colors.get(node_type.lower().replace(" ", "_"), '#4338CA')
    
    # Heat-based gradient coloring for stocks
    if heat_level > 85:
        return '#FF1744'    # Intense red - very hot
    elif heat_level > 70:
        return '#FF5722'    # Red-orange - hot  
    elif heat_level > 55:
        return '#FF9800'    # Orange - warm
    elif heat_level > 40:
        return '#FFC107'    # Amber - medium
    elif heat_level > 25:
        return '#4CAF50'    # Green - cool
    else:
        return '#607D8B'    # Blue-gray - cold

@router.get("/live-data")
async def get_live_data():
    """Get live graph data (when market is open)"""
    graph_data = generate_mock_graph_data()
    
    # Add some "live" variation for market hours
    if is_market_open():
        # Simulate more volatile changes during market hours
        for node in graph_data.nodes:
            node.heat_level = min(100, max(0, node.heat_level + random.uniform(-10, 10)))
            node.change = node.change + random.uniform(-1, 1)
    
    return format_for_frontend(graph_data)

@router.get("/synthetic-data") 
async def get_synthetic_data():
    """Get synthetic graph data (when market is closed)"""
    graph_data = generate_mock_graph_data()
    
    # Add synthetic/historical simulation
    for node in graph_data.nodes:
        # More stable changes for synthetic data
        node.heat_level = min(100, max(0, node.heat_level + random.uniform(-5, 5)))
        
    return format_for_frontend(graph_data)

@router.get("/status")
async def get_graph_status():
    """Get graph system status"""
    return {
        "success": True,
        "market_open": is_market_open(),
        "data_source": "live" if is_market_open() else "synthetic",
        "last_updated": datetime.now().isoformat(),
        "endpoints_available": [
            "/api/graph/live-data",
            "/api/graph/synthetic-data", 
            "/api/graph/status"
        ]
    }

# Add stock analysis endpoint that frontend expects
stock_router = APIRouter(prefix="/api/stock", tags=["Stock Analysis"])

@stock_router.get("/{stock_id}/analysis")
async def get_stock_analysis(stock_id: str):
    """Get detailed stock analysis"""
    
    # Generate mock analysis data
    analysis = {
        "success": True,
        "data": {
            "symbol": stock_id,
            "name": f"Analysis for {stock_id}",
            "current_price": random.uniform(50, 500),
            "heat_level": random.uniform(0, 100),
            "sentiment_score": random.uniform(-1, 1),
            "technical_indicators": {
                "rsi": random.uniform(0, 100),
                "macd": random.uniform(-1, 1),
                "bollinger_bands": {
                    "upper": random.uniform(100, 200),
                    "middle": random.uniform(80, 120),
                    "lower": random.uniform(60, 100)
                }
            },
            "correlations": [
                {"symbol": "AAPL", "correlation": 0.75},
                {"symbol": "MSFT", "correlation": 0.68},
                {"symbol": "GOOGL", "correlation": 0.62}
            ],
            "news_sentiment": {
                "score": random.uniform(-1, 1),
                "articles_count": random.randint(5, 50),
                "last_updated": datetime.now().isoformat()
            }
        }
    }
    
    return analysis