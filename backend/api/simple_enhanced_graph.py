"""
Simple Enhanced Graph API - Basic implementation for frontend
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import random
import json
from datetime import datetime
import pytz

router = APIRouter(prefix="/api/enhanced-graph", tags=["Enhanced Graph"])

class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    heat_level: float
    price: float
    change: float
    volume: int

class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    weight: float
    correlation: float

class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    timestamp: str
    market_status: str

def is_market_open():
    """Check if US market is currently open"""
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.now(ny_tz)
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    time_minutes = hour * 60 + minute
    
    # Market hours: Monday-Friday 9:30 AM - 4:00 PM ET
    if weekday >= 5:  # Weekend
        return False
    if time_minutes < 9*60 + 30 or time_minutes >= 16*60:  # Outside market hours
        return False
    return True

def generate_mock_graph_data():
    """Generate mock graph data for visualization"""
    
    # Sample stocks with sectors
    stocks = [
        {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
        {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
        {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Discretionary"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
        {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology"},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services"},
        {"symbol": "V", "name": "Visa Inc.", "sector": "Financial Services"},
        {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
        {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consumer Staples"},
        {"symbol": "PG", "name": "Procter & Gamble Co.", "sector": "Consumer Staples"},
        {"symbol": "UNH", "name": "UnitedHealth Group Inc.", "sector": "Healthcare"}
    ]
    
    # Generate nodes
    nodes = []
    for stock in stocks:
        heat_level = random.uniform(0, 100)
        price = random.uniform(50, 500)
        change = random.uniform(-5, 5)
        volume = random.randint(1000000, 50000000)
        
        node = GraphNode(
            id=stock["symbol"],
            label=f"{stock['symbol']} - {stock['name'][:15]}",
            type=stock["sector"],
            heat_level=heat_level,
            price=round(price, 2),
            change=round(change, 2),
            volume=volume
        )
        nodes.append(node)
    
    # Add sector nodes
    sectors = list(set([stock["sector"] for stock in stocks]))
    for sector in sectors:
        node = GraphNode(
            id=f"SECTOR_{sector.replace(' ', '_').upper()}",
            label=sector,
            type="Sector",
            heat_level=random.uniform(30, 80),
            price=0,
            change=random.uniform(-2, 2),
            volume=0
        )
        nodes.append(node)
    
    # Generate edges
    edges = []
    
    # Connect stocks to their sectors
    for stock in stocks:
        sector_id = f"SECTOR_{stock['sector'].replace(' ', '_').upper()}"
        edge = GraphEdge(
            source=stock["symbol"],
            target=sector_id,
            label="belongs_to",
            weight=random.uniform(0.5, 1.0),
            correlation=random.uniform(0.3, 0.9)
        )
        edges.append(edge)
    
    # Connect related stocks
    correlations = [
        ("AAPL", "MSFT", 0.75),
        ("GOOGL", "MSFT", 0.68),
        ("TSLA", "NVDA", 0.62),
        ("JPM", "V", 0.71),
        ("JNJ", "UNH", 0.58),
        ("WMT", "PG", 0.54),
        ("AAPL", "NVDA", 0.69),
        ("AMZN", "GOOGL", 0.73)
    ]
    
    for source, target, corr in correlations:
        edge = GraphEdge(
            source=source,
            target=target,
            label="correlated",
            weight=corr,
            correlation=corr
        )
        edges.append(edge)
    
    return GraphData(
        nodes=nodes,
        edges=edges,
        timestamp=datetime.now().isoformat(),
        market_status="open" if is_market_open() else "closed"
    )

@router.get("/status")
async def get_status():
    """Get enhanced graph system status"""
    return {
        "status": "active",
        "market_open": is_market_open(),
        "last_updated": datetime.now().isoformat(),
        "features": [
            "Real-time heat visualization",
            "Market correlation analysis", 
            "Sector-based grouping",
            "Interactive graph exploration"
        ]
    }

@router.get("/data", response_model=GraphData)
async def get_graph_data():
    """Get current graph data for visualization"""
    return generate_mock_graph_data()

@router.get("/nodes")
async def get_nodes():
    """Get all nodes in the graph"""
    data = generate_mock_graph_data()
    return {"nodes": [node.dict() for node in data.nodes]}

@router.get("/edges") 
async def get_edges():
    """Get all edges in the graph"""
    data = generate_mock_graph_data()
    return {"edges": [edge.dict() for edge in data.edges]}

@router.get("/heat-analysis")
async def get_heat_analysis():
    """Get heat analysis data"""
    data = generate_mock_graph_data()
    
    # Categorize nodes by heat level
    hot_nodes = [n for n in data.nodes if n.heat_level > 70]
    warm_nodes = [n for n in data.nodes if 40 <= n.heat_level <= 70] 
    cool_nodes = [n for n in data.nodes if n.heat_level < 40]
    
    return {
        "summary": {
            "total_nodes": len(data.nodes),
            "hot_nodes": len(hot_nodes),
            "warm_nodes": len(warm_nodes),
            "cool_nodes": len(cool_nodes)
        },
        "hot_nodes": [{"id": n.id, "label": n.label, "heat": n.heat_level} for n in hot_nodes],
        "market_status": data.market_status,
        "timestamp": data.timestamp
    }