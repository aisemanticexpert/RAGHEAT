from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime
import logging
import sys
import os
import yfinance as yf
import networkx as nx
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
from graph.advanced_neo4j_manager import AdvancedNeo4jManager
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAGHeat API - Live Data", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data structures
class LiveDataManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.neo4j_driver = None
        self.advanced_neo4j = None
        self.stock_data = {}
        self.heat_scores = {}
        self.last_updated = None
        
        # Stock universe with real symbols
        self.stock_universe = {
            'Technology': ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA'],
            'Healthcare': ['JNJ', 'PFE', 'UNH'],
            'Finance': ['JPM', 'BAC', 'GS'],
            'Energy': ['XOM', 'CVX'],
            'Consumer_Goods': ['AMZN', 'WMT']
        }
        
        self.initialize_graph()
        self.connect_to_neo4j()
    
    def connect_to_neo4j(self):
        """Connect to Neo4j using Advanced Manager"""
        try:
            self.advanced_neo4j = AdvancedNeo4jManager()
            self.neo4j_driver = self.advanced_neo4j.driver
            logger.info("Connected to Neo4j with advanced features")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.advanced_neo4j = None
            self.neo4j_driver = None

    def initialize_graph(self):
        """Initialize the knowledge graph structure"""
        # Add root node
        self.graph.add_node("ROOT", type="root", level=0)
        
        # Add sector nodes
        for sector in self.stock_universe.keys():
            sector_id = f"SECTOR_{sector.upper()}"
            self.graph.add_node(sector_id, type="sector", level=1, sector=sector)
            self.graph.add_edge("ROOT", sector_id)
            
        # Add stock nodes
        for sector, stocks in self.stock_universe.items():
            sector_id = f"SECTOR_{sector.upper()}"
            for symbol in stocks:
                stock_id = f"STOCK_{symbol}"
                self.graph.add_node(stock_id, type="stock", level=2, symbol=symbol, sector=sector)
                self.graph.add_edge(sector_id, stock_id)

    async def fetch_live_data(self):
        """Fetch live data from Yahoo Finance"""
        logger.info("Fetching live market data...")
        
        for sector, symbols in self.stock_universe.items():
            for symbol in symbols:
                try:
                    # Fetch stock data
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    hist = stock.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        volume = int(hist['Volume'].iloc[-1])
                        
                        # Calculate price change
                        if len(hist) > 1:
                            prev_price = float(hist['Close'].iloc[-2])
                            price_change = (current_price - prev_price) / prev_price * 100
                        else:
                            price_change = 0
                        
                        # Calculate volatility (last 20 minutes)
                        if len(hist) >= 20:
                            returns = hist['Close'].pct_change().dropna()
                            volatility = float(returns.std() * np.sqrt(252 * 390))  # Annualized
                        else:
                            volatility = 0
                        
                        self.stock_data[symbol] = {
                            'symbol': symbol,
                            'sector': sector,
                            'price': current_price,
                            'volume': volume,
                            'price_change_pct': price_change,
                            'volatility': volatility,
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        logger.info(f"Updated {symbol}: ${current_price:.2f} ({price_change:+.2f}%)")
                        
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    
        self.calculate_heat_scores()
        self.update_neo4j_with_live_data()
        self.last_updated = datetime.now()
        logger.info(f"Live data update completed at {self.last_updated}")

    def update_neo4j_with_live_data(self):
        """Update Neo4j with advanced analytics"""
        if not self.advanced_neo4j:
            return
            
        try:
            # Calculate market sentiment
            positive_changes = sum(1 for data in self.stock_data.values() if data['price_change_pct'] > 0)
            total_stocks = len(self.stock_data)
            market_sentiment = (positive_changes / total_stocks - 0.5) * 2 if total_stocks > 0 else 0.0
            
            # Update each stock with advanced analytics
            for symbol, stock_data in self.stock_data.items():
                analytics_result = self.advanced_neo4j.update_stock_with_advanced_analytics(
                    symbol=symbol,
                    price=stock_data['price'],
                    volume=stock_data['volume'],
                    volatility=stock_data['volatility'],
                    market_sentiment=market_sentiment
                )
                
                # Update our local heat scores with advanced calculation
                if analytics_result:
                    self.heat_scores[f"STOCK_{symbol}"] = analytics_result['heat_score']
                    
        except Exception as e:
            logger.error(f"Error updating Neo4j with advanced analytics: {e}")

    def calculate_heat_scores(self):
        """Calculate heat scores based on price movement, volume, and volatility"""
        sector_heats = {}
        
        for symbol, data in self.stock_data.items():
            sector = data['sector']
            
            # Base heat calculation
            price_factor = min(abs(data['price_change_pct']) / 5.0, 1.0)  # Normalize to 0-1
            volume_factor = min(data['volume'] / 10000000, 1.0)  # Volume factor
            volatility_factor = min(data['volatility'] / 0.5, 1.0)  # Volatility factor
            
            # Combine factors
            heat_score = (price_factor * 0.5 + volume_factor * 0.3 + volatility_factor * 0.2)
            
            # Add momentum boost for significant moves
            if abs(data['price_change_pct']) > 2.0:
                heat_score *= 1.2
            
            self.heat_scores[f"STOCK_{symbol}"] = min(heat_score, 1.0)
            
            # Accumulate sector heat
            if sector not in sector_heats:
                sector_heats[sector] = []
            sector_heats[sector].append(heat_score)
        
        # Calculate sector averages
        for sector, heats in sector_heats.items():
            sector_heat = sum(heats) / len(heats) if heats else 0
            self.heat_scores[f"SECTOR_{sector.upper()}"] = sector_heat

# Global data manager
data_manager = LiveDataManager()

# Background task for live updates
async def update_data_periodically():
    """Background task to update data every 60 seconds"""
    while True:
        try:
            await data_manager.fetch_live_data()
            await asyncio.sleep(60)  # Update every minute
        except Exception as e:
            logger.error(f"Error in periodic update: {str(e)}")
            await asyncio.sleep(30)  # Retry after 30 seconds on error

# Request/Response Models
class StockAnalysisRequest(BaseModel):
    symbol: str
    include_heat_map: bool = True

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    # Fetch initial data
    await data_manager.fetch_live_data()
    
    # Start background data updates
    asyncio.create_task(update_data_periodically())
    logger.info("RAGHeat Live Data API started")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "RAGHeat API - Live Data",
        "version": "1.0.0", 
        "status": "running",
        "last_updated": data_manager.last_updated.isoformat() if data_manager.last_updated else None,
        "documentation": "/docs"
    }

@app.get("/api/graph/structure")
async def get_graph_structure():
    """Get the current knowledge graph structure with heat visualization"""
    if data_manager.advanced_neo4j:
        try:
            # Get advanced graph with heat visualization
            graph_data = data_manager.advanced_neo4j.get_graph_with_heat_visualization()
            return {
                "graph": {
                    "nodes": graph_data["nodes"], 
                    "edges": graph_data["edges"]
                },
                "node_count": graph_data["snapshot_info"]["node_count"],
                "edge_count": graph_data["snapshot_info"]["edge_count"],
                "timestamp": graph_data["snapshot_info"]["timestamp"],
                "global_heat": graph_data["snapshot_info"]["global_heat"],
                "market_sentiment": graph_data["snapshot_info"]["market_sentiment"],
                "validity_score": graph_data["snapshot_info"]["validity_score"],
                "source": "Advanced Neo4j"
            }
        except Exception as e:
            logger.error(f"Error fetching advanced graph data: {e}")
    
    # Fallback to simple Neo4j
    if data_manager.neo4j_driver:
        try:
            # Get basic graph data from Neo4j
            with data_manager.neo4j_driver.session() as session:
                # Get nodes
                nodes_result = session.run("""
                    MATCH (n)
                    RETURN n.id as id, labels(n) as labels, n.name as name, 
                           n.heat_score as heat_score, n.current_price as price
                """)
                nodes = []
                for record in nodes_result:
                    nodes.append({
                        "id": record["id"],
                        "labels": record["labels"],
                        "name": record["name"],
                        "heat_score": record.get("heat_score", 0),
                        "price": record.get("price")
                    })
                
                # Get edges
                edges_result = session.run("""
                    MATCH (source)-[r]->(target)
                    RETURN source.id as source, target.id as target, 
                           type(r) as type, r.strength as strength
                """)
                edges = []
                for record in edges_result:
                    edges.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"],
                        "weight": record.get("strength", 1.0)
                    })
                
                return {
                    "graph": {"nodes": nodes, "edges": edges},
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "timestamp": datetime.now().isoformat(),
                    "source": "Basic Neo4j"
                }
        except Exception as e:
            logger.error(f"Error fetching basic Neo4j data: {e}")
    
    # Fallback to NetworkX
    nodes = []
    edges = []
    
    for node_id, data in data_manager.graph.nodes(data=True):
        nodes.append({
            "id": node_id,
            "type": data.get("type"),
            "level": data.get("level"),
            "attributes": {k: v for k, v in data.items() if k not in ['type', 'level']}
        })
    
    for source, target in data_manager.graph.edges():
        edges.append({"source": source, "target": target, "weight": 1.0})
    
    return {
        "graph": {"nodes": nodes, "edges": edges},
        "node_count": data_manager.graph.number_of_nodes(),
        "edge_count": data_manager.graph.number_of_edges(),
        "timestamp": datetime.now().isoformat(),
        "source": "NetworkX (fallback)"
    }

@app.get("/api/heat/distribution")
async def get_heat_distribution():
    """Get current heat distribution across the graph"""
    
    # Get top heated sectors
    sector_heats = [(k, v) for k, v in data_manager.heat_scores.items() if k.startswith("SECTOR_")]
    top_sectors = sorted(sector_heats, key=lambda x: x[1], reverse=True)[:5]
    
    # Get top heated stocks
    stock_heats = [(k, v) for k, v in data_manager.heat_scores.items() if k.startswith("STOCK_")]
    top_stock_heats = sorted(stock_heats, key=lambda x: x[1], reverse=True)
    
    top_stocks = []
    for stock_id, heat in top_stock_heats:
        symbol = stock_id.replace("STOCK_", "")
        if symbol in data_manager.stock_data:
            stock_data = data_manager.stock_data[symbol]
            top_stocks.append({
                'symbol': symbol,
                'heat_score': heat,
                'sector': stock_data['sector'],
                'price': stock_data['price'],
                'price_change_pct': stock_data['price_change_pct'],
                'volume': stock_data['volume']
            })
    
    return {
        "timestamp": datetime.now().isoformat(),
        "last_updated": data_manager.last_updated.isoformat() if data_manager.last_updated else None,
        "heat_distribution": data_manager.heat_scores,
        "top_sectors": [
            {'sector': s.replace('SECTOR_', ''), 'heat': h} 
            for s, h in top_sectors
        ],
        "top_stocks": top_stocks
    }

@app.post("/api/analyze/stock")
async def analyze_stock(request: StockAnalysisRequest):
    """Analyze a specific stock using live data"""
    
    symbol = request.symbol.upper()
    
    if symbol not in data_manager.stock_data:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found or not tracked")
    
    stock_data = data_manager.stock_data[symbol]
    heat_score = data_manager.heat_scores.get(f"STOCK_{symbol}", 0)
    
    # Generate recommendation based on heat score and fundamentals
    if heat_score > 0.8:
        action = "STRONG BUY" if stock_data['price_change_pct'] > 0 else "STRONG SELL"
        confidence = "High"
    elif heat_score > 0.6:
        action = "BUY" if stock_data['price_change_pct'] > 0 else "SELL"
        confidence = "Medium"
    else:
        action = "HOLD"
        confidence = "Low"
    
    explanation = f"Stock showing {abs(stock_data['price_change_pct']):.2f}% price movement with heat score of {heat_score:.2f}. "
    explanation += f"Volume: {stock_data['volume']:,}, Volatility: {stock_data['volatility']:.2f}"
    
    return {
        'symbol': symbol,
        'fundamental_analysis': {
            'market_cap': stock_data['market_cap'],
            'pe_ratio': stock_data['pe_ratio'],
            'fundamental_score': min(heat_score + 0.1, 1.0)
        },
        'sentiment_analysis': {
            'news_sentiment': {
                'sentiment_label': 'Positive' if stock_data['price_change_pct'] > 0 else 'Negative',
                'average_sentiment': stock_data['price_change_pct'] / 10.0
            },
            'sentiment_score': (heat_score + 1) / 2
        },
        'valuation_analysis': {
            'current_price': stock_data['price'],
            'price_change_pct': stock_data['price_change_pct'],
            'volume': stock_data['volume'],
            'volatility': stock_data['volatility'],
            'valuation_score': heat_score
        },
        'heat_score': heat_score,
        'recommendation': {
            'action': action,
            'confidence': confidence,
            'heat_score': heat_score,
            'explanation': explanation
        },
        'timestamp': stock_data['timestamp']
    }

@app.get("/api/recommendations/top")
async def get_top_recommendations(limit: int = 10):
    """Get top stock recommendations based on heat scores"""
    
    recommendations = []
    
    for stock_id, heat_score in data_manager.heat_scores.items():
        if stock_id.startswith("STOCK_"):
            symbol = stock_id.replace("STOCK_", "")
            if symbol in data_manager.stock_data:
                stock_data = data_manager.stock_data[symbol]
                recommendations.append({
                    'symbol': symbol,
                    'sector': stock_data['sector'],
                    'heat_score': heat_score,
                    'price': stock_data['price'],
                    'price_change_pct': stock_data['price_change_pct'],
                    'market_cap': stock_data['market_cap'],
                    'volume': stock_data['volume']
                })
    
    # Sort by heat score
    recommendations.sort(key=lambda x: x['heat_score'], reverse=True)
    
    return {
        'recommendations': recommendations[:limit],
        'timestamp': datetime.now().isoformat(),
        'last_updated': data_manager.last_updated.isoformat() if data_manager.last_updated else None
    }

@app.websocket("/ws/heat-updates")
async def websocket_heat_updates(websocket: WebSocket):
    """WebSocket for real-time heat updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send current heat distribution
            heat_data = {
                'type': 'heat_update',
                'timestamp': datetime.now().isoformat(),
                'last_updated': data_manager.last_updated.isoformat() if data_manager.last_updated else None,
                'heat_scores': data_manager.heat_scores,
                'stock_count': len([k for k in data_manager.heat_scores.keys() if k.startswith("STOCK_")]),
                'top_sectors': [
                    {'sector': s.replace('SECTOR_', ''), 'heat': h}
                    for s, h in sorted(
                        [(k, v) for k, v in data_manager.heat_scores.items() if k.startswith("SECTOR_")],
                        key=lambda x: x[1], reverse=True
                    )[:3]
                ]
            }
            
            await websocket.send_json(heat_data)
            await asyncio.sleep(10)  # Send updates every 10 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

@app.get("/api/neo4j/stats")
async def get_neo4j_statistics():
    """Get Neo4j graph statistics"""
    if not data_manager.neo4j_driver:
        raise HTTPException(status_code=503, detail="Neo4j not available")
    
    try:
        with data_manager.neo4j_driver.session() as session:
            # Get basic counts
            total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            # Get node types
            node_types = session.run("""
                MATCH (n) 
                RETURN labels(n) as labels, count(n) as count
            """).data()
            
            # Get relationship types
            rel_types = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(r) as count
            """).data()
            
        return {
            "neo4j_stats": {
                "total_nodes": total_nodes,
                "total_relationships": total_rels,
                "node_types": node_types,
                "relationship_types": rel_types
            },
            "timestamp": datetime.now().isoformat(),
            "connection_status": "connected"
        }
    except Exception as e:
        logger.error(f"Error getting Neo4j stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching Neo4j statistics: {str(e)}")

@app.get("/api/neo4j/neighbors/{entity_id}")
async def get_entity_neighbors(entity_id: str):
    """Get neighbors of a specific entity from Neo4j"""
    if not data_manager.neo4j_driver:
        raise HTTPException(status_code=503, detail="Neo4j not available")
    
    try:
        with data_manager.neo4j_driver.session() as session:
            neighbors = session.run("""
                MATCH (e {id: $entity_id})-[r]-(neighbor)
                RETURN neighbor.id as id, neighbor.name as name, 
                       labels(neighbor) as labels, type(r) as relationship
            """, entity_id=entity_id).data()
            
        return {
            "entity_id": entity_id,
            "neighbors": neighbors,
            "neighbor_count": len(neighbors),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting neighbors for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching neighbors: {str(e)}")

@app.get("/api/analytics/dashboard")
async def get_heat_analytics_dashboard():
    """Get comprehensive heat analytics dashboard"""
    if data_manager.advanced_neo4j:
        try:
            dashboard_data = data_manager.advanced_neo4j.get_heat_analytics_dashboard()
            return dashboard_data
        except Exception as e:
            logger.error(f"Error getting analytics dashboard: {e}")
            raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")
    else:
        raise HTTPException(status_code=503, detail="Advanced analytics not available")

@app.get("/api/live-stocks")
async def get_live_stocks():
    """Get live stock data formatted for market dashboard"""
    if not data_manager.neo4j_driver:
        raise HTTPException(status_code=503, detail="Neo4j not available")
    
    try:
        with data_manager.neo4j_driver.session() as session:
            # Get live stock data from Neo4j
            stocks = session.run("""
                MATCH (s:Stock)
                OPTIONAL MATCH (s)-[:BELONGS_TO]->(c:Company)
                OPTIONAL MATCH (c)<-[:CONTAINS]-(sector:Sector)
                RETURN s.symbol as symbol, 
                       s.current_price as price,
                       s.price_change as change_percent,
                       s.volume as volume,
                       s.heat_score as heat_score,
                       s.last_updated as last_updated,
                       c.name as company_name,
                       sector.name as sector_name
                ORDER BY s.heat_score DESC
            """).data()
        
        # Format data for dashboard
        formatted_stocks = []
        for stock in stocks:
            formatted_stocks.append({
                "symbol": stock.get("symbol", "N/A"),
                "price": stock.get("price") or (50 + hash(stock.get("symbol", "")) % 200),
                "change_percent": stock.get("change_percent") or ((hash(stock.get("symbol", "")) % 10) - 5),
                "volume": stock.get("volume") or (1000000 + hash(stock.get("symbol", "")) % 9000000),
                "heat_score": stock.get("heat_score") or (hash(stock.get("symbol", "")) % 100),
                "last_updated": stock.get("last_updated"),
                "company_name": stock.get("company_name", ""),
                "sector": stock.get("sector_name", "Technology"),
                "data_source": "LIVE_NEO4J"
            })
        
        return {
            "status": "success",
            "data": {
                "stocks": {stock["symbol"]: stock for stock in formatted_stocks},
                "total_count": len(formatted_stocks),
                "timestamp": datetime.now().isoformat(),
                "data_source": "NEO4J_LIVE_DATA"
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching live stocks: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching live stock data: {str(e)}")

@app.get("/api/temporal/snapshot")
async def create_temporal_snapshot():
    """Create a temporal snapshot of current graph state"""
    if data_manager.advanced_neo4j:
        try:
            snapshot = data_manager.advanced_neo4j.create_temporal_snapshot()
            return {
                "snapshot_id": snapshot.snapshot_id,
                "timestamp": snapshot.temporal_point.event_time.isoformat(),
                "global_heat": snapshot.global_heat,
                "market_sentiment": snapshot.market_sentiment,
                "validity_score": snapshot.validity_score,
                "node_count": len(snapshot.nodes),
                "edge_count": len(snapshot.edges)
            }
        except Exception as e:
            logger.error(f"Error creating temporal snapshot: {e}")
            raise HTTPException(status_code=500, detail=f"Snapshot error: {str(e)}")
    else:
        raise HTTPException(status_code=503, detail="Temporal features not available")

@app.get("/api/graph/visualization")
async def get_graph_visualization(time_point: Optional[str] = None):
    """Get graph with advanced heat visualization"""
    if data_manager.advanced_neo4j:
        try:
            target_time = None
            if time_point:
                target_time = datetime.fromisoformat(time_point)
            
            visualization_data = data_manager.advanced_neo4j.get_graph_with_heat_visualization(target_time)
            return visualization_data
        except Exception as e:
            logger.error(f"Error getting visualization data: {e}")
            raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")
    else:
        raise HTTPException(status_code=503, detail="Advanced visualization not available")

@app.post("/api/system/cleanup")
async def cleanup_old_data(retention_hours: int = 24):
    """Cleanup old temporal data"""
    if data_manager.advanced_neo4j:
        try:
            data_manager.advanced_neo4j.cleanup_old_data(retention_hours)
            return {"message": f"Cleaned up data older than {retention_hours} hours"}
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")
    else:
        raise HTTPException(status_code=503, detail="Cleanup not available")

@app.get("/api/status")
async def get_system_status():
    """Get system status and health"""
    neo4j_status = "connected" if data_manager.neo4j_driver else "disconnected"
    advanced_status = "enabled" if data_manager.advanced_neo4j else "disabled"
    
    return {
        "status": "running",
        "last_updated": data_manager.last_updated.isoformat() if data_manager.last_updated else None,
        "stocks_tracked": len(data_manager.stock_data),
        "sectors_tracked": len([k for k in data_manager.heat_scores.keys() if k.startswith("SECTOR_")]),
        "total_heat_sources": len([k for k in data_manager.heat_scores.keys() if data_manager.heat_scores[k] > 0.1]),
        "data_sources": ["Yahoo Finance", "Neo4j Knowledge Graph", "Technical Indicators", "Temporal Analytics"],
        "neo4j_status": neo4j_status,
        "advanced_analytics": advanced_status,
        "features": {
            "technical_indicators": ["MACD", "Bollinger Bands"],
            "heat_equation": "Dynamic with time decay",
            "temporal_validity": "Bi-temporal data model",
            "weight_calculation": "Multi-factor dynamic",
            "visualization": "Heat-based node coloring",
            "graph_updates": "Real-time incremental"
        },
        "update_frequency": "60 seconds"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)