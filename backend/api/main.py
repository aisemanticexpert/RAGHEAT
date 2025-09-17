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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.knowledge_graph import FinancialKnowledgeGraph
from graph.heat_diffusion import HeatDiffusionEngine
from agents.orchestrator_agent import OrchestratorAgent
from data_pipeline.stream_processor import MarketDataStreamer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAGHeat API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
knowledge_graph = FinancialKnowledgeGraph()
heat_engine = HeatDiffusionEngine(knowledge_graph.graph)
orchestrator = OrchestratorAgent()
streamer = MarketDataStreamer(knowledge_graph, heat_engine, orchestrator)

# Request/Response Models
class StockAnalysisRequest(BaseModel):
    symbol: str
    include_heat_map: bool = True

class HeatMapResponse(BaseModel):
    timestamp: str
    heat_distribution: Dict[str, float]
    top_sectors: List[Dict]
    top_stocks: List[Dict]

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    # Start data streaming in background
    asyncio.create_task(streamer.start_streaming())
    logger.info("RAGHeat system started")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "RAGHeat API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs"
    }

@app.get("/api/graph/structure")
async def get_graph_structure():
    """Get the current knowledge graph structure"""
    return {
        "graph": json.loads(knowledge_graph.to_json()),
        "node_count": knowledge_graph.graph.number_of_nodes(),
        "edge_count": knowledge_graph.graph.number_of_edges(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/heat/distribution")
async def get_heat_distribution():
    """Get current heat distribution across the graph"""

    # Get top heated sectors
    top_sectors = heat_engine.get_heated_sectors(top_k=5)

    # Get top stocks from each heated sector
    top_stocks = []
    for sector_id, sector_heat in top_sectors[:3]:
        stocks = heat_engine.get_heated_stocks_in_sector(sector_id, top_k=3)
        for stock_id, stock_heat in stocks:
            stock_node = knowledge_graph.node_mapping.get(stock_id)
            if stock_node:
                top_stocks.append({
                    'symbol': stock_node.attributes.get('symbol'),
                    'heat_score': stock_heat,
                    'sector': stock_node.attributes.get('sector')
                })

    return {
        "timestamp": datetime.now().isoformat(),
        "heat_distribution": knowledge_graph.heat_scores,
        "top_sectors": [
            {'sector': s.replace('SECTOR_', ''), 'heat': h} 
            for s, h in top_sectors
        ],
        "top_stocks": sorted(top_stocks, key=lambda x: x.get('heat_score', 0), reverse=True)
    }

@app.post("/api/analyze/stock")
async def analyze_stock(request: StockAnalysisRequest):
    """Analyze a specific stock using multi-agent system"""

    try:
        # Perform comprehensive analysis
        analysis = orchestrator.analyze_stock_comprehensive(request.symbol)

        if 'error' in analysis:
            raise HTTPException(status_code=400, detail=analysis['error'])

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing stock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommendations/top")
async def get_top_recommendations(limit: int = 10):
    """Get top stock recommendations based on heat scores"""

    recommendations = []

    # Get all stocks and their heat scores
    for node_id, heat_score in knowledge_graph.heat_scores.items():
        if node_id.startswith("STOCK_"):
            node = knowledge_graph.node_mapping.get(node_id)
            if node:
                symbol = node.attributes.get('symbol')
                if symbol:
                    recommendations.append({
                        'symbol': symbol,
                        'sector': node.attributes.get('sector'),
                        'heat_score': heat_score,
                        'price': node.attributes.get('price', 0),
                        'market_cap': node.attributes.get('market_cap', 0)
                    })

    # Sort by heat score
    recommendations.sort(key=lambda x: x.get('heat_score', 0), reverse=True)

    return {
        'recommendations': recommendations[:limit],
        'timestamp': datetime.now().isoformat()
    }

@app.websocket("/ws/heat-updates")
async def websocket_heat_updates(websocket: WebSocket):
    """WebSocket for real-time heat updates"""
    await websocket.accept()

    try:
        while True:
            # Send heat distribution every 10 seconds
            heat_data = {
                'type': 'heat_update',
                'timestamp': datetime.now().isoformat(),
                'heat_scores': knowledge_graph.heat_scores,
                'top_sectors': [
                    {'sector': s.replace('SECTOR_', ''), 'heat': h}
                    for s, h in heat_engine.get_heated_sectors(top_k=3)
                ]
            }

            await websocket.send_json(heat_data)
            await asyncio.sleep(10)

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)