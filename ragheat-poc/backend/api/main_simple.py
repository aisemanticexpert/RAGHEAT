from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime
import logging
import random

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

# Mock data for demo
mock_sectors = [
    {'sector': 'Technology', 'heat': 0.85},
    {'sector': 'Healthcare', 'heat': 0.72},
    {'sector': 'Finance', 'heat': 0.68},
    {'sector': 'Energy', 'heat': 0.55},
    {'sector': 'Consumer_Goods', 'heat': 0.62}
]

mock_stocks = [
    {'symbol': 'AAPL', 'sector': 'Technology', 'heat_score': 0.89, 'price': 175.50, 'market_cap': 2800000000000},
    {'symbol': 'GOOGL', 'sector': 'Technology', 'heat_score': 0.83, 'price': 142.30, 'market_cap': 1800000000000},
    {'symbol': 'MSFT', 'sector': 'Technology', 'heat_score': 0.81, 'price': 412.20, 'market_cap': 3100000000000},
    {'symbol': 'JNJ', 'sector': 'Healthcare', 'heat_score': 0.74, 'price': 165.80, 'market_cap': 420000000000},
    {'symbol': 'JPM', 'sector': 'Finance', 'heat_score': 0.69, 'price': 198.40, 'market_cap': 580000000000}
]

# Request/Response Models
class StockAnalysisRequest(BaseModel):
    symbol: str
    include_heat_map: bool = True

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
        "graph": {
            "nodes": [
                {"id": "ROOT_SECTOR", "type": "root", "level": 0},
                {"id": "SECTOR_TECHNOLOGY", "type": "sector", "level": 1},
                {"id": "SECTOR_HEALTHCARE", "type": "sector", "level": 1},
                {"id": "STOCK_AAPL", "type": "stock", "level": 2},
                {"id": "STOCK_GOOGL", "type": "stock", "level": 2}
            ],
            "edges": [
                {"source": "ROOT_SECTOR", "target": "SECTOR_TECHNOLOGY", "weight": 1.0},
                {"source": "SECTOR_TECHNOLOGY", "target": "STOCK_AAPL", "weight": 1.0}
            ]
        },
        "node_count": 5,
        "edge_count": 2,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/heat/distribution")
async def get_heat_distribution():
    """Get current heat distribution across the graph"""
    # Add some randomness to simulate real-time updates
    for sector in mock_sectors:
        sector['heat'] += random.uniform(-0.05, 0.05)
        sector['heat'] = max(0, min(1, sector['heat']))
    
    for stock in mock_stocks:
        stock['heat_score'] += random.uniform(-0.03, 0.03)
        stock['heat_score'] = max(0, min(1, stock['heat_score']))
    
    return {
        "timestamp": datetime.now().isoformat(),
        "heat_distribution": {f"STOCK_{stock['symbol']}": stock['heat_score'] for stock in mock_stocks},
        "top_sectors": mock_sectors[:3],
        "top_stocks": sorted(mock_stocks, key=lambda x: x['heat_score'], reverse=True)
    }

@app.post("/api/analyze/stock")
async def analyze_stock(request: StockAnalysisRequest):
    """Analyze a specific stock using multi-agent system"""
    
    # Find stock in mock data
    stock = next((s for s in mock_stocks if s['symbol'] == request.symbol), None)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")
    
    # Generate mock analysis
    heat_score = stock['heat_score']
    
    if heat_score > 0.8:
        action = "STRONG BUY"
        confidence = "High"
        explanation = "Strong fundamentals with positive sentiment and favorable technical indicators"
    elif heat_score > 0.7:
        action = "BUY"
        confidence = "Medium"
        explanation = "Good fundamentals with moderate positive sentiment"
    elif heat_score > 0.5:
        action = "HOLD"
        confidence = "Medium"
        explanation = "Mixed signals with neutral market conditions"
    else:
        action = "SELL"
        confidence = "Medium"
        explanation = "Weak fundamentals with negative market sentiment"
    
    return {
        'symbol': request.symbol,
        'fundamental_analysis': {
            'market_cap': stock['market_cap'],
            'pe_ratio': 25.4,
            'fundamental_score': heat_score - 0.1
        },
        'sentiment_analysis': {
            'news_sentiment': {
                'sentiment_label': 'Positive' if heat_score > 0.6 else 'Neutral',
                'average_sentiment': heat_score - 0.5
            },
            'sentiment_score': heat_score
        },
        'valuation_analysis': {
            'current_price': stock['price'],
            'rsi': 45.2,
            'valuation_score': heat_score + 0.05
        },
        'heat_score': heat_score,
        'recommendation': {
            'action': action,
            'confidence': confidence,
            'heat_score': heat_score,
            'explanation': explanation
        },
        'timestamp': datetime.now().isoformat()
    }

@app.get("/api/recommendations/top")
async def get_top_recommendations(limit: int = 10):
    """Get top stock recommendations based on heat scores"""
    
    recommendations = sorted(mock_stocks, key=lambda x: x['heat_score'], reverse=True)
    
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
                'heat_scores': {f"STOCK_{stock['symbol']}": stock['heat_score'] for stock in mock_stocks},
                'top_sectors': mock_sectors[:3]
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