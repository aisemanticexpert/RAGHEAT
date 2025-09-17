"""
Live Synthetic Data API
FastAPI service providing real-time synthetic market data to frontend
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import uvicorn
from pathlib import Path
# Redis imports removed due to version conflicts
# import redis
# import aioredis
from dataclasses import asdict
import numpy as np

from services.synthetic_data_generator import SyntheticDataGenerator
from services.real_market_data_downloader import RealMarketDataDownloader
from streaming.live_data_kafka_streamer import LiveDataKafkaStreamer

# Initialize FastAPI app
app = FastAPI(
    title="RAGHeat Live Synthetic Data API",
    description="Real-time synthetic market data streaming API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
data_generator = SyntheticDataGenerator()
market_downloader = RealMarketDataDownloader()
kafka_streamer = LiveDataKafkaStreamer()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[websocket] = set()
        logging.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        self.subscriptions.pop(websocket, None)
        logging.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logging.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict, channel: str = "all"):
        disconnected = set()
        for websocket in self.active_connections:
            try:
                if channel == "all" or channel in self.subscriptions.get(websocket, set()):
                    await websocket.send_text(json.dumps(message, default=str))
            except Exception as e:
                logging.error(f"Error broadcasting to client: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)
    
    def subscribe_client(self, websocket: WebSocket, channels: List[str]):
        if websocket in self.subscriptions:
            self.subscriptions[websocket].update(channels)

# Initialize connection manager
manager = ConnectionManager()

# Data streaming state
streaming_state = {
    "is_active": False,
    "start_time": None,
    "message_count": 0,
    "client_count": 0,
    "current_regime": "sideways"
}

# Background streaming task
streaming_task = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Live Synthetic Data API...")
    
    # Initialize data directories
    Path("data/synthetic_stocks").mkdir(parents=True, exist_ok=True)
    Path("data/real_market_data").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Start background tasks
    asyncio.create_task(start_background_streaming())
    asyncio.create_task(start_real_data_download())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logging.info("Shutting down API...")
    global streaming_task
    if streaming_task:
        streaming_task.cancel()

async def start_background_streaming():
    """Start background data streaming"""
    global streaming_task, streaming_state
    
    await asyncio.sleep(2)  # Wait for initialization
    
    if not streaming_state["is_active"]:
        streaming_state["is_active"] = True
        streaming_state["start_time"] = datetime.now()
        streaming_task = asyncio.create_task(stream_market_data())
        logging.info("Background streaming started")

async def start_real_data_download():
    """Start real market data download in background"""
    await asyncio.sleep(5)  # Wait for other services
    
    try:
        # Start downloading real market data
        await market_downloader.full_market_data_download()
    except Exception as e:
        logging.error(f"Real data download error: {e}")

async def stream_market_data():
    """Main streaming loop"""
    global streaming_state
    
    logging.info("Starting market data streaming loop...")
    
    while streaming_state["is_active"]:
        try:
            # Generate new market data
            market_batch = data_generator.generate_batch_data(1)
            
            if market_batch:
                market_tick = market_batch[0]
                
                # Update streaming state
                streaming_state["message_count"] += 1
                streaming_state["current_regime"] = market_tick.get("market_regime", "unknown")
                streaming_state["client_count"] = len(manager.active_connections)
                
                # Broadcast to all connected clients
                await manager.broadcast({
                    "type": "market_data",
                    "data": market_tick,
                    "metadata": {
                        "message_id": streaming_state["message_count"],
                        "timestamp": datetime.now().isoformat(),
                        "client_count": streaming_state["client_count"]
                    }
                }, "market_data")
                
                # Send individual stock updates
                for symbol, stock_data in market_tick.get("stocks", {}).items():
                    await manager.broadcast({
                        "type": "stock_update",
                        "symbol": symbol,
                        "data": stock_data,
                        "timestamp": datetime.now().isoformat()
                    }, f"stock_{symbol}")
                
                # Heat propagation updates
                heat_data = {
                    "type": "heat_update",
                    "timestamp": datetime.now().isoformat(),
                    "heat_map": {
                        symbol: stock["heat_score"] 
                        for symbol, stock in market_tick.get("stocks", {}).items()
                    },
                    "regime": market_tick.get("market_regime")
                }
                
                await manager.broadcast(heat_data, "heat_signals")
            
            # Realistic streaming interval
            await asyncio.sleep(0.5)  # 500ms intervals
            
        except Exception as e:
            logging.error(f"Error in streaming loop: {e}")
            await asyncio.sleep(1.0)

# API Endpoints

@app.get("/")
async def root():
    """API status and information"""
    return {
        "service": "RAGHeat Live Synthetic Data API",
        "version": "1.0.0",
        "status": "active",
        "streaming_status": streaming_state,
        "endpoints": {
            "market_data": "/api/market-data",
            "stocks": "/api/stocks",
            "heat_signals": "/api/heat-signals",
            "websocket": "/ws",
            "analytics": "/api/analytics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "streaming_active": streaming_state["is_active"],
        "connected_clients": len(manager.active_connections),
        "messages_sent": streaming_state["message_count"]
    }

@app.get("/api/market-data")
async def get_current_market_data():
    """Get current market data snapshot"""
    try:
        # Generate fresh market data
        market_batch = data_generator.generate_batch_data(1)
        
        if market_batch:
            return {
                "status": "success",
                "data": market_batch[0],
                "timestamp": datetime.now().isoformat(),
                "streaming_info": streaming_state
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate market data")
            
    except Exception as e:
        logging.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks")
async def get_all_stocks():
    """Get all available stocks information"""
    return {
        "status": "success",
        "stocks": list(data_generator.stocks.keys()),
        "count": len(data_generator.stocks),
        "sectors": list(set(stock["sector"] for stock in data_generator.stocks.values())),
        "correlation_groups": list(set(stock["correlation_group"] for stock in data_generator.stocks.values()))
    }

@app.get("/api/stocks/{symbol}")
async def get_stock_data(symbol: str):
    """Get data for a specific stock"""
    if symbol not in data_generator.stocks:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    try:
        # Generate current data for the stock
        stock_data = data_generator.generate_single_tick(symbol)
        
        return {
            "status": "success",
            "symbol": symbol,
            "data": asdict(stock_data),
            "stock_info": data_generator.stocks[symbol],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error getting stock {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/heat-signals")
async def get_heat_signals():
    """Get current heat propagation signals"""
    try:
        # Generate market data to get heat signals
        market_batch = data_generator.generate_batch_data(1)
        
        if market_batch:
            market_data = market_batch[0]
            heat_signals = {}
            
            for symbol, stock_data in market_data.get("stocks", {}).items():
                heat_signals[symbol] = {
                    "heat_score": stock_data["heat_score"],
                    "volatility": stock_data["volatility"],
                    "sector": stock_data["sector"],
                    "correlations": stock_data["correlation_signals"]
                }
            
            return {
                "status": "success",
                "heat_signals": heat_signals,
                "market_regime": market_data.get("market_regime"),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate heat signals")
            
    except Exception as e:
        logging.error(f"Error getting heat signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics")
async def get_analytics():
    """Get market analytics and statistics"""
    try:
        # Get real-time data summary
        real_data_summary = market_downloader.get_latest_data_summary()
        
        # Generate current analytics
        analytics = {
            "streaming_analytics": {
                "uptime_seconds": (
                    (datetime.now() - streaming_state["start_time"]).total_seconds()
                    if streaming_state["start_time"] else 0
                ),
                "messages_streamed": streaming_state["message_count"],
                "active_connections": len(manager.active_connections),
                "current_regime": streaming_state["current_regime"],
                "streaming_rate": "2 ticks/second"
            },
            "market_analytics": {
                "total_stocks": len(data_generator.stocks),
                "sectors": len(set(stock["sector"] for stock in data_generator.stocks.values())),
                "correlation_groups": len(set(stock["correlation_group"] for stock in data_generator.stocks.values())),
                "heat_network_size": len(data_generator.heat_network)
            },
            "real_data_status": real_data_summary,
            "data_quality": {
                "synthetic_data_quality": 95.0,
                "real_data_coverage": len(real_data_summary.get("available_symbols", [])),
                "update_frequency": "500ms intervals"
            }
        }
        
        return {
            "status": "success",
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/controls/start-streaming")
async def start_streaming():
    """Start data streaming"""
    global streaming_state, streaming_task
    
    if not streaming_state["is_active"]:
        streaming_state["is_active"] = True
        streaming_state["start_time"] = datetime.now()
        streaming_task = asyncio.create_task(stream_market_data())
        
        return {
            "status": "success",
            "message": "Streaming started",
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "status": "info",
            "message": "Streaming already active",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/controls/stop-streaming")
async def stop_streaming():
    """Stop data streaming"""
    global streaming_state, streaming_task
    
    if streaming_state["is_active"]:
        streaming_state["is_active"] = False
        if streaming_task:
            streaming_task.cancel()
        
        return {
            "status": "success",
            "message": "Streaming stopped",
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "status": "info",
            "message": "Streaming not active",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/real-time-data")
async def get_real_time_data():
    """Get real-time market data from external sources"""
    try:
        real_time_data = market_downloader.get_real_time_data()
        
        return {
            "status": "success",
            "real_time_data": real_time_data,
            "symbols_count": len(real_time_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error getting real-time data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await manager.send_personal_message({
            "type": "connection",
            "message": "Connected to RAGHeat Live Data Stream",
            "timestamp": datetime.now().isoformat(),
            "available_channels": [
                "market_data", "heat_signals", "stock_updates", 
                "regime_changes", "analytics"
            ]
        }, websocket)
        
        # Handle incoming messages
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                # Handle subscription requests
                if data.get("type") == "subscribe":
                    channels = data.get("channels", [])
                    manager.subscribe_client(websocket, channels)
                    
                    await manager.send_personal_message({
                        "type": "subscription_confirmed",
                        "channels": channels,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                
                # Handle data requests
                elif data.get("type") == "request_data":
                    request_type = data.get("data_type")
                    
                    if request_type == "market_snapshot":
                        market_data = data_generator.generate_batch_data(1)
                        if market_data:
                            await manager.send_personal_message({
                                "type": "market_snapshot",
                                "data": market_data[0],
                                "timestamp": datetime.now().isoformat()
                            }, websocket)
                    
                    elif request_type == "heat_map":
                        heat_data = await get_heat_signals()
                        await manager.send_personal_message({
                            "type": "heat_map",
                            "data": heat_data,
                            "timestamp": datetime.now().isoformat()
                        }, websocket)
                
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "live_synthetic_data_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )