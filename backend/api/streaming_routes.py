"""
WebSocket Streaming Routes for Real-time Market Data
Provides live streaming data to frontend with multiple API sources
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import logging
import json
import asyncio
from typing import List
from datetime import datetime
import sys
import os

# Add services path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

from high_speed_streaming_service import streaming_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/streaming", tags=["Real-time Streaming"])

class StreamingManager:
    """Manage WebSocket connections and streaming"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.streaming_task = None

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        await streaming_service.add_websocket_connection(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        asyncio.create_task(streaming_service.remove_websocket_connection(websocket))
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def start_streaming_if_needed(self):
        """Start streaming if we have connections and it's not already running"""
        if self.active_connections and not streaming_service.is_streaming:
            # Start streaming in background
            self.streaming_task = asyncio.create_task(streaming_service.start_streaming())
            logger.info("🚀 Started streaming service for WebSocket clients")

    async def stop_streaming_if_no_connections(self):
        """Stop streaming if no connections"""
        if not self.active_connections and streaming_service.is_streaming:
            streaming_service.stop_streaming()
            if self.streaming_task:
                self.streaming_task.cancel()
            logger.info("⏹️ Stopped streaming service - no active connections")

# Global streaming manager
streaming_manager = StreamingManager()

@router.websocket("/live-data")
async def websocket_live_data(websocket: WebSocket):
    """WebSocket endpoint for live market data streaming"""
    await streaming_manager.connect(websocket)
    
    try:
        # Start streaming service
        await streaming_manager.start_streaming_if_needed()
        
        # Send initial status
        status = await streaming_service.get_streaming_status()
        await websocket.send_text(json.dumps({
            'type': 'connection_status',
            'status': 'connected',
            'streaming_info': status
        }))
        
        # Keep connection alive and handle client messages
        while True:
            # Wait for client messages (like ping/pong)
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                data = json.loads(message)
                
                if data.get('type') == 'ping':
                    await websocket.send_text(json.dumps({'type': 'pong', 'timestamp': streaming_service._get_timestamp()}))
                elif data.get('type') == 'subscribe':
                    symbols = data.get('symbols', streaming_service.symbols)
                    streaming_service.symbols = symbols
                    await websocket.send_text(json.dumps({
                        'type': 'subscribed',
                        'symbols': symbols
                    }))
                    
            except asyncio.TimeoutError:
                # Send heartbeat every 30 seconds
                await websocket.send_text(json.dumps({
                    'type': 'heartbeat',
                    'timestamp': datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        streaming_manager.disconnect(websocket)
        await streaming_manager.stop_streaming_if_no_connections()

@router.get("/status")
async def get_streaming_status():
    """Get current streaming service status"""
    try:
        status = await streaming_service.get_streaming_status()
        return {
            'streaming_service': status,
            'websocket_connections': len(streaming_manager.active_connections),
            'total_connections': len(streaming_service.active_connections)
        }
    except Exception as e:
        logger.error(f"Error getting streaming status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start")
async def start_streaming():
    """Manually start streaming service"""
    try:
        if not streaming_service.is_streaming:
            streaming_manager.streaming_task = asyncio.create_task(streaming_service.start_streaming())
            return {
                'status': 'started',
                'message': 'High-speed streaming service started',
                'connections': len(streaming_manager.active_connections)
            }
        else:
            return {
                'status': 'already_running',
                'message': 'Streaming service is already running',
                'connections': len(streaming_manager.active_connections)
            }
    except Exception as e:
        logger.error(f"Error starting streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_streaming():
    """Manually stop streaming service"""
    try:
        streaming_service.stop_streaming()
        if streaming_manager.streaming_task:
            streaming_manager.streaming_task.cancel()
        
        return {
            'status': 'stopped',
            'message': 'Streaming service stopped',
            'connections': len(streaming_manager.active_connections)
        }
    except Exception as e:
        logger.error(f"Error stopping streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test-apis")
async def test_api_speeds():
    """Test speed of different APIs"""
    try:
        import time
        test_symbol = "AAPL"
        results = {}
        
        for api_name, api_func in streaming_service.apis.items():
            start_time = time.time()
            try:
                data = await api_func(test_symbol)
                end_time = time.time()
                
                results[api_name] = {
                    'success': data is not None,
                    'response_time_ms': int((end_time - start_time) * 1000),
                    'data': data.__dict__ if data else None
                }
            except Exception as e:
                results[api_name] = {
                    'success': False,
                    'error': str(e),
                    'response_time_ms': 0
                }
        
        # Find fastest API
        fastest_api = min(
            [k for k, v in results.items() if v['success']], 
            key=lambda k: results[k]['response_time_ms'],
            default=None
        )
        
        return {
            'test_symbol': test_symbol,
            'fastest_api': fastest_api,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error testing APIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/demo")
async def get_streaming_demo():
    """Get HTML demo page for WebSocket streaming"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>RAGHeat Live Data Stream</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .stock { margin: 10px; padding: 15px; background: #2d2d2d; border-radius: 8px; }
        .positive { color: #00ff88; }
        .negative { color: #ff4444; }
        .price { font-size: 1.2em; font-weight: bold; }
        .change { font-size: 1em; }
        .status { padding: 10px; background: #333; margin-bottom: 20px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🔥 RAGHeat Live Stream</h1>
    <div id="status" class="status">Connecting...</div>
    <div id="stocks"></div>
    
    <script>
        const ws = new WebSocket('ws://localhost:8000/api/streaming/live-data');
        const statusDiv = document.getElementById('status');
        const stocksDiv = document.getElementById('stocks');
        
        ws.onopen = function(event) {
            statusDiv.innerHTML = '✅ Connected to live stream';
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'market_update') {
                statusDiv.innerHTML = `📡 Live Stream Active - ${data.count} stocks updated in ${data.fetch_time_ms}ms`;
                
                stocksDiv.innerHTML = '';
                for (const [symbol, stock] of Object.entries(data.data)) {
                    const changeClass = stock.change >= 0 ? 'positive' : 'negative';
                    const changeSymbol = stock.change >= 0 ? '+' : '';
                    
                    stocksDiv.innerHTML += `
                        <div class="stock">
                            <strong>${symbol}</strong>
                            <div class="price">$${stock.price}</div>
                            <div class="change ${changeClass}">
                                ${changeSymbol}${stock.change} (${changeSymbol}${stock.change_percent}%)
                            </div>
                            <small>Vol: ${stock.volume.toLocaleString()} | Source: ${stock.source}</small>
                        </div>
                    `;
                }
            }
        };
        
        ws.onclose = function(event) {
            statusDiv.innerHTML = '❌ Disconnected from stream';
        };
        
        ws.onerror = function(error) {
            statusDiv.innerHTML = '💥 Stream error: ' + error;
        };
        
        // Send ping every 25 seconds
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'ping'}));
            }
        }, 25000);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)