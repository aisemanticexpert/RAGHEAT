"""
Enhanced Graph API Routes - Supporting professional knowledge graph visualization
Provides real-time data streaming with market hours detection and heated node analysis
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json
import random
import time

from fastapi import APIRouter, WebSocket, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel

# Initialize router
router = APIRouter()
logger = logging.getLogger(__name__)

# Data models
class NodeData(BaseModel):
    id: str
    label: str
    type: str  # 'market', 'sector', 'stock'
    heat_level: float
    price: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    volatility: Optional[float] = None
    ai_prediction: Optional[float] = None

class RelationshipData(BaseModel):
    source: str
    target: str
    type: str  # 'CONTAINS', 'BELONGS_TO', 'CORRELATES_WITH', etc.
    weight: float
    strength: float
    properties: Optional[Dict[str, Any]] = None

class GraphData(BaseModel):
    nodes: List[NodeData]
    relationships: List[RelationshipData]
    metadata: Dict[str, Any]

# Global state for WebSocket connections and data streaming
active_connections: List[WebSocket] = []
current_graph_data: Optional[GraphData] = None
streaming_task: Optional[asyncio.Task] = None
last_update: Optional[datetime] = None

# Market data simulation with heat physics
class MarketDataSimulator:
    def __init__(self):
        self.sectors = {
            'TECHNOLOGY': {
                'name': 'Technology',
                'stocks': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'CRM', 'ADBE', 'ORCL'],
                'base_heat': 0.6,
                'volatility': 0.3
            },
            'FINANCIALS': {
                'name': 'Financials', 
                'stocks': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC'],
                'base_heat': 0.4,
                'volatility': 0.2
            },
            'HEALTHCARE': {
                'name': 'Healthcare',
                'stocks': ['JNJ', 'PFE', 'UNH', 'ABBV', 'LLY', 'TMO', 'ABT', 'MRK'],
                'base_heat': 0.3,
                'volatility': 0.15
            },
            'ENERGY': {
                'name': 'Energy',
                'stocks': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'KMI', 'OKE'],
                'base_heat': 0.5,
                'volatility': 0.4
            },
            'CONSUMER_DISCRETIONARY': {
                'name': 'Consumer Discretionary',
                'stocks': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'TJX', 'SBUX'],
                'base_heat': 0.45,
                'volatility': 0.25
            }
        }
        
        self.heat_states = {}  # Track heat evolution over time
        self.correlation_matrix = self._generate_correlation_matrix()
        self.market_sentiment = 0.0  # Global market sentiment (-1 to 1)
        
    def _generate_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Generate realistic stock correlation matrix"""
        correlations = {}
        all_stocks = []
        
        for sector_data in self.sectors.values():
            all_stocks.extend(sector_data['stocks'])
            
        for stock1 in all_stocks:
            correlations[stock1] = {}
            sector1 = self._get_stock_sector(stock1)
            
            for stock2 in all_stocks:
                if stock1 == stock2:
                    correlations[stock1][stock2] = 1.0
                else:
                    sector2 = self._get_stock_sector(stock2)
                    
                    # Higher correlation within same sector
                    if sector1 == sector2:
                        base_corr = 0.6 + random.uniform(-0.2, 0.3)
                    else:
                        base_corr = 0.1 + random.uniform(-0.1, 0.3)
                    
                    correlations[stock1][stock2] = max(-0.5, min(0.95, base_corr))
        
        return correlations
    
    def _get_stock_sector(self, stock: str) -> str:
        """Get sector for a given stock"""
        for sector_key, sector_data in self.sectors.items():
            if stock in sector_data['stocks']:
                return sector_key
        return 'TECHNOLOGY'  # Default fallback
    
    def _apply_heat_diffusion(self, current_heat: Dict[str, float]) -> Dict[str, float]:
        """Apply heat diffusion physics across the graph"""
        new_heat = current_heat.copy()
        
        # Apply correlation-based heat diffusion
        for stock1, heat1 in current_heat.items():
            if stock1 not in self.correlation_matrix:
                continue
                
            heat_influence = 0.0
            influence_count = 0
            
            for stock2, correlation in self.correlation_matrix[stock1].items():
                if stock2 != stock1 and stock2 in current_heat:
                    heat2 = current_heat[stock2]
                    # Heat flows from high to low, influenced by correlation
                    heat_influence += correlation * (heat2 - heat1) * 0.1
                    influence_count += 1
            
            if influence_count > 0:
                # Apply heat diffusion with decay
                new_heat[stock1] = max(0.05, min(0.95, heat1 + heat_influence))
        
        return new_heat
    
    def _generate_realistic_price_data(self, stock: str, current_heat: float) -> Dict[str, float]:
        """Generate realistic price and volume data based on heat level"""
        sector = self._get_stock_sector(stock)
        sector_data = self.sectors[sector]
        
        # Base price influenced by heat and market sentiment
        base_price = 100 + random.uniform(50, 400)
        
        # Change percent influenced by heat level and volatility
        heat_influence = (current_heat - 0.5) * 2  # Scale to -1 to 1
        volatility_factor = sector_data['volatility']
        sentiment_influence = self.market_sentiment * 0.3
        
        change_percent = (
            heat_influence * 5 +  # Heat drives price movement
            sentiment_influence * 3 +  # Market sentiment
            random.normalvariate(0, volatility_factor * 2)  # Random volatility
        )
        
        # Volume inversely related to price stability, directly to heat
        base_volume = 1000000 + int(current_heat * 50000000)
        volume_noise = random.uniform(0.5, 2.0)
        volume = int(base_volume * volume_noise)
        
        # Market cap calculation
        shares_outstanding = random.uniform(100, 10000) * 1000000  # 100M to 10B shares
        current_price = base_price * (1 + change_percent / 100)
        market_cap = current_price * shares_outstanding
        
        return {
            'price': current_price,
            'change_percent': change_percent,
            'volume': volume,
            'market_cap': market_cap,
            'volatility': volatility_factor + (current_heat * 0.2)
        }
    
    def generate_synthetic_data(self) -> GraphData:
        """Generate comprehensive synthetic market data with heat physics"""
        nodes = []
        relationships = []
        
        # Update market sentiment (slow drift)
        self.market_sentiment += random.normalvariate(0, 0.05)
        self.market_sentiment = max(-1.0, min(1.0, self.market_sentiment))
        
        # Generate or update heat states
        all_stocks = []
        for sector_data in self.sectors.values():
            all_stocks.extend(sector_data['stocks'])
        
        if not self.heat_states:
            # Initialize heat states
            for stock in all_stocks:
                sector = self._get_stock_sector(stock)
                base_heat = self.sectors[sector]['base_heat']
                self.heat_states[stock] = base_heat + random.uniform(-0.2, 0.2)
        else:
            # Apply heat diffusion physics
            self.heat_states = self._apply_heat_diffusion(self.heat_states)
            
            # Add random heat events (news, earnings, etc.)
            for stock in all_stocks:
                if random.random() < 0.05:  # 5% chance of heat event
                    heat_event = random.uniform(-0.3, 0.4)  # Bias towards heating
                    self.heat_states[stock] = max(0.05, min(0.95, 
                        self.heat_states[stock] + heat_event))
        
        # Create market root node
        market_heat = np.mean(list(self.heat_states.values()))
        nodes.append(NodeData(
            id='MARKET',
            label='Global Market',
            type='market',
            heat_level=market_heat,
            ai_prediction=market_heat + random.uniform(-0.1, 0.1)
        ))
        
        # Create sector nodes and relationships
        for sector_key, sector_data in self.sectors.items():
            # Calculate sector heat as average of its stocks
            sector_stocks = sector_data['stocks']
            sector_heat = np.mean([self.heat_states[stock] for stock in sector_stocks])
            
            # Generate sector-level metrics
            sector_performance = (sector_heat - 0.5) * 20  # Scale to percentage
            sector_volatility = sector_data['volatility'] + (sector_heat * 0.1)
            
            nodes.append(NodeData(
                id=sector_key,
                label=sector_data['name'],
                type='sector',
                heat_level=sector_heat,
                change_percent=sector_performance,
                volatility=sector_volatility,
                ai_prediction=sector_heat + random.uniform(-0.1, 0.15)
            ))
            
            # Connect sector to market
            relationships.append(RelationshipData(
                source='MARKET',
                target=sector_key,
                type='CONTAINS',
                weight=4,
                strength=sector_heat,
                properties={'stock_count': len(sector_stocks)}
            ))
            
            # Create stock nodes and connect to sectors
            for stock in sector_stocks:
                stock_heat = self.heat_states[stock]
                price_data = self._generate_realistic_price_data(stock, stock_heat)
                
                nodes.append(NodeData(
                    id=stock,
                    label=stock,
                    type='stock',
                    heat_level=stock_heat,
                    sector=sector_key.lower(),
                    price=price_data['price'],
                    change_percent=price_data['change_percent'],
                    volume=price_data['volume'],
                    market_cap=price_data['market_cap'],
                    volatility=price_data['volatility'],
                    ai_prediction=stock_heat + random.uniform(-0.15, 0.15)
                ))
                
                # Connect stock to sector
                relationships.append(RelationshipData(
                    source=sector_key,
                    target=stock,
                    type='BELONGS_TO',
                    weight=3,
                    strength=stock_heat,
                    properties={'heat_level': stock_heat}
                ))
        
        # Add correlation-based relationships between stocks
        tech_stocks = self.sectors['TECHNOLOGY']['stocks'][:5]
        for i, stock1 in enumerate(tech_stocks):
            for stock2 in tech_stocks[i+1:]:
                correlation = self.correlation_matrix.get(stock1, {}).get(stock2, 0)
                if correlation > 0.7:  # High correlation threshold
                    relationships.append(RelationshipData(
                        source=stock1,
                        target=stock2,
                        type='CORRELATES_WITH',
                        weight=2,
                        strength=correlation,
                        properties={
                            'correlation': correlation,
                            'value': f"{int(correlation * 100)}% corr"
                        }
                    ))
        
        metadata = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_source': 'synthetic',
            'market_sentiment': self.market_sentiment,
            'total_nodes': len(nodes),
            'total_relationships': len(relationships),
            'heat_stats': {
                'avg_heat': float(np.mean(list(self.heat_states.values()))),
                'max_heat': float(np.max(list(self.heat_states.values()))),
                'min_heat': float(np.min(list(self.heat_states.values()))),
                'hot_stocks': [stock for stock, heat in self.heat_states.items() if heat > 0.7]
            }
        }
        
        return GraphData(
            nodes=nodes,
            relationships=relationships,
            metadata=metadata
        )

# Initialize simulator
market_simulator = MarketDataSimulator()

def is_market_open() -> bool:
    """Determine if the market is currently open"""
    now = datetime.now(timezone.utc)
    
    # Convert to ET (market timezone)
    et_offset = -5 if now.month in [11, 12, 1, 2, 3] else -4  # EST/EDT
    market_time = now.replace(hour=now.hour + et_offset)
    
    # Market is open Mon-Fri, 9:30 AM - 4:00 PM ET
    weekday = market_time.weekday()  # 0 = Monday
    hour = market_time.hour
    minute = market_time.minute
    
    if weekday > 4:  # Weekend
        return False
    
    # Check market hours (9:30 AM to 4:00 PM ET)
    market_open_time = 9.5  # 9:30 AM
    market_close_time = 16.0  # 4:00 PM
    current_time = hour + minute / 60.0
    
    return market_open_time <= current_time < market_close_time

async def fetch_live_market_data() -> GraphData:
    """Fetch real market data (placeholder for actual data providers)"""
    # In production, this would connect to:
    # - Yahoo Finance API
    # - Alpha Vantage
    # - IEX Cloud
    # - Bloomberg API
    # - Polygon.io
    # - Financial Modeling Prep API
    
    # For now, return enhanced synthetic data during market hours
    return market_simulator.generate_synthetic_data()

@router.get("/api/graph/live-data")
async def get_live_graph_data():
    """Get current graph data - live during market hours, synthetic otherwise"""
    try:
        global current_graph_data, last_update
        
        market_open = is_market_open()
        data_source = 'live' if market_open else 'synthetic'
        
        logger.info(f"📊 Fetching {data_source} graph data (market {'open' if market_open else 'closed'})")
        
        if market_open:
            graph_data = await fetch_live_market_data()
        else:
            graph_data = market_simulator.generate_synthetic_data()
        
        # Update global state
        current_graph_data = graph_data
        last_update = datetime.now(timezone.utc)
        
        # Broadcast to WebSocket connections
        if active_connections:
            await broadcast_graph_update({
                'type': 'graph_update',
                'data': graph_data.dict(),
                'market_status': 'open' if market_open else 'closed'
            })
        
        return JSONResponse({
            'success': True,
            'data': graph_data.dict(),
            'market_status': 'open' if market_open else 'closed',
            'last_update': last_update.isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error fetching graph data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch graph data: {str(e)}")

@router.get("/api/graph/synthetic-data")
async def get_synthetic_graph_data():
    """Get synthetic graph data for testing and after-hours"""
    try:
        logger.info("🎲 Generating synthetic graph data")
        
        graph_data = market_simulator.generate_synthetic_data()
        
        return JSONResponse({
            'success': True,
            'data': graph_data.dict(),
            'market_status': 'synthetic',
            'last_update': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error generating synthetic data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate synthetic data: {str(e)}")

@router.get("/api/graph/heat-analysis")
async def get_heat_analysis():
    """Get detailed heat analysis and predictions"""
    try:
        if not current_graph_data:
            # Generate fresh data if none exists
            current_graph_data = market_simulator.generate_synthetic_data()
        
        # Extract heat statistics
        stock_nodes = [node for node in current_graph_data.nodes if node.type == 'stock']
        heat_levels = [node.heat_level for node in stock_nodes]
        
        if not heat_levels:
            raise HTTPException(status_code=404, detail="No heat data available")
        
        # Calculate heat statistics
        heat_stats = {
            'average_heat': float(np.mean(heat_levels)),
            'median_heat': float(np.median(heat_levels)),
            'std_heat': float(np.std(heat_levels)),
            'min_heat': float(np.min(heat_levels)),
            'max_heat': float(np.max(heat_levels))
        }
        
        # Categorize stocks by heat
        hot_stocks = [node for node in stock_nodes if node.heat_level > 0.7]
        warm_stocks = [node for node in stock_nodes if 0.4 < node.heat_level <= 0.7]
        cool_stocks = [node for node in stock_nodes if node.heat_level <= 0.4]
        
        # Generate heat predictions (simple trend analysis)
        predictions = []
        for node in stock_nodes[:10]:  # Top 10 for performance
            # Simulate prediction based on current heat and volatility
            current_heat = node.heat_level
            trend = random.uniform(-0.2, 0.3) if current_heat > 0.5 else random.uniform(-0.1, 0.2)
            predicted_heat = max(0.05, min(0.95, current_heat + trend))
            
            predictions.append({
                'stock': node.id,
                'current_heat': current_heat,
                'predicted_heat': predicted_heat,
                'trend': 'heating' if trend > 0.05 else 'cooling' if trend < -0.05 else 'stable',
                'confidence': random.uniform(0.6, 0.9)
            })
        
        analysis_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'heat_statistics': heat_stats,
            'categorization': {
                'hot_stocks': [{'id': node.id, 'heat': node.heat_level, 'change': node.change_percent} for node in hot_stocks],
                'warm_stocks': [{'id': node.id, 'heat': node.heat_level, 'change': node.change_percent} for node in warm_stocks],
                'cool_stocks': [{'id': node.id, 'heat': node.heat_level, 'change': node.change_percent} for node in cool_stocks]
            },
            'predictions': predictions,
            'market_sentiment': market_simulator.market_sentiment,
            'heat_diffusion_active': True
        }
        
        return JSONResponse({
            'success': True,
            'data': analysis_data
        })
        
    except Exception as e:
        logger.error(f"❌ Error in heat analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Heat analysis failed: {str(e)}")

@router.websocket("/ws/live-graph")
async def websocket_live_graph(websocket: WebSocket):
    """WebSocket endpoint for real-time graph data streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    
    logger.info(f"🔌 WebSocket connected. Active connections: {len(active_connections)}")
    
    try:
        # Send initial data
        if current_graph_data:
            await websocket.send_json({
                'type': 'initial_data',
                'data': current_graph_data.dict(),
                'market_status': 'open' if is_market_open() else 'closed'
            })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client message with timeout
                message = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                if message.get('type') == 'ping':
                    await websocket.send_json({'type': 'pong', 'timestamp': time.time()})
                elif message.get('type') == 'request_update':
                    # Send latest data
                    if current_graph_data:
                        await websocket.send_json({
                            'type': 'graph_update',
                            'data': current_graph_data.dict(),
                            'market_status': 'open' if is_market_open() else 'closed'
                        })
                
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({'type': 'keepalive', 'timestamp': time.time()})
                
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected. Active connections: {len(active_connections)}")

async def broadcast_graph_update(message: Dict):
    """Broadcast message to all active WebSocket connections"""
    if not active_connections:
        return
    
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.error(f"❌ Failed to send to WebSocket: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for connection in disconnected:
        if connection in active_connections:
            active_connections.remove(connection)

async def start_background_streaming():
    """Start background task for continuous data streaming"""
    global streaming_task
    
    if streaming_task and not streaming_task.done():
        return  # Already running
    
    async def streaming_loop():
        while True:
            try:
                # Generate new data every 30 seconds during market hours, 60 seconds after hours
                market_open = is_market_open()
                update_interval = 30 if market_open else 60
                
                # Update global data
                if market_open:
                    new_data = await fetch_live_market_data()
                else:
                    new_data = market_simulator.generate_synthetic_data()
                
                global current_graph_data, last_update
                current_graph_data = new_data
                last_update = datetime.now(timezone.utc)
                
                # Broadcast to WebSocket clients
                await broadcast_graph_update({
                    'type': 'graph_update',
                    'data': new_data.dict(),
                    'market_status': 'open' if market_open else 'closed',
                    'timestamp': last_update.isoformat()
                })
                
                # Broadcast heat-specific updates
                heat_updates = {
                    node.id: node.heat_level 
                    for node in new_data.nodes 
                    if node.type == 'stock' and node.heat_level > 0.6
                }
                
                if heat_updates:
                    await broadcast_graph_update({
                        'type': 'heat_update',
                        'data': heat_updates,
                        'timestamp': last_update.isoformat()
                    })
                
                logger.info(f"📡 Broadcast update to {len(active_connections)} connections")
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"❌ Streaming loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    streaming_task = asyncio.create_task(streaming_loop())
    logger.info("🚀 Background streaming task started")

@router.on_event("startup")
async def startup_event():
    """Initialize background streaming on startup"""
    await start_background_streaming()

@router.get("/api/graph/status")
async def get_graph_status():
    """Get current graph system status"""
    return JSONResponse({
        'success': True,
        'data': {
            'market_open': is_market_open(),
            'active_connections': len(active_connections),
            'last_update': last_update.isoformat() if last_update else None,
            'streaming_active': streaming_task is not None and not streaming_task.done(),
            'data_available': current_graph_data is not None,
            'market_sentiment': market_simulator.market_sentiment
        }
    })

# Export router
__all__ = ['router']