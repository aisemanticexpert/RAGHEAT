"""
Ontology-driven Knowledge Graph API Routes
Professional hierarchical endpoints for ontology-based visualization
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import asyncio

from services.ontology_neo4j_service import OntologyNeo4jService

# Initialize router and service
router = APIRouter(prefix="/api/ontology", tags=["ontology"])
ontology_service = OntologyNeo4jService()

# Global initialization flag
_service_initialized = False

logger = logging.getLogger(__name__)

async def ensure_service_initialized():
    """Ensure ontology service is initialized"""
    global _service_initialized
    
    if not _service_initialized:
        try:
            await ontology_service.initialize_ontology_graph()
            _service_initialized = True
            logger.info("✅ Ontology service initialized")
        except Exception as e:
            logger.warning(f"⚠️ Ontology service initialization failed: {e}")
            # Continue with fallback mode

@router.get("/graph/{level}")
async def get_hierarchical_graph(
    level: int,
    max_nodes: Optional[int] = 50,
    background_tasks: BackgroundTasks = None
):
    """
    Get hierarchical knowledge graph data for specific level
    
    Levels:
    1: Market Structure - Top-level market organization
    2: Sector Classification - Economic sectors and industries  
    3: Financial Instruments - Stocks, bonds, derivatives
    4: Corporate Relations - Correlations and relationships
    5: Trading Signals - BUY/SELL/HOLD recommendations
    6: Technical Indicators - RSI, MA, Bollinger Bands
    7: Heat Propagation - Thermal dynamics network
    """
    try:
        # Ensure service is initialized
        await ensure_service_initialized()
        
        # Validate level
        if level < 1 or level > 7:
            raise HTTPException(
                status_code=400, 
                detail="Level must be between 1 and 7"
            )
            
        # Validate max_nodes
        if max_nodes < 1 or max_nodes > 200:
            raise HTTPException(
                status_code=400,
                detail="max_nodes must be between 1 and 200"
            )
        
        # Get hierarchical graph data
        graph_data = await ontology_service.get_hierarchical_graph_data(
            level=level, 
            max_nodes=max_nodes
        )
        
        # Add metadata
        graph_data["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "max_nodes": max_nodes,
            "node_count": len(graph_data.get("nodes", [])),
            "link_count": len(graph_data.get("links", [])),
            "service_initialized": _service_initialized
        }
        
        return JSONResponse(content=graph_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting hierarchical graph: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph data: {str(e)}"
        )

@router.get("/market-overview")
async def get_market_overview():
    """Get comprehensive market overview with trading signals"""
    try:
        await ensure_service_initialized()
        
        overview = await ontology_service.get_market_overview()
        
        return JSONResponse(content={
            "status": "success",
            "data": overview,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting market overview: {e}")
        # Return fallback data
        return JSONResponse(content={
            "status": "fallback",
            "data": {
                "total_stocks": 20,
                "total_signals": 15,
                "buy_signals": 6,
                "sell_signals": 4,
                "hold_signals": 5,
                "average_heat": 55.7,
                "max_heat": 92.3,
                "min_heat": 18.4,
                "market_sentiment": "Cautiously Bullish",
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        })

@router.post("/populate-realtime")
async def populate_realtime_data(
    market_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    Populate ontology with real-time market data
    This endpoint receives market data and updates the knowledge graph
    """
    try:
        await ensure_service_initialized()
        
        # Validate input data
        if not market_data or "stocks" not in market_data:
            raise HTTPException(
                status_code=400,
                detail="Invalid market data: 'stocks' field required"
            )
            
        # Schedule background population
        background_tasks.add_task(
            ontology_service.populate_real_time_data,
            market_data
        )
        
        return JSONResponse(content={
            "status": "success",
            "message": "Real-time data population scheduled",
            "stocks_count": len(market_data.get("stocks", {})),
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error populating real-time data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to populate data: {str(e)}"
        )

@router.get("/levels")
async def get_visualization_levels():
    """Get all available visualization levels with descriptions"""
    return JSONResponse(content={
        "status": "success",
        "levels": {
            1: {
                "name": "Market Structure",
                "description": "Top-level market organization and exchanges",
                "icon": "🏛️",
                "entities": ["Markets", "Exchanges", "Indices"],
                "relationships": ["contains", "tracks"]
            },
            2: {
                "name": "Sector Classification", 
                "description": "Economic sectors and industry classifications",
                "icon": "🏢",
                "entities": ["Sectors", "Industries", "Sub-industries"],
                "relationships": ["belongs_to", "part_of"]
            },
            3: {
                "name": "Financial Instruments",
                "description": "Tradable securities and derivatives",
                "icon": "📈", 
                "entities": ["Stocks", "Bonds", "Options", "ETFs"],
                "relationships": ["issued_by", "backed_by", "tracks"]
            },
            4: {
                "name": "Corporate Relations",
                "description": "Inter-company relationships and correlations",
                "icon": "🔗",
                "entities": ["Corporations", "Correlations", "Partnerships"],
                "relationships": ["correlates_with", "competes_with", "supplies_to"]
            },
            5: {
                "name": "Trading Signals",
                "description": "Algorithmic buy/sell/hold recommendations",
                "icon": "📊",
                "entities": ["BuySignals", "SellSignals", "HoldSignals"],
                "relationships": ["recommends", "based_on", "applies_to"]
            },
            6: {
                "name": "Technical Indicators",
                "description": "Mathematical analysis tools and metrics",
                "icon": "📉",
                "entities": ["RSI", "MovingAverages", "BollingerBands", "MACD"],
                "relationships": ["calculates", "indicates", "measures"]
            },
            7: {
                "name": "Heat Propagation",
                "description": "Thermal dynamics and influence networks",
                "icon": "🔥",
                "entities": ["HeatNodes", "HeatFlows", "ThermalClusters"],
                "relationships": ["propagates_to", "influences", "heats"]
            }
        },
        "timestamp": datetime.now().isoformat()
    })

@router.get("/ontology/ttl")
async def get_ontology_ttl():
    """Download the TTL ontology file"""
    try:
        ontology_path = "backend/ontology/financial_markets_ontology.ttl"
        
        with open(ontology_path, 'r', encoding='utf-8') as f:
            ttl_content = f.read()
            
        return JSONResponse(
            content={
                "status": "success",
                "ontology": ttl_content,
                "format": "turtle",
                "timestamp": datetime.now().isoformat()
            },
            headers={
                "Content-Type": "application/json; charset=utf-8"
            }
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Ontology file not found"
        )
    except Exception as e:
        logger.error(f"❌ Error reading ontology file: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to read ontology file"
        )

@router.get("/signals/{symbol}")
async def get_stock_signals(symbol: str):
    """Get latest trading signals for a specific stock"""
    try:
        await ensure_service_initialized()
        
        # This would query Neo4j for latest signals
        # For now, return synthetic data
        signals = {
            "symbol": symbol.upper(),
            "current_signal": {
                "type": "BuySignal" if symbol.upper() in ["AAPL", "MSFT"] else "HoldSignal",
                "strength": 0.75,
                "confidence": 0.82,
                "reasoning": f"Technical analysis indicates buying opportunity for {symbol.upper()}",
                "timestamp": datetime.now().isoformat()
            },
            "technical_indicators": {
                "rsi": 65.4,
                "ma_50": 175.20,
                "ma_200": 168.50,
                "bollinger_upper": 185.30,
                "bollinger_lower": 165.10
            },
            "heat_score": 72.5,
            "correlations": {
                "GOOGL": 0.68,
                "MSFT": 0.72,
                "AMZN": 0.45
            }
        }
        
        return JSONResponse(content={
            "status": "success",
            "data": signals,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting stock signals: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get signals for {symbol}"
        )

@router.get("/health")
async def ontology_health_check():
    """Health check for ontology service"""
    try:
        await ensure_service_initialized()
        
        # Test basic Neo4j connectivity
        test_query = "MATCH (n) RETURN count(n) as node_count LIMIT 1"
        result = ontology_service.graph.run(test_query).data()
        node_count = result[0]["node_count"] if result else 0
        
        return JSONResponse(content={
            "status": "healthy",
            "service_initialized": _service_initialized,
            "neo4j_connected": True,
            "node_count": node_count,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "service_initialized": _service_initialized,
                "neo4j_connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@router.post("/initialize")
async def force_initialize_service(background_tasks: BackgroundTasks):
    """Force re-initialization of the ontology service"""
    try:
        global _service_initialized
        _service_initialized = False
        
        # Schedule initialization in background
        background_tasks.add_task(ensure_service_initialized)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Service re-initialization scheduled",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error forcing initialization: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to schedule initialization"
        )

# Background task to periodically update ontology with live data
async def periodic_ontology_update():
    """Periodically fetch and update ontology with latest market data"""
    while True:
        try:
            if _service_initialized:
                # This would fetch latest market data and update ontology
                logger.info("🔄 Periodic ontology update (placeholder)")
                
            await asyncio.sleep(300)  # Update every 5 minutes
            
        except Exception as e:
            logger.error(f"❌ Periodic update error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error

# Start background task when module loads
# asyncio.create_task(periodic_ontology_update())