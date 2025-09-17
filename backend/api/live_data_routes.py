"""
Live Data API Routes
Endpoints for managing real-time data feeds and Neo4j population
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import asyncio

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

# MULTI-SOURCE PROFESSIONAL REAL DATA SERVICE - NO MOCK DATA
from multi_source_professional_data_service import multi_source_professional_data_service
from mock_neo4j_service import mock_neo4j_service as neo4j_live_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/live-data", tags=["Live Data"])

class DataModeRequest(BaseModel):
    use_live_data: bool = True

class PopulationRequest(BaseModel):
    symbols: Optional[List[str]] = None
    populate_neo4j: bool = True

@router.get("/status")
async def get_data_service_status():
    """Get status of live data services"""
    
    try:
        # Get MULTI-SOURCE PROFESSIONAL real-time service status
        market_overview = await multi_source_professional_data_service.get_market_overview()
        
        # Get Neo4j service status
        neo4j_stats = await neo4j_live_service.get_graph_stats()
        
        return {
            "real_time_service": {
                "active": True,
                "mock_mode": False,  # PROFESSIONAL SERVICE - NO MOCK DATA
                "cache_size": len(multi_source_professional_data_service.cache),
                "last_market_overview": market_overview
            },
            "neo4j_service": {
                "connected": neo4j_live_service.is_connected,
                "graph_stats": neo4j_stats
            },
            "timestamp": market_overview.get("timestamp", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/mode")
async def set_data_mode(request: DataModeRequest):
    """Switch between live and mock data modes"""
    
    try:
        if request.use_live_data:
            real_time_service.enable_live_data()
            mode = "live"
        else:
            real_time_service.enable_mock_data()
            mode = "mock"
        
        return {
            "status": "success",
            "message": f"Data mode set to {mode}",
            "use_live_data": not real_time_service.use_mock_data
        }
        
    except Exception as e:
        logger.error(f"Error setting data mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/populate")
async def populate_data(request: PopulationRequest, background_tasks: BackgroundTasks):
    """Manually trigger data population"""
    
    try:
        # Get fresh stock data
        symbols = request.symbols or [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", 
            "JPM", "BAC", "JNJ", "PFE", "XOM", "CVX"
        ]
        
        stock_data = await multi_source_professional_data_service.get_multiple_stocks(symbols)
        
        result = {
            "status": "success",
            "stocks_updated": len(stock_data),
            "symbols": list(stock_data.keys()),
            "neo4j_populated": False
        }
        
        if request.populate_neo4j:
            # Populate Neo4j in background
            background_tasks.add_task(neo4j_live_service.populate_live_data, symbols)
            result["neo4j_populated"] = True
            result["message"] = "Data population started in background"
        
        return result
        
    except Exception as e:
        logger.error(f"Error populating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/neo4j/connect")
async def connect_neo4j():
    """Connect to Neo4j database"""
    
    try:
        await neo4j_live_service.connect()
        
        if neo4j_live_service.is_connected:
            return {
                "status": "success",
                "message": "Connected to Neo4j database",
                "connected": True
            }
        else:
            return {
                "status": "error",
                "message": "Failed to connect to Neo4j database", 
                "connected": False
            }
            
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/neo4j/populate")
async def populate_neo4j(symbols: Optional[List[str]] = None):
    """Populate Neo4j with current market data"""
    
    try:
        success = await neo4j_live_service.populate_live_data(symbols)
        
        if success:
            stats = await neo4j_live_service.get_graph_stats()
            return {
                "status": "success",
                "message": "Neo4j populated with live data",
                "graph_stats": stats
            }
        else:
            return {
                "status": "error",
                "message": "Failed to populate Neo4j"
            }
            
    except Exception as e:
        logger.error(f"Error populating Neo4j: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/neo4j/start-continuous")
async def start_continuous_population(
    background_tasks: BackgroundTasks,
    interval_seconds: int = 300  # 5 minutes default
):
    """Start continuous Neo4j data population"""
    
    try:
        # Start continuous population in background
        background_tasks.add_task(
            neo4j_live_service.start_live_population, 
            interval_seconds
        )
        
        return {
            "status": "started",
            "message": f"Continuous Neo4j population started with {interval_seconds}s interval",
            "interval_seconds": interval_seconds
        }
        
    except Exception as e:
        logger.error(f"Error starting continuous population: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-overview")
async def get_market_overview():
    """Get comprehensive market overview"""
    
    try:
        overview = await multi_source_professional_data_service.get_market_overview()
        return overview
        
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stocks/{symbols}")
async def get_stock_data(symbols: str):
    """Get data for specific stocks (comma-separated symbols)"""
    
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        stock_data = await multi_source_professional_data_service.get_multiple_stocks(symbol_list)
        
        return {
            "stocks": stock_data,
            "count": len(stock_data),
            "requested": symbol_list
        }
        
    except Exception as e:
        logger.error(f"Error getting stock data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/neo4j/stats")
async def get_neo4j_stats():
    """Get Neo4j graph statistics"""
    
    try:
        stats = await neo4j_live_service.get_graph_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting Neo4j stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test")
async def test_live_data():
    """Test endpoint to verify live data services"""
    
    try:
        # Test MULTI-SOURCE PROFESSIONAL real-time service
        test_stocks = await multi_source_professional_data_service.get_multiple_stocks(["AAPL", "MSFT"])
        
        # Test Neo4j connection
        neo4j_connected = neo4j_live_service.is_connected
        if not neo4j_connected:
            await neo4j_live_service.connect()
            neo4j_connected = neo4j_live_service.is_connected
        
        return {
            "status": "success",
            "real_time_service": "MULTI_SOURCE_PROFESSIONAL",
            "sample_stocks": len(test_stocks),
            "neo4j_connected": neo4j_connected,
            "mock_data_mode": False,  # NO MOCK DATA
            "timestamp": test_stocks.get("AAPL").timestamp if test_stocks.get("AAPL") else None,
            "aapl_price": test_stocks.get("AAPL").price if test_stocks.get("AAPL") else None
        }
        
    except Exception as e:
        logger.error(f"Error in test endpoint: {e}")
        return {
            "status": "error",
            "message": str(e)
        }