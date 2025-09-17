"""
Neo4j Streaming API Routes - Test streaming data population to Neo4j
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime

from backend.services.streaming_neo4j_service import StreamingNeo4jService, get_streaming_neo4j_service
from backend.services.high_speed_streaming_service import HighSpeedStreamingService

logger = logging.getLogger(__name__)

router = APIRouter()

class MockStreamingData:
    """Mock streaming data for testing"""
    def __init__(self, symbol, price, change, change_percent, volume):
        self.symbol = symbol
        self.price = price
        self.change = change
        self.change_percent = change_percent
        self.volume = volume
        self.high = price * 1.05
        self.low = price * 0.95
        self.open_price = price * 0.98
        self.source = "docker_test"

@router.get("/neo4j/test-connection")
async def test_neo4j_connection():
    """Test Neo4j connection within Docker environment"""
    try:
        neo4j_service = get_streaming_neo4j_service()
        await neo4j_service.ensure_connected()
        
        if neo4j_service.is_connected:
            stats = await neo4j_service.get_live_graph_stats()
            return {
                "status": "success",
                "connected": True,
                "message": "Neo4j connection successful",
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "connected": False,
                "message": "Failed to connect to Neo4j",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Neo4j connection test failed: {e}")
        return {
            "status": "error",
            "connected": False,
            "message": f"Neo4j connection error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@router.post("/neo4j/populate-test-data")
async def populate_test_data():
    """Populate Neo4j with test streaming data"""
    try:
        neo4j_service = get_streaming_neo4j_service()
        await neo4j_service.ensure_connected()
        
        if not neo4j_service.is_connected:
            raise HTTPException(status_code=503, detail="Neo4j service not available")
        
        # Create test streaming data
        test_data = {
            "AAPL": MockStreamingData("AAPL", 175.25, 3.50, 2.04, 52123456),
            "MSFT": MockStreamingData("MSFT", 315.80, -2.20, -0.69, 31967834),
            "GOOGL": MockStreamingData("GOOGL", 2750.45, 25.30, 0.93, 15456789),
            "NVDA": MockStreamingData("NVDA", 445.67, 12.90, 2.98, 78890123),
            "TSLA": MockStreamingData("TSLA", 268.32, -8.68, -3.14, 99012345),
            "META": MockStreamingData("META", 325.89, 5.67, 1.77, 44567890),
            "AMZN": MockStreamingData("AMZN", 135.45, 2.12, 1.59, 66789012)
        }
        
        # Populate Neo4j
        success = await neo4j_service.populate_streaming_data(test_data)
        
        if success:
            # Get updated stats
            stats = await neo4j_service.get_live_graph_stats()
            
            return {
                "status": "success",
                "message": f"Successfully populated Neo4j with {len(test_data)} stocks",
                "stocks_processed": len(test_data),
                "stocks": list(test_data.keys()),
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to populate Neo4j with test data")
            
    except Exception as e:
        logger.error(f"Failed to populate test data: {e}")
        raise HTTPException(status_code=500, detail=f"Population error: {str(e)}")

@router.post("/neo4j/populate-live-data")
async def populate_live_data():
    """Populate Neo4j with real-time streaming data"""
    try:
        neo4j_service = get_streaming_neo4j_service()
        await neo4j_service.ensure_connected()
        
        if not neo4j_service.is_connected:
            raise HTTPException(status_code=503, detail="Neo4j service not available")
        
        # Get live streaming data
        streaming_service = HighSpeedStreamingService()
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN"]
        
        logger.info(f"Fetching live data for {len(symbols)} symbols...")
        live_data = await streaming_service.get_fastest_data(symbols)
        
        if not live_data:
            raise HTTPException(status_code=503, detail="Unable to fetch live streaming data")
        
        # Populate Neo4j with live data
        success = await neo4j_service.populate_streaming_data(live_data)
        
        if success:
            # Get updated stats
            stats = await neo4j_service.get_live_graph_stats()
            
            return {
                "status": "success",
                "message": f"Successfully populated Neo4j with live data for {len(live_data)} stocks",
                "stocks_processed": len(live_data),
                "stocks": list(live_data.keys()),
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to populate Neo4j with live data")
            
    except Exception as e:
        logger.error(f"Failed to populate live data: {e}")
        raise HTTPException(status_code=500, detail=f"Live data population error: {str(e)}")

@router.get("/neo4j/query/top-performers")
async def get_top_performers():
    """Get top performing stocks from Neo4j"""
    try:
        neo4j_service = get_streaming_neo4j_service()
        await neo4j_service.ensure_connected()
        
        if not neo4j_service.is_connected:
            raise HTTPException(status_code=503, detail="Neo4j service not available")
        
        query = """
        MATCH (s:Stock)-[:BELONGS_TO]->(sec:Sector)
        WHERE s.last_updated IS NOT NULL AND s.price IS NOT NULL
        RETURN s.symbol as symbol, s.price as price, s.change_percent as change_percent, 
               s.volume as volume, sec.name as sector, s.last_updated as last_updated
        ORDER BY s.change_percent DESC
        LIMIT 10
        """
        
        results = await neo4j_service.execute_live_query(query)
        
        return {
            "status": "success",
            "message": f"Retrieved {len(results)} top performing stocks",
            "data": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to query top performers: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@router.get("/neo4j/query/sector-performance")
async def get_sector_performance():
    """Get sector performance from Neo4j"""
    try:
        neo4j_service = get_streaming_neo4j_service()
        await neo4j_service.ensure_connected()
        
        if not neo4j_service.is_connected:
            raise HTTPException(status_code=503, detail="Neo4j service not available")
        
        query = """
        MATCH (s:Stock)-[:BELONGS_TO]->(sec:Sector)
        WHERE s.last_updated IS NOT NULL AND s.price IS NOT NULL
        WITH sec.name as sector, 
             avg(s.change_percent) as avg_change,
             count(s) as stock_count,
             sum(s.volume) as total_volume,
             max(s.last_updated) as last_updated
        RETURN sector, avg_change, stock_count, total_volume, last_updated
        ORDER BY avg_change DESC
        """
        
        results = await neo4j_service.execute_live_query(query)
        
        return {
            "status": "success",
            "message": f"Retrieved performance data for {len(results)} sectors",
            "data": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to query sector performance: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@router.get("/neo4j/query/market-sessions")
async def get_market_sessions():
    """Get recent market sessions from Neo4j"""
    try:
        neo4j_service = get_streaming_neo4j_service()
        await neo4j_service.ensure_connected()
        
        if not neo4j_service.is_connected:
            raise HTTPException(status_code=503, detail="Neo4j service not available")
        
        query = """
        MATCH (ms:MarketSession)
        RETURN ms.session_id as session_id, ms.timestamp as timestamp, 
               ms.total_stocks as total_stocks, ms.status as status
        ORDER BY ms.timestamp DESC
        LIMIT 10
        """
        
        results = await neo4j_service.execute_live_query(query)
        
        return {
            "status": "success",
            "message": f"Retrieved {len(results)} market sessions",
            "data": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to query market sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")