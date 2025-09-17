"""
Real-Time API Routes
Serves real market data from Redis storage (no mock data)
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.redis_json_storage import RedisJSONStorage

router = APIRouter(prefix="/api/realtime", tags=["Real-Time Data"])

# Initialize Redis storage
redis_storage = RedisJSONStorage()

@router.on_event("startup")
async def startup_event():
    """Connect to Redis on startup"""
    if not redis_storage.connect():
        print("⚠️  Warning: Could not connect to Redis. Real-time endpoints may not work.")

@router.get("/market-overview")
async def get_realtime_market_overview():
    """Get latest real-time market overview data (no mock data)"""
    try:
        data = redis_storage.get_latest_data('market_overview')
        if not data:
            raise HTTPException(
                status_code=404, 
                detail="No real-time market overview data available. Check if Kafka producer is running."
            )
        
        return {
            "status": "success",
            "message": "Real-time market overview data",
            "data": data.get('data', {}),
            "stored_at": data.get('stored_at'),
            "timeframe": data.get('timeframe', '5s'),
            "source": "kafka_redis_pipeline"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving real-time data: {str(e)}")

@router.get("/heat-distribution")
async def get_realtime_heat_distribution():
    """Get latest real-time heat distribution data (no mock data)"""
    try:
        data = redis_storage.get_latest_data('heat_distribution')
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No real-time heat distribution data available. Check if Kafka producer is running."
            )
        
        return {
            "status": "success", 
            "message": "Real-time heat distribution data",
            "data": data.get('data', {}),
            "stored_at": data.get('stored_at'),
            "timeframe": data.get('timeframe', '5s'),
            "source": "kafka_redis_pipeline"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving heat data: {str(e)}")

@router.get("/stock-performance")
async def get_realtime_stock_performance():
    """Get latest real-time stock performance data (no mock data)"""
    try:
        data = redis_storage.get_latest_data('stock_performance')
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No real-time stock performance data available. Check if Kafka producer is running."
            )
        
        return {
            "status": "success",
            "message": "Real-time stock performance data", 
            "data": data.get('data', {}),
            "stored_at": data.get('stored_at'),
            "timeframe": data.get('timeframe', '5s'),
            "source": "kafka_redis_pipeline"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stock performance: {str(e)}")

@router.get("/sector-performance")
async def get_realtime_sector_performance():
    """Get latest real-time sector performance data (no mock data)"""
    try:
        data = redis_storage.get_latest_data('sector_performance')
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No real-time sector performance data available. Check if Kafka producer is running."
            )
        
        return {
            "status": "success",
            "message": "Real-time sector performance data",
            "data": data.get('data', {}), 
            "stored_at": data.get('stored_at'),
            "timeframe": data.get('timeframe', '5s'),
            "source": "kafka_redis_pipeline"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sector performance: {str(e)}")

@router.get("/recent-data/{data_type}")
async def get_recent_data(data_type: str, minutes: int = 5):
    """Get recent data for the last N minutes"""
    try:
        valid_types = ['market_overview', 'heat_distribution', 'stock_performance', 'sector_performance']
        if data_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data type. Must be one of: {valid_types}"
            )
        
        data = redis_storage.get_recent_data(data_type, minutes)
        
        return {
            "status": "success",
            "message": f"Recent {data_type} data for last {minutes} minutes",
            "data": data,
            "count": len(data),
            "timeframe": "5s",
            "source": "kafka_redis_pipeline"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving recent data: {str(e)}")

@router.get("/dashboard-data")
async def get_dashboard_data():
    """Get all latest data needed for dashboard (no mock data)"""
    try:
        dashboard_data = {}
        
        # Get latest market overview
        market_overview = redis_storage.get_latest_data('market_overview')
        if market_overview:
            dashboard_data['market_overview'] = market_overview.get('data', {})
        
        # Get latest heat distribution
        heat_data = redis_storage.get_latest_data('heat_distribution')
        if heat_data:
            dashboard_data['heat_distribution'] = heat_data.get('data', {})
        
        # Get latest stock performance
        stock_performance = redis_storage.get_latest_data('stock_performance')
        if stock_performance:
            dashboard_data['stock_performance'] = stock_performance.get('data', {})
            
        # Get latest sector performance
        sector_performance = redis_storage.get_latest_data('sector_performance')
        if sector_performance:
            dashboard_data['sector_performance'] = sector_performance.get('data', {})
        
        if not dashboard_data:
            raise HTTPException(
                status_code=404,
                detail="No real-time data available. Check if Kafka producer and consumer are running."
            )
        
        return {
            "status": "success",
            "message": "Complete dashboard data from real-time pipeline",
            "data": dashboard_data,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "kafka_redis_pipeline",
            "data_available": list(dashboard_data.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dashboard data: {str(e)}")

@router.get("/storage-stats")
async def get_storage_stats():
    """Get Redis storage statistics"""
    try:
        stats = redis_storage.get_storage_stats()
        return {
            "status": "success",
            "message": "Redis storage statistics",
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving storage stats: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for real-time data pipeline"""
    try:
        # Check Redis connection
        redis_connected = redis_storage.redis_client is not None
        
        # Check if we have recent data
        latest_market = redis_storage.get_latest_data('market_overview')
        latest_heat = redis_storage.get_latest_data('heat_distribution')
        
        data_freshness = "stale"
        if latest_market or latest_heat:
            data_freshness = "fresh"
        
        return {
            "status": "success",
            "redis_connected": redis_connected,
            "data_freshness": data_freshness,
            "has_market_data": latest_market is not None,
            "has_heat_data": latest_heat is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }