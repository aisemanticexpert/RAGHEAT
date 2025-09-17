from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from options_routes import router as options_router, monitor_options_signals
except ImportError:
    options_router = None
    monitor_options_signals = None
    print("Options routes not available - some dependencies may be missing")

try:
    from sector_routes import router as sector_router
except ImportError:
    sector_router = None
    print("Sector routes not available - some dependencies may be missing")

try:
    from heat_routes import router as heat_router
except ImportError:
    heat_router = None
    print("Heat routes not available - some dependencies may be missing")

try:
    from advanced_analysis_routes import router as advanced_router
except ImportError:
    advanced_router = None
    print("Advanced analysis routes not available - some dependencies may be missing")

try:
    from professional_live_signal_routes import router as live_signal_router
    print("Using PROFESSIONAL live signal routes with Finnhub + Alpha Vantage APIs")
except ImportError:
    try:
        from hybrid_live_signal_routes import router as live_signal_router
        print("Using HYBRID live signal routes with real market data + realistic fallback")
    except ImportError:
        try:
            from real_live_signal_routes import router as live_signal_router
            print("Using REAL live signal routes with actual market data")
        except ImportError:
            try:
                from live_signal_routes import router as live_signal_router
            except ImportError:
                try:
                    from minimal_live_signal_routes import router as live_signal_router
                    print("Using minimal live signal routes for testing")
                except ImportError:
                    live_signal_router = None
                    print("Live signal routes not available - some dependencies may be missing")

try:
    from live_data_routes import router as live_data_router
except ImportError:
    live_data_router = None
    print("Live data routes not available - some dependencies may be missing")

try:
    from streaming_routes import router as streaming_router
    print("High-speed streaming routes loaded with WebSocket support")
except ImportError:
    streaming_router = None
    print("Streaming routes not available - some dependencies may be missing")

try:
    from neo4j_streaming_routes import router as neo4j_streaming_router
    print("Neo4j streaming routes loaded for live data integration")
except ImportError:
    neo4j_streaming_router = None

try:
    from realtime_routes import router as realtime_router
    print("Real-time Redis pipeline routes loaded")
except ImportError:
    realtime_router = None
    print("Real-time routes not available - some dependencies may be missing")
    print("Neo4j streaming routes not available - some dependencies may be missing")

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

# Include routers if available
if options_router:
    app.include_router(options_router)

if sector_router:
    app.include_router(sector_router)

if heat_router:
    app.include_router(heat_router)

if advanced_router:
    app.include_router(advanced_router)

if live_signal_router:
    app.include_router(live_signal_router)

if live_data_router:
    app.include_router(live_data_router)

if streaming_router:
    app.include_router(streaming_router)

if neo4j_streaming_router:
    app.include_router(neo4j_streaming_router, prefix="/api/streaming")

if realtime_router:
    app.include_router(realtime_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "RAGHeat API",
        "version": "1.0.0",
        "status": "running",
        "message": "Docker setup successful! Backend is running.",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ragheat-backend"}

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    if monitor_options_signals:
        asyncio.create_task(monitor_options_signals())
        logger.info("Options monitoring started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)