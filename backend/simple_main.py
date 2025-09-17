"""
Simple RAGHeat API - Minimal implementation with enhanced graph support
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import pytz

# Import routers
from api.simple_enhanced_graph import router as enhanced_graph_router
from api.graph_routes import router as graph_router, stock_router
# from api.professional_streaming_routes import router as professional_router  # Needs dependency fix

app = FastAPI(
    title="RAGHeat Simple API",
    description="Simplified RAGHeat API with Enhanced Graph Visualization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(enhanced_graph_router)
app.include_router(graph_router)
app.include_router(stock_router)
# app.include_router(professional_router)  # Needs dependency fix

@app.get("/")
async def root():
    return {"message": "RAGHeat Simple API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": True,
        "components": {
            "enhanced_graph": True,
            "api": True
        }
    }

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.now(ny_tz)
    
    return {
        "status": "active",
        "time": now.isoformat(),
        "market_hours": "9:30 AM - 4:00 PM ET",
        "features": [
            "Enhanced Graph Visualization",
            "Real-time Heat Analysis", 
            "Market Status Detection",
            "Interactive Node Exploration"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)