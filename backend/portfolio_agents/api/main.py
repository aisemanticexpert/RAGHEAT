"""
Main FastAPI Application for RAGHeat Portfolio System
====================================================

Production-ready API server for the multi-agent portfolio construction system.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import uvicorn

from .routes.portfolio_routes import router as portfolio_router
from .routes.agent_routes import router as agent_router  
from .routes.workflow_routes import router as workflow_router
from .routes.analysis_routes import router as analysis_router
from .routes.system_routes import router as system_router
from ..core.portfolio_system import RAGHeatPortfolioSystem
from ..config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global system instance
portfolio_system: RAGHeatPortfolioSystem = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    global portfolio_system
    logger.info("Starting RAGHeat Portfolio API...")
    
    try:
        portfolio_system = RAGHeatPortfolioSystem()
        app.state.portfolio_system = portfolio_system
        logger.info("Portfolio system initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing portfolio system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAGHeat Portfolio API...")

# Create FastAPI application
app = FastAPI(
    title="RAGHeat Portfolio Construction API",
    description="""
    Advanced Multi-Agent Portfolio Construction System using Heat Diffusion and AI Consensus
    
    ## Features
    
    * **Multi-Agent Analysis**: Specialized agents for fundamental, sentiment, technical, and network analysis
    * **Heat Diffusion Modeling**: Physics-inspired influence propagation on financial networks
    * **Consensus Building**: Structured debate and weighted voting among AI agents
    * **Portfolio Optimization**: Modern portfolio theory with advanced risk management
    * **Explainable AI**: Transparent decision-making with chain-of-thought reasoning
    
    ## Architecture
    
    The system employs a multi-agent architecture where specialized AI agents analyze different aspects 
    of the financial markets and collaborate to construct optimal portfolios. Each agent has access to 
    domain-specific tools and data sources, and they engage in structured debates to reach consensus.
    
    ## Usage
    
    1. **System Status**: Check `/system/status` for system health
    2. **Portfolio Construction**: POST to `/portfolio/construct` with stock list
    3. **Individual Analysis**: GET `/analysis/{agent_type}` for specific agent insights
    4. **Workflow Management**: Use `/workflow/*` endpoints to manage execution flows
    
    ## Support
    
    For technical support and documentation, visit the project repository.
    """,
    version="1.0.0",
    contact={
        "name": "RAGHeat Team",
        "email": "support@ragheat.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(portfolio_router, prefix="/portfolio", tags=["Portfolio Construction"])
app.include_router(agent_router, prefix="/agents", tags=["Agent Management"]) 
app.include_router(workflow_router, prefix="/workflow", tags=["Workflow Management"])
app.include_router(analysis_router, prefix="/analysis", tags=["Analysis Services"])
app.include_router(system_router, prefix="/system", tags=["System Management"])

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system information."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAGHeat Portfolio Construction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .header { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }
            .status { background-color: #e8f5e8; padding: 15px; border-radius: 4px; margin: 20px 0; }
            .endpoints { background-color: #f8f9fa; padding: 15px; border-radius: 4px; }
            .endpoint { margin: 5px 0; font-family: monospace; }
            a { color: #007acc; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="header">🚀 RAGHeat Portfolio Construction API</h1>
            
            <div class="status">
                <h3>✅ System Status: ONLINE</h3>
                <p>Multi-agent portfolio construction system is ready for requests.</p>
                <p><strong>Version:</strong> 1.0.0</p>
                <p><strong>Timestamp:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC') + """</p>
            </div>
            
            <h3>📚 API Documentation</h3>
            <p>Interactive API documentation is available at:</p>
            <div class="endpoints">
                <div class="endpoint">📖 <a href="/docs">Swagger UI Documentation</a></div>
                <div class="endpoint">📋 <a href="/redoc">ReDoc Documentation</a></div>
                <div class="endpoint">⚙️ <a href="/system/status">System Status</a></div>
            </div>
            
            <h3>🔥 Key Features</h3>
            <ul>
                <li><strong>Multi-Agent Analysis</strong>: Fundamental, sentiment, technical, and network analysis</li>
                <li><strong>Heat Diffusion Modeling</strong>: Physics-inspired influence propagation</li>
                <li><strong>AI Consensus Building</strong>: Structured debate and weighted voting</li>
                <li><strong>Portfolio Optimization</strong>: Advanced risk-adjusted portfolio construction</li>
                <li><strong>Explainable Decisions</strong>: Transparent AI reasoning and explanations</li>
            </ul>
            
            <h3>🚀 Quick Start</h3>
            <p>Construct a portfolio by sending a POST request to <code>/portfolio/construct</code> with your stock list.</p>
            
            <p style="margin-top: 30px; color: #666; border-top: 1px solid #eee; padding-top: 15px;">
                Powered by RAGHeat Multi-Agent Portfolio Construction System
            </p>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if hasattr(app.state, 'portfolio_system'):
            system_health = app.state.portfolio_system.health_check()
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system_health": system_health,
                "version": "1.0.0"
            }
        else:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": "Portfolio system not initialized"
            }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy", 
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="RAGHeat Portfolio Construction API",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# Dependency to get portfolio system
def get_portfolio_system() -> RAGHeatPortfolioSystem:
    """Dependency to get the portfolio system instance."""
    if hasattr(app.state, 'portfolio_system'):
        return app.state.portfolio_system
    else:
        raise HTTPException(
            status_code=500,
            detail="Portfolio system not initialized"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )