"""
Revolutionary RAGHeat Trading System - Main API Server
Integrates all advanced AI components into a unified API
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import asyncio
from datetime import datetime
import logging
import sys
import os
import uvicorn
import numpy as np
from pathlib import Path

# Add parent directories to path
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

# Import our revolutionary components
try:
    from unified_trading_system import UnifiedTradingSystem, SystemConfig, TradingMode
    from models.heat_propagation.viral_heat_engine import ViralHeatEngine
    from models.hierarchical_analysis.sector_stock_analyzer import HierarchicalSectorStockAnalyzer
    from models.machine_learning.advanced_heat_predictor import AdvancedHeatPredictor
    from models.knowledge_graph.market_ontology_engine import MarketOntologyEngine
    from models.options.advanced_option_pricing import AdvancedOptionPricingEngine
    from config.sector_stocks import SECTOR_STOCKS, PRIORITY_STOCKS
    from api.enhanced_graph_routes import router as enhanced_graph_router
    from api.ontology_routes import router as ontology_router
except ImportError as e:
    logging.error(f"Import error: {e}")
    print(f"Import error: {e}")
    enhanced_graph_router = None
    ontology_router = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAGHeat Revolutionary Trading System",
    description="AI-powered trading platform with viral heat propagation",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include enhanced graph router
if enhanced_graph_router:
    app.include_router(enhanced_graph_router)
    logger.info("✅ Enhanced graph routes included")
else:
    logger.warning("⚠️ Enhanced graph routes not available")

# Include ontology router
if ontology_router:
    app.include_router(ontology_router)
    logger.info("✅ Ontology routes included")
else:
    logger.warning("⚠️ Ontology routes not available")

# Global system instance
trading_system: Optional[UnifiedTradingSystem] = None
websocket_connections: List[WebSocket] = []

# Request/Response Models
class SystemInitRequest(BaseModel):
    trading_mode: str = "simulation"
    enable_all_features: bool = True

class SignalRequest(BaseModel):
    symbols: List[str]
    timeframe: str = "1d"

class AnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str = "comprehensive"

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                await self.disconnect(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the revolutionary trading system"""
    global trading_system
    
    logger.info("🚀 Starting Revolutionary RAGHeat Trading System...")
    
    try:
        # Create system configuration
        config = SystemConfig(
            trading_mode=TradingMode.SIMULATION,
            enable_heat_propagation=True,
            enable_hierarchical_analysis=True,
            enable_ml_predictions=True,
            enable_knowledge_graph=True,
            enable_option_pricing=True,
            target_annual_return=10.0,  # 1000% target
            max_portfolio_risk=0.02
        )
        
        # Initialize the unified trading system
        trading_system = UnifiedTradingSystem(config)
        
        # Initialize all components
        success = await trading_system.initialize()
        
        if success:
            # Start trading operations
            await trading_system.start_trading()
            logger.info("✅ Revolutionary Trading System started successfully!")
        else:
            logger.error("❌ Failed to start trading system")
            
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the trading system"""
    global trading_system
    
    if trading_system:
        await trading_system.shutdown()
        logger.info("Trading system shutdown complete")

# API Routes

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "RAGHeat Revolutionary Trading System",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Viral Heat Propagation",
            "Hierarchical Analysis", 
            "Advanced ML Predictions",
            "Knowledge Graph Reasoning",
            "Option Pricing",
            "Real-time Signals"
        ],
        "documentation": "/docs",
        "dashboard": "/dashboard"
    }

@app.get("/api/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    if not trading_system:
        return {"status": "not_initialized", "error": "Trading system not initialized"}
    
    try:
        status = trading_system.get_system_status()
        return {
            "status": "success",
            "data": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return {"status": "error", "error": str(e)}

@app.post("/api/system/initialize")
async def initialize_system(request: SystemInitRequest):
    """Initialize or reinitialize the trading system"""
    global trading_system
    
    try:
        # Create new configuration
        config = SystemConfig(
            trading_mode=TradingMode(request.trading_mode),
            enable_heat_propagation=request.enable_all_features,
            enable_hierarchical_analysis=request.enable_all_features,
            enable_ml_predictions=request.enable_all_features,
            enable_knowledge_graph=request.enable_all_features,
            enable_option_pricing=request.enable_all_features
        )
        
        # Shutdown existing system if running
        if trading_system:
            await trading_system.shutdown()
        
        # Create new system
        trading_system = UnifiedTradingSystem(config)
        success = await trading_system.initialize()
        
        if success:
            await trading_system.start_trading()
            return {"status": "success", "message": "System initialized successfully"}
        else:
            return {"status": "error", "message": "Failed to initialize system"}
            
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        return {"status": "error", "error": str(e)}

@app.get("/api/signals/recent")
async def get_recent_signals(limit: int = 20):
    """Get recent trading signals"""
    if not trading_system:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    try:
        signals = trading_system.get_recent_signals(limit)
        return {
            "status": "success",
            "signals": signals,
            "count": len(signals),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/comprehensive")
async def comprehensive_analysis(request: AnalysisRequest):
    """Perform comprehensive analysis on a symbol"""
    if not trading_system:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    try:
        # Get comprehensive analysis from the system
        symbol = request.symbol.upper()
        
        # Check if we have data for this symbol
        if symbol not in trading_system.market_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Generate unified signal (this performs comprehensive analysis)
        signal = await trading_system._analyze_symbol_comprehensive(symbol)
        
        if not signal:
            raise HTTPException(status_code=500, detail="Failed to analyze symbol")
        
        # Get additional insights
        portfolio_state = trading_system.portfolio
        risk_assessment = trading_system.risk_manager.evaluate_trade_risk(signal, portfolio_state)
        
        return {
            "status": "success",
            "symbol": symbol,
            "analysis": {
                "unified_signal": {
                    "signal_type": signal.signal_type,
                    "strength": signal.strength,
                    "confidence": signal.confidence,
                    "expected_return": signal.expected_return,
                    "position_size": signal.position_size,
                    "reasoning_factors": signal.reasoning_factors
                },
                "component_signals": {
                    "heat_propagation": signal.heat_propagation_signal,
                    "hierarchical": signal.hierarchical_signal,
                    "ml_prediction": signal.ml_prediction_signal,
                    "knowledge_graph": signal.knowledge_graph_signal,
                    "technical": signal.technical_signal
                },
                "predictions": {
                    "price_target_1d": signal.price_target_1d,
                    "price_target_5d": signal.price_target_5d,
                    "price_target_20d": signal.price_target_20d,
                    "sharpe_prediction": signal.sharpe_prediction
                },
                "risk_assessment": risk_assessment
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/heat/propagation")
async def get_heat_propagation():
    """Get current heat propagation data"""
    if not trading_system or not trading_system.heat_engine:
        raise HTTPException(status_code=503, detail="Heat propagation engine not available")
    
    try:
        # Get heat propagation results
        results = trading_system.heat_engine.get_propagation_results()
        
        # Format for frontend visualization
        nodes = []
        links = []
        
        for node_id, result in results.items():
            nodes.append({
                "id": node_id,
                "symbol": node_id,
                "heat_level": result.final_heat,
                "initial_heat": result.initial_heat,
                "propagation_efficiency": result.propagation_efficiency,
                "market_cap_billions": 100,  # Placeholder
                "sector": trading_system.market_data.get(node_id, {}).get("sector", "Unknown"),
                "prediction": result.final_heat / 10  # Normalize
            })
        
        # Create links based on propagation
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                # Simple correlation-based linking
                heat_diff = abs(node1["heat_level"] - node2["heat_level"])
                if heat_diff < 0.3:  # Similar heat levels suggest connection
                    links.append({
                        "source": node1["id"],
                        "target": node2["id"],
                        "heat_flow": min(node1["heat_level"], node2["heat_level"])
                    })
        
        return {
            "status": "success",
            "data": {
                "nodes": nodes,
                "links": links[:20]  # Limit links for performance
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting heat propagation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sectors/analysis")
async def get_sector_analysis():
    """Get hierarchical sector analysis"""
    if not trading_system:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    try:
        # Generate sector analysis data
        sector_data = []
        
        for sector_key, sector_info in SECTOR_STOCKS.items():
            # Calculate sector metrics
            sector_performance = 0.0
            sector_heat = 0.0
            stock_count = 0
            
            for stock in sector_info["all_stocks"][:5]:  # Sample 5 stocks
                if stock in trading_system.market_data:
                    df = trading_system.market_data[stock]
                    if len(df) > 20:
                        recent_return = df['Close'].pct_change(20).iloc[-1]
                        sector_performance += recent_return
                        sector_heat += abs(recent_return) * 10  # Heat calculation
                        stock_count += 1
            
            if stock_count > 0:
                avg_performance = (sector_performance / stock_count) * 100
                avg_heat = min(sector_heat / stock_count, 1.0)
            else:
                avg_performance = 0.0
                avg_heat = 0.0
            
            sector_data.append({
                "key": sector_key,
                "name": sector_info["sector_name"],
                "performance": avg_performance,
                "heat_score": avg_heat,
                "volume_change": np.random.uniform(-10, 20),  # Placeholder
                "volatility": np.random.uniform(0.1, 0.4),    # Placeholder
                "ai_prediction": 1 if avg_performance > 0 else -1,
                "top_stocks": sector_info["top_stocks"]
            })
        
        return {
            "status": "success",
            "data": sector_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting sector analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/status")
async def get_portfolio_status():
    """Get current portfolio status"""
    if not trading_system:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    try:
        portfolio = trading_system.portfolio
        performance_metrics = trading_system.performance_tracker.calculate_performance_metrics(portfolio)
        
        return {
            "status": "success",
            "portfolio": {
                "total_value": portfolio.total_value,
                "cash": portfolio.cash,
                "total_return": portfolio.total_return,
                "position_count": portfolio.position_count,
                "positions": dict(portfolio.positions),
                "sector_allocations": dict(portfolio.sector_allocations)
            },
            "performance": performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{symbol}/analysis")
async def get_stock_analysis(symbol: str):
    """Get detailed stock analysis with buy/sell recommendation"""
    
    symbol = symbol.upper()
    
    # Mock comprehensive stock analysis for demonstration
    stock_data = {
        "AAPL": {"name": "Apple Inc.", "price": 189.46, "sector": "Technology"},
        "MSFT": {"name": "Microsoft Corporation", "price": 415.26, "sector": "Technology"},
        "NVDA": {"name": "NVIDIA Corporation", "price": 456.12, "sector": "Technology"},
        "TSLA": {"name": "Tesla, Inc.", "price": 267.48, "sector": "Consumer_Discretionary"},
        "JPM": {"name": "JPMorgan Chase & Co.", "price": 158.23, "sector": "Financial"}
    }
    
    if symbol not in stock_data:
        # Generate default data for unknown stocks
        stock_data[symbol] = {
            "name": f"{symbol} Corporation",
            "price": 100 + np.random.uniform(50, 200),
            "sector": "Unknown"
        }
    
    stock = stock_data[symbol]
    
    # Generate realistic buy/sell recommendation
    buy_signals = np.random.randint(1, 4)
    sell_signals = np.random.randint(0, 3)
    
    if buy_signals > sell_signals + 1:
        action = "STRONG_BUY" if buy_signals >= 3 else "BUY"
        confidence = 0.8 + (buy_signals * 0.05)
        price_target = stock["price"] * np.random.uniform(1.15, 1.35)
    elif sell_signals > buy_signals:
        action = "SELL"
        confidence = 0.7 + (sell_signals * 0.05)
        price_target = stock["price"] * np.random.uniform(0.85, 0.95)
    else:
        action = "HOLD"
        confidence = np.random.uniform(0.6, 0.8)
        price_target = stock["price"] * np.random.uniform(0.95, 1.10)
    
    # Generate RAG explanation based on action
    if action in ["BUY", "STRONG_BUY"]:
        reasoning_path = [
            f"Technical analysis shows strong momentum for {symbol}",
            f"RSI indicates oversold conditions with potential for reversal",
            f"Moving averages suggest upward trend continuation",
            f"Fundamental metrics show {stock['name']} is undervalued relative to peers",
            f"Market sentiment and sector rotation favor {stock['sector']} stocks"
        ]
        supporting_evidence = [
            f"P/E ratio below sector median",
            f"Strong institutional ownership with recent insider buying",
            f"Earnings growth projected at 15-20% for next fiscal year",
            f"Market leadership position in key growth segments",
            f"Strong balance sheet with low debt-to-equity ratio"
        ]
        risk_factors = [
            "Market volatility could impact short-term performance",
            "Regulatory changes in technology sector",
            "Competition from emerging market players",
            "Interest rate sensitivity affecting valuation multiples"
        ]
    elif action == "SELL":
        reasoning_path = [
            f"Technical indicators signal weakness in {symbol}",
            f"RSI shows overbought conditions suggesting pullback",
            f"Breaking below key support levels",
            f"Fundamental valuation appears stretched at current levels",
            f"Sector headwinds and competitive pressures mounting"
        ]
        supporting_evidence = [
            f"P/E ratio exceeds historical averages",
            f"Recent earnings miss and guidance reduction",
            f"Declining market share in core business segments",
            f"High institutional ownership suggesting limited upside",
            f"Rising input costs pressuring profit margins"
        ]
        risk_factors = [
            "Potential for further downside if macro conditions deteriorate",
            "Execution risks related to strategic initiatives",
            "Currency headwinds for international exposure",
            "Liquidity concerns in current market environment"
        ]
    else:  # HOLD
        reasoning_path = [
            f"Mixed signals for {symbol} suggest wait-and-see approach",
            f"Technical indicators are conflicting with no clear direction",
            f"Fundamental valuation appears fairly priced",
            f"Market conditions create uncertainty for near-term performance",
            f"Better entry opportunities may emerge with patience"
        ]
        supporting_evidence = [
            f"Current valuation aligns with sector averages",
            f"Stable earnings outlook with modest growth expectations",
            f"Balanced risk-reward profile at current price levels",
            f"Strong competitive position but limited catalysts",
            f"Dividend yield provides income while waiting"
        ]
        risk_factors = [
            "Opportunity cost of holding versus other alternatives",
            "Market timing risk if conditions change rapidly",
            "Sector rotation could impact relative performance",
            "Inflation concerns affecting consumer spending"
        ]
    
    return {
        "status": "success",
        "data": {
            "basic_info": stock,
            "current_price": stock["price"],
            "recommendation": {
                "symbol": symbol,
                "action": action,
                "confidence": min(confidence, 0.95),
                "price_target": price_target,
                "timeframe": "3-6 months",
                "explanation": {
                    "reasoning_path": reasoning_path,
                    "supporting_evidence": supporting_evidence,
                    "risk_factors": risk_factors,
                    "confidence_score": min(confidence, 0.95)
                }
            },
            "last_updated": datetime.now().isoformat()
        }
    }

@app.get("/api/options/strategies")
async def get_option_strategies(symbol: str):
    """Get option strategies for a symbol"""
    if not trading_system or not trading_system.option_engine:
        raise HTTPException(status_code=503, detail="Option pricing engine not available")
    
    try:
        # Get current market data
        if symbol.upper() not in trading_system.market_data:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        df = trading_system.market_data[symbol.upper()]
        current_price = df['Close'].iloc[-1]
        volatility = df['Close'].pct_change().std() * np.sqrt(252)
        
        # Get optimized strategies
        strategies = trading_system.option_engine.strategy_optimizer.optimize_strategy(
            spot_price=current_price,
            volatility=volatility,
            time_to_expiry=30/365,  # 30 days
            market_outlook='bullish'  # Default
        )
        
        strategy_data = []
        for strategy in strategies:
            strategy_data.append({
                "name": strategy.name,
                "strategy_type": strategy.strategy_type,
                "max_profit": strategy.max_profit,
                "max_loss": strategy.max_loss,
                "probability_of_profit": strategy.probability_of_profit,
                "expected_return": strategy.expected_return,
                "risk_reward_ratio": strategy.risk_reward_ratio,
                "breakeven_points": strategy.breakeven_points
            })
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "current_price": current_price,
            "volatility": volatility,
            "strategies": strategy_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting option strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket Endpoints

@app.websocket("/ws/live-signals")
async def websocket_live_signals(websocket: WebSocket):
    """WebSocket for live trading signals"""
    await manager.connect(websocket)
    
    try:
        while True:
            if trading_system:
                # Get recent signals
                recent_signals = trading_system.get_recent_signals(5)
                system_status = trading_system.get_system_status()
                
                message = {
                    "type": "live_signals",
                    "signals": recent_signals,
                    "system_status": {
                        "total_value": system_status.get("portfolio", {}).get("total_value", 0),
                        "total_return": system_status.get("portfolio", {}).get("total_return", 0),
                        "signals_generated": system_status.get("total_signals_generated", 0),
                        "trades_executed": system_status.get("total_trades_executed", 0)
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send_json(message)
            
            await asyncio.sleep(10)  # Send updates every 10 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

@app.websocket("/ws/heat-propagation")
async def websocket_heat_propagation(websocket: WebSocket):
    """WebSocket for heat propagation updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            if trading_system and trading_system.heat_engine:
                # Get heat propagation data
                results = trading_system.heat_engine.get_propagation_results()
                
                heat_data = {
                    "type": "heat_propagation",
                    "heat_sources": len([r for r in results.values() if r.initial_heat > 0.1]),
                    "total_heat": sum(r.final_heat for r in results.values()),
                    "top_heated_stocks": [
                        {"symbol": symbol, "heat": result.final_heat}
                        for symbol, result in sorted(results.items(), 
                                                   key=lambda x: x[1].final_heat, reverse=True)[:10]
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send_json(heat_data)
            
            await asyncio.sleep(15)  # Send updates every 15 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

@app.websocket("/ws/sector-analysis")
async def websocket_sector_analysis(websocket: WebSocket):
    """WebSocket for sector analysis updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            if trading_system:
                # Generate mock sector data for now
                sector_updates = []
                for sector_key, sector_info in list(SECTOR_STOCKS.items())[:5]:
                    sector_updates.append({
                        "key": sector_key,
                        "name": sector_info["sector_name"],
                        "performance": np.random.uniform(-5, 15),
                        "heat_score": np.random.uniform(0.2, 0.9),
                        "volume_change": np.random.uniform(-20, 30),
                        "timestamp": datetime.now().isoformat()
                    })
                
                message = {
                    "type": "sector_analysis",
                    "sectors": sector_updates,
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send_json(message)
            
            await asyncio.sleep(20)  # Send updates every 20 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

# Serve static files (frontend)
frontend_path = Path(__file__).parent.parent.parent / "frontend" / "build"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")
    
    @app.get("/dashboard")
    async def serve_dashboard():
        """Serve the React dashboard"""
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return index_file.read_text()
        else:
            return {"message": "Dashboard not built. Run 'npm run build' in frontend directory."}

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": trading_system is not None,
        "components": {
            "heat_engine": trading_system.heat_engine is not None if trading_system else False,
            "ml_predictor": trading_system.ml_predictor is not None if trading_system else False,
            "ontology_engine": trading_system.ontology_engine is not None if trading_system else False,
            "option_engine": trading_system.option_engine is not None if trading_system else False
        }
    }

if __name__ == "__main__":
    print("🚀 Starting RAGHeat Revolutionary Trading System")
    print("=" * 60)
    print("🔥 Features:")
    print("   • Viral Heat Propagation")
    print("   • Hierarchical Sector Analysis")
    print("   • Advanced ML Predictions")
    print("   • Knowledge Graph Reasoning")
    print("   • Option Pricing & Strategies")
    print("   • Real-time WebSocket Streams")
    print("=" * 60)
    print("📊 API Documentation: http://localhost:8001/docs")
    print("🎯 Dashboard: http://localhost:8001/dashboard")
    print("💹 Live Signals: ws://localhost:8001/ws/live-signals")
    print("🔥 Heat Propagation: ws://localhost:8001/ws/heat-propagation")
    print("📈 Sector Analysis: ws://localhost:8001/ws/sector-analysis")
    print("=" * 60)
    
    uvicorn.run(
        "revolutionary_main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
        access_log=True
    )