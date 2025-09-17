"""
Standalone Ontology API - Professional Knowledge Graph Visualization
Works without Neo4j dependency for demonstration
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import asyncio
import uvicorn
import numpy as np
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Professional Ontology Knowledge Graph API",
    description="Standalone ontology-driven knowledge graph visualization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Professional financial market data
SECTORS = ["Technology", "Healthcare", "Finance", "Energy", "Consumer_Discretionary", "Consumer", "Retail", "Entertainment"]

STOCK_DATA = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "price": 189.46, "market_cap": 2950},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "price": 415.26, "market_cap": 3080},
    "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology", "price": 456.12, "market_cap": 1120},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "price": 138.21, "market_cap": 1740},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer_Discretionary", "price": 155.89, "market_cap": 1620},
    "TSLA": {"name": "Tesla Inc.", "sector": "Consumer_Discretionary", "price": 267.48, "market_cap": 850},
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Finance", "price": 158.23, "market_cap": 465},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare", "price": 160.15, "market_cap": 420},
    "V": {"name": "Visa Inc.", "sector": "Finance", "price": 282.07, "market_cap": 575},
    "WMT": {"name": "Walmart Inc.", "sector": "Consumer", "price": 84.25, "market_cap": 685},
    "XOM": {"name": "Exxon Mobil Corporation", "sector": "Energy", "price": 118.27, "market_cap": 485},
    "PG": {"name": "Procter & Gamble Co.", "sector": "Consumer", "price": 165.89, "market_cap": 385},
    "UNH": {"name": "UnitedHealth Group Inc.", "sector": "Healthcare", "price": 521.47, "market_cap": 480},
    "HD": {"name": "Home Depot Inc.", "sector": "Retail", "price": 415.26, "market_cap": 425},
    "BAC": {"name": "Bank of America Corp.", "sector": "Finance", "price": 45.23, "market_cap": 365},
    "NFLX": {"name": "Netflix Inc.", "sector": "Entertainment", "price": 485.73, "market_cap": 215},
    "DIS": {"name": "Walt Disney Co.", "sector": "Entertainment", "price": 113.89, "market_cap": 205},
    "CVX": {"name": "Chevron Corporation", "sector": "Energy", "price": 155.84, "market_cap": 295},
    "KO": {"name": "Coca-Cola Co.", "sector": "Consumer", "price": 63.47, "market_cap": 275},
    "PFE": {"name": "Pfizer Inc.", "sector": "Healthcare", "price": 28.96, "market_cap": 165}
}

ONTOLOGY_LEVELS = {
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
}

def generate_trading_signals():
    """Generate realistic trading signals based on technical analysis"""
    signals = []
    
    for symbol, data in STOCK_DATA.items():
        # Generate technical indicators
        rsi = random.uniform(20, 80)
        ma_50 = data["price"] * random.uniform(0.95, 1.05)
        ma_200 = data["price"] * random.uniform(0.90, 1.10)
        
        # Determine signal based on indicators
        if rsi < 30 and data["price"] > ma_50:
            signal_type = "BuySignal"
            strength = random.uniform(0.7, 0.95)
            confidence = random.uniform(0.75, 0.90)
        elif rsi > 70 and data["price"] < ma_50:
            signal_type = "SellSignal"
            strength = random.uniform(0.6, 0.85)
            confidence = random.uniform(0.65, 0.80)
        else:
            signal_type = "HoldSignal"
            strength = random.uniform(0.5, 0.75)
            confidence = random.uniform(0.60, 0.75)
        
        signals.append({
            "id": f"{symbol}_{signal_type}",
            "symbol": symbol,
            "signal_type": signal_type,
            "strength": strength,
            "confidence": confidence,
            "technical_indicators": {
                "rsi": rsi,
                "ma_50": ma_50,
                "ma_200": ma_200,
                "bollinger_upper": data["price"] * 1.02,
                "bollinger_lower": data["price"] * 0.98
            }
        })
    
    return signals

def generate_heat_propagation_data():
    """Generate heat propagation network data"""
    nodes = []
    links = []
    
    for symbol, data in STOCK_DATA.items():
        heat_score = random.uniform(0.1, 1.0)
        heat_intensity = random.uniform(0.2, 0.9)
        
        nodes.append({
            "id": symbol,
            "symbol": symbol,
            "name": data["name"],
            "sector": data["sector"],
            "heat_score": heat_score,
            "heat_intensity": heat_intensity,
            "heat_capacity": random.uniform(0.3, 0.8),
            "node_type": "HeatNode",
            "market_cap": data["market_cap"],
            "price": data["price"]
        })
    
    # Create heat flow links based on sector similarity and correlation
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            if node1["sector"] == node2["sector"]:
                # Same sector = higher correlation
                heat_flow = min(node1["heat_score"], node2["heat_score"]) * random.uniform(0.6, 0.9)
                if heat_flow > 0.3:
                    links.append({
                        "source": node1["id"],
                        "target": node2["id"],
                        "heat_flow": heat_flow,
                        "relationship_type": "sector_correlation",
                        "strength": heat_flow
                    })
            elif random.random() > 0.8:  # Some cross-sector correlations
                heat_flow = min(node1["heat_score"], node2["heat_score"]) * random.uniform(0.2, 0.5)
                if heat_flow > 0.2:
                    links.append({
                        "source": node1["id"],
                        "target": node2["id"],
                        "heat_flow": heat_flow,
                        "relationship_type": "market_correlation",
                        "strength": heat_flow
                    })
    
    return nodes, links

def generate_level_data(level: int, max_nodes: int = 50):
    """Generate ontology-driven data for specific visualization level"""
    
    if level == 1:  # Market Structure
        nodes = [
            {"id": "NASDAQ", "name": "NASDAQ Exchange", "type": "Market", "level": 1, "size": 50},
            {"id": "NYSE", "name": "New York Stock Exchange", "type": "Market", "level": 1, "size": 45},
            {"id": "SP500", "name": "S&P 500 Index", "type": "Index", "level": 1, "size": 40},
            {"id": "DOWJONES", "name": "Dow Jones Industrial", "type": "Index", "level": 1, "size": 35}
        ]
        links = [
            {"source": "NYSE", "target": "SP500", "relationship": "tracks"},
            {"source": "NASDAQ", "target": "SP500", "relationship": "tracks"},
            {"source": "NYSE", "target": "DOWJONES", "relationship": "contains"}
        ]
    
    elif level == 2:  # Sector Classification
        nodes = [{"id": sector, "name": f"{sector} Sector", "type": "Sector", "level": 2, "size": 30 + random.randint(0, 20)} 
                 for sector in SECTORS]
        links = []
        for i, sector1 in enumerate(SECTORS):
            for sector2 in SECTORS[i+1:]:
                if random.random() > 0.7:  # Some sector relationships
                    links.append({
                        "source": sector1,
                        "target": sector2,
                        "relationship": "market_segment"
                    })
    
    elif level == 3:  # Financial Instruments
        nodes = []
        for symbol, data in list(STOCK_DATA.items())[:max_nodes]:
            nodes.append({
                "id": symbol,
                "name": data["name"],
                "symbol": symbol,
                "type": "Stock",
                "sector": data["sector"],
                "level": 3,
                "price": data["price"],
                "market_cap": data["market_cap"],
                "size": min(50, data["market_cap"] / 20)
            })
        
        # Create links between stocks in the same sector
        links = []
        sector_groups = {}
        for node in nodes:
            sector = node["sector"]
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(node["id"])
        
        # Create intra-sector links
        for sector, stocks in sector_groups.items():
            for i, stock1 in enumerate(stocks):
                for stock2 in stocks[i+1:]:
                    if random.random() > 0.6:  # 40% chance of connection
                        links.append({
                            "source": stock1,
                            "target": stock2,
                            "relationship": "same_sector",
                            "sector": sector
                        })
    
    elif level == 4:  # Corporate Relations
        nodes = []
        for symbol, data in list(STOCK_DATA.items())[:max_nodes]:
            nodes.append({
                "id": symbol,
                "name": data["name"],
                "type": "Corporation",
                "sector": data["sector"],
                "level": 4,
                "size": 25 + random.randint(0, 15)
            })
        
        links = []
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if node1["sector"] == node2["sector"] and random.random() > 0.6:
                    correlation = random.uniform(0.3, 0.9)
                    links.append({
                        "source": node1["id"],
                        "target": node2["id"],
                        "relationship": "correlates_with",
                        "correlation": correlation,
                        "strength": correlation
                    })
    
    elif level == 5:  # Trading Signals
        signals = generate_trading_signals()
        nodes = []
        links = []
        
        # Create both signal nodes and stock nodes
        stock_symbols = set()
        for signal in signals[:max_nodes]:
            stock_symbols.add(signal["symbol"])
        
        # Add stock nodes
        for symbol in stock_symbols:
            if symbol in STOCK_DATA:
                data = STOCK_DATA[symbol]
                nodes.append({
                    "id": symbol,
                    "name": data["name"],
                    "type": "Stock",
                    "level": 5,
                    "size": 15
                })
        
        # Add signal nodes
        for signal in signals[:max_nodes]:
            signal_node = {
                "id": signal["id"],
                "name": f"{signal['symbol']} {signal['signal_type']}",
                "type": signal["signal_type"],
                "symbol": signal["symbol"],
                "level": 5,
                "strength": signal["strength"],
                "confidence": signal["confidence"],
                "size": 20 + (signal["strength"] * 20)
            }
            nodes.append(signal_node)
            
            # Link signal to stock (both nodes exist now)
            links.append({
                "source": signal["symbol"],
                "target": signal["id"],
                "relationship": "generates_signal"
            })
    
    elif level == 6:  # Technical Indicators
        nodes = []
        links = []
        
        # Create indicator nodes
        indicators = ["RSI", "MA_50", "MA_200", "MACD", "BollingerBands"]
        for indicator in indicators:
            nodes.append({
                "id": indicator,
                "name": indicator.replace("_", " "),
                "type": "TechnicalIndicator",
                "level": 6,
                "size": 25 + random.randint(0, 15)
            })
        
        # Add stock nodes that will be linked
        selected_stocks = list(STOCK_DATA.keys())[:max_nodes//2]
        for symbol in selected_stocks:
            data = STOCK_DATA[symbol]
            nodes.append({
                "id": symbol,
                "name": data["name"],
                "type": "Stock",
                "level": 6,
                "size": 12
            })
        
        # Link indicators to stocks (both exist in nodes now)
        for symbol in selected_stocks:
            for indicator in indicators[:3]:  # Limit connections
                if random.random() > 0.5:
                    links.append({
                        "source": indicator,
                        "target": symbol,
                        "relationship": "calculates_for",
                        "value": random.uniform(0.2, 0.8)
                    })
    
    elif level == 7:  # Heat Propagation
        nodes, links = generate_heat_propagation_data()
        nodes = nodes[:max_nodes]
        # Limit links for performance
        links = [link for link in links if link["heat_flow"] > 0.4][:50]
    
    else:
        nodes, links = [], []
    
    return {"nodes": nodes, "links": links}

@app.get("/")
async def root():
    return {
        "name": "Professional Ontology Knowledge Graph API",
        "version": "1.0.0",
        "status": "running",
        "ontology_levels": 7,
        "total_entities": len(STOCK_DATA),
        "documentation": "/docs"
    }

@app.get("/api/ontology/levels")
async def get_visualization_levels():
    """Get all available visualization levels with descriptions"""
    return JSONResponse(content={
        "status": "success",
        "levels": ONTOLOGY_LEVELS,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/api/ontology/graph/{level}")
async def get_hierarchical_graph(
    level: int,
    max_nodes: Optional[int] = 50
):
    """Get hierarchical knowledge graph data for specific level"""
    try:
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
        graph_data = generate_level_data(level, max_nodes)
        
        # Add metadata
        graph_data["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "level_name": ONTOLOGY_LEVELS[level]["name"],
            "level_description": ONTOLOGY_LEVELS[level]["description"],
            "max_nodes": max_nodes,
            "node_count": len(graph_data.get("nodes", [])),
            "link_count": len(graph_data.get("links", [])),
            "service_mode": "standalone"
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

@app.get("/api/ontology/market-overview")
async def get_market_overview():
    """Get comprehensive market overview with trading signals"""
    try:
        signals = generate_trading_signals()
        
        buy_signals = [s for s in signals if s["signal_type"] == "BuySignal"]
        sell_signals = [s for s in signals if s["signal_type"] == "SellSignal"] 
        hold_signals = [s for s in signals if s["signal_type"] == "HoldSignal"]
        
        # Calculate heat scores
        heat_nodes, _ = generate_heat_propagation_data()
        avg_heat = np.mean([node["heat_score"] for node in heat_nodes]) * 100
        max_heat = np.max([node["heat_score"] for node in heat_nodes]) * 100
        min_heat = np.min([node["heat_score"] for node in heat_nodes]) * 100
        
        # Market sentiment based on signals
        if len(buy_signals) > len(sell_signals):
            if len(buy_signals) > len(sell_signals) * 1.5:
                sentiment = "Bullish"
            else:
                sentiment = "Cautiously Bullish"
        elif len(sell_signals) > len(buy_signals):
            if len(sell_signals) > len(buy_signals) * 1.5:
                sentiment = "Bearish"
            else:
                sentiment = "Cautiously Bearish"
        else:
            sentiment = "Neutral"
        
        overview = {
            "total_stocks": len(STOCK_DATA),
            "total_signals": len(signals),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "hold_signals": len(hold_signals),
            "average_heat": round(avg_heat, 1),
            "max_heat": round(max_heat, 1),
            "min_heat": round(min_heat, 1),
            "market_sentiment": sentiment,
            "top_buy_signals": [
                {"symbol": s["symbol"], "strength": s["strength"], "confidence": s["confidence"]}
                for s in sorted(buy_signals, key=lambda x: x["strength"], reverse=True)[:5]
            ],
            "top_sell_signals": [
                {"symbol": s["symbol"], "strength": s["strength"], "confidence": s["confidence"]}
                for s in sorted(sell_signals, key=lambda x: x["strength"], reverse=True)[:5]
            ]
        }
        
        return JSONResponse(content={
            "status": "success",
            "data": overview,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting market overview: {e}")
        return JSONResponse(content={
            "status": "fallback",
            "data": {
                "total_stocks": len(STOCK_DATA),
                "total_signals": 15,
                "buy_signals": 6,
                "sell_signals": 4,
                "hold_signals": 5,
                "average_heat": 55.7,
                "max_heat": 92.3,
                "min_heat": 18.4,
                "market_sentiment": "Cautiously Bullish"
            },
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        })

@app.get("/api/ontology/signals/{symbol}")
async def get_stock_signals(symbol: str):
    """Get latest trading signals for a specific stock"""
    try:
        symbol = symbol.upper()
        
        if symbol not in STOCK_DATA:
            raise HTTPException(
                status_code=404,
                detail=f"Stock {symbol} not found"
            )
        
        stock_data = STOCK_DATA[symbol]
        signals = generate_trading_signals()
        stock_signals = [s for s in signals if s["symbol"] == symbol]
        
        if not stock_signals:
            raise HTTPException(
                status_code=404,
                detail=f"No signals found for {symbol}"
            )
        
        signal = stock_signals[0]
        
        # Generate correlations with other stocks in same sector
        correlations = {}
        for other_symbol, other_data in STOCK_DATA.items():
            if other_symbol != symbol and other_data["sector"] == stock_data["sector"]:
                correlations[other_symbol] = random.uniform(0.3, 0.8)
        
        return JSONResponse(content={
            "status": "success",
            "data": {
                "symbol": symbol,
                "current_signal": {
                    "type": signal["signal_type"],
                    "strength": signal["strength"],
                    "confidence": signal["confidence"],
                    "reasoning": f"Technical analysis indicates {signal['signal_type'].lower()} opportunity for {symbol}",
                    "timestamp": datetime.now().isoformat()
                },
                "technical_indicators": signal["technical_indicators"],
                "heat_score": random.uniform(40, 90),
                "correlations": dict(list(correlations.items())[:3]),
                "stock_info": stock_data
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting stock signals: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get signals for {symbol}"
        )

@app.get("/api/ontology/health")
async def ontology_health_check():
    """Health check for ontology service"""
    try:
        return JSONResponse(content={
            "status": "healthy",
            "service_mode": "standalone",
            "neo4j_connected": False,
            "ontology_levels": 7,
            "total_stocks": len(STOCK_DATA),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service_mode": "standalone", 
                "neo4j_connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/ontology/ttl")
async def get_ontology_ttl():
    """Download the TTL ontology file"""
    try:
        ontology_path = "ontology/financial_markets_ontology.ttl"
        
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

if __name__ == "__main__":
    print("🧠 Starting Professional Ontology Knowledge Graph API")
    print("=" * 60)
    print("🔬 Features:")
    print("   • 7-Level Hierarchical Ontology")
    print("   • Professional KG Visualization")
    print("   • Trading Signal Generation")
    print("   • Heat Propagation Analysis")
    print("   • Technical Indicator Integration")
    print("   • Sector-based Analysis")
    print("=" * 60)
    print("📊 API Documentation: http://localhost:8001/docs")
    print("🧠 Ontology Levels: http://localhost:8001/api/ontology/levels")
    print("📈 Market Overview: http://localhost:8001/api/ontology/market-overview")
    print("🔥 Graph Level 7: http://localhost:8001/api/ontology/graph/7")
    print("=" * 60)
    
    uvicorn.run(
        "standalone_ontology_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )