"""
Heat map distribution API routes for market analysis
Provides heat map data for visualization components
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
import logging
import random
import math

import sys
import os

# Add paths for imports
sys.path.append('/app')
sys.path.append('/app/backend')

try:
    from backend.services.multi_sector_service import MultiSectorService
    from backend.config.sector_stocks import SECTOR_STOCKS
except ImportError:
    # Try relative import
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from services.multi_sector_service import MultiSectorService
    from config.sector_stocks import SECTOR_STOCKS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/heat", tags=["Heat Map Analysis"])

def generate_trading_signal(stock_data) -> tuple[str, float]:
    """
    Generate buy/sell signals using efficient heat equation analysis
    Returns: (signal, confidence) where signal is "buy"/"sell"/"hold" and confidence is 0-1
    """
    # Heat equation factors
    heat_score = stock_data.heat_score
    change_percent = stock_data.change_percent
    volatility = stock_data.volatility
    volume_factor = min(stock_data.volume / 50000000, 2.0)  # Normalize volume
    
    # Multi-factor analysis
    momentum_score = change_percent * 0.4  # Price momentum weight
    heat_factor = heat_score * 0.3         # Heat intensity weight  
    volume_signal = (volume_factor - 1) * 0.2  # Volume above average
    volatility_penalty = volatility * -0.1     # High volatility reduces confidence
    
    # Combined signal strength
    signal_strength = momentum_score + heat_factor + volume_signal + volatility_penalty
    
    # Generate signal based on thresholds
    if signal_strength > 1.5:
        signal = "buy"
        confidence = min(0.95, 0.5 + (signal_strength - 1.5) * 0.3)
    elif signal_strength < -1.0:
        signal = "sell"  
        confidence = min(0.95, 0.5 + abs(signal_strength + 1.0) * 0.3)
    else:
        signal = "hold"
        confidence = 0.3 + abs(signal_strength) * 0.2
    
    return signal, round(confidence, 3)

class HeatMapCell(BaseModel):
    symbol: str
    sector: str
    value: float  # Heat intensity
    change_percent: float
    volume: int
    market_cap: float
    x: int  # Grid position
    y: int  # Grid position
    color: str
    size: float
    signal: str  # "buy", "sell", "hold"
    confidence: float  # Signal confidence 0-1

class HeatDistributionResponse(BaseModel):
    cells: List[HeatMapCell]
    sectors: List[str]
    timestamp: str
    market_overview: Dict[str, float]
    data_status: str  # "live", "cached", "sample"
    market_open: bool

@router.get("/distribution", response_model=HeatDistributionResponse)
async def get_heat_distribution():
    """Get heat map distribution data for all sectors"""
    try:
        async with MultiSectorService() as service:
            all_data = await service.get_all_sector_data()
        
        # Log data retrieval status
        logger.info(f"Retrieved sector data: {len(all_data)} sectors")
        for sector, stocks in all_data.items():
            logger.info(f"  {sector}: {len(stocks)} stocks")
        
        # Force real data - only fallback if absolutely no data
        if not all_data:
            logger.warning("No market data retrieved, falling back to sample data")
            return await get_sample_heat_distribution()
        
        # Create heat map cells
        cells = []
        sectors = []
        x_pos = 0
        y_pos = 0
        grid_cols = 8  # 8 stocks per row
        
        total_change = 0
        total_volume = 0
        stock_count = 0
        
        for sector_name, stocks in all_data.items():
            if sector_name not in sectors:
                sectors.append(sector_name)
            
            # Get top 6 stocks from each sector for heat map
            top_stocks = sorted(stocks, key=lambda x: x.heat_score, reverse=True)[:6]
            
            for stock in top_stocks:
                # Calculate heat intensity (0-100)
                heat_value = min(100, max(0, stock.heat_score * 100))
                
                # Generate buy/sell signals based on heat equation
                signal, confidence = generate_trading_signal(stock)
                
                # Determine color based on trading signal and change percentage
                if signal == "buy" and confidence > 0.7:
                    color = '#00ff88'  # Strong buy - bright green
                elif signal == "buy" and confidence > 0.5:
                    color = '#88ff44'  # Moderate buy - light green  
                elif signal == "sell" and confidence > 0.7:
                    color = '#ff3366'  # Strong sell - red
                elif signal == "sell" and confidence > 0.5:
                    color = '#ff6b35'  # Moderate sell - orange-red
                else:
                    color = '#ffaa00'  # Hold/neutral - orange
                
                # Calculate size based on volume, market cap, and confidence
                base_size = (stock.volume / 10000000) + (stock.market_cap / 1000000000)
                size = min(1.5, max(0.3, base_size * (1 + confidence * 0.5)))
                
                cells.append(HeatMapCell(
                    symbol=stock.symbol,
                    sector=sector_name,
                    value=heat_value,
                    change_percent=stock.change_percent,
                    volume=stock.volume,
                    market_cap=stock.market_cap,
                    x=x_pos,
                    y=y_pos,
                    color=color,
                    size=size,
                    signal=signal,
                    confidence=confidence
                ))
                
                total_change += stock.change_percent
                total_volume += stock.volume
                stock_count += 1
                
                # Update grid position
                x_pos += 1
                if x_pos >= grid_cols:
                    x_pos = 0
                    y_pos += 1
        
        # Calculate market overview metrics
        market_overview = {
            "avg_change": total_change / stock_count if stock_count > 0 else 0,
            "total_volume": total_volume,
            "hot_stocks": len([c for c in cells if c.value > 70]),
            "gainers": len([c for c in cells if c.change_percent > 0]),
            "losers": len([c for c in cells if c.change_percent < 0]),
            "market_heat_index": sum(c.value for c in cells) / len(cells) if cells else 0
        }
        
        # Determine data status
        now = datetime.now()
        market_hours = 9 <= now.hour < 16  # Simple market hours check (9 AM - 4 PM)
        has_real_data = len([c for c in cells if c.symbol in ["AAPL", "MSFT", "GOOGL"]]) > 0
        
        if has_real_data and market_hours:
            data_status = "live"
        elif has_real_data:
            data_status = "cached"  
        else:
            data_status = "sample"

        return HeatDistributionResponse(
            cells=cells,
            sectors=sectors,
            timestamp=datetime.now().isoformat(),
            market_overview=market_overview,
            data_status=data_status,
            market_open=market_hours
        )
        
    except Exception as e:
        logger.error(f"Error generating heat distribution: {e}")
        # Fallback to sample data if real data fails
        return await get_sample_heat_distribution()

@router.get("/sector/{sector_name}")
async def get_sector_heat(sector_name: str):
    """Get heat map data for a specific sector"""
    try:
        async with MultiSectorService() as service:
            all_data = await service.get_all_sector_data()
        
        if sector_name not in all_data:
            raise HTTPException(status_code=404, detail=f"Sector {sector_name} not found")
        
        stocks = all_data[sector_name]
        
        heat_data = []
        for stock in stocks:
            heat_data.append({
                "symbol": stock.symbol,
                "heat_score": stock.heat_score * 100,
                "change_percent": stock.change_percent,
                "volume": stock.volume,
                "market_cap": stock.market_cap,
                "volatility": stock.volatility
            })
        
        # Sort by heat score
        heat_data.sort(key=lambda x: x["heat_score"], reverse=True)
        
        return {
            "sector": sector_name,
            "stocks": heat_data,
            "sector_summary": {
                "avg_heat": sum(s["heat_score"] for s in heat_data) / len(heat_data) if heat_data else 0,
                "avg_change": sum(s["change_percent"] for s in heat_data) / len(heat_data) if heat_data else 0,
                "total_volume": sum(s["volume"] for s in heat_data),
                "hottest_stock": heat_data[0]["symbol"] if heat_data else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting sector heat data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/live-updates")
async def get_live_heat_updates():
    """Get live heat map updates for WebSocket replacement"""
    try:
        # This simulates live updates - in production would connect to real-time data
        async with MultiSectorService() as service:
            all_data = await service.get_all_sector_data()
        
        # Get random sample of hot stocks for live updates
        all_stocks = []
        for stocks in all_data.values():
            all_stocks.extend(stocks)
        
        # Sort by heat score and get top 10
        hot_stocks = sorted(all_stocks, key=lambda x: x.heat_score, reverse=True)[:10]
        
        live_updates = []
        for stock in hot_stocks:
            # Simulate small random changes for live effect
            change_delta = random.uniform(-0.1, 0.1)
            
            live_updates.append({
                "symbol": stock.symbol,
                "sector": stock.sector,
                "current_change": stock.change_percent + change_delta,
                "heat_score": min(1.0, max(0.0, stock.heat_score + random.uniform(-0.05, 0.05))),
                "volume": stock.volume + random.randint(-10000, 10000),
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "type": "live_heat_update",
            "updates": live_updates,
            "market_pulse": random.uniform(0.5, 0.9),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting live heat updates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-pulse")
async def get_market_pulse():
    """Get overall market heat pulse indicator"""
    try:
        async with MultiSectorService() as service:
            all_data = await service.get_all_sector_data()
        
        all_stocks = []
        for stocks in all_data.values():
            all_stocks.extend(stocks)
        
        if not all_stocks:
            return {"pulse": 0.5, "status": "neutral"}
        
        # Calculate market pulse based on heat scores and changes
        total_heat = sum(stock.heat_score for stock in all_stocks)
        avg_heat = total_heat / len(all_stocks)
        
        positive_changes = len([s for s in all_stocks if s.change_percent > 0])
        pulse_ratio = positive_changes / len(all_stocks)
        
        # Combine heat and momentum for pulse
        market_pulse = (avg_heat * 0.6) + (pulse_ratio * 0.4)
        
        # Determine status
        if market_pulse > 0.7:
            status = "hot"
        elif market_pulse > 0.5:
            status = "warm"
        elif market_pulse > 0.3:
            status = "neutral"
        else:
            status = "cool"
        
        return {
            "pulse": round(market_pulse, 3),
            "status": status,
            "heat_index": round(avg_heat * 100, 1),
            "positive_ratio": round(pulse_ratio * 100, 1),
            "total_stocks": len(all_stocks),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating market pulse: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_sample_heat_distribution():
    """Generate sample heat map data when real data is unavailable"""
    sample_stocks = [
        # Technology
        {"symbol": "AAPL", "sector": "Technology", "change": 2.45, "volume": 75000000, "heat": 0.85},
        {"symbol": "MSFT", "sector": "Technology", "change": 1.87, "volume": 45000000, "heat": 0.78},
        {"symbol": "NVDA", "sector": "Technology", "change": 3.21, "volume": 85000000, "heat": 0.92},
        {"symbol": "TSLA", "sector": "Technology", "change": 4.56, "volume": 95000000, "heat": 0.94},
        {"symbol": "GOOGL", "sector": "Technology", "change": 1.23, "volume": 35000000, "heat": 0.71},
        {"symbol": "META", "sector": "Technology", "change": -1.45, "volume": 55000000, "heat": 0.65},
        
        # Healthcare
        {"symbol": "JNJ", "sector": "Healthcare", "change": 0.87, "volume": 25000000, "heat": 0.68},
        {"symbol": "PFE", "sector": "Healthcare", "change": -0.34, "volume": 30000000, "heat": 0.55},
        {"symbol": "UNH", "sector": "Healthcare", "change": 2.11, "volume": 15000000, "heat": 0.76},
        {"symbol": "MRNA", "sector": "Healthcare", "change": -2.87, "volume": 40000000, "heat": 0.48},
        {"symbol": "BMY", "sector": "Healthcare", "change": 1.45, "volume": 20000000, "heat": 0.72},
        {"symbol": "ABBV", "sector": "Healthcare", "change": 0.98, "volume": 18000000, "heat": 0.69},
        
        # Finance
        {"symbol": "JPM", "sector": "Finance", "change": 1.56, "volume": 42000000, "heat": 0.74},
        {"symbol": "BAC", "sector": "Finance", "change": 2.34, "volume": 65000000, "heat": 0.81},
        {"symbol": "WFC", "sector": "Finance", "change": 0.45, "volume": 38000000, "heat": 0.63},
        {"symbol": "GS", "sector": "Finance", "change": -1.23, "volume": 28000000, "heat": 0.52},
        {"symbol": "MS", "sector": "Finance", "change": 1.89, "volume": 35000000, "heat": 0.77},
        {"symbol": "C", "sector": "Finance", "change": 0.67, "volume": 45000000, "heat": 0.66},
        
        # Energy
        {"symbol": "XOM", "sector": "Energy", "change": 3.45, "volume": 55000000, "heat": 0.88},
        {"symbol": "CVX", "sector": "Energy", "change": 2.87, "volume": 48000000, "heat": 0.83},
        {"symbol": "COP", "sector": "Energy", "change": 4.12, "volume": 62000000, "heat": 0.91},
        {"symbol": "EOG", "sector": "Energy", "change": 1.78, "volume": 32000000, "heat": 0.75},
        {"symbol": "SLB", "sector": "Energy", "change": -0.89, "volume": 38000000, "heat": 0.58},
        {"symbol": "OXY", "sector": "Energy", "change": 2.45, "volume": 45000000, "heat": 0.79},
    ]
    
    cells = []
    sectors = set()
    x_pos = 0
    y_pos = 0
    grid_cols = 8
    
    for i, stock in enumerate(sample_stocks):
        sectors.add(stock["sector"])
        
        # Calculate heat intensity
        heat_value = stock["heat"] * 100
        
        # Determine color based on change
        if stock["change"] > 2:
            color = '#00ff88'  # Strong green
        elif stock["change"] > 0:
            color = '#88ff44'  # Light green
        elif stock["change"] > -2:
            color = '#ffaa00'  # Orange
        else:
            color = '#ff3366'  # Red
        
        size = min(1.5, max(0.3, stock["volume"] / 50000000))
        
        # Generate mock signal for sample data
        if stock["change"] > 1.5 and stock["heat"] > 0.7:
            signal, confidence = "buy", 0.85
        elif stock["change"] < -1.0:
            signal, confidence = "sell", 0.75
        else:
            signal, confidence = "hold", 0.6
            
        cells.append(HeatMapCell(
            symbol=stock["symbol"],
            sector=stock["sector"],
            value=heat_value,
            change_percent=stock["change"],
            volume=stock["volume"],
            market_cap=stock["volume"] * 100,  # Mock market cap
            x=x_pos,
            y=y_pos,
            color=color,
            size=size,
            signal=signal,
            confidence=confidence
        ))
        
        x_pos += 1
        if x_pos >= grid_cols:
            x_pos = 0
            y_pos += 1
    
    # Calculate market overview
    total_change = sum(s["change"] for s in sample_stocks)
    market_overview = {
        "avg_change": total_change / len(sample_stocks),
        "total_volume": sum(s["volume"] for s in sample_stocks),
        "hot_stocks": len([s for s in sample_stocks if s["heat"] > 0.7]),
        "gainers": len([s for s in sample_stocks if s["change"] > 0]),
        "losers": len([s for s in sample_stocks if s["change"] < 0]),
        "market_heat_index": sum(s["heat"] for s in sample_stocks) / len(sample_stocks) * 100
    }
    
    now = datetime.now()
    market_hours = 9 <= now.hour < 16
    
    return HeatDistributionResponse(
        cells=cells,
        sectors=list(sectors),
        timestamp=datetime.now().isoformat(),
        market_overview=market_overview,
        data_status="sample",
        market_open=market_hours
    )

@router.get("/sample-distribution", response_model=HeatDistributionResponse)
async def get_sample_heat_distribution_endpoint():
    """Get sample heat map distribution data for testing"""
    return await get_sample_heat_distribution()