"""
Multi-sector analysis API routes with efficient data handling
Supports 150+ NASDAQ stocks across 10 sectors including Tesla
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
import logging

import sys
import os

# Add paths for imports
sys.path.append('/app')
sys.path.append('/app/backend')

try:
    from backend.services.multi_sector_service import MultiSectorService
    from backend.config.sector_stocks import SECTOR_STOCKS, PRIORITY_STOCKS
except ImportError:
    # Try relative import
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from services.multi_sector_service import MultiSectorService
    from config.sector_stocks import SECTOR_STOCKS, PRIORITY_STOCKS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sectors", tags=["Multi-Sector Analysis"])

# Global service instance
sector_service = MultiSectorService()

class SectorRequest(BaseModel):
    sectors: Optional[List[str]] = None
    top_only: bool = True

class BubbleChartResponse(BaseModel):
    symbol: str
    sector: str
    sector_name: str
    x: float  # Price change %
    y: float  # Volatility
    size: float  # Heat score
    color: str
    price: float
    volume: int
    market_cap: float

class StockResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: float
    sector: str
    heat_score: float
    volatility: float
    timestamp: str

class SectorSummaryResponse(BaseModel):
    sector: str
    sector_name: str
    color: str
    total_stocks: int
    top_performers: List[str]
    avg_change: float
    total_volume: int
    heat_index: float

@router.get("/bubble-chart", response_model=List[BubbleChartResponse])
async def get_bubble_chart_data():
    """Get bubble chart data for all sectors"""
    try:
        async with MultiSectorService() as service:
            bubble_data = await service.get_bubble_chart_data()
            
        return [
            BubbleChartResponse(**data) for data in bubble_data
        ]
        
    except Exception as e:
        logger.error(f"Error generating bubble chart data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summaries", response_model=List[SectorSummaryResponse])
async def get_sector_summaries():
    """Get comprehensive summaries for all sectors"""
    try:
        async with MultiSectorService() as service:
            summaries = await service.get_sector_summaries()
            
        return [
            SectorSummaryResponse(
                sector=s.sector,
                sector_name=s.sector_name,
                color=s.color,
                total_stocks=s.total_stocks,
                top_performers=s.top_performers,
                avg_change=s.avg_change,
                total_volume=s.total_volume,
                heat_index=s.heat_index
            )
            for s in summaries
        ]
        
    except Exception as e:
        logger.error(f"Error generating sector summaries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top-stocks", response_model=Dict[str, List[StockResponse]])
async def get_top_stocks_by_sectors():
    """Get top 4 stocks from each sector"""
    try:
        async with MultiSectorService() as service:
            top_stocks = await service.get_top_stocks_by_sectors()
            
        result = {}
        for sector, stocks in top_stocks.items():
            result[sector] = [
                StockResponse(
                    symbol=stock.symbol,
                    price=stock.price,
                    change=stock.change,
                    change_percent=stock.change_percent,
                    volume=stock.volume,
                    market_cap=stock.market_cap,
                    sector=stock.sector,
                    heat_score=stock.heat_score,
                    volatility=stock.volatility,
                    timestamp=stock.timestamp.isoformat()
                )
                for stock in stocks
            ]
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting top stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/all-data", response_model=Dict[str, List[StockResponse]])
async def get_all_sector_data():
    """Get all stock data organized by sectors"""
    try:
        async with MultiSectorService() as service:
            all_data = await service.get_all_sector_data()
            
        result = {}
        for sector, stocks in all_data.items():
            result[sector] = [
                StockResponse(
                    symbol=stock.symbol,
                    price=stock.price,
                    change=stock.change,
                    change_percent=stock.change_percent,
                    volume=stock.volume,
                    market_cap=stock.market_cap,
                    sector=stock.sector,
                    heat_score=stock.heat_score,
                    volatility=stock.volatility,
                    timestamp=stock.timestamp.isoformat()
                )
                for stock in stocks
            ]
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting all sector data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tesla-analysis", response_model=StockResponse)
async def get_tesla_analysis():
    """Get detailed analysis for Tesla (TSLA)"""
    try:
        async with MultiSectorService() as service:
            tesla_data = await service.get_tesla_analysis()
            
        if not tesla_data:
            raise HTTPException(status_code=404, detail="Tesla data not available")
            
        return StockResponse(
            symbol=tesla_data.symbol,
            price=tesla_data.price,
            change=tesla_data.change,
            change_percent=tesla_data.change_percent,
            volume=tesla_data.volume,
            market_cap=tesla_data.market_cap,
            sector=tesla_data.sector,
            heat_score=tesla_data.heat_score,
            volatility=tesla_data.volatility,
            timestamp=tesla_data.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting Tesla analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sector-config")
async def get_sector_configuration():
    """Get sector configuration with stock lists"""
    try:
        return {
            "sectors": SECTOR_STOCKS,
            "priority_stocks": PRIORITY_STOCKS,
            "total_stocks": sum(len(data["all_stocks"]) for data in SECTOR_STOCKS.values()),
            "total_sectors": len(SECTOR_STOCKS)
        }
        
    except Exception as e:
        logger.error(f"Error getting sector config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hot-stocks", response_model=List[StockResponse])
async def get_hot_stocks(limit: int = 20):
    """Get hottest stocks across all sectors based on heat score"""
    try:
        async with MultiSectorService() as service:
            all_data = await service.get_all_sector_data()
            
        # Flatten all stocks and sort by heat score
        all_stocks = []
        for sector_stocks in all_data.values():
            all_stocks.extend(sector_stocks)
        
        # Sort by combined heat score and volatility
        hot_stocks = sorted(
            all_stocks, 
            key=lambda x: x.heat_score * 0.6 + x.volatility * 0.4,
            reverse=True
        )[:limit]
        
        return [
            StockResponse(
                symbol=stock.symbol,
                price=stock.price,
                change=stock.change,
                change_percent=stock.change_percent,
                volume=stock.volume,
                market_cap=stock.market_cap,
                sector=stock.sector,
                heat_score=stock.heat_score,
                volatility=stock.volatility,
                timestamp=stock.timestamp.isoformat()
            )
            for stock in hot_stocks
        ]
        
    except Exception as e:
        logger.error(f"Error getting hot stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance-summary")
async def get_performance_summary():
    """Get overall market performance summary"""
    try:
        async with MultiSectorService() as service:
            all_data = await service.get_all_sector_data()
            
        total_stocks = sum(len(stocks) for stocks in all_data.values())
        
        if total_stocks == 0:
            return {"message": "No data available"}
            
        # Calculate overall metrics
        all_changes = []
        all_volumes = []
        all_heat_scores = []
        
        for sector_stocks in all_data.values():
            for stock in sector_stocks:
                all_changes.append(stock.change_percent)
                all_volumes.append(stock.volume)
                all_heat_scores.append(stock.heat_score)
        
        gainers = len([c for c in all_changes if c > 0])
        losers = len([c for c in all_changes if c < 0])
        
        return {
            "total_stocks": total_stocks,
            "gainers": gainers,
            "losers": losers,
            "avg_change": sum(all_changes) / len(all_changes) if all_changes else 0,
            "total_volume": sum(all_volumes),
            "market_heat_index": sum(all_heat_scores) / len(all_heat_scores) if all_heat_scores else 0,
            "sectors_analyzed": len(all_data),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))