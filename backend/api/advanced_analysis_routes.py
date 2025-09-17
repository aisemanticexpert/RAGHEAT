"""
Advanced Market Analysis API Routes
Provides 90%+ probability options trading analysis with real-time Neo4j integration
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio
import logging
from dataclasses import asdict

from services.market_orchestrator import MarketOrchestrator, MarketAnalysisResult
from services.sector_performance_analyzer import SectorPerformance
from services.options_analyzer import OptionsOpportunity
from services.news_sentiment_analyzer import SentimentAnalysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/advanced", tags=["Advanced Market Analysis"])

# Pydantic models for API responses
class SectorPerformanceResponse(BaseModel):
    sector: str
    sector_name: str
    performance_score: float
    avg_return: float
    volatility: float
    momentum: float
    volume_surge: float
    news_sentiment: float
    top_stocks: List[str]
    timestamp: str

class OptionsOpportunityResponse(BaseModel):
    symbol: str
    sector: str
    win_probability: float
    strategy: str
    target_price: float
    expected_move: float
    volatility_rank: float
    volume_surge: float
    earnings_date: Optional[str]
    news_catalyst: str
    risk_reward_ratio: float
    confidence_score: float
    entry_signals: List[str]

class MarketAnalysisResponse(BaseModel):
    session_id: str
    timestamp: str
    analysis_duration: float
    top_sectors: List[SectorPerformanceResponse]
    worst_sectors: List[SectorPerformanceResponse]
    high_probability_opportunities: List[OptionsOpportunityResponse]
    market_summary: Dict[str, float]
    winning_probability: float
    recommended_actions: List[str]
    graph_session_id: str
    data_freshness: str

class LiveGraphAnalysisResponse(BaseModel):
    session_id: str
    timestamp: str
    live_opportunities: List[Dict[str, Any]]
    cypher_query: str

# Global orchestrator instance
orchestrator = MarketOrchestrator()

# Background task for continuous analysis
analysis_task_running = False

@router.post("/start-continuous-analysis")
async def start_continuous_analysis(background_tasks: BackgroundTasks):
    """Start continuous market analysis (runs every 5 minutes)"""
    global analysis_task_running
    
    if analysis_task_running:
        return {"status": "already_running", "message": "Continuous analysis already active"}
    
    background_tasks.add_task(run_continuous_analysis)
    analysis_task_running = True
    
    return {
        "status": "started", 
        "message": "Continuous analysis started",
        "interval_minutes": 5
    }

@router.get("/market-analysis", response_model=MarketAnalysisResponse)
async def get_comprehensive_market_analysis():
    """Get comprehensive market analysis with 90%+ probability opportunities"""
    try:
        start_time = datetime.now()
        
        # Check for cached analysis first
        cached_analysis = await orchestrator.get_cached_analysis(max_age_minutes=5)
        
        if cached_analysis:
            logger.info("Returning cached analysis")
            analysis_result = cached_analysis
            data_freshness = "cached"
        else:
            logger.info("Running new comprehensive analysis...")
            analysis_result = await orchestrator.run_comprehensive_analysis()
            data_freshness = "fresh"
        
        # Convert dataclasses to response models
        top_sectors = [
            SectorPerformanceResponse(
                sector=s.sector,
                sector_name=s.sector_name,
                performance_score=s.performance_score,
                avg_return=s.avg_return,
                volatility=s.volatility,
                momentum=s.momentum,
                volume_surge=s.volume_surge,
                news_sentiment=s.news_sentiment,
                top_stocks=s.top_stocks,
                timestamp=s.timestamp.isoformat()
            ) for s in analysis_result.top_sectors
        ]
        
        worst_sectors = [
            SectorPerformanceResponse(
                sector=s.sector,
                sector_name=s.sector_name,
                performance_score=s.performance_score,
                avg_return=s.avg_return,
                volatility=s.volatility,
                momentum=s.momentum,
                volume_surge=s.volume_surge,
                news_sentiment=s.news_sentiment,
                top_stocks=s.top_stocks,
                timestamp=s.timestamp.isoformat()
            ) for s in analysis_result.worst_sectors
        ]
        
        opportunities = [
            OptionsOpportunityResponse(
                symbol=opp.symbol,
                sector=opp.sector,
                win_probability=opp.win_probability,
                strategy=opp.strategy,
                target_price=opp.target_price,
                expected_move=opp.expected_move,
                volatility_rank=opp.volatility_rank,
                volume_surge=opp.volume_surge,
                earnings_date=opp.earnings_date,
                news_catalyst=opp.news_catalyst,
                risk_reward_ratio=opp.risk_reward_ratio,
                confidence_score=opp.confidence_score,
                entry_signals=opp.entry_signals
            ) for opp in analysis_result.high_probability_opportunities
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return MarketAnalysisResponse(
            session_id=analysis_result.session_id,
            timestamp=analysis_result.timestamp.isoformat(),
            analysis_duration=execution_time,
            top_sectors=top_sectors,
            worst_sectors=worst_sectors,
            high_probability_opportunities=opportunities,
            market_summary=analysis_result.market_summary,
            winning_probability=analysis_result.winning_probability,
            recommended_actions=analysis_result.recommended_actions,
            graph_session_id=analysis_result.graph_session_id,
            data_freshness=data_freshness
        )
        
    except Exception as e:
        logger.error(f"Error in comprehensive market analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/sector-analysis/{sector_name}")
async def get_sector_deep_dive(sector_name: str):
    """Get detailed analysis for a specific sector"""
    try:
        # Get cached analysis
        cached_analysis = await orchestrator.get_cached_analysis()
        
        if not cached_analysis:
            raise HTTPException(status_code=404, detail="No recent analysis available. Run comprehensive analysis first.")
        
        # Find sector data
        target_sector = None
        for sector in cached_analysis.top_sectors + cached_analysis.worst_sectors:
            if sector.sector_name.lower() == sector_name.lower() or sector.sector.lower() == sector_name.lower():
                target_sector = sector
                break
        
        if not target_sector:
            raise HTTPException(status_code=404, detail=f"Sector '{sector_name}' not found in analysis")
        
        # Get sector opportunities
        sector_opportunities = cached_analysis.sector_opportunities.get(target_sector.sector, [])
        
        # Get sentiment data for sector stocks
        sector_sentiment = {}
        for stock in target_sector.top_stocks:
            if stock in cached_analysis.sentiment_analysis:
                sentiment = cached_analysis.sentiment_analysis[stock]
                sector_sentiment[stock] = {
                    "overall_sentiment": sentiment.overall_sentiment,
                    "sentiment_strength": sentiment.sentiment_strength,
                    "catalyst_potential": sentiment.catalyst_potential,
                    "news_count": sentiment.news_count,
                    "key_topics": sentiment.key_topics
                }
        
        return {
            "sector_info": {
                "name": target_sector.sector_name,
                "performance_score": target_sector.performance_score,
                "avg_return": target_sector.avg_return,
                "volatility": target_sector.volatility,
                "momentum": target_sector.momentum,
                "volume_surge": target_sector.volume_surge,
                "news_sentiment": target_sector.news_sentiment
            },
            "top_stocks": target_sector.top_stocks,
            "opportunities": [asdict(opp) for opp in sector_opportunities],
            "sentiment_analysis": sector_sentiment,
            "high_probability_count": len([opp for opp in sector_opportunities if opp.win_probability >= 0.90]),
            "avg_win_probability": sum(opp.win_probability for opp in sector_opportunities) / len(sector_opportunities) if sector_opportunities else 0,
            "timestamp": target_sector.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in sector analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock-analysis/{symbol}")
async def get_stock_deep_dive(symbol: str):
    """Get comprehensive analysis for a specific stock"""
    try:
        cached_analysis = await orchestrator.get_cached_analysis()
        
        if not cached_analysis:
            raise HTTPException(status_code=404, detail="No recent analysis available")
        
        symbol = symbol.upper()
        
        # Find all opportunities for this stock
        stock_opportunities = []
        for opportunities in cached_analysis.sector_opportunities.values():
            stock_opportunities.extend([opp for opp in opportunities if opp.symbol == symbol])
        
        if not stock_opportunities:
            raise HTTPException(status_code=404, detail=f"No opportunities found for {symbol}")
        
        # Get sentiment analysis
        sentiment = cached_analysis.sentiment_analysis.get(symbol)
        
        # Find sector
        stock_sector = None
        for sector in cached_analysis.top_sectors + cached_analysis.worst_sectors:
            if symbol in sector.top_stocks:
                stock_sector = sector
                break
        
        return {
            "symbol": symbol,
            "sector": stock_sector.sector_name if stock_sector else "Unknown",
            "opportunities": [asdict(opp) for opp in stock_opportunities],
            "best_opportunity": asdict(max(stock_opportunities, key=lambda x: x.win_probability)),
            "sentiment_analysis": asdict(sentiment) if sentiment else None,
            "sector_performance": stock_sector.performance_score if stock_sector else 0,
            "total_opportunities": len(stock_opportunities),
            "high_probability_opportunities": len([opp for opp in stock_opportunities if opp.win_probability >= 0.90]),
            "strategies_available": list(set(opp.strategy for opp in stock_opportunities)),
            "avg_win_probability": sum(opp.win_probability for opp in stock_opportunities) / len(stock_opportunities),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stock analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/live-graph-analysis/{graph_session_id}", response_model=LiveGraphAnalysisResponse)
async def get_live_graph_analysis(graph_session_id: str):
    """Get live analysis from Neo4j graph database"""
    try:
        live_data = await orchestrator.get_live_graph_analysis(graph_session_id)
        cypher_query = await orchestrator.graph_builder.get_live_analysis_query(graph_session_id)
        
        return LiveGraphAnalysisResponse(
            session_id=graph_session_id,
            timestamp=datetime.now().isoformat(),
            live_opportunities=live_data,
            cypher_query=cypher_query
        )
        
    except Exception as e:
        logger.error(f"Error in live graph analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-pulse-advanced")
async def get_advanced_market_pulse():
    """Get advanced market pulse with sentiment and volatility metrics"""
    try:
        cached_analysis = await orchestrator.get_cached_analysis()
        
        if not cached_analysis:
            return {
                "status": "no_analysis",
                "message": "No recent analysis available",
                "timestamp": datetime.now().isoformat()
            }
        
        market_summary = cached_analysis.market_summary
        
        # Calculate advanced pulse metrics
        bullish_ratio = market_summary.get("call_opportunities", 0) / max(market_summary.get("total_opportunities", 1), 1)
        bearish_ratio = market_summary.get("put_opportunities", 0) / max(market_summary.get("total_opportunities", 1), 1)
        
        # Market direction signal
        if bullish_ratio > 0.6:
            direction = "BULLISH"
            strength = bullish_ratio
        elif bearish_ratio > 0.6:
            direction = "BEARISH" 
            strength = bearish_ratio
        else:
            direction = "NEUTRAL"
            strength = 0.5
        
        return {
            "market_direction": direction,
            "direction_strength": strength,
            "overall_winning_probability": cached_analysis.winning_probability,
            "avg_sector_performance": market_summary.get("avg_top_sector_performance", 0),
            "market_sentiment": market_summary.get("avg_market_sentiment", 0),
            "catalyst_potential": market_summary.get("avg_catalyst_potential", 0),
            "total_opportunities": market_summary.get("total_opportunities", 0),
            "high_probability_count": len([
                opp for opp in cached_analysis.high_probability_opportunities 
                if opp.win_probability >= 0.90
            ]),
            "ultra_high_probability_count": len([
                opp for opp in cached_analysis.high_probability_opportunities 
                if opp.win_probability >= 0.95
            ]),
            "top_opportunity": {
                "symbol": cached_analysis.high_probability_opportunities[0].symbol,
                "strategy": cached_analysis.high_probability_opportunities[0].strategy,
                "win_probability": cached_analysis.high_probability_opportunities[0].win_probability
            } if cached_analysis.high_probability_opportunities else None,
            "market_breadth": market_summary.get("market_breadth", 0),
            "analysis_age_minutes": (
                datetime.now() - cached_analysis.timestamp
            ).total_seconds() / 60,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in advanced market pulse: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top-opportunities")
async def get_top_opportunities(limit: int = 10, min_probability: float = 0.90):
    """Get top opportunities filtered by probability"""
    try:
        cached_analysis = await orchestrator.get_cached_analysis()
        
        if not cached_analysis:
            raise HTTPException(status_code=404, detail="No recent analysis available")
        
        # Filter and sort opportunities
        filtered_opportunities = [
            opp for opp in cached_analysis.high_probability_opportunities
            if opp.win_probability >= min_probability
        ]
        
        # Sort by probability and confidence
        filtered_opportunities.sort(
            key=lambda x: (x.win_probability, x.confidence_score),
            reverse=True
        )
        
        # Limit results
        top_opportunities = filtered_opportunities[:limit]
        
        return {
            "total_opportunities": len(cached_analysis.high_probability_opportunities),
            "filtered_count": len(filtered_opportunities),
            "returned_count": len(top_opportunities),
            "filter_criteria": {
                "min_probability": min_probability,
                "limit": limit
            },
            "opportunities": [asdict(opp) for opp in top_opportunities],
            "avg_win_probability": sum(opp.win_probability for opp in top_opportunities) / len(top_opportunities) if top_opportunities else 0,
            "strategy_breakdown": {
                "calls": len([opp for opp in top_opportunities if opp.strategy == "call"]),
                "puts": len([opp for opp in top_opportunities if opp.strategy == "put"]),
                "straddles": len([opp for opp in top_opportunities if opp.strategy == "straddle"])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting top opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_continuous_analysis():
    """Background task for continuous analysis"""
    global analysis_task_running
    
    logger.info("Starting continuous analysis background task")
    
    try:
        while analysis_task_running:
            try:
                logger.info("Running scheduled comprehensive analysis...")
                await orchestrator.run_comprehensive_analysis()
                logger.info("Scheduled analysis completed successfully")
                
                # Wait 5 minutes before next analysis
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in continuous analysis: {e}")
                # Wait shorter time on error, then retry
                await asyncio.sleep(60)
                
    except asyncio.CancelledError:
        logger.info("Continuous analysis task cancelled")
    finally:
        analysis_task_running = False
        logger.info("Continuous analysis background task ended")

@router.delete("/stop-continuous-analysis")
async def stop_continuous_analysis():
    """Stop continuous market analysis"""
    global analysis_task_running
    
    analysis_task_running = False
    
    return {
        "status": "stopped",
        "message": "Continuous analysis stopped"
    }