"""
Analysis Services API Routes
============================

API endpoints for specific analysis services and tools.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime

from ...core.portfolio_system import RAGHeatPortfolioSystem

router = APIRouter()

class AnalysisRequest(BaseModel):
    """Request model for analysis services."""
    stocks: List[str]
    analysis_parameters: Dict[str, Any] = {}

def get_portfolio_system() -> RAGHeatPortfolioSystem:
    """Get portfolio system instance."""
    raise HTTPException(status_code=500, detail="Portfolio system dependency not configured")

@router.post("/fundamental", response_model=Dict[str, Any])
async def fundamental_analysis(
    request: AnalysisRequest,
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """Perform fundamental analysis on stocks."""
    try:
        result = portfolio_system.run_individual_analysis(
            agent_type="fundamental_analyst",
            stocks=request.stocks,
            analysis_type="comprehensive"
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in fundamental analysis: {str(e)}"
        )

@router.post("/sentiment", response_model=Dict[str, Any])
async def sentiment_analysis(
    request: AnalysisRequest,
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """Perform sentiment analysis on stocks."""
    try:
        result = portfolio_system.run_individual_analysis(
            agent_type="sentiment_analyst",
            stocks=request.stocks,
            analysis_type="comprehensive"
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in sentiment analysis: {str(e)}"
        )

@router.post("/technical", response_model=Dict[str, Any])
async def technical_analysis(
    request: AnalysisRequest,
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """Perform technical analysis on stocks."""
    try:
        result = portfolio_system.run_individual_analysis(
            agent_type="valuation_analyst",
            stocks=request.stocks,
            analysis_type="comprehensive"
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in technical analysis: {str(e)}"
        )

@router.post("/heat-diffusion", response_model=Dict[str, Any])
async def heat_diffusion_analysis(
    request: AnalysisRequest,
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """Perform heat diffusion analysis."""
    try:
        result = portfolio_system.run_individual_analysis(
            agent_type="heat_diffusion_analyst",
            stocks=request.stocks,
            analysis_type="comprehensive"
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in heat diffusion analysis: {str(e)}"
        )

@router.get("/tools", response_model=Dict[str, Any])
async def get_analysis_tools():
    """Get information about available analysis tools."""
    return {
        "available_tools": {
            "fundamental": ["SEC filings analysis", "Financial ratio calculation", "Growth assessment"],
            "sentiment": ["News aggregation", "Social media monitoring", "Analyst rating tracking"],
            "technical": ["Price/volume analysis", "Technical indicators", "Risk metrics"],
            "network": ["Knowledge graph construction", "Heat diffusion simulation", "Influence propagation"]
        },
        "timestamp": datetime.now().isoformat()
    }