"""
Agent Management API Routes
===========================

API endpoints for managing and monitoring individual agents.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime

from ...core.portfolio_system import RAGHeatPortfolioSystem

router = APIRouter()

class AgentRequest(BaseModel):
    """Request model for agent operations."""
    stocks: List[str]
    analysis_type: str = "comprehensive"
    context: Dict[str, Any] = {}

def get_portfolio_system() -> RAGHeatPortfolioSystem:
    """Get portfolio system instance."""
    raise HTTPException(status_code=500, detail="Portfolio system dependency not configured")

@router.get("/", response_model=Dict[str, Any])
async def list_agents(
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """List all available agents and their capabilities."""
    try:
        agents_info = {}
        
        for agent_type, agent in portfolio_system.agents.items():
            agents_info[agent_type] = agent.get_agent_status()
        
        return {
            "agents": agents_info,
            "total_agents": len(agents_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing agents: {str(e)}"
        )

@router.get("/{agent_type}", response_model=Dict[str, Any])
async def get_agent_info(
    agent_type: str,
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """Get detailed information about a specific agent."""
    try:
        if agent_type not in portfolio_system.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_type} not found")
        
        agent = portfolio_system.agents[agent_type]
        return agent.get_agent_status()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting agent info: {str(e)}"
        )

@router.post("/{agent_type}/analyze", response_model=Dict[str, Any])
async def run_agent_analysis(
    agent_type: str,
    request: AgentRequest,
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """Run analysis using a specific agent."""
    try:
        result = portfolio_system.run_individual_analysis(
            agent_type=agent_type,
            stocks=request.stocks,
            analysis_type=request.analysis_type
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running agent analysis: {str(e)}"
        )

@router.get("/{agent_type}/performance", response_model=Dict[str, Any])
async def get_agent_performance(
    agent_type: str,
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """Get performance metrics for a specific agent."""
    try:
        if agent_type not in portfolio_system.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_type} not found")
        
        agent = portfolio_system.agents[agent_type]
        return agent.get_performance_summary()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting agent performance: {str(e)}"
        )