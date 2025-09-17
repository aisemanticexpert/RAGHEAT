"""
Workflow Management API Routes
==============================

API endpoints for managing and executing workflows.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime

from ...core.portfolio_system import RAGHeatPortfolioSystem

router = APIRouter()

class WorkflowRequest(BaseModel):
    """Request model for workflow execution."""
    workflow_type: str
    input_data: Dict[str, Any]

def get_portfolio_system() -> RAGHeatPortfolioSystem:
    """Get portfolio system instance."""
    raise HTTPException(status_code=500, detail="Portfolio system dependency not configured")

@router.get("/templates", response_model=Dict[str, Any])
async def get_workflow_templates(
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """Get available workflow templates."""
    try:
        templates = portfolio_system.workflow_manager.get_workflow_templates()
        return {
            "templates": templates,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting workflow templates: {str(e)}"
        )

@router.post("/execute", response_model=Dict[str, Any])
async def execute_workflow(
    request: WorkflowRequest,
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """Execute a specific workflow."""
    try:
        result = portfolio_system.workflow_manager.execute_workflow(
            workflow_type=request.workflow_type,
            input_data=request.input_data,
            agents=portfolio_system.agents
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing workflow: {str(e)}"
        )

@router.get("/status", response_model=Dict[str, Any])
async def get_workflow_status(
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """Get current workflow status."""
    try:
        status = portfolio_system.workflow_manager.get_workflow_status()
        return status
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting workflow status: {str(e)}"
        )

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_workflow_history(
    limit: int = 10,
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """Get workflow execution history."""
    try:
        history = portfolio_system.workflow_manager.get_execution_history(limit=limit)
        return history
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting workflow history: {str(e)}"
        )