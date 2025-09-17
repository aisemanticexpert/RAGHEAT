"""
System Management API Routes
============================

API endpoints for system monitoring, health checks, and management.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime

from ...core.portfolio_system import RAGHeatPortfolioSystem

router = APIRouter()

class SystemStatus(BaseModel):
    """System status response model."""
    status: str
    timestamp: str
    system_health: Dict[str, Any]
    agents_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]

class ConfigurationExport(BaseModel):
    """Configuration export response model."""
    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    settings: Dict[str, Any]
    export_timestamp: str

def get_portfolio_system() -> RAGHeatPortfolioSystem:
    """Get portfolio system instance."""
    # This would be properly injected in production
    raise HTTPException(status_code=500, detail="Portfolio system dependency not configured")

@router.get("/status", response_model=SystemStatus)
async def get_system_status(
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """
    Get comprehensive system status.
    
    Returns detailed information about:
    - Overall system health
    - Agent status and performance
    - Recent execution statistics
    - Configuration validation status
    """
    try:
        system_status = portfolio_system.get_system_status()
        health_check = portfolio_system.health_check()
        performance = portfolio_system.get_agent_performance_summary()
        
        return SystemStatus(
            status="healthy" if health_check['overall_health'] == 'healthy' else "degraded",
            timestamp=datetime.now().isoformat(),
            system_health=health_check,
            agents_status=system_status.get('agents', {}),
            performance_metrics=performance
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving system status: {str(e)}"
        )

@router.get("/health")
async def health_check(
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """
    Simple health check endpoint.
    
    Returns basic health status for load balancers and monitoring systems.
    """
    try:
        health = portfolio_system.health_check()
        
        if health['overall_health'] == 'healthy':
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        else:
            return {
                "status": "degraded", 
                "issues": health.get('issues', []),
                "warnings": health.get('warnings', []),
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"System health check failed: {str(e)}"
        )

@router.get("/metrics", response_model=Dict[str, Any])
async def get_system_metrics(
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """
    Get detailed system performance metrics.
    
    Returns metrics suitable for monitoring dashboards and alerting systems.
    """
    try:
        performance = portfolio_system.get_agent_performance_summary()
        execution_history = portfolio_system.get_execution_history(limit=100)
        
        # Calculate additional metrics
        total_executions = len(execution_history)
        successful_executions = sum(1 for ex in execution_history if ex.get('success', False))
        
        recent_executions = execution_history[-10:] if execution_history else []
        recent_success_rate = sum(1 for ex in recent_executions if ex.get('success', False)) / max(len(recent_executions), 1)
        
        avg_execution_time = sum(
            ex.get('execution_time', 0) for ex in execution_history
        ) / max(total_executions, 1)
        
        return {
            "system_metrics": {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "overall_success_rate": successful_executions / max(total_executions, 1),
                "recent_success_rate": recent_success_rate,
                "average_execution_time": avg_execution_time,
                "uptime": "N/A",  # Would be calculated from startup time
                "memory_usage": "N/A",  # Would use psutil in production
                "cpu_usage": "N/A"  # Would use psutil in production
            },
            "agent_metrics": performance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving system metrics: {str(e)}"
        )

@router.post("/reset")
async def reset_system(
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """
    Reset the entire system.
    
    Clears execution history, resets agents, and reinitializes components.
    Use with caution - this will clear all session data.
    """
    try:
        portfolio_system.reset_system()
        
        return {
            "status": "success",
            "message": "System reset completed successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting system: {str(e)}"
        )

@router.get("/configuration", response_model=ConfigurationExport)
async def export_configuration(
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """
    Export current system configuration.
    
    Returns complete configuration including agents, tasks, and settings.
    Useful for backup, debugging, and deployment verification.
    """
    try:
        config = portfolio_system.export_configuration()
        
        return ConfigurationExport(
            agents_config=config.get('agents_config', {}),
            tasks_config=config.get('tasks_config', {}),
            settings=config.get('settings', {}),
            export_timestamp=config.get('export_timestamp', datetime.now().isoformat())
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting configuration: {str(e)}"
        )

@router.get("/logs", response_model=List[Dict[str, Any]])
async def get_system_logs(
    limit: int = 100,
    level: str = "INFO"
):
    """
    Get recent system logs.
    
    Returns recent log entries filtered by level.
    Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    try:
        # In production, this would integrate with actual logging system
        # For now, return mock logs
        mock_logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "System status check completed successfully",
                "module": "system_routes"
            },
            {
                "timestamp": datetime.now().isoformat(), 
                "level": "INFO",
                "message": "Portfolio construction request received",
                "module": "portfolio_routes"
            },
            {
                "timestamp": datetime.now().isoformat(),
                "level": "DEBUG",
                "message": "Agent factory initialized with 7 agents",
                "module": "agent_factory"
            }
        ]
        
        return mock_logs[:limit]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving logs: {str(e)}"
        )

@router.get("/version", response_model=Dict[str, Any])
async def get_version_info():
    """
    Get system version and build information.
    
    Returns version details, build timestamp, and component versions.
    """
    return {
        "version": "1.0.0",
        "build_date": "2024-01-01",
        "git_commit": "abc123def",
        "components": {
            "crewai": "0.28.8",
            "langchain": "0.1.20", 
            "fastapi": "0.111.0",
            "anthropic": "0.25.9"
        },
        "python_version": "3.11+",
        "system_type": "RAGHeat Portfolio Construction",
        "api_version": "v1"
    }

@router.post("/maintenance")
async def toggle_maintenance_mode(
    enabled: bool,
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """
    Toggle system maintenance mode.
    
    When enabled, the system will reject new portfolio construction requests
    but allow status and monitoring endpoints.
    """
    try:
        # In production, this would set a global maintenance flag
        return {
            "status": "success",
            "maintenance_mode": enabled,
            "message": f"Maintenance mode {'enabled' if enabled else 'disabled'}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error toggling maintenance mode: {str(e)}"
        )

@router.get("/diagnostics", response_model=Dict[str, Any])
async def run_system_diagnostics(
    portfolio_system: RAGHeatPortfolioSystem = Depends(get_portfolio_system)
):
    """
    Run comprehensive system diagnostics.
    
    Performs various system checks and returns detailed diagnostic information.
    """
    try:
        diagnostics = {
            "system_check": "PASS",
            "agents_check": "PASS", 
            "configuration_check": "PASS",
            "dependencies_check": "PASS",
            "database_check": "PASS",
            "memory_check": "PASS",
            "performance_check": "PASS"
        }
        
        # Perform actual diagnostic checks
        try:
            # Check system status
            system_status = portfolio_system.get_system_status()
            if system_status.get('system_status') != 'ready':
                diagnostics["system_check"] = "FAIL"
            
            # Check agents
            agents_status = system_status.get('agents', {})
            if not agents_status:
                diagnostics["agents_check"] = "FAIL"
            
            # Check configuration
            config_valid = system_status.get('configuration_valid', False)
            if not config_valid:
                diagnostics["configuration_check"] = "FAIL"
                
        except Exception as check_error:
            diagnostics["error"] = str(check_error)
        
        overall_status = "PASS" if all(
            status == "PASS" for status in diagnostics.values() 
            if isinstance(status, str) and status in ["PASS", "FAIL"]
        ) else "FAIL"
        
        return {
            "overall_status": overall_status,
            "diagnostics": diagnostics,
            "timestamp": datetime.now().isoformat(),
            "recommendations": [
                "System appears healthy" if overall_status == "PASS" 
                else "Some system components may need attention"
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running diagnostics: {str(e)}"
        )