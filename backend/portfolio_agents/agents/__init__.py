"""
Agent Implementation Module for RAGHeat Portfolio Construction
============================================================

This module contains the agent implementations based on the CrewAI framework.
Each agent is specialized for a specific aspect of portfolio construction.
"""

from .agent_factory import AgentFactory
from .base_portfolio_agent import BasePortfolioAgent

__all__ = [
    'AgentFactory',
    'BasePortfolioAgent'
]