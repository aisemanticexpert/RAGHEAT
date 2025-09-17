"""
RAGHeat Multi-Agent Portfolio Construction System
================================================

A sophisticated multi-agent system for equity portfolio construction using CrewAI framework.
Implements heat diffusion models, knowledge graphs, and AI-driven consensus building.

Agents:
- Fundamental Analyst: Deep fundamental analysis from SEC filings
- Sentiment Analyst: News and social media sentiment analysis  
- Valuation Analyst: Technical and quantitative valuation
- Knowledge Graph Engineer: Builds financial relationship networks
- Heat Diffusion Analyst: Models influence propagation
- Portfolio Coordinator: Orchestrates agent debates and consensus
- Explanation Generator: Creates transparent investment rationales

Based on BlackRock research and AlphaAgents papers.
"""

__version__ = "1.0.0"
__author__ = "RAGHeat Team"

from .core.portfolio_system import RAGHeatPortfolioSystem
from .agents.agent_factory import AgentFactory
from .tools.tool_registry import ToolRegistry

__all__ = [
    'RAGHeatPortfolioSystem',
    'AgentFactory', 
    'ToolRegistry'
]