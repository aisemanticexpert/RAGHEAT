"""
RAGHeat CrewAI Multi-Agent Portfolio Construction System
========================================================

This module implements a sophisticated multi-agent system for financial portfolio construction
using CrewAI framework, based on RAGHeat and AlphaAgents research papers.

Key Components:
- Knowledge Graph Construction and Maintenance
- Heat Diffusion Analysis for Influence Propagation
- Multi-Agent Consensus Building through Structured Debate
- Portfolio Optimization with Risk Management

Agents:
- Fundamental Analyst: Deep fundamental analysis using SEC filings
- Sentiment Analyst: Market sentiment and news analysis  
- Valuation Analyst: Quantitative valuation and technical analysis
- Knowledge Graph Engineer: Graph construction and semantic modeling
- Heat Diffusion Analyst: Influence propagation modeling
- Portfolio Coordinator: Multi-agent coordination and consensus building
- Explanation Generator: Investment rationale and transparency
"""

__version__ = "1.0.0"
__author__ = "RAGHeat Team"

from .crews.portfolio_crew import PortfolioConstructionCrew
from .config.settings import CrewAISettings

__all__ = ["PortfolioConstructionCrew", "CrewAISettings"]