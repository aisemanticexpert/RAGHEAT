"""
RAGHeat CrewAI Tools Module
===========================

This module contains all tool implementations for the RAGHeat portfolio construction system.
Tools are organized by functional area and agent specialization.
"""

from .fundamental_tools import *
from .sentiment_tools import *
from .valuation_tools import *
from .graph_tools import *
from .heat_diffusion_tools import *
from .portfolio_tools import *
from .explanation_tools import *
from .tool_registry import get_tools_for_agent, register_tool, get_all_tools

__all__ = [
    # Fundamental analysis tools
    "FundamentalReportPull",
    "FinancialReportRAG", 
    "SECFilingAnalyzer",
    "FinancialRatioCalculator",
    
    # Sentiment analysis tools
    "NewsAggregator",
    "SentimentAnalyzer",
    "SocialMediaMonitor",
    "AnalystRatingTracker",
    "FinBERTSentiment",
    
    # Valuation analysis tools
    "PriceVolumeAnalyzer",
    "TechnicalIndicatorCalculator",
    "VolatilityCalculator",
    "SharpeRatioCalculator",
    "CorrelationAnalyzer",
    
    # Knowledge graph tools
    "GraphConstructor",
    "SPARQLQueryEngine",
    "Neo4jInterface",
    "OntologyMapper",
    "TripleExtractor",
    
    # Heat diffusion tools
    "HeatEquationSolver",
    "GraphLaplacianCalculator",
    "DiffusionSimulator",
    "InfluencePropagator",
    "HeatKernelCalculator",
    
    # Portfolio management tools
    "ConsensusBuilder",
    "DebateModerator",
    "PortfolioOptimizer",
    "RiskAssessor",
    "WeightAllocator",
    
    # Explanation tools
    "ChainOfThoughtGenerator",
    "VisualizationCreator",
    "ReportGenerator",
    "LangChainRAG",
    "GPT4Interface",
    
    # Registry functions
    "get_tools_for_agent",
    "register_tool",
    "get_all_tools"
]