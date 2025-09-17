"""
Portfolio Construction Tasks for RAGHeat CrewAI System
======================================================

CrewAI Task implementations for the portfolio construction workflow.
"""

from crewai import Task
from typing import Dict, Any, List, Optional

class ConstructKnowledgeGraphTask:
    """Task for constructing financial knowledge graph."""
    
    @staticmethod
    def create(agent, context_tasks: List[Task] = None) -> Task:
        return Task(
            description="""
            Build and update the financial knowledge graph with current market data.
            Extract entities (stocks, sectors, events, indicators) and relationships 
            (correlatesWith, affectedBy, belongsToSector) from multiple data sources.
            Integrate SEC filings, news, price data, and economic indicators into a 
            unified RDF/OWL knowledge structure.
            
            Key Requirements:
            - Extract company entities with ticker, sector, market cap attributes
            - Create event nodes for Fed decisions, earnings, news
            - Model relationships with weights and temporal annotations
            - Ensure graph is optimized for heat diffusion analysis
            """,
            agent=agent,
            expected_output="""
            A comprehensive knowledge graph containing:
            - Company nodes with attributes (ticker, sector, market cap)
            - Event nodes (Fed decisions, earnings, news)
            - Relationship edges with weights
            - Temporal annotations for dynamic updates
            - Graph statistics and quality metrics
            """,
            context=context_tasks or []
        )

class AnalyzeFundamentalsTask:
    """Task for fundamental analysis."""
    
    @staticmethod
    def create(agent, context_tasks: List[Task] = None) -> Task:
        return Task(
            description="""
            Perform deep fundamental analysis on target stocks by analyzing:
            - Latest 10-K and 10-Q reports
            - Revenue growth, profit margins, and cash flow trends
            - Debt-to-equity ratios and financial health metrics
            - Management discussion and forward guidance
            - Competitive positioning and market share
            
            Generate a fundamental score and long-term outlook for each stock.
            Use the knowledge graph context to understand sector relationships.
            """,
            agent=agent,
            expected_output="""
            Fundamental analysis report including:
            - Financial health score (1-10)
            - Growth trajectory assessment
            - Key risks and opportunities
            - Long-term BUY/HOLD/SELL recommendation
            - Supporting financial metrics and ratios
            - Confidence levels for each recommendation
            """,
            context=context_tasks or []
        )

class AssessMarketSentimentTask:
    """Task for market sentiment analysis."""
    
    @staticmethod
    def create(agent, context_tasks: List[Task] = None) -> Task:
        return Task(
            description="""
            Analyze current market sentiment and news flow by:
            - Processing financial news from Bloomberg, Reuters, WSJ
            - Monitoring social media buzz (Reddit, Twitter, StockTwits)
            - Tracking analyst rating changes and price targets
            - Identifying insider trading patterns
            - Measuring sentiment scores using FinBERT
            
            Determine short-term momentum and sentiment-driven opportunities.
            Leverage knowledge graph to understand sentiment propagation.
            """,
            agent=agent,
            expected_output="""
            Sentiment analysis report containing:
            - Aggregate sentiment score (-1 to +1)
            - Key news events and their impact
            - Social media momentum indicators
            - Analyst consensus and recent changes
            - Short-term sentiment-based recommendation
            - Sentiment trend analysis and momentum indicators
            """,
            context=context_tasks or []
        )

class CalculateValuationsTask:
    """Task for valuation analysis."""
    
    @staticmethod
    def create(agent, context_tasks: List[Task] = None) -> Task:
        return Task(
            description="""
            Perform quantitative valuation analysis including:
            - Calculate current P/E, P/B, EV/EBITDA ratios
            - Analyze price and volume trends over multiple timeframes
            - Compute technical indicators (RSI, MACD, Bollinger Bands)
            - Calculate volatility and risk metrics
            - Determine if stocks are over/undervalued relative to historical ranges
            
            Provide entry/exit signals and risk-adjusted position sizing recommendations.
            """,
            agent=agent,
            expected_output="""
            Valuation report with:
            - Current valuation metrics vs historical averages
            - Technical indicator signals
            - Volatility and risk assessment
            - Price targets based on multiple methodologies
            - Valuation-based BUY/HOLD/SELL signal
            - Entry/exit points and stop-loss levels
            """,
            context=context_tasks or []
        )

class SimulateHeatDiffusionTask:
    """Task for heat diffusion simulation."""
    
    @staticmethod
    def create(agent, context_tasks: List[Task] = None) -> Task:
        return Task(
            description="""
            Model influence propagation using heat diffusion equations:
            - Identify recent shock events (Fed decisions, earnings surprises)
            - Set initial heat values at event source nodes
            - Simulate heat propagation through graph using Laplacian
            - Calculate heat scores for all stocks after N iterations
            - Identify stocks most/least affected by cascading influences
            
            Use diffusion coefficient β and iterate until convergence.
            Integrate with fundamental and sentiment analysis for context.
            """,
            agent=agent,
            expected_output="""
            Heat diffusion analysis including:
            - Heat map visualization of influence spread
            - Ranked list of stocks by heat intensity
            - Influence paths from events to stocks
            - Cascade risk assessment
            - Structural importance scores
            - Systemic risk metrics and early warning indicators
            """,
            context=context_tasks or []
        )

class FacilitateAgentDebateTask:
    """Task for agent debate facilitation."""
    
    @staticmethod
    def create(agent, context_tasks: List[Task] = None) -> Task:
        return Task(
            description="""
            Coordinate multi-agent debate and consensus building:
            - Present each specialist's analysis to the group
            - Identify areas of agreement and disagreement
            - Facilitate structured debate using Round Robin approach
            - Challenge agents to defend their positions with evidence
            - Guide agents toward consensus on each stock
            - Resolve conflicts through weighted scoring
            
            Continue debate until consensus or maximum iterations reached.
            Synthesize insights from fundamental, sentiment, valuation, and heat diffusion analyses.
            """,
            agent=agent,
            expected_output="""
            Consensus report containing:
            - Final BUY/SELL decisions for each stock
            - Areas of agent agreement/disagreement
            - Debate transcript and key arguments
            - Confidence scores for each decision
            - Risk-adjusted consensus recommendations
            - Minority opinions and alternative scenarios
            """,
            context=context_tasks or []
        )

class ConstructPortfolioTask:
    """Task for portfolio construction."""
    
    @staticmethod
    def create(agent, context_tasks: List[Task] = None) -> Task:
        return Task(
            description="""
            Build final portfolio based on multi-agent consensus:
            - Select stocks with strongest BUY consensus
            - Determine position weights based on conviction and risk
            - Apply risk tolerance constraints (conservative/neutral/aggressive)
            - Ensure appropriate diversification across sectors
            - Calculate expected portfolio metrics (return, volatility, Sharpe)
            - Generate rebalancing recommendations
            
            Optimize for risk-adjusted returns while respecting constraints.
            """,
            agent=agent,
            expected_output="""
            Portfolio construction report with:
            - Final stock selections and weights
            - Expected return and risk metrics
            - Sharpe ratio and other performance indicators
            - Sector allocation breakdown
            - Rebalancing schedule and triggers
            - Implementation guidance and execution plan
            """,
            context=context_tasks or []
        )

class GenerateInvestmentRationaleTask:
    """Task for investment rationale generation."""
    
    @staticmethod
    def create(agent, context_tasks: List[Task] = None) -> Task:
        return Task(
            description="""
            Create comprehensive explanation for portfolio decisions:
            - Trace causal chains from macro events to stock impacts
            - Generate narrative explanations using chain-of-thought
            - Create visual heat maps showing influence propagation
            - Document evidence supporting each decision
            - Ensure regulatory compliance and auditability
            - Produce both technical and simplified explanations
            
            Make AI decision-making transparent and trustworthy.
            """,
            agent=agent,
            expected_output="""
            Investment explanation package including:
            - Executive summary of portfolio strategy
            - Detailed rationale for each position
            - Visual heat maps and influence graphs
            - Evidence trail and source citations
            - Risk disclosures and limitations
            - Both technical and plain-language versions
            - Regulatory compliance documentation
            """,
            context=context_tasks or []
        )