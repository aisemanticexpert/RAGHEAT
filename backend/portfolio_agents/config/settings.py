"""
Configuration Settings for RAGHeat Portfolio Construction System
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class PortfolioSettings(BaseSettings):
    """Configuration settings for the portfolio construction system."""
    
    # API Keys
    ANTHROPIC_API_KEY: str = Field(
        default="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",
        description="Anthropic Claude API key"
    )
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    FINNHUB_API_KEY: Optional[str] = Field(default=None, description="Finnhub API key")
    
    # Database Settings
    NEO4J_URI: str = Field(default="neo4j://localhost:7687", description="Neo4j database URI")
    NEO4J_USERNAME: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="password", description="Neo4j password")
    
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis cache URL")
    
    # Agent Configuration
    MAX_ITERATIONS: int = Field(default=10, description="Maximum iterations for agent tasks")
    AGENT_TIMEOUT: int = Field(default=300, description="Agent timeout in seconds")
    ENABLE_MEMORY: bool = Field(default=True, description="Enable agent memory")
    VERBOSE_LOGGING: bool = Field(default=True, description="Enable verbose logging")
    
    # Portfolio Parameters
    RISK_FREE_RATE: float = Field(default=0.05, description="Risk-free rate for calculations")
    DEFAULT_PORTFOLIO_SIZE: int = Field(default=10, description="Default number of stocks in portfolio")
    REBALANCING_FREQUENCY: str = Field(default="monthly", description="Portfolio rebalancing frequency")
    
    # Heat Diffusion Parameters
    DIFFUSION_COEFFICIENT: float = Field(default=0.1, description="Heat diffusion coefficient β")
    DIFFUSION_ITERATIONS: int = Field(default=50, description="Number of diffusion iterations")
    CONVERGENCE_THRESHOLD: float = Field(default=1e-6, description="Convergence threshold for heat diffusion")
    
    # Data Sources
    DEFAULT_STOCK_UNIVERSE: list = Field(
        default=[
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.B", 
            "JNJ", "JPM", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE", 
            "NFLX", "CRM", "INTC", "VZ", "KO", "PFE", "T", "CSCO", "ABT", "XOM"
        ],
        description="Default stock universe for analysis"
    )
    
    # Sentiment Analysis
    SENTIMENT_SOURCES: Dict[str, bool] = Field(
        default={
            "reddit": True,
            "twitter": True, 
            "news": True,
            "analyst_ratings": True,
            "insider_trading": True
        },
        description="Enabled sentiment analysis sources"
    )
    
    # Risk Management
    MAX_POSITION_SIZE: float = Field(default=0.15, description="Maximum position size per stock")
    MAX_SECTOR_ALLOCATION: float = Field(default=0.30, description="Maximum sector allocation")
    STOP_LOSS_THRESHOLD: float = Field(default=0.08, description="Stop loss threshold")
    
    # API Rate Limits
    API_RATE_LIMIT: int = Field(default=100, description="API requests per minute limit")
    BATCH_SIZE: int = Field(default=10, description="Batch size for processing stocks")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = PortfolioSettings()

# Environment setup for AI models
os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY
if settings.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY