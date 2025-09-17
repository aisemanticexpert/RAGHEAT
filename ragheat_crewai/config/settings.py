"""
CrewAI Configuration Settings for RAGHeat Portfolio Construction System
"""

import os
from typing import Dict, Any, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from pathlib import Path

class CrewAISettings(BaseSettings):
    """Configuration settings for CrewAI system"""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )
    
    # API Keys and External Services
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    alpha_vantage_api_key: str = Field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY", ""))
    finnhub_api_key: str = Field(default_factory=lambda: os.getenv("FINNHUB_API_KEY", ""))
    
    # Database Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))
    
    # Redis Configuration
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    
    # CrewAI Configuration
    default_llm_model: str = Field(default="claude-3-5-sonnet-20241022")
    max_execution_time: int = Field(default=300)  # 5 minutes
    max_iterations: int = Field(default=10)
    
    # Heat Diffusion Parameters
    diffusion_coefficient: float = Field(default=0.1)
    diffusion_iterations: int = Field(default=50)
    convergence_threshold: float = Field(default=1e-6)
    
    # Portfolio Construction Parameters
    default_risk_tolerance: str = Field(default="neutral")  # conservative, neutral, aggressive
    max_portfolio_positions: int = Field(default=20)
    min_position_weight: float = Field(default=0.01)  # 1%
    max_position_weight: float = Field(default=0.1)   # 10%
    
    # Data Sources
    data_update_frequency: str = Field(default="daily")  # real-time, hourly, daily
    sentiment_sources: List[str] = Field(default=["news", "social_media", "analyst_ratings"])
    
    # Agent Behavior Configuration
    enable_agent_memory: bool = Field(default=True)
    agent_collaboration_mode: str = Field(default="structured_debate")  # independent, collaborative, structured_debate
    consensus_threshold: float = Field(default=0.7)  # 70% agreement for consensus
    
    # Logging and Monitoring
    log_level: str = Field(default="INFO")
    enable_telemetry: bool = Field(default=True)
    performance_tracking: bool = Field(default=True)
    
    # File Paths
    config_dir: Path = Field(default_factory=lambda: Path(__file__).parent)
    agents_config_path: Path = Field(default_factory=lambda: Path(__file__).parent / "agents.yaml")
    tasks_config_path: Path = Field(default_factory=lambda: Path(__file__).parent / "tasks.yaml")
    
    
    @property
    def anthropic_config(self) -> Dict[str, Any]:
        """Get Anthropic API configuration"""
        return {
            "api_key": self.anthropic_api_key,
            "model": self.default_llm_model,
            "max_tokens": 4096,
            "temperature": 0.1
        }
    
    @property
    def neo4j_config(self) -> Dict[str, Any]:
        """Get Neo4j database configuration"""
        return {
            "uri": self.neo4j_uri,
            "user": self.neo4j_user,
            "password": self.neo4j_password
        }
    
    @property
    def redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "host": self.redis_host,
            "port": self.redis_port,
            "db": self.redis_db
        }
    
    @property
    def agent_configs(self) -> Dict[str, Any]:
        """Load agent configurations from YAML file"""
        import yaml
        
        try:
            with open(self.agents_config_path, 'r', encoding='utf-8') as f:
                configs = yaml.safe_load(f)
            return configs.get('agents', {})
        except FileNotFoundError:
            # Return default agent configurations if file not found
            return {
                "fundamental_analyst": {
                    "role": "Fundamental Analyst",
                    "goal": "Analyze fundamental data and financial health of companies",
                    "backstory": "Expert in financial analysis with deep knowledge of company fundamentals",
                    "tools": ["fundamental_report_pull", "financial_ratio_calculator"],
                    "verbose": True,
                    "allow_delegation": False,
                    "max_iter": 5
                }
            }
    
    def get_risk_parameters(self, risk_tolerance: Optional[str] = None) -> Dict[str, float]:
        """Get risk parameters based on tolerance level"""
        tolerance = risk_tolerance or self.default_risk_tolerance
        
        risk_params = {
            "conservative": {
                "max_volatility": 0.15,
                "min_sharpe_ratio": 1.0,
                "max_drawdown": 0.05,
                "concentration_limit": 0.05
            },
            "neutral": {
                "max_volatility": 0.25,
                "min_sharpe_ratio": 0.8,
                "max_drawdown": 0.1,
                "concentration_limit": 0.08
            },
            "aggressive": {
                "max_volatility": 0.4,
                "min_sharpe_ratio": 0.6,
                "max_drawdown": 0.2,
                "concentration_limit": 0.12
            }
        }
        
        return risk_params.get(tolerance, risk_params["neutral"])
    
    def validate_configuration(self) -> bool:
        """Validate that all required configuration is present"""
        required_fields = [
            "anthropic_api_key",
            "neo4j_password"
        ]
        
        missing_fields = []
        for field in required_fields:
            if not getattr(self, field):
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {missing_fields}")
        
        return True

# Global settings instance
settings = CrewAISettings()