import os
from dataclasses import dataclass
from typing import List, Dict
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    '''Application configuration'''
    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_TOPIC_MARKET_DATA: str = "market-data-stream"
    KAFKA_TOPIC_NEWS: str = "news-stream"

    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # Heat Diffusion Parameters
    HEAT_DISSIPATION_RATE: float = 0.85
    PROPAGATION_STEPS: int = 5
    TIME_WINDOW: int = 24  # hours

    # Sectors Configuration
    SECTORS: List[str] = ["Technology", "Healthcare", "Finance", "Energy", "Consumer_Goods", "Industrial", "Utilities", "Real_Estate", "Materials", "Communication"]

    # Free Data APIs
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "demo")
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_KEY", "")

    # Update Intervals (seconds)
    PRICE_UPDATE_INTERVAL: int = 60
    NEWS_UPDATE_INTERVAL: int = 300
    HEAT_CALCULATION_INTERVAL: int = 30

config = Config()