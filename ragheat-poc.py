#!/usr/bin/env python3
"""
RAGHeat POC Deployment Script
This script creates the complete project structure and files for the RAGHeat system
"""

import os
import sys
from pathlib import Path
import textwrap

def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {path}")


def write_file(filepath, content):
    """Write content to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent(content).strip())
    print(f"✓ Created file: {filepath}")


def deploy_ragheat_project():
    """Deploy the complete RAGHeat project structure"""

    print("=" * 60)
    print("RAGHeat POC Deployment Script")
    print("=" * 60)

    # Get project root directory
    project_name = "ragheat-poc"
    base_dir = Path.cwd() / project_name

    print(f"\nDeploying project to: {base_dir}\n")

    # Create main directories
    directories = [
        base_dir,
        base_dir / "backend",
        base_dir / "backend" / "agents",
        base_dir / "backend" / "graph",
        base_dir / "backend" / "data_pipeline",
        base_dir / "backend" / "ontology",
        base_dir / "backend" / "api",
        base_dir / "backend" / "api" / "routes",
        base_dir / "backend" / "config",
        base_dir / "backend" / "tests",
        base_dir / "frontend",
        base_dir / "frontend" / "src",
        base_dir / "frontend" / "src" / "components",
        base_dir / "frontend" / "public",
    ]

    for directory in directories:
        create_directory(directory)

    # Configuration files
    print("\n📝 Creating configuration files...")

    # .env file
    write_file(base_dir / ".env", """
    # API Keys - Replace with your actual keys
    ANTHROPIC_API_KEY=your_anthropic_api_key_here
    ALPHA_VANTAGE_KEY=demo
    FINNHUB_KEY=your_finnhub_key_here
    NEO4J_PASSWORD=password

    # Service URLs
    NEO4J_URI=bolt://localhost:7687
    KAFKA_BOOTSTRAP_SERVERS=localhost:9092
    """)

    # .gitignore
    write_file(base_dir / ".gitignore", """
    # Python
    __pycache__/
    *.py[cod]
    *$py.class
    *.so
    .Python
    build/
    develop-eggs/
    dist/
    downloads/
    eggs/
    .eggs/
    lib/
    lib64/
    parts/
    sdist/
    var/
    wheels/
    *.egg-info/
    .installed.cfg
    *.egg
    MANIFEST

    # Virtual Environment
    venv/
    ENV/
    env/

    # IDE
    .vscode/
    .idea/
    *.swp
    *.swo

    # Environment
    .env
    .env.local

    # Node
    node_modules/
    npm-debug.log*
    yarn-debug.log*
    yarn-error.log*

    # Docker
    *.log

    # Neo4j
    neo4j_data/

    # OS
    .DS_Store
    Thumbs.db
    """)

    # requirements.txt
    write_file(base_dir / "requirements.txt", """
    fastapi==0.104.1
    uvicorn==0.24.0
    crewai==0.30.0
    langchain==0.1.0
    networkx==3.2
    numpy==1.24.3
    pandas==2.1.1
    scipy==1.11.3
    yfinance==0.2.31
    kafka-python==2.0.2
    neo4j==5.14.0
    rdflib==7.0.0
    textblob==0.17.1
    pydantic==2.4.2
    python-dotenv==1.0.0
    websockets==12.0
    aiofiles==23.2.1
    httpx==0.25.0
    anthropic==0.7.0
    """)

    # docker-compose.yml
    write_file(base_dir / "docker-compose.yml", """
    version: '3.8'

    services:
      neo4j:
        image: neo4j:5.14
        ports:
          - "7474:7474"
          - "7687:7687"
        environment:
          - NEO4J_AUTH=neo4j/password
          - NEO4J_PLUGINS=["graph-data-science"]
        volumes:
          - neo4j_data:/data

      kafka:
        image: confluentinc/cp-kafka:7.5.0
        ports:
          - "9092:9092"
        environment:
          - KAFKA_BROKER_ID=1
          - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
          - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
          - KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1
        depends_on:
          - zookeeper

      zookeeper:
        image: confluentinc/cp-zookeeper:7.5.0
        ports:
          - "2181:2181"
        environment:
          - ZOOKEEPER_CLIENT_PORT=2181
          - ZOOKEEPER_TICK_TIME=2000

    volumes:
      neo4j_data:
    """)

    # Backend Configuration
    print("\n⚙️ Creating backend configuration...")

    write_file(base_dir / "backend" / "__init__.py", "")
    write_file(base_dir / "backend" / "agents" / "__init__.py", "")
    write_file(base_dir / "backend" / "graph" / "__init__.py", "")
    write_file(base_dir / "backend" / "data_pipeline" / "__init__.py", "")
    write_file(base_dir / "backend" / "api" / "__init__.py", "")

    # config/settings.py
    write_file(base_dir / "backend" / "config" / "settings.py", """
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
        SECTORS: List[str] = [
            "Technology", "Healthcare", "Finance", "Energy", 
            "Consumer_Goods", "Industrial", "Utilities", 
            "Real_Estate", "Materials", "Communication"
        ]

        # Free Data APIs
        ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "demo")
        FINNHUB_API_KEY: str = os.getenv("FINNHUB_KEY", "")

        # Update Intervals (seconds)
        PRICE_UPDATE_INTERVAL: int = 60
        NEWS_UPDATE_INTERVAL: int = 300
        HEAT_CALCULATION_INTERVAL: int = 30

    config = Config()
    """)

    # Graph Module
    print("\n📊 Creating graph module...")

    write_file(base_dir / "backend" / "graph" / "knowledge_graph.py", """
    import networkx as nx
    import numpy as np
    from typing import Dict, List, Tuple, Optional
    from datetime import datetime
    import logging
    from dataclasses import dataclass
    import json

    logger = logging.getLogger(__name__)

    @dataclass
    class Node:
        '''Represents a node in the knowledge graph'''
        id: str
        type: str  # 'sector', 'stock', 'feature', 'event'
        level: int
        attributes: Dict
        heat_score: float = 0.0
        timestamp: datetime = None

    class FinancialKnowledgeGraph:
        '''
        Manages the financial knowledge graph with hierarchical structure:
        Level 0: Root (SECTOR)
        Level 1: Sectors (Technology, Healthcare, etc.)
        Level 2: Individual Stocks
        '''

        def __init__(self):
            self.graph = nx.DiGraph()
            self.node_mapping = {}
            self.heat_scores = {}
            self.initialize_graph()

        def initialize_graph(self):
            '''Initialize the graph with root and sector nodes'''
            # Add root node
            root = Node(
                id="ROOT_SECTOR",
                type="root",
                level=0,
                attributes={"name": "Market"},
                heat_score=0.0,
                timestamp=datetime.now()
            )
            self.add_node(root)

            # Add sector nodes
            sectors = [
                "Technology", "Healthcare", "Finance", "Energy",
                "Consumer_Goods", "Industrial", "Utilities",
                "Real_Estate", "Materials", "Communication"
            ]

            for sector in sectors:
                sector_node = Node(
                    id=f"SECTOR_{sector.upper()}",
                    type="sector",
                    level=1,
                    attributes={
                        "name": sector,
                        "market_cap": 0,
                        "avg_pe": 0,
                        "volatility": 0
                    },
                    heat_score=0.0,
                    timestamp=datetime.now()
                )
                self.add_node(sector_node)
                self.add_edge("ROOT_SECTOR", sector_node.id, weight=1.0)

        def add_node(self, node: Node):
            '''Add a node to the graph'''
            self.graph.add_node(
                node.id,
                type=node.type,
                level=node.level,
                attributes=node.attributes,
                heat_score=node.heat_score,
                timestamp=node.timestamp
            )
            self.node_mapping[node.id] = node
            self.heat_scores[node.id] = node.heat_score

        def add_edge(self, source: str, target: str, weight: float = 1.0):
            '''Add an edge between nodes'''
            self.graph.add_edge(source, target, weight=weight)

        def add_stock(self, stock_symbol: str, sector: str, attributes: Dict):
            '''Add a stock node to the appropriate sector'''
            stock_node = Node(
                id=f"STOCK_{stock_symbol}",
                type="stock",
                level=2,
                attributes={
                    "symbol": stock_symbol,
                    "sector": sector,
                    **attributes
                },
                heat_score=0.0,
                timestamp=datetime.now()
            )

            self.add_node(stock_node)
            sector_id = f"SECTOR_{sector.upper()}"
            if sector_id in self.node_mapping:
                self.add_edge(sector_id, stock_node.id)

            return stock_node

        def update_node_attributes(self, node_id: str, attributes: Dict):
            '''Update attributes of an existing node'''
            if node_id in self.graph.nodes:
                self.graph.nodes[node_id]['attributes'].update(attributes)
                self.graph.nodes[node_id]['timestamp'] = datetime.now()

        def get_neighbors(self, node_id: str) -> List[str]:
            '''Get all neighbors of a node'''
            return list(self.graph.neighbors(node_id))

        def get_node_by_level(self, level: int) -> List[Node]:
            '''Get all nodes at a specific level'''
            nodes = []
            for node_id, data in self.graph.nodes(data=True):
                if data.get('level') == level:
                    nodes.append(self.node_mapping[node_id])
            return nodes

        def to_json(self) -> str:
            '''Export graph to JSON format'''
            nodes = []
            edges = []

            for node_id, data in self.graph.nodes(data=True):
                nodes.append({
                    "id": node_id,
                    "type": data.get("type"),
                    "level": data.get("level"),
                    "attributes": data.get("attributes"),
                    "heat_score": self.heat_scores.get(node_id, 0)
                })

            for source, target, data in self.graph.edges(data=True):
                edges.append({
                    "source": source,
                    "target": target,
                    "weight": data.get("weight", 1.0)
                })

            return json.dumps({"nodes": nodes, "edges": edges}, indent=2)
    """)

    write_file(base_dir / "backend" / "graph" / "heat_diffusion.py", """
    import numpy as np
    import networkx as nx
    from typing import Dict, List, Tuple, Optional
    from scipy import sparse
    import logging

    logger = logging.getLogger(__name__)

    class HeatDiffusionEngine:
        '''
        Implements heat diffusion over the financial knowledge graph
        Based on the heat equation: dh(t)/dt = -βL·h(t)
        '''

        def __init__(self, graph: nx.DiGraph, beta: float = 0.85):
            self.graph = graph
            self.beta = beta  # Diffusion coefficient
            self.heat_values = {}
            self.node_list = list(graph.nodes())
            self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}

        def compute_laplacian(self) -> np.ndarray:
            '''Compute the graph Laplacian matrix'''
            n = len(self.node_list)
            A = nx.adjacency_matrix(self.graph, nodelist=self.node_list)
            D = sparse.diags(A.sum(axis=1).A1)
            L = D - A
            return L

        def apply_heat_source(self, source_nodes: Dict[str, float]):
            '''
            Apply heat to source nodes (e.g., stocks affected by events)
            source_nodes: {node_id: heat_value}
            '''
            n = len(self.node_list)
            heat_vector = np.zeros(n)

            for node_id, heat_value in source_nodes.items():
                if node_id in self.node_to_idx:
                    idx = self.node_to_idx[node_id]
                    heat_vector[idx] = heat_value

            return heat_vector

        def propagate_heat(self, initial_heat: np.ndarray, time_steps: int = 5) -> np.ndarray:
            '''
            Propagate heat through the graph using iterative method
            '''
            L = self.compute_laplacian()

            # For numerical stability, use iterative approach
            heat = initial_heat.copy()
            dt = 0.1  # Time step

            for _ in range(time_steps):
                # Euler method: h(t+dt) = h(t) - β·dt·L·h(t)
                heat_change = -self.beta * dt * L.dot(heat)
                heat = heat + heat_change

                # Ensure non-negative heat values
                heat = np.maximum(heat, 0)

                # Normalize to prevent explosion
                if heat.max() > 0:
                    heat = heat / heat.max()

            return heat

        def calculate_heat_distribution(
            self, 
            event_impacts: Dict[str, float],
            propagation_steps: int = 5
        ) -> Dict[str, float]:
            '''
            Calculate heat distribution across the graph given event impacts
            '''
            # Apply heat sources
            initial_heat = self.apply_heat_source(event_impacts)

            # Propagate heat
            final_heat = self.propagate_heat(initial_heat, propagation_steps)

            # Convert back to dictionary
            heat_distribution = {}
            for idx, node_id in enumerate(self.node_list):
                heat_distribution[node_id] = float(final_heat[idx])

            self.heat_values = heat_distribution
            return heat_distribution

        def get_heated_sectors(self, top_k: int = 3) -> List[Tuple[str, float]]:
            '''Get the most heated sectors'''
            sector_heats = {}

            for node_id, heat in self.heat_values.items():
                if node_id.startswith("SECTOR_"):
                    sector_heats[node_id] = heat

            # Sort by heat score
            sorted_sectors = sorted(
                sector_heats.items(), 
                key=lambda x: x[1], 
                reverse=True
            )

            return sorted_sectors[:top_k]

        def get_heated_stocks_in_sector(
            self, 
            sector_id: str, 
            top_k: int = 5
        ) -> List[Tuple[str, float]]:
            '''Get the most heated stocks within a specific sector'''
            stock_heats = {}

            # Get all stocks in the sector
            sector_stocks = [
                n for n in self.graph.neighbors(sector_id)
                if n.startswith("STOCK_")
            ]

            for stock_id in sector_stocks:
                if stock_id in self.heat_values:
                    stock_heats[stock_id] = self.heat_values[stock_id]

            # Sort by heat score
            sorted_stocks = sorted(
                stock_heats.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return sorted_stocks[:top_k]
    """)

    # Agents Module
    print("\n🤖 Creating agents module...")

    write_file(base_dir / "backend" / "agents" / "base_agent.py", """
    from crewai import Agent, Task
    from typing import Dict, List, Optional
    import logging

    logger = logging.getLogger(__name__)

    class BaseFinancialAgent:
        '''Base class for all financial agents'''

        def __init__(self, role: str, goal: str, backstory: str):
            self.agent = Agent(
                role=role,
                goal=goal,
                backstory=backstory,
                verbose=True,
                allow_delegation=False
            )
            self.logger = logging.getLogger(self.__class__.__name__)
    """)

    write_file(base_dir / "backend" / "agents" / "fundamental_agent.py", """
    from crewai import Agent, Task
    from typing import Dict, List, Optional
    import yfinance as yf
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)

    class FundamentalAgent:
        '''Agent responsible for fundamental analysis of stocks'''

        def __init__(self):
            self.agent = Agent(
                role='Fundamental Analyst',
                goal='Analyze company fundamentals and financial health',
                backstory='''You are an experienced fundamental analyst with 
                20 years of experience analyzing financial statements, 
                evaluating company performance, and identifying value opportunities.
                You focus on metrics like P/E ratio, revenue growth, debt levels, 
                and cash flow.''',
                verbose=True,
                allow_delegation=False
            )

        def analyze_stock(self, symbol: str) -> Dict:
            '''Perform fundamental analysis on a stock'''
            try:
                stock = yf.Ticker(symbol)
                info = stock.info

                # Extract key fundamental metrics
                fundamentals = {
                    'symbol': symbol,
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'peg_ratio': info.get('pegRatio', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'roe': info.get('returnOnEquity', 0),
                    'revenue_growth': info.get('revenueGrowth', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'free_cash_flow': info.get('freeCashflow', 0),
                    'recommendation': info.get('recommendationKey', 'hold')
                }

                # Calculate fundamental score (0-1)
                score = self._calculate_fundamental_score(fundamentals)
                fundamentals['fundamental_score'] = score

                return fundamentals

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                return {'symbol': symbol, 'error': str(e), 'fundamental_score': 0.5}

        def _calculate_fundamental_score(self, metrics: Dict) -> float:
            '''Calculate a composite fundamental score'''
            score = 0.5  # Base score

            # P/E Ratio (lower is better, typically)
            pe = metrics.get('pe_ratio', 0)
            if 0 < pe < 15:
                score += 0.1
            elif 15 <= pe < 25:
                score += 0.05
            elif pe > 40:
                score -= 0.1

            # ROE (higher is better)
            roe = metrics.get('roe', 0)
            if roe > 0.20:
                score += 0.1
            elif roe > 0.15:
                score += 0.05

            # Debt to Equity (lower is better)
            de = metrics.get('debt_to_equity', 0)
            if de < 0.5:
                score += 0.1
            elif de > 2.0:
                score -= 0.1

            # Revenue Growth (positive is good)
            growth = metrics.get('revenue_growth', 0)
            if growth > 0.15:
                score += 0.1
            elif growth > 0.05:
                score += 0.05
            elif growth < 0:
                score -= 0.1

            # Ensure score is between 0 and 1
            return max(0, min(1, score))
    """)

    write_file(base_dir / "backend" / "agents" / "sentiment_agent.py", """
    from crewai import Agent, Task
    import requests
    from typing import Dict, List
    from textblob import TextBlob
    import logging

    logger = logging.getLogger(__name__)

    class SentimentAgent:
        '''Agent responsible for analyzing market sentiment from news and social media'''

        def __init__(self, finnhub_api_key: str = None):
            self.finnhub_api_key = finnhub_api_key or "demo"
            self.agent = Agent(
                role='Sentiment Analyst',
                goal='Analyze market sentiment from news and social media',
                backstory='''You are a sentiment analysis expert who monitors 
                news feeds, social media, and market chatter to gauge investor 
                sentiment. You can identify trends, detect market mood shifts, 
                and predict sentiment-driven price movements.''',
                verbose=True,
                allow_delegation=False
            )

        def analyze_sentiment(self, symbol: str) -> Dict:
            '''Analyze sentiment for a stock'''
            news_sentiment = self._get_news_sentiment(symbol)

            return {
                'symbol': symbol,
                'news_sentiment': news_sentiment,
                'sentiment_score': self._calculate_sentiment_score(news_sentiment)
            }

        def _get_news_sentiment(self, symbol: str) -> Dict:
            '''Get news sentiment (mock implementation for demo)'''
            try:
                # For demo, use mock data
                mock_news = [
                    f"{symbol} reports strong quarterly earnings",
                    f"Analysts upgrade {symbol} to buy",
                    f"{symbol} announces new product launch"
                ]

                sentiments = []
                for headline in mock_news:
                    blob = TextBlob(headline)
                    sentiments.append(blob.sentiment.polarity)

                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

                return {
                    'headline_count': len(mock_news),
                    'average_sentiment': avg_sentiment,
                    'sentiment_label': self._get_sentiment_label(avg_sentiment)
                }

            except Exception as e:
                logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
                return {'error': str(e), 'average_sentiment': 0}

        def _get_sentiment_label(self, score: float) -> str:
            '''Convert sentiment score to label'''
            if score > 0.5:
                return "Very Positive"
            elif score > 0.1:
                return "Positive"
            elif score > -0.1:
                return "Neutral"
            elif score > -0.5:
                return "Negative"
            else:
                return "Very Negative"

        def _calculate_sentiment_score(self, sentiment_data: Dict) -> float:
            '''Calculate normalized sentiment score (0-1)'''
            avg_sentiment = sentiment_data.get('average_sentiment', 0)
            # Normalize from [-1, 1] to [0, 1]
            return (avg_sentiment + 1) / 2
    """)

    write_file(base_dir / "backend" / "agents" / "valuation_agent.py", """
    from crewai import Agent, Task
    import yfinance as yf
    import numpy as np
    import pandas as pd
    from typing import Dict, List
    import logging

    logger = logging.getLogger(__name__)

    class ValuationAgent:
        '''Agent responsible for technical analysis and valuation'''

        def __init__(self):
            self.agent = Agent(
                role='Valuation Analyst',
                goal='Analyze stock valuation and technical indicators',
                backstory='''You are a quantitative analyst specializing in 
                stock valuation and technical analysis. You use mathematical 
                models, price patterns, and technical indicators to identify 
                trading opportunities and assess if stocks are over or undervalued.''',
                verbose=True,
                allow_delegation=False
            )

        def analyze_valuation(self, symbol: str) -> Dict:
            '''Perform valuation analysis on a stock'''
            try:
                stock = yf.Ticker(symbol)

                # Get historical data
                hist = stock.history(period="3mo")

                if hist.empty:
                    return {'symbol': symbol, 'error': 'No data available', 'valuation_score': 0.5}

                # Calculate technical indicators
                valuation = {
                    'symbol': symbol,
                    'current_price': float(hist['Close'].iloc[-1]) if not hist.empty else 0,
                    'volatility': float(hist['Close'].pct_change().std() * np.sqrt(252)) if len(hist) > 1 else 0,
                    'rsi': self._calculate_rsi(hist['Close']) if len(hist) > 14 else 50,
                    'macd': self._calculate_macd(hist['Close']) if len(hist) > 26 else {},
                    'price_change_30d': float(
                        (hist['Close'].iloc[-1] / hist['Close'].iloc[-30] - 1) * 100
                    ) if len(hist) > 30 else 0,
                    'volume_avg': float(hist['Volume'].mean()) if 'Volume' in hist else 0,
                    'valuation_score': 0.5  # Will be calculated
                }

                # Calculate valuation score
                valuation['valuation_score'] = self._calculate_valuation_score(valuation)

                return valuation

            except Exception as e:
                logger.error(f"Error analyzing valuation for {symbol}: {str(e)}")
                return {'symbol': symbol, 'error': str(e), 'valuation_score': 0.5}

        def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
            '''Calculate RSI indicator'''
            if len(prices) < period:
                return 50.0

            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1]) if not rsi.empty else 50

        def _calculate_macd(self, prices: pd.Series) -> Dict:
            '''Calculate MACD indicator'''
            if len(prices) < 26:
                return {'macd': 0, 'signal': 0, 'histogram': 0}

            ema12 = prices.ewm(span=12, adjust=False).mean()
            ema26 = prices.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            return {
                'macd': float(macd_line.iloc[-1]) if not macd_line.empty else 0,
                'signal': float(signal_line.iloc[-1]) if not signal_line.empty else 0,
                'histogram': float(macd_line.iloc[-1] - signal_line.iloc[-1]) 
                            if not macd_line.empty else 0
            }

        def _calculate_valuation_score(self, metrics: Dict) -> float:
            '''Calculate composite valuation score'''
            score = 0.5

            # RSI (30-70 is normal range)
            rsi = metrics.get('rsi', 50)
            if rsi < 30:
                score += 0.15  # Oversold
            elif rsi > 70:
                score -= 0.15  # Overbought

            # Price momentum
            price_change = metrics.get('price_change_30d', 0)
            if price_change > 10:
                score += 0.1
            elif price_change < -10:
                score -= 0.1

            # MACD signal
            macd_data = metrics.get('macd', {})
            if isinstance(macd_data, dict):
                histogram = macd_data.get('histogram', 0)
                if histogram > 0:
                    score += 0.05
                else:
                    score -= 0.05

            return max(0, min(1, score))
    """)

    write_file(base_dir / "backend" / "agents" / "orchestrator_agent.py", """
    from crewai import Agent, Task, Crew, Process
    from typing import Dict, List, Optional
    import pandas as pd
    import logging
    from .fundamental_agent import FundamentalAgent
    from .sentiment_agent import SentimentAgent
    from .valuation_agent import ValuationAgent

    logger = logging.getLogger(__name__)

    class OrchestratorAgent:
        '''Master agent that coordinates other agents and makes final decisions'''

        def __init__(self):
            self.fundamental_agent = FundamentalAgent()
            self.sentiment_agent = SentimentAgent()
            self.valuation_agent = ValuationAgent()

            self.orchestrator = Agent(
                role='Portfolio Manager',
                goal='Synthesize all analyses and make investment recommendations',
                backstory='''You are a senior portfolio manager with 20 years of 
                experience. You synthesize fundamental, sentiment, and valuation 
                analyses to make informed investment decisions. You understand 
                how to balance different factors and identify the best opportunities 
                while managing risk.''',
                verbose=True,
                allow_delegation=True
            )

        def analyze_stock_comprehensive(self, symbol: str) -> Dict:
            '''Perform comprehensive analysis using all agents'''
            try:
                # Gather analyses from all agents
                fundamental = self.fundamental_agent.analyze_stock(symbol)
                sentiment = self.sentiment_agent.analyze_sentiment(symbol)
                valuation = self.valuation_agent.analyze_valuation(symbol)

                # Calculate heat score based on all factors
                heat_score = self._calculate_heat_score(fundamental, sentiment, valuation)

                # Create recommendation
                recommendation = self._generate_recommendation(
                    fundamental, sentiment, valuation, heat_score
                )

                return {
                    'symbol': symbol,
                    'fundamental_analysis': fundamental,
                    'sentiment_analysis': sentiment,
                    'valuation_analysis': valuation,
                    'heat_score': heat_score,
                    'recommendation': recommendation,
                    'timestamp': pd.Timestamp.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Error in comprehensive analysis for {symbol}: {str(e)}")
                return {'symbol': symbol, 'error': str(e)}

        def _calculate_heat_score(
            self, 
            fundamental: Dict, 
            sentiment: Dict, 
            valuation: Dict
        ) -> float:
            '''Calculate heat score based on multiple factors'''
            weights = {
                'fundamental': 0.4,
                'sentiment': 0.3,
                'valuation': 0.3
            }

            scores = {
                'fundamental': fundamental.get('fundamental_score', 0.5),
                'sentiment': sentiment.get('sentiment_score', 0.5),
                'valuation': valuation.get('valuation_score', 0.5)
            }

            # Weighted average
            heat_score = sum(
                weights[key] * scores[key] 
                for key in weights
            )

            # Add momentum boost if all signals are positive
            if all(score > 0.6 for score in scores.values()):
                heat_score *= 1.1

            return min(1.0, heat_score)

        def _generate_recommendation(
            self,
            fundamental: Dict,
            sentiment: Dict,
            valuation: Dict,
            heat_score: float
        ) -> Dict:
            '''Generate investment recommendation'''

            # Determine action based on heat score
            if heat_score > 0.7:
                action = "STRONG BUY"
                confidence = "High"
            elif heat_score > 0.6:
                action = "BUY"
                confidence = "Medium"
            elif heat_score > 0.4:
                action = "HOLD"
                confidence = "Medium"
            elif heat_score > 0.3:
                action = "SELL"
                confidence = "Medium"
            else:
                action = "STRONG SELL"
                confidence = "High"

            explanation = self._generate_explanation(
                fundamental, sentiment, valuation, heat_score
            )

            return {
                'action': action,
                'confidence': confidence,
                'heat_score': heat_score,
                'explanation': explanation
            }

        def _generate_explanation(
            self,
            fundamental: Dict,
            sentiment: Dict,
            valuation: Dict,
            heat_score: float
        ) -> str:
            '''Generate human-readable explanation'''

            explanations = []

            # Fundamental explanation
            if fundamental.get('fundamental_score', 0) > 0.6:
                explanations.append("Strong fundamentals with healthy financial metrics")
            elif fundamental.get('fundamental_score', 0) < 0.4:
                explanations.append("Weak fundamentals raise concerns")

            # Sentiment explanation
            sentiment_label = sentiment.get('news_sentiment', {}).get('sentiment_label', 'Neutral')
            explanations.append(f"Market sentiment is {sentiment_label}")

            # Valuation explanation
            if valuation.get('rsi', 50) < 30:
                explanations.append("Technical indicators suggest oversold conditions")
            elif valuation.get('rsi', 50) > 70:
                explanations.append("Technical indicators suggest overbought conditions")

            # Heat score explanation
            explanations.append(f"Overall heat score of {heat_score:.2f} indicates {self._heat_interpretation(heat_score)}")

            return ". ".join(explanations)

        def _heat_interpretation(self, heat_score: float) -> str:
            '''Interpret heat score in human terms'''
            if heat_score > 0.7:
                return "very high momentum and strong opportunity"
            elif heat_score > 0.5:
                return "positive momentum"
            elif heat_score > 0.3:
                return "neutral to slightly negative conditions"
            else:
                return "significant headwinds"
    """)

    # Data Pipeline
    print("\n📡 Creating data pipeline...")

    write_file(base_dir / "backend" / "data_pipeline" / "stream_processor.py", """
    import asyncio
    import json
    from typing import Dict, List, Optional
    from datetime import datetime
    import yfinance as yf
    import logging
    import time

    logger = logging.getLogger(__name__)

    class MarketDataStreamer:
        '''Streams market data and updates the knowledge graph in real-time'''

        def __init__(self, knowledge_graph, heat_engine, orchestrator):
            self.knowledge_graph = knowledge_graph
            self.heat_engine = heat_engine
            self.orchestrator = orchestrator
            self.running = False

            # Stock universe - for demo, using major tech stocks
            self.stock_universe = {
                'Technology': ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA'],
                'Healthcare': ['JNJ', 'PFE', 'UNH'],
                'Finance': ['JPM', 'BAC', 'GS'],
                'Energy': ['XOM', 'CVX'],
                'Consumer_Goods': ['AMZN', 'WMT']
            }

        async def start_streaming(self):
            '''Start the data streaming process'''
            self.running = True

            # Initialize stocks in knowledge graph
            await self._initialize_stocks()

            # Start concurrent tasks
            tasks = [
                asyncio.create_task(self._stream_prices()),
                asyncio.create_task(self._calculate_heat_periodically())
            ]

            await asyncio.gather(*tasks)

        async def _initialize_stocks(self):
            '''Initialize all stocks in the knowledge graph'''
            for sector, stocks in self.stock_universe.items():
                for symbol in stocks:
                    try:
                        stock = yf.Ticker(symbol)
                        info = stock.info

                        attributes = {
                            'symbol': symbol,
                            'name': info.get('longName', symbol),
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0),
                            'volume': info.get('volume', 0),
                            'price': info.get('currentPrice', info.get('regularMarketPrice', 0))
                        }

                        self.knowledge_graph.add_stock(symbol, sector, attributes)
                        logger.info(f"Added {symbol} to {sector}")

                    except Exception as e:
                        logger.error(f"Error initializing {symbol}: {str(e)}")

            logger.info("Stock initialization complete")

        async def _stream_prices(self):
            '''Stream price updates'''
            while self.running:
                try:
                    for sector, stocks in self.stock_universe.items():
                        for symbol in stocks:
                            try:
                                # Get latest price
                                stock = yf.Ticker(symbol)
                                info = stock.info

                                # Update node attributes
                                node_id = f"STOCK_{symbol}"
                                self.knowledge_graph.update_node_attributes(
                                    node_id,
                                    {
                                        'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                                        'volume': info.get('volume', 0),
                                        'last_update': datetime.now().isoformat()
                                    }
                                )
                            except Exception as e:
                                logger.error(f"Error updating {symbol}: {str(e)}")

                    # Wait before next update
                    await asyncio.sleep(60)  # Update every minute

                except Exception as e:
                    logger.error(f"Error streaming prices: {str(e)}")
                    await asyncio.sleep(10)

        async def _calculate_heat_periodically(self):
            '''Periodically recalculate heat distribution'''
            while self.running:
                try:
                    # Get current heat sources (simplified for demo)
                    heat_sources = {}

                    # For demo, assign random heat to some stocks
                    import random
                    all_stocks = [
                        f"STOCK_{stock}"
                        for stocks in self.stock_universe.values() 
                        for stock in stocks
                    ]

                    # Randomly select stocks as heat sources
                    num_sources = min(5, len(all_stocks))
                    selected_stocks = random.sample(all_stocks, num_sources)

                    for stock_id in selected_stocks:
                        heat_sources[stock_id] = random.uniform(0.5, 1.0)

                    # Calculate and propagate heat
                    if heat_sources:
                        heat_distribution = self.heat_engine.calculate_heat_distribution(
                            heat_sources,
                            propagation_steps=5
                        )

                        # Update all nodes with new heat scores
                        for node_id, heat in heat_distribution.items():
                            self.knowledge_graph.heat_scores[node_id] = heat

                        # Log top heated sectors
                        top_sectors = self.heat_engine.get_heated_sectors(top_k=3)
                        logger.info(f"Top heated sectors: {top_sectors}")

                    await asyncio.sleep(30)  # Recalculate every 30 seconds

                except Exception as e:
                    logger.error(f"Error calculating heat: {str(e)}")
                    await asyncio.sleep(60)

        def stop_streaming(self):
            '''Stop the streaming process'''
            self.running = False
    """)

    # API Module
    print("\n🚀 Creating API module...")

    write_file(base_dir / "backend" / "api" / "main.py", '''
    from fastapi import FastAPI, HTTPException, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Dict, List, Optional
    import json
    import asyncio
    from datetime import datetime
    import logging
    import sys
    import os

    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from graph.knowledge_graph import FinancialKnowledgeGraph
    from graph.heat_diffusion import HeatDiffusionEngine
    from agents.orchestrator_agent import OrchestratorAgent
    from data_pipeline.stream_processor import MarketDataStreamer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    app = FastAPI(title="RAGHeat API", version="1.0.0")

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize components
    knowledge_graph = FinancialKnowledgeGraph()
    heat_engine = HeatDiffusionEngine(knowledge_graph.graph)
    orchestrator = OrchestratorAgent()
    streamer = MarketDataStreamer(knowledge_graph, heat_engine, orchestrator)

    # Request/Response Models
    class StockAnalysisRequest(BaseModel):
        symbol: str
        include_heat_map: bool = True

    class HeatMapResponse(BaseModel):
        timestamp: str
        heat_distribution: Dict[str, float]
        top_sectors: List[Dict]
        top_stocks: List[Dict]

    @app.on_event("startup")
    async def startup_event():
        """Initialize the system on startup"""
        # Start data streaming in background
        asyncio.create_task(streamer.start_streaming())
        logger.info("RAGHeat system started")

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "name": "RAGHeat API",
            "version": "1.0.0",
            "status": "running",
            "documentation": "/docs"
        }

    @app.get("/api/graph/structure")
    async def get_graph_structure():
        """Get the current knowledge graph structure"""
        return {
            "graph": json.loads(knowledge_graph.to_json()),
            "node_count": knowledge_graph.graph.number_of_nodes(),
            "edge_count": knowledge_graph.graph.number_of_edges(),
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/api/heat/distribution")
    async def get_heat_distribution():
        """Get current heat distribution across the graph"""

        # Get top heated sectors
        top_sectors = heat_engine.get_heated_sectors(top_k=5)

        # Get top stocks from each heated sector
        top_stocks = []
        for sector_id, sector_heat in top_sectors[:3]:
            stocks = heat_engine.get_heated_stocks_in_sector(sector_id, top_k=3)
            for stock_id, stock_heat in stocks:
                stock_node = knowledge_graph.node_mapping.get(stock_id)
                if stock_node:
                    top_stocks.append({
                        'symbol': stock_node.attributes.get('symbol'),
                        'heat_score': stock_heat,
                        'sector': stock_node.attributes.get('sector')
                    })

        return {
            "timestamp": datetime.now().isoformat(),
            "heat_distribution": knowledge_graph.heat_scores,
            "top_sectors": [
                {'sector': s.replace('SECTOR_', ''), 'heat': h} 
                for s, h in top_sectors
            ],
            "top_stocks": sorted(top_stocks, key=lambda x: x.get('heat_score', 0), reverse=True)
        }

    @app.post("/api/analyze/stock")
    async def analyze_stock(request: StockAnalysisRequest):
        """Analyze a specific stock using multi-agent system"""

        try:
            # Perform comprehensive analysis
            analysis = orchestrator.analyze_stock_comprehensive(request.symbol)

            if 'error' in analysis:
                raise HTTPException(status_code=400, detail=analysis['error'])

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing stock: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/recommendations/top")
    async def get_top_recommendations(limit: int = 10):
        """Get top stock recommendations based on heat scores"""

        recommendations = []

        # Get all stocks and their heat scores
        for node_id, heat_score in knowledge_graph.heat_scores.items():
            if node_id.startswith("STOCK_"):
                node = knowledge_graph.node_mapping.get(node_id)
                if node:
                    symbol = node.attributes.get('symbol')
                    if symbol:
                        recommendations.append({
                            'symbol': symbol,
                            'sector': node.attributes.get('sector'),
                            'heat_score': heat_score,
                            'price': node.attributes.get('price', 0),
                            'market_cap': node.attributes.get('market_cap', 0)
                        })

        # Sort by heat score
        recommendations.sort(key=lambda x: x.get('heat_score', 0), reverse=True)

        return {
            'recommendations': recommendations[:limit],
            'timestamp': datetime.now().isoformat()
        }

    @app.websocket("/ws/heat-updates")
    async def websocket_heat_updates(websocket: WebSocket):
        """WebSocket for real-time heat updates"""
        await websocket.accept()

        try:
            while True:
                # Send heat distribution every 10 seconds
                heat_data = {
                    'type': 'heat_update',
                    'timestamp': datetime.now().isoformat(),
                    'heat_scores': knowledge_graph.heat_scores,
                    'top_sectors': [
                        {'sector': s.replace('SECTOR_', ''), 'heat': h}
                        for s, h in heat_engine.get_heated_sectors(top_k=3)
                    ]
                }

                await websocket.send_json(heat_data)
                await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            await websocket.close()

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    ''')

    # Frontend files
    print("\n🎨 Creating frontend files...")

    write_file(base_dir / "frontend" / "package.json", """
    {
      "name": "ragheat-frontend",
      "version": "1.0.0",
      "private": true,
      "dependencies": {
        "react": "^18.2.0",
        "react-dom": "^18.2.0",
        "react-scripts": "5.0.1",
        "axios": "^1.5.0",
        "d3": "^7.8.5",
        "lucide-react": "^0.263.1",
        "web-vitals": "^3.4.0"
      },
      "scripts": {
        "start": "react-scripts start",
        "build": "react-scripts build",
        "test": "react-scripts test",
        "eject": "react-scripts eject"
      },
      "eslintConfig": {
        "extends": [
          "react-app"
        ]
      },
      "browserslist": {
        "production": [
          ">0.2%",
          "not dead",
          "not op_mini all"
        ],
        "development": [
          "last 1 chrome version",
          "last 1 firefox version",
          "last 1 safari version"
        ]
      }
    }
    """)

    write_file(base_dir / "frontend" / "src" / "App.js", """
    import React from 'react';
    import HeatMapDashboard from './components/HeatMapDashboard';
    import './App.css';

    function App() {
      return (
        <div className="App">
          <header className="App-header">
            <h1>RAGHeat - Real-time Stock Recommendation System</h1>
          </header>
          <main>
            <HeatMapDashboard />
          </main>
        </div>
      );
    }

    export default App;
    """)

    write_file(base_dir / "frontend" / "src" / "components" / "HeatMapDashboard.js", '''
    import React, { useState, useEffect } from 'react';
    import axios from 'axios';

    const API_BASE_URL = 'http://localhost:8000';

    const HeatMapDashboard = () => {
      const [heatData, setHeatData] = useState(null);
      const [recommendations, setRecommendations] = useState([]);
      const [selectedStock, setSelectedStock] = useState(null);
      const [loading, setLoading] = useState(true);
      const [ws, setWs] = useState(null);

      useEffect(() => {
        // Fetch initial data
        fetchHeatDistribution();
        fetchRecommendations();

        // Setup WebSocket connection
        const websocket = new WebSocket('ws://localhost:8000/ws/heat-updates');

        websocket.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'heat_update') {
            setHeatData(data);
          }
        };

        websocket.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

        setWs(websocket);

        return () => {
          if (websocket) {
            websocket.close();
          }
        };
      }, []);

      const fetchHeatDistribution = async () => {
        try {
          const response = await axios.get(`${API_BASE_URL}/api/heat/distribution`);
          setHeatData(response.data);
          setLoading(false);
        } catch (error) {
          console.error('Error fetching heat distribution:', error);
          setLoading(false);
        }
      };

      const fetchRecommendations = async () => {
        try {
          const response = await axios.get(`${API_BASE_URL}/api/recommendations/top`);
          setRecommendations(response.data.recommendations);
        } catch (error) {
          console.error('Error fetching recommendations:', error);
        }
      };

      const analyzeStock = async (symbol) => {
        try {
          const response = await axios.post(`${API_BASE_URL}/api/analyze/stock`, {
            symbol: symbol,
            include_heat_map: true
          });
          setSelectedStock(response.data);
        } catch (error) {
          console.error('Error analyzing stock:', error);
        }
      };

      if (loading) {
        return <div className="loading">Loading RAGHeat System...</div>;
      }

      return (
        <div className="dashboard">
          <div className="dashboard-grid">
            <div className="panel">
              <h2>Top Heated Sectors</h2>
              {heatData?.top_sectors && (
                <ul className="sector-list">
                  {heatData.top_sectors.map((sector, idx) => (
                    <li key={idx} className="sector-item">
                      <span className="sector-name">{sector.sector}</span>
                      <span className="heat-score">{(sector.heat * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="panel">
              <h2>Top Stock Recommendations</h2>
              {recommendations.length > 0 && (
                <ul className="stock-list">
                  {recommendations.slice(0, 10).map((stock, idx) => (
                    <li 
                      key={idx} 
                      className="stock-item"
                      onClick={() => analyzeStock(stock.symbol)}
                    >
                      <span className="stock-symbol">{stock.symbol}</span>
                      <span className="stock-sector">{stock.sector}</span>
                      <span className="heat-score">{(stock.heat_score * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {selectedStock && (
              <div className="panel analysis-panel">
                <h2>Stock Analysis: {selectedStock.symbol}</h2>
                <div className="analysis-content">
                  <div className="metric">
                    <label>Recommendation:</label>
                    <span className={`recommendation ${selectedStock.recommendation?.action}`}>
                      {selectedStock.recommendation?.action}
                    </span>
                  </div>
                  <div className="metric">
                    <label>Heat Score:</label>
                    <span>{(selectedStock.heat_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="metric">
                    <label>Confidence:</label>
                    <span>{selectedStock.recommendation?.confidence}</span>
                  </div>
                  <div className="explanation">
                    <label>Analysis:</label>
                    <p>{selectedStock.recommendation?.explanation}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      );
    };

    export default HeatMapDashboard;
    ''')

    write_file(base_dir / "frontend" / "src" / "App.css", """
    .App {
      text-align: center;
      background-color: #f5f5f5;
      min-height: 100vh;
    }

    .App-header {
      background-color: #282c34;
      padding: 20px;
      color: white;
    }

    .dashboard {
      padding: 20px;
      max-width: 1400px;
      margin: 0 auto;
    }

    .dashboard-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 20px;
    }

    .panel {
      background: white;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .panel h2 {
      margin-top: 0;
      color: #333;
      border-bottom: 2px solid #4CAF50;
      padding-bottom: 10px;
    }

    .sector-list, .stock-list {
      list-style: none;
      padding: 0;
    }

    .sector-item, .stock-item {
      display: flex;
      justify-content: space-between;
      padding: 10px;
      border-bottom: 1px solid #eee;
      cursor: pointer;
      transition: background-color 0.3s;
    }

    .sector-item:hover, .stock-item:hover {
      background-color: #f0f0f0;
    }

    .heat-score {
      color: #ff6b6b;
      font-weight: bold;
    }

    .stock-symbol {
      font-weight: bold;
      color: #2c3e50;
    }

    .stock-sector {
      color: #7f8c8d;
      font-size: 0.9em;
    }

    .analysis-panel {
      grid-column: span 2;
    }

    .analysis-content {
      margin-top: 20px;
    }

    .metric {
      display: flex;
      justify-content: space-between;
      margin-bottom: 15px;
      padding: 10px;
      background-color: #f9f9f9;
      border-radius: 4px;
    }

    .metric label {
      font-weight: bold;
      color: #555;
    }

    .recommendation {
      font-weight: bold;
      padding: 5px 10px;
      border-radius: 4px;
    }

    .recommendation.BUY,
    .recommendation.STRONG.BUY {
      background-color: #4CAF50;
      color: white;
    }

    .recommendation.SELL,
    .recommendation.STRONG.SELL {
      background-color: #f44336;
      color: white;
    }

    .recommendation.HOLD {
      background-color: #FFC107;
      color: white;
    }

    .explanation {
      margin-top: 20px;
      padding: 15px;
      background-color: #f9f9f9;
      border-radius: 4px;
      border-left: 4px solid #4CAF50;
    }

    .explanation label {
      font-weight: bold;
      color: #555;
      display: block;
      margin-bottom: 10px;
    }

    .loading {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      font-size: 1.5em;
      color: #666;
    }
    """)

    write_file(base_dir / "frontend" / "src" / "index.js", """
    import React from 'react';
    import ReactDOM from 'react-dom/client';
    import './index.css';
    import App from './App';

    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(
      <React.StrictMode>
        <App />
      </React.StrictMode>
    );
    """)

    write_file(base_dir / "frontend" / "src" / "index.css", """
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
        sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    code {
      font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
        monospace;
    }
    """)

    write_file(base_dir / "frontend" / "public" / "index.html", """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#000000" />
        <meta name="description" content="RAGHeat - Real-time Stock Recommendation System" />
        <title>RAGHeat</title>
      </head>
      <body>
        <noscript>You need to enable JavaScript to run this app.</noscript>
        <div id="root"></div>
      </body>
    </html>
    """)

    # README
    print("\n📚 Creating documentation...")

    write_file(base_dir / "README.md", """
    # RAGHeat POC - Real-time Stock Recommendation System

    ## 🚀 Quick Start

    ### Prerequisites
    - Python 3.9+
    - Node.js 16+
    - Docker and Docker Compose

    ### Installation & Setup

    1. **Clone and navigate to project:**
```bash
    cd ragheat-poc
    2. **Set up environment variables:**
    # Edit .env file and add your API keys
    ANTHROPIC_API_KEY=your_anthropic_key_here
    ALPHA_VANTAGE_KEY=your_alpha_vantage_key
    FINNHUB_KEY=your_finnhub_key

    3. **Start infrastructure services:**
    docker-compose up -d
    4. **Install Python dependencies:**
    pip install -r requirements.txt
    5. **Start the backend:**
    cd backend
    python -m api.main
    6. **In a new terminal, set up frontend:**
    cd frontend
    npm install
    npm start
    7. **Access the application:**
    - Frontend: http://localhost:3000
    - API Documentation: http://localhost:8000/docs
    - Neo4j Browser: http://localhost:7474
    
    ## 📊 Features
    
    - **Real-time Knowledge Graph**: Hierarchical structure (Root → Sectors → Stocks)
    - **Heat Diffusion Algorithm**: Mathematical model for influence propagation
    - **Multi-Agent System**: Fundamental, Sentiment, and Valuation agents using CrewAI
    - **Real-time Updates**: WebSocket support for live heat distribution
    - **RESTful API**: Comprehensive endpoints with Swagger documentation
    - **Interactive Dashboard**: React-based visualization
    
    ## 🏗️ Architecture
    
    ### Heat Diffusion Model
    The system implements the discrete heat equation on graphs:
    - dh(t)/dt = -βL·h(t)
    - Heat propagates from high-momentum stocks to related entities
    - Multi-hop influence paths are calculated
    
    ### Multi-Agent Collaboration
    - **Fundamental Agent**: Analyzes financial statements and metrics
    - **Sentiment Agent**: Processes news and social media sentiment
    - **Valuation Agent**: Performs technical analysis
    - **Orchestrator**: Synthesizes analyses and generates recommendations
    
    ## 📡 API Endpoints
    
    - `GET /api/graph/structure` - Get knowledge graph structure
    - `GET /api/heat/distribution` - Get current heat distribution
    - `POST /api/analyze/stock` - Analyze specific stock
    - `GET /api/recommendations/top` - Get top recommendations
    - `WS /ws/heat-updates` - Real-time heat updates
    
    ## 🧪 Testing
    # Run backend tests
        pytest backend/tests/
    
        # Run frontend tests
        cd frontend && npm test
        ## 🔧 Configuration
    
    Edit `backend/config/settings.py` to customize:
    - Heat diffusion parameters
    - Update intervals
    - Stock universe
    - API configurations
    
    ## 📈 Production Deployment
    
    1. Use environment variables for sensitive data
    2. Deploy with Docker/Kubernetes
    3. Set up monitoring (Prometheus/Grafana)
    4. Configure rate limiting and authentication
    5. Use production database (PostgreSQL/MongoDB)
    
    ## 🤝 Contributing
    
    1. Fork the repository
    2. Create feature branch
    3. Commit changes
    4. Push to branch
    5. Create Pull Request
    
    ## 📄 License
    
    MIT License
    
    ## 🆘 Support
    
    For issues or questions, please open a GitHub issue.
    """)

    # Create run script
    write_file(base_dir / "run.sh", """
#!/bin/bash

echo "Starting RAGHeat POC..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Start infrastructure
echo "Starting infrastructure services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start backend
echo "Starting backend API..."
cd backend
python -m api.main &
BACKEND_PID=$!

cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install

# Start frontend
echo "Starting frontend..."
npm start &
FRONTEND_PID=$!

echo "RAGHeat is running!"
echo "Frontend: http://localhost:3000"
echo "API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; docker-compose down; exit" INT
wait
""")

    # Make run script executable
    os.chmod(base_dir / "run.sh", 0o755)

    print("\n" + "=" * 60)
    print("✅ RAGHeat POC deployment complete!")
    print("=" * 60)
    print(f"\n📁 Project created at: {base_dir}")
    print("\n🚀 To start the system:")
    print(f"   1. cd {project_name}")
    print("   2. Edit .env file with your API keys")
    print("   3. ./run.sh (Linux/Mac) or follow README instructions")
    print("\n📖 Documentation: README.md")
    print("🌐 API Docs will be at: http://localhost:8000/docs")
    print("\n⚠️  Important: Add your API keys to the .env file before running!")


if __name__ == "__main__":
    deploy_ragheat_project()
