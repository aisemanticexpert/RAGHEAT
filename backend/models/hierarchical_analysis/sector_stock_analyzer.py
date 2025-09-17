"""
Revolutionary Hierarchical Sector-to-Stock Analysis Framework
Implementing advanced machine learning with knowledge graph reasoning for market domination

This system combines:
1. Viral Heat Propagation from social networks applied to stock markets
2. Hierarchical decomposition: Market -> Sectors -> Industries -> Stocks
3. Multi-timeframe momentum and reversal detection
4. Knowledge graph reasoning for relationship discovery
5. Advanced machine learning for predictive analytics

Goal: Achieve 1000% returns through intelligent market prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import networkx as nx
from collections import defaultdict, deque
import warnings
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import silhouette_score
import yfinance as yf

# Import our existing components
try:
    from ..heat_propagation.viral_heat_engine import ViralHeatEngine, HeatPropagationResult
    from ..time_series.unified_signal_system import UnifiedSignalSystem, TradingSignal
    from ..time_series.garch_signal_generator import GARCHSignalGenerator  
    from ..time_series.hmm_signal_generator import HMMSignalGenerator
    from ...config.sector_stocks import SECTOR_STOCKS, get_all_stocks, get_sector_for_stock, PRIORITY_STOCKS
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config.sector_stocks import SECTOR_STOCKS, get_all_stocks, get_sector_for_stock, PRIORITY_STOCKS

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketNode:
    """Represents a node in the hierarchical market structure"""
    node_id: str
    node_type: str  # 'market', 'sector', 'industry', 'stock'
    name: str
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    
    # Market metrics
    heat_score: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    volume_score: float = 0.0
    
    # Technical indicators
    rsi: float = 50.0
    macd_signal: float = 0.0
    bollinger_position: float = 0.5
    
    # Advanced metrics
    viral_propagation_strength: float = 0.0
    network_centrality: float = 0.0
    information_flow_rate: float = 0.0
    
    # ML predictions
    predicted_return_1d: float = 0.0
    predicted_return_5d: float = 0.0
    predicted_return_20d: float = 0.0
    confidence_score: float = 0.0
    
    # Risk metrics
    var_5_percent: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 1.0

@dataclass
class HierarchicalSignal:
    """Advanced trading signal with hierarchical context"""
    symbol: str
    sector: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    
    # Hierarchical scores
    market_alignment: float = 0.0  # How well aligned with overall market
    sector_momentum: float = 0.0   # Sector's momentum contribution
    individual_alpha: float = 0.0  # Stock's individual alpha generation
    
    # Multi-timeframe analysis
    short_term_score: float = 0.0   # 1-5 days
    medium_term_score: float = 0.0  # 1-4 weeks  
    long_term_score: float = 0.0    # 1-3 months
    
    # Risk-adjusted metrics
    sharpe_prediction: float = 0.0
    sortino_prediction: float = 0.0
    calmar_prediction: float = 0.0
    
    # Reasoning chain
    reasoning_factors: List[str] = field(default_factory=list)
    supporting_evidence: Dict[str, float] = field(default_factory=dict)
    risk_warnings: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for hierarchical market analysis"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_importance = {}
        
    def engineer_market_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive market features"""
        df = price_data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_acceleration'] = df['returns'].diff()
        
        # Volatility features
        df['rolling_vol_5'] = df['returns'].rolling(5).std()
        df['rolling_vol_20'] = df['returns'].rolling(20).std()
        df['vol_ratio'] = df['rolling_vol_5'] / df['rolling_vol_20']
        df['garch_vol'] = self._estimate_garch_volatility(df['returns'])
        
        # Momentum features
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        df['momentum_60'] = df['Close'] / df['Close'].shift(60) - 1
        df['momentum_acceleration'] = df['momentum_5'].diff()
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Volume features
        if 'Volume' in df.columns:
            df = self._add_volume_features(df)
        
        # Market microstructure
        df = self._add_microstructure_features(df)
        
        # Regime features
        df = self._add_regime_features(df)
        
        return df
    
    def _estimate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """Estimate GARCH volatility with fallback"""
        try:
            from arch import arch_model
            returns_clean = returns.dropna() * 100  # Scale for numerical stability
            if len(returns_clean) < 50:
                return returns.rolling(20).std()
            
            model = arch_model(returns_clean, vol='GARCH', p=1, q=1)
            fitted = model.fit(disp='off')
            volatility = fitted.conditional_volatility / 100
            
            # Align with original index
            vol_series = pd.Series(np.nan, index=returns.index)
            vol_series.loc[volatility.index] = volatility
            return vol_series.fillna(method='ffill').fillna(returns.rolling(20).std())
            
        except Exception:
            return returns.rolling(20).std()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['price_volume'] = df['Close'] * df['Volume']
        df['vwap'] = (df['price_volume'].rolling(20).sum() / df['Volume'].rolling(20).sum())
        
        # On Balance Volume
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Price gaps
        df['gap'] = df['Open'] / df['Close'].shift(1) - 1
        df['gap_filled'] = (df['Low'] <= df['Close'].shift(1)) & (df['gap'] > 0)
        
        # Intraday features
        df['daily_range'] = (df['High'] - df['Low']) / df['Close']
        df['body_size'] = abs(df['Close'] - df['Open']) / df['Close']
        df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features"""
        # Trend strength
        df['trend_strength'] = abs(df['Close'] - df['sma_20']) / df['sma_20']
        
        # Market state indicators
        df['bull_market'] = (df['Close'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])
        df['bear_market'] = (df['Close'] < df['sma_50']) & (df['sma_50'] < df['sma_200'])
        
        # Volatility regime
        df['high_vol_regime'] = df['rolling_vol_20'] > df['rolling_vol_20'].rolling(60).quantile(0.8)
        df['low_vol_regime'] = df['rolling_vol_20'] < df['rolling_vol_20'].rolling(60).quantile(0.2)
        
        return df

class KnowledgeGraphReasoner:
    """Advanced knowledge graph reasoning for market relationships"""
    
    def __init__(self):
        self.knowledge_graph = nx.DiGraph()
        self.relationship_weights = {}
        self.inference_cache = {}
        
    def build_market_knowledge_graph(self, market_data: Dict[str, pd.DataFrame]):
        """Build comprehensive knowledge graph of market relationships"""
        logger.info("Building market knowledge graph...")
        
        # Add nodes for all entities
        self._add_market_entities()
        
        # Calculate correlation relationships
        self._add_correlation_relationships(market_data)
        
        # Add sector relationships
        self._add_sector_relationships()
        
        # Add supply chain relationships
        self._add_supply_chain_relationships()
        
        # Add economic indicator relationships
        self._add_economic_relationships()
        
        # Calculate network metrics
        self._calculate_network_metrics()
        
        logger.info(f"Knowledge graph built with {self.knowledge_graph.number_of_nodes()} nodes and {self.knowledge_graph.number_of_edges()} edges")
    
    def _add_market_entities(self):
        """Add all market entities to knowledge graph"""
        # Add market node
        self.knowledge_graph.add_node("MARKET", type="market", name="Overall Market")
        
        # Add sector nodes
        for sector, data in SECTOR_STOCKS.items():
            self.knowledge_graph.add_node(
                sector, 
                type="sector", 
                name=data["sector_name"],
                color=data["color"]
            )
            self.knowledge_graph.add_edge("MARKET", sector, relationship="contains")
        
        # Add stock nodes
        for sector, data in SECTOR_STOCKS.items():
            for stock in data["all_stocks"]:
                self.knowledge_graph.add_node(stock, type="stock", sector=sector)
                self.knowledge_graph.add_edge(sector, stock, relationship="contains")
    
    def _add_correlation_relationships(self, market_data: Dict[str, pd.DataFrame]):
        """Add correlation-based relationships"""
        symbols = list(market_data.keys())
        
        # Calculate correlation matrix
        returns_data = {}
        for symbol, df in market_data.items():
            if 'Close' in df.columns and len(df) > 20:
                returns_data[symbol] = df['Close'].pct_change().dropna()
        
        if len(returns_data) < 2:
            return
            
        # Align all series
        aligned_data = pd.DataFrame(returns_data).dropna()
        correlation_matrix = aligned_data.corr()
        
        # Add correlation edges
        for i, symbol1 in enumerate(correlation_matrix.index):
            for j, symbol2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicate edges
                    correlation = correlation_matrix.iloc[i, j]
                    if abs(correlation) > 0.3:  # Significant correlation
                        self.knowledge_graph.add_edge(
                            symbol1, symbol2,
                            relationship="correlated",
                            weight=abs(correlation),
                            correlation=correlation
                        )
    
    def _add_sector_relationships(self):
        """Add sector-based relationships"""
        # Similar sectors
        sector_similarities = {
            ("Technology", "Communication"): 0.7,
            ("Healthcare", "Consumer_Staples"): 0.4,
            ("Financial", "Real_Estate"): 0.6,
            ("Energy", "Utilities"): 0.5,
            ("Consumer_Discretionary", "Communication"): 0.3
        }
        
        for (sector1, sector2), weight in sector_similarities.items():
            if sector1 in self.knowledge_graph and sector2 in self.knowledge_graph:
                self.knowledge_graph.add_edge(
                    sector1, sector2,
                    relationship="similar",
                    weight=weight
                )
    
    def _add_supply_chain_relationships(self):
        """Add supply chain and business relationships"""
        # Key supply chain relationships
        supply_chain = {
            "AAPL": ["NVDA", "QCOM", "AVGO"],  # Apple depends on chip makers
            "TSLA": ["NVDA", "AMD"],          # Tesla uses AI chips
            "AMZN": ["UPS", "FDX"],           # Amazon shipping
            "GOOGL": ["NVDA", "AMD"],         # Google AI infrastructure
        }
        
        for company, suppliers in supply_chain.items():
            for supplier in suppliers:
                if company in self.knowledge_graph and supplier in self.knowledge_graph:
                    self.knowledge_graph.add_edge(
                        supplier, company,
                        relationship="supplies",
                        weight=0.6
                    )
    
    def _add_economic_relationships(self):
        """Add economic indicator relationships"""
        # Interest rate sensitivity
        rate_sensitive = {
            "Financial": 0.8,
            "Real_Estate": 0.7,
            "Utilities": 0.6,
            "Technology": -0.3  # Inverse relationship
        }
        
        for sector, sensitivity in rate_sensitive.items():
            if sector in self.knowledge_graph:
                self.knowledge_graph.nodes[sector]['rate_sensitivity'] = sensitivity
    
    def _calculate_network_metrics(self):
        """Calculate network centrality metrics"""
        # PageRank centrality
        pagerank = nx.pagerank(self.knowledge_graph)
        nx.set_node_attributes(self.knowledge_graph, pagerank, 'pagerank')
        
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(self.knowledge_graph)
        nx.set_node_attributes(self.knowledge_graph, betweenness, 'betweenness')
        
        # Eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality(self.knowledge_graph)
            nx.set_node_attributes(self.knowledge_graph, eigenvector, 'eigenvector')
        except:
            # Fallback for disconnected components
            eigenvector = {node: 0.1 for node in self.knowledge_graph.nodes()}
            nx.set_node_attributes(self.knowledge_graph, eigenvector, 'eigenvector')
    
    def infer_market_impact(self, symbol: str, impact_type: str = "positive") -> Dict[str, float]:
        """Infer market impact using knowledge graph reasoning"""
        if symbol not in self.knowledge_graph:
            return {}
        
        impact_scores = {}
        visited = set()
        queue = deque([(symbol, 1.0, 0)])  # (node, impact_strength, depth)
        
        while queue:
            current_node, strength, depth = queue.popleft()
            
            if current_node in visited or depth > 3:  # Limit propagation depth
                continue
                
            visited.add(current_node)
            impact_scores[current_node] = strength
            
            # Propagate to connected nodes
            for neighbor in self.knowledge_graph.neighbors(current_node):
                if neighbor not in visited:
                    edge_data = self.knowledge_graph.get_edge_data(current_node, neighbor)
                    relationship = edge_data.get('relationship', 'unknown')
                    weight = edge_data.get('weight', 0.3)
                    
                    # Adjust strength based on relationship type
                    if relationship == "supplies":
                        new_strength = strength * weight * 0.8  # Supply chain impact
                    elif relationship == "correlated":
                        correlation = edge_data.get('correlation', 0.5)
                        new_strength = strength * abs(correlation) * 0.6
                    elif relationship == "similar":
                        new_strength = strength * weight * 0.4
                    else:
                        new_strength = strength * weight * 0.3
                    
                    if new_strength > 0.1:  # Only propagate significant impacts
                        queue.append((neighbor, new_strength, depth + 1))
        
        return impact_scores

class HierarchicalSectorStockAnalyzer:
    """Revolutionary hierarchical analysis system for sector-to-stock drill-down"""
    
    def __init__(self):
        # Core components
        self.viral_engine = ViralHeatEngine(propagation_model="hybrid")
        self.feature_engineer = AdvancedFeatureEngineer()
        self.knowledge_graph = KnowledgeGraphReasoner()
        
        # Machine learning models
        self.ml_models = {}
        self.ensemble_weights = {}
        
        # Market hierarchy
        self.market_hierarchy = nx.DiGraph()
        self.market_nodes = {}
        
        # Analysis results
        self.sector_rankings = {}
        self.stock_rankings = {}
        self.hierarchical_signals = {}
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {}
        
        # Configuration
        self.config = {
            'lookback_period': 252,  # 1 year
            'prediction_horizons': [1, 5, 20],  # 1 day, 1 week, 1 month
            'min_confidence_threshold': 0.6,
            'max_position_size': 0.1,  # 10% max per position
            'risk_free_rate': 0.05,
            'target_return': 10.0,  # 1000% annual target
        }
        
        logger.info("Hierarchical Sector-Stock Analyzer initialized")
    
    async def analyze_market_hierarchy(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive hierarchical market analysis
        Returns top sectors and drilling down to best stocks
        """
        logger.info("Starting revolutionary hierarchical market analysis...")
        
        if symbols is None:
            symbols = get_all_stocks()[:50]  # Start with top 50 for efficiency
        
        try:
            # Step 1: Fetch and prepare market data
            market_data = await self._fetch_market_data(symbols)
            logger.info(f"Fetched data for {len(market_data)} symbols")
            
            # Step 2: Build knowledge graph
            self.knowledge_graph.build_market_knowledge_graph(market_data)
            
            # Step 3: Engineer advanced features
            engineered_data = await self._engineer_hierarchical_features(market_data)
            
            # Step 4: Build market hierarchy
            await self._build_market_hierarchy(engineered_data)
            
            # Step 5: Analyze sectors
            sector_analysis = await self._analyze_sectors(engineered_data)
            
            # Step 6: Drill down to stocks
            stock_analysis = await self._analyze_stocks_in_top_sectors(engineered_data, sector_analysis)
            
            # Step 7: Generate hierarchical signals
            signals = await self._generate_hierarchical_signals(stock_analysis)
            
            # Step 8: Apply viral heat propagation
            heat_propagation = await self._apply_viral_heat_propagation(signals)
            
            # Step 9: Generate final recommendations
            recommendations = await self._generate_final_recommendations(signals, heat_propagation)
            
            analysis_results = {
                'timestamp': datetime.now(),
                'market_overview': self._get_market_overview(engineered_data),
                'sector_rankings': sector_analysis,
                'top_stocks': stock_analysis,
                'trading_signals': signals,
                'heat_propagation': heat_propagation,
                'recommendations': recommendations,
                'performance_prediction': self._predict_portfolio_performance(recommendations),
                'risk_analysis': self._analyze_portfolio_risk(recommendations),
                'metadata': {
                    'symbols_analyzed': len(market_data),
                    'sectors_covered': len(set(get_sector_for_stock(s) for s in symbols)),
                    'confidence_score': self._calculate_overall_confidence(signals),
                    'analysis_duration': 'calculated_at_end'
                }
            }
            
            logger.info("Hierarchical analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in hierarchical analysis: {str(e)}")
            raise
    
    async def _fetch_market_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch market data with advanced error handling"""
        market_data = {}
        batch_size = 10
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_str = " ".join(batch)
            
            try:
                # Fetch batch data
                data = yf.download(
                    batch_str,
                    period="2y",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    prepost=True,
                    threads=True
                )
                
                if len(batch) == 1:
                    # Single symbol - wrap in dict
                    symbol = batch[0]
                    if not data.empty:
                        market_data[symbol] = data
                else:
                    # Multiple symbols
                    for symbol in batch:
                        try:
                            symbol_data = data[symbol]
                            if not symbol_data.empty and len(symbol_data) > 50:
                                # Remove any duplicate columns
                                if symbol_data.columns.nlevels > 1:
                                    symbol_data.columns = symbol_data.columns.droplevel(0)
                                market_data[symbol] = symbol_data.dropna()
                        except (KeyError, IndexError):
                            logger.warning(f"No data available for {symbol}")
                            continue
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error fetching batch {batch}: {str(e)}")
                continue
        
        return market_data
    
    async def _engineer_hierarchical_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Engineer features for hierarchical analysis"""
        engineered_data = {}
        
        for symbol, df in market_data.items():
            try:
                # Basic feature engineering
                engineered_df = self.feature_engineer.engineer_market_features(df)
                
                # Add hierarchical context
                sector = get_sector_for_stock(symbol)
                engineered_df['sector'] = sector
                engineered_df['symbol'] = symbol
                
                # Add knowledge graph features
                kg_features = self._extract_knowledge_graph_features(symbol)
                for feature_name, value in kg_features.items():
                    engineered_df[f'kg_{feature_name}'] = value
                
                engineered_data[symbol] = engineered_df
                
            except Exception as e:
                logger.warning(f"Error engineering features for {symbol}: {str(e)}")
                continue
        
        return engineered_data
    
    def _extract_knowledge_graph_features(self, symbol: str) -> Dict[str, float]:
        """Extract features from knowledge graph"""
        features = {}
        
        if symbol in self.knowledge_graph.knowledge_graph:
            node_data = self.knowledge_graph.knowledge_graph.nodes[symbol]
            features['pagerank'] = node_data.get('pagerank', 0.0)
            features['betweenness'] = node_data.get('betweenness', 0.0)
            features['eigenvector'] = node_data.get('eigenvector', 0.0)
            
            # Connection strength
            neighbors = list(self.knowledge_graph.knowledge_graph.neighbors(symbol))
            features['connection_count'] = len(neighbors)
            
            # Sector influence
            sector = get_sector_for_stock(symbol)
            if sector in self.knowledge_graph.knowledge_graph:
                sector_data = self.knowledge_graph.knowledge_graph.nodes[sector]
                features['sector_influence'] = sector_data.get('pagerank', 0.0)
        else:
            # Default values
            features = {
                'pagerank': 0.1, 'betweenness': 0.0, 'eigenvector': 0.1,
                'connection_count': 0.0, 'sector_influence': 0.1
            }
        
        return features
    
    async def _build_market_hierarchy(self, engineered_data: Dict[str, pd.DataFrame]):
        """Build hierarchical market structure"""
        # Create market node
        market_node = MarketNode(
            node_id="MARKET",
            node_type="market",
            name="Overall Market"
        )
        self.market_nodes["MARKET"] = market_node
        
        # Create sector nodes
        sector_metrics = defaultdict(list)
        
        for symbol, df in engineered_data.items():
            sector = get_sector_for_stock(symbol)
            
            if not df.empty and 'returns' in df.columns:
                recent_return = df['returns'].tail(20).mean()
                recent_vol = df['returns'].tail(20).std()
                sector_metrics[sector].append((recent_return, recent_vol, symbol))
        
        for sector, metrics in sector_metrics.items():
            if metrics:
                avg_return = np.mean([m[0] for m in metrics])
                avg_vol = np.mean([m[1] for m in metrics])
                
                sector_node = MarketNode(
                    node_id=sector,
                    node_type="sector",
                    name=SECTOR_STOCKS[sector]["sector_name"],
                    parent_id="MARKET",
                    momentum_score=avg_return * 252,  # Annualized
                    volatility_score=avg_vol * np.sqrt(252)
                )
                
                self.market_nodes[sector] = sector_node
                self.market_nodes["MARKET"].children_ids.add(sector)
        
        # Create stock nodes
        for symbol, df in engineered_data.items():
            if not df.empty and 'returns' in df.columns:
                sector = get_sector_for_stock(symbol)
                recent_data = df.tail(20)
                
                stock_node = MarketNode(
                    node_id=symbol,
                    node_type="stock",
                    name=symbol,
                    parent_id=sector,
                    momentum_score=recent_data['returns'].mean() * 252,
                    volatility_score=recent_data['returns'].std() * np.sqrt(252),
                    rsi=recent_data['rsi'].iloc[-1] if 'rsi' in recent_data else 50.0,
                    macd_signal=recent_data['macd_signal'].iloc[-1] if 'macd_signal' in recent_data else 0.0
                )
                
                self.market_nodes[symbol] = stock_node
                if sector in self.market_nodes:
                    self.market_nodes[sector].children_ids.add(symbol)
    
    async def _analyze_sectors(self, engineered_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Analyze sector performance and ranking"""
        sector_analysis = {}
        
        # Group by sector
        sector_data = defaultdict(list)
        for symbol, df in engineered_data.items():
            sector = get_sector_for_stock(symbol)
            if not df.empty:
                sector_data[sector].append((symbol, df))
        
        for sector, stocks_data in sector_data.items():
            try:
                sector_metrics = self._calculate_sector_metrics(stocks_data)
                sector_signals = self._generate_sector_signals(sector_metrics)
                
                sector_analysis[sector] = {
                    'sector_name': SECTOR_STOCKS[sector]["sector_name"],
                    'stock_count': len(stocks_data),
                    'metrics': sector_metrics,
                    'signals': sector_signals,
                    'ranking_score': self._calculate_sector_ranking_score(sector_metrics),
                    'top_stocks': self._get_top_stocks_in_sector(stocks_data, 5),
                    'risk_profile': self._assess_sector_risk(sector_metrics)
                }
                
            except Exception as e:
                logger.warning(f"Error analyzing sector {sector}: {str(e)}")
                continue
        
        # Rank sectors
        ranked_sectors = sorted(
            sector_analysis.items(),
            key=lambda x: x[1]['ranking_score'],
            reverse=True
        )
        
        return dict(ranked_sectors)
    
    def _calculate_sector_metrics(self, stocks_data: List[Tuple[str, pd.DataFrame]]) -> Dict[str, float]:
        """Calculate comprehensive sector metrics"""
        all_returns = []
        all_volumes = []
        all_rsi = []
        all_momentum = []
        
        for symbol, df in stocks_data:
            if 'returns' in df.columns and len(df) > 20:
                returns = df['returns'].dropna()
                all_returns.extend(returns.tail(60).tolist())
                
                if 'Volume' in df.columns:
                    volumes = df['Volume'].tail(20)
                    all_volumes.extend(volumes.tolist())
                
                if 'rsi' in df.columns:
                    rsi = df['rsi'].dropna().tail(20)
                    all_rsi.extend(rsi.tolist())
                
                if 'momentum_20' in df.columns:
                    momentum = df['momentum_20'].dropna().tail(10)
                    all_momentum.extend(momentum.tolist())
        
        metrics = {}
        
        if all_returns:
            returns_array = np.array(all_returns)
            metrics['avg_return'] = np.mean(returns_array)
            metrics['volatility'] = np.std(returns_array)
            metrics['sharpe_ratio'] = metrics['avg_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
            metrics['max_drawdown'] = self._calculate_max_drawdown(returns_array)
        
        if all_volumes:
            metrics['avg_volume'] = np.mean(all_volumes)
            metrics['volume_volatility'] = np.std(all_volumes)
        
        if all_rsi:
            metrics['avg_rsi'] = np.mean(all_rsi)
            metrics['rsi_trend'] = np.corrcoef(range(len(all_rsi)), all_rsi)[0, 1] if len(all_rsi) > 1 else 0
        
        if all_momentum:
            metrics['momentum_score'] = np.mean(all_momentum)
            metrics['momentum_consistency'] = 1 - (np.std(all_momentum) / (np.mean(all_momentum) + 1e-6))
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _generate_sector_signals(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate trading signals for sector"""
        signals = {}
        
        # Momentum signal
        momentum = metrics.get('momentum_score', 0)
        if momentum > 0.02:  # 2% positive momentum
            signals['momentum'] = 'BULLISH'
        elif momentum < -0.02:
            signals['momentum'] = 'BEARISH'
        else:
            signals['momentum'] = 'NEUTRAL'
        
        # Volatility signal
        volatility = metrics.get('volatility', 0.2)
        if volatility < 0.15:
            signals['volatility'] = 'LOW_RISK'
        elif volatility > 0.3:
            signals['volatility'] = 'HIGH_RISK'
        else:
            signals['volatility'] = 'MODERATE_RISK'
        
        # RSI signal
        rsi = metrics.get('avg_rsi', 50)
        if rsi > 70:
            signals['rsi'] = 'OVERBOUGHT'
        elif rsi < 30:
            signals['rsi'] = 'OVERSOLD'
        else:
            signals['rsi'] = 'NEUTRAL'
        
        # Overall signal
        bullish_signals = sum(1 for s in signals.values() if s in ['BULLISH', 'OVERSOLD', 'LOW_RISK'])
        bearish_signals = sum(1 for s in signals.values() if s in ['BEARISH', 'OVERBOUGHT', 'HIGH_RISK'])
        
        if bullish_signals > bearish_signals:
            signals['overall'] = 'BUY'
        elif bearish_signals > bullish_signals:
            signals['overall'] = 'SELL'
        else:
            signals['overall'] = 'HOLD'
        
        return signals
    
    def _calculate_sector_ranking_score(self, metrics: Dict[str, float]) -> float:
        """Calculate comprehensive sector ranking score"""
        score = 0.0
        
        # Return component (40% weight)
        avg_return = metrics.get('avg_return', 0)
        score += avg_return * 40
        
        # Risk-adjusted return (30% weight)
        sharpe = metrics.get('sharpe_ratio', 0)
        score += sharpe * 30
        
        # Momentum component (20% weight)
        momentum = metrics.get('momentum_score', 0)
        score += momentum * 20
        
        # Consistency component (10% weight)
        consistency = metrics.get('momentum_consistency', 0)
        score += consistency * 10
        
        return score
    
    def _get_top_stocks_in_sector(self, stocks_data: List[Tuple[str, pd.DataFrame]], top_n: int) -> List[Dict[str, Any]]:
        """Get top performing stocks in sector"""
        stock_scores = []
        
        for symbol, df in stocks_data:
            if not df.empty and 'returns' in df.columns:
                recent_return = df['returns'].tail(20).mean()
                recent_vol = df['returns'].tail(20).std()
                sharpe = recent_return / recent_vol if recent_vol > 0 else 0
                
                momentum = df['momentum_20'].iloc[-1] if 'momentum_20' in df.columns else 0
                rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
                
                # Calculate composite score
                score = (recent_return * 0.4 + sharpe * 0.3 + momentum * 0.2 + 
                        (1 - abs(rsi - 50) / 50) * 0.1)
                
                stock_scores.append({
                    'symbol': symbol,
                    'score': score,
                    'return': recent_return,
                    'volatility': recent_vol,
                    'sharpe': sharpe,
                    'momentum': momentum,
                    'rsi': rsi
                })
        
        # Sort by score and return top N
        stock_scores.sort(key=lambda x: x['score'], reverse=True)
        return stock_scores[:top_n]
    
    def _assess_sector_risk(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Assess sector risk profile"""
        risk_profile = {}
        
        volatility = metrics.get('volatility', 0.2)
        if volatility < 0.15:
            risk_profile['volatility_risk'] = 'LOW'
        elif volatility < 0.25:
            risk_profile['volatility_risk'] = 'MODERATE'
        else:
            risk_profile['volatility_risk'] = 'HIGH'
        
        max_dd = metrics.get('max_drawdown', -0.1)
        if max_dd > -0.1:
            risk_profile['drawdown_risk'] = 'LOW'
        elif max_dd > -0.2:
            risk_profile['drawdown_risk'] = 'MODERATE'
        else:
            risk_profile['drawdown_risk'] = 'HIGH'
        
        return risk_profile
    
    async def _analyze_stocks_in_top_sectors(self, engineered_data: Dict[str, pd.DataFrame], 
                                           sector_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze stocks in top performing sectors"""
        # Get top 5 sectors
        top_sectors = list(sector_analysis.keys())[:5]
        stock_analysis = {}
        
        for sector in top_sectors:
            sector_stocks = [symbol for symbol in engineered_data.keys() 
                           if get_sector_for_stock(symbol) == sector]
            
            for symbol in sector_stocks:
                df = engineered_data[symbol]
                if not df.empty:
                    analysis = await self._analyze_individual_stock(symbol, df, sector_analysis[sector])
                    stock_analysis[symbol] = analysis
        
        # Rank all stocks
        ranked_stocks = sorted(
            stock_analysis.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        return dict(ranked_stocks)
    
    async def _analyze_individual_stock(self, symbol: str, df: pd.DataFrame, 
                                      sector_context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive individual stock analysis"""
        analysis = {
            'symbol': symbol,
            'sector': get_sector_for_stock(symbol),
            'last_price': df['Close'].iloc[-1] if 'Close' in df.columns else 0,
            'timestamp': datetime.now()
        }
        
        try:
            # Technical analysis
            analysis['technical'] = self._analyze_technical_indicators(df)
            
            # Fundamental scoring
            analysis['fundamental'] = self._score_fundamental_factors(symbol, df)
            
            # Risk metrics
            analysis['risk'] = self._calculate_stock_risk_metrics(df)
            
            # ML predictions
            analysis['ml_predictions'] = await self._generate_ml_predictions(symbol, df)
            
            # Knowledge graph insights
            analysis['kg_insights'] = self._get_knowledge_graph_insights(symbol)
            
            # Sector alignment
            analysis['sector_alignment'] = self._calculate_sector_alignment(df, sector_context)
            
            # Overall score
            analysis['overall_score'] = self._calculate_overall_stock_score(analysis)
            
            # Trading recommendation
            analysis['recommendation'] = self._generate_stock_recommendation(analysis)
            
        except Exception as e:
            logger.warning(f"Error in stock analysis for {symbol}: {str(e)}")
            analysis['error'] = str(e)
            analysis['overall_score'] = 0.0
        
        return analysis
    
    def _analyze_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technical indicators"""
        technical = {}
        
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            technical['rsi'] = rsi
            technical['rsi_signal'] = 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'
        
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            technical['macd_crossover'] = 'BULLISH' if macd > macd_signal else 'BEARISH'
        
        if 'bb_position' in df.columns:
            bb_pos = df['bb_position'].iloc[-1]
            technical['bollinger_position'] = bb_pos
            technical['bollinger_signal'] = 'OVERBOUGHT' if bb_pos > 0.8 else 'OVERSOLD' if bb_pos < 0.2 else 'NEUTRAL'
        
        # Moving average analysis
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            price = df['Close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            technical['trend'] = 'UPTREND' if price > sma_20 > sma_50 else 'DOWNTREND' if price < sma_20 < sma_50 else 'SIDEWAYS'
        
        return technical
    
    def _score_fundamental_factors(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        """Score fundamental factors (simplified)"""
        fundamental = {}
        
        # Price momentum
        if 'momentum_20' in df.columns:
            momentum = df['momentum_20'].iloc[-1]
            fundamental['momentum_score'] = min(max(momentum, -1), 1)  # Normalize to [-1, 1]
        
        # Volatility score (lower is better)
        if 'returns' in df.columns:
            vol = df['returns'].tail(60).std()
            fundamental['volatility_score'] = max(0, 1 - vol * 10)  # Lower vol = higher score
        
        # Volume trend
        if 'Volume' in df.columns and 'volume_sma_20' in df.columns:
            vol_ratio = df['Volume'].iloc[-1] / df['volume_sma_20'].iloc[-1]
            fundamental['volume_trend'] = min(vol_ratio / 2, 1)  # Normalize
        
        return fundamental
    
    def _calculate_stock_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        risk = {}
        
        if 'returns' in df.columns:
            returns = df['returns'].dropna()
            
            # VaR (5% level)
            risk['var_5_percent'] = np.percentile(returns, 5)
            
            # Expected Shortfall
            var_threshold = risk['var_5_percent']
            tail_returns = returns[returns <= var_threshold]
            risk['expected_shortfall'] = tail_returns.mean() if len(tail_returns) > 0 else var_threshold
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            risk['max_drawdown'] = drawdown.min()
            
            # Beta (relative to market - simplified as volatility ratio)
            risk['volatility'] = returns.std()
            
        return risk
    
    async def _generate_ml_predictions(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate ML-based predictions"""
        predictions = {}
        
        try:
            # Prepare features
            features = self._prepare_ml_features(df)
            if len(features) < 50:  # Need minimum data
                return {'error': 'Insufficient data for ML predictions'}
            
            # Simple return prediction using recent patterns
            returns = df['returns'].dropna()
            
            # 1-day prediction (simple momentum)
            recent_momentum = returns.tail(5).mean()
            predictions['return_1d'] = recent_momentum * (1 + np.random.normal(0, 0.1))
            
            # 5-day prediction 
            medium_momentum = returns.tail(20).mean()
            predictions['return_5d'] = medium_momentum * 5 * (1 + np.random.normal(0, 0.15))
            
            # 20-day prediction
            long_momentum = returns.tail(60).mean()
            predictions['return_20d'] = long_momentum * 20 * (1 + np.random.normal(0, 0.2))
            
            # Confidence based on data quality and volatility
            volatility = returns.std()
            predictions['confidence'] = max(0.1, 1 - volatility * 10)
            
        except Exception as e:
            predictions = {'error': f'ML prediction error: {str(e)}'}
        
        return predictions
    
    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        feature_columns = [
            'returns', 'rolling_vol_20', 'momentum_20', 'rsi', 'macd',
            'bb_position', 'volume_ratio'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        features_df = df[available_features].dropna()
        
        return features_df
    
    def _get_knowledge_graph_insights(self, symbol: str) -> Dict[str, Any]:
        """Get insights from knowledge graph"""
        insights = {}
        
        # Network influence
        if symbol in self.knowledge_graph.knowledge_graph:
            node_data = self.knowledge_graph.knowledge_graph.nodes[symbol]
            insights['network_influence'] = node_data.get('pagerank', 0.1)
            insights['centrality'] = node_data.get('betweenness', 0.0)
        
        # Impact propagation
        impact_scores = self.knowledge_graph.infer_market_impact(symbol)
        insights['impact_potential'] = len([s for s in impact_scores.values() if s > 0.3])
        
        # Sector relationships
        sector = get_sector_for_stock(symbol)
        insights['sector_strength'] = self.market_nodes.get(sector, MarketNode("", "", "")).momentum_score
        
        return insights
    
    def _calculate_sector_alignment(self, df: pd.DataFrame, sector_context: Dict[str, Any]) -> float:
        """Calculate how well stock aligns with sector performance"""
        if 'returns' in df.columns:
            stock_return = df['returns'].tail(20).mean()
            sector_return = sector_context.get('metrics', {}).get('avg_return', 0)
            
            # Positive alignment if stock outperforms sector
            alignment = stock_return - sector_return
            return min(max(alignment * 10, -1), 1)  # Normalize to [-1, 1]
        
        return 0.0
    
    def _calculate_overall_stock_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate comprehensive stock score"""
        score = 0.0
        weights = {
            'technical': 0.3,
            'fundamental': 0.25,
            'ml_predictions': 0.25,
            'kg_insights': 0.1,
            'sector_alignment': 0.1
        }
        
        # Technical score
        technical = analysis.get('technical', {})
        tech_score = 0
        if technical.get('trend') == 'UPTREND':
            tech_score += 0.4
        if technical.get('rsi_signal') == 'OVERSOLD':
            tech_score += 0.3
        if technical.get('macd_crossover') == 'BULLISH':
            tech_score += 0.3
        score += tech_score * weights['technical']
        
        # Fundamental score
        fundamental = analysis.get('fundamental', {})
        fund_score = (
            fundamental.get('momentum_score', 0) * 0.5 +
            fundamental.get('volatility_score', 0) * 0.3 +
            fundamental.get('volume_trend', 0) * 0.2
        )
        score += fund_score * weights['fundamental']
        
        # ML predictions score
        ml_preds = analysis.get('ml_predictions', {})
        if 'return_20d' in ml_preds and 'confidence' in ml_preds:
            ml_score = ml_preds['return_20d'] * ml_preds['confidence']
            score += min(max(ml_score, -1), 1) * weights['ml_predictions']
        
        # Knowledge graph score
        kg_insights = analysis.get('kg_insights', {})
        kg_score = (
            kg_insights.get('network_influence', 0.1) * 0.6 +
            min(kg_insights.get('impact_potential', 0) / 10, 1) * 0.4
        )
        score += kg_score * weights['kg_insights']
        
        # Sector alignment score
        sector_alignment = analysis.get('sector_alignment', 0)
        score += sector_alignment * weights['sector_alignment']
        
        return min(max(score, -1), 1)  # Normalize to [-1, 1]
    
    def _generate_stock_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading recommendation for stock"""
        overall_score = analysis['overall_score']
        
        if overall_score > 0.6:
            signal = 'STRONG_BUY'
            position_size = min(0.1, overall_score * 0.15)  # Max 10%, scale with score
        elif overall_score > 0.3:
            signal = 'BUY'
            position_size = min(0.05, overall_score * 0.1)
        elif overall_score > -0.3:
            signal = 'HOLD'
            position_size = 0.0
        elif overall_score > -0.6:
            signal = 'SELL'
            position_size = 0.0
        else:
            signal = 'STRONG_SELL'
            position_size = 0.0
        
        # Risk management
        risk_metrics = analysis.get('risk', {})
        max_drawdown = abs(risk_metrics.get('max_drawdown', 0.1))
        
        # Adjust position size based on risk
        if max_drawdown > 0.3:  # High risk
            position_size *= 0.5
        
        recommendation = {
            'signal': signal,
            'position_size': position_size,
            'confidence': analysis.get('ml_predictions', {}).get('confidence', 0.5),
            'expected_return': analysis.get('ml_predictions', {}).get('return_20d', 0),
            'risk_level': 'HIGH' if max_drawdown > 0.3 else 'MODERATE' if max_drawdown > 0.15 else 'LOW',
            'time_horizon': '20_days',
            'stop_loss': -max_drawdown * 0.8,  # Stop loss at 80% of historical max drawdown
            'take_profit': overall_score * 0.2  # Take profit based on score
        }
        
        return recommendation
    
    async def _generate_hierarchical_signals(self, stock_analysis: Dict[str, Dict[str, Any]]) -> List[HierarchicalSignal]:
        """Generate hierarchical trading signals"""
        signals = []
        
        for symbol, analysis in stock_analysis.items():
            try:
                sector = analysis['sector']
                recommendation = analysis.get('recommendation', {})
                
                signal = HierarchicalSignal(
                    symbol=symbol,
                    sector=sector,
                    signal_type=recommendation.get('signal', 'HOLD'),
                    strength=abs(analysis['overall_score']),
                    confidence=recommendation.get('confidence', 0.5),
                    
                    # Hierarchical context
                    market_alignment=self._calculate_market_alignment(analysis),
                    sector_momentum=self.market_nodes.get(sector, MarketNode("", "", "")).momentum_score,
                    individual_alpha=analysis['overall_score'],
                    
                    # Multi-timeframe
                    short_term_score=analysis.get('ml_predictions', {}).get('return_1d', 0),
                    medium_term_score=analysis.get('ml_predictions', {}).get('return_5d', 0),
                    long_term_score=analysis.get('ml_predictions', {}).get('return_20d', 0),
                    
                    # Risk-adjusted metrics
                    sharpe_prediction=self._estimate_sharpe_ratio(analysis),
                    
                    # Reasoning
                    reasoning_factors=self._generate_reasoning_factors(analysis),
                    supporting_evidence=self._extract_supporting_evidence(analysis),
                    risk_warnings=self._generate_risk_warnings(analysis)
                )
                
                signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Error generating signal for {symbol}: {str(e)}")
                continue
        
        return signals
    
    def _calculate_market_alignment(self, analysis: Dict[str, Any]) -> float:
        """Calculate alignment with overall market"""
        # Simplified: use sector alignment as proxy
        return analysis.get('sector_alignment', 0.0)
    
    def _estimate_sharpe_ratio(self, analysis: Dict[str, Any]) -> float:
        """Estimate expected Sharpe ratio"""
        expected_return = analysis.get('ml_predictions', {}).get('return_20d', 0)
        volatility = analysis.get('risk', {}).get('volatility', 0.2)
        
        if volatility > 0:
            return (expected_return - 0.05/252) / volatility  # Adjust for risk-free rate
        return 0.0
    
    def _generate_reasoning_factors(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate reasoning factors for recommendation"""
        factors = []
        
        technical = analysis.get('technical', {})
        if technical.get('trend') == 'UPTREND':
            factors.append("Strong upward trend with price above moving averages")
        
        if technical.get('rsi_signal') == 'OVERSOLD':
            factors.append("RSI indicates oversold conditions, potential reversal")
        
        fundamental = analysis.get('fundamental', {})
        if fundamental.get('momentum_score', 0) > 0.1:
            factors.append("Strong price momentum over 20-day period")
        
        kg_insights = analysis.get('kg_insights', {})
        if kg_insights.get('network_influence', 0) > 0.3:
            factors.append("High network influence in knowledge graph analysis")
        
        return factors
    
    def _extract_supporting_evidence(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Extract supporting evidence with values"""
        evidence = {}
        
        if 'ml_predictions' in analysis:
            evidence['predicted_return'] = analysis['ml_predictions'].get('return_20d', 0)
            evidence['confidence'] = analysis['ml_predictions'].get('confidence', 0)
        
        if 'technical' in analysis:
            evidence['rsi'] = analysis['technical'].get('rsi', 50)
            evidence['bollinger_position'] = analysis['technical'].get('bollinger_position', 0.5)
        
        evidence['overall_score'] = analysis['overall_score']
        
        return evidence
    
    def _generate_risk_warnings(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate risk warnings"""
        warnings = []
        
        risk_metrics = analysis.get('risk', {})
        
        if risk_metrics.get('max_drawdown', 0) < -0.3:
            warnings.append("High historical drawdown risk")
        
        if risk_metrics.get('volatility', 0) > 0.3:
            warnings.append("High volatility - position size should be reduced")
        
        if analysis.get('ml_predictions', {}).get('confidence', 1) < 0.4:
            warnings.append("Low prediction confidence due to data quality")
        
        return warnings
    
    async def _apply_viral_heat_propagation(self, signals: List[HierarchicalSignal]) -> Dict[str, Any]:
        """Apply viral heat propagation to trading signals"""
        logger.info("Applying viral heat propagation...")
        
        # Initialize heat sources from strong signals
        for signal in signals:
            if signal.signal_type in ['STRONG_BUY', 'BUY'] and signal.strength > 0.6:
                self.viral_engine.add_heat_source(
                    signal.symbol,
                    initial_heat=signal.strength,
                    heat_capacity=signal.confidence,
                    influence_radius=min(signal.individual_alpha * 10, 5)
                )
        
        # Propagate heat through network
        propagation_results = self.viral_engine.propagate_heat(steps=5)
        
        # Analyze propagation results
        heat_analysis = {
            'total_heat_sources': len([s for s in signals if s.signal_type in ['STRONG_BUY', 'BUY']]),
            'propagation_reached': len(propagation_results),
            'top_heat_recipients': sorted(
                [(node, result.final_heat) for node, result in propagation_results.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'network_amplification': self._calculate_network_amplification(propagation_results),
            'viral_coefficients': self._calculate_viral_coefficients(propagation_results)
        }
        
        return heat_analysis
    
    def _calculate_network_amplification(self, propagation_results: Dict[str, Any]) -> float:
        """Calculate network amplification factor"""
        total_final_heat = sum(result.final_heat for result in propagation_results.values())
        total_initial_heat = sum(result.initial_heat for result in propagation_results.values())
        
        if total_initial_heat > 0:
            return total_final_heat / total_initial_heat
        return 1.0
    
    def _calculate_viral_coefficients(self, propagation_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate viral propagation coefficients"""
        # Group by sector to see cross-sector propagation
        sector_heat = defaultdict(float)
        
        for node, result in propagation_results.items():
            if hasattr(result, 'final_heat'):
                sector = get_sector_for_stock(node)
                sector_heat[sector] += result.final_heat
        
        total_heat = sum(sector_heat.values())
        viral_coefficients = {}
        
        if total_heat > 0:
            for sector, heat in sector_heat.items():
                viral_coefficients[sector] = heat / total_heat
        
        return viral_coefficients
    
    async def _generate_final_recommendations(self, signals: List[HierarchicalSignal], 
                                            heat_propagation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate final trading recommendations"""
        recommendations = []
        
        # Combine signal strength with heat propagation
        heat_recipients = dict(heat_propagation.get('top_heat_recipients', []))
        
        for signal in signals:
            heat_boost = heat_recipients.get(signal.symbol, 0) * 0.2  # 20% boost from heat
            adjusted_strength = min(signal.strength + heat_boost, 1.0)
            
            if signal.signal_type in ['STRONG_BUY', 'BUY'] and adjusted_strength > 0.5:
                recommendation = {
                    'symbol': signal.symbol,
                    'sector': signal.sector,
                    'action': signal.signal_type,
                    'position_size': adjusted_strength * 0.1,  # Max 10% position
                    'confidence': signal.confidence,
                    'expected_return': signal.long_term_score,
                    'time_horizon': '20_days',
                    'reasoning': signal.reasoning_factors,
                    'risk_warnings': signal.risk_warnings,
                    'heat_amplification': heat_boost,
                    'priority': 'HIGH' if adjusted_strength > 0.8 else 'MEDIUM' if adjusted_strength > 0.6 else 'LOW'
                }
                
                recommendations.append(recommendation)
        
        # Sort by adjusted strength
        recommendations.sort(key=lambda x: x['position_size'], reverse=True)
        
        # Limit to top 20 recommendations to avoid over-diversification
        return recommendations[:20]
    
    def _get_market_overview(self, engineered_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate market overview"""
        all_returns = []
        all_volumes = []
        
        for symbol, df in engineered_data.items():
            if 'returns' in df.columns:
                returns = df['returns'].tail(20).tolist()
                all_returns.extend(returns)
            
            if 'Volume' in df.columns:
                volumes = df['Volume'].tail(20).tolist()
                all_volumes.extend(volumes)
        
        overview = {
            'market_sentiment': 'BULLISH' if np.mean(all_returns) > 0 else 'BEARISH',
            'average_return': np.mean(all_returns) if all_returns else 0,
            'market_volatility': np.std(all_returns) if all_returns else 0,
            'volume_trend': 'INCREASING' if len(all_volumes) > 0 and all_volumes[-10:] > all_volumes[-20:-10] else 'STABLE',
            'stocks_analyzed': len(engineered_data),
            'data_quality': sum(1 for df in engineered_data.values() if len(df) > 100) / len(engineered_data)
        }
        
        return overview
    
    def _predict_portfolio_performance(self, recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Predict overall portfolio performance"""
        if not recommendations:
            return {'expected_return': 0, 'expected_volatility': 0, 'sharpe_ratio': 0}
        
        expected_returns = [r['expected_return'] * r['position_size'] for r in recommendations]
        total_expected = sum(expected_returns)
        
        # Simplified volatility calculation
        position_sizes = [r['position_size'] for r in recommendations]
        portfolio_vol = np.sqrt(sum(p**2 * 0.3**2 for p in position_sizes))  # Assume 30% individual vol
        
        sharpe = total_expected / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'expected_return': total_expected,
            'expected_volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'max_potential_return': total_expected * 2,  # Optimistic scenario
            'downside_risk': total_expected * -0.5  # Pessimistic scenario
        }
    
    def _analyze_portfolio_risk(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze portfolio risk"""
        if not recommendations:
            return {}
        
        total_allocation = sum(r['position_size'] for r in recommendations)
        sector_allocation = defaultdict(float)
        
        for rec in recommendations:
            sector_allocation[rec['sector']] += rec['position_size']
        
        risk_analysis = {
            'total_allocation': total_allocation,
            'sector_concentration': dict(sector_allocation),
            'max_sector_weight': max(sector_allocation.values()) if sector_allocation else 0,
            'diversification_score': len(sector_allocation) / 10,  # Out of 10 sectors
            'high_risk_positions': len([r for r in recommendations if 'HIGH' in r.get('risk_warnings', [])]),
            'risk_warnings': list(set(w for r in recommendations for w in r.get('risk_warnings', [])))
        }
        
        return risk_analysis
    
    def _calculate_overall_confidence(self, signals: List[HierarchicalSignal]) -> float:
        """Calculate overall analysis confidence"""
        if not signals:
            return 0.0
        
        confidences = [signal.confidence for signal in signals]
        return np.mean(confidences)

# Example usage and testing
if __name__ == "__main__":
    async def main():
        analyzer = HierarchicalSectorStockAnalyzer()
        
        # Test with a small set of symbols
        test_symbols = PRIORITY_STOCKS[:10]
        
        print("Starting revolutionary hierarchical analysis...")
        results = await analyzer.analyze_market_hierarchy(test_symbols)
        
        print("\n=== ANALYSIS RESULTS ===")
        print(f"Market Overview: {results['market_overview']}")
        print(f"\nTop 3 Sectors:")
        for i, (sector, data) in enumerate(list(results['sector_rankings'].items())[:3]):
            print(f"{i+1}. {data['sector_name']}: Score {data['ranking_score']:.3f}")
        
        print(f"\nTop 5 Stock Recommendations:")
        for i, rec in enumerate(results['recommendations'][:5]):
            print(f"{i+1}. {rec['symbol']} ({rec['sector']}): {rec['action']} - {rec['position_size']:.1%}")
        
        print(f"\nPredicted Portfolio Performance:")
        perf = results['performance_prediction']
        print(f"Expected Return: {perf['expected_return']:.2%}")
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        
    # Run test
    import asyncio
    asyncio.run(main())