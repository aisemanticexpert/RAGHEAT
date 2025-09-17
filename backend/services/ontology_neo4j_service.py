"""
Ontology-driven Neo4j Service with Neosemantics
Implements professional knowledge graph with TTL ontology integration
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from py2neo import Graph, Node, Relationship
from dataclasses import dataclass
import asyncio
import numpy as np

@dataclass
class TradingSignal:
    signal_type: str  # BUY, SELL, HOLD
    strength: float   # 0.0 to 1.0
    confidence: float # 0.0 to 1.0
    indicators: Dict[str, float]
    reasoning: str

@dataclass 
class HeatNode:
    entity_id: str
    heat_score: float
    heat_intensity: float
    heat_capacity: float
    level: int
    node_type: str

class OntologyNeo4jService:
    """Professional ontology-driven Neo4j service with hierarchical KG"""
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.graph = Graph(uri, auth=(user, password))
        self.ontology_file = "backend/ontology/financial_markets_ontology.ttl"
        self.logger = logging.getLogger(__name__)
        
        # Professional visualization levels
        self.visualization_levels = {
            1: "Market Structure",      # Markets, Exchanges
            2: "Sector Classification", # Sectors, Industries  
            3: "Financial Instruments", # Stocks, Options, Bonds
            4: "Corporate Entities",    # Companies, Correlations
            5: "Trading Signals",       # BUY/SELL/HOLD signals
            6: "Technical Indicators",  # RSI, MA, Bollinger Bands
            7: "Heat Propagation"       # Heat flows, thermal dynamics
        }
        
    async def initialize_ontology_graph(self):
        """Initialize Neo4j with Neosemantics and load TTL ontology"""
        self.logger.info("🚀 Initializing ontology-driven knowledge graph...")
        
        try:
            # Install Neosemantics plugin
            await self._setup_neosemantics()
            
            # Load TTL ontology
            await self._load_ttl_ontology()
            
            # Create hierarchical structure
            await self._create_hierarchical_structure()
            
            # Setup indexes for performance
            await self._create_performance_indexes()
            
            self.logger.info("✅ Ontology graph initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ontology graph: {e}")
            raise
            
    async def _setup_neosemantics(self):
        """Setup Neosemantics plugin and configuration"""
        try:
            # Initialize Neosemantics
            cypher = """
            CALL n10s.graphconfig.init({
                handleVocabUris: 'MAP',
                handleMultival: 'ARRAY',
                handleRDFTypes: 'LABELS',
                keepLangTag: true,
                keepCustomDataTypes: true
            })
            """
            self.graph.run(cypher)
            
            # Create namespace mappings
            namespaces = [
                ("fin", "http://ragheat.ai/ontology/finance#"),
                ("market", "http://ragheat.ai/ontology/market#"), 
                ("trading", "http://ragheat.ai/ontology/trading#"),
                ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
                ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
                ("owl", "http://www.w3.org/2002/07/owl#")
            ]
            
            for prefix, uri in namespaces:
                cypher = f"CALL n10s.nsprefixes.add('{prefix}', '{uri}')"
                self.graph.run(cypher)
                
            self.logger.info("✅ Neosemantics configured with namespaces")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Neosemantics setup warning: {e}")
            # Continue without Neosemantics if plugin not available
            
    async def _load_ttl_ontology(self):
        """Load TTL ontology into Neo4j"""
        try:
            # Try to load via Neosemantics first
            cypher = f"""
            CALL n10s.rdf.import.fetch(
                'file://{self.ontology_file}',
                'Turtle'
            )
            """
            result = self.graph.run(cypher)
            self.logger.info("✅ TTL ontology loaded via Neosemantics")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Neosemantics not available, using manual ontology creation")
            await self._create_manual_ontology_structure()
            
    async def _create_manual_ontology_structure(self):
        """Create ontology structure manually if Neosemantics not available"""
        
        # Level 1: Market Structure
        sectors = [
            ("Technology", "Technology Sector", "#00D4FF"),
            ("Healthcare", "Healthcare Sector", "#FF6B35"), 
            ("Finance", "Financial Services", "#00FF88"),
            ("Energy", "Energy Sector", "#FFA726"),
            ("Consumer_Discretionary", "Consumer Discretionary", "#E91E63"),
            ("Consumer", "Consumer Staples", "#9C27B0"),
            ("Retail", "Retail Sector", "#FF5722"),
            ("Entertainment", "Entertainment", "#795548")
        ]
        
        for sector_id, name, color in sectors:
            sector = Node(
                "Sector", "FinancialEntity", "MarketSegment",
                name=name,
                sector_id=sector_id,
                level=2,
                visualization_color=color,
                heat_capacity=1.0,
                created_at=datetime.now().isoformat()
            )
            self.graph.merge(sector, "Sector", "sector_id")
            
        self.logger.info("✅ Manual ontology structure created")
        
    async def _create_hierarchical_structure(self):
        """Create professional hierarchical level structure"""
        
        # Create level nodes for navigation
        for level, description in self.visualization_levels.items():
            level_node = Node(
                "VisualizationLevel",
                level=level,
                name=description,
                description=f"Level {level}: {description}",
                created_at=datetime.now().isoformat()
            )
            self.graph.merge(level_node, "VisualizationLevel", "level")
            
        self.logger.info("✅ Hierarchical level structure created")
        
    async def _create_performance_indexes(self):
        """Create indexes for optimal query performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Stock) ON (n.symbol)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Sector) ON (n.sector_id)",  
            "CREATE INDEX IF NOT EXISTS FOR (n:TradingSignal) ON (n.signal_type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:HeatNode) ON (n.heat_score)",
            "CREATE INDEX IF NOT EXISTS FOR (n:FinancialEntity) ON (n.timestamp)",
            "CREATE INDEX IF NOT EXISTS FOR (n:VisualizationLevel) ON (n.level)"
        ]
        
        for index_query in indexes:
            try:
                self.graph.run(index_query)
            except Exception as e:
                self.logger.warning(f"Index creation warning: {e}")
                
        self.logger.info("✅ Performance indexes created")
        
    async def populate_real_time_data(self, market_data: Dict[str, Any]):
        """Populate ontology with real-time market data"""
        try:
            timestamp = datetime.now()
            
            # Process each stock from market data
            for symbol, stock_data in market_data.get("stocks", {}).items():
                await self._create_stock_entity(symbol, stock_data, timestamp)
                
            # Create correlations
            await self._create_correlation_network(market_data.get("stocks", {}))
            
            # Generate trading signals
            await self._generate_trading_signals(market_data.get("stocks", {}))
            
            # Update heat propagation
            await self._update_heat_propagation(market_data.get("stocks", {}))
            
            self.logger.info(f"✅ Real-time data populated for {len(market_data.get('stocks', {}))} stocks")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to populate real-time data: {e}")
            raise
            
    async def _create_stock_entity(self, symbol: str, stock_data: Dict, timestamp: datetime):
        """Create comprehensive stock entity with ontology relationships"""
        
        # Create stock node with all properties
        stock = Node(
            "Stock", "Security", "FinancialInstrument", "FinancialEntity", "HeatNode",
            symbol=symbol,
            name=symbol,  # Could be enhanced with company name lookup
            price=float(stock_data.get("price", 0)),
            volume=int(stock_data.get("volume", 0)),
            change_percent=float(stock_data.get("change_percent", 0)),
            volatility=float(stock_data.get("volatility", 0)),
            heat_score=float(stock_data.get("heat_score", 0)),
            heat_intensity=float(stock_data.get("heat_score", 0) / 100.0),
            heat_capacity=self._calculate_heat_capacity(stock_data),
            market_cap=float(stock_data.get("market_cap", 0)),
            level=3,  # Level 3: Financial Instruments
            timestamp=timestamp.isoformat(),
            last_updated=timestamp.isoformat()
        )
        
        self.graph.merge(stock, "Stock", "symbol")
        
        # Link to sector
        sector_name = stock_data.get("sector", "Technology")
        cypher = """
        MATCH (s:Stock {symbol: $symbol}), (sector:Sector {sector_id: $sector_id})
        MERGE (s)-[r:BELONGS_TO_SECTOR]->(sector)
        SET r.timestamp = $timestamp
        """
        self.graph.run(cypher, symbol=symbol, sector_id=sector_name, timestamp=timestamp.isoformat())
        
    def _calculate_heat_capacity(self, stock_data: Dict) -> float:
        """Calculate heat capacity based on market cap and volatility"""
        market_cap = stock_data.get("market_cap", 1000000000)  # Default 1B
        volatility = stock_data.get("volatility", 1.0)
        
        # Larger market cap = higher heat capacity (slower temperature changes)
        # Higher volatility = lower heat capacity (faster temperature changes)
        base_capacity = np.log10(market_cap / 1000000000)  # Normalize to billions
        volatility_factor = 1.0 / (1.0 + volatility)
        
        return max(0.1, base_capacity * volatility_factor)
        
    async def _create_correlation_network(self, stocks_data: Dict):
        """Create correlation relationships between stocks"""
        stock_symbols = list(stocks_data.keys())
        
        for i, symbol1 in enumerate(stock_symbols):
            stock1_data = stocks_data[symbol1]
            correlations = stock1_data.get("correlation_signals", {})
            
            for symbol2, correlation_value in correlations.items():
                if symbol2 in stock_symbols and abs(correlation_value) > 0.3:
                    
                    # Determine correlation strength
                    if abs(correlation_value) > 0.7:
                        rel_type = "HAS_STRONG_CORRELATION"
                        strength = "strong"
                    elif abs(correlation_value) > 0.5:
                        rel_type = "HAS_MODERATE_CORRELATION"  
                        strength = "moderate"
                    else:
                        rel_type = "HAS_WEAK_CORRELATION"
                        strength = "weak"
                        
                    cypher = f"""
                    MATCH (s1:Stock {{symbol: $symbol1}}), (s2:Stock {{symbol: $symbol2}})
                    MERGE (s1)-[r:{rel_type}]->(s2)
                    SET r.correlation_value = $correlation_value,
                        r.strength = $strength,
                        r.timestamp = $timestamp,
                        r.level = 4
                    """
                    
                    self.graph.run(cypher, 
                        symbol1=symbol1, 
                        symbol2=symbol2,
                        correlation_value=correlation_value,
                        strength=strength,
                        timestamp=datetime.now().isoformat()
                    )
                    
    async def _generate_trading_signals(self, stocks_data: Dict):
        """Generate BUY/SELL/HOLD signals based on technical analysis"""
        
        for symbol, stock_data in stocks_data.items():
            signal = self._calculate_trading_signal(stock_data)
            
            # Create trading signal node
            signal_node = Node(
                "TradingSignal", signal.signal_type,
                symbol=symbol,
                signal_type=signal.signal_type,
                strength=signal.strength,
                confidence=signal.confidence,
                reasoning=signal.reasoning,
                rsi_value=signal.indicators.get("rsi", 0),
                ma_signal=signal.indicators.get("ma_signal", 0),
                bollinger_position=signal.indicators.get("bollinger_position", 0),
                level=5,  # Level 5: Trading Signals
                timestamp=datetime.now().isoformat()
            )
            
            self.graph.merge(signal_node, "TradingSignal", ["symbol", "timestamp"])
            
            # Link signal to stock
            cypher = """
            MATCH (s:Stock {symbol: $symbol}), (sig:TradingSignal {symbol: $symbol})
            WHERE sig.timestamp = $timestamp
            MERGE (s)-[r:HAS_SIGNAL]->(sig)
            SET r.timestamp = $timestamp
            """
            
            self.graph.run(cypher, symbol=symbol, timestamp=signal_node["timestamp"])
            
    def _calculate_trading_signal(self, stock_data: Dict) -> TradingSignal:
        """Calculate trading signal using multiple technical indicators"""
        
        price = stock_data.get("price", 100)
        heat_score = stock_data.get("heat_score", 50)
        volatility = stock_data.get("volatility", 1.0)
        change_percent = stock_data.get("change_percent", 0)
        
        # Simulate technical indicators
        rsi = self._calculate_rsi_simulation(price, volatility, change_percent)
        ma_signal = self._calculate_ma_signal(price, change_percent)
        bollinger_position = self._calculate_bollinger_position(price, volatility)
        
        indicators = {
            "rsi": rsi,
            "ma_signal": ma_signal, 
            "bollinger_position": bollinger_position
        }
        
        # Decision logic
        buy_signals = 0
        sell_signals = 0
        
        # RSI signals
        if rsi < 30:
            buy_signals += 2  # Oversold
        elif rsi > 70:
            sell_signals += 2  # Overbought
            
        # Moving Average signals
        if ma_signal > 1.02:  # Price 2% above MA
            buy_signals += 1
        elif ma_signal < 0.98:  # Price 2% below MA  
            sell_signals += 1
            
        # Heat-based signals
        if heat_score > 80:
            buy_signals += 1  # High heat = potential breakout
        elif heat_score < 20:
            sell_signals += 1  # Low heat = potential weakness
            
        # Volatility signals
        if volatility > 3.0:
            sell_signals += 1  # High volatility = risk
            
        # Determine final signal
        total_signals = buy_signals + sell_signals
        
        if total_signals == 0:
            signal_type = "HoldSignal" 
            strength = 0.5
            reasoning = "Neutral market conditions"
        elif buy_signals > sell_signals:
            signal_type = "BuySignal"
            strength = min(0.95, buy_signals / max(1, total_signals))
            reasoning = f"Buy indicators: RSI={rsi:.1f}, Heat={heat_score:.1f}"
        else:
            signal_type = "SellSignal"
            strength = min(0.95, sell_signals / max(1, total_signals))
            reasoning = f"Sell indicators: RSI={rsi:.1f}, Volatility={volatility:.1f}"
            
        confidence = min(0.95, abs(buy_signals - sell_signals) / max(1, total_signals))
        
        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            indicators=indicators,
            reasoning=reasoning
        )
        
    def _calculate_rsi_simulation(self, price: float, volatility: float, change: float) -> float:
        """Simulate RSI calculation"""
        # Simplified RSI simulation based on price momentum and volatility
        base_rsi = 50.0
        
        # Adjust based on recent change
        if change > 0:
            base_rsi += min(30, change * 10)
        else:
            base_rsi -= min(30, abs(change) * 10)
            
        # Adjust for volatility
        if volatility > 2.0:
            base_rsi += np.random.normal(0, 10)  # High volatility adds noise
            
        return np.clip(base_rsi, 0, 100)
        
    def _calculate_ma_signal(self, price: float, change: float) -> float:
        """Calculate moving average signal (price relative to MA)"""
        # Simulate MA as price adjusted for recent trend
        simulated_ma = price * (1 - change / 100)
        return price / simulated_ma if simulated_ma > 0 else 1.0
        
    def _calculate_bollinger_position(self, price: float, volatility: float) -> float:
        """Calculate position within Bollinger Bands"""
        # Simulate Bollinger Band position
        # 0 = at lower band, 0.5 = at middle, 1 = at upper band
        volatility_factor = volatility / 2.0
        return np.clip(0.5 + np.random.normal(0, volatility_factor), 0, 1)
        
    async def _update_heat_propagation(self, stocks_data: Dict):
        """Update heat propagation network"""
        
        for symbol, stock_data in stocks_data.items():
            heat_score = stock_data.get("heat_score", 0)
            correlations = stock_data.get("correlation_signals", {})
            
            # Create heat flows to correlated stocks
            for corr_symbol, correlation in correlations.items():
                if abs(correlation) > 0.5:  # Only strong correlations
                    heat_flow_strength = abs(correlation) * (heat_score / 100.0)
                    
                    cypher = """
                    MATCH (s1:Stock {symbol: $symbol1}), (s2:Stock {symbol: $symbol2})
                    MERGE (s1)-[hf:PROPAGATES_HEAT_TO]->(s2)
                    SET hf.flow_strength = $flow_strength,
                        hf.heat_transfer = $heat_transfer,
                        hf.level = 7,
                        hf.timestamp = $timestamp
                    """
                    
                    self.graph.run(cypher,
                        symbol1=symbol,
                        symbol2=corr_symbol,
                        flow_strength=heat_flow_strength,
                        heat_transfer=heat_score * abs(correlation) * 0.1,
                        timestamp=datetime.now().isoformat()
                    )
                    
    async def get_hierarchical_graph_data(self, level: int = 3, max_nodes: int = 50) -> Dict[str, Any]:
        """Get professional hierarchical graph data for visualization"""
        
        try:
            if level == 1:
                return await self._get_market_structure_level()
            elif level == 2:
                return await self._get_sector_level() 
            elif level == 3:
                return await self._get_stock_level(max_nodes)
            elif level == 4:
                return await self._get_correlation_level(max_nodes)
            elif level == 5:
                return await self._get_signals_level(max_nodes)
            elif level == 6:
                return await self._get_technical_indicators_level()
            elif level == 7:
                return await self._get_heat_propagation_level(max_nodes)
            else:
                return await self._get_stock_level(max_nodes)  # Default
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get hierarchical data: {e}")
            return {"nodes": [], "links": [], "level": level, "error": str(e)}
            
    async def _get_sector_level(self) -> Dict[str, Any]:
        """Get Level 2: Sector-based visualization"""
        
        cypher = """
        MATCH (s:Sector)
        OPTIONAL MATCH (stock:Stock)-[:BELONGS_TO_SECTOR]->(s)
        WITH s, count(stock) as stock_count, avg(stock.heat_score) as avg_heat
        RETURN s.sector_id as id, s.name as name, s.visualization_color as color,
               stock_count, coalesce(avg_heat, 0) as heat_score,
               'sector' as type, 2 as level
        ORDER BY stock_count DESC
        """
        
        result = self.graph.run(cypher)
        
        nodes = []
        for record in result:
            nodes.append({
                "id": record["id"],
                "name": record["name"], 
                "type": record["type"],
                "level": record["level"],
                "heat_score": float(record["heat_score"]),
                "stock_count": record["stock_count"],
                "color": record["color"],
                "size": 20 + (record["stock_count"] * 2)  # Size based on stocks
            })
            
        # Create sector relationships (industries within sectors could be added)
        links = []
        
        return {
            "nodes": nodes,
            "links": links,
            "level": 2,
            "level_name": "Sector Classification",
            "description": "Economic sectors with aggregated metrics"
        }
        
    async def _get_stock_level(self, max_nodes: int) -> Dict[str, Any]:
        """Get Level 3: Stock-based visualization with signals"""
        
        cypher = """
        MATCH (stock:Stock)
        OPTIONAL MATCH (stock)-[:HAS_SIGNAL]->(signal:TradingSignal)
        WHERE signal.timestamp > datetime() - duration('PT1H')
        OPTIONAL MATCH (stock)-[:BELONGS_TO_SECTOR]->(sector:Sector)
        RETURN stock.symbol as id, stock.symbol as name, 'stock' as type,
               stock.price as price, stock.heat_score as heat_score,
               stock.change_percent as change_percent, stock.volatility as volatility,
               sector.name as sector, sector.visualization_color as sector_color,
               signal.signal_type as signal_type, signal.strength as signal_strength,
               signal.confidence as signal_confidence,
               3 as level
        ORDER BY stock.heat_score DESC
        LIMIT $max_nodes
        """
        
        result = self.graph.run(cypher, max_nodes=max_nodes)
        
        nodes = []
        for record in result:
            # Determine node color based on signal
            color = record["sector_color"] or "#666666"
            if record["signal_type"]:
                if "Buy" in record["signal_type"]:
                    color = "#00FF88"  # Green for buy
                elif "Sell" in record["signal_type"]:
                    color = "#FF6B35"  # Red for sell
                else:
                    color = "#FFA726"  # Orange for hold
                    
            nodes.append({
                "id": record["id"],
                "name": record["name"],
                "type": record["type"],
                "level": record["level"],
                "heat_score": float(record["heat_score"] or 0),
                "price": float(record["price"] or 0),
                "change_percent": float(record["change_percent"] or 0),
                "volatility": float(record["volatility"] or 0),
                "sector": record["sector"],
                "signal_type": record["signal_type"],
                "signal_strength": float(record["signal_strength"] or 0),
                "signal_confidence": float(record["signal_confidence"] or 0),
                "color": color,
                "size": 10 + (record["heat_score"] or 0) / 5  # Size based on heat
            })
            
        # Get correlation relationships
        cypher_links = """
        MATCH (s1:Stock)-[r:HAS_STRONG_CORRELATION|HAS_MODERATE_CORRELATION]->(s2:Stock)
        WHERE s1.symbol IN $symbols AND s2.symbol IN $symbols
        RETURN s1.symbol as source, s2.symbol as target, 
               r.correlation_value as correlation, r.strength as strength,
               type(r) as relationship_type
        """
        
        symbols = [node["id"] for node in nodes]
        link_result = self.graph.run(cypher_links, symbols=symbols)
        
        links = []
        for record in link_result:
            # Color based on correlation strength
            if record["strength"] == "strong":
                color = "#FF0066"
                width = 3
            elif record["strength"] == "moderate": 
                color = "#FF6600"
                width = 2
            else:
                color = "#CCCCCC"
                width = 1
                
            links.append({
                "source": record["source"],
                "target": record["target"],
                "type": "correlation",
                "correlation": float(record["correlation"]),
                "strength": record["strength"],
                "color": color,
                "width": width
            })
            
        return {
            "nodes": nodes,
            "links": links,
            "level": 3,
            "level_name": "Financial Instruments",
            "description": "Stocks with real-time signals and correlations"
        }
        
    async def _get_signals_level(self, max_nodes: int) -> Dict[str, Any]:
        """Get Level 5: Trading Signals visualization"""
        
        cypher = """
        MATCH (stock:Stock)-[:HAS_SIGNAL]->(signal:TradingSignal)
        WHERE signal.timestamp > datetime() - duration('PT1H')
        RETURN stock.symbol as stock_symbol, stock.heat_score as heat_score,
               signal.signal_type as signal_type, signal.strength as strength,
               signal.confidence as confidence, signal.reasoning as reasoning,
               signal.rsi_value as rsi, signal.ma_signal as ma_signal,
               5 as level
        ORDER BY signal.strength DESC
        LIMIT $max_nodes
        """
        
        result = self.graph.run(cypher, max_nodes=max_nodes)
        
        nodes = []
        links = []
        
        for record in result:
            stock_id = record["stock_symbol"]
            signal_id = f"{stock_id}_signal"
            
            # Stock node
            nodes.append({
                "id": stock_id,
                "name": record["stock_symbol"],
                "type": "stock",
                "level": 3,
                "heat_score": float(record["heat_score"] or 0),
                "color": "#666666",
                "size": 15
            })
            
            # Signal node  
            signal_color = "#00FF88" if "Buy" in record["signal_type"] else \
                          "#FF6B35" if "Sell" in record["signal_type"] else "#FFA726"
                          
            nodes.append({
                "id": signal_id,
                "name": record["signal_type"].replace("Signal", ""),
                "type": "signal",
                "level": 5,
                "signal_type": record["signal_type"],
                "strength": float(record["strength"]),
                "confidence": float(record["confidence"]),
                "reasoning": record["reasoning"],
                "rsi": float(record["rsi"] or 0),
                "color": signal_color,
                "size": 10 + (record["strength"] * 10)
            })
            
            # Link stock to signal
            links.append({
                "source": stock_id,
                "target": signal_id,
                "type": "has_signal",
                "color": signal_color,
                "width": 2
            })
            
        return {
            "nodes": nodes,
            "links": links,
            "level": 5,
            "level_name": "Trading Signals",
            "description": "BUY/SELL/HOLD recommendations with confidence levels"
        }
        
    async def _get_heat_propagation_level(self, max_nodes: int) -> Dict[str, Any]:
        """Get Level 7: Heat Propagation Network"""
        
        cypher = """
        MATCH (s1:Stock)-[hf:PROPAGATES_HEAT_TO]->(s2:Stock)
        WHERE hf.flow_strength > 0.1
        RETURN s1.symbol as source_symbol, s1.heat_score as source_heat,
               s2.symbol as target_symbol, s2.heat_score as target_heat,
               hf.flow_strength as flow_strength, hf.heat_transfer as heat_transfer
        ORDER BY hf.flow_strength DESC
        LIMIT $max_nodes
        """
        
        result = self.graph.run(cypher, max_nodes=max_nodes)
        
        nodes = {}
        links = []
        
        for record in result:
            source_id = record["source_symbol"]
            target_id = record["target_symbol"]
            
            # Add source node
            if source_id not in nodes:
                nodes[source_id] = {
                    "id": source_id,
                    "name": source_id,
                    "type": "heat_node",
                    "level": 7,
                    "heat_score": float(record["source_heat"]),
                    "color": self._get_heat_color(record["source_heat"]),
                    "size": 10 + (record["source_heat"] / 5)
                }
                
            # Add target node
            if target_id not in nodes:
                nodes[target_id] = {
                    "id": target_id,
                    "name": target_id, 
                    "type": "heat_node",
                    "level": 7,
                    "heat_score": float(record["target_heat"]),
                    "color": self._get_heat_color(record["target_heat"]),
                    "size": 10 + (record["target_heat"] / 5)
                }
                
            # Heat flow link
            links.append({
                "source": source_id,
                "target": target_id,
                "type": "heat_flow",
                "flow_strength": float(record["flow_strength"]),
                "heat_transfer": float(record["heat_transfer"]),
                "color": "#FF6600",
                "width": max(1, record["flow_strength"] * 5)
            })
            
        return {
            "nodes": list(nodes.values()),
            "links": links,
            "level": 7,
            "level_name": "Heat Propagation",
            "description": "Thermal dynamics and heat flow networks"
        }
        
    def _get_heat_color(self, heat_score: float) -> str:
        """Get color based on heat score"""
        if heat_score > 80:
            return "#FF0033"  # Hot red
        elif heat_score > 60:
            return "#FF6600"  # Orange
        elif heat_score > 40:
            return "#FFAA00"  # Yellow-orange
        elif heat_score > 20:
            return "#0099FF"  # Cool blue
        else:
            return "#0066CC"  # Cold blue
            
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get comprehensive market overview"""
        
        cypher = """
        MATCH (stock:Stock)
        OPTIONAL MATCH (stock)-[:HAS_SIGNAL]->(signal:TradingSignal)
        WHERE signal.timestamp > datetime() - duration('PT1H')
        WITH count(stock) as total_stocks,
             count(signal) as total_signals,
             sum(CASE WHEN 'Buy' IN labels(signal) THEN 1 ELSE 0 END) as buy_signals,
             sum(CASE WHEN 'Sell' IN labels(signal) THEN 1 ELSE 0 END) as sell_signals,
             sum(CASE WHEN 'Hold' IN labels(signal) THEN 1 ELSE 0 END) as hold_signals,
             avg(stock.heat_score) as avg_heat,
             max(stock.heat_score) as max_heat,
             min(stock.heat_score) as min_heat
        RETURN total_stocks, total_signals, buy_signals, sell_signals, hold_signals,
               avg_heat, max_heat, min_heat
        """
        
        result = self.graph.run(cypher).data()
        if result:
            data = result[0]
            return {
                "total_stocks": data["total_stocks"],
                "total_signals": data["total_signals"],
                "buy_signals": data["buy_signals"],
                "sell_signals": data["sell_signals"], 
                "hold_signals": data["hold_signals"],
                "average_heat": float(data["avg_heat"] or 0),
                "max_heat": float(data["max_heat"] or 0),
                "min_heat": float(data["min_heat"] or 0),
                "market_sentiment": self._calculate_market_sentiment(data),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"error": "No market data available"}
            
    def _calculate_market_sentiment(self, data: Dict) -> str:
        """Calculate overall market sentiment"""
        buy_signals = data.get("buy_signals", 0)
        sell_signals = data.get("sell_signals", 0)
        total_signals = data.get("total_signals", 0)
        
        if total_signals == 0:
            return "Neutral"
            
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        
        if buy_ratio > 0.6:
            return "Bullish"
        elif sell_ratio > 0.6:
            return "Bearish"
        elif buy_ratio > sell_ratio:
            return "Cautiously Bullish"
        elif sell_ratio > buy_ratio:
            return "Cautiously Bearish"
        else:
            return "Neutral"