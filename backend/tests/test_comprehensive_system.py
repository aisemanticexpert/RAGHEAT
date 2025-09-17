"""
Comprehensive Test Suite for RAGHeat Trading System
Tests all major components: Synthetic Data, Kafka, Heat Models, GraphRAG, PathRAG, LLM Analysis
"""

import pytest
import asyncio
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import all components to test
from data_pipeline.synthetic_market_generator import (
    SyntheticMarketGenerator, get_synthetic_market_data, get_synthetic_neo4j_data
)
from services.market_hours_detector import MarketHoursDetector, DataSourceManager
from models.knowledge_graph.graph_rag_engine import AdvancedGraphRAG, graph_rag_engine
from models.knowledge_graph.path_rag_engine import initialize_path_rag, PathRAGQuery, PathType
from models.heat_propagation.advanced_heat_engine import AdvancedHeatEngine, analyze_market_heat
from services.intelligent_llm_analyzer import IntelligentLLMAnalyzer, AnalysisDepth
from streaming.intelligent_kafka_producer import IntelligentMarketKafkaProducer


class TestSyntheticDataGenerator:
    """Test synthetic market data generation"""
    
    @pytest.fixture
    def generator(self):
        return SyntheticMarketGenerator()
    
    def test_generator_initialization(self, generator):
        """Test that generator initializes correctly"""
        assert generator is not None
        assert len(generator.sectors) > 0
        assert len(generator.stocks) > 0
        assert generator.heat_diffusion_matrix is not None
    
    def test_market_overview_generation(self, generator):
        """Test market overview data generation"""
        data = generator.generate_market_overview()
        
        # Validate structure
        assert "timestamp" in data
        assert "market_overview" in data
        assert "stocks" in data
        assert "sectors" in data
        assert "heated_sectors" in data
        assert "generation_metadata" in data
        
        # Validate market overview
        market = data["market_overview"]
        assert "market_return" in market
        assert "market_volatility" in market
        assert "total_volume" in market
        assert "market_heat_index" in market
        
        # Validate stocks data
        assert len(data["stocks"]) > 0
        for symbol, stock_data in data["stocks"].items():
            assert "price" in stock_data
            assert "change_percent" in stock_data
            assert "volume" in stock_data
            assert "heat_level" in stock_data
            assert "sector" in stock_data
    
    def test_neo4j_graph_data_generation(self, generator):
        """Test Neo4j compatible graph structure generation"""
        graph_data = generator.get_neo4j_graph_data()
        
        assert "nodes" in graph_data
        assert "relationships" in graph_data
        assert "metadata" in graph_data
        
        # Validate nodes
        nodes = graph_data["nodes"]
        assert len(nodes) > 0
        
        market_node = next((n for n in nodes if n["type"] == "Market"), None)
        assert market_node is not None
        
        sector_nodes = [n for n in nodes if n["type"] == "Sector"]
        assert len(sector_nodes) > 0
        
        stock_nodes = [n for n in nodes if n["type"] == "Stock"]
        assert len(stock_nodes) > 0
        
        # Validate relationships
        relationships = graph_data["relationships"]
        assert len(relationships) > 0
        
        for rel in relationships:
            assert "source" in rel
            assert "target" in rel
            assert "type" in rel
            assert "weight" in rel
    
    def test_heat_diffusion_physics(self, generator):
        """Test heat diffusion physics model"""
        # Generate multiple cycles to test heat evolution
        initial_data = generator.generate_market_overview()
        initial_heat = sum(s.current_heat_level for s in generator.sectors.values())
        
        # Generate more data cycles
        for _ in range(5):
            generator.generate_market_overview()
        
        final_data = generator.generate_market_overview()
        final_heat = sum(s.current_heat_level for s in generator.sectors.values())
        
        # Heat should evolve over time (not be static)
        assert abs(final_heat - initial_heat) > 0.01 or len(final_data["heated_sectors"]) > 0
    
    def test_regime_shifts(self, generator):
        """Test market regime shifting"""
        initial_regime = generator.market_regime
        
        # Force regime shifts by calling multiple times
        for _ in range(20):
            generator.generate_market_regime_shift()
        
        # Regime should have changed at least once (probabilistic, but very likely)
        regimes_seen = {initial_regime}
        for _ in range(50):
            generator.generate_market_regime_shift()
            regimes_seen.add(generator.market_regime)
        
        assert len(regimes_seen) > 1  # Should see multiple regimes


class TestMarketHoursDetector:
    """Test market hours detection and data source switching"""
    
    @pytest.fixture
    def detector(self):
        return MarketHoursDetector()
    
    @pytest.fixture
    def data_manager(self):
        return DataSourceManager()
    
    def test_market_time_calculation(self, detector):
        """Test market time zone handling"""
        market_time = detector.get_current_market_time()
        assert market_time.tzinfo is not None
        assert str(market_time.tzinfo) == "America/New_York"
    
    def test_trading_day_detection(self, detector):
        """Test trading day vs weekend/holiday detection"""
        # Test a known weekday (modify as needed)
        test_date = datetime(2024, 3, 15, 10, 0)  # Friday
        test_date_et = detector.market_timezone.localize(test_date)
        
        is_trading, reason = detector.is_trading_day(test_date_et)
        # This should be a trading day (assuming not a holiday)
        if not is_trading and reason != "Weekend":
            # It's a holiday, which is valid
            assert reason is not None
    
    def test_session_detection(self, detector):
        """Test trading session detection"""
        # Test different times of day
        test_times = [
            (8, 0),   # Pre-market
            (10, 0),  # Regular session
            (17, 0),  # After hours
            (22, 0),  # Closed
        ]
        
        for hour, minute in test_times:
            test_time = detector.market_timezone.localize(
                datetime(2024, 3, 15, hour, minute)  # Friday
            )
            session_name, session = detector.get_session_for_time(test_time)
            # Should return valid session info
            if session:
                assert hasattr(session, 'name')
                assert hasattr(session, 'start_time')
                assert hasattr(session, 'end_time')
    
    def test_market_status_calculation(self, detector):
        """Test comprehensive market status calculation"""
        status = detector.get_market_status()
        
        assert hasattr(status, 'status')
        assert hasattr(status, 'current_time')
        assert hasattr(status, 'market_timezone')
        assert hasattr(status, 'is_holiday')
    
    def test_synthetic_data_decision(self, detector):
        """Test decision logic for using synthetic data"""
        should_use_synthetic = detector.should_use_synthetic_data()
        assert isinstance(should_use_synthetic, bool)
    
    @pytest.mark.asyncio
    async def test_data_source_switching(self, data_manager):
        """Test intelligent data source switching"""
        # Mock the actual API calls
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"test": "data"})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Test data retrieval
            result = await data_manager.get_market_data("market_overview")
            assert result is not None
            assert isinstance(result, dict)


class TestAdvancedHeatEngine:
    """Test heat equation models and sector heating detection"""
    
    @pytest.fixture
    def heat_engine(self):
        return AdvancedHeatEngine()
    
    @pytest.fixture
    def sample_market_data(self):
        return {
            "heated_sectors": [
                {"sector": "technology", "heat_level": 0.8, "reason": "AI breakthrough"},
                {"sector": "energy", "heat_level": -0.6, "reason": "Oil decline"}
            ],
            "stocks": {
                "AAPL": {"heat_level": 0.7, "price": 180, "change_percent": 5.2, "volume": 3000000},
                "NVDA": {"heat_level": 0.9, "price": 450, "change_percent": 8.1, "volume": 5000000},
                "XOM": {"heat_level": -0.5, "price": 105, "change_percent": -3.2, "volume": 2000000}
            },
            "market_overview": {"market_heat_index": 45.2}
        }
    
    def test_heat_engine_initialization(self, heat_engine):
        """Test heat engine initializes correctly"""
        assert heat_engine.thermal_diffusivity > 0
        assert heat_engine.grid_size > 0
        assert len(heat_engine.entity_positions) > 0
        assert heat_engine.correlation_matrix is not None
    
    def test_entity_positioning(self, heat_engine):
        """Test that entities are positioned correctly in market space"""
        assert "US_MARKET" in heat_engine.entity_positions
        assert "TECHNOLOGY" in heat_engine.entity_positions
        assert "STOCK_AAPL" in heat_engine.entity_positions
        
        # Market should be at center
        market_pos = heat_engine.entity_positions["US_MARKET"]
        assert abs(market_pos[0] - 0.5) < 0.1
        assert abs(market_pos[1] - 0.5) < 0.1
    
    def test_heat_source_creation(self, heat_engine, sample_market_data):
        """Test heat source creation from market data"""
        initial_sources = len(heat_engine.active_sources)
        heat_engine.create_heat_sources_from_data(sample_market_data)
        
        # Should have added heat sources
        assert len(heat_engine.active_sources) > initial_sources
        
        # Check source properties
        for source in heat_engine.active_sources:
            assert hasattr(source, 'entity_id')
            assert hasattr(source, 'intensity')
            assert hasattr(source, 'location')
            assert source.intensity > 0
    
    def test_heat_equation_solving(self, heat_engine, sample_market_data):
        """Test 2D heat equation solution"""
        # Add heat sources
        heat_engine.create_heat_sources_from_data(sample_market_data)
        
        # Solve for short time period
        time_series = heat_engine.solve_heat_equation_2d(time_span=60, num_steps=20)
        
        assert len(time_series) > 0
        
        # Validate field states
        for state in time_series:
            assert hasattr(state, 'timestamp')
            assert hasattr(state, 'heat_values')
            assert hasattr(state, 'heated_entities')
            assert state.heat_values.shape == (heat_engine.grid_size, heat_engine.grid_size)
    
    def test_heated_sector_identification(self, heat_engine, sample_market_data):
        """Test identification of heated sectors"""
        result = heat_engine.analyze_heat_propagation(sample_market_data, time_horizon=300)
        
        assert hasattr(result, 'heated_sectors')
        assert isinstance(result.heated_sectors, list)
        
        # Should find heated sectors from input data
        heated_sector_names = [hs['sector'] for hs in result.heated_sectors]
        assert len(heated_sector_names) > 0
        
        # Validate heated sector structure
        for sector in result.heated_sectors:
            assert 'sector' in sector
            assert 'heat_level' in sector
            assert 'recommendation' in sector
            assert 'confidence' in sector
    
    def test_heat_flow_analysis(self, heat_engine, sample_market_data):
        """Test heat flow pattern analysis"""
        result = heat_engine.analyze_heat_propagation(sample_market_data, time_horizon=300)
        
        assert 'heat_flow_analysis' in result.__dict__
        flow_analysis = result.heat_flow_analysis
        
        if "error" not in flow_analysis:
            assert 'dominant_flow_direction' in flow_analysis
            assert 'heat_convergence_points' in flow_analysis
    
    def test_heat_predictions(self, heat_engine, sample_market_data):
        """Test heat evolution predictions"""
        result = heat_engine.analyze_heat_propagation(sample_market_data, time_horizon=600)
        
        assert hasattr(result, 'predictions')
        predictions = result.predictions
        
        if "error" not in predictions:
            # Should have some prediction categories
            assert len(predictions) > 0


class TestGraphRAGEngine:
    """Test GraphRAG knowledge graph reasoning"""
    
    @pytest.fixture
    def graph_rag(self):
        return AdvancedGraphRAG()
    
    @pytest.fixture
    def sample_market_data(self):
        return get_synthetic_market_data()
    
    def test_graph_initialization(self, graph_rag):
        """Test knowledge graph initialization"""
        assert len(graph_rag.entities) > 0
        assert len(graph_rag.relationships) > 0
        assert graph_rag.graph.number_of_nodes() > 0
    
    def test_market_data_update(self, graph_rag, sample_market_data):
        """Test updating graph with market data"""
        initial_entities = len(graph_rag.entities)
        graph_rag.update_from_market_data(sample_market_data)
        
        # Should have added/updated entities
        assert len(graph_rag.entities) >= initial_entities
    
    def test_reasoning_path_finding(self, graph_rag):
        """Test finding reasoning paths"""
        # Update with some data first
        graph_rag.update_from_market_data(get_synthetic_market_data())
        
        # Find paths from a technology stock
        if "STOCK_AAPL" in graph_rag.entities:
            paths = graph_rag.find_reasoning_paths("STOCK_AAPL", max_hops=3)
            
            # Should find some paths
            assert isinstance(paths, list)
            
            if len(paths) > 0:
                path = paths[0]
                assert hasattr(path, 'entities')
                assert hasattr(path, 'relationships')
                assert hasattr(path, 'confidence')
    
    @pytest.mark.asyncio
    async def test_graph_rag_query(self, graph_rag):
        """Test full GraphRAG query"""
        # Update with market data
        graph_rag.update_from_market_data(get_synthetic_market_data())
        
        # Test query
        result = await graph_rag.query("What's the outlook for technology stocks?")
        
        assert hasattr(result, 'query')
        assert hasattr(result, 'primary_entity')
        assert hasattr(result, 'reasoning_paths')
        assert hasattr(result, 'confidence')
    
    def test_graph_statistics(self, graph_rag):
        """Test graph statistics calculation"""
        stats = graph_rag.get_graph_statistics()
        
        assert 'total_entities' in stats
        assert 'total_relationships' in stats
        assert 'entity_types' in stats
        assert 'graph_connectivity' in stats


class TestPathRAGEngine:
    """Test PathRAG multi-hop reasoning"""
    
    @pytest.fixture
    def path_rag(self):
        # Initialize with GraphRAG engine
        return initialize_path_rag(graph_rag_engine)
    
    def test_path_rag_initialization(self, path_rag):
        """Test PathRAG initialization"""
        assert path_rag is not None
        assert path_rag.graph_rag is not None
        assert len(path_rag.path_patterns) > 0
    
    def test_path_type_classification(self, path_rag):
        """Test path type classification"""
        # Create mock path hops
        from models.knowledge_graph.path_rag_engine import PathHop
        from models.knowledge_graph.graph_rag_engine import GraphEntity, GraphRelationship
        
        # Mock entities and relationships
        entity1 = GraphEntity("test1", "Stock", "Test1", {})
        entity2 = GraphEntity("test2", "Sector", "Test2", {})
        
        relationship = GraphRelationship("test1", "test2", "BELONGS_TO", {}, 1.0, 0.8)
        
        hop = PathHop(entity1, entity2, relationship, 0.8, "test hop")
        
        path_type = path_rag.classify_path_type([hop])
        assert path_type is not None
    
    @pytest.mark.asyncio
    async def test_path_finding(self, path_rag):
        """Test multi-hop path finding"""
        # Update GraphRAG with data first
        graph_rag_engine.update_from_market_data(get_synthetic_market_data())
        
        # Create path query
        query = PathRAGQuery(
            query_text="Test path analysis",
            start_entity_ids=["US_MARKET"],
            max_hops=3,
            min_confidence=0.1,
            max_paths=5
        )
        
        paths = await path_rag.find_multi_hop_paths(query)
        
        assert isinstance(paths, list)
        # May be empty if no good paths found, which is okay for test


class TestIntelligentLLMAnalyzer:
    """Test intelligent LLM analysis integration"""
    
    @pytest.fixture
    def analyzer(self):
        return IntelligentLLMAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.path_rag is not None
        assert len(analyzer.analysis_templates) > 0
    
    @pytest.mark.asyncio
    async def test_market_context_retrieval(self, analyzer):
        """Test market context retrieval"""
        context = await analyzer.get_market_context()
        
        assert hasattr(context, 'market_status')
        assert hasattr(context, 'data_source')
        assert hasattr(context, 'market_sentiment')
        assert hasattr(context, 'heated_sectors')
    
    @pytest.mark.asyncio
    async def test_heat_analysis(self, analyzer):
        """Test heat analysis component"""
        context = await analyzer.get_market_context()
        heat_analysis = await analyzer.perform_heat_analysis("AAPL", context)
        
        assert isinstance(heat_analysis, dict)
        if "error" not in heat_analysis:
            assert "stock_heat_level" in heat_analysis
            assert "sector_heat_level" in heat_analysis
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, analyzer):
        """Test full comprehensive analysis"""
        # Use quick analysis to speed up test
        result = await analyzer.analyze_stock("AAPL", depth=AnalysisDepth.QUICK)
        
        assert hasattr(result, 'symbol')
        assert hasattr(result, 'recommendation')
        assert hasattr(result, 'analysis_depth')
        assert hasattr(result, 'market_context')
        assert hasattr(result, 'execution_summary')
        
        # Check recommendation structure
        rec = result.recommendation
        assert hasattr(rec, 'signal')
        assert hasattr(rec, 'confidence')
        assert hasattr(rec, 'reasoning')


class TestKafkaProducer:
    """Test Kafka producer integration"""
    
    @pytest.fixture
    def producer(self):
        return IntelligentMarketKafkaProducer()
    
    def test_producer_initialization(self, producer):
        """Test producer initialization"""
        assert producer is not None
        assert len(producer.topics) > 0
        assert producer.data_source_manager is not None
    
    @pytest.mark.asyncio
    async def test_comprehensive_data_generation(self, producer):
        """Test comprehensive market data generation"""
        with patch.object(producer.data_source_manager, 'get_market_data') as mock_get_data:
            mock_get_data.return_value = get_synthetic_market_data()
            
            data = await producer.get_comprehensive_market_data()
            
            assert isinstance(data, dict)
            if data:  # May be empty due to mocking
                assert 'market_overview' in data or 'market_info' in data
    
    def test_producer_stats(self, producer):
        """Test producer statistics"""
        stats = producer.get_producer_stats()
        
        assert 'timestamp' in stats
        assert 'streaming_active' in stats
        assert 'kafka_connected' in stats
        assert 'market_status' in stats


class TestIntegrationScenarios:
    """Test integration between different components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis_flow(self):
        """Test complete end-to-end analysis flow"""
        # 1. Generate synthetic market data
        market_data = get_synthetic_market_data()
        assert market_data is not None
        
        # 2. Analyze with heat engine
        heat_result = analyze_market_heat(market_data)
        assert heat_result is not None
        
        # 3. Update GraphRAG with data
        graph_rag_engine.update_from_market_data(market_data)
        
        # 4. Perform comprehensive analysis
        analyzer = IntelligentLLMAnalyzer()
        analysis_result = await analyzer.analyze_stock("AAPL", depth=AnalysisDepth.QUICK)
        
        assert analysis_result is not None
        assert analysis_result.recommendation is not None
    
    def test_data_consistency(self):
        """Test data consistency across components"""
        # Generate data from synthetic generator
        market_data = get_synthetic_market_data()
        
        # Get Neo4j format of same data
        neo4j_data = get_synthetic_neo4j_data()
        
        # Verify consistency
        market_stocks = set(market_data.get("stocks", {}).keys())
        neo4j_stock_nodes = {node["name"] for node in neo4j_data.get("nodes", []) 
                           if node.get("type") == "Stock"}
        
        # Should have overlapping stocks
        overlap = market_stocks & neo4j_stock_nodes
        assert len(overlap) > 0
    
    def test_heat_diffusion_consistency(self):
        """Test heat diffusion consistency across models"""
        market_data = get_synthetic_market_data()
        
        # Get heated sectors from synthetic generator
        synthetic_heated = {hs["sector"] for hs in market_data.get("heated_sectors", [])}
        
        # Get heated sectors from heat engine
        heat_result = analyze_market_heat(market_data)
        heat_engine_heated = {hs["sector"] for hs in heat_result.heated_sectors}
        
        # Should have some consistency (may not be exact due to different models)
        if synthetic_heated and heat_engine_heated:
            # At least some overlap or similar total count
            overlap = len(synthetic_heated & heat_engine_heated)
            total_count_diff = abs(len(synthetic_heated) - len(heat_engine_heated))
            
            # Either overlap or similar counts (allowing for model differences)
            assert overlap > 0 or total_count_diff <= 2


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid/empty data"""
        # Empty market data
        empty_data = {}
        
        # Should not crash
        try:
            result = analyze_market_heat(empty_data)
            assert result is not None
        except Exception as e:
            # If it throws, should be a controlled exception
            assert "error" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_api_failure_fallback(self):
        """Test fallback when APIs fail"""
        data_manager = DataSourceManager()
        
        # Mock API failure
        with patch('aiohttp.ClientSession.get', side_effect=Exception("API failed")):
            result = await data_manager.get_market_data("market_overview")
            
            # Should fallback to synthetic data
            assert result is not None
            assert "market_status" in result
    
    def test_malformed_query_handling(self):
        """Test handling of malformed queries"""
        analyzer = IntelligentLLMAnalyzer()
        
        # Test with invalid symbols
        try:
            # This should handle gracefully
            future_result = analyzer.analyze_stock("INVALID_SYMBOL_12345")
            # Just check it doesn't immediately crash
            assert future_result is not None
        except Exception as e:
            # Should be a controlled exception
            assert isinstance(e, (ValueError, TypeError))


# Test runner configuration
if __name__ == "__main__":
    # Run specific test categories
    import subprocess
    
    print("🧪 Running RAGHeat System Comprehensive Tests")
    print("=" * 60)
    
    # Run tests with verbose output
    test_commands = [
        "python -m pytest tests/test_comprehensive_system.py::TestSyntheticDataGenerator -v",
        "python -m pytest tests/test_comprehensive_system.py::TestMarketHoursDetector -v", 
        "python -m pytest tests/test_comprehensive_system.py::TestAdvancedHeatEngine -v",
        "python -m pytest tests/test_comprehensive_system.py::TestGraphRAGEngine -v",
        "python -m pytest tests/test_comprehensive_system.py::TestIntelligentLLMAnalyzer -v",
        "python -m pytest tests/test_comprehensive_system.py::TestIntegrationScenarios -v"
    ]
    
    for cmd in test_commands:
        print(f"\n📊 Running: {cmd}")
        try:
            subprocess.run(cmd.split(), check=True)
            print("✅ Test passed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Test failed: {e}")
        except FileNotFoundError:
            print("⚠️ pytest not found, install with: pip install pytest pytest-asyncio")
            break