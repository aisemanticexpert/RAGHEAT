#!/usr/bin/env python3
"""
Test script to verify Neo4j streaming data population
"""

import asyncio
import sys
import logging
from backend import StreamingNeo4jService
from backend import HighSpeedStreamingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockStreamingData:
    """Mock streaming data for testing"""
    def __init__(self, symbol, price, change, change_percent, volume):
        self.symbol = symbol
        self.price = price
        self.change = change
        self.change_percent = change_percent
        self.volume = volume
        self.high = price * 1.05
        self.low = price * 0.95
        self.open_price = price * 0.98
        self.source = "test"

async def test_neo4j_connection():
    """Test Neo4j connection and data population"""
    logger.info("🧪 Testing Neo4j streaming service...")
    
    # Create Neo4j service instance (using localhost for testing outside Docker)
    neo4j_service = StreamingNeo4jService(uri="bolt://localhost:7687")
    
    # Ensure connection is established
    await neo4j_service.ensure_connected()
    
    if not neo4j_service.is_connected:
        logger.error("❌ Failed to connect to Neo4j")
        return False
    
    logger.info("✅ Connected to Neo4j successfully")
    
    # Test with mock data first
    test_data = {
        "AAPL": MockStreamingData("AAPL", 150.25, 2.50, 1.69, 45123456),
        "MSFT": MockStreamingData("MSFT", 305.80, -1.20, -0.39, 28967834),
        "GOOGL": MockStreamingData("GOOGL", 2650.45, 15.30, 0.58, 12456789),
        "NVDA": MockStreamingData("NVDA", 425.67, 8.90, 2.13, 67890123),
        "TSLA": MockStreamingData("TSLA", 248.32, -5.68, -2.24, 89012345)
    }
    
    logger.info("📊 Populating Neo4j with test streaming data...")
    success = await neo4j_service.populate_streaming_data(test_data)
    
    if success:
        logger.info("✅ Successfully populated Neo4j with streaming data")
        
        # Get stats
        stats = await neo4j_service.get_live_graph_stats()
        logger.info(f"📈 Neo4j Stats: {stats}")
        
        # Test visualization queries
        queries = await neo4j_service.create_live_visualization_queries()
        logger.info(f"📋 Generated {len(queries)} visualization queries")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"🔍 Testing query {i}...")
            results = await neo4j_service.execute_live_query(query)
            logger.info(f"✅ Query {i} returned {len(results)} records")
            if results:
                logger.info(f"Sample result: {results[0]}")
        
        return True
    else:
        logger.error("❌ Failed to populate Neo4j with streaming data")
        return False

async def test_real_streaming_data():
    """Test with real streaming data from APIs"""
    logger.info("🔄 Testing with real streaming data...")
    
    # Create streaming service
    streaming_service = HighSpeedStreamingService()
    
    # Get real data
    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    real_data = await streaming_service.get_fastest_data(symbols)
    
    if real_data:
        logger.info(f"📡 Retrieved real data for {len(real_data)} stocks")
        
        # Create Neo4j service (using localhost for testing outside Docker)
        neo4j_service = StreamingNeo4jService(uri="bolt://localhost:7687")
        await neo4j_service.ensure_connected()
        
        if neo4j_service.is_connected:
            success = await neo4j_service.populate_streaming_data(real_data)
            if success:
                logger.info("✅ Successfully populated Neo4j with real streaming data")
                
                # Get updated stats
                stats = await neo4j_service.get_live_graph_stats()
                logger.info(f"📈 Updated Neo4j Stats: {stats}")
                
                return True
            else:
                logger.error("❌ Failed to populate Neo4j with real data")
                return False
        else:
            logger.error("❌ Neo4j not connected for real data test")
            return False
    else:
        logger.error("❌ Failed to retrieve real streaming data")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Neo4j streaming data tests...")
    
    # Test 1: Connection and mock data
    test1_success = await test_neo4j_connection()
    
    if test1_success:
        logger.info("✅ Test 1 (Mock data) PASSED")
        
        # Test 2: Real streaming data
        test2_success = await test_real_streaming_data()
        
        if test2_success:
            logger.info("🎉 All tests PASSED! Neo4j streaming is working correctly.")
            return True
        else:
            logger.error("❌ Test 2 (Real data) FAILED")
            return False
    else:
        logger.error("❌ Test 1 (Mock data) FAILED")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if result:
            logger.info("🎯 Neo4j streaming service is fully operational!")
            sys.exit(0)
        else:
            logger.error("💥 Neo4j streaming service has issues!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Test failed with exception: {e}")
        sys.exit(1)