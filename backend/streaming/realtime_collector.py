"""
Real-Time Data Collector
Collects market data every 5 seconds and stores in Redis JSON format
No mock data - only real API data
"""
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List
import requests
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.redis_json_storage import RedisJSONStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDataCollector:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.redis_storage = RedisJSONStorage()
        self.collection_interval = 5  # 5 seconds as requested
        self.running = False
        
        # Working API endpoints identified from audit
        self.api_endpoints = {
            'market_overview': '/api/live-data/market-overview',
            'heat_distribution': '/api/heat/distribution', 
            'neo4j_stocks': '/api/streaming/neo4j/query/top-performers',
            'neo4j_sectors': '/api/streaming/neo4j/query/sector-performance'
        }
        
    def connect(self):
        """Connect to Redis storage"""
        if not self.redis_storage.connect():
            logger.error("❌ Cannot connect to Redis storage")
            return False
        logger.info("✅ Connected to Redis storage")
        return True
    
    def fetch_api_data(self, endpoint: str, timeout: int = 5) -> Dict[str, Any]:
        """Fetch data from API endpoint (no mock data)"""
        try:
            url = f"{self.api_base_url}{endpoint}"
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Fetched real data from {endpoint}")
                return data
            else:
                logger.warning(f"⚠️  API {endpoint} returned {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error fetching {endpoint}: {e}")
            
        return {}
    
    def collect_and_store_data(self):
        """Collect real data from all working APIs and store in Redis"""
        collection_timestamp = datetime.utcnow()
        logger.info(f"🔄 Starting data collection cycle at {collection_timestamp}")
        
        collected_data = {}
        
        # Collect from all working API endpoints
        for data_type, endpoint in self.api_endpoints.items():
            real_data = self.fetch_api_data(endpoint)
            
            if real_data:
                # Add metadata
                enhanced_data = {
                    'raw_data': real_data,
                    'collection_timestamp': collection_timestamp.isoformat(),
                    'data_source': 'real_api',
                    'endpoint': endpoint,
                    'timeframe': '5s'
                }
                
                # Store in Redis with 5-second timeframe
                success = self.redis_storage.store_market_data(data_type, enhanced_data)
                if success:
                    collected_data[data_type] = enhanced_data
                    logger.info(f"📦 Stored {data_type} data in Redis")
                else:
                    logger.error(f"❌ Failed to store {data_type} data")
            else:
                logger.warning(f"⚠️  No real data available for {data_type}")
        
        logger.info(f"📊 Collection cycle completed: {len(collected_data)}/{len(self.api_endpoints)} data types stored")
        return collected_data
    
    def run_continuous_collection(self):
        """Run continuous data collection every 5 seconds"""
        if not self.connect():
            logger.error("Cannot start collection without Redis connection")
            return
            
        self.running = True
        logger.info(f"🚀 Starting continuous real-time data collection (every {self.collection_interval} seconds)")
        logger.info(f"📡 Monitoring APIs: {list(self.api_endpoints.keys())}")
        logger.info("🔥 NO MOCK DATA - Only real API data will be collected and stored")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Collect and store data
                collected_data = self.collect_and_store_data()
                
                # Log collection summary
                end_time = time.time()
                collection_duration = end_time - start_time
                
                logger.info(f"⏱️  Collection took {collection_duration:.2f}s, sleeping for {self.collection_interval}s")
                
                # Wait for next collection cycle
                time.sleep(self.collection_interval)
                
        except KeyboardInterrupt:
            logger.info("⏹️  Data collection stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in collection loop: {e}")
        finally:
            self.running = False
            logger.info("🔌 Data collector stopped")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics from Redis storage"""
        return self.redis_storage.get_storage_stats()
    
    def test_apis(self):
        """Test all API endpoints to see what's working"""
        logger.info("🧪 Testing API endpoints...")
        
        working_apis = []
        broken_apis = []
        
        for data_type, endpoint in self.api_endpoints.items():
            data = self.fetch_api_data(endpoint)
            if data:
                working_apis.append((data_type, endpoint))
                logger.info(f"✅ {data_type}: WORKING")
            else:
                broken_apis.append((data_type, endpoint))
                logger.error(f"❌ {data_type}: BROKEN")
        
        logger.info(f"📊 API Test Results: {len(working_apis)} working, {len(broken_apis)} broken")
        
        if broken_apis:
            logger.warning("⚠️  BROKEN APIs detected:")
            for data_type, endpoint in broken_apis:
                logger.warning(f"   - {data_type}: {endpoint}")
                
        return {
            'working_apis': working_apis,
            'broken_apis': broken_apis,
            'total_tested': len(self.api_endpoints),
            'success_rate': len(working_apis) / len(self.api_endpoints)
        }

if __name__ == "__main__":
    collector = RealTimeDataCollector()
    
    # Test APIs first
    api_test_results = collector.test_apis()
    
    if api_test_results['success_rate'] > 0:
        logger.info(f"🎯 {api_test_results['success_rate']*100:.0f}% of APIs are working - starting collection")
        collector.run_continuous_collection()
    else:
        logger.error("❌ No working APIs found - cannot start collection")
        logger.error("Please check if the backend server is running on http://localhost:8000")