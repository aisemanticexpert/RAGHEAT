"""
Redis JSON Storage with 5-Second Timeframes
Stores market data in JSON format with automatic TTL and timeframe management
"""
import json
import redis
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisJSONStorage:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = None
        self.host = host
        self.port = port
        self.db = db
        self.timeframe_seconds = 5
        self.default_ttl = 3600  # 1 hour TTL for stored data
        
    def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.host, 
                port=self.port, 
                db=self.db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False
    
    def generate_timeframe_key(self, data_type: str, timestamp: Optional[datetime] = None) -> str:
        """Generate Redis key with 5-second timeframe"""
        if not timestamp:
            timestamp = datetime.utcnow()
            
        # Round down to nearest 5-second interval
        seconds = timestamp.second
        rounded_seconds = (seconds // self.timeframe_seconds) * self.timeframe_seconds
        
        timeframe_timestamp = timestamp.replace(second=rounded_seconds, microsecond=0)
        timeframe_key = timeframe_timestamp.strftime("%Y%m%d_%H%M%S")
        
        return f"ragheat:{data_type}:{timeframe_key}"
    
    def store_market_data(self, data_type: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store market data with 5-second timeframe key"""
        if not self.redis_client:
            logger.error("Redis client not connected")
            return False
            
        try:
            # Add storage metadata
            storage_data = {
                'data': data,
                'stored_at': datetime.utcnow().isoformat(),
                'timeframe': '5s',
                'data_type': data_type
            }
            
            key = self.generate_timeframe_key(data_type)
            ttl_seconds = ttl or self.default_ttl
            
            # Store as JSON with TTL
            self.redis_client.setex(
                key, 
                ttl_seconds,
                json.dumps(storage_data)
            )
            
            # Also store latest data pointer
            latest_key = f"ragheat:{data_type}:latest"
            self.redis_client.setex(latest_key, ttl_seconds, key)
            
            logger.info(f"📦 Stored {data_type} data in {key} (TTL: {ttl_seconds}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing {data_type} data: {e}")
            return False
    
    def get_latest_data(self, data_type: str) -> Optional[Dict[str, Any]]:
        """Get the latest stored data for a data type"""
        if not self.redis_client:
            return None
            
        try:
            latest_key = f"ragheat:{data_type}:latest"
            actual_key = self.redis_client.get(latest_key)
            
            if not actual_key:
                logger.warning(f"No latest data found for {data_type}")
                return None
                
            data_json = self.redis_client.get(actual_key)
            if data_json:
                stored_data = json.loads(data_json)
                logger.info(f"📖 Retrieved latest {data_type} data from {actual_key}")
                return stored_data
                
        except Exception as e:
            logger.error(f"❌ Error retrieving latest {data_type} data: {e}")
            
        return None
    
    def get_timeframe_data(self, data_type: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get data for a specific 5-second timeframe"""
        if not self.redis_client:
            return None
            
        try:
            key = self.generate_timeframe_key(data_type, timestamp)
            data_json = self.redis_client.get(key)
            
            if data_json:
                stored_data = json.loads(data_json)
                logger.info(f"📖 Retrieved {data_type} data for timeframe {key}")
                return stored_data
                
        except Exception as e:
            logger.error(f"❌ Error retrieving timeframe data: {e}")
            
        return None
    
    def get_recent_data(self, data_type: str, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get all data for the last N minutes"""
        if not self.redis_client:
            return []
            
        try:
            # Generate keys for the last N minutes
            recent_data = []
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=minutes)
            
            current_time = start_time
            while current_time <= end_time:
                key = self.generate_timeframe_key(data_type, current_time)
                data_json = self.redis_client.get(key)
                
                if data_json:
                    stored_data = json.loads(data_json)
                    recent_data.append(stored_data)
                
                current_time += timedelta(seconds=self.timeframe_seconds)
            
            logger.info(f"📊 Retrieved {len(recent_data)} timeframes for {data_type} (last {minutes} minutes)")
            return recent_data
            
        except Exception as e:
            logger.error(f"❌ Error retrieving recent data: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        if not self.redis_client:
            return {}
            
        try:
            # Get all ragheat keys
            all_keys = self.redis_client.keys("ragheat:*")
            data_types = set()
            
            for key in all_keys:
                if ':latest' not in key:
                    parts = key.split(':')
                    if len(parts) >= 2:
                        data_types.add(parts[1])
            
            stats = {
                'total_keys': len(all_keys),
                'data_types': list(data_types),
                'latest_keys': len([k for k in all_keys if ':latest' in k]),
                'timeframe_keys': len([k for k in all_keys if ':latest' not in k]),
                'redis_info': self.redis_client.info('memory')
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting storage stats: {e}")
            return {}
    
    def clear_data_type(self, data_type: str) -> int:
        """Clear all data for a specific data type"""
        if not self.redis_client:
            return 0
            
        try:
            pattern = f"ragheat:{data_type}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"🗑️  Cleared {deleted} keys for {data_type}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Error clearing {data_type} data: {e}")
            return 0

# Singleton instance for global use
redis_storage = RedisJSONStorage()

if __name__ == "__main__":
    # Test the storage system
    storage = RedisJSONStorage()
    
    if storage.connect():
        # Test storing some data
        test_data = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "test_value": 123.45,
            "test_array": [1, 2, 3]
        }
        
        storage.store_market_data("test", test_data)
        
        # Retrieve it
        latest = storage.get_latest_data("test")
        print("Latest data:", latest)
        
        # Get stats
        stats = storage.get_storage_stats()
        print("Storage stats:", stats)