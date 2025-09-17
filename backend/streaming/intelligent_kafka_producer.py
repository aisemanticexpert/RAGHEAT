"""
Intelligent Kafka Producer with Market Hours Detection
Automatically switches between real-time and synthetic data based on market status
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

# Import our services
from services.market_hours_detector import data_source_manager, get_market_hours_info
from data_pipeline.synthetic_market_generator import synthetic_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentMarketKafkaProducer:
    """
    Enhanced Kafka producer that intelligently switches data sources
    """
    
    def __init__(self):
        self.producer = None
        self.topics = {
            'market_overview': 'ragheat.market.overview',
            'heat_distribution': 'ragheat.market.heat',
            'neo4j_graph': 'ragheat.graph.neo4j',
            'market_analysis': 'ragheat.analysis.signals',
            'heat_diffusion': 'ragheat.physics.heat'
        }
        self.data_source_manager = data_source_manager
        self.last_market_status = None
        self.streaming_active = False
        
    def connect_kafka(self) -> bool:
        """Connect to Kafka broker with enhanced configuration"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None,
                retries=5,
                retry_backoff_ms=1000,
                max_in_flight_requests_per_connection=1,
                enable_idempotence=True,
                compression_type='gzip',
                batch_size=16384,
                linger_ms=10
            )
            logger.info("✅ Connected to Kafka broker with intelligent data switching")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            return False
    
    async def get_comprehensive_market_data(self) -> Dict[str, Any]:
        """Get comprehensive market data from appropriate source"""
        try:
            # Get market status
            market_info = get_market_hours_info()
            
            # Get market overview data
            market_data = await self.data_source_manager.get_market_data('market_overview')
            
            # Get Neo4j graph data
            if market_info['data_source_recommendation'] == 'synthetic':
                neo4j_data = synthetic_generator.get_neo4j_graph_data()
            else:
                neo4j_data = await self.data_source_manager.get_market_data('neo4j_stocks')
            
            # Generate analysis data
            analysis_data = self.generate_market_analysis(market_data, neo4j_data, market_info)
            
            # Generate heat diffusion data
            heat_data = self.generate_heat_diffusion_data(market_data, neo4j_data)
            
            return {
                'market_overview': market_data,
                'neo4j_graph': neo4j_data,
                'market_analysis': analysis_data,
                'heat_diffusion': heat_data,
                'market_info': market_info
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting comprehensive market data: {e}")
            return {}
    
    def generate_market_analysis(self, market_data: Dict, neo4j_data: Dict, market_info: Dict) -> Dict[str, Any]:
        """Generate market analysis with buy/sell signals"""
        try:
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "market_sentiment": market_data.get('market_overview', {}).get('market_return', 0),
                "volatility_regime": "high" if abs(market_data.get('market_overview', {}).get('market_volatility', 0)) > 2 else "normal",
                "data_source": market_info.get('data_source_recommendation', 'unknown'),
                "signals": [],
                "sector_recommendations": [],
                "heated_sectors": market_data.get('heated_sectors', []),
                "risk_assessment": self.assess_market_risk(market_data)
            }
            
            # Generate signals for heated sectors/stocks
            if 'heated_sectors' in market_data:
                for heated in market_data['heated_sectors']:
                    signal = {
                        "type": "sector_heat",
                        "symbol": heated['sector'],
                        "signal": "BUY" if heated['heat_level'] > 0.3 else "WATCH",
                        "confidence": min(abs(heated['heat_level']) * 100, 95),
                        "reason": heated.get('reason', 'Heat diffusion detected'),
                        "heat_level": heated['heat_level'],
                        "timestamp": datetime.now().isoformat()
                    }
                    analysis['signals'].append(signal)
            
            # Analyze individual stocks from market data
            if 'stocks' in market_data:
                for symbol, stock_data in market_data['stocks'].items():
                    if abs(stock_data.get('heat_level', 0)) > 0.5:
                        signal_type = "STRONG_BUY" if stock_data['heat_level'] > 0.7 else "BUY" if stock_data['heat_level'] > 0.3 else "SELL"
                        
                        signal = {
                            "type": "stock_heat",
                            "symbol": symbol,
                            "signal": signal_type,
                            "confidence": min(abs(stock_data['heat_level']) * 80 + 20, 95),
                            "price": stock_data.get('price', 0),
                            "change_percent": stock_data.get('change_percent', 0),
                            "heat_level": stock_data['heat_level'],
                            "sector": stock_data.get('sector', 'unknown'),
                            "volume": stock_data.get('volume', 0),
                            "reason": f"Heat level {stock_data['heat_level']:.2f} with volume spike" if stock_data.get('volume', 0) > 2000000 else f"Heat level {stock_data['heat_level']:.2f}",
                            "timestamp": datetime.now().isoformat()
                        }
                        analysis['signals'].append(signal)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error generating market analysis: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def assess_market_risk(self, market_data: Dict) -> Dict[str, Any]:
        """Assess overall market risk"""
        try:
            market_overview = market_data.get('market_overview', {})
            volatility = market_overview.get('market_volatility', 0)
            heat_index = market_overview.get('market_heat_index', 0)
            
            # Risk scoring
            vol_risk = min(abs(volatility) / 5 * 100, 100)  # Normalize volatility risk
            heat_risk = abs(heat_index)  # Heat index is already 0-100
            
            overall_risk = (vol_risk + heat_risk) / 2
            
            if overall_risk < 30:
                risk_level = "LOW"
            elif overall_risk < 60:
                risk_level = "MODERATE"
            elif overall_risk < 80:
                risk_level = "HIGH"
            else:
                risk_level = "EXTREME"
            
            return {
                "overall_risk_score": round(overall_risk, 1),
                "risk_level": risk_level,
                "volatility_risk": round(vol_risk, 1),
                "heat_risk": round(heat_risk, 1),
                "recommendation": "REDUCE_EXPOSURE" if risk_level in ["HIGH", "EXTREME"] else "NORMAL_TRADING",
                "factors": {
                    "market_volatility": volatility,
                    "heat_index": heat_index,
                    "heated_sectors": len(market_data.get('heated_sectors', []))
                }
            }
            
        except Exception as e:
            return {"error": str(e), "overall_risk_score": 50, "risk_level": "UNKNOWN"}
    
    def generate_heat_diffusion_data(self, market_data: Dict, neo4j_data: Dict) -> Dict[str, Any]:
        """Generate heat diffusion physics data"""
        try:
            # Extract heat levels from sectors and stocks
            sector_heat = {}
            stock_heat = {}
            
            if 'sectors' in market_data:
                for sector, data in market_data['sectors'].items():
                    sector_heat[sector] = data.get('heat_level', 0)
            
            if 'stocks' in market_data:
                for symbol, data in market_data['stocks'].items():
                    stock_heat[symbol] = data.get('heat_level', 0)
            
            # Calculate heat gradients and flow
            heat_gradients = []
            for sector, heat in sector_heat.items():
                if abs(heat) > 0.1:
                    heat_gradients.append({
                        "entity": sector,
                        "type": "sector",
                        "heat_level": heat,
                        "gradient": heat / 0.5,  # Normalize gradient
                        "flow_direction": "outward" if heat > 0 else "inward"
                    })
            
            # Calculate diffusion rate
            total_heat = sum(abs(h) for h in sector_heat.values())
            diffusion_rate = min(total_heat / len(sector_heat) if sector_heat else 0, 1.0)
            
            # Heat sources and sinks
            heat_sources = [s for s, h in sector_heat.items() if h > 0.3]
            heat_sinks = [s for s, h in sector_heat.items() if h < -0.3]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "diffusion_rate": round(diffusion_rate, 3),
                "total_heat_energy": round(total_heat, 3),
                "sector_heat_map": sector_heat,
                "stock_heat_map": stock_heat,
                "heat_gradients": heat_gradients,
                "heat_sources": heat_sources,
                "heat_sinks": heat_sinks,
                "equilibrium_status": "stable" if diffusion_rate < 0.3 else "unstable",
                "physics_model": {
                    "equation": "∂H/∂t = α∇²H + S(x,t)",
                    "thermal_diffusivity": 0.1,
                    "time_step": 5.0,
                    "boundary_conditions": "zero_flux"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating heat diffusion data: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def send_to_kafka(self, topic: str, key: str, data: Dict[str, Any]) -> bool:
        """Send data to Kafka topic with error handling"""
        if not data or not self.producer:
            return False
        
        try:
            # Add metadata
            enhanced_data = {
                **data,
                "kafka_metadata": {
                    "producer": "intelligent_kafka_producer",
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                    "data_size_bytes": len(json.dumps(data, default=str)),
                    "compression": "gzip"
                }
            }
            
            future = self.producer.send(topic, key=key, value=enhanced_data)
            record_metadata = future.get(timeout=10)
            
            logger.info(f"✅ Sent to {topic}: {len(str(enhanced_data))} bytes (offset: {record_metadata.offset})")
            return True
            
        except KafkaError as e:
            logger.error(f"❌ Kafka send error for topic {topic}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error sending to {topic}: {e}")
            return False
    
    def log_market_status_change(self, new_status: Dict[str, Any]):
        """Log when market status changes"""
        if self.last_market_status != new_status['status']:
            logger.info(f"📈 Market status changed: {self.last_market_status} → {new_status['status']}")
            logger.info(f"📊 Data source: {new_status.get('data_source_recommendation', 'unknown')}")
            
            if new_status.get('next_open'):
                logger.info(f"⏰ Next market open: {new_status['next_open']}")
            if new_status.get('next_close'):
                logger.info(f"⏰ Next market close: {new_status['next_close']}")
                
            self.last_market_status = new_status['status']
    
    async def stream_intelligent_market_data(self):
        """Main streaming loop with intelligent data source switching"""
        if not self.connect_kafka():
            logger.error("Cannot start streaming without Kafka connection")
            return
        
        logger.info("🧠 Starting intelligent market data streaming (5-second intervals)")
        logger.info("🔄 Automatic switching between real-time and synthetic data")
        
        self.streaming_active = True
        cycle_count = 0
        
        while self.streaming_active:
            try:
                cycle_count += 1
                current_time = datetime.now()
                timestamp_key = current_time.strftime("%Y%m%d_%H%M%S")
                
                # Get comprehensive market data
                comprehensive_data = await self.get_comprehensive_market_data()
                
                if not comprehensive_data:
                    logger.warning("⚠️ No data received, skipping cycle")
                    await asyncio.sleep(5)
                    continue
                
                # Log market status changes
                market_info = comprehensive_data.get('market_info', {})
                self.log_market_status_change(market_info)
                
                # Send market overview
                if 'market_overview' in comprehensive_data:
                    self.send_to_kafka(
                        self.topics['market_overview'],
                        f"market_{timestamp_key}",
                        comprehensive_data['market_overview']
                    )
                
                # Send Neo4j graph data
                if 'neo4j_graph' in comprehensive_data:
                    self.send_to_kafka(
                        self.topics['neo4j_graph'],
                        f"graph_{timestamp_key}",
                        comprehensive_data['neo4j_graph']
                    )
                
                # Send market analysis
                if 'market_analysis' in comprehensive_data:
                    self.send_to_kafka(
                        self.topics['market_analysis'],
                        f"analysis_{timestamp_key}",
                        comprehensive_data['market_analysis']
                    )
                
                # Send heat diffusion data
                if 'heat_diffusion' in comprehensive_data:
                    self.send_to_kafka(
                        self.topics['heat_diffusion'],
                        f"heat_{timestamp_key}",
                        comprehensive_data['heat_diffusion']
                    )
                
                # Log cycle completion
                data_source = market_info.get('data_source_recommendation', 'unknown')
                signals_count = len(comprehensive_data.get('market_analysis', {}).get('signals', []))
                heated_sectors = len(comprehensive_data.get('market_overview', {}).get('heated_sectors', []))
                
                logger.info(f"🚀 Cycle {cycle_count} completed - Source: {data_source}, Signals: {signals_count}, Heated Sectors: {heated_sectors}")
                
                # Wait 5 seconds for next cycle
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("⏹️ Streaming stopped by user")
                self.streaming_active = False
                break
            except Exception as e:
                logger.error(f"❌ Error in streaming loop: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    def stop_streaming(self):
        """Stop the streaming loop"""
        self.streaming_active = False
        logger.info("🛑 Streaming stop requested")
    
    def close(self):
        """Close Kafka producer"""
        self.streaming_active = False
        if self.producer:
            self.producer.flush()  # Ensure all messages are sent
            self.producer.close()
            logger.info("🔌 Intelligent Kafka producer closed")
    
    def get_producer_stats(self) -> Dict[str, Any]:
        """Get producer statistics and status"""
        market_info = get_market_hours_info()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "streaming_active": self.streaming_active,
            "kafka_connected": self.producer is not None,
            "topics_configured": len(self.topics),
            "market_status": market_info,
            "data_source": market_info.get('data_source_recommendation', 'unknown'),
            "producer_config": {
                "compression": "gzip",
                "batch_size": 16384,
                "linger_ms": 10,
                "retries": 5,
                "enable_idempotence": True
            }
        }


# Global producer instance
intelligent_producer = IntelligentMarketKafkaProducer()


# Convenience functions
async def start_intelligent_streaming():
    """Start the intelligent streaming service"""
    await intelligent_producer.stream_intelligent_market_data()


def get_streaming_status() -> Dict[str, Any]:
    """Get current streaming status"""
    return intelligent_producer.get_producer_stats()


if __name__ == "__main__":
    async def main():
        producer = IntelligentMarketKafkaProducer()
        try:
            await producer.stream_intelligent_market_data()
        finally:
            producer.close()
    
    # Run the streaming service
    asyncio.run(main())