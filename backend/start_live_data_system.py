#!/usr/bin/env python3
"""
Start Live Data System
Launch the complete synthetic data streaming system
"""
import asyncio
import logging
import sys
import signal
import json
from pathlib import Path
from datetime import datetime

# Import our services
from services.synthetic_data_generator import SyntheticDataGenerator
from services.real_market_data_downloader import RealMarketDataDownloader
from services.realtime_analytics_engine import RealTimeAnalyticsEngine

class LiveDataSystem:
    def __init__(self):
        self.data_generator = SyntheticDataGenerator()
        self.market_downloader = RealMarketDataDownloader()
        self.analytics_engine = RealTimeAnalyticsEngine()
        
        self.is_running = False
        self.data_dir = Path("data")
        
        # Ensure directories exist
        for subdir in ["synthetic_stocks", "real_market_data", "processed", "cache"]:
            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "start_time": None,
            "data_points_generated": 0,
            "json_files_created": 0,
            "real_data_downloads": 0,
            "analytics_signals": 0
        }
    
    async def initialize_system(self):
        """Initialize all components"""
        logging.info("🚀 Initializing Live Data System...")
        
        # Generate initial synthetic data files
        logging.info("📊 Generating initial synthetic data...")
        initial_data = self.data_generator.generate_batch_data(50)
        
        # Save as individual JSON files
        for i, batch in enumerate(initial_data):
            timestamp = batch['timestamp'].replace(':', '-').replace('.', '-')
            filename = f"market_data_{timestamp}_{i:04d}.json"
            filepath = self.data_dir / "synthetic_stocks" / filename
            
            with open(filepath, 'w') as f:
                json.dump(batch, f, indent=2, default=str)
            
            self.stats["json_files_created"] += 1
        
        self.stats["data_points_generated"] += len(initial_data)
        
        logging.info(f"✅ Generated {len(initial_data)} initial data files")
        
        # Start real market data download
        logging.info("🌐 Starting real market data download...")
        try:
            asyncio.create_task(self.download_real_data())
        except Exception as e:
            logging.warning(f"Real data download failed: {e}, continuing with synthetic data")
        
        logging.info("🔧 System initialization complete!")
    
    async def download_real_data(self):
        """Download real market data in background"""
        try:
            # Download current market data
            yahoo_data = await self.market_downloader.download_yahoo_finance_data(
                ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'], 
                period='1d', 
                interval='5m'
            )
            
            self.stats["real_data_downloads"] += len(yahoo_data)
            logging.info(f"📈 Downloaded real data for {len(yahoo_data)} symbols")
            
            # Download news and sentiment
            news_data = await self.market_downloader.download_news_sentiment_data()
            logging.info(f"📰 Downloaded news from {len(news_data)} sources")
            
            # Download economic indicators
            economic_data = await self.market_downloader.download_economic_indicators()
            logging.info(f"📊 Downloaded {len(economic_data)} economic indicators")
            
        except Exception as e:
            logging.error(f"Real data download error: {e}")
    
    async def start_streaming(self):
        """Start the live streaming loop"""
        logging.info("🔄 Starting live data streaming...")
        self.is_running = True
        self.stats["start_time"] = datetime.now()
        
        stream_count = 0
        
        while self.is_running:
            try:
                # Generate new market tick
                market_batch = self.data_generator.generate_batch_data(1)
                
                if market_batch:
                    market_tick = market_batch[0]
                    
                    # Update analytics engine with new data
                    for symbol, stock_data in market_tick['stocks'].items():
                        self.analytics_engine.update_data(
                            symbol=symbol,
                            price=stock_data['price'],
                            volume=stock_data['volume'],
                            heat_score=stock_data['heat_score'],
                            volatility=stock_data['volatility']
                        )
                    
                    # Save streaming data
                    if stream_count % 10 == 0:  # Save every 10th tick
                        timestamp = market_tick['timestamp'].replace(':', '-').replace('.', '-')
                        filename = f"stream_data_{timestamp}_{stream_count:06d}.json"
                        filepath = self.data_dir / "processed" / filename
                        
                        # Add analytics data
                        enhanced_data = market_tick.copy()
                        enhanced_data['analytics'] = {
                            'active_signals': self.analytics_engine.get_active_signals(),
                            'signal_summary': self.analytics_engine.get_signal_summary(),
                            'heat_analysis': self.analytics_engine.get_heat_analysis()
                        }
                        
                        with open(filepath, 'w') as f:
                            json.dump(enhanced_data, f, indent=2, default=str)
                    
                    self.stats["data_points_generated"] += 1
                    self.stats["analytics_signals"] += len(self.analytics_engine.get_active_signals())
                    stream_count += 1
                    
                    # Log progress periodically
                    if stream_count % 100 == 0:
                        signals = self.analytics_engine.get_signal_summary()
                        logging.info(f"📊 Streamed {stream_count} ticks | Active signals: {signals.get('total_signals', 0)} | Regime: {market_tick.get('market_regime', 'unknown')}")
                
                # Stream at realistic intervals
                await asyncio.sleep(0.5)  # 500ms intervals = 2 ticks per second
                
            except Exception as e:
                logging.error(f"Streaming error: {e}")
                await asyncio.sleep(1.0)
    
    async def run_analytics_loop(self):
        """Run analytics processing loop"""
        while self.is_running:
            try:
                # Run fast analytics
                await self.analytics_engine.fast_analysis()
                await asyncio.sleep(1.0)
                
                # Run medium analytics every 5 seconds
                if self.stats["data_points_generated"] % 10 == 0:
                    await self.analytics_engine.medium_analysis()
                
                # Run slow analytics every 30 seconds
                if self.stats["data_points_generated"] % 60 == 0:
                    await self.analytics_engine.slow_analysis()
                    
                    # Save analytics summary
                    analytics_summary = {
                        "timestamp": datetime.now().isoformat(),
                        "system_stats": self.get_system_stats(),
                        "signal_summary": self.analytics_engine.get_signal_summary(),
                        "active_signals": self.analytics_engine.get_active_signals(),
                        "heat_analysis": self.analytics_engine.get_heat_analysis(),
                        "analytics_status": self.analytics_engine.get_analytics_status()
                    }
                    
                    filepath = self.data_dir / "processed" / f"analytics_summary_{int(datetime.now().timestamp())}.json"
                    with open(filepath, 'w') as f:
                        json.dump(analytics_summary, f, indent=2, default=str)
                
            except Exception as e:
                logging.error(f"Analytics loop error: {e}")
                await asyncio.sleep(5.0)
    
    def get_system_stats(self):
        """Get current system statistics"""
        runtime = (datetime.now() - self.stats["start_time"]).total_seconds() if self.stats["start_time"] else 0
        
        return {
            "runtime_seconds": runtime,
            "data_points_generated": self.stats["data_points_generated"],
            "json_files_created": self.stats["json_files_created"],
            "real_data_downloads": self.stats["real_data_downloads"],
            "analytics_signals": self.stats["analytics_signals"],
            "generation_rate": self.stats["data_points_generated"] / runtime if runtime > 0 else 0,
            "active_symbols": len(self.data_generator.stocks),
            "current_regime": getattr(self.data_generator, 'current_regime', 'unknown')
        }
    
    async def start(self):
        """Start the complete system"""
        try:
            # Initialize
            await self.initialize_system()
            
            # Start background tasks
            streaming_task = asyncio.create_task(self.start_streaming())
            analytics_task = asyncio.create_task(self.run_analytics_loop())
            
            logging.info("🎯 Live Data System is running!")
            logging.info("📊 Real-time data generation active")
            logging.info("🔍 Analytics engine processing")
            logging.info("📁 Data being saved to JSON files")
            logging.info("🌐 Real market data downloading")
            logging.info("\n🚀 System Status:")
            logging.info(f"   - {len(self.data_generator.stocks)} stocks tracked")
            logging.info(f"   - Data generated every 500ms")
            logging.info(f"   - Analytics running at multiple intervals")
            logging.info(f"   - JSON files saved in {self.data_dir}/")
            logging.info("\n⏹️  Press Ctrl+C to stop the system\n")
            
            # Wait for both tasks
            await asyncio.gather(streaming_task, analytics_task)
            
        except KeyboardInterrupt:
            logging.info("\n🛑 Stopping Live Data System...")
            await self.stop()
        except Exception as e:
            logging.error(f"System error: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the system gracefully"""
        self.is_running = False
        
        # Final statistics
        final_stats = self.get_system_stats()
        
        logging.info("📊 Final Statistics:")
        logging.info(f"   - Runtime: {final_stats['runtime_seconds']:.1f} seconds")
        logging.info(f"   - Data points generated: {final_stats['data_points_generated']}")
        logging.info(f"   - JSON files created: {final_stats['json_files_created']}")
        logging.info(f"   - Real data downloads: {final_stats['real_data_downloads']}")
        logging.info(f"   - Analytics signals: {final_stats['analytics_signals']}")
        logging.info(f"   - Generation rate: {final_stats['generation_rate']:.2f} points/sec")
        
        # Save final stats
        filepath = self.data_dir / "processed" / f"final_stats_{int(datetime.now().timestamp())}.json"
        with open(filepath, 'w') as f:
            json.dump(final_stats, f, indent=2, default=str)
        
        logging.info(f"✅ System stopped. Data saved in {self.data_dir}/")

async def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('live_data_system.log')
        ]
    )
    
    # Create and start system
    system = LiveDataSystem()
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        logging.info(f"\n🔔 Received signal {signum}")
        system.is_running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await system.start()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🎯 RAGHeat Live Data System")
    print("=" * 50)
    print("🔥 Continuous synthetic market data generation")
    print("📊 Real-time analytics and signal processing")
    print("🌐 Real market data integration")
    print("📁 JSON file streaming and storage")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)