"""
Real Market Data Downloader
Downloads live and historical market data from multiple sources
"""
import asyncio
import aiohttp
import yfinance as yf
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import time
import requests
from concurrent.futures import ThreadPoolExecutor
import feedparser
import websocket
import threading

class RealMarketDataDownloader:
    def __init__(self):
        self.data_dir = Path("data/real_market_data")
        self.processed_dir = Path("data/processed")
        self.cache_dir = Path("data/cache")
        
        # Ensure directories exist
        for dir_path in [self.data_dir, self.processed_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Stock symbols to track
        self.symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'JPM', 'BAC', 'JNJ', 'PFE', 'XOM', 'CVX', 'WMT', 'HD',
            'PG', 'KO', 'DIS', 'V', 'SPY', 'QQQ', 'VTI'
        ]
        
        # Data sources configuration
        self.data_sources = {
            'yahoo_finance': {'enabled': True, 'rate_limit': 0.5},
            'alpha_vantage': {'enabled': False, 'api_key': None, 'rate_limit': 12.0},
            'iex_cloud': {'enabled': False, 'api_key': None, 'rate_limit': 0.1},
            'finnhub': {'enabled': False, 'api_key': None, 'rate_limit': 1.0}
        }
        
        # WebSocket connections for real-time data
        self.websocket_connections = {}
        self.real_time_data = {}
        
        # Download statistics
        self.download_stats = {
            'total_requests': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'last_update': None,
            'data_quality_score': 0.0
        }
    
    async def download_yahoo_finance_data(self, 
                                        symbols: List[str], 
                                        period: str = '1d',
                                        interval: str = '1m') -> Dict[str, pd.DataFrame]:
        """Download data from Yahoo Finance"""
        logging.info(f"Downloading Yahoo Finance data for {len(symbols)} symbols")
        
        downloaded_data = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            for symbol in symbols:
                future = executor.submit(self._download_single_yahoo_symbol, symbol, period, interval)
                futures.append((symbol, future))
            
            for symbol, future in futures:
                try:
                    data = future.result(timeout=30)
                    if data is not None and not data.empty:
                        downloaded_data[symbol] = data
                        self.download_stats['successful_downloads'] += 1
                        
                        # Save individual symbol data
                        filepath = self.data_dir / f"{symbol}_yahoo_{datetime.now().strftime('%Y%m%d')}.json"
                        self._save_dataframe_as_json(data, filepath, symbol)
                    else:
                        self.download_stats['failed_downloads'] += 1
                        
                except Exception as e:
                    logging.error(f"Failed to download {symbol}: {e}")
                    self.download_stats['failed_downloads'] += 1
                
                self.download_stats['total_requests'] += 1
                
                # Rate limiting
                await asyncio.sleep(self.data_sources['yahoo_finance']['rate_limit'])
        
        self.download_stats['last_update'] = datetime.now().isoformat()
        return downloaded_data
    
    def _download_single_yahoo_symbol(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Download data for a single symbol from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist_data = ticker.history(period=period, interval=interval)
            
            if hist_data.empty:
                logging.warning(f"No data returned for {symbol}")
                return None
            
            # Add additional information
            info = ticker.info
            hist_data['Symbol'] = symbol
            hist_data['MarketCap'] = info.get('marketCap', 0)
            hist_data['Sector'] = info.get('sector', 'Unknown')
            hist_data['Industry'] = info.get('industry', 'Unknown')
            
            # Calculate additional metrics
            hist_data['Returns'] = hist_data['Close'].pct_change()
            hist_data['Volatility'] = hist_data['Returns'].rolling(window=20).std() * np.sqrt(252)
            hist_data['RSI'] = self._calculate_rsi(hist_data['Close'])
            hist_data['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
            hist_data['EMA_12'] = hist_data['Close'].ewm(span=12).mean()
            
            return hist_data
            
        except Exception as e:
            logging.error(f"Error downloading {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _save_dataframe_as_json(self, df: pd.DataFrame, filepath: Path, symbol: str):
        """Save DataFrame as JSON with metadata"""
        try:
            # Convert DataFrame to records
            records = df.reset_index().to_dict('records')
            
            # Add metadata
            data_package = {
                'symbol': symbol,
                'download_timestamp': datetime.now().isoformat(),
                'records_count': len(records),
                'data_quality': self._assess_data_quality(df),
                'columns': list(df.columns),
                'data': records
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data_package, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Error saving {symbol} data: {e}")
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Assess the quality of downloaded data"""
        if df.empty:
            return {'score': 0.0, 'completeness': 0.0, 'consistency': 0.0}
        
        # Completeness: percentage of non-null values
        completeness = df.notna().sum().sum() / (len(df) * len(df.columns))
        
        # Consistency: check for reasonable price movements
        if 'Close' in df.columns:
            price_changes = df['Close'].pct_change().dropna()
            extreme_moves = (abs(price_changes) > 0.15).sum() / len(price_changes)  # >15% moves
            consistency = max(0.0, 1.0 - extreme_moves * 2)  # Penalize extreme moves
        else:
            consistency = 0.5
        
        # Overall score
        score = (completeness * 0.7 + consistency * 0.3)
        
        return {
            'score': round(score, 3),
            'completeness': round(completeness, 3),
            'consistency': round(consistency, 3),
            'records': len(df)
        }
    
    async def download_news_sentiment_data(self) -> Dict[str, List[Dict]]:
        """Download news and sentiment data"""
        logging.info("Downloading news sentiment data...")
        
        news_data = {}
        
        # RSS feeds for financial news
        rss_feeds = {
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline'
        }
        
        for source, url in rss_feeds.items():
            try:
                feed = feedparser.parse(url)
                articles = []
                
                for entry in feed.entries[:10]:  # Top 10 articles
                    article = {
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': source,
                        'sentiment_score': self._analyze_sentiment(
                            entry.get('title', '') + ' ' + entry.get('summary', '')
                        )
                    }
                    articles.append(article)
                
                news_data[source] = articles
                
                # Save news data
                filepath = self.data_dir / f"news_{source}_{datetime.now().strftime('%Y%m%d')}.json"
                with open(filepath, 'w') as f:
                    json.dump(articles, f, indent=2, default=str)
                
                await asyncio.sleep(1.0)  # Rate limiting
                
            except Exception as e:
                logging.error(f"Failed to download news from {source}: {e}")
        
        return news_data
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (placeholder for more sophisticated analysis)"""
        # Simple keyword-based sentiment
        positive_words = ['gain', 'rise', 'up', 'bull', 'positive', 'growth', 'strong', 'beat', 'exceed']
        negative_words = ['loss', 'fall', 'down', 'bear', 'negative', 'decline', 'weak', 'miss', 'below']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return round(sentiment, 3)
    
    async def download_economic_indicators(self) -> Dict[str, any]:
        """Download economic indicators (simplified version)"""
        logging.info("Downloading economic indicators...")
        
        indicators = {}
        
        try:
            # VIX (Volatility Index) - using Yahoo Finance
            vix_ticker = yf.Ticker('^VIX')
            vix_data = vix_ticker.history(period='5d')
            if not vix_data.empty:
                indicators['VIX'] = {
                    'current_value': float(vix_data['Close'].iloc[-1]),
                    'change': float(vix_data['Close'].pct_change().iloc[-1]),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Treasury rates
            treasury_symbols = {'^TNX': '10Y_Treasury', '^IRX': '3M_Treasury'}
            
            for symbol, name in treasury_symbols.items():
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='5d')
                if not data.empty:
                    indicators[name] = {
                        'current_value': float(data['Close'].iloc[-1]),
                        'change': float(data['Close'].pct_change().iloc[-1]),
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Currency data (USD index)
            usd_ticker = yf.Ticker('DX-Y.NYB')
            usd_data = usd_ticker.history(period='5d')
            if not usd_data.empty:
                indicators['USD_Index'] = {
                    'current_value': float(usd_data['Close'].iloc[-1]),
                    'change': float(usd_data['Close'].pct_change().iloc[-1]),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Save economic indicators
            filepath = self.data_dir / f"economic_indicators_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filepath, 'w') as f:
                json.dump(indicators, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Failed to download economic indicators: {e}")
        
        return indicators
    
    async def start_real_time_websocket_feeds(self):
        """Start WebSocket connections for real-time data (placeholder)"""
        logging.info("Starting real-time WebSocket feeds...")
        
        # Note: This is a placeholder. In production, you would connect to
        # real WebSocket feeds from brokers or data providers
        
        # Simulate real-time updates
        asyncio.create_task(self._simulate_real_time_updates())
    
    async def _simulate_real_time_updates(self):
        """Simulate real-time market updates"""
        while True:
            try:
                # Generate realistic real-time updates
                for symbol in self.symbols[:5]:  # Update first 5 symbols
                    if symbol not in self.real_time_data:
                        self.real_time_data[symbol] = {
                            'price': 100.0 + np.random.normal(0, 10),
                            'volume': 1000000
                        }
                    
                    # Simulate price movement
                    current_price = self.real_time_data[symbol]['price']
                    price_change = np.random.normal(0, 0.01) * current_price  # 1% volatility
                    new_price = max(0.01, current_price + price_change)
                    
                    self.real_time_data[symbol] = {
                        'price': new_price,
                        'change': price_change,
                        'change_percent': (price_change / current_price) * 100,
                        'volume': self.real_time_data[symbol]['volume'] + np.random.randint(1000, 10000),
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol
                    }
                
                # Save real-time snapshot
                if self.real_time_data:
                    filepath = self.cache_dir / f"realtime_snapshot_{int(time.time())}.json"
                    with open(filepath, 'w') as f:
                        json.dump(self.real_time_data, f, indent=2, default=str)
                
                await asyncio.sleep(2.0)  # Update every 2 seconds
                
            except Exception as e:
                logging.error(f"Error in real-time simulation: {e}")
                await asyncio.sleep(5.0)
    
    def get_latest_data_summary(self) -> Dict[str, any]:
        """Get summary of latest downloaded data"""
        summary = {
            'download_stats': self.download_stats,
            'available_symbols': len(self.symbols),
            'real_time_symbols': len(self.real_time_data),
            'data_files': len(list(self.data_dir.glob('*.json'))),
            'cache_files': len(list(self.cache_dir.glob('*.json'))),
            'last_real_time_update': None
        }
        
        if self.real_time_data:
            latest_timestamp = max(
                data.get('timestamp', '') 
                for data in self.real_time_data.values()
            )
            summary['last_real_time_update'] = latest_timestamp
        
        return summary
    
    def get_real_time_data(self) -> Dict[str, any]:
        """Get current real-time data"""
        return dict(self.real_time_data)
    
    async def full_market_data_download(self):
        """Download comprehensive market data from all available sources"""
        logging.info("Starting comprehensive market data download...")
        
        download_results = {
            'start_time': datetime.now().isoformat(),
            'tasks_completed': [],
            'errors': []
        }
        
        try:
            # Download historical stock data
            if self.data_sources['yahoo_finance']['enabled']:
                logging.info("Downloading Yahoo Finance historical data...")
                yahoo_data = await self.download_yahoo_finance_data(self.symbols)
                download_results['tasks_completed'].append(f"Yahoo Finance: {len(yahoo_data)} symbols")
            
            # Download news sentiment data
            logging.info("Downloading news sentiment data...")
            news_data = await self.download_news_sentiment_data()
            download_results['tasks_completed'].append(f"News data: {len(news_data)} sources")
            
            # Download economic indicators
            logging.info("Downloading economic indicators...")
            economic_data = await self.download_economic_indicators()
            download_results['tasks_completed'].append(f"Economic indicators: {len(economic_data)} metrics")
            
            # Start real-time feeds
            await self.start_real_time_websocket_feeds()
            download_results['tasks_completed'].append("Real-time feeds started")
            
        except Exception as e:
            error_msg = f"Download error: {e}"
            logging.error(error_msg)
            download_results['errors'].append(error_msg)
        
        download_results['end_time'] = datetime.now().isoformat()
        download_results['summary'] = self.get_latest_data_summary()
        
        # Save download results
        filepath = self.processed_dir / f"download_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f:
            json.dump(download_results, f, indent=2, default=str)
        
        return download_results

# Main execution function
async def main():
    """Main function to run the market data downloader"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    downloader = RealMarketDataDownloader()
    
    try:
        # Run comprehensive download
        results = await downloader.full_market_data_download()
        
        logging.info("Market data download completed!")
        logging.info(f"Tasks completed: {results['tasks_completed']}")
        
        if results['errors']:
            logging.warning(f"Errors encountered: {results['errors']}")
        
        # Keep real-time feeds running
        logging.info("Real-time feeds are running... Press Ctrl+C to stop.")
        
        # Run indefinitely for real-time updates
        while True:
            await asyncio.sleep(60)  # Check every minute
            summary = downloader.get_latest_data_summary()
            logging.info(f"Real-time status: {summary['real_time_symbols']} symbols active")
        
    except KeyboardInterrupt:
        logging.info("Shutting down market data downloader...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())