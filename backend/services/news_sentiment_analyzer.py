"""
News Sentiment Analysis Service
Integrates with Alpha Vantage and Yahoo Finance for real-time news sentiment
"""

import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
import re
from dataclasses import dataclass
from textblob import TextBlob
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    published_time: datetime
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0 to 1
    topics: List[str]
    url: str

@dataclass
class SentimentAnalysis:
    symbol: str
    sector: str
    overall_sentiment: float  # -1 to 1
    sentiment_strength: float  # 0 to 1
    news_count: int
    key_topics: List[str]
    major_headlines: List[str]
    catalyst_potential: float  # 0 to 1
    timestamp: datetime

class NewsSentimentAnalyzer:
    
    def __init__(self):
        # In production, use environment variables
        self.alpha_vantage_key = "demo"  # Replace with actual API key
        self.session = None
        
        # Sentiment keywords
        self.bullish_keywords = [
            'beat', 'beats', 'exceed', 'exceeds', 'strong', 'growth', 'positive',
            'upgrade', 'buy', 'bullish', 'surge', 'rally', 'breakout', 'momentum',
            'innovation', 'expansion', 'acquisition', 'partnership', 'revenue',
            'profit', 'earnings', 'dividend', 'breakthrough'
        ]
        
        self.bearish_keywords = [
            'miss', 'misses', 'weak', 'decline', 'negative', 'downgrade', 'sell',
            'bearish', 'crash', 'drop', 'fall', 'concern', 'risk', 'loss',
            'layoffs', 'lawsuit', 'investigation', 'scandal', 'warning',
            'recession', 'inflation', 'volatility'
        ]
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def analyze_stock_sentiment(self, symbol: str, sector: str) -> SentimentAnalysis:
        """Analyze sentiment for a specific stock"""
        try:
            # Get news from multiple sources
            news_items = []
            
            # Yahoo Finance news
            yahoo_news = await self._get_yahoo_news(symbol)
            news_items.extend(yahoo_news)
            
            # Alpha Vantage news (if API key available)
            if self.alpha_vantage_key != "demo":
                av_news = await self._get_alpha_vantage_news(symbol)
                news_items.extend(av_news)
            
            # Analyze sentiment
            return self._analyze_sentiment(symbol, sector, news_items)
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return self._create_neutral_sentiment(symbol, sector)
    
    async def analyze_sector_sentiment(self, sector: str, stocks: List[str]) -> Dict[str, SentimentAnalysis]:
        """Analyze sentiment for multiple stocks in a sector"""
        tasks = []
        for symbol in stocks:
            task = asyncio.create_task(self.analyze_stock_sentiment(symbol, sector))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        sentiment_data = {}
        for i, result in enumerate(results):
            if isinstance(result, SentimentAnalysis):
                sentiment_data[stocks[i]] = result
        
        return sentiment_data
    
    async def _get_yahoo_news(self, symbol: str) -> List[NewsItem]:
        """Get news from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            news_items = []
            for item in news[:10]:  # Top 10 news items
                try:
                    published_time = datetime.fromtimestamp(item['providerPublishTime'])
                    
                    # Calculate relevance based on title/summary keywords
                    text = f"{item['title']} {item.get('summary', '')}"
                    relevance = self._calculate_relevance(text, symbol)
                    
                    news_items.append(NewsItem(
                        title=item['title'],
                        summary=item.get('summary', ''),
                        source=item.get('publisher', 'Yahoo'),
                        published_time=published_time,
                        sentiment_score=0,  # Will be calculated later
                        relevance_score=relevance,
                        topics=[],
                        url=item.get('link', '')
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing news item: {e}")
                    continue
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo news for {symbol}: {e}")
            return []
    
    async def _get_alpha_vantage_news(self, symbol: str) -> List[NewsItem]:
        """Get news from Alpha Vantage News API"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.alpha_vantage_key,
                'limit': 50
            }
            
            if not self.session:
                return []
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                
                if 'feed' not in data:
                    return []
                
                news_items = []
                for item in data['feed']:
                    try:
                        published_time = datetime.fromisoformat(
                            item['time_published'].replace('T', ' ')
                        )
                        
                        # Get sentiment from Alpha Vantage
                        sentiment_score = float(item.get('overall_sentiment_score', 0))
                        
                        # Calculate relevance
                        relevance = float(item.get('relevance_score', 0.5))
                        
                        news_items.append(NewsItem(
                            title=item['title'],
                            summary=item['summary'],
                            source=item['source'],
                            published_time=published_time,
                            sentiment_score=sentiment_score,
                            relevance_score=relevance,
                            topics=item.get('topics', []),
                            url=item['url']
                        ))
                        
                    except Exception as e:
                        logger.warning(f"Error parsing Alpha Vantage news item: {e}")
                        continue
                
                return news_items
                
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news for {symbol}: {e}")
            return []
    
    def _calculate_relevance(self, text: str, symbol: str) -> float:
        """Calculate news relevance to stock"""
        text_lower = text.lower()
        symbol_lower = symbol.lower()
        
        relevance = 0.0
        
        # Direct symbol mention
        if symbol_lower in text_lower:
            relevance += 0.5
        
        # Company name recognition (simplified)
        company_keywords = self._get_company_keywords(symbol)
        for keyword in company_keywords:
            if keyword.lower() in text_lower:
                relevance += 0.3
                break
        
        # Financial keywords
        financial_keywords = ['earnings', 'revenue', 'profit', 'guidance', 'forecast']
        for keyword in financial_keywords:
            if keyword in text_lower:
                relevance += 0.2
                break
        
        return min(1.0, relevance)
    
    def _get_company_keywords(self, symbol: str) -> List[str]:
        """Get company-specific keywords for relevance scoring"""
        # Simplified mapping - in production, use a comprehensive database
        company_map = {
            'AAPL': ['apple', 'iphone', 'ipad', 'mac'],
            'MSFT': ['microsoft', 'windows', 'azure', 'office'],
            'GOOGL': ['google', 'alphabet', 'youtube', 'android'],
            'TSLA': ['tesla', 'elon musk', 'electric vehicle', 'ev'],
            'AMZN': ['amazon', 'aws', 'bezos', 'prime'],
            'META': ['meta', 'facebook', 'instagram', 'whatsapp']
        }
        return company_map.get(symbol, [])
    
    def _analyze_sentiment(self, symbol: str, sector: str, news_items: List[NewsItem]) -> SentimentAnalysis:
        """Analyze overall sentiment from news items"""
        if not news_items:
            return self._create_neutral_sentiment(symbol, sector)
        
        # Calculate sentiment for each news item if not already calculated
        for item in news_items:
            if item.sentiment_score == 0:  # Not calculated by API
                item.sentiment_score = self._calculate_text_sentiment(
                    f"{item.title} {item.summary}"
                )
        
        # Weighted sentiment calculation
        total_weighted_sentiment = 0
        total_weight = 0
        
        # Recent news gets higher weight
        now = datetime.now()
        
        for item in news_items:
            # Time decay weight (recent news more important)
            hours_old = (now - item.published_time).total_seconds() / 3600
            time_weight = max(0.1, 1.0 - (hours_old / 168))  # Decay over week
            
            # Relevance weight
            relevance_weight = item.relevance_score
            
            # Combined weight
            weight = time_weight * relevance_weight
            
            total_weighted_sentiment += item.sentiment_score * weight
            total_weight += weight
        
        # Calculate overall metrics
        overall_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
        sentiment_strength = abs(overall_sentiment)
        
        # Extract key topics and headlines
        key_topics = self._extract_key_topics(news_items)
        major_headlines = [item.title for item in news_items[:5] if item.relevance_score > 0.5]
        
        # Calculate catalyst potential
        catalyst_potential = self._calculate_catalyst_potential(news_items, overall_sentiment)
        
        return SentimentAnalysis(
            symbol=symbol,
            sector=sector,
            overall_sentiment=overall_sentiment,
            sentiment_strength=sentiment_strength,
            news_count=len(news_items),
            key_topics=key_topics,
            major_headlines=major_headlines,
            catalyst_potential=catalyst_potential,
            timestamp=datetime.now()
        )
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score from text using TextBlob and keyword analysis"""
        try:
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment.polarity
            
            # Keyword-based sentiment
            text_lower = text.lower()
            bullish_score = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
            bearish_score = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
            
            keyword_sentiment = 0
            if bullish_score + bearish_score > 0:
                keyword_sentiment = (bullish_score - bearish_score) / (bullish_score + bearish_score)
            
            # Combine scores
            combined_sentiment = (textblob_sentiment * 0.6) + (keyword_sentiment * 0.4)
            
            return max(-1.0, min(1.0, combined_sentiment))
            
        except Exception as e:
            logger.error(f"Error calculating sentiment: {e}")
            return 0.0
    
    def _extract_key_topics(self, news_items: List[NewsItem]) -> List[str]:
        """Extract key topics from news items"""
        # Simplified topic extraction
        topics = set()
        
        for item in news_items:
            if item.topics:
                topics.update(item.topics)
        
        # Add common financial topics found in text
        all_text = ' '.join([f"{item.title} {item.summary}" for item in news_items]).lower()
        
        financial_topics = [
            'earnings', 'revenue', 'guidance', 'acquisition', 'merger',
            'product launch', 'regulatory', 'competition', 'market share'
        ]
        
        for topic in financial_topics:
            if topic in all_text:
                topics.add(topic.title())
        
        return list(topics)[:10]  # Top 10 topics
    
    def _calculate_catalyst_potential(self, news_items: List[NewsItem], overall_sentiment: float) -> float:
        """Calculate potential for news to be a market catalyst"""
        if not news_items:
            return 0.0
        
        catalyst_score = 0.0
        
        # High relevance news
        high_relevance_count = len([item for item in news_items if item.relevance_score > 0.7])
        catalyst_score += min(0.3, high_relevance_count * 0.1)
        
        # Strong sentiment
        catalyst_score += min(0.3, abs(overall_sentiment) * 0.3)
        
        # Recent news
        recent_count = len([
            item for item in news_items 
            if (datetime.now() - item.published_time).total_seconds() < 86400  # 24 hours
        ])
        catalyst_score += min(0.2, recent_count * 0.05)
        
        # Volume of news
        news_volume_score = min(0.2, len(news_items) * 0.02)
        catalyst_score += news_volume_score
        
        return min(1.0, catalyst_score)
    
    def _create_neutral_sentiment(self, symbol: str, sector: str) -> SentimentAnalysis:
        """Create neutral sentiment analysis when no news available"""
        return SentimentAnalysis(
            symbol=symbol,
            sector=sector,
            overall_sentiment=0.0,
            sentiment_strength=0.0,
            news_count=0,
            key_topics=[],
            major_headlines=[],
            catalyst_potential=0.0,
            timestamp=datetime.now()
        )

# Global instance
news_analyzer = NewsSentimentAnalyzer()