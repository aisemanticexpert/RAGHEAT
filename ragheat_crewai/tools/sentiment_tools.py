"""
Sentiment Analysis Tools for RAGHeat CrewAI System
=================================================

Tools for sentiment analysis including news, social media, and analyst sentiment.
"""

from typing import Dict, Any, List, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NewsAggregatorInput(BaseModel):
    """Input schema for news aggregator tool."""
    tickers: List[str] = Field(..., description="Stock ticker symbols")
    days_back: int = Field(default=30, description="Days to look back for news")
    sources: List[str] = Field(default=["general"], description="News sources to include")

class NewsAggregator(BaseTool):
    """Tool for aggregating financial news for sentiment analysis."""
    
    name: str = Field(default="news_aggregator")
    description: str = Field(default="Aggregate financial news for sentiment analysis")
    args_schema: type[BaseModel] = NewsAggregatorInput
    
    def _run(self, tickers: List[str], days_back: int = 30, sources: List[str] = None) -> Dict[str, Any]:
        """
        Aggregate news for given tickers.
        
        Args:
            tickers: Stock ticker symbols
            days_back: Days to look back for news
            sources: News sources to include
            
        Returns:
            Dictionary containing aggregated news data
        """
        try:
            sources = sources or ["general"]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Simulate news aggregation (in real implementation, would use news APIs)
            aggregated_news = {}
            
            for ticker in tickers:
                # Simulate fetching news for each ticker
                ticker_news = self._fetch_ticker_news(ticker, start_date, end_date, sources)
                aggregated_news[ticker] = ticker_news
            
            result = {
                "tickers": tickers,
                "news_data": aggregated_news,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "sources": sources,
                "total_articles": sum(len(news["articles"]) for news in aggregated_news.values()),
                "summary": self._generate_news_summary(aggregated_news),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully aggregated news for {len(tickers)} tickers")
            return result
            
        except Exception as e:
            logger.error(f"Error aggregating news: {e}")
            return {
                "tickers": tickers,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _fetch_ticker_news(self, ticker: str, start_date: datetime, end_date: datetime, sources: List[str]) -> Dict[str, Any]:
        """Fetch news for a specific ticker."""
        # Simulate news fetching
        # In real implementation, would integrate with news APIs like NewsAPI, Alpha Vantage, etc.
        
        sample_articles = [
            {
                "title": f"{ticker} Reports Strong Q3 Earnings",
                "summary": f"{ticker} exceeded analyst expectations with strong revenue growth.",
                "sentiment_score": 0.7,
                "source": "Financial Times",
                "published_date": (end_date - timedelta(days=5)).isoformat(),
                "url": f"https://example.com/{ticker}-earnings",
                "relevance_score": 0.9
            },
            {
                "title": f"Analyst Upgrades {ticker} Price Target",
                "summary": f"Major investment bank raises {ticker} target price citing strong fundamentals.",
                "sentiment_score": 0.6,
                "source": "Reuters",
                "published_date": (end_date - timedelta(days=10)).isoformat(),
                "url": f"https://example.com/{ticker}-upgrade",
                "relevance_score": 0.8
            },
            {
                "title": f"{ticker} Faces Regulatory Scrutiny",
                "summary": f"Regulators are investigating {ticker}'s business practices.",
                "sentiment_score": -0.4,
                "source": "Wall Street Journal",
                "published_date": (end_date - timedelta(days=15)).isoformat(),
                "url": f"https://example.com/{ticker}-regulatory",
                "relevance_score": 0.7
            }
        ]
        
        return {
            "ticker": ticker,
            "articles": sample_articles,
            "article_count": len(sample_articles),
            "average_sentiment": sum(article["sentiment_score"] for article in sample_articles) / len(sample_articles),
            "sentiment_distribution": self._calculate_sentiment_distribution(sample_articles)
        }
    
    def _calculate_sentiment_distribution(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate sentiment distribution of articles."""
        distribution = {"positive": 0, "neutral": 0, "negative": 0}
        
        for article in articles:
            sentiment = article["sentiment_score"]
            if sentiment > 0.1:
                distribution["positive"] += 1
            elif sentiment < -0.1:
                distribution["negative"] += 1
            else:
                distribution["neutral"] += 1
        
        return distribution
    
    def _generate_news_summary(self, aggregated_news: Dict[str, Any]) -> str:
        """Generate summary of aggregated news."""
        total_articles = sum(len(news["articles"]) for news in aggregated_news.values())
        
        if total_articles == 0:
            return "No relevant news articles found for the specified tickers."
        
        # Calculate overall sentiment
        all_sentiments = []
        for news in aggregated_news.values():
            for article in news["articles"]:
                all_sentiments.append(article["sentiment_score"])
        
        avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
        
        sentiment_desc = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
        
        return f"Found {total_articles} articles with overall {sentiment_desc} sentiment (score: {avg_sentiment:.2f})"

class SentimentAnalyzerInput(BaseModel):
    """Input schema for sentiment analyzer tool."""
    text_data: List[str] = Field(..., description="Text data to analyze for sentiment")
    analysis_type: str = Field(default="financial", description="Type of sentiment analysis")

class SentimentAnalyzer(BaseTool):
    """Tool for analyzing sentiment of text data."""
    
    name: str = Field(default="sentiment_analyzer")
    description: str = Field(default="Analyze sentiment of text data using NLP models")
    args_schema: type[BaseModel] = SentimentAnalyzerInput
    
    def _run(self, text_data: List[str], analysis_type: str = "financial") -> Dict[str, Any]:
        """
        Analyze sentiment of text data.
        
        Args:
            text_data: List of text strings to analyze
            analysis_type: Type of sentiment analysis
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            if not text_data:
                return {
                    "error": "No text data provided for analysis",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Simulate sentiment analysis (in real implementation, would use models like FinBERT, VADER, etc.)
            sentiment_results = []
            
            for i, text in enumerate(text_data):
                sentiment_score = self._analyze_text_sentiment(text, analysis_type)
                sentiment_results.append({
                    "text_id": i,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment_score": sentiment_score,
                    "sentiment_label": self._score_to_label(sentiment_score),
                    "confidence": 0.85  # Simulated confidence
                })
            
            # Calculate aggregate metrics
            scores = [result["sentiment_score"] for result in sentiment_results]
            avg_sentiment = sum(scores) / len(scores)
            sentiment_volatility = self._calculate_sentiment_volatility(scores)
            
            result = {
                "analysis_type": analysis_type,
                "text_count": len(text_data),
                "individual_results": sentiment_results,
                "aggregate_metrics": {
                    "average_sentiment": avg_sentiment,
                    "sentiment_volatility": sentiment_volatility,
                    "sentiment_label": self._score_to_label(avg_sentiment),
                    "positive_count": sum(1 for s in scores if s > 0.1),
                    "negative_count": sum(1 for s in scores if s < -0.1),
                    "neutral_count": sum(1 for s in scores if -0.1 <= s <= 0.1)
                },
                "insights": self._generate_sentiment_insights(sentiment_results, avg_sentiment),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully analyzed sentiment for {len(text_data)} texts")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_text_sentiment(self, text: str, analysis_type: str) -> float:
        """Analyze sentiment of individual text."""
        # Simplified sentiment analysis simulation
        # In real implementation, would use proper NLP models
        
        positive_words = ["good", "great", "excellent", "positive", "bullish", "strong", "growth", "profit", "beat", "upgrade"]
        negative_words = ["bad", "terrible", "negative", "bearish", "weak", "decline", "loss", "miss", "downgrade", "risk"]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Simple scoring based on word counts
        if positive_count + negative_count == 0:
            return 0.0  # Neutral
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        
        # Add some randomness to simulate model uncertainty
        import random
        sentiment += random.uniform(-0.1, 0.1)
        
        return max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
    
    def _score_to_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_sentiment_volatility(self, scores: List[float]) -> float:
        """Calculate sentiment volatility."""
        if len(scores) < 2:
            return 0.0
        
        avg = sum(scores) / len(scores)
        variance = sum((score - avg) ** 2 for score in scores) / len(scores)
        return variance ** 0.5
    
    def _generate_sentiment_insights(self, results: List[Dict[str, Any]], avg_sentiment: float) -> List[str]:
        """Generate insights from sentiment analysis."""
        insights = []
        
        # Overall sentiment insight
        if avg_sentiment > 0.3:
            insights.append("Strong positive sentiment detected across texts")
        elif avg_sentiment > 0.1:
            insights.append("Moderate positive sentiment observed")
        elif avg_sentiment < -0.3:
            insights.append("Strong negative sentiment detected")
        elif avg_sentiment < -0.1:
            insights.append("Moderate negative sentiment observed")
        else:
            insights.append("Neutral sentiment with mixed opinions")
        
        # Sentiment distribution insight
        positive_count = sum(1 for r in results if r["sentiment_score"] > 0.1)
        negative_count = sum(1 for r in results if r["sentiment_score"] < -0.1)
        
        if positive_count > negative_count * 2:
            insights.append("Sentiment is predominantly positive")
        elif negative_count > positive_count * 2:
            insights.append("Sentiment is predominantly negative")
        else:
            insights.append("Mixed sentiment with both positive and negative views")
        
        return insights

class SocialMediaMonitor(BaseTool):
    """Tool for monitoring social media sentiment."""
    
    name: str = Field(default="social_media_monitor")
    description: str = Field(default="Monitor social media sentiment for stocks and markets")
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        return {"tool": "social_media_monitor", "status": "placeholder"}

class AnalystRatingTracker(BaseTool):
    """Tool for tracking analyst ratings and recommendations."""
    
    name: str = Field(default="analyst_rating_tracker")
    description: str = Field(default="Track analyst ratings and price target changes")
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        return {"tool": "analyst_rating_tracker", "status": "placeholder"}

class FinBERTSentiment(BaseTool):
    """Tool for financial sentiment analysis using FinBERT."""
    
    name: str = Field(default="finbert_sentiment")
    description: str = Field(default="Advanced financial sentiment analysis using FinBERT model")
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        return {"tool": "finbert_sentiment", "status": "placeholder"}

# Register tools
from .tool_registry import register_tool

register_tool("news_aggregator", NewsAggregator)
register_tool("sentiment_analyzer", SentimentAnalyzer)
register_tool("social_media_monitor", SocialMediaMonitor)
register_tool("analyst_rating_tracker", AnalystRatingTracker)
register_tool("finbert_sentiment", FinBERTSentiment)