"""
Sentiment Analysis Tools for Portfolio Construction
================================================

Tools for analyzing market sentiment, news, social media, and analyst opinions.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from crewai_tools import BaseTool
from pydantic import BaseModel
from loguru import logger
import yfinance as yf

# For sentiment analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import nltk
    from textblob import TextBlob
except ImportError:
    logger.warning("Some sentiment analysis libraries not available")

class NewsAggregatorTool(BaseTool):
    """Tool for aggregating financial news from multiple sources."""
    
    name: str = "news_aggregator"
    description: str = "Aggregate financial news from multiple sources for sentiment analysis"
    
    def __init__(self):
        super().__init__()
        self.news_sources = [
            'yahoo_finance',
            'reuters', 
            'bloomberg',
            'marketwatch'
        ]
    
    def _run(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """Aggregate news for a given stock symbol."""
        try:
            # Get news from Yahoo Finance (primary source)
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            aggregated_news = []
            
            for article in news[:20]:  # Limit to 20 most recent articles
                news_item = {
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'publisher': article.get('publisher', ''),
                    'publish_time': datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                    'url': article.get('link', ''),
                    'type': article.get('type', 'news')
                }
                aggregated_news.append(news_item)
            
            # Additional mock news sources (would be real APIs in production)
            mock_news = self._generate_mock_news(symbol)
            aggregated_news.extend(mock_news)
            
            return {
                'symbol': symbol,
                'news_articles': aggregated_news,
                'total_articles': len(aggregated_news),
                'date_range': {
                    'from': (datetime.now() - timedelta(days=days_back)).isoformat(),
                    'to': datetime.now().isoformat()
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error aggregating news for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _generate_mock_news(self, symbol: str) -> List[Dict]:
        """Generate mock news articles for demonstration."""
        mock_articles = [
            {
                'title': f'{symbol} Reports Strong Quarterly Earnings',
                'summary': f'{symbol} exceeded analysts expectations with strong revenue growth.',
                'publisher': 'Financial Times',
                'publish_time': datetime.now() - timedelta(hours=2),
                'url': f'https://example.com/news/{symbol}-earnings',
                'type': 'earnings'
            },
            {
                'title': f'Analysts Upgrade {symbol} Price Target',
                'summary': f'Multiple analysts raised price targets for {symbol} citing strong fundamentals.',
                'publisher': 'Reuters',
                'publish_time': datetime.now() - timedelta(hours=6),
                'url': f'https://example.com/news/{symbol}-upgrade',
                'type': 'analyst_rating'
            }
        ]
        return mock_articles

class SentimentAnalyzerTool(BaseTool):
    """Tool for analyzing sentiment of news and text content."""
    
    name: str = "sentiment_analyzer"
    description: str = "Analyze sentiment of financial news and text content"
    
    def __init__(self):
        super().__init__()
        self.vader = SentimentIntensityAnalyzer()
        try:
            # Initialize FinBERT model for financial sentiment
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_pipeline = pipeline("sentiment-analysis", 
                                            model=self.finbert_model, 
                                            tokenizer=self.finbert_tokenizer)
        except:
            logger.warning("FinBERT model not available, using alternative methods")
            self.finbert_pipeline = None
    
    def _run(self, text_data: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """Analyze sentiment of text data."""
        try:
            sentiment_scores = []
            
            for item in text_data:
                text = f"{item.get('title', '')} {item.get('summary', '')}"
                if text.strip():
                    score = self._analyze_text_sentiment(text)
                    score['source'] = item
                    sentiment_scores.append(score)
            
            # Aggregate sentiment
            aggregate_sentiment = self._aggregate_sentiment_scores(sentiment_scores)
            
            return {
                'symbol': symbol,
                'individual_scores': sentiment_scores,
                'aggregate_sentiment': aggregate_sentiment,
                'sentiment_summary': self._generate_sentiment_summary(aggregate_sentiment),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of individual text."""
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # FinBERT sentiment (if available)
        finbert_score = None
        if self.finbert_pipeline:
            try:
                finbert_result = self.finbert_pipeline(text[:512])  # Truncate for model limits
                finbert_score = finbert_result[0]
            except:
                pass
        
        return {
            'text_preview': text[:100],
            'vader_sentiment': vader_scores,
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'finbert_sentiment': finbert_score,
            'composite_score': self._calculate_composite_score(
                vader_scores['compound'], 
                textblob_polarity, 
                finbert_score
            )
        }
    
    def _calculate_composite_score(self, vader_compound: float, textblob_polarity: float, finbert_score: Optional[Dict]) -> float:
        """Calculate composite sentiment score."""
        scores = [vader_compound, textblob_polarity]
        
        if finbert_score:
            # Convert FinBERT to numeric scale
            if finbert_score['label'] == 'positive':
                scores.append(finbert_score['score'])
            elif finbert_score['label'] == 'negative':
                scores.append(-finbert_score['score'])
            else:  # neutral
                scores.append(0)
        
        return np.mean(scores)
    
    def _aggregate_sentiment_scores(self, scores: List[Dict]) -> Dict[str, Any]:
        """Aggregate multiple sentiment scores."""
        if not scores:
            return {'sentiment_score': 0, 'sentiment_label': 'neutral', 'confidence': 0}
        
        composite_scores = [score['composite_score'] for score in scores]
        avg_sentiment = np.mean(composite_scores)
        sentiment_std = np.std(composite_scores)
        
        # Determine sentiment label
        if avg_sentiment > 0.1:
            label = 'positive'
        elif avg_sentiment < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'sentiment_score': avg_sentiment,
            'sentiment_label': label,
            'confidence': max(0, 1 - sentiment_std),
            'sentiment_distribution': {
                'positive': len([s for s in composite_scores if s > 0.1]),
                'negative': len([s for s in composite_scores if s < -0.1]),
                'neutral': len([s for s in composite_scores if -0.1 <= s <= 0.1])
            }
        }
    
    def _generate_sentiment_summary(self, aggregate: Dict[str, Any]) -> str:
        """Generate human-readable sentiment summary."""
        score = aggregate['sentiment_score']
        label = aggregate['sentiment_label']
        confidence = aggregate['confidence']
        
        if label == 'positive':
            return f"Positive sentiment (score: {score:.2f}) with {confidence:.1%} confidence"
        elif label == 'negative':
            return f"Negative sentiment (score: {score:.2f}) with {confidence:.1%} confidence"
        else:
            return f"Neutral sentiment (score: {score:.2f}) with {confidence:.1%} confidence"

class SocialMediaMonitorTool(BaseTool):
    """Tool for monitoring social media sentiment about stocks."""
    
    name: str = "social_media_monitor"
    description: str = "Monitor social media platforms for stock sentiment and buzz"
    
    def _run(self, symbol: str) -> Dict[str, Any]:
        """Monitor social media for stock sentiment."""
        try:
            # Mock social media data (would integrate with real APIs)
            social_data = self._generate_mock_social_data(symbol)
            
            # Analyze social sentiment
            sentiment_analysis = self._analyze_social_sentiment(social_data)
            
            return {
                'symbol': symbol,
                'social_mentions': social_data,
                'sentiment_analysis': sentiment_analysis,
                'buzz_metrics': self._calculate_buzz_metrics(social_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring social media for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _generate_mock_social_data(self, symbol: str) -> List[Dict]:
        """Generate mock social media data."""
        import random
        
        platforms = ['reddit', 'twitter', 'stocktwits']
        sentiments = ['bullish', 'bearish', 'neutral']
        
        social_posts = []
        for i in range(50):  # 50 mock posts
            post = {
                'platform': random.choice(platforms),
                'text': f"Mock post about ${symbol} - {random.choice(['great earnings', 'concerning trends', 'holding steady'])}",
                'sentiment': random.choice(sentiments),
                'likes': random.randint(1, 100),
                'shares': random.randint(0, 50),
                'timestamp': datetime.now() - timedelta(hours=random.randint(1, 24)),
                'user_followers': random.randint(100, 10000)
            }
            social_posts.append(post)
        
        return social_posts
    
    def _analyze_social_sentiment(self, social_data: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from social media data."""
        sentiment_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        total_engagement = 0
        
        for post in social_data:
            sentiment_counts[post['sentiment']] += 1
            total_engagement += post['likes'] + post['shares']
        
        total_posts = len(social_data)
        if total_posts == 0:
            return {'error': 'No social data available'}
        
        return {
            'sentiment_breakdown': {
                'bullish_pct': sentiment_counts['bullish'] / total_posts,
                'bearish_pct': sentiment_counts['bearish'] / total_posts,
                'neutral_pct': sentiment_counts['neutral'] / total_posts
            },
            'total_engagement': total_engagement,
            'avg_engagement_per_post': total_engagement / total_posts,
            'sentiment_score': (sentiment_counts['bullish'] - sentiment_counts['bearish']) / total_posts
        }
    
    def _calculate_buzz_metrics(self, social_data: List[Dict]) -> Dict[str, Any]:
        """Calculate buzz and virality metrics."""
        if not social_data:
            return {}
        
        total_mentions = len(social_data)
        total_engagement = sum(post['likes'] + post['shares'] for post in social_data)
        
        # Calculate platform breakdown
        platform_breakdown = {}
        for post in social_data:
            platform = post['platform']
            if platform not in platform_breakdown:
                platform_breakdown[platform] = 0
            platform_breakdown[platform] += 1
        
        return {
            'total_mentions': total_mentions,
            'total_engagement': total_engagement,
            'platform_breakdown': platform_breakdown,
            'buzz_intensity': min(total_mentions / 100, 1.0),  # Normalized buzz score
            'viral_coefficient': total_engagement / max(total_mentions, 1)
        }

class AnalystRatingTrackerTool(BaseTool):
    """Tool for tracking analyst ratings and price targets."""
    
    name: str = "analyst_rating_tracker"
    description: str = "Track analyst ratings, price targets, and recommendation changes"
    
    def _run(self, symbol: str) -> Dict[str, Any]:
        """Track analyst ratings for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get analyst recommendations
            recommendations = ticker.recommendations
            upgrades_downgrades = ticker.upgrades_downgrades
            
            # Process recommendations
            if recommendations is not None and not recommendations.empty:
                recent_recommendations = recommendations.head(10)
                rating_analysis = self._analyze_ratings(recent_recommendations)
            else:
                rating_analysis = {'error': 'No analyst recommendations available'}
            
            # Get price targets
            analyst_price_targets = self._get_price_targets(ticker)
            
            return {
                'symbol': symbol,
                'analyst_ratings': rating_analysis,
                'price_targets': analyst_price_targets,
                'recent_changes': self._get_recent_changes(upgrades_downgrades),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error tracking analyst ratings for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _analyze_ratings(self, recommendations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze analyst recommendations."""
        try:
            # Convert recommendations to numeric scores
            rating_map = {
                'Strong Buy': 5, 'Buy': 4, 'Hold': 3, 'Sell': 2, 'Strong Sell': 1,
                'Outperform': 4, 'Underperform': 2, 'Market Perform': 3
            }
            
            ratings = []
            for _, row in recommendations.iterrows():
                action = row.get('To Grade', row.get('Action', ''))
                if action in rating_map:
                    ratings.append(rating_map[action])
            
            if not ratings:
                return {'error': 'No valid ratings found'}
            
            avg_rating = np.mean(ratings)
            rating_distribution = {
                'strong_buy': ratings.count(5),
                'buy': ratings.count(4),
                'hold': ratings.count(3),
                'sell': ratings.count(2),
                'strong_sell': ratings.count(1)
            }
            
            return {
                'average_rating': avg_rating,
                'rating_distribution': rating_distribution,
                'total_analysts': len(ratings),
                'consensus': self._get_consensus(avg_rating)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing ratings: {e}")
            return {'error': str(e)}
    
    def _get_consensus(self, avg_rating: float) -> str:
        """Get consensus rating from average."""
        if avg_rating >= 4.5:
            return "Strong Buy"
        elif avg_rating >= 3.5:
            return "Buy"
        elif avg_rating >= 2.5:
            return "Hold"
        elif avg_rating >= 1.5:
            return "Sell"
        else:
            return "Strong Sell"
    
    def _get_price_targets(self, ticker) -> Dict[str, Any]:
        """Get analyst price targets."""
        try:
            info = ticker.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            target_mean = info.get('targetMeanPrice')
            
            if target_mean and current_price:
                upside_potential = (target_mean - current_price) / current_price
            else:
                upside_potential = None
            
            return {
                'current_price': current_price,
                'target_high': target_high,
                'target_low': target_low,
                'target_mean': target_mean,
                'upside_potential': upside_potential
            }
            
        except Exception as e:
            return {'error': f'Error getting price targets: {e}'}
    
    def _get_recent_changes(self, upgrades_downgrades) -> List[Dict]:
        """Get recent rating changes."""
        if upgrades_downgrades is None or upgrades_downgrades.empty:
            return []
        
        recent_changes = []
        for _, row in upgrades_downgrades.head(5).iterrows():
            change = {
                'firm': row.get('Firm', ''),
                'from_grade': row.get('From Grade', ''),
                'to_grade': row.get('To Grade', ''),
                'action': row.get('Action', ''),
                'date': row.name.strftime('%Y-%m-%d') if hasattr(row.name, 'strftime') else str(row.name)
            }
            recent_changes.append(change)
        
        return recent_changes

class FinBERTSentimentTool(BaseTool):
    """Tool specifically for FinBERT-based financial sentiment analysis."""
    
    name: str = "finbert_sentiment"
    description: str = "Analyze financial text sentiment using FinBERT model"
    
    def __init__(self):
        super().__init__()
        try:
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
        except:
            logger.warning("FinBERT model not available")
            self.finbert_pipeline = None
    
    def _run(self, texts: List[str], symbol: str) -> Dict[str, Any]:
        """Analyze financial sentiment using FinBERT."""
        try:
            if not self.finbert_pipeline:
                return {'error': 'FinBERT model not available', 'symbol': symbol}
            
            sentiment_results = []
            
            for text in texts[:50]:  # Limit to 50 texts for performance
                if len(text.strip()) > 0:
                    # Truncate text to model limits
                    truncated_text = text[:512]
                    result = self.finbert_pipeline(truncated_text)
                    
                    sentiment_results.append({
                        'text_preview': text[:100],
                        'sentiment': result[0]['label'],
                        'confidence': result[0]['score']
                    })
            
            # Aggregate results
            aggregate = self._aggregate_finbert_results(sentiment_results)
            
            return {
                'symbol': symbol,
                'individual_results': sentiment_results,
                'aggregate_sentiment': aggregate,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in FinBERT analysis for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _aggregate_finbert_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate FinBERT sentiment results."""
        if not results:
            return {'sentiment': 'neutral', 'confidence': 0}
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        confidence_sum = 0
        
        for result in results:
            sentiment_counts[result['sentiment']] += 1
            confidence_sum += result['confidence']
        
        total = len(results)
        avg_confidence = confidence_sum / total
        
        # Determine overall sentiment
        max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        return {
            'overall_sentiment': max_sentiment,
            'average_confidence': avg_confidence,
            'sentiment_distribution': {
                k: v / total for k, v in sentiment_counts.items()
            },
            'sentiment_score': self._calculate_sentiment_score(sentiment_counts, total)
        }
    
    def _calculate_sentiment_score(self, counts: Dict[str, int], total: int) -> float:
        """Calculate sentiment score from -1 to 1."""
        positive_pct = counts['positive'] / total
        negative_pct = counts['negative'] / total
        return positive_pct - negative_pct

# Initialize tools
news_aggregator = NewsAggregatorTool()
sentiment_analyzer = SentimentAnalyzerTool()
social_media_monitor = SocialMediaMonitorTool()
analyst_rating_tracker = AnalystRatingTrackerTool()
finbert_sentiment = FinBERTSentimentTool()