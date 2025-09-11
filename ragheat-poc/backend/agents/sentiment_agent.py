from crewai import Agent, Task
import requests
from typing import Dict, List
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)

class SentimentAgent:
    '''Agent responsible for analyzing market sentiment from news and social media'''

    def __init__(self, finnhub_api_key: str = None):
        self.finnhub_api_key = finnhub_api_key or "demo"
        self.agent = Agent(
            role='Sentiment Analyst',
            goal='Analyze market sentiment from news and social media',
            backstory='''You are a sentiment analysis expert who monitors 
            news feeds, social media, and market chatter to gauge investor 
            sentiment. You can identify trends, detect market mood shifts, 
            and predict sentiment-driven price movements.''',
            verbose=True,
            allow_delegation=False
        )

    def analyze_sentiment(self, symbol: str) -> Dict:
        '''Analyze sentiment for a stock'''
        news_sentiment = self._get_news_sentiment(symbol)

        return {
            'symbol': symbol,
            'news_sentiment': news_sentiment,
            'sentiment_score': self._calculate_sentiment_score(news_sentiment)
        }

    def _get_news_sentiment(self, symbol: str) -> Dict:
        '''Get news sentiment (mock implementation for demo)'''
        try:
            # For demo, use mock data
            mock_news = [
                f"{symbol} reports strong quarterly earnings",
                f"Analysts upgrade {symbol} to buy",
                f"{symbol} announces new product launch"
            ]

            sentiments = []
            for headline in mock_news:
                blob = TextBlob(headline)
                sentiments.append(blob.sentiment.polarity)

            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

            return {
                'headline_count': len(mock_news),
                'average_sentiment': avg_sentiment,
                'sentiment_label': self._get_sentiment_label(avg_sentiment)
            }

        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
            return {'error': str(e), 'average_sentiment': 0}

    def _get_sentiment_label(self, score: float) -> str:
        '''Convert sentiment score to label'''
        if score > 0.5:
            return "Very Positive"
        elif score > 0.1:
            return "Positive"
        elif score > -0.1:
            return "Neutral"
        elif score > -0.5:
            return "Negative"
        else:
            return "Very Negative"

    def _calculate_sentiment_score(self, sentiment_data: Dict) -> float:
        '''Calculate normalized sentiment score (0-1)'''
        avg_sentiment = sentiment_data.get('average_sentiment', 0)
        # Normalize from [-1, 1] to [0, 1]
        return (avg_sentiment + 1) / 2