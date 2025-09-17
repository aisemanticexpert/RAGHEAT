"""
Sentiment Analyst Agent for RAGHeat CrewAI System
=================================================

This agent specializes in analyzing market sentiment through news, social media,
analyst ratings, and other sentiment indicators to gauge short-term momentum.
"""

from typing import Dict, Any, List
from .base_agent import RAGHeatBaseAgent
import logging

logger = logging.getLogger(__name__)

class SentimentAnalystAgent(RAGHeatBaseAgent):
    """
    Sentiment Analyst Agent for market sentiment and news analysis.
    
    Specializes in:
    - Financial news sentiment analysis
    - Social media sentiment monitoring
    - Analyst rating changes and consensus
    - Market sentiment indicators
    - Short-term momentum assessment
    """
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis on given stocks.
        
        Args:
            input_data: Dictionary containing:
                - stocks: List of stock tickers to analyze
                - time_window: Time window for sentiment analysis (default: 30d)
                - sentiment_sources: Sources to analyze (news, social, analyst)
                - baseline_sentiment: Historical sentiment baseline
        
        Returns:
            Sentiment analysis results with momentum indicators
        """
        try:
            stocks = input_data.get("stocks", [])
            time_window = input_data.get("time_window", "30d")
            sentiment_sources = input_data.get("sentiment_sources", ["news", "social_media", "analyst_ratings"])
            
            if not stocks:
                return {"error": "No stocks provided for sentiment analysis"}
            
            logger.info(f"Sentiment analysis starting for {len(stocks)} stocks")
            
            # Prepare analysis context
            analysis_context = {
                "stocks": stocks,
                "time_window": time_window,
                "sentiment_sources": sentiment_sources,
                "analysis_type": "sentiment",
                "focus_areas": [
                    "news_sentiment",
                    "social_media_buzz",
                    "analyst_consensus",
                    "insider_activity",
                    "market_momentum"
                ]
            }
            
            # Execute sentiment analysis task
            task_description = f"""
            Conduct comprehensive sentiment analysis for the following stocks: {', '.join(stocks)}
            
            Time Window: {time_window}
            Sources: {', '.join(sentiment_sources)}
            
            For each stock, analyze:
            
            1. NEWS SENTIMENT ANALYSIS:
               - Recent news articles and press releases (last {time_window})
               - Sentiment scoring using FinBERT and other NLP models
               - Key themes and topics in news coverage
               - Impact of major news events on sentiment trends
               - Source credibility and reach analysis
            
            2. SOCIAL MEDIA SENTIMENT:
               - Twitter/X mentions and sentiment
               - Reddit discussions (WallStreetBets, investing subreddits)
               - StockTwits sentiment and message volume
               - Trending hashtags and viral content
               - Retail investor sentiment indicators
            
            3. ANALYST SENTIMENT AND RATINGS:
               - Recent analyst rating changes (upgrades/downgrades)
               - Price target revisions and consensus
               - Analyst sentiment trends and distribution
               - Institutional vs retail analyst views
               - Earnings revision trends
            
            4. INSIDER ACTIVITY SENTIMENT:
               - Recent insider buying/selling activity
               - Form 4 filings and their sentiment implications
               - Management confidence indicators
               - Institutional ownership changes
            
            5. MARKET MOMENTUM INDICATORS:
               - Options flow sentiment (put/call ratios)
               - Short interest changes
               - ETF flow sentiment
               - Technical sentiment indicators
               - Fear & Greed index correlations
            
            For each stock, provide:
            - Overall Sentiment Score (-1 to +1, where -1 = very negative, +1 = very positive)
            - News Sentiment Score (-1 to +1)
            - Social Media Sentiment Score (-1 to +1)
            - Analyst Sentiment Score (-1 to +1)
            - Momentum Indicator (Strong Positive, Positive, Neutral, Negative, Strong Negative)
            - Sentiment Trend (Improving, Stable, Deteriorating)
            - Key Sentiment Drivers (top 3 factors influencing sentiment)
            - Sentiment-based recommendation (Strong Buy, Buy, Hold, Sell, Strong Sell)
            - Confidence level (0-100%)
            
            Identify:
            - Sentiment inflection points and trend changes
            - Contrarian opportunities (oversold/overbought sentiment)
            - Event-driven sentiment catalysts
            - Cross-asset sentiment correlations
            
            Focus on short to medium-term sentiment dynamics and their potential impact on stock performance.
            """
            
            result = self.execute_task(task_description, analysis_context)
            
            # Post-process results to ensure structured output
            processed_result = self._structure_sentiment_analysis(result, stocks)
            
            logger.info(f"Sentiment analysis completed for {len(stocks)} stocks")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                "error": str(e),
                "agent": "sentiment_analyst",
                "analysis_type": "sentiment"
            }
    
    def _structure_sentiment_analysis(self, raw_result: Dict[str, Any], stocks: List[str]) -> Dict[str, Any]:
        """Structure the sentiment analysis results."""
        
        structured_result = {
            "analysis_type": "sentiment",
            "agent": "sentiment_analyst",
            "timestamp": self._get_current_timestamp(),
            "stocks_analyzed": stocks,
            "overall_analysis": raw_result.get("result", ""),
            "sentiment_scores": {},
            "sentiment_trends": {},
            "key_sentiment_drivers": [],
            "market_sentiment_overview": "",
            "contrarian_opportunities": [],
            "sentiment_alerts": []
        }
        
        # Extract structured data from result text
        result_text = str(raw_result.get("result", ""))
        
        # Extract key sentiment drivers
        drivers = self._extract_sentiment_drivers(result_text)
        structured_result["key_sentiment_drivers"] = drivers
        
        # Extract contrarian opportunities
        contrarian = self._extract_contrarian_opportunities(result_text)
        structured_result["contrarian_opportunities"] = contrarian
        
        # For each stock, extract specific sentiment data
        for stock in stocks:
            sentiment_data = self._extract_stock_sentiment_data(result_text, stock)
            if sentiment_data:
                structured_result["sentiment_scores"][stock] = sentiment_data
        
        return structured_result
    
    def _extract_sentiment_drivers(self, text: str) -> List[str]:
        """Extract key sentiment drivers from analysis text."""
        drivers = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['driver:', 'catalyst:', 'factor:', 'theme:']):
                clean_driver = line.split(':', 1)[-1].strip()
                if clean_driver and len(clean_driver) > 15:
                    drivers.append(clean_driver)
        
        return drivers[:5]  # Limit to top 5 drivers
    
    def _extract_contrarian_opportunities(self, text: str) -> List[str]:
        """Extract contrarian opportunities from analysis text."""
        opportunities = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['contrarian', 'oversold', 'overbought', 'reversal']):
                if len(line) > 20:
                    opportunities.append(line)
        
        return opportunities[:3]  # Limit to top 3 opportunities
    
    def _extract_stock_sentiment_data(self, text: str, stock: str) -> Dict[str, Any]:
        """Extract stock-specific sentiment data from analysis text."""
        sentiment_data = {
            "ticker": stock,
            "overall_sentiment": 0.0,  # Default neutral
            "news_sentiment": 0.0,
            "social_sentiment": 0.0,
            "analyst_sentiment": 0.0,
            "momentum": "Neutral",
            "trend": "Stable",
            "confidence": 60,
            "recommendation": "HOLD",
            "key_themes": [],
            "sentiment_alerts": []
        }
        
        # Look for stock-specific sentiment indicators
        text_lower = text.lower()
        stock_lower = stock.lower()
        
        # Extract sentiment indicators
        if f"{stock_lower} positive" in text_lower or f"bullish on {stock_lower}" in text_lower:
            sentiment_data["overall_sentiment"] = 0.6
            sentiment_data["recommendation"] = "BUY"
        elif f"{stock_lower} negative" in text_lower or f"bearish on {stock_lower}" in text_lower:
            sentiment_data["overall_sentiment"] = -0.6
            sentiment_data["recommendation"] = "SELL"
        
        return sentiment_data
    
    def analyze_news_sentiment(self, stocks: List[str], time_window: str = "7d") -> Dict[str, Any]:
        """
        Specialized method for news sentiment analysis.
        
        Args:
            stocks: List of stock tickers
            time_window: Time window for news analysis
            
        Returns:
            Detailed news sentiment analysis
        """
        task_description = f"""
        Perform detailed news sentiment analysis for: {', '.join(stocks)}
        
        Time Window: {time_window}
        
        Analyze:
        1. NEWS VOLUME AND COVERAGE:
           - Number of articles and mentions
           - Source diversity and credibility
           - Coverage intensity trends
        
        2. SENTIMENT SCORING:
           - Article-level sentiment scores
           - Headline sentiment vs body content
           - Sentiment distribution and volatility
        
        3. TOPIC ANALYSIS:
           - Key themes and topics discussed
           - Earnings-related vs operational news
           - Regulatory or legal news impact
        
        4. TEMPORAL PATTERNS:
           - Sentiment trend over time window
           - Event-driven sentiment spikes
           - Pre/post market sentiment patterns
        
        5. SOURCE ANALYSIS:
           - Mainstream media vs financial press
           - Analyst research coverage
           - Company communications sentiment
        
        Provide sentiment momentum indicators and news-driven catalysts.
        """
        
        context = {
            "stocks": stocks,
            "time_window": time_window,
            "analysis_type": "news_sentiment"
        }
        
        return self.execute_task(task_description, context)
    
    def analyze_social_media_sentiment(self, stocks: List[str], platforms: List[str] = None) -> Dict[str, Any]:
        """
        Specialized method for social media sentiment analysis.
        
        Args:
            stocks: List of stock tickers
            platforms: Social media platforms to analyze
            
        Returns:
            Social media sentiment analysis
        """
        platforms = platforms or ["twitter", "reddit", "stocktwits"]
        
        task_description = f"""
        Analyze social media sentiment for: {', '.join(stocks)}
        
        Platforms: {', '.join(platforms)}
        
        For each platform, evaluate:
        
        1. VOLUME METRICS:
           - Mention volume and frequency
           - User engagement (likes, shares, comments)
           - Viral content and trending status
        
        2. SENTIMENT METRICS:
           - Positive/negative sentiment ratios
           - Emotion analysis (fear, greed, excitement)
           - Sentiment intensity and conviction
        
        3. INFLUENTIAL VOICES:
           - Key opinion leaders and their sentiment
           - Verified account sentiment vs general public
           - Institutional vs retail sentiment divergence
        
        4. BEHAVIORAL INDICATORS:
           - Discussion quality and depth
           - FOMO vs fear indicators
           - Bandwagon effects and herd behavior
        
        5. PLATFORM-SPECIFIC ANALYSIS:
           - Twitter: Real-time sentiment and trending
           - Reddit: Community sentiment and discussions
           - StockTwits: Trader sentiment and positioning
        
        Identify social sentiment-driven momentum and potential reversals.
        """
        
        context = {
            "stocks": stocks,
            "platforms": platforms,
            "analysis_type": "social_media_sentiment"
        }
        
        return self.execute_task(task_description, context)
    
    def analyze_analyst_sentiment(self, stocks: List[str]) -> Dict[str, Any]:
        """
        Specialized method for analyst sentiment and rating analysis.
        
        Args:
            stocks: List of stock tickers
            
        Returns:
            Analyst sentiment analysis
        """
        task_description = f"""
        Analyze analyst sentiment and ratings for: {', '.join(stocks)}
        
        Evaluate:
        
        1. RATING TRENDS:
           - Recent rating changes (last 90 days)
           - Upgrade/downgrade frequency and magnitude
           - Rating distribution (Buy/Hold/Sell)
        
        2. PRICE TARGET ANALYSIS:
           - Price target revisions and trends
           - Consensus price target vs current price
           - Price target dispersion and conviction
        
        3. EARNINGS ESTIMATES:
           - Earnings revision trends (up/down)
           - Estimate dispersion and uncertainty
           - Beat/miss expectations analysis
        
        4. ANALYST SENTIMENT INDICATORS:
           - Language tone in research reports
           - Conviction level in recommendations
           - Initiations vs downgrades sentiment
        
        5. CONSENSUS DYNAMICS:
           - Analyst consensus strength
           - Outlier opinions and contrarian views
           - Institutional vs independent analyst sentiment
        
        Identify momentum shifts in professional sentiment.
        """
        
        context = {
            "stocks": stocks,
            "analysis_type": "analyst_sentiment"
        }
        
        return self.execute_task(task_description, context)