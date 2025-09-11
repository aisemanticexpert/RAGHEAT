from crewai import Agent, Task, Crew, Process
from typing import Dict, List, Optional
import pandas as pd
import logging
from .fundamental_agent import FundamentalAgent
from .sentiment_agent import SentimentAgent
from .valuation_agent import ValuationAgent

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    '''Master agent that coordinates other agents and makes final decisions'''

    def __init__(self):
        self.fundamental_agent = FundamentalAgent()
        self.sentiment_agent = SentimentAgent()
        self.valuation_agent = ValuationAgent()

        self.orchestrator = Agent(
            role='Portfolio Manager',
            goal='Synthesize all analyses and make investment recommendations',
            backstory='''You are a senior portfolio manager with 20 years of 
            experience. You synthesize fundamental, sentiment, and valuation 
            analyses to make informed investment decisions. You understand 
            how to balance different factors and identify the best opportunities 
            while managing risk.''',
            verbose=True,
            allow_delegation=True
        )

    def analyze_stock_comprehensive(self, symbol: str) -> Dict:
        '''Perform comprehensive analysis using all agents'''
        try:
            # Gather analyses from all agents
            fundamental = self.fundamental_agent.analyze_stock(symbol)
            sentiment = self.sentiment_agent.analyze_sentiment(symbol)
            valuation = self.valuation_agent.analyze_valuation(symbol)

            # Calculate heat score based on all factors
            heat_score = self._calculate_heat_score(fundamental, sentiment, valuation)

            # Create recommendation
            recommendation = self._generate_recommendation(
                fundamental, sentiment, valuation, heat_score
            )

            return {
                'symbol': symbol,
                'fundamental_analysis': fundamental,
                'sentiment_analysis': sentiment,
                'valuation_analysis': valuation,
                'heat_score': heat_score,
                'recommendation': recommendation,
                'timestamp': pd.Timestamp.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}

    def _calculate_heat_score(
        self, 
        fundamental: Dict, 
        sentiment: Dict, 
        valuation: Dict
    ) -> float:
        '''Calculate heat score based on multiple factors'''
        weights = {
            'fundamental': 0.4,
            'sentiment': 0.3,
            'valuation': 0.3
        }

        scores = {
            'fundamental': fundamental.get('fundamental_score', 0.5),
            'sentiment': sentiment.get('sentiment_score', 0.5),
            'valuation': valuation.get('valuation_score', 0.5)
        }

        # Weighted average
        heat_score = sum(
            weights[key] * scores[key] 
            for key in weights
        )

        # Add momentum boost if all signals are positive
        if all(score > 0.6 for score in scores.values()):
            heat_score *= 1.1

        return min(1.0, heat_score)

    def _generate_recommendation(
        self,
        fundamental: Dict,
        sentiment: Dict,
        valuation: Dict,
        heat_score: float
    ) -> Dict:
        '''Generate investment recommendation'''

        # Determine action based on heat score
        if heat_score > 0.7:
            action = "STRONG BUY"
            confidence = "High"
        elif heat_score > 0.6:
            action = "BUY"
            confidence = "Medium"
        elif heat_score > 0.4:
            action = "HOLD"
            confidence = "Medium"
        elif heat_score > 0.3:
            action = "SELL"
            confidence = "Medium"
        else:
            action = "STRONG SELL"
            confidence = "High"

        explanation = self._generate_explanation(
            fundamental, sentiment, valuation, heat_score
        )

        return {
            'action': action,
            'confidence': confidence,
            'heat_score': heat_score,
            'explanation': explanation
        }

    def _generate_explanation(
        self,
        fundamental: Dict,
        sentiment: Dict,
        valuation: Dict,
        heat_score: float
    ) -> str:
        '''Generate human-readable explanation'''

        explanations = []

        # Fundamental explanation
        if fundamental.get('fundamental_score', 0) > 0.6:
            explanations.append("Strong fundamentals with healthy financial metrics")
        elif fundamental.get('fundamental_score', 0) < 0.4:
            explanations.append("Weak fundamentals raise concerns")

        # Sentiment explanation
        sentiment_label = sentiment.get('news_sentiment', {}).get('sentiment_label', 'Neutral')
        explanations.append(f"Market sentiment is {sentiment_label}")

        # Valuation explanation
        if valuation.get('rsi', 50) < 30:
            explanations.append("Technical indicators suggest oversold conditions")
        elif valuation.get('rsi', 50) > 70:
            explanations.append("Technical indicators suggest overbought conditions")

        # Heat score explanation
        explanations.append(f"Overall heat score of {heat_score:.2f} indicates {self._heat_interpretation(heat_score)}")

        return ". ".join(explanations)

    def _heat_interpretation(self, heat_score: float) -> str:
        '''Interpret heat score in human terms'''
        if heat_score > 0.7:
            return "very high momentum and strong opportunity"
        elif heat_score > 0.5:
            return "positive momentum"
        elif heat_score > 0.3:
            return "neutral to slightly negative conditions"
        else:
            return "significant headwinds"