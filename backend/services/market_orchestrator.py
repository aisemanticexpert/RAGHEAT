"""
Market Analysis Orchestrator
Coordinates all analysis services for comprehensive market insights with 90%+ win probability
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import json

from services.sector_performance_analyzer import SectorPerformanceAnalyzer, SectorPerformance
from services.options_analyzer import AdvancedOptionsAnalyzer, OptionsOpportunity  
from services.news_sentiment_analyzer import NewsSentimentAnalyzer, SentimentAnalysis
from services.dynamic_graph_builder import DynamicGraphBuilder

logger = logging.getLogger(__name__)

@dataclass
class MarketAnalysisResult:
    session_id: str
    timestamp: datetime
    top_sectors: List[SectorPerformance]
    worst_sectors: List[SectorPerformance]
    high_probability_opportunities: List[OptionsOpportunity]
    sector_opportunities: Dict[str, List[OptionsOpportunity]]
    sentiment_analysis: Dict[str, SentimentAnalysis]
    market_summary: Dict[str, float]
    winning_probability: float
    recommended_actions: List[str]
    graph_session_id: str

class MarketOrchestrator:
    
    def __init__(self):
        self.sector_analyzer = SectorPerformanceAnalyzer()
        self.options_analyzer = AdvancedOptionsAnalyzer()
        self.news_analyzer = NewsSentimentAnalyzer()
        self.graph_builder = DynamicGraphBuilder()
        
        self.analysis_cache = {}
        self.last_analysis_time = None
        self.cache_duration = 300  # 5 minutes cache
        
    async def run_comprehensive_analysis(self) -> MarketAnalysisResult:
        """Run complete market analysis with 90%+ probability filtering"""
        
        start_time = datetime.now()
        logger.info("Starting comprehensive market analysis...")
        
        try:
            # Step 1: Analyze sector performance
            logger.info("Step 1: Analyzing sector performance...")
            top_sectors, worst_sectors = await self.sector_analyzer.get_top_sectors(count=5)
            
            # Step 2: Get options opportunities for selected sectors
            logger.info("Step 2: Analyzing options opportunities...")
            sector_opportunities = {}
            high_prob_opportunities = []
            
            # Analyze top performing sectors
            for sector in top_sectors:
                opportunities = await self.options_analyzer.analyze_sector_options(
                    sector.top_stocks, sector.sector
                )
                sector_opportunities[sector.sector] = opportunities
                
                # Filter for ultra-high probability (90%+)
                ultra_high_prob = [opp for opp in opportunities if opp.win_probability >= 0.90]
                high_prob_opportunities.extend(ultra_high_prob)
            
            # Analyze worst performing sectors (for put opportunities)
            for sector in worst_sectors:
                opportunities = await self.options_analyzer.analyze_sector_options(
                    sector.top_stocks, sector.sector
                )
                sector_opportunities[sector.sector] = opportunities
                
                # Focus on put opportunities in bad sectors
                put_opportunities = [opp for opp in opportunities 
                                   if opp.strategy == "put" and opp.win_probability >= 0.85]
                high_prob_opportunities.extend(put_opportunities)
            
            # Step 3: Analyze news sentiment for all relevant stocks
            logger.info("Step 3: Analyzing news sentiment...")
            all_stocks = set()
            for opportunities in sector_opportunities.values():
                for opp in opportunities:
                    all_stocks.add(opp.symbol)
            
            sentiment_tasks = []
            async with self.news_analyzer as news_service:
                for stock in all_stocks:
                    sector = self._get_stock_sector(stock, top_sectors + worst_sectors)
                    task = asyncio.create_task(
                        news_service.analyze_stock_sentiment(stock, sector)
                    )
                    sentiment_tasks.append((stock, task))
                
                sentiment_results = await asyncio.gather(
                    *[task for _, task in sentiment_tasks], 
                    return_exceptions=True
                )
            
            # Process sentiment results
            sentiment_analysis = {}
            for (stock, _), result in zip(sentiment_tasks, sentiment_results):
                if isinstance(result, SentimentAnalysis):
                    sentiment_analysis[stock] = result
            
            # Step 4: Apply sentiment boost to opportunities
            logger.info("Step 4: Applying sentiment analysis...")
            enhanced_opportunities = []
            
            for opp in high_prob_opportunities:
                sentiment = sentiment_analysis.get(opp.symbol)
                if sentiment:
                    # Boost probability based on sentiment
                    sentiment_boost = self._calculate_sentiment_boost(sentiment, opp.strategy)
                    enhanced_prob = min(0.98, opp.win_probability + sentiment_boost)
                    
                    # Create enhanced opportunity
                    enhanced_opp = OptionsOpportunity(
                        symbol=opp.symbol,
                        sector=opp.sector,
                        win_probability=enhanced_prob,
                        strategy=opp.strategy,
                        target_price=opp.target_price,
                        expected_move=opp.expected_move,
                        volatility_rank=opp.volatility_rank,
                        volume_surge=opp.volume_surge,
                        earnings_date=opp.earnings_date,
                        news_catalyst=f"Sentiment: {sentiment.overall_sentiment:.2f}",
                        risk_reward_ratio=opp.risk_reward_ratio,
                        time_decay_factor=opp.time_decay_factor,
                        confidence_score=enhanced_prob,
                        entry_signals=opp.entry_signals + [f"Sentiment boost: +{sentiment_boost:.3f}"]
                    )
                    enhanced_opportunities.append(enhanced_opp)
                else:
                    enhanced_opportunities.append(opp)
            
            # Step 5: Build dynamic graph relationships
            logger.info("Step 5: Building market graph...")
            graph_session_id = await self.graph_builder.build_market_graph(
                top_sectors + worst_sectors,
                sector_opportunities,
                sentiment_analysis
            )
            
            # Step 6: Calculate overall market metrics
            market_summary = self._calculate_market_summary(
                top_sectors, worst_sectors, enhanced_opportunities, sentiment_analysis
            )
            
            # Step 7: Filter final recommendations (90%+ probability)
            final_opportunities = [
                opp for opp in enhanced_opportunities 
                if opp.win_probability >= 0.90
            ]
            
            # Sort by probability and confidence
            final_opportunities.sort(
                key=lambda x: (x.win_probability, x.confidence_score), 
                reverse=True
            )
            
            # Step 8: Generate recommended actions
            recommended_actions = self._generate_recommendations(
                final_opportunities, market_summary, top_sectors, worst_sectors
            )
            
            # Calculate overall winning probability
            overall_winning_prob = self._calculate_overall_winning_probability(final_opportunities)
            
            # Create result
            result = MarketAnalysisResult(
                session_id=f"analysis_{int(start_time.timestamp())}",
                timestamp=start_time,
                top_sectors=top_sectors,
                worst_sectors=worst_sectors,
                high_probability_opportunities=final_opportunities[:20],  # Top 20
                sector_opportunities=sector_opportunities,
                sentiment_analysis=sentiment_analysis,
                market_summary=market_summary,
                winning_probability=overall_winning_prob,
                recommended_actions=recommended_actions,
                graph_session_id=graph_session_id
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Analysis completed in {execution_time:.2f} seconds")
            logger.info(f"Found {len(final_opportunities)} opportunities with 90%+ probability")
            logger.info(f"Overall winning probability: {overall_winning_prob:.1%}")
            
            # Cache result
            self.analysis_cache[result.session_id] = result
            self.last_analysis_time = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            raise
    
    def _get_stock_sector(self, stock: str, sectors: List[SectorPerformance]) -> str:
        """Find sector for a stock"""
        for sector in sectors:
            if stock in sector.top_stocks:
                return sector.sector
        return "Unknown"
    
    def _calculate_sentiment_boost(self, sentiment: SentimentAnalysis, strategy: str) -> float:
        """Calculate probability boost based on sentiment and strategy"""
        
        base_boost = sentiment.sentiment_strength * sentiment.catalyst_potential * 0.1
        
        # Strategy-specific adjustments
        if strategy == "call" and sentiment.overall_sentiment > 0:
            return base_boost * 1.5  # Positive sentiment boosts calls
        elif strategy == "put" and sentiment.overall_sentiment < 0:
            return base_boost * 1.5  # Negative sentiment boosts puts
        elif strategy == "straddle":
            return base_boost * sentiment.catalyst_potential  # High catalyst potential boosts straddles
        
        return base_boost * 0.5  # Reduced boost for misaligned sentiment
    
    def _calculate_market_summary(self, top_sectors: List[SectorPerformance], 
                                worst_sectors: List[SectorPerformance],
                                opportunities: List[OptionsOpportunity],
                                sentiment_analysis: Dict[str, SentimentAnalysis]) -> Dict[str, float]:
        """Calculate overall market summary metrics"""
        
        # Sector metrics
        avg_top_performance = sum(s.performance_score for s in top_sectors) / len(top_sectors)
        avg_worst_performance = sum(s.performance_score for s in worst_sectors) / len(worst_sectors)
        
        # Opportunity metrics
        avg_win_probability = sum(opp.win_probability for opp in opportunities) / len(opportunities) if opportunities else 0
        avg_risk_reward = sum(opp.risk_reward_ratio for opp in opportunities) / len(opportunities) if opportunities else 0
        
        # Sentiment metrics
        sentiments = [s.overall_sentiment for s in sentiment_analysis.values()]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        avg_catalyst_potential = sum(s.catalyst_potential for s in sentiment_analysis.values()) / len(sentiment_analysis) if sentiment_analysis else 0
        
        # Strategy distribution
        call_count = len([opp for opp in opportunities if opp.strategy == "call"])
        put_count = len([opp for opp in opportunities if opp.strategy == "put"]) 
        straddle_count = len([opp for opp in opportunities if opp.strategy == "straddle"])
        
        return {
            "avg_top_sector_performance": avg_top_performance,
            "avg_worst_sector_performance": avg_worst_performance,
            "avg_win_probability": avg_win_probability,
            "avg_risk_reward_ratio": avg_risk_reward,
            "avg_market_sentiment": avg_sentiment,
            "avg_catalyst_potential": avg_catalyst_potential,
            "total_opportunities": len(opportunities),
            "call_opportunities": call_count,
            "put_opportunities": put_count,
            "straddle_opportunities": straddle_count,
            "market_breadth": len(top_sectors) + len(worst_sectors),
            "sentiment_coverage": len(sentiment_analysis)
        }
    
    def _generate_recommendations(self, opportunities: List[OptionsOpportunity], 
                                market_summary: Dict[str, float],
                                top_sectors: List[SectorPerformance],
                                worst_sectors: List[SectorPerformance]) -> List[str]:
        """Generate actionable trading recommendations"""
        
        recommendations = []
        
        if not opportunities:
            recommendations.append("⚠️ No high-probability opportunities found. Consider waiting for better setups.")
            return recommendations
        
        # Top opportunity
        best_opp = opportunities[0]
        recommendations.append(
            f"🎯 HIGHEST PROBABILITY: {best_opp.strategy.upper()} {best_opp.symbol} "
            f"({best_opp.win_probability:.1%} probability, {best_opp.confidence_score:.1%} confidence)"
        )
        
        # Sector recommendations
        if top_sectors:
            best_sector = top_sectors[0]
            recommendations.append(
                f"📈 TOP SECTOR: {best_sector.sector_name} "
                f"(Performance: {best_sector.performance_score:.2f})"
            )
        
        if worst_sectors and worst_sectors[0].performance_score < -2:
            worst_sector = worst_sectors[0]
            recommendations.append(
                f"📉 SHORT CANDIDATE: {worst_sector.sector_name} "
                f"(Performance: {worst_sector.performance_score:.2f})"
            )
        
        # Strategy mix recommendations
        call_ratio = market_summary["call_opportunities"] / market_summary["total_opportunities"]
        if call_ratio > 0.7:
            recommendations.append("🟢 BULLISH BIAS: High number of call opportunities suggests upward momentum")
        elif call_ratio < 0.3:
            recommendations.append("🔴 BEARISH BIAS: High number of put opportunities suggests downward pressure")
        else:
            recommendations.append("🟡 NEUTRAL BIAS: Mixed opportunities suggest range-bound market")
        
        # Risk management
        avg_risk_reward = market_summary["avg_risk_reward_ratio"]
        if avg_risk_reward > 3.0:
            recommendations.append("✅ EXCELLENT RISK/REWARD: Average R:R ratio > 3:1")
        elif avg_risk_reward < 2.0:
            recommendations.append("⚠️ MODERATE RISK/REWARD: Consider position sizing carefully")
        
        # Sentiment-based recommendations
        avg_sentiment = market_summary["avg_market_sentiment"]
        if avg_sentiment > 0.3:
            recommendations.append("📰 POSITIVE NEWS FLOW: Strong positive sentiment supporting positions")
        elif avg_sentiment < -0.3:
            recommendations.append("📰 NEGATIVE NEWS FLOW: Caution advised due to negative sentiment")
        
        # Final portfolio recommendation
        high_prob_count = len([opp for opp in opportunities if opp.win_probability > 0.92])
        if high_prob_count >= 5:
            recommendations.append(f"💎 DIVERSIFY: {high_prob_count} ultra-high probability plays available for portfolio diversification")
        
        return recommendations
    
    def _calculate_overall_winning_probability(self, opportunities: List[OptionsOpportunity]) -> float:
        """Calculate portfolio-level winning probability"""
        
        if not opportunities:
            return 0.0
        
        # For independent events, portfolio win probability is complex
        # Using simplified approach: average of top opportunities weighted by confidence
        
        top_opportunities = opportunities[:10]  # Top 10
        
        if not top_opportunities:
            return 0.0
        
        weighted_prob = sum(
            opp.win_probability * opp.confidence_score 
            for opp in top_opportunities
        ) / sum(opp.confidence_score for opp in top_opportunities)
        
        return weighted_prob
    
    async def get_cached_analysis(self, max_age_minutes: int = 5) -> Optional[MarketAnalysisResult]:
        """Get cached analysis if available and fresh"""
        
        if not self.last_analysis_time:
            return None
        
        age = (datetime.now() - self.last_analysis_time).total_seconds() / 60
        if age > max_age_minutes:
            return None
        
        # Return most recent analysis
        if self.analysis_cache:
            latest_key = max(self.analysis_cache.keys())
            return self.analysis_cache[latest_key]
        
        return None
    
    async def get_live_graph_analysis(self, graph_session_id: str) -> List[Dict]:
        """Get live analysis from Neo4j graph"""
        return await self.graph_builder.execute_live_query(graph_session_id)

# Global instance
market_orchestrator = MarketOrchestrator()