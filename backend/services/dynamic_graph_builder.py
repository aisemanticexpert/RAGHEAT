"""
Dynamic Neo4j Graph Builder
Creates and updates real-time relationships between sectors, stocks, and market events
"""

import asyncio
from neo4j import GraphDatabase
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import numpy as np

from services.sector_performance_analyzer import SectorPerformance
from services.options_analyzer import OptionsOpportunity
from services.news_sentiment_analyzer import SentimentAnalysis

logger = logging.getLogger(__name__)

class DynamicGraphBuilder:
    
    def __init__(self, uri="bolt://neo4j:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        if self.driver:
            self.driver.close()
    
    async def build_market_graph(self, 
                                sector_data: List[SectorPerformance],
                                options_data: Dict[str, List[OptionsOpportunity]],
                                sentiment_data: Dict[str, SentimentAnalysis]) -> str:
        """Build comprehensive market graph with dynamic relationships"""
        
        timestamp = datetime.now()
        session_id = f"session_{int(timestamp.timestamp())}"
        
        with self.driver.session() as session:
            # Create market session node
            session.run("""
                CREATE (ms:MarketSession {
                    session_id: $session_id,
                    timestamp: $timestamp,
                    market_status: $status
                })
            """, session_id=session_id, timestamp=timestamp.isoformat(), status="ACTIVE")
            
            # Create sector nodes and relationships
            await self._create_sector_nodes(session, sector_data, session_id)
            
            # Create stock nodes and relationships
            await self._create_stock_nodes(session, options_data, sentiment_data, session_id)
            
            # Create options opportunity relationships
            await self._create_options_relationships(session, options_data, session_id)
            
            # Create sentiment relationships
            await self._create_sentiment_relationships(session, sentiment_data, session_id)
            
            # Create inter-sector correlations
            await self._create_sector_correlations(session, sector_data, session_id)
            
            # Create risk-reward clusters
            await self._create_risk_clusters(session, options_data, session_id)
            
        logger.info(f"Built market graph for session {session_id}")
        return session_id
    
    async def _create_sector_nodes(self, session, sector_data: List[SectorPerformance], session_id: str):
        """Create sector nodes with performance metrics"""
        
        for sector in sector_data:
            session.run("""
                MERGE (s:Sector {name: $sector_name})
                SET s.performance_score = $performance,
                    s.avg_return = $avg_return,
                    s.volatility = $volatility,
                    s.momentum = $momentum,
                    s.volume_surge = $volume_surge,
                    s.news_sentiment = $news_sentiment,
                    s.last_updated = $timestamp,
                    s.session_id = $session_id
                
                WITH s
                MATCH (ms:MarketSession {session_id: $session_id})
                MERGE (ms)-[:CONTAINS_SECTOR {
                    strength: $performance,
                    rank: $rank
                }]->(s)
            """, 
            sector_name=sector.sector_name,
            performance=sector.performance_score,
            avg_return=sector.avg_return,
            volatility=sector.volatility,
            momentum=sector.momentum,
            volume_surge=sector.volume_surge,
            news_sentiment=sector.news_sentiment,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            rank=sector_data.index(sector) + 1
            )
    
    async def _create_stock_nodes(self, session, options_data: Dict[str, List[OptionsOpportunity]], 
                                sentiment_data: Dict[str, SentimentAnalysis], session_id: str):
        """Create stock nodes with comprehensive metrics"""
        
        all_symbols = set()
        for opportunities in options_data.values():
            for opp in opportunities:
                all_symbols.add(opp.symbol)
        
        for symbol in all_symbols:
            # Get data for this symbol
            stock_opportunities = []
            for sector_opps in options_data.values():
                stock_opportunities.extend([opp for opp in sector_opps if opp.symbol == symbol])
            
            sentiment = sentiment_data.get(symbol)
            best_opportunity = max(stock_opportunities, key=lambda x: x.win_probability) if stock_opportunities else None
            
            if not best_opportunity:
                continue
            
            session.run("""
                MERGE (st:Stock {symbol: $symbol})
                SET st.sector = $sector,
                    st.win_probability = $win_prob,
                    st.volatility_rank = $vol_rank,
                    st.volume_surge = $volume_surge,
                    st.sentiment_score = $sentiment,
                    st.catalyst_potential = $catalyst,
                    st.last_updated = $timestamp,
                    st.session_id = $session_id
                
                WITH st
                MATCH (s:Sector {name: $sector_full})
                MERGE (s)-[:CONTAINS_STOCK {
                    performance_rank: $rank,
                    opportunity_score: $opp_score
                }]->(st)
            """,
            symbol=symbol,
            sector=best_opportunity.sector,
            sector_full=self._get_sector_full_name(best_opportunity.sector),
            win_prob=best_opportunity.win_probability,
            vol_rank=best_opportunity.volatility_rank,
            volume_surge=best_opportunity.volume_surge,
            sentiment=sentiment.overall_sentiment if sentiment else 0.0,
            catalyst=sentiment.catalyst_potential if sentiment else 0.0,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            rank=1,  # Would be calculated based on sector ranking
            opp_score=best_opportunity.confidence_score
            )
    
    async def _create_options_relationships(self, session, options_data: Dict[str, List[OptionsOpportunity]], session_id: str):
        """Create options opportunity relationships"""
        
        for sector, opportunities in options_data.items():
            for opp in opportunities:
                session.run("""
                    MATCH (st:Stock {symbol: $symbol})
                    CREATE (op:OptionsOpportunity {
                        id: $opp_id,
                        strategy: $strategy,
                        win_probability: $win_prob,
                        expected_move: $expected_move,
                        risk_reward: $risk_reward,
                        confidence: $confidence,
                        session_id: $session_id,
                        timestamp: $timestamp
                    })
                    
                    CREATE (st)-[:HAS_OPPORTUNITY {
                        strength: $win_prob,
                        priority: $priority
                    }]->(op)
                """,
                symbol=opp.symbol,
                opp_id=f"{opp.symbol}_{opp.strategy}_{int(datetime.now().timestamp())}",
                strategy=opp.strategy,
                win_prob=opp.win_probability,
                expected_move=opp.expected_move,
                risk_reward=opp.risk_reward_ratio,
                confidence=opp.confidence_score,
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                priority=1 if opp.win_probability > 0.9 else 2
                )
    
    async def _create_sentiment_relationships(self, session, sentiment_data: Dict[str, SentimentAnalysis], session_id: str):
        """Create news sentiment relationships"""
        
        for symbol, sentiment in sentiment_data.items():
            if sentiment.news_count == 0:
                continue
                
            session.run("""
                MATCH (st:Stock {symbol: $symbol})
                CREATE (ns:NewsSentiment {
                    id: $news_id,
                    overall_sentiment: $sentiment,
                    sentiment_strength: $strength,
                    news_count: $count,
                    catalyst_potential: $catalyst,
                    key_topics: $topics,
                    session_id: $session_id,
                    timestamp: $timestamp
                })
                
                CREATE (st)-[:HAS_NEWS_SENTIMENT {
                    impact_score: $impact,
                    recency: $recency
                }]->(ns)
            """,
            symbol=symbol,
            news_id=f"{symbol}_news_{int(datetime.now().timestamp())}",
            sentiment=sentiment.overall_sentiment,
            strength=sentiment.sentiment_strength,
            count=sentiment.news_count,
            catalyst=sentiment.catalyst_potential,
            topics=json.dumps(sentiment.key_topics),
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            impact=sentiment.sentiment_strength * sentiment.catalyst_potential,
            recency=1.0  # Would calculate based on news timestamp
            )
    
    async def _create_sector_correlations(self, session, sector_data: List[SectorPerformance], session_id: str):
        """Create correlation relationships between sectors"""
        
        for i, sector1 in enumerate(sector_data):
            for j, sector2 in enumerate(sector_data[i+1:], i+1):
                # Calculate correlation based on performance similarity
                correlation = self._calculate_sector_correlation(sector1, sector2)
                
                if abs(correlation) > 0.5:  # Only create significant correlations
                    session.run("""
                        MATCH (s1:Sector {name: $sector1})
                        MATCH (s2:Sector {name: $sector2})
                        CREATE (s1)-[:CORRELATED_WITH {
                            correlation: $correlation,
                            strength: $strength,
                            session_id: $session_id
                        }]->(s2)
                    """,
                    sector1=sector1.sector_name,
                    sector2=sector2.sector_name,
                    correlation=correlation,
                    strength=abs(correlation),
                    session_id=session_id
                    )
    
    async def _create_risk_clusters(self, session, options_data: Dict[str, List[OptionsOpportunity]], session_id: str):
        """Create risk-reward clusters"""
        
        # Flatten all opportunities
        all_opportunities = []
        for opportunities in options_data.values():
            all_opportunities.extend(opportunities)
        
        # Create risk clusters based on volatility and probability
        high_prob_low_vol = [opp for opp in all_opportunities 
                           if opp.win_probability > 0.9 and opp.volatility_rank < 0.5]
        high_prob_high_vol = [opp for opp in all_opportunities 
                            if opp.win_probability > 0.9 and opp.volatility_rank >= 0.5]
        
        clusters = [
            ("HIGH_PROB_LOW_VOL", high_prob_low_vol),
            ("HIGH_PROB_HIGH_VOL", high_prob_high_vol)
        ]
        
        for cluster_name, opportunities in clusters:
            if not opportunities:
                continue
                
            session.run("""
                CREATE (rc:RiskCluster {
                    name: $cluster_name,
                    opportunity_count: $count,
                    avg_win_prob: $avg_prob,
                    avg_volatility: $avg_vol,
                    session_id: $session_id,
                    timestamp: $timestamp
                })
            """,
            cluster_name=cluster_name,
            count=len(opportunities),
            avg_prob=np.mean([opp.win_probability for opp in opportunities]),
            avg_vol=np.mean([opp.volatility_rank for opp in opportunities]),
            session_id=session_id,
            timestamp=datetime.now().isoformat()
            )
            
            # Link stocks to risk cluster
            for opp in opportunities:
                session.run("""
                    MATCH (st:Stock {symbol: $symbol})
                    MATCH (rc:RiskCluster {name: $cluster_name})
                    MERGE (st)-[:BELONGS_TO_CLUSTER {
                        score: $score
                    }]->(rc)
                """,
                symbol=opp.symbol,
                cluster_name=cluster_name,
                score=opp.confidence_score
                )
    
    def _calculate_sector_correlation(self, sector1: SectorPerformance, sector2: SectorPerformance) -> float:
        """Calculate correlation between two sectors"""
        # Simple correlation based on performance metrics
        metrics1 = [sector1.avg_return, sector1.momentum, sector1.volume_surge, sector1.news_sentiment]
        metrics2 = [sector2.avg_return, sector2.momentum, sector2.volume_surge, sector2.news_sentiment]
        
        return np.corrcoef(metrics1, metrics2)[0, 1] if len(metrics1) > 1 else 0.0
    
    def _get_sector_full_name(self, sector_code: str) -> str:
        """Convert sector code to full name"""
        sector_map = {
            "Technology": "Technology",
            "Healthcare": "Healthcare & Biotech", 
            "Financial": "Financial Services",
            "Consumer_Discretionary": "Consumer Discretionary",
            "Communication": "Communication Services",
            "Consumer_Staples": "Consumer Staples",
            "Industrial": "Industrial",
            "Energy": "Energy",
            "Utilities": "Utilities",
            "Real_Estate": "Real Estate"
        }
        return sector_map.get(sector_code, sector_code)
    
    async def get_live_analysis_query(self, session_id: str) -> str:
        """Generate Cypher query for live analysis"""
        
        query = f"""
        // Live Market Analysis for Session {session_id}
        MATCH (ms:MarketSession {{session_id: '{session_id}'}})-[:CONTAINS_SECTOR]->(s:Sector)
        MATCH (s)-[:CONTAINS_STOCK]->(st:Stock)
        MATCH (st)-[:HAS_OPPORTUNITY]->(op:OptionsOpportunity)
        OPTIONAL MATCH (st)-[:HAS_NEWS_SENTIMENT]->(ns:NewsSentiment)
        
        RETURN s.name as sector,
               st.symbol as stock,
               op.strategy as strategy,
               op.win_probability as win_probability,
               op.confidence as confidence,
               st.sentiment_score as sentiment,
               ns.catalyst_potential as catalyst_potential,
               s.performance_score as sector_performance
        
        ORDER BY op.win_probability DESC, op.confidence DESC
        LIMIT 20
        """
        
        return query
    
    async def execute_live_query(self, session_id: str) -> List[Dict[str, Any]]:
        """Execute live analysis query"""
        
        query = await self.get_live_analysis_query(session_id)
        
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

# Global instance
graph_builder = DynamicGraphBuilder()