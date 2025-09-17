// Advanced Real-time Market Analysis Cypher Queries
// These queries run at regular intervals for live market insights

// Query 1: Top 5 Sectors with Best Performance Today
MATCH (s:Sector)
WHERE s.last_updated > datetime() - duration('PT1H') // Last hour
RETURN s.name as sector,
       s.performance_score as performance,
       s.avg_return as return_1d,
       s.momentum as momentum,
       s.volume_surge as volume_activity
ORDER BY s.performance_score DESC
LIMIT 5;

// Query 2: Ultra High Probability Options Opportunities (90%+)
MATCH (st:Stock)-[:HAS_OPPORTUNITY]->(op:OptionsOpportunity)
WHERE op.win_probability >= 0.90
  AND op.timestamp > datetime() - duration('PT30M') // Last 30 minutes
WITH st, op, 
     CASE op.strategy
       WHEN 'call' THEN '📈 CALL'
       WHEN 'put' THEN '📉 PUT'  
       WHEN 'straddle' THEN '⚡ STRADDLE'
       ELSE op.strategy
     END as strategy_emoji
RETURN st.symbol as stock,
       st.sector as sector,
       strategy_emoji as strategy,
       round(op.win_probability * 100, 1) as win_probability_pct,
       round(op.confidence * 100, 1) as confidence_pct,
       round(op.expected_move, 2) as expected_move,
       round(op.risk_reward, 1) as risk_reward_ratio
ORDER BY op.win_probability DESC, op.confidence DESC
LIMIT 20;

// Query 3: Sector Correlation Network Analysis
MATCH (s1:Sector)-[r:CORRELATED_WITH]->(s2:Sector)
WHERE r.strength > 0.7  // Strong correlations only
  AND s1.session_id = s2.session_id
RETURN s1.name as sector_1,
       s2.name as sector_2,
       round(r.correlation, 3) as correlation,
       round(r.strength, 3) as strength,
       CASE 
         WHEN r.correlation > 0.8 THEN 'VERY_STRONG_POSITIVE'
         WHEN r.correlation > 0.5 THEN 'STRONG_POSITIVE'  
         WHEN r.correlation < -0.8 THEN 'VERY_STRONG_NEGATIVE'
         WHEN r.correlation < -0.5 THEN 'STRONG_NEGATIVE'
         ELSE 'MODERATE'
       END as correlation_type
ORDER BY r.strength DESC;

// Query 4: News Catalyst Impact Analysis
MATCH (st:Stock)-[:HAS_NEWS_SENTIMENT]->(ns:NewsSentiment)
WHERE ns.catalyst_potential > 0.7  // High catalyst potential
  AND ns.news_count >= 3           // Significant news volume
  AND ns.timestamp > datetime() - duration('PT2H') // Recent news
OPTIONAL MATCH (st)-[:HAS_OPPORTUNITY]->(op:OptionsOpportunity)
WHERE op.win_probability >= 0.85
RETURN st.symbol as stock,
       st.sector as sector,
       round(ns.overall_sentiment, 3) as sentiment,
       round(ns.catalyst_potential, 3) as catalyst_score,
       ns.news_count as news_volume,
       ns.key_topics as topics,
       collect(DISTINCT op.strategy) as available_strategies,
       max(op.win_probability) as best_win_probability
ORDER BY ns.catalyst_potential DESC, ns.overall_sentiment DESC;

// Query 5: Risk Cluster Analysis - Portfolio Construction
MATCH (st:Stock)-[:BELONGS_TO_CLUSTER]->(rc:RiskCluster)
WHERE rc.name IN ['HIGH_PROB_LOW_VOL', 'HIGH_PROB_HIGH_VOL']
OPTIONAL MATCH (st)-[:HAS_OPPORTUNITY]->(op:OptionsOpportunity)
WHERE op.win_probability >= 0.88
RETURN rc.name as risk_cluster,
       collect(DISTINCT st.symbol) as stocks_in_cluster,
       count(st) as stock_count,
       round(rc.avg_win_prob, 3) as cluster_avg_win_prob,
       round(rc.avg_volatility, 3) as cluster_avg_volatility,
       collect(DISTINCT op.strategy) as strategies_available
ORDER BY rc.avg_win_prob DESC;

// Query 6: Real-time Market Heat Map Data
MATCH (s:Sector)-[:CONTAINS_STOCK]->(st:Stock)
WHERE s.last_updated > datetime() - duration('PT15M') // Last 15 minutes
OPTIONAL MATCH (st)-[:HAS_OPPORTUNITY]->(op:OptionsOpportunity)
WHERE op.win_probability >= 0.85
OPTIONAL MATCH (st)-[:HAS_NEWS_SENTIMENT]->(ns:NewsSentiment)
RETURN s.name as sector,
       st.symbol as stock,
       s.performance_score as sector_performance,
       st.win_probability as stock_win_probability,
       st.volatility_rank as volatility,
       st.volume_surge as volume_activity,
       coalesce(ns.overall_sentiment, 0) as news_sentiment,
       coalesce(op.strategy, 'none') as best_strategy,
       coalesce(op.win_probability, 0) as strategy_win_prob,
       // Heat score calculation
       round((s.performance_score * 0.3 + 
              st.win_probability * 100 * 0.4 + 
              st.volume_surge * 10 * 0.2 +
              coalesce(ns.overall_sentiment, 0) * 50 * 0.1), 2) as heat_score
ORDER BY heat_score DESC;

// Query 7: Time-based Performance Tracking
MATCH (s:Sector)
WHERE s.last_updated > datetime() - duration('PT6H') // Last 6 hours
WITH s.name as sector,
     s.performance_score as current_score,
     s.last_updated as timestamp
ORDER BY s.name, timestamp
RETURN sector,
       collect(current_score) as performance_history,
       collect(timestamp) as time_points,
       max(current_score) - min(current_score) as performance_range,
       // Simple momentum calculation
       CASE 
         WHEN size(collect(current_score)) >= 2 
         THEN last(collect(current_score)) - head(collect(current_score))
         ELSE 0 
       END as momentum_change
ORDER BY momentum_change DESC;

// Query 8: Cross-Sector Opportunity Discovery
MATCH (s1:Sector)-[:CONTAINS_STOCK]->(st1:Stock)-[:HAS_OPPORTUNITY]->(op1:OptionsOpportunity)
MATCH (s2:Sector)-[:CONTAINS_STOCK]->(st2:Stock)-[:HAS_OPPORTUNITY]->(op2:OptionsOpportunity)
WHERE s1 <> s2  // Different sectors
  AND op1.win_probability >= 0.90
  AND op2.win_probability >= 0.90
  AND op1.strategy <> op2.strategy  // Different strategies
  AND s1.session_id = s2.session_id
RETURN s1.name as sector_1,
       st1.symbol as stock_1,
       op1.strategy as strategy_1,
       round(op1.win_probability, 3) as win_prob_1,
       s2.name as sector_2,
       st2.symbol as stock_2,
       op2.strategy as strategy_2,
       round(op2.win_probability, 3) as win_prob_2,
       round((op1.win_probability + op2.win_probability) / 2, 3) as combined_prob,
       // Portfolio risk assessment
       CASE 
         WHEN s1.performance_score > 0 AND s2.performance_score > 0 THEN 'BOTH_BULLISH'
         WHEN s1.performance_score < 0 AND s2.performance_score < 0 THEN 'BOTH_BEARISH'
         ELSE 'MIXED_SIGNALS'
       END as portfolio_bias
ORDER BY combined_prob DESC
LIMIT 15;

// Query 9: Volatility Expansion Detection
MATCH (st:Stock)-[:HAS_OPPORTUNITY]->(op:OptionsOpportunity)
WHERE op.strategy = 'straddle'
  AND op.win_probability >= 0.85
OPTIONAL MATCH (st)-[:HAS_NEWS_SENTIMENT]->(ns:NewsSentiment)
RETURN st.symbol as stock,
       st.sector as sector,
       round(op.win_probability, 3) as straddle_win_prob,
       round(op.expected_move, 2) as expected_move_pct,
       st.volatility_rank as current_vol_rank,
       coalesce(ns.catalyst_potential, 0) as news_catalyst,
       // Volatility expansion score
       round((op.win_probability * 0.4 + 
              (1 - st.volatility_rank) * 0.3 +  // Low current vol is good
              coalesce(ns.catalyst_potential, 0) * 0.3), 3) as vol_expansion_score
ORDER BY vol_expansion_score DESC
LIMIT 10;

// Query 10: Market Session Performance Summary
MATCH (ms:MarketSession)-[:CONTAINS_SECTOR]->(s:Sector)
OPTIONAL MATCH (s)-[:CONTAINS_STOCK]->(st:Stock)
OPTIONAL MATCH (st)-[:HAS_OPPORTUNITY]->(op:OptionsOpportunity)
WHERE op.win_probability >= 0.90
RETURN ms.session_id as session_id,
       ms.timestamp as analysis_time,
       count(DISTINCT s) as sectors_analyzed,
       count(DISTINCT st) as stocks_analyzed,
       count(DISTINCT op) as high_prob_opportunities,
       round(avg(s.performance_score), 3) as avg_sector_performance,
       round(avg(op.win_probability), 3) as avg_opportunity_win_rate,
       // Market sentiment summary  
       round(avg(st.sentiment_score), 3) as avg_market_sentiment,
       // Strategy distribution
       size([o IN collect(op) WHERE o.strategy = 'call']) as call_opportunities,
       size([o IN collect(op) WHERE o.strategy = 'put']) as put_opportunities,
       size([o IN collect(op) WHERE o.strategy = 'straddle']) as straddle_opportunities;

// Scheduled Query: Clean old sessions (run every hour)
MATCH (ms:MarketSession)
WHERE ms.timestamp < datetime() - duration('P1D') // Older than 1 day
DETACH DELETE ms;