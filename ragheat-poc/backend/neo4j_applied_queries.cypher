// RAGHeat Applied Neo4j Visualization Queries
// Copy these into Neo4j Browser

// MAIN HEAT VISUALIZATION
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN s
ORDER BY s.heat_score DESC

==================================================

// COMPLETE GRAPH
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
OPTIONAL MATCH (s)-[r]-(connected)
RETURN s, r, connected
LIMIT 100

==================================================

// LIVE UPDATES
MATCH (s:Stock)
WHERE datetime(s.last_updated) > datetime() - duration('PT5M')
RETURN s
ORDER BY s.heat_score DESC

==================================================

// DEBUG QUERY
MATCH (s:Stock)
RETURN count(s) as total_stocks, 
       max(s.heat_score) as max_heat,
       min(s.heat_score) as min_heat

==================================================

// HEAT TABLE
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN 
  s.symbol as Symbol,
  s.heat_score as Heat_Score, 
  s.heat_level as Heat_Level,
  s.current_price as Price,
  s.last_updated as Last_Updated
ORDER BY s.heat_score DESC

==================================================

