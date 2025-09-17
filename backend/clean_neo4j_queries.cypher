// Clean Neo4j Queries for IDE - No Separators
// These queries work in both Neo4j Browser and IDE GraphDB plugin

// Main Heat Visualization Query
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN s
ORDER BY s.heat_score DESC;

// Complete Graph with Relationships
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
OPTIONAL MATCH (s)-[r]-(connected)
RETURN s, r, connected
LIMIT 100;

// Live Updates Query
MATCH (s:Stock)
WHERE datetime(s.last_updated) > datetime() - duration('PT5M')
RETURN s
ORDER BY s.heat_score DESC;

// Debug Query - Check Stock Count
MATCH (s:Stock)
RETURN count(s) as total_stocks;

// Heat Table View
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN 
  s.symbol as Symbol,
  s.heat_score as Heat_Score, 
  s.heat_level as Heat_Level,
  s.current_price as Price,
  s.last_updated as Last_Updated
ORDER BY s.heat_score DESC;