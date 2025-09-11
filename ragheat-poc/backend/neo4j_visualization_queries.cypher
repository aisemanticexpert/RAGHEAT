// RAGHeat Neo4j Visualization Queries
// Copy and paste these into Neo4j Browser

// 1. Heat Map Overview
// Shows top 20 heated stocks with heat-based coloring
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN s
ORDER BY s.heat_score DESC
LIMIT 20

============================================================

// 2. Full Graph with Heat
// Shows interconnected stocks with relationships and heat visualization
MATCH (s:Stock)-[r]-(connected)
WHERE s.heat_score IS NOT NULL
RETURN s, r, connected
LIMIT 100

============================================================

// 3. Extreme Heat Nodes
// Focus on extremely heated stocks and their connections
MATCH (s:Stock)
WHERE s.heat_level = 'extreme' OR s.heat_score > 0.7
OPTIONAL MATCH (s)-[r]-(connected)
RETURN s, r, connected

============================================================

// 4. Live Heat Updates
// Shows recently updated stocks with current heat levels
MATCH (s:Stock)
WHERE datetime(s.last_updated) > datetime() - duration('PT5M')
OPTIONAL MATCH (s)-[r:CORRELATES]-(other)
RETURN s, r, other
ORDER BY s.heat_score DESC

============================================================

// 5. Stock and Sector Analysis
// Shows stocks with their sectors and heat levels
MATCH (s:Stock)-[:SECTOR_OF]->(sector:Sector)
WHERE s.heat_score IS NOT NULL
RETURN s, sector
ORDER BY s.heat_score DESC
LIMIT 30

============================================================

// 6. Heat Distribution by Level
// Shows count of stocks by heat level
MATCH (s:Stock)
WHERE s.heat_level IS NOT NULL
RETURN s.heat_level as heat_level, count(s) as count, 
       min(s.heat_score) as min_heat, max(s.heat_score) as max_heat,
       avg(s.heat_score) as avg_heat
ORDER BY avg_heat DESC

============================================================

// 7. Top Performing Stocks (Heat Based)
// Shows highest heat stocks with details
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN s.symbol as Symbol, s.heat_score as Heat_Score, 
       s.heat_level as Heat_Level, s.current_price as Price,
       s.composite_weight as Weight
ORDER BY s.heat_score DESC
LIMIT 15

============================================================

// 8. Real-time Graph Snapshot
// Complete view with all relationships and heat coloring
MATCH (n)-[r]-(m)
WHERE n.heat_score IS NOT NULL OR m.heat_score IS NOT NULL
RETURN n, r, m
LIMIT 200

============================================================

// 9. Heat Propagation Analysis
// Shows how heat might propagate through correlations
MATCH path = (hot:Stock)-[:CORRELATES*1..2]-(connected:Stock)
WHERE hot.heat_score > 0.02
RETURN path
LIMIT 50

============================================================

// 10. Dynamic Update Query (Run periodically)
// This query should be run every minute to see live updates
MATCH (s:Stock)
WHERE datetime(s.last_updated) > datetime() - duration('PT2M')
OPTIONAL MATCH (s)-[r]-(connected)
RETURN s, r, connected
ORDER BY s.heat_score DESC