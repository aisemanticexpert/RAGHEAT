// CORRECT Neo4j Browser Queries for RAGHeat Dynamic Visualization
// Use these queries to see ALL stocks with heat-based colors

// ==========================================
// 1. MAIN HEAT VISUALIZATION QUERY - USE THIS FIRST
// ==========================================
// Shows all Stock nodes with heat scores - THIS IS WHAT YOU NEED
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN s
ORDER BY s.heat_score DESC

// ==========================================
// 2. STOCK NODES ONLY WITH DETAILS
// ==========================================
// Table view of all heated stocks with their data
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN 
  s.symbol as Symbol,
  s.heat_score as Heat_Score, 
  s.heat_level as Heat_Level,
  s.current_price as Price,
  s.background_color as Color,
  s.last_updated as Last_Updated
ORDER BY s.heat_score DESC

// ==========================================
// 3. FULL GRAPH WITH STOCKS AND RELATIONSHIPS
// ==========================================
// Shows stocks with their company and sector connections
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
OPTIONAL MATCH (s)-[r1]-(company:Company)
OPTIONAL MATCH (company)-[r2]-(sector:Sector)
RETURN s, r1, company, r2, sector
LIMIT 50

// ==========================================
// 4. TOP HEATED STOCKS ONLY
// ==========================================
// Focus on highest heat stocks
MATCH (s:Stock)
WHERE s.heat_score > 0.04
RETURN s
ORDER BY s.heat_score DESC

// ==========================================
// 5. REAL-TIME UPDATES QUERY
// ==========================================
// Run this every minute to see live updates
MATCH (s:Stock)
WHERE datetime(s.last_updated) > datetime() - duration('PT5M')
RETURN s
ORDER BY s.heat_score DESC

// ==========================================
// 6. STOCK CORRELATION NETWORK
// ==========================================
// Shows how stocks are connected through correlations
MATCH (s1:Stock)-[r:CORRELATES]-(s2:Stock)
WHERE s1.heat_score IS NOT NULL AND s2.heat_score IS NOT NULL
RETURN s1, r, s2
ORDER BY s1.heat_score DESC
LIMIT 30

// ==========================================
// 7. COMPLETE KNOWLEDGE GRAPH
// ==========================================
// Shows everything: stocks, companies, sectors, market
MATCH (n)
WHERE n.heat_score IS NOT NULL OR 
      labels(n)[0] IN ['Company', 'Sector', 'Market']
OPTIONAL MATCH (n)-[r]-(connected)
RETURN n, r, connected
LIMIT 100

// ==========================================
// 8. HEAT DISTRIBUTION ANALYSIS
// ==========================================
// Count stocks by heat level
MATCH (s:Stock)
WHERE s.heat_level IS NOT NULL
RETURN 
  s.heat_level as Level,
  count(s) as Count,
  min(s.heat_score) as Min_Heat,
  max(s.heat_score) as Max_Heat,
  avg(s.heat_score) as Avg_Heat
ORDER BY Avg_Heat DESC

// ==========================================
// 9. CURRENT MARKET LEADERS
// ==========================================
// Top 10 stocks with highest heat right now
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN s
ORDER BY s.heat_score DESC
LIMIT 10

// ==========================================
// 10. DEBUGGING QUERY - CHECK ALL DATA
// ==========================================
// Use this to verify all properties are set correctly
MATCH (s:Stock)
RETURN 
  s.symbol,
  s.heat_score,
  s.heat_level,
  s.background_color,
  s.current_price,
  s.last_updated,
  properties(s) as all_properties
ORDER BY s.heat_score DESC
LIMIT 5