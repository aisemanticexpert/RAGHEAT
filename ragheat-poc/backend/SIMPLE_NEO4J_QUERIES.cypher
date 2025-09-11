// =================================================================
// SIMPLE NEO4J QUERIES FOR RAGHEAT VISUALIZATION
// Copy and paste these directly into Neo4j Browser
// =================================================================

// 1. MAIN HEAT VISUALIZATION - START WITH THIS ONE
MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN s
ORDER BY s.heat_score DESC

// =================================================================

// 2. VERIFY DATA EXISTS - RUN THIS IF MAIN QUERY SHOWS NO RESULTS
MATCH (s:Stock)
RETURN count(s) as total_stocks

// =================================================================

// 3. CHECK HEAT VALUES - SHOWS CURRENT DATA
MATCH (s:Stock)
RETURN s.symbol as Symbol, s.heat_score as Heat, s.heat_level as Level
ORDER BY s.heat_score DESC
LIMIT 10

// =================================================================

// 4. COMPLETE GRAPH WITH RELATIONSHIPS
MATCH (s:Stock)-[r]-(other)
RETURN s, r, other
LIMIT 50

// =================================================================

// 5. LIVE UPDATES - RUN THIS EVERY MINUTE FOR REAL-TIME
MATCH (s:Stock)
WHERE datetime(s.last_updated) > datetime() - duration('PT5M')
RETURN s
ORDER BY s.heat_score DESC

// =================================================================

// 6. DEBUG - CHECK IF NODES HAVE REQUIRED PROPERTIES
MATCH (s:Stock)
RETURN s.symbol, s.heat_score, s.heat_level, s.last_updated
ORDER BY s.heat_score DESC NULLS LAST
LIMIT 5