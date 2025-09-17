MATCH (s:Stock)
WHERE datetime(s.last_updated) > datetime() - duration('PT5M')
RETURN s
ORDER BY s.heat_score DESC