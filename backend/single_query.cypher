MATCH (s:Stock)
WHERE s.heat_score IS NOT NULL
RETURN s
ORDER BY s.heat_score DESC


MATCH (s:Stock)
WHERE datetime(s.last_updated) > datetime() - duration('PT5M')
RETURN s
ORDER BY s.heat_score DESC

 MATCH (s:Stock)
     WHERE s.heat_score IS NOT NULL AND s.trading_signal IS NOT NULL
     RETURN s.symbol as Symbol,
            s.trading_signal as Signal,
            s.signal_confidence as Confidence,
            s.price_target as Target,
            s.stop_loss as StopLoss,
            s.current_price as Price,
            s.heat_score as Heat,
            s
