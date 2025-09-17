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
ORDER BY 
  CASE s.trading_signal
    WHEN 'STRONG_BUY' THEN 1
    WHEN 'BUY' THEN 2
    WHEN 'WEAK_BUY' THEN 3
    WHEN 'HOLD' THEN 4
    WHEN 'WEAK_SELL' THEN 5
    WHEN 'SELL' THEN 6
    WHEN 'STRONG_SELL' THEN 7
    ELSE 8
  END,
  s.signal_confidence DESC