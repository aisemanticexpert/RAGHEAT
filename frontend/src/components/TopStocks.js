import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, Zap, Star } from 'lucide-react';
import CountUp from 'react-countup';

const TopStocks = ({ recommendations, onStockSelect, selectedStock, realStocks, dataSource }) => {
  console.log('🔄 TopStocks received:', {
    recommendations: recommendations?.length,
    realStocks: realStocks?.length,
    dataSource
  });

  // PROFESSIONAL DATA ONLY - NO MOCK DATA
  let stocks = [];
  
  if (recommendations && recommendations.length > 0) {
    // Use professional recommendations (from Dashboard's realStocks)
    stocks = recommendations.map(stock => ({
      symbol: stock.symbol,
      sector: stock.sector || 'Technology',
      heat_score: stock.heat_score || (Math.abs(stock.change_percent || 0) / 10),
      price: stock.price || 0,
      change: (stock.change_percent || 0) * (stock.price || 0) / 100, // Calculate dollar change
      changePercent: stock.change_percent || 0,
      data_source: stock.data_source
    }));
    console.log('✅ Using professional recommendations:', stocks.length);
  } else if (realStocks && realStocks.length > 0) {
    // Fallback to realStocks if recommendations not available
    stocks = realStocks.map(stock => ({
      symbol: stock.symbol,
      sector: stock.sector || 'Technology',
      heat_score: stock.heat_score || (Math.abs(stock.change_percent || 0) / 10),
      price: stock.price || 0,
      change: (stock.change_percent || 0) * (stock.price || 0) / 100,
      changePercent: stock.change_percent || 0,
      data_source: stock.data_source
    })).slice(0, 8);
    console.log('✅ Using professional realStocks:', stocks.length);
  } else {
    console.error('❌ NO PROFESSIONAL DATA AVAILABLE - TopStocks will show empty');
  }

  const getTrendIcon = (change) => {
    if (change > 0) return <TrendingUp size={14} />;
    if (change < 0) return <TrendingDown size={14} />;
    return <Minus size={14} />;
  };

  const getTrendClass = (change) => {
    if (change > 0) return 'positive';
    if (change < 0) return 'negative';
    return 'neutral';
  };

  const getHeatLevel = (score) => {
    if (score >= 0.8) return 'hot';
    if (score >= 0.6) return 'warm';
    return 'cool';
  };

  return (
    <div className="top-stocks">
      <div className="panel-header">
        <h2>Top Heat Signals</h2>
        <div className="heat-legend">
          <span className="legend-item hot">
            <Zap size={12} /> Hot
          </span>
          <span className="legend-item warm">
            <Zap size={12} /> Warm
          </span>
          <span className="legend-item cool">
            <Zap size={12} /> Cool
          </span>
        </div>
      </div>

      <div className="stocks-list">
        <AnimatePresence>
          {stocks.slice(0, 8).map((stock, index) => (
            <motion.div
              key={stock.symbol}
              className={`stock-item ${selectedStock?.symbol === stock.symbol ? 'selected' : ''}`}
              onClick={() => onStockSelect(stock)}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="stock-rank">
                {index + 1}
                {index < 3 && <Star size={12} className="top-performer" />}
              </div>

              <div className="stock-info">
                <div className="stock-symbol">{stock.symbol}</div>
                <div className="stock-sector">{stock.sector}</div>
              </div>

              <div className="stock-metrics">
                <div className="stock-price">
                  $<CountUp 
                    end={stock.price || 0} 
                    duration={1.5} 
                    decimals={2}
                    preserveValue={true}
                  />
                </div>
                <div className={`stock-change ${getTrendClass(stock.change || 0)}`}>
                  {getTrendIcon(stock.change || 0)}
                  <span>{stock.changePercent ? `${stock.changePercent.toFixed(2)}%` : '0.00%'}</span>
                </div>
              </div>

              <div className="heat-indicator">
                <div 
                  className={`heat-bar ${getHeatLevel(stock.heat_score)}`}
                  style={{ width: `${(stock.heat_score * 100)}%` }}
                />
                <div className="heat-score">
                  <CountUp 
                    end={stock.heat_score * 100} 
                    duration={2} 
                    decimals={0}
                  />%
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      <style jsx>{`
        .top-stocks {
          height: 100%;
        }

        .heat-legend {
          display: flex;
          gap: 1rem;
          font-size: 0.75rem;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          padding: 0.25rem 0.5rem;
          border-radius: 12px;
          background: rgba(255, 255, 255, 0.1);
        }

        .legend-item.hot {
          color: #ff6b35;
        }

        .legend-item.warm {
          color: #ffd700;
        }

        .legend-item.cool {
          color: #00d4ff;
        }

        .stocks-list {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          max-height: 500px;
          overflow-y: auto;
        }

        .stock-item {
          display: grid;
          grid-template-columns: 40px 1fr auto 80px;
          align-items: center;
          gap: 1rem;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(0, 212, 255, 0.2);
          border-radius: 12px;
          cursor: pointer;
          transition: all 0.3s ease;
          position: relative;
          overflow: hidden;
        }

        .stock-item:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: #00d4ff;
          transform: translateX(4px);
        }

        .stock-item.selected {
          border-color: #00ff88;
          background: rgba(0, 255, 136, 0.1);
        }

        .stock-item::before {
          content: '';
          position: absolute;
          left: 0;
          top: 0;
          bottom: 0;
          width: 3px;
          background: var(--accent-color, #00d4ff);
          transition: all 0.3s ease;
        }

        .stock-item.selected::before {
          background: #00ff88;
        }

        .stock-rank {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: rgba(0, 212, 255, 0.2);
          color: #00d4ff;
          font-weight: 700;
          font-size: 0.875rem;
          position: relative;
        }

        .top-performer {
          position: absolute;
          top: -2px;
          right: -2px;
          color: #ffd700;
        }

        .stock-info {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .stock-symbol {
          font-weight: 700;
          font-size: 1rem;
          color: white;
        }

        .stock-sector {
          font-size: 0.75rem;
          color: #a0a0c0;
        }

        .stock-metrics {
          text-align: right;
        }

        .stock-price {
          font-weight: 700;
          font-family: 'Courier New', monospace;
          margin-bottom: 0.25rem;
        }

        .stock-change {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 0.25rem;
          font-size: 0.75rem;
          font-weight: 600;
        }

        .stock-change.positive {
          color: #00ff88;
        }

        .stock-change.negative {
          color: #ff6b35;
        }

        .stock-change.neutral {
          color: #a0a0c0;
        }

        .heat-indicator {
          position: relative;
          width: 60px;
        }

        .heat-bar {
          height: 4px;
          border-radius: 2px;
          background: #333;
          position: relative;
          overflow: hidden;
        }

        .heat-bar::after {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          height: 100%;
          width: 100%;
          border-radius: 2px;
          transition: all 0.8s ease;
        }

        .heat-bar.hot::after {
          background: linear-gradient(90deg, #ff6b35, #ff8c42);
        }

        .heat-bar.warm::after {
          background: linear-gradient(90deg, #ffd700, #ffed4a);
        }

        .heat-bar.cool::after {
          background: linear-gradient(90deg, #00d4ff, #42a5f5);
        }

        .heat-score {
          font-size: 0.75rem;
          font-weight: 700;
          text-align: center;
          margin-top: 0.25rem;
          font-family: 'Courier New', monospace;
        }

        .stocks-list::-webkit-scrollbar {
          width: 4px;
        }

        .stocks-list::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 2px;
        }

        .stocks-list::-webkit-scrollbar-thumb {
          background: #00d4ff;
          border-radius: 2px;
        }
      `}</style>
    </div>
  );
};

export default TopStocks;