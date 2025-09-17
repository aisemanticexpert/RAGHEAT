import React from 'react';
import { motion } from 'framer-motion';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area 
} from 'recharts';
import { TrendingUp, TrendingDown, Target, Zap } from 'lucide-react';

const StockChart = ({ selectedStock }) => {
  // Generate mock price data for the selected stock
  const generatePriceData = (stock) => {
    const basePrice = stock.price || 100;
    const data = [];
    const now = new Date();
    
    for (let i = 30; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60 * 1000); // 1 minute intervals
      const randomChange = (Math.random() - 0.5) * (basePrice * 0.02);
      const price = basePrice + randomChange;
      
      data.push({
        time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        price: price,
        volume: Math.floor(Math.random() * 10000) + 5000,
        heat: Math.random() * 100
      });
    }
    
    return data;
  };

  const priceData = generatePriceData(selectedStock);
  const currentPrice = priceData[priceData.length - 1]?.price || 0;
  const previousPrice = priceData[priceData.length - 2]?.price || currentPrice;
  const priceChange = currentPrice - previousPrice;
  const priceChangePercent = (priceChange / previousPrice) * 100;

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-time">{`Time: ${label}`}</p>
          <p className="tooltip-price">
            Price: <span>${payload[0].value.toFixed(2)}</span>
          </p>
          {payload[1] && (
            <p className="tooltip-volume">
              Volume: <span>{payload[1].value.toLocaleString()}</span>
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div 
      className="stock-chart"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="chart-header">
        <div className="stock-title">
          <h2>{selectedStock.symbol}</h2>
          <div className="stock-details">
            <span className="current-price">
              ${currentPrice.toFixed(2)}
            </span>
            <span className={`price-change ${priceChange >= 0 ? 'positive' : 'negative'}`}>
              {priceChange >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
              {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} 
              ({priceChangePercent >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%)
            </span>
          </div>
        </div>

        <div className="chart-metrics">
          <div className="metric">
            <Zap size={16} />
            <span>Heat Score: {(selectedStock.heat_score * 100).toFixed(0)}%</span>
          </div>
          <div className="metric">
            <Target size={16} />
            <span>Sector: {selectedStock.sector}</span>
          </div>
        </div>
      </div>

      <div className="charts-container">
        <div className="price-chart">
          <h3>Price Movement (Last 30 Minutes)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={priceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="time" 
                stroke="#a0a0c0"
                fontSize={12}
                interval="preserveStartEnd"
              />
              <YAxis 
                stroke="#a0a0c0"
                fontSize={12}
                domain={['dataMin - 0.5', 'dataMax + 0.5']}
                tickFormatter={(value) => `$${value.toFixed(2)}`}
              />
              <Tooltip content={<CustomTooltip />} />
              <defs>
                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#00d4ff" stopOpacity={0.05}/>
                </linearGradient>
              </defs>
              <Area
                type="monotone"
                dataKey="price"
                stroke="#00d4ff"
                strokeWidth={2}
                fill="url(#priceGradient)"
                dot={false}
                activeDot={{ r: 4, fill: "#00d4ff", stroke: "#fff", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="volume-chart">
          <h3>Volume & Heat Activity</h3>
          <ResponsiveContainer width="100%" height={150}>
            <AreaChart data={priceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="time" 
                stroke="#a0a0c0"
                fontSize={12}
                interval="preserveStartEnd"
              />
              <YAxis 
                stroke="#a0a0c0"
                fontSize={12}
                tickFormatter={(value) => `${(value/1000).toFixed(0)}K`}
              />
              <Tooltip />
              <defs>
                <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ffd700" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#ffd700" stopOpacity={0.05}/>
                </linearGradient>
              </defs>
              <Area
                type="monotone"
                dataKey="volume"
                stroke="#ffd700"
                strokeWidth={2}
                fill="url(#volumeGradient)"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Analysis Panel */}
      <div className="analysis-panel">
        <h3>AI Analysis</h3>
        <div className="analysis-grid">
          <div className="analysis-item">
            <div className="analysis-label">Recommendation</div>
            <div className="analysis-value buy">STRONG BUY</div>
          </div>
          <div className="analysis-item">
            <div className="analysis-label">Confidence</div>
            <div className="analysis-value">87%</div>
          </div>
          <div className="analysis-item">
            <div className="analysis-label">Target Price</div>
            <div className="analysis-value">${(currentPrice * 1.15).toFixed(2)}</div>
          </div>
          <div className="analysis-item">
            <div className="analysis-label">Risk Level</div>
            <div className="analysis-value moderate">MODERATE</div>
          </div>
        </div>
        <div className="analysis-summary">
          <p>
            Strong momentum detected with increasing volume and positive heat signals. 
            Technical indicators suggest continued upward movement with key support at ${(currentPrice * 0.95).toFixed(2)}.
          </p>
        </div>
      </div>

      <style jsx>{`
        .stock-chart {
          background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
          border: 1px solid rgba(0, 212, 255, 0.3);
          border-radius: 16px;
          padding: 1.5rem;
          backdrop-filter: blur(20px);
        }

        .chart-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 2rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid rgba(0, 212, 255, 0.2);
        }

        .stock-title h2 {
          font-size: 1.5rem;
          font-weight: 700;
          color: #00d4ff;
          margin-bottom: 0.5rem;
        }

        .stock-details {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .current-price {
          font-size: 1.25rem;
          font-weight: 700;
          font-family: 'Courier New', monospace;
        }

        .price-change {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          font-size: 1rem;
          font-weight: 600;
        }

        .price-change.positive {
          color: #00ff88;
        }

        .price-change.negative {
          color: #ff6b35;
        }

        .chart-metrics {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .metric {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.875rem;
          color: #a0a0c0;
        }

        .charts-container {
          margin-bottom: 2rem;
        }

        .price-chart,
        .volume-chart {
          margin-bottom: 2rem;
        }

        .price-chart h3,
        .volume-chart h3 {
          font-size: 1rem;
          margin-bottom: 1rem;
          color: #00d4ff;
        }

        .custom-tooltip {
          background: rgba(15, 15, 35, 0.95);
          border: 1px solid #00d4ff;
          border-radius: 8px;
          padding: 1rem;
          backdrop-filter: blur(10px);
        }

        .tooltip-time {
          color: #a0a0c0;
          margin-bottom: 0.5rem;
          font-size: 0.875rem;
        }

        .tooltip-price,
        .tooltip-volume {
          margin-bottom: 0.25rem;
          font-size: 0.875rem;
        }

        .tooltip-price span,
        .tooltip-volume span {
          color: #00d4ff;
          font-weight: 600;
        }

        .analysis-panel {
          border-top: 1px solid rgba(0, 212, 255, 0.2);
          padding-top: 1.5rem;
        }

        .analysis-panel h3 {
          font-size: 1.125rem;
          margin-bottom: 1rem;
          color: #00d4ff;
        }

        .analysis-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 1rem;
          margin-bottom: 1.5rem;
        }

        .analysis-item {
          text-align: center;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 12px;
          border: 1px solid rgba(0, 212, 255, 0.2);
        }

        .analysis-label {
          font-size: 0.75rem;
          color: #a0a0c0;
          margin-bottom: 0.5rem;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .analysis-value {
          font-size: 1.125rem;
          font-weight: 700;
          font-family: 'Courier New', monospace;
        }

        .analysis-value.buy {
          color: #00ff88;
        }

        .analysis-value.moderate {
          color: #ffd700;
        }

        .analysis-summary {
          background: rgba(255, 255, 255, 0.05);
          padding: 1rem;
          border-radius: 12px;
          border-left: 3px solid #00d4ff;
        }

        .analysis-summary p {
          margin: 0;
          line-height: 1.6;
          color: #a0a0c0;
        }

        @media (max-width: 768px) {
          .chart-header {
            flex-direction: column;
            gap: 1rem;
          }

          .analysis-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }
      `}</style>
    </motion.div>
  );
};

export default StockChart;