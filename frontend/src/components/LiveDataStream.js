import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity,
  Wifi,
  WifiOff,
  RefreshCw,
  Zap,
  BarChart3,
  Clock,
  Target
} from 'lucide-react';
import CountUp from 'react-countup';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8002';
const WS_URL = API_BASE_URL.replace('http', 'ws');

const LiveDataStream = () => {
  const [stockData, setStockData] = useState({});
  const [connectionStatus, setConnectionStatus] = useState('Connecting...');
  const [lastUpdate, setLastUpdate] = useState(null);
  const [streamStats, setStreamStats] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const connectWebSocket = () => {
    try {
      // Close existing connection if any
      if (wsRef.current) {
        wsRef.current.close();
      }

      console.log('Connecting to WebSocket:', `${WS_URL}/ws`);
      wsRef.current = new WebSocket(`${WS_URL}/ws`);

      wsRef.current.onopen = (event) => {
        console.log('WebSocket connected:', event);
        setConnectionStatus('🔥 Live Stream Connected');
        setReconnectAttempts(0);
        
        // Send a ping to keep connection alive
        const pingInterval = setInterval(() => {
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'ping' }));
          }
        }, 25000);

        wsRef.current.pingInterval = pingInterval;
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message received:', data.type);

          if (data.type === 'market_update') {
            setStockData(data.data);
            setLastUpdate(new Date(data.timestamp));
            setStreamStats({
              count: data.count,
              fetchTime: data.fetch_time_ms,
              timestamp: data.timestamp
            });
            setConnectionStatus('📡 Live Data Streaming');
          } else if (data.type === 'connection_status') {
            setConnectionStatus('✅ Connected - Waiting for data...');
          } else if (data.type === 'pong') {
            console.log('Pong received');
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setConnectionStatus('❌ Connection Lost - Reconnecting...');
        
        // Clear ping interval
        if (wsRef.current && wsRef.current.pingInterval) {
          clearInterval(wsRef.current.pingInterval);
        }

        // Attempt to reconnect
        if (reconnectAttempts < 5) {
          const timeout = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts(prev => prev + 1);
            connectWebSocket();
          }, timeout);
        } else {
          setConnectionStatus('💥 Connection Failed - Max retries reached');
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('⚠️ Connection Error');
      };

    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setConnectionStatus('💥 Failed to Connect');
    }
  };

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        if (wsRef.current.pingInterval) {
          clearInterval(wsRef.current.pingInterval);
        }
        wsRef.current.close();
      }
    };
  }, []);

  const getChangeColor = (changePercent) => {
    if (changePercent > 0) return 'text-green-400';
    if (changePercent < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  const getChangeIcon = (changePercent) => {
    if (changePercent > 0) return <TrendingUp size={16} className="text-green-400" />;
    if (changePercent < 0) return <TrendingDown size={16} className="text-red-400" />;
    return <Activity size={16} className="text-gray-400" />;
  };

  const getHeatColor = (heatScore) => {
    if (heatScore > 0.8) return 'bg-purple-500';
    if (heatScore > 0.6) return 'bg-orange-500';
    if (heatScore > 0.4) return 'bg-yellow-500';
    return 'bg-blue-500';
  };

  const manualRefresh = async () => {
    try {
      // Trigger a manual refresh by reconnecting
      connectWebSocket();
    } catch (error) {
      console.error('Manual refresh failed:', error);
    }
  };

  const stockEntries = Object.entries(stockData).sort((a, b) => 
    Math.abs(b[1].change_percent) - Math.abs(a[1].change_percent)
  );

  return (
    <div className="live-data-stream">
      {/* Header */}
      <div className="stream-header">
        <div className="header-left">
          <h2>🔥 Live Market Data Stream</h2>
          <div className="connection-status">
            <div className={`status-indicator ${connectionStatus.includes('Connected') || connectionStatus.includes('Streaming') ? 'connected' : 'disconnected'}`}>
              {connectionStatus.includes('Connected') || connectionStatus.includes('Streaming') ? 
                <Wifi size={16} /> : <WifiOff size={16} />
              }
              <span>{connectionStatus}</span>
            </div>
          </div>
        </div>
        
        <div className="header-controls">
          <button onClick={manualRefresh} className="refresh-btn">
            <RefreshCw size={16} />
            Reconnect
          </button>
        </div>
      </div>

      {/* Stream Stats */}
      {streamStats && (
        <div className="stream-stats">
          <div className="stat-card">
            <BarChart3 size={16} />
            <div>
              <span className="stat-label">Stocks Streaming</span>
              <span className="stat-value">
                <CountUp end={streamStats.count} duration={0.5} />
              </span>
            </div>
          </div>
          
          <div className="stat-card">
            <Zap size={16} />
            <div>
              <span className="stat-label">Fetch Speed</span>
              <span className="stat-value">
                <CountUp end={streamStats.fetchTime} duration={0.5} />ms
              </span>
            </div>
          </div>
          
          <div className="stat-card">
            <Clock size={16} />
            <div>
              <span className="stat-label">Last Update</span>
              <span className="stat-value">
                {lastUpdate ? lastUpdate.toLocaleTimeString() : 'Never'}
              </span>
            </div>
          </div>

          <div className="stat-card">
            <Target size={16} />
            <div>
              <span className="stat-label">Data Source</span>
              <span className="stat-value">Yahoo Finance</span>
            </div>
          </div>
        </div>
      )}

      {/* Live Stock Data */}
      <div className="stocks-grid">
        <AnimatePresence>
          {stockEntries.map(([symbol, stock], index) => (
            <motion.div
              key={symbol}
              className="stock-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="stock-header">
                <div className="symbol-info">
                  <span className="symbol">{symbol}</span>
                  <span className="source">{stock.source || 'yahoo'}</span>
                </div>
                <div className="change-indicator">
                  {getChangeIcon(stock.change_percent)}
                </div>
              </div>

              <div className="price-section">
                <div className="current-price">
                  $<CountUp 
                    end={stock.price} 
                    duration={1} 
                    decimals={2}
                    preserveValue
                  />
                </div>
                
                <div className={`price-change ${getChangeColor(stock.change_percent)}`}>
                  <span className="change-amount">
                    {stock.change >= 0 ? '+' : ''}
                    <CountUp 
                      end={stock.change} 
                      duration={1} 
                      decimals={2}
                      preserveValue
                    />
                  </span>
                  <span className="change-percent">
                    ({stock.change_percent >= 0 ? '+' : ''}
                    <CountUp 
                      end={stock.change_percent} 
                      duration={1} 
                      decimals={2}
                      preserveValue
                    />%)
                  </span>
                </div>
              </div>

              <div className="stock-details">
                <div className="detail-row">
                  <span className="label">Volume:</span>
                  <span className="value">{stock.volume?.toLocaleString() || 'N/A'}</span>
                </div>
                
                <div className="detail-row">
                  <span className="label">High:</span>
                  <span className="value">${stock.high?.toFixed(2) || stock.price?.toFixed(2)}</span>
                </div>
                
                <div className="detail-row">
                  <span className="label">Low:</span>
                  <span className="value">${stock.low?.toFixed(2) || stock.price?.toFixed(2)}</span>
                </div>
                
                <div className="detail-row">
                  <span className="label">Updated:</span>
                  <span className="value">
                    {stock.timestamp ? new Date(stock.timestamp).toLocaleTimeString() : 'Live'}
                  </span>
                </div>
              </div>

              <div className="ranking-badge">
                #{index + 1}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {Object.keys(stockData).length === 0 && (
        <div className="no-data">
          <Activity size={48} />
          <h3>Waiting for Live Data...</h3>
          <p>Connecting to real-time market data stream</p>
          <button onClick={manualRefresh} className="retry-btn">
            <RefreshCw size={16} />
            Retry Connection
          </button>
        </div>
      )}

      <style jsx>{`
        .live-data-stream {
          padding: 2rem;
          color: white;
          background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
          min-height: 100vh;
        }

        .stream-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid rgba(0, 212, 255, 0.2);
        }

        .header-left h2 {
          font-size: 2rem;
          margin-bottom: 0.5rem;
          color: #00d4ff;
          text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }

        .connection-status {
          margin-top: 0.5rem;
        }

        .status-indicator {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          border-radius: 20px;
          font-size: 0.875rem;
          font-weight: 600;
          transition: all 0.3s ease;
        }

        .status-indicator.connected {
          background: rgba(0, 255, 136, 0.2);
          color: #00ff88;
          border: 1px solid #00ff88;
          animation: pulse-green 2s infinite;
        }

        .status-indicator.disconnected {
          background: rgba(255, 107, 53, 0.2);
          color: #ff6b35;
          border: 1px solid #ff6b35;
          animation: pulse-red 2s infinite;
        }

        .refresh-btn, .retry-btn {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.75rem 1.5rem;
          background: linear-gradient(135deg, #00d4ff, #42a5f5);
          color: white;
          border: none;
          border-radius: 8px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .refresh-btn:hover, .retry-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4);
        }

        .stream-stats {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1rem;
          margin-bottom: 2rem;
        }

        .stat-card {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(0, 212, 255, 0.2);
          border-radius: 12px;
          color: #00d4ff;
          backdrop-filter: blur(10px);
        }

        .stat-label {
          display: block;
          font-size: 0.75rem;
          color: #a0a0c0;
          margin-bottom: 0.25rem;
        }

        .stat-value {
          font-size: 1.25rem;
          font-weight: 700;
          font-family: 'Courier New', monospace;
          color: white;
        }

        .stocks-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
          gap: 1.5rem;
        }

        .stock-card {
          position: relative;
          padding: 1.5rem;
          background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
          border: 1px solid rgba(0, 212, 255, 0.2);
          border-radius: 16px;
          backdrop-filter: blur(20px);
          transition: all 0.3s ease;
          overflow: hidden;
        }

        .stock-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 3px;
          background: linear-gradient(90deg, #00d4ff, #00ff88, #ffd700);
          opacity: 0;
          transition: opacity 0.3s ease;
        }

        .stock-card:hover::before {
          opacity: 1;
        }

        .stock-card:hover {
          border-color: #00d4ff;
          transform: translateY(-5px);
          box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
        }

        .stock-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .symbol {
          font-size: 1.5rem;
          font-weight: 700;
          color: white;
          text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
        }

        .source {
          font-size: 0.75rem;
          color: #a0a0c0;
          background: rgba(0, 212, 255, 0.2);
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          margin-top: 0.25rem;
          display: block;
        }

        .price-section {
          margin-bottom: 1.5rem;
        }

        .current-price {
          font-size: 2.5rem;
          font-weight: 700;
          color: white;
          font-family: 'Courier New', monospace;
          text-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
          margin-bottom: 0.5rem;
        }

        .price-change {
          display: flex;
          gap: 0.5rem;
          font-size: 1rem;
          font-weight: 600;
        }

        .change-amount, .change-percent {
          font-family: 'Courier New', monospace;
        }

        .stock-details {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .detail-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.875rem;
        }

        .detail-row .label {
          color: #a0a0c0;
        }

        .detail-row .value {
          color: white;
          font-weight: 600;
          font-family: 'Courier New', monospace;
        }

        .ranking-badge {
          position: absolute;
          top: 1rem;
          right: 1rem;
          background: linear-gradient(135deg, #00d4ff, #00ff88);
          color: white;
          font-size: 0.75rem;
          font-weight: 700;
          padding: 0.25rem 0.5rem;
          border-radius: 12px;
          min-width: 30px;
          text-align: center;
        }

        .no-data {
          text-align: center;
          padding: 4rem 2rem;
          color: #a0a0c0;
        }

        .no-data svg {
          margin-bottom: 1rem;
          opacity: 0.5;
          animation: pulse 2s infinite;
        }

        .no-data h3 {
          font-size: 1.5rem;
          margin-bottom: 0.5rem;
          color: #00d4ff;
        }

        .text-green-400 {
          color: #4ade80;
        }

        .text-red-400 {
          color: #f87171;
        }

        .text-gray-400 {
          color: #9ca3af;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @keyframes pulse-green {
          0%, 100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
          70% { box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
        }

        @keyframes pulse-red {
          0%, 100% { box-shadow: 0 0 0 0 rgba(255, 107, 53, 0.7); }
          70% { box-shadow: 0 0 0 10px rgba(255, 107, 53, 0); }
        }

        @media (max-width: 768px) {
          .stocks-grid {
            grid-template-columns: 1fr;
          }
          
          .stream-header {
            flex-direction: column;
            gap: 1rem;
          }
          
          .stream-stats {
            grid-template-columns: repeat(2, 1fr);
          }
        }
      `}</style>
    </div>
  );
};

export default LiveDataStream;