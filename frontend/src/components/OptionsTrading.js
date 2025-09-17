import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  Target, 
  Zap, 
  AlertCircle, 
  Clock, 
  DollarSign,
  BarChart3,
  Bell,
  Settings,
  PlayCircle,
  PauseCircle,
  RefreshCw,
  ArrowUp,
  ArrowDown,
  Activity
} from 'lucide-react';
import CountUp from 'react-countup';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8002';

const OptionsTrading = () => {
  const [signals, setSignals] = useState({});
  const [hotSignals, setHotSignals] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  const [marketStatus, setMarketStatus] = useState('CLOSED');
  const [performance, setPerformance] = useState(null);
  const [settings, setSettings] = useState({
    min_probability: 0.7,
    min_heat_score: 0.6,
    symbols: ['SPY', 'QQQ', 'SOXS', 'IWM', 'ARKK'],
    actions: ['BUY_CALL', 'BUY_PUT']
  });
  const [loading, setLoading] = useState(false);

  // Fetch options signals
  const fetchOptionsSignals = async () => {
    if (!isMonitoring) return;
    
    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE_URL}/api/options/bulk-signals`, {
        symbols: settings.symbols
      });
      
      if (response.data) {
        const signalsMap = {};
        response.data.forEach(signal => {
          signalsMap[signal.symbol] = signal;
        });
        setSignals(signalsMap);
      }
    } catch (error) {
      console.error('Error fetching options signals:', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch hot signals for notifications
  const fetchHotSignals = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/options/hot-signals`);
      if (response.data?.signals) {
        const newHotSignals = response.data.signals;
        
        // Check for new hot signals to create notifications
        newHotSignals.forEach(signal => {
          const existingNotification = notifications.find(n => 
            n.symbol === signal.symbol && 
            n.action === signal.action &&
            Math.abs(new Date(n.timestamp) - new Date(signal.timestamp)) < 60000 // Within 1 minute
          );
          
          if (!existingNotification) {
            const notification = {
              id: Date.now() + Math.random(),
              symbol: signal.symbol,
              action: signal.action,
              probability: signal.probability,
              heat_score: signal.heat_score,
              strike_price: signal.strike_price,
              pathrag_reasoning: signal.pathrag_reasoning,
              timestamp: signal.timestamp,
              type: signal.probability > 0.8 ? 'high' : signal.probability > 0.7 ? 'medium' : 'low'
            };
            
            setNotifications(prev => [notification, ...prev.slice(0, 9)]); // Keep last 10
          }
        });
        
        setHotSignals(newHotSignals);
      }
    } catch (error) {
      console.error('Error fetching hot signals:', error);
    }
  };

  // Fetch market status
  const fetchMarketStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/options/market-hours`);
      if (response.data) {
        setMarketStatus(response.data.status);
      }
    } catch (error) {
      console.error('Error fetching market status:', error);
    }
  };

  // Fetch performance metrics
  const fetchPerformance = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/options/performance-metrics`);
      if (response.data) {
        setPerformance(response.data);
      }
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
    }
  };

  // Auto-refresh data
  useEffect(() => {
    if (isMonitoring) {
      fetchOptionsSignals();
      fetchHotSignals();
      fetchMarketStatus();
      fetchPerformance();
      
      const interval = setInterval(() => {
        fetchOptionsSignals();
        fetchHotSignals();
        fetchMarketStatus();
      }, 15000); // 15 seconds
      
      const performanceInterval = setInterval(fetchPerformance, 60000); // 1 minute
      
      return () => {
        clearInterval(interval);
        clearInterval(performanceInterval);
      };
    }
  }, [isMonitoring, settings.symbols]);

  const getActionIcon = (action) => {
    switch (action) {
      case 'BUY_CALL':
        return <TrendingUp size={16} className="text-green-400" />;
      case 'BUY_PUT':
        return <TrendingDown size={16} className="text-red-400" />;
      case 'SELL_CALL':
        return <ArrowDown size={16} className="text-orange-400" />;
      case 'SELL_PUT':
        return <ArrowUp size={16} className="text-blue-400" />;
      default:
        return <Activity size={16} className="text-gray-400" />;
    }
  };

  const getActionColor = (action) => {
    switch (action) {
      case 'BUY_CALL':
        return 'text-green-400 bg-green-400/20';
      case 'BUY_PUT':
        return 'text-red-400 bg-red-400/20';
      case 'SELL_CALL':
        return 'text-orange-400 bg-orange-400/20';
      case 'SELL_PUT':
        return 'text-blue-400 bg-blue-400/20';
      default:
        return 'text-gray-400 bg-gray-400/20';
    }
  };

  const getStrengthColor = (strength) => {
    switch (strength) {
      case 'STRONG_BUY':
      case 'STRONG_SELL':
        return 'text-purple-400';
      case 'BUY':
      case 'SELL':
        return 'text-green-400';
      case 'WEAK_BUY':
      case 'WEAK_SELL':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="options-trading">
      {/* Header Controls */}
      <div className="trading-header">
        <div className="header-left">
          <h2>🎯 Options Trading Signals</h2>
          <div className="market-status">
            <div className={`status-indicator ${marketStatus.toLowerCase()}`}>
              <div className="pulse-dot"></div>
              <span>Market {marketStatus}</span>
            </div>
          </div>
        </div>
        
        <div className="header-controls">
          <button
            onClick={() => setIsMonitoring(!isMonitoring)}
            className={`monitor-btn ${isMonitoring ? 'active' : 'inactive'}`}
          >
            {isMonitoring ? <PauseCircle size={16} /> : <PlayCircle size={16} />}
            {isMonitoring ? 'Pause' : 'Start'} Monitoring
          </button>
          
          <button onClick={fetchOptionsSignals} className="refresh-btn" disabled={loading}>
            <RefreshCw size={16} className={loading ? 'spinning' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* Performance Metrics */}
      {performance && (
        <div className="performance-metrics">
          <div className="metric-card">
            <BarChart3 size={16} />
            <div>
              <span className="metric-label">Total Signals</span>
              <span className="metric-value">
                <CountUp end={performance.total_signals || 0} duration={1} />
              </span>
            </div>
          </div>
          
          <div className="metric-card">
            <Target size={16} />
            <div>
              <span className="metric-label">High Prob Rate</span>
              <span className="metric-value">
                <CountUp end={(performance.high_prob_rate || 0) * 100} duration={1} decimals={1} />%
              </span>
            </div>
          </div>
          
          <div className="metric-card">
            <Zap size={16} />
            <div>
              <span className="metric-label">Avg Heat Score</span>
              <span className="metric-value">
                <CountUp end={(performance.avg_heat_score || 0) * 100} duration={1} decimals={0} />%
              </span>
            </div>
          </div>
          
          <div className="metric-card">
            <Clock size={16} />
            <div>
              <span className="metric-label">Today's Signals</span>
              <span className="metric-value">
                <CountUp end={performance.last_24h_signals || 0} duration={1} />
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="trading-grid">
        {/* Live Options Signals */}
        <div className="signals-panel">
          <div className="panel-header">
            <h3>🔥 Live Options Signals</h3>
            <div className="signal-count">
              {Object.keys(signals).length} Active
            </div>
          </div>
          
          <div className="signals-list">
            <AnimatePresence>
              {Object.entries(signals).map(([symbol, signal]) => (
                <motion.div
                  key={symbol}
                  className={`signal-card ${selectedSymbol === symbol ? 'selected' : ''}`}
                  onClick={() => setSelectedSymbol(symbol)}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="signal-header">
                    <div className="symbol-info">
                      <span className="symbol">{symbol}</span>
                      <div className={`action-badge ${getActionColor(signal.action)}`}>
                        {getActionIcon(signal.action)}
                        <span>{signal.action.replace('_', ' ')}</span>
                      </div>
                    </div>
                    
                    <div className="signal-metrics">
                      <div className="probability">
                        <span className="label">Probability</span>
                        <span className="value">
                          <CountUp end={(signal.probability || 0) * 100} duration={1} decimals={0} />%
                        </span>
                      </div>
                      <div className="heat-score">
                        <span className="label">Heat</span>
                        <span className="value">
                          <CountUp end={(signal.heat_score || 0) * 100} duration={1} decimals={0} />%
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="signal-details">
                    <div className="strike-expiry">
                      <DollarSign size={14} />
                      <span>Strike: ${signal.strike_price || 'N/A'}</span>
                      <Clock size={14} />
                      <span>Exp: {signal.expiration || 'N/A'}</span>
                    </div>
                    
                    <div className={`strength ${getStrengthColor(signal.strength)}`}>
                      {(signal.strength || 'NEUTRAL').replace('_', ' ')}
                    </div>
                  </div>
                  
                  <div className="risk-reward">
                    <div className="rr-ratio">
                      R/R: {signal.risk_reward?.ratio?.toFixed(2) || 'N/A'}
                    </div>
                    <div className="timestamp">
                      {new Date(signal.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* PATHRAG Analysis */}
        {selectedSymbol && signals[selectedSymbol] && (
          <div className="pathrag-panel">
            <div className="panel-header">
              <h3>🧠 PATHRAG Analysis - {selectedSymbol}</h3>
              <div className="analysis-score">
                AI Confidence: <span className="score">
                  <CountUp end={(signals[selectedSymbol].probability || 0) * 100} duration={1} decimals={0} />%
                </span>
              </div>
            </div>
            
            <div className="pathrag-content">
              <div className="reasoning-text">
                {signals[selectedSymbol].pathrag_reasoning || 'Analysis not available'}
              </div>
              
              <div className="technical-summary">
                <h4>Technical Indicators Summary</h4>
                <div className="indicators-grid">
                  {signals[selectedSymbol].technical_indicators && (
                    <>
                      <div className="indicator">
                        <span className="label">Price vs MA5:</span>
                        <span className={`value ${signals[selectedSymbol].technical_indicators.price > signals[selectedSymbol].technical_indicators.ma_5 ? 'bullish' : 'bearish'}`}>
                          {signals[selectedSymbol].technical_indicators.price > signals[selectedSymbol].technical_indicators.ma_5 ? 'Above' : 'Below'}
                        </span>
                      </div>
                      
                      <div className="indicator">
                        <span className="label">MACD Signal:</span>
                        <span className={`value ${signals[selectedSymbol].technical_indicators.macd?.macd > signals[selectedSymbol].technical_indicators.macd?.signal ? 'bullish' : 'bearish'}`}>
                          {signals[selectedSymbol].technical_indicators.macd?.macd > signals[selectedSymbol].technical_indicators.macd?.signal ? 'Bullish' : 'Bearish'}
                        </span>
                      </div>
                      
                      <div className="indicator">
                        <span className="label">RSI:</span>
                        <span className={`value ${signals[selectedSymbol].technical_indicators.rsi > 70 ? 'overbought' : signals[selectedSymbol].technical_indicators.rsi < 30 ? 'oversold' : 'neutral'}`}>
                          {signals[selectedSymbol].technical_indicators.rsi?.toFixed(0)}
                        </span>
                      </div>
                      
                      <div className="indicator">
                        <span className="label">Volume:</span>
                        <span className={`value ${signals[selectedSymbol].technical_indicators.volume_ratio > 1.5 ? 'high' : 'normal'}`}>
                          {signals[selectedSymbol].technical_indicators.volume_ratio?.toFixed(1)}x
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Real-time Notifications */}
        <div className="notifications-panel">
          <div className="panel-header">
            <h3>🔔 Live Alerts</h3>
            <Bell size={16} className={notifications.length > 0 ? 'active' : ''} />
          </div>
          
          <div className="notifications-list">
            <AnimatePresence>
              {notifications.map((notification) => (
                <motion.div
                  key={notification.id}
                  className={`notification-item ${notification.type}`}
                  initial={{ opacity: 0, x: 300 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -300 }}
                  layout
                >
                  <div className="notification-header">
                    <div className="symbol-action">
                      <span className="symbol">{notification.symbol}</span>
                      <div className={`action-badge ${getActionColor(notification.action)}`}>
                        {getActionIcon(notification.action)}
                        <span>{notification.action.replace('_', ' ')}</span>
                      </div>
                    </div>
                    <AlertCircle size={16} className="alert-icon" />
                  </div>
                  
                  <div className="notification-details">
                    <div className="prob-heat">
                      <span>P: {(notification.probability * 100).toFixed(0)}%</span>
                      <span>H: {(notification.heat_score * 100).toFixed(0)}%</span>
                      <span>${notification.strike_price}</span>
                    </div>
                    <div className="timestamp">
                      {new Date(notification.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            
            {notifications.length === 0 && (
              <div className="no-notifications">
                <Bell size={32} />
                <p>No active alerts</p>
                <small>High probability signals will appear here</small>
              </div>
            )}
          </div>
        </div>
      </div>

      <style jsx>{`
        .options-trading {
          padding: 2rem;
          color: white;
        }

        .trading-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid rgba(0, 212, 255, 0.2);
        }

        .header-left h2 {
          font-size: 1.5rem;
          margin-bottom: 0.5rem;
          color: #00d4ff;
        }

        .market-status {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .status-indicator {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          border-radius: 20px;
          font-size: 0.875rem;
          font-weight: 600;
        }

        .status-indicator.open {
          background: rgba(0, 255, 136, 0.2);
          color: #00ff88;
          border: 1px solid #00ff88;
        }

        .status-indicator.closed {
          background: rgba(255, 107, 53, 0.2);
          color: #ff6b35;
          border: 1px solid #ff6b35;
        }

        .status-indicator.premarket,
        .status-indicator.afterhours {
          background: rgba(255, 215, 0, 0.2);
          color: #ffd700;
          border: 1px solid #ffd700;
        }

        .pulse-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: currentColor;
          animation: pulse 1.5s infinite;
        }

        .header-controls {
          display: flex;
          gap: 1rem;
        }

        .monitor-btn, .refresh-btn {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 8px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .monitor-btn.active {
          background: linear-gradient(135deg, #ff6b35, #ff8c42);
          color: white;
        }

        .monitor-btn.inactive {
          background: linear-gradient(135deg, #00ff88, #42a5f5);
          color: white;
        }

        .refresh-btn {
          background: rgba(0, 212, 255, 0.2);
          color: #00d4ff;
          border: 1px solid #00d4ff;
        }

        .refresh-btn:hover {
          background: rgba(0, 212, 255, 0.3);
        }

        .refresh-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .spinning {
          animation: spin 1s linear infinite;
        }

        .performance-metrics {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1rem;
          margin-bottom: 2rem;
        }

        .metric-card {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(0, 212, 255, 0.2);
          border-radius: 12px;
          color: #00d4ff;
        }

        .metric-label {
          display: block;
          font-size: 0.75rem;
          color: #a0a0c0;
          margin-bottom: 0.25rem;
        }

        .metric-value {
          font-size: 1.25rem;
          font-weight: 700;
          font-family: 'Courier New', monospace;
        }

        .trading-grid {
          display: grid;
          grid-template-columns: 1fr 1fr 350px;
          gap: 2rem;
        }

        .signals-panel,
        .pathrag-panel,
        .notifications-panel {
          background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
          border: 1px solid rgba(0, 212, 255, 0.2);
          border-radius: 16px;
          padding: 1.5rem;
          backdrop-filter: blur(20px);
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid rgba(0, 212, 255, 0.2);
        }

        .panel-header h3 {
          font-size: 1.125rem;
          color: #00d4ff;
        }

        .signal-count,
        .analysis-score {
          font-size: 0.875rem;
          color: #a0a0c0;
        }

        .analysis-score .score {
          color: #00ff88;
          font-weight: 700;
        }

        .signals-list {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          max-height: 500px;
          overflow-y: auto;
        }

        .signal-card {
          padding: 1rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(0, 212, 255, 0.2);
          border-radius: 12px;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .signal-card:hover {
          border-color: #00d4ff;
          transform: translateY(-2px);
        }

        .signal-card.selected {
          border-color: #00ff88;
          background: rgba(0, 255, 136, 0.1);
        }

        .signal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.75rem;
        }

        .symbol {
          font-size: 1rem;
          font-weight: 700;
          color: white;
        }

        .action-badge {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          padding: 0.25rem 0.75rem;
          border-radius: 12px;
          font-size: 0.75rem;
          font-weight: 600;
          margin-top: 0.25rem;
        }

        .signal-metrics {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          align-items: flex-end;
        }

        .probability,
        .heat-score {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
        }

        .probability .label,
        .heat-score .label {
          font-size: 0.6rem;
          color: #a0a0c0;
          text-transform: uppercase;
        }

        .probability .value,
        .heat-score .value {
          font-size: 0.9rem;
          font-weight: 700;
          font-family: 'Courier New', monospace;
        }

        .signal-details {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .strike-expiry {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.75rem;
          color: #a0a0c0;
        }

        .strength {
          font-size: 0.75rem;
          font-weight: 600;
          text-transform: uppercase;
        }

        .strength.text-purple-400 {
          color: #a855f7;
        }

        .strength.text-green-400 {
          color: #4ade80;
        }

        .strength.text-yellow-400 {
          color: #facc15;
        }

        .strength.text-gray-400 {
          color: #9ca3af;
        }

        .risk-reward {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.75rem;
          color: #a0a0c0;
        }

        .rr-ratio {
          font-weight: 600;
        }

        .pathrag-content {
          color: #e5e7eb;
        }

        .reasoning-text {
          padding: 1rem;
          background: rgba(0, 212, 255, 0.1);
          border-left: 3px solid #00d4ff;
          border-radius: 8px;
          margin-bottom: 1.5rem;
          font-size: 0.875rem;
          line-height: 1.6;
        }

        .technical-summary h4 {
          color: #00d4ff;
          margin-bottom: 1rem;
          font-size: 1rem;
        }

        .indicators-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 0.75rem;
        }

        .indicator {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
        }

        .indicator .label {
          font-size: 0.75rem;
          color: #a0a0c0;
        }

        .indicator .value {
          font-size: 0.75rem;
          font-weight: 600;
        }

        .indicator .value.bullish {
          color: #00ff88;
        }

        .indicator .value.bearish {
          color: #ff6b35;
        }

        .indicator .value.neutral {
          color: #a0a0c0;
        }

        .indicator .value.overbought {
          color: #ff6b35;
        }

        .indicator .value.oversold {
          color: #00ff88;
        }

        .indicator .value.high {
          color: #ffd700;
        }

        .notifications-list {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          max-height: 400px;
          overflow-y: auto;
        }

        .notification-item {
          padding: 1rem;
          border-radius: 12px;
          border-left: 4px solid;
        }

        .notification-item.high {
          background: rgba(168, 85, 247, 0.1);
          border-left-color: #a855f7;
        }

        .notification-item.medium {
          background: rgba(255, 215, 0, 0.1);
          border-left-color: #ffd700;
        }

        .notification-item.low {
          background: rgba(0, 212, 255, 0.1);
          border-left-color: #00d4ff;
        }

        .notification-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .symbol-action {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .notification-details {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .prob-heat {
          display: flex;
          gap: 0.75rem;
          font-size: 0.75rem;
          font-weight: 600;
          color: #a0a0c0;
        }

        .timestamp {
          font-size: 0.7rem;
          color: #6b7280;
        }

        .no-notifications {
          text-align: center;
          padding: 2rem;
          color: #6b7280;
        }

        .no-notifications svg {
          margin-bottom: 1rem;
          opacity: 0.5;
        }

        .alert-icon.active {
          color: #00ff88;
          animation: pulse 2s infinite;
        }

        @media (max-width: 1400px) {
          .trading-grid {
            grid-template-columns: 1fr;
            gap: 1.5rem;
          }
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default OptionsTrading;