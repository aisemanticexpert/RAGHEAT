import React, { useState, useEffect } from 'react';
import './LiveSignalsDashboard.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8002';

const SimpleLiveSignals = () => {
  const [signals, setSignals] = useState([]);
  const [heatAnalysis, setHeatAnalysis] = useState({});
  const [loading, setLoading] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState('connecting');

  const fetchData = async () => {
    try {
      setLoading(true);
      setConnectionStatus('connecting');

      // Fetch signals
      const signalsResponse = await fetch(`${API_BASE_URL}/api/live-signals/current?min_priority=5&max_results=10`);
      if (signalsResponse.ok) {
        const signalsData = await signalsResponse.json();
        setSignals(signalsData);
        setConnectionStatus('connected');
      }

      // Fetch heat analysis
      const heatResponse = await fetch(`${API_BASE_URL}/api/live-signals/heat-analysis`);
      if (heatResponse.ok) {
        const heatData = await heatResponse.json();
        setHeatAnalysis(heatData);
      }

      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setConnectionStatus('error');
      setLoading(false);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchData();

    // Set up polling every 10 seconds
    const interval = setInterval(fetchData, 10000);

    return () => clearInterval(interval);
  }, []);

  const getSignalIcon = (signalType) => {
    switch (signalType) {
      case 'BULLISH_CALL': return '📈';
      case 'BEARISH_PUT': return '📉';
      case 'STRADDLE': return '🎯';
      case 'IRON_CONDOR': return '🦅';
      default: return '💹';
    }
  };

  const getStrengthColor = (strength) => {
    switch (strength) {
      case 'ULTRA_STRONG': return '#00ff88';
      case 'STRONG': return '#00d4ff';
      case 'MODERATE': return '#ffaa00';
      case 'WEAK': return '#ff6b35';
      default: return '#a0a0c0';
    }
  };

  const getPriorityColor = (priority) => {
    if (priority >= 9) return '#ff0066';
    if (priority >= 7) return '#ff6b35';
    if (priority >= 5) return '#ffaa00';
    return '#88ff44';
  };

  return (
    <div className="live-signals-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-left">
          <h2>🔥 Live Options Signals</h2>
          <div className="connection-status">
            <div className={`status ${connectionStatus === 'connected' ? 'connected' : 'disconnected'}`}>
              {connectionStatus === 'connected' ? '🟢' : '🔴'} {connectionStatus}
            </div>
          </div>
        </div>
      </div>

      {/* Heat Analysis Panel */}
      {heatAnalysis.market_heat_status && (
        <div className="heat-analysis-panel">
          <h3>🌡️ Market Heat Analysis</h3>
          <div className="heat-metrics">
            <div className="heat-metric">
              <span className="label">Market Status:</span>
              <span className={`value heat-${heatAnalysis.market_heat_status.toLowerCase()}`}>
                {heatAnalysis.market_heat_status}
              </span>
            </div>
            <div className="heat-metric">
              <span className="label">Avg Heat:</span>
              <span className="value">{(heatAnalysis.average_heat_score * 100).toFixed(1)}%</span>
            </div>
            <div className="heat-metric">
              <span className="label">Ultra Strong:</span>
              <span className="value">{heatAnalysis.ultra_strong_signals}</span>
            </div>
            <div className="heat-metric">
              <span className="label">Total Signals:</span>
              <span className="value">{heatAnalysis.total_signals}</span>
            </div>
          </div>
        </div>
      )}

      {/* Signals Grid */}
      <div className="signals-grid">
        {loading ? (
          <div className="empty-state">
            <h3>Loading Signals...</h3>
            <p>Fetching latest options opportunities...</p>
          </div>
        ) : signals.length === 0 ? (
          <div className="empty-state">
            <h3>No Active Signals</h3>
            <p>Waiting for high-probability options opportunities...</p>
          </div>
        ) : (
          signals.map((signal) => (
            <div key={signal.signal_id} className="signal-card">
              {/* Signal Header */}
              <div className="signal-header">
                <div className="signal-title">
                  <span className="signal-icon">{getSignalIcon(signal.signal_type)}</span>
                  <div className="signal-info">
                    <h3>{signal.symbol}</h3>
                    <span className="sector">{signal.sector}</span>
                  </div>
                </div>
                <div 
                  className="priority-badge"
                  style={{ backgroundColor: getPriorityColor(signal.priority) }}
                >
                  {signal.priority}
                </div>
              </div>

              {/* Signal Metrics */}
              <div className="signal-metrics">
                <div className="metric">
                  <span>${(signal.entry_price_low || 0).toFixed(2)}</span>
                  <small>Entry Low</small>
                </div>
                <div className="metric">
                  <span>${(signal.target_price || 0).toFixed(2)}</span>
                  <small>Target</small>
                </div>
                <div className="metric">
                  <span>{((signal.win_probability || 0) * 100).toFixed(0)}%</span>
                  <small>Win Prob</small>
                </div>
                <div className="metric">
                  <span>{((signal.heat_score || 0) * 100).toFixed(0)}%</span>
                  <small>Heat</small>
                </div>
              </div>

              {/* Price Targets */}
              <div className="price-targets">
                <div className="price-row">
                  <span className="label">Entry Range:</span>
                  <span className="value">
                    ${(signal.entry_price_low || 0).toFixed(2)} - ${(signal.entry_price_high || 0).toFixed(2)}
                  </span>
                </div>
                <div className="price-row">
                  <span className="label">Target:</span>
                  <span className="value target">${(signal.target_price || 0).toFixed(2)}</span>
                </div>
                <div className="price-row">
                  <span className="label">Stop Loss:</span>
                  <span className="value stop">${(signal.stop_loss || 0).toFixed(2)}</span>
                </div>
              </div>

              {/* Signal Details */}
              <div className="signal-details">
                <span 
                  className="strategy-badge"
                  style={{ color: getStrengthColor(signal.strength) }}
                >
                  {signal.strategy.toUpperCase()}
                </span>
                <span 
                  className="strength-badge"
                  style={{ backgroundColor: getStrengthColor(signal.strength) }}
                >
                  {signal.strength}
                </span>
              </div>

              {/* Entry Signals */}
              <div className="entry-signals">
                <small>Entry Signals</small>
                <ul>
                  {signal.entry_signals.slice(0, 3).map((entrySignal, idx) => (
                    <li key={idx}>{entrySignal}</li>
                  ))}
                </ul>
              </div>

              {/* Risk Factors */}
              <div className="risk-factors">
                <small>Risk Factors</small>
                <div className="risk-list">
                  {signal.risk_factors.slice(0, 3).map((risk, idx) => (
                    <span key={idx} className="risk-tag">{risk}</span>
                  ))}
                </div>
              </div>

              {/* Timestamp */}
              <div className="signal-timestamp">
                Generated: {new Date(signal.generated_at).toLocaleTimeString()}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer Stats */}
      <div className="dashboard-footer">
        <div className="stats">
          <div className="stat">
            <span className="label">Active Signals:</span>
            <span className="value">{signals.length}</span>
          </div>
          <div className="stat">
            <span className="label">Last Update:</span>
            <span className="value">{new Date().toLocaleTimeString()}</span>
          </div>
          <div className="stat">
            <span className="label">Status:</span>
            <span className="value" style={{ color: connectionStatus === 'connected' ? '#00ff88' : '#ff6b35' }}>
              {connectionStatus}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimpleLiveSignals;