import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import Header from './components/Header';
import RealTimeOptionsTrading from './components/RealTimeOptionsTrading';
import LiveDataStream from './components/LiveDataStream';
import AdvancedTradingDashboard from './components/AdvancedTradingDashboard';
import ProfessionalNeo4jGraph from './components/ProfessionalNeo4jGraph';
import EnhancedKnowledgeGraph from './components/EnhancedKnowledgeGraph';
import TempD3KnowledgeGraph from './components/TempD3KnowledgeGraph';
import ProfessionalOntologyGraph from './components/ProfessionalOntologyGraph';
import AdvancedOntologyGraph from './components/AdvancedOntologyGraph';
import PortfolioConstructionDashboard from './components/PortfolioConstructionDashboard';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('revolutionary');
  const [systemStatus, setSystemStatus] = useState('loading');
  const [apiUrl] = useState('http://localhost:8003');

  useEffect(() => {
    // Check if the revolutionary API is running
    checkApiStatus();
    const interval = setInterval(checkApiStatus, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/status`);
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'running') {
          setSystemStatus('connected');
        } else {
          setSystemStatus('error');
        }
      } else {
        setSystemStatus('error');
      }
    } catch (error) {
      setSystemStatus('disconnected');
    }
  };

  const getStatusMessage = () => {
    switch (systemStatus) {
      case 'loading':
        return '🔄 Initializing Revolutionary Trading System...';
      case 'connected':
        return '✅ Revolutionary AI Trading System Active';
      case 'disconnected':
        return '❌ API Server Disconnected - Please start the backend';
      case 'error':
        return '⚠️ API Server Error - Check backend logs';
      default:
        return '🔄 Loading...';
    }
  };

  const getStatusColor = () => {
    switch (systemStatus) {
      case 'connected':
        return '#00ff88';
      case 'disconnected':
      case 'error':
        return '#ff6b35';
      default:
        return '#ffa726';
    }
  };

  return (
    <div className="App">
      <Header />
      
      {/* System Status Bar */}
      <div style={{
        background: `linear-gradient(90deg, ${getStatusColor()}22, transparent)`,
        border: `1px solid ${getStatusColor()}33`,
        padding: '0.5rem 1rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '0.5rem'
      }}>
        <div style={{ 
          width: '8px', 
          height: '8px', 
          borderRadius: '50%', 
          backgroundColor: getStatusColor(),
          animation: systemStatus === 'connected' ? 'pulse 2s infinite' : 'none'
        }}></div>
        <span style={{ 
          color: getStatusColor(),
          fontSize: '0.9rem',
          fontWeight: '500'
        }}>
          {getStatusMessage()}
        </span>
      </div>
      
      <nav style={{ 
        display: 'flex', 
        gap: '20px', 
        padding: '20px', 
        backgroundColor: '#1a1a3a',
        justifyContent: 'center',
        flexWrap: 'wrap'
      }}>
        <button 
          onClick={() => setCurrentView('revolutionary')}
          style={{
            padding: '10px 20px',
            backgroundColor: currentView === 'revolutionary' ? '#ff6b35' : '#333',
            color: currentView === 'revolutionary' ? '#fff' : '#fff',
            border: currentView === 'revolutionary' ? '2px solid #ff8c42' : 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '14px',
            transition: 'all 0.3s ease'
          }}
        >
          🚀 Revolutionary Dashboard
        </button>
        <button 
          onClick={() => setCurrentView('dashboard')}
          style={{
            padding: '10px 20px',
            backgroundColor: currentView === 'dashboard' ? '#00d4ff' : '#333',
            color: currentView === 'dashboard' ? '#000' : '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          📊 Market Dashboard
        </button>
        <button 
          onClick={() => setCurrentView('live-signals')}
          style={{
            padding: '10px 20px',
            backgroundColor: currentView === 'live-signals' ? '#00d4ff' : '#333',
            color: currentView === 'live-signals' ? '#000' : '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          📈 Live Options Signals
        </button>
        <button 
          onClick={() => setCurrentView('live-stream')}
          style={{
            padding: '10px 20px',
            backgroundColor: currentView === 'live-stream' ? '#00d4ff' : '#333',
            color: currentView === 'live-stream' ? '#000' : '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          🔥 Live Data Stream
        </button>
        <button 
          onClick={() => setCurrentView('graph-viz')}
          style={{
            padding: '10px 20px',
            backgroundColor: currentView === 'graph-viz' ? '#00ff88' : '#333',
            color: currentView === 'graph-viz' ? '#000' : '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          🌐 Knowledge Graph
        </button>
        <button 
          onClick={() => setCurrentView('enhanced-graph')}
          style={{
            padding: '10px 20px',
            backgroundColor: currentView === 'enhanced-graph' ? '#6366F1' : '#333',
            color: currentView === 'enhanced-graph' ? '#fff' : '#fff',
            border: currentView === 'enhanced-graph' ? '2px solid #8B5CF6' : 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '14px',
            transition: 'all 0.3s ease'
          }}
        >
          🚀 Enhanced Graph
        </button>
        <button 
          onClick={() => setCurrentView('revolutionary-sigma')}
          style={{
            padding: '10px 20px',
            backgroundColor: currentView === 'revolutionary-sigma' ? '#ff0080' : '#333',
            color: '#fff',
            border: currentView === 'revolutionary-sigma' ? '2px solid #ff33aa' : 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '14px',
            transition: 'all 0.3s ease',
            boxShadow: currentView === 'revolutionary-sigma' ? '0 0 15px rgba(255, 0, 128, 0.5)' : 'none'
          }}
        >
          ⚡ Sigma.js Revolution
        </button>
        <button 
          onClick={() => setCurrentView('ontology-graph')}
          style={{
            padding: '10px 20px',
            backgroundColor: currentView === 'ontology-graph' ? '#FFD700' : '#333',
            color: currentView === 'ontology-graph' ? '#000' : '#fff',
            border: currentView === 'ontology-graph' ? '2px solid #FFA500' : 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '14px',
            transition: 'all 0.3s ease',
            boxShadow: currentView === 'ontology-graph' ? '0 0 20px rgba(255, 215, 0, 0.4)' : 'none'
          }}
        >
          🧠 Ontology Graph
        </button>
        <button 
          onClick={() => setCurrentView('advanced-ontology')}
          style={{
            padding: '10px 20px',
            backgroundColor: currentView === 'advanced-ontology' ? '#9966FF' : '#333',
            color: currentView === 'advanced-ontology' ? '#fff' : '#fff',
            border: currentView === 'advanced-ontology' ? '2px solid #BB88FF' : 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '14px',
            transition: 'all 0.3s ease',
            boxShadow: currentView === 'advanced-ontology' ? '0 0 20px rgba(153, 102, 255, 0.4)' : 'none'
          }}
        >
          🚀 Advanced KG
        </button>
        <button 
          onClick={() => setCurrentView('portfolio-construction')}
          style={{
            padding: '10px 20px',
            backgroundColor: currentView === 'portfolio-construction' ? '#FF6B35' : '#333',
            color: currentView === 'portfolio-construction' ? '#fff' : '#fff',
            border: currentView === 'portfolio-construction' ? '3px solid #FF8C42' : 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '14px',
            transition: 'all 0.3s ease',
            boxShadow: currentView === 'portfolio-construction' ? '0 0 25px rgba(255, 107, 53, 0.6)' : 'none',
            transform: currentView === 'portfolio-construction' ? 'scale(1.05)' : 'scale(1)'
          }}
        >
          🤖 Portfolio AI
        </button>
      </nav>
      
      {currentView === 'revolutionary' && (
        systemStatus === 'connected' ? (
          <AdvancedTradingDashboard apiUrl={apiUrl} />
        ) : (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '60vh',
            gap: '2rem',
            padding: '2rem',
            background: 'linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%)'
          }}>
            <div style={{
              fontSize: '4rem',
              marginBottom: '1rem'
            }}>
              {systemStatus === 'loading' ? '🔄' : systemStatus === 'disconnected' ? '📡' : '⚠️'}
            </div>
            
            <div style={{
              textAlign: 'center',
              maxWidth: '600px'
            }}>
              <h2 style={{ 
                color: '#ffffff',
                marginBottom: '1rem',
                fontSize: '1.5rem'
              }}>
                {getStatusMessage()}
              </h2>
              
              {systemStatus === 'disconnected' && (
                <div style={{
                  background: 'rgba(255, 107, 53, 0.1)',
                  border: '1px solid rgba(255, 107, 53, 0.3)',
                  borderRadius: '8px',
                  padding: '1.5rem',
                  marginTop: '1rem'
                }}>
                  <h3 style={{ color: '#ff6b35', marginBottom: '1rem' }}>
                    🚀 Start the Revolutionary Backend
                  </h3>
                  <div style={{ 
                    textAlign: 'left',
                    background: 'rgba(0, 0, 0, 0.3)',
                    padding: '1rem',
                    borderRadius: '4px',
                    fontFamily: 'monospace',
                    fontSize: '0.9rem',
                    color: '#00ff88'
                  }}>
                    <div>cd /Users/rajeshgupta/PycharmProjects/ragheat-poc/backend</div>
                    <div>PYTHONPATH=. python api/revolutionary_main.py</div>
                  </div>
                  <p style={{ 
                    color: '#cccccc',
                    marginTop: '1rem',
                    fontSize: '0.9rem'
                  }}>
                    This will start the revolutionary AI trading system on port 8001!
                  </p>
                </div>
              )}
              
              <button 
                onClick={checkApiStatus}
                style={{
                  marginTop: '1rem',
                  padding: '0.75rem 1.5rem',
                  background: 'linear-gradient(135deg, #ff6b35, #ff8c42)',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'white',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  fontSize: '1rem',
                  transition: 'transform 0.2s ease'
                }}
                onMouseOver={(e) => e.target.style.transform = 'scale(1.05)'}
                onMouseOut={(e) => e.target.style.transform = 'scale(1)'}
              >
                🔄 Retry Connection
              </button>
            </div>
          </div>
        )
      )}
      {currentView === 'dashboard' && <Dashboard />}
      {currentView === 'live-signals' && <RealTimeOptionsTrading />}
      {currentView === 'live-stream' && <LiveDataStream />}
      {currentView === 'graph-viz' && <ProfessionalNeo4jGraph apiUrl={apiUrl} />}
      {currentView === 'enhanced-graph' && <EnhancedKnowledgeGraph apiUrl={apiUrl} />}
      {currentView === 'revolutionary-sigma' && <TempD3KnowledgeGraph apiUrl={apiUrl} />}
      {currentView === 'ontology-graph' && <ProfessionalOntologyGraph apiUrl="http://localhost:8001" />}
      {currentView === 'advanced-ontology' && <AdvancedOntologyGraph apiUrl="http://localhost:8001" />}
      {currentView === 'portfolio-construction' && <PortfolioConstructionDashboard apiUrl="http://localhost:8001" />}
    </div>
  );
}

export default App;