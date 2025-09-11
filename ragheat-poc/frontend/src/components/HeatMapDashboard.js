import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const HeatMapDashboard = () => {
  const [heatData, setHeatData] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [selectedStock, setSelectedStock] = useState(null);
  const [loading, setLoading] = useState(true);
  const [ws, setWs] = useState(null);

  useEffect(() => {
    // Fetch initial data
    fetchHeatDistribution();
    fetchRecommendations();

    // Setup WebSocket connection
    const websocket = new WebSocket('ws://localhost:8000/ws/heat-updates');

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'heat_update') {
        setHeatData(data);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setWs(websocket);

    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, []);

  const fetchHeatDistribution = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/heat/distribution`);
      setHeatData(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching heat distribution:', error);
      setLoading(false);
    }
  };

  const fetchRecommendations = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/recommendations/top`);
      setRecommendations(response.data.recommendations);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    }
  };

  const analyzeStock = async (symbol) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/analyze/stock`, {
        symbol: symbol,
        include_heat_map: true
      });
      setSelectedStock(response.data);
    } catch (error) {
      console.error('Error analyzing stock:', error);
    }
  };

  if (loading) {
    return <div className="loading">Loading RAGHeat System...</div>;
  }

  return (
    <div className="dashboard">
      <div className="dashboard-grid">
        <div className="panel">
          <h2>Top Heated Sectors</h2>
          {heatData?.top_sectors && (
            <ul className="sector-list">
              {heatData.top_sectors.map((sector, idx) => (
                <li key={idx} className="sector-item">
                  <span className="sector-name">{sector.sector}</span>
                  <span className="heat-score">{(sector.heat * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="panel">
          <h2>Top Stock Recommendations</h2>
          {recommendations.length > 0 && (
            <ul className="stock-list">
              {recommendations.slice(0, 10).map((stock, idx) => (
                <li 
                  key={idx} 
                  className="stock-item"
                  onClick={() => analyzeStock(stock.symbol)}
                >
                  <span className="stock-symbol">{stock.symbol}</span>
                  <span className="stock-sector">{stock.sector}</span>
                  <span className="heat-score">{(stock.heat_score * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        {selectedStock && (
          <div className="panel analysis-panel">
            <h2>Stock Analysis: {selectedStock.symbol}</h2>
            <div className="analysis-content">
              <div className="metric">
                <label>Recommendation:</label>
                <span className={`recommendation ${selectedStock.recommendation?.action}`}>
                  {selectedStock.recommendation?.action}
                </span>
              </div>
              <div className="metric">
                <label>Heat Score:</label>
                <span>{(selectedStock.heat_score * 100).toFixed(1)}%</span>
              </div>
              <div className="metric">
                <label>Confidence:</label>
                <span>{selectedStock.recommendation?.confidence}</span>
              </div>
              <div className="explanation">
                <label>Analysis:</label>
                <p>{selectedStock.recommendation?.explanation}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HeatMapDashboard;