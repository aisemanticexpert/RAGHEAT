import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ResponsiveCirclePacking } from '@nivo/circle-packing';
import { TrendingUp, TrendingDown, Activity, Zap, Target } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8002';

const MultiSectorBubbleChart = () => {
  const [bubbleData, setBubbleData] = useState([]);
  const [sectorSummaries, setSectorSummaries] = useState([]);
  const [topStocks, setTopStocks] = useState({});
  const [selectedSector, setSelectedSector] = useState(null);
  const [teslaData, setTeslaData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  const fetchData = async () => {
    try {
      setIsLoading(true);
      
      const [bubbleResponse, summariesResponse, topStocksResponse, teslaResponse] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/sectors/bubble-chart`).catch(() => ({ data: [] })),
        axios.get(`${API_BASE_URL}/api/sectors/summaries`).catch(() => ({ data: [] })),
        axios.get(`${API_BASE_URL}/api/sectors/top-stocks`).catch(() => ({ data: {} })),
        axios.get(`${API_BASE_URL}/api/sectors/tesla-analysis`).catch(() => ({ data: null }))
      ]);

      setBubbleData(bubbleResponse.data || []);
      setSectorSummaries(summariesResponse.data || []);
      setTopStocks(topStocksResponse.data || {});
      setTeslaData(teslaResponse.data);
      setLastUpdate(new Date());
      
    } catch (error) {
      console.error('Error fetching multi-sector data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Transform data for bubble chart
  const transformedBubbleData = {
    name: "NASDAQ Sectors",
    color: "hsl(0, 0%, 0%)",
    children: sectorSummaries.map(sector => ({
      name: sector.sector_name,
      color: sector.color,
      children: bubbleData
        .filter(stock => stock.sector === sector.sector)
        .slice(0, 8) // Top 8 stocks per sector for visualization
        .map(stock => ({
          name: stock.symbol || 'N/A',
          value: Math.abs(stock.change_percent || 0) * 10 + 20, // Base size
          color: (stock.change_percent || 0) > 0 ? '#00ff88' : '#ff3366',
          data: stock
        }))
    })).filter(sector => sector.children.length > 0)
  };

  if (isLoading) {
    return (
      <div className="multi-sector-loading">
        <motion.div 
          className="loading-spinner"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        >
          <Zap size={48} />
        </motion.div>
        <h3>Loading Multi-Sector Analysis...</h3>
        <p>Analyzing 150+ NASDAQ stocks across 10 sectors</p>
      </div>
    );
  }

  return (
    <div className="multi-sector-container">
      {/* Header */}
      <div className="sector-header">
        <h2>Multi-Sector NASDAQ Analysis</h2>
        <div className="sector-stats">
          <div className="stat">
            <Activity size={16} />
            <span>{bubbleData.length} Stocks</span>
          </div>
          <div className="stat">
            <Target size={16} />
            <span>{sectorSummaries.length} Sectors</span>
          </div>
          <div className="stat">
            <Zap size={16} />
            <span>Updated: {lastUpdate.toLocaleTimeString()}</span>
          </div>
        </div>
      </div>

      {/* Tesla Highlight */}
      {teslaData && (
        <motion.div 
          className="tesla-highlight"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="tesla-icon">⚡</div>
          <div className="tesla-info">
            <h3>TESLA (TSLA)</h3>
            <div className="tesla-metrics">
              <span className="price">${(teslaData.price || 0).toFixed(2)}</span>
              <span className={`change ${(teslaData.change_percent || 0) > 0 ? 'positive' : 'negative'}`}>
                {(teslaData.change_percent || 0) > 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                {(teslaData.change_percent || 0).toFixed(2)}%
              </span>
              <span className="heat">Heat: {((teslaData.heat_score || 0) * 100).toFixed(0)}%</span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Sector Overview */}
      <div className="sector-overview">
        <h3>Sector Performance</h3>
        <div className="sector-grid">
          {sectorSummaries.map((sector, index) => (
            <motion.div
              key={sector.sector}
              className={`sector-card ${selectedSector === sector.sector ? 'selected' : ''}`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => setSelectedSector(selectedSector === sector.sector ? null : sector.sector)}
            >
              <div className="sector-color" style={{ backgroundColor: sector.color }}></div>
              <div className="sector-info">
                <h4>{sector.sector_name}</h4>
                <div className="sector-metrics">
                  <span className={`avg-change ${(sector.avg_change || 0) > 0 ? 'positive' : 'negative'}`}>
                    {(sector.avg_change || 0).toFixed(2)}%
                  </span>
                  <span className="stock-count">{sector.total_stocks || 0} stocks</span>
                </div>
                <div className="top-performers">
                  {(sector.top_performers || []).slice(0, 3).join(', ')}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Bubble Chart */}
      <div className="bubble-chart-container">
        <h3>Interactive Sector Bubble Chart</h3>
        <div className="chart-wrapper">
          {transformedBubbleData.children.length > 0 ? (
            <ResponsiveCirclePacking
              data={transformedBubbleData}
              margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
              identity="name"
              value="value"
              colors={{ scheme: 'category10' }}
              padding={6}
              labelTextColor={{
                from: 'color',
                modifiers: [['darker', 0.8]]
              }}
              borderWidth={2}
              borderColor={{
                from: 'color',
                modifiers: [['darker', 0.3]]
              }}
              animate={true}
              motionStiffness={90}
              motionDamping={12}
              tooltip={({ node }) => (
                <div className="bubble-tooltip">
                  <strong>{node.id}</strong>
                  {node.data.data && (
                    <>
                      <div>Price: ${node.data.data.price?.toFixed(2)}</div>
                      <div>Change: {node.data.data.change_percent?.toFixed(2)}%</div>
                      <div>Volume: {node.data.data.volume?.toLocaleString()}</div>
                      <div>Heat Score: {(node.data.data.heat_score * 100)?.toFixed(0)}%</div>
                    </>
                  )}
                </div>
              )}
            />
          ) : (
            <div className="no-data">Loading bubble chart data...</div>
          )}
        </div>
      </div>

      {/* Top Stocks by Sector */}
      {selectedSector && topStocks[selectedSector] && (
        <motion.div 
          className="top-stocks-section"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
        >
          <h3>Top Stocks in {sectorSummaries.find(s => s.sector === selectedSector)?.sector_name}</h3>
          <div className="top-stocks-grid">
            {topStocks[selectedSector].map((stock, index) => (
              <div key={stock.symbol} className="stock-card">
                <div className="stock-header">
                  <span className="symbol">{stock.symbol}</span>
                  <span className={`change ${(stock.change_percent || 0) > 0 ? 'positive' : 'negative'}`}>
                    {(stock.change_percent || 0) > 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                    {(stock.change_percent || 0).toFixed(2)}%
                  </span>
                </div>
                <div className="stock-price">${(stock.price || 0).toFixed(2)}</div>
                <div className="stock-metrics">
                  <span>Vol: {((stock.volume || 0) / 1000000).toFixed(1)}M</span>
                  <span>Heat: {((stock.heat_score || 0) * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Performance Legend */}
      <div className="performance-legend">
        <h4>Chart Legend</h4>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-color positive"></div>
            <span>Gainers</span>
          </div>
          <div className="legend-item">
            <div className="legend-color negative"></div>
            <span>Losers</span>
          </div>
          <div className="legend-item">
            <div className="legend-bubble"></div>
            <span>Bubble Size = Heat Score</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MultiSectorBubbleChart;