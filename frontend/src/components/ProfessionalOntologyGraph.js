/**
 * Professional Ontology-Driven Knowledge Graph
 * Multi-level hierarchical visualization with semantic structure
 * Based on professional KG visualization best practices
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import './ProfessionalOntologyGraph.css';

const ProfessionalOntologyGraph = ({ apiUrl = 'http://localhost:8002' }) => {
  const svgRef = useRef();
  const containerRef = useRef();
  
  // State management
  const [currentLevel, setCurrentLevel] = useState(3);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [marketOverview, setMarketOverview] = useState(null);
  const [error, setError] = useState(null);
  
  // Professional visualization levels
  const visualizationLevels = {
    1: { name: "Market Structure", description: "Top-level market organization", icon: "🏛️" },
    2: { name: "Sector Classification", description: "Economic sectors and industries", icon: "🏢" },
    3: { name: "Financial Instruments", description: "Stocks, bonds, derivatives", icon: "📈" },
    4: { name: "Corporate Relations", description: "Correlations and relationships", icon: "🔗" },
    5: { name: "Trading Signals", description: "BUY/SELL/HOLD recommendations", icon: "📊" },
    6: { name: "Technical Indicators", description: "RSI, MA, Bollinger Bands", icon: "📉" },
    7: { name: "Heat Propagation", description: "Thermal dynamics network", icon: "🔥" }
  };
  
  // Color schemes for different entity types
  const colorSchemes = {
    sector: "#00D4FF",
    stock: "#00FF88", 
    signal: {
      buy: "#00FF88",
      sell: "#FF6B35", 
      hold: "#FFA726"
    },
    correlation: {
      strong: "#FF0066",
      moderate: "#FF6600",
      weak: "#CCCCCC"
    },
    heat: {
      hot: "#FF0033",
      warm: "#FF6600",
      cool: "#0099FF",
      cold: "#0066CC"
    }
  };
  
  // Fetch hierarchical graph data
  const fetchGraphData = useCallback(async (level) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${apiUrl}/api/ontology/graph/${level}?max_nodes=50`);
      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setGraphData(data);
      
      // Also fetch market overview
      const overviewResponse = await fetch(`${apiUrl}/api/ontology/market-overview`);
      const overview = await overviewResponse.json();
      setMarketOverview(overview);
      
    } catch (error) {
      console.error('Error fetching graph data:', error);
      setError(error.message);
      // Fallback to synthetic data for demo
      setGraphData(generateFallbackData(level));
    } finally {
      setIsLoading(false);
    }
  }, [apiUrl]);
  
  // Generate fallback data for demo purposes
  const generateFallbackData = (level) => {
    if (level === 2) {
      // Sector level
      return {
        nodes: [
          { id: "Technology", name: "Technology", type: "sector", level: 2, heat_score: 75, stock_count: 5, color: "#00D4FF", size: 30 },
          { id: "Healthcare", name: "Healthcare", type: "sector", level: 2, heat_score: 65, stock_count: 3, color: "#FF6B35", size: 25 },
          { id: "Finance", name: "Finance", type: "sector", level: 2, heat_score: 45, stock_count: 4, color: "#00FF88", size: 28 },
          { id: "Energy", name: "Energy", type: "sector", level: 2, heat_score: 35, stock_count: 2, color: "#FFA726", size: 22 }
        ],
        links: [],
        level: 2,
        level_name: "Sector Classification",
        description: "Economic sectors with aggregated metrics"
      };
    } else if (level === 3) {
      // Stock level
      return {
        nodes: [
          { id: "AAPL", name: "AAPL", type: "stock", level: 3, heat_score: 85, price: 175.50, signal_type: "BuySignal", color: "#00FF88", size: 20 },
          { id: "GOOGL", name: "GOOGL", type: "stock", level: 3, heat_score: 72, price: 142.80, signal_type: "HoldSignal", color: "#FFA726", size: 18 },
          { id: "MSFT", name: "MSFT", type: "stock", level: 3, heat_score: 68, price: 378.90, signal_type: "BuySignal", color: "#00FF88", size: 17 },
          { id: "TSLA", name: "TSLA", type: "stock", level: 3, heat_score: 92, price: 248.50, signal_type: "SellSignal", color: "#FF6B35", size: 22 }
        ],
        links: [
          { source: "AAPL", target: "GOOGL", type: "correlation", strength: "strong", correlation: 0.75, color: "#FF0066", width: 3 },
          { source: "GOOGL", target: "MSFT", type: "correlation", strength: "moderate", correlation: 0.55, color: "#FF6600", width: 2 }
        ],
        level: 3,
        level_name: "Financial Instruments",
        description: "Stocks with real-time signals"
      };
    } else if (level === 5) {
      // Signals level
      return {
        nodes: [
          { id: "AAPL", name: "AAPL", type: "stock", level: 3, heat_score: 85, color: "#666666", size: 15 },
          { id: "AAPL_signal", name: "BUY", type: "signal", level: 5, signal_type: "BuySignal", strength: 0.85, color: "#00FF88", size: 18 },
          { id: "TSLA", name: "TSLA", type: "stock", level: 3, heat_score: 92, color: "#666666", size: 15 },
          { id: "TSLA_signal", name: "SELL", type: "signal", level: 5, signal_type: "SellSignal", strength: 0.78, color: "#FF6B35", size: 17 }
        ],
        links: [
          { source: "AAPL", target: "AAPL_signal", type: "has_signal", color: "#00FF88", width: 2 },
          { source: "TSLA", target: "TSLA_signal", type: "has_signal", color: "#FF6B35", width: 2 }
        ],
        level: 5,
        level_name: "Trading Signals",
        description: "BUY/SELL/HOLD recommendations"
      };
    }
    
    return { nodes: [], links: [], level: level, level_name: "No Data", description: "No data available" };
  };
  
  // Initialize and render the graph
  useEffect(() => {
    if (!svgRef.current || !graphData.nodes.length) return;
    
    renderProfessionalGraph();
  }, [graphData, currentLevel]);
  
  // Load initial data
  useEffect(() => {
    fetchGraphData(currentLevel);
  }, [currentLevel, fetchGraphData]);
  
  const renderProfessionalGraph = () => {
    const svg = d3.select(svgRef.current);
    const width = containerRef.current?.clientWidth || 1200;
    const height = containerRef.current?.clientHeight || 800;
    
    svg.attr('width', width).attr('height', height);
    svg.selectAll('*').remove();
    
    // Create professional gradient definitions
    const defs = svg.append('defs');
    
    // Radial gradient for nodes
    const nodeGradient = defs.append('radialGradient')
      .attr('id', 'nodeGradient')
      .attr('cx', '30%')
      .attr('cy', '30%');
      
    nodeGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#FFFFFF')
      .attr('stop-opacity', 0.8);
      
    nodeGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#000000')
      .attr('stop-opacity', 0.1);
    
    // Professional drop shadow filter
    const filter = defs.append('filter')
      .attr('id', 'dropshadow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');
      
    filter.append('feDropShadow')
      .attr('dx', 2)
      .attr('dy', 2)
      .attr('stdDeviation', 3)
      .attr('flood-color', '#000000')
      .attr('flood-opacity', 0.3);
    
    // Create simulation with professional physics
    const simulation = d3.forceSimulation(graphData.nodes)
      .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(d => {
        // Dynamic link distance based on relationship type
        if (d.type === 'correlation') return 150;
        if (d.type === 'has_signal') return 100;
        if (d.type === 'heat_flow') return 200;
        return 120;
      }))
      .force('charge', d3.forceManyBody().strength(d => {
        // Node repulsion based on size and level
        const baseStrength = -300;
        const levelMultiplier = Math.max(0.5, (8 - d.level) / 4);
        const sizeMultiplier = (d.size || 10) / 15;
        return baseStrength * levelMultiplier * sizeMultiplier;
      }))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => (d.size || 10) + 5))
      .force('x', d3.forceX(width / 2).strength(0.1))
      .force('y', d3.forceY(height / 2).strength(0.1));
    
    // Create link elements with professional styling
    const linkGroup = svg.append('g').attr('class', 'links');
    const link = linkGroup.selectAll('line')
      .data(graphData.links)
      .enter().append('line')
      .attr('stroke', d => d.color || '#666666')
      .attr('stroke-width', d => d.width || 1)
      .attr('stroke-opacity', d => {
        if (d.type === 'correlation') return 0.6 + (Math.abs(d.correlation || 0) * 0.4);
        return 0.6;
      })
      .attr('stroke-dasharray', d => {
        if (d.type === 'heat_flow') return '5,5';
        return null;
      });
    
    // Add link labels for important connections
    const linkLabels = linkGroup.selectAll('text')
      .data(graphData.links.filter(d => d.correlation && Math.abs(d.correlation) > 0.6))
      .enter().append('text')
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#FFFFFF')
      .attr('stroke', '#000000')
      .attr('stroke-width', 0.5)
      .text(d => d.correlation ? `${(d.correlation * 100).toFixed(0)}%` : '');
    
    // Create node groups with professional hierarchy
    const nodeGroup = svg.append('g').attr('class', 'nodes');
    const node = nodeGroup.selectAll('g')
      .data(graphData.nodes)
      .enter().append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));
    
    // Professional node circles with level-based styling
    node.append('circle')
      .attr('r', d => d.size || 10)
      .attr('fill', d => {
        if (d.type === 'signal') {
          return d.color;
        } else if (d.type === 'stock' && d.signal_type) {
          if (d.signal_type.includes('Buy')) return colorSchemes.signal.buy;
          if (d.signal_type.includes('Sell')) return colorSchemes.signal.sell;
          return colorSchemes.signal.hold;
        }
        return d.color || colorSchemes.sector;
      })
      .attr('stroke', '#FFFFFF')
      .attr('stroke-width', d => d.level <= 3 ? 3 : 2)
      .attr('filter', 'url(#dropshadow)')
      .style('fill-opacity', d => {
        if (d.type === 'sector') return 0.9;
        if (d.type === 'signal') return 0.8;
        return 0.85;
      });
    
    // Add inner circles for multi-level nodes
    node.filter(d => d.level <= 2)
      .append('circle')
      .attr('r', d => (d.size || 10) * 0.6)
      .attr('fill', 'url(#nodeGradient)')
      .attr('fill-opacity', 0.3);
    
    // Professional node labels with hierarchy
    node.append('text')
      .text(d => d.name)
      .attr('text-anchor', 'middle')
      .attr('dy', d => d.level <= 2 ? '0.35em' : '-1.2em')
      .attr('font-family', 'Arial, sans-serif')
      .attr('font-size', d => {
        if (d.level === 1) return '16px';
        if (d.level === 2) return '14px';
        return '12px';
      })
      .attr('font-weight', d => d.level <= 2 ? 'bold' : 'normal')
      .attr('fill', '#FFFFFF')
      .attr('stroke', '#000000')
      .attr('stroke-width', 0.8)
      .attr('paint-order', 'stroke');
    
    // Add level indicators
    node.filter(d => d.level)
      .append('text')
      .text(d => `L${d.level}`)
      .attr('text-anchor', 'middle')
      .attr('dy', d => (d.size || 10) + 15)
      .attr('font-size', '10px')
      .attr('fill', '#CCCCCC')
      .attr('opacity', 0.7);
    
    // Add heat score indicators for relevant nodes
    node.filter(d => d.heat_score !== undefined)
      .append('text')
      .text(d => `${d.heat_score.toFixed(1)}°`)
      .attr('text-anchor', 'middle')
      .attr('dy', '2.8em')
      .attr('font-size', '10px')
      .attr('fill', d => d.heat_score > 70 ? '#FF3333' : d.heat_score > 40 ? '#FF9900' : '#00CCFF')
      .attr('font-weight', 'bold');
    
    // Professional interaction handlers
    node.on('click', function(event, d) {
      setSelectedNode(d);
      
      // Highlight connected nodes
      const connectedNodeIds = new Set();
      graphData.links.forEach(link => {
        if (link.source.id === d.id) connectedNodeIds.add(link.target.id);
        if (link.target.id === d.id) connectedNodeIds.add(link.source.id);
      });
      
      // Update visual emphasis
      node.selectAll('circle')
        .attr('stroke-width', nodeData => {
          if (nodeData.id === d.id) return 4;
          if (connectedNodeIds.has(nodeData.id)) return 3;
          return nodeData.level <= 3 ? 3 : 2;
        })
        .attr('stroke', nodeData => {
          if (nodeData.id === d.id) return '#FFD700';
          if (connectedNodeIds.has(nodeData.id)) return '#FF6600';
          return '#FFFFFF';
        });
        
      link.attr('stroke-opacity', linkData => {
        if (linkData.source.id === d.id || linkData.target.id === d.id) return 1.0;
        return 0.2;
      });
    })
    .on('mouseenter', function(event, d) {
      d3.select(this).select('circle')
        .transition()
        .duration(200)
        .attr('r', (d.size || 10) * 1.2);
    })
    .on('mouseleave', function(event, d) {
      d3.select(this).select('circle')
        .transition()
        .duration(200)
        .attr('r', d.size || 10);
    });
    
    // Simulation tick function with smooth animation
    simulation.on('tick', () => {
      // Update links
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
        
      // Update link labels
      linkLabels
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2);
      
      // Update nodes
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
    
    // Drag functions
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  };
  
  return (
    <div className="professional-ontology-graph">
      {/* Header with Level Controls */}
      <div className="graph-header">
        <div className="level-info">
          <h2>🧠 Professional Knowledge Graph</h2>
          <p>Ontology-Driven Hierarchical Visualization</p>
        </div>
        
        <div className="level-controls">
          <label>Visualization Level:</label>
          <select 
            value={currentLevel} 
            onChange={(e) => setCurrentLevel(parseInt(e.target.value))}
            className="level-select"
          >
            {Object.entries(visualizationLevels).map(([level, info]) => (
              <option key={level} value={level}>
                {info.icon} Level {level}: {info.name}
              </option>
            ))}
          </select>
        </div>
      </div>
      
      {/* Current Level Description */}
      {graphData.level && (
        <div className="level-description">
          <div className="level-badge">
            {visualizationLevels[graphData.level]?.icon} Level {graphData.level}
          </div>
          <h3>{graphData.level_name}</h3>
          <p>{graphData.description}</p>
        </div>
      )}
      
      {/* Market Overview Stats */}
      {marketOverview && (
        <div className="market-stats">
          <div className="stat-item">
            <span className="stat-label">Stocks:</span>
            <span className="stat-value">{marketOverview.total_stocks || 0}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Signals:</span>
            <span className="stat-value">{marketOverview.total_signals || 0}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Sentiment:</span>
            <span className={`stat-value sentiment-${(marketOverview.market_sentiment || 'neutral').toLowerCase()}`}>
              {marketOverview.market_sentiment || 'Neutral'}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Avg Heat:</span>
            <span className="stat-value">{(marketOverview.average_heat || 0).toFixed(1)}°</span>
          </div>
        </div>
      )}
      
      {/* Error Display */}
      {error && (
        <div className="error-message">
          ⚠️ {error} (Using demo data)
        </div>
      )}
      
      {/* Loading Indicator */}
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Loading Level {currentLevel} Data...</p>
        </div>
      )}
      
      {/* Main Graph Container */}
      <div ref={containerRef} className="graph-container">
        <svg ref={svgRef}></svg>
      </div>
      
      {/* Node Information Panel */}
      {selectedNode && (
        <div className="node-info-panel">
          <div className="panel-header">
            <h4>{selectedNode.name}</h4>
            <button 
              onClick={() => setSelectedNode(null)}
              className="close-button"
            >
              ×
            </button>
          </div>
          
          <div className="node-details">
            <div className="detail-row">
              <span>Type:</span>
              <span className="detail-value">{selectedNode.type}</span>
            </div>
            <div className="detail-row">
              <span>Level:</span>
              <span className="detail-value">L{selectedNode.level}</span>
            </div>
            
            {selectedNode.heat_score !== undefined && (
              <div className="detail-row">
                <span>Heat Score:</span>
                <span className="detail-value heat-score">{selectedNode.heat_score.toFixed(1)}°</span>
              </div>
            )}
            
            {selectedNode.price && (
              <div className="detail-row">
                <span>Price:</span>
                <span className="detail-value">${selectedNode.price.toFixed(2)}</span>
              </div>
            )}
            
            {selectedNode.signal_type && (
              <div className="detail-row">
                <span>Signal:</span>
                <span className={`detail-value signal-${selectedNode.signal_type.toLowerCase()}`}>
                  {selectedNode.signal_type.replace('Signal', '')}
                </span>
              </div>
            )}
            
            {selectedNode.strength !== undefined && (
              <div className="detail-row">
                <span>Strength:</span>
                <span className="detail-value">{(selectedNode.strength * 100).toFixed(0)}%</span>
              </div>
            )}
            
            {selectedNode.stock_count && (
              <div className="detail-row">
                <span>Stocks:</span>
                <span className="detail-value">{selectedNode.stock_count}</span>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Legend */}
      <div className="graph-legend">
        <h4>Legend</h4>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: colorSchemes.sector }}></div>
            <span>Sectors</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: colorSchemes.signal.buy }}></div>
            <span>Buy Signal</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: colorSchemes.signal.sell }}></div>
            <span>Sell Signal</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: colorSchemes.signal.hold }}></div>
            <span>Hold Signal</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfessionalOntologyGraph;