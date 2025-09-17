import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import './Neo4jGraphVisualization.css';

const Neo4jGraphVisualization = ({ apiUrl = 'http://localhost:8001' }) => {
  const svgRef = useRef();
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [heatThreshold, setHeatThreshold] = useState(0.3);
  const [showStocks, setShowStocks] = useState(true);
  const [simulationRunning, setSimulationRunning] = useState(false);

  // Fetch data from backend
  const fetchGraphData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Try to fetch heat propagation data, use mock data if not available
      let heatData = null;
      try {
        const heatResponse = await fetch(`${apiUrl}/api/heat/propagation`);
        if (heatResponse.ok) {
          const result = await heatResponse.json();
          heatData = result.data;
        }
      } catch (error) {
        // Use mock heat data for demonstration
        console.log('Using mock heat data for demonstration');
        heatData = {
          nodes: [
            { id: 'AAPL', heat_level: 0.85 },
            { id: 'MSFT', heat_level: 0.72 },
            { id: 'NVDA', heat_level: 0.93 },
            { id: 'GOOGL', heat_level: 0.68 },
            { id: 'TSLA', heat_level: 0.89 },
            { id: 'AMZN', heat_level: 0.76 },
            { id: 'META', heat_level: 0.64 },
            { id: 'JPM', heat_level: 0.42 },
            { id: 'JNJ', heat_level: 0.38 },
            { id: 'XOM', heat_level: 0.55 }
          ]
        };
      }

      // Fetch sector analysis
      const sectorResponse = await fetch(`${apiUrl}/api/sectors/analysis`);
      let sectorData = null;
      if (sectorResponse.ok) {
        const result = await sectorResponse.json();
        sectorData = result.data;
      }

      // Generate graph structure
      const nodes = [];
      const links = [];

      // Add root node
      nodes.push({
        id: 'market',
        label: 'Market',
        type: 'root',
        heat: 0.5,
        radius: 30,
        x: 400,
        y: 300,
        fx: 400,
        fy: 300
      });

      // Add sector nodes
      if (sectorData) {
        sectorData.forEach((sector, index) => {
          const angle = (index / sectorData.length) * 2 * Math.PI;
          const radius = 150;
          const x = 400 + Math.cos(angle) * radius;
          const y = 300 + Math.sin(angle) * radius;

          // Calculate heat based on multiple factors
          const baseHeat = sector.heat_score || Math.abs(sector.performance || 0) / 10;
          const volatilityHeat = (sector.volatility || 0.2) * 2;
          const volumeHeat = Math.abs(sector.volume_change || 0) / 20;
          const combinedHeat = Math.min(0.9, Math.max(0.1, (baseHeat + volatilityHeat + volumeHeat) / 3));

          nodes.push({
            id: sector.key,
            label: sector.name,
            type: 'sector',
            heat: combinedHeat,
            performance: sector.performance || 0,
            radius: 20 + combinedHeat * 15,
            x: x,
            y: y,
            volume_change: sector.volume_change || 0,
            volatility: sector.volatility || 0.2
          });

          // Link to market
          links.push({
            source: 'market',
            target: sector.key,
            value: sector.heat_score || 0.5,
            type: 'sector-link'
          });

          // Add top stocks for this sector if heat is above threshold
          if (showStocks && (sector.heat_score || 0.5) > heatThreshold) {
            sector.top_stocks?.slice(0, 3).forEach((stock, stockIndex) => {
              const stockAngle = angle + ((stockIndex - 1) * 0.3);
              const stockRadius = 220;
              const stockX = 400 + Math.cos(stockAngle) * stockRadius;
              const stockY = 300 + Math.sin(stockAngle) * stockRadius;

              // Find heat data for this stock
              const stockHeat = heatData?.nodes?.find(n => n.id === stock)?.heat_level || Math.random() * 0.6 + 0.2;

              nodes.push({
                id: stock,
                label: stock,
                type: 'stock',
                sector: sector.key,
                heat: stockHeat,
                radius: 8 + stockHeat * 10,
                x: stockX,
                y: stockY,
                price: Math.random() * 200 + 50,
                change: (Math.random() - 0.5) * 10
              });

              // Link to sector
              links.push({
                source: sector.key,
                target: stock,
                value: stockHeat,
                type: 'stock-link'
              });
            });
          }
        });
      }

      setGraphData({ nodes, links });
      setLoading(false);
    } catch (err) {
      console.error('Error fetching graph data:', err);
      setError(err.message);
      setLoading(false);
    }
  }, [apiUrl, heatThreshold, showStocks]);

  // Initialize D3 visualization
  useEffect(() => {
    if (!graphData.nodes.length || loading) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 800;
    const height = 600;

    svg.attr("width", width).attr("height", height);

    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 3])
      .on("zoom", (event) => {
        container.attr("transform", event.transform);
      });

    svg.call(zoom);

    const container = svg.append("g");

    // Create simulation
    const simulation = d3.forceSimulation(graphData.nodes)
      .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(d => {
        if (d.type === 'sector-link') return 120;
        if (d.type === 'stock-link') return 60;
        return 80;
      }))
      .force("charge", d3.forceManyBody().strength(d => {
        if (d.type === 'root') return -1000;
        if (d.type === 'sector') return -400;
        return -200;
      }))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(d => d.radius + 5));

    // Create gradients for heat visualization
    const defs = container.append("defs");
    
    // Heat gradient
    const heatGradient = defs.append("radialGradient")
      .attr("id", "heatGradient");
    
    heatGradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#ff6b35")
      .attr("stop-opacity", 0.8);
    
    heatGradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#ff6b35")
      .attr("stop-opacity", 0.2);

    // Create links
    const link = container.append("g")
      .selectAll("line")
      .data(graphData.links)
      .enter().append("line")
      .attr("class", "link")
      .attr("stroke", d => {
        const intensity = d.value || 0.5;
        return d3.interpolateRdYlBu(1 - intensity);
      })
      .attr("stroke-width", d => Math.max(1, (d.value || 0.5) * 6))
      .attr("stroke-opacity", d => 0.3 + (d.value || 0.5) * 0.7);

    // Create nodes
    const node = container.append("g")
      .selectAll("g")
      .data(graphData.nodes)
      .enter().append("g")
      .attr("class", "node")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Add heat aura for hot stocks
    node.filter(d => d.heat > 0.6)
      .append("circle")
      .attr("r", d => d.radius + 15)
      .attr("fill", "url(#heatGradient)")
      .attr("opacity", d => d.heat * 0.5);

    // Add main circles
    node.append("circle")
      .attr("r", d => d.radius)
      .attr("fill", d => {
        if (d.type === 'root') return "#2563eb";
        if (d.type === 'sector') {
          const intensity = d.heat || 0.5;
          return d3.interpolateRdYlBu(1 - intensity);
        }
        if (d.type === 'stock') {
          const intensity = d.heat || 0.5;
          return d3.interpolateRdBu(intensity);
        }
        return "#64748b";
      })
      .attr("stroke", d => {
        if (d.type === 'root') return "#1e40af";
        if (d.type === 'sector') return "#374151";
        return "#475569";
      })
      .attr("stroke-width", d => d.type === 'root' ? 3 : 2);

    // Add labels
    node.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", d => d.type === 'stock' ? ".35em" : "0.7em")
      .attr("font-size", d => {
        if (d.type === 'root') return "14px";
        if (d.type === 'sector') return "12px";
        return "10px";
      })
      .attr("font-weight", d => d.type === 'root' ? "bold" : "normal")
      .attr("fill", d => d.type === 'stock' ? "#ffffff" : "#000000")
      .text(d => {
        if (d.type === 'sector') {
          return d.label.length > 12 ? d.label.substring(0, 12) + "..." : d.label;
        }
        return d.label;
      });

    // Add heat intensity indicators for stocks
    node.filter(d => d.type === 'stock')
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "-1.2em")
      .attr("font-size", "8px")
      .attr("fill", "#ff6b35")
      .text(d => `🔥${(d.heat * 100).toFixed(0)}%`);

    // Click handler
    node.on("click", (event, d) => {
      setSelectedNode(d);
      event.stopPropagation();
    });

    // Simulation update
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("transform", d => `translate(${d.x},${d.y})`);
    });

    setSimulationRunning(true);

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
      if (d.type !== 'root') {
        d.fx = null;
        d.fy = null;
      }
    }

    // Cleanup
    return () => {
      simulation.stop();
      setSimulationRunning(false);
    };

  }, [graphData]);

  // Auto-refresh data
  useEffect(() => {
    fetchGraphData();
    const interval = setInterval(fetchGraphData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [fetchGraphData]);

  if (loading) {
    return (
      <div className="neo4j-graph-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading knowledge graph...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="neo4j-graph-container">
        <div className="error-message">
          <h3>Error loading graph data</h3>
          <p>{error}</p>
          <button onClick={fetchGraphData} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="neo4j-graph-container">
      <div className="graph-controls">
        <div className="control-group">
          <label>Heat Threshold:</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={heatThreshold}
            onChange={(e) => setHeatThreshold(parseFloat(e.target.value))}
          />
          <span>{(heatThreshold * 100).toFixed(0)}%</span>
        </div>
        
        <div className="control-group">
          <label>
            <input
              type="checkbox"
              checked={showStocks}
              onChange={(e) => setShowStocks(e.target.checked)}
            />
            Show Stocks
          </label>
        </div>

        <div className="control-group">
          <button onClick={fetchGraphData} disabled={loading}>
            🔄 Refresh
          </button>
        </div>

        <div className="status-indicator">
          <span className={`status ${simulationRunning ? 'running' : 'stopped'}`}>
            {simulationRunning ? '🟢 Live' : '🔴 Stopped'}
          </span>
        </div>
      </div>

      <div className="graph-content">
        <svg ref={svgRef}></svg>
        
        {selectedNode && (
          <div className="node-details">
            <div className="details-header">
              <h3>{selectedNode.label}</h3>
              <button onClick={() => setSelectedNode(null)}>×</button>
            </div>
            <div className="details-content">
              <p><strong>Type:</strong> {selectedNode.type}</p>
              <p><strong>Heat Level:</strong> {(selectedNode.heat * 100).toFixed(1)}%</p>
              
              {selectedNode.type === 'sector' && (
                <>
                  <p><strong>Performance:</strong> {selectedNode.performance?.toFixed(2)}%</p>
                  <p><strong>Volume Change:</strong> {selectedNode.volume_change?.toFixed(2)}%</p>
                  <p><strong>Volatility:</strong> {selectedNode.volatility?.toFixed(2)}</p>
                </>
              )}
              
              {selectedNode.type === 'stock' && (
                <>
                  <p><strong>Sector:</strong> {selectedNode.sector}</p>
                  <p><strong>Price:</strong> ${selectedNode.price?.toFixed(2)}</p>
                  <p><strong>Change:</strong> {selectedNode.change?.toFixed(2)}%</p>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="graph-legend">
        <h4>Heat Map Legend</h4>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-circle root"></div>
            <span>Market Root</span>
          </div>
          <div className="legend-item">
            <div className="legend-circle sector"></div>
            <span>Sector</span>
          </div>
          <div className="legend-item">
            <div className="legend-circle stock"></div>
            <span>Stock</span>
          </div>
          <div className="legend-item">
            <div className="legend-circle hot"></div>
            <span>High Heat (60%+)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Neo4jGraphVisualization;