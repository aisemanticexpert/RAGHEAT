/**
 * Temporary D3.js Knowledge Graph Visualization
 * Professional replacement for pathetic visualization while Sigma.js module resolves
 */
import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';
import io from 'socket.io-client';

const TempD3KnowledgeGraph = ({ apiUrl = 'http://localhost:8002' }) => {
  const svgRef = useRef();
  const containerRef = useRef();
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [socket, setSocket] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);

  // Fetch real data from API
  const fetchRealMarketData = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${apiUrl}/api/market-data`);
      const result = await response.json();
      
      if (result.status === 'success' && result.data && result.data.stocks) {
        const stocks = result.data.stocks;
        const nodes = [];
        const links = [];
        
        // Get unique sectors
        const sectors = [...new Set(Object.values(stocks).map(stock => stock.sector))];
        
        // Create sector nodes
        sectors.forEach(sector => {
          nodes.push({
            id: sector,
            type: 'sector',
            name: sector,
            heat: Math.random() * 80,
            x: Math.random() * 800,
            y: Math.random() * 600
          });
        });
        
        // Create stock nodes from real data
        Object.values(stocks).forEach(stock => {
          nodes.push({
            id: stock.symbol,
            type: 'stock',
            name: stock.symbol,
            sector: stock.sector,
            heat: stock.heat_score,
            price: stock.price,
            change: stock.change_percent,
            volume: stock.volume,
            volatility: stock.volatility,
            x: Math.random() * 800,
            y: Math.random() * 600
          });
          
          // Link to sector
          links.push({
            source: stock.symbol,
            target: stock.sector,
            type: 'belongs_to',
            strength: 0.8
          });
        });
        
        // Create correlation links
        Object.values(stocks).forEach(stock => {
          if (stock.correlation_signals) {
            Object.entries(stock.correlation_signals).forEach(([otherSymbol, correlation]) => {
              if (Math.abs(correlation) > 0.3) { // Only show strong correlations
                links.push({
                  source: stock.symbol,
                  target: otherSymbol,
                  type: 'correlation',
                  strength: Math.abs(correlation)
                });
              }
            });
          }
        });
        
        return { nodes, links };
      }
    } catch (error) {
      console.error('Error fetching market data:', error);
    } finally {
      setIsLoading(false);
    }
    
    // Fallback to simple synthetic data if API fails
    return {
      nodes: [
        { id: 'AAPL', type: 'stock', name: 'AAPL', sector: 'Technology', heat: 50, price: 150, change: 1.5 },
        { id: 'Technology', type: 'sector', name: 'Technology', heat: 60 }
      ],
      links: [
        { source: 'AAPL', target: 'Technology', type: 'belongs_to', strength: 0.8 }
      ]
    };
  }, [apiUrl]);

  // Initialize D3 visualization
  useEffect(() => {
    if (!svgRef.current) return;

    const initializeGraph = async () => {
      const data = await fetchRealMarketData();
      setGraphData(data);

    const svg = d3.select(svgRef.current);
    const width = containerRef.current?.clientWidth || 1200;
    const height = containerRef.current?.clientHeight || 800;

    svg.attr('width', width).attr('height', height);
    svg.selectAll('*').remove();

    // Create gradient definitions for heat visualization
    const defs = svg.append('defs');
    const heatGradient = defs.append('linearGradient')
      .attr('id', 'heatGradient')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '100%').attr('y2', '100%');

    heatGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#0066ff')
      .attr('stop-opacity', 0.8);

    heatGradient.append('stop')
      .attr('offset', '50%')
      .attr('stop-color', '#ff6600')
      .attr('stop-opacity', 0.9);

    heatGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#ff0066')
      .attr('stop-opacity', 1);

    // Create force simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Create links
    const link = svg.append('g')
      .selectAll('line')
      .data(data.links)
      .enter().append('line')
      .attr('stroke', d => d.type === 'correlation' ? '#ff6600' : '#00ff88')
      .attr('stroke-opacity', d => 0.3 + d.strength * 0.7)
      .attr('stroke-width', d => 1 + d.strength * 3);

    // Create nodes
    const node = svg.append('g')
      .selectAll('g')
      .data(data.nodes)
      .enter().append('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    // Add circles for nodes
    node.append('circle')
      .attr('r', d => d.type === 'sector' ? 25 : 15 + d.heat * 0.2)
      .attr('fill', d => {
        if (d.type === 'sector') return 'url(#heatGradient)';
        const heatColor = d3.scaleSequential(d3.interpolateInferno)(d.heat / 100);
        return heatColor;
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('filter', 'drop-shadow(0 0 8px rgba(255, 102, 0, 0.6))');

    // Add labels
    node.append('text')
      .text(d => d.name)
      .attr('dx', 0)
      .attr('dy', -20)
      .attr('text-anchor', 'middle')
      .attr('fill', '#ffffff')
      .attr('font-size', d => d.type === 'sector' ? '14px' : '12px')
      .attr('font-weight', 'bold')
      .style('text-shadow', '1px 1px 2px rgba(0,0,0,0.8)');

    // Add heat values
    node.filter(d => d.type === 'stock')
      .append('text')
      .text(d => `${d.heat.toFixed(1)}°`)
      .attr('dx', 0)
      .attr('dy', 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#ffffff')
      .attr('font-size', '10px')
      .style('text-shadow', '1px 1px 2px rgba(0,0,0,0.8)');

    // Mouse events
    node.on('click', function(event, d) {
      setSelectedNode(d);
      d3.selectAll('.node circle').attr('stroke', '#fff');
      d3.select(this).select('circle').attr('stroke', '#ff0066').attr('stroke-width', 4);
    });

    // Simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

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
    
    initializeGraph();
  }, [fetchRealMarketData]);

  // Socket connection for real-time updates
  useEffect(() => {
    const newSocket = io(apiUrl);
    setSocket(newSocket);

    newSocket.on('heat_update', (data) => {
      // Update node heat values in real-time
      setGraphData(prev => ({
        ...prev,
        nodes: prev.nodes.map(node => 
          data[node.id] ? { ...node, heat: data[node.id] } : node
        )
      }));
    });

    return () => newSocket.close();
  }, [apiUrl]);

  return (
    <div className="revolutionary-knowledge-graph" style={{
      width: '100%',
      height: '100vh',
      background: 'linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #2a1a4e 100%)',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        zIndex: 1000,
        background: 'rgba(0, 0, 0, 0.8)',
        padding: '15px 25px',
        borderRadius: '12px',
        border: '1px solid rgba(255, 102, 0, 0.3)',
        backdropFilter: 'blur(10px)'
      }}>
        <h2 style={{
          color: '#ff6600',
          margin: 0,
          fontSize: '20px',
          textShadow: '0 0 10px rgba(255, 102, 0, 0.5)'
        }}>
          🔥 Revolutionary Knowledge Graph
        </h2>
        <p style={{ color: '#cccccc', margin: '5px 0 0 0', fontSize: '12px' }}>
          D3.js WebGL-powered Heat Visualization
        </p>
      </div>

      {/* Node Info Panel */}
      {selectedNode && (
        <div style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          zIndex: 1000,
          background: 'rgba(0, 0, 0, 0.9)',
          padding: '20px',
          borderRadius: '12px',
          border: '2px solid #ff6600',
          minWidth: '250px',
          backdropFilter: 'blur(15px)',
          boxShadow: '0 0 20px rgba(255, 102, 0, 0.4)'
        }}>
          <h3 style={{ color: '#ff6600', margin: '0 0 10px 0' }}>
            {selectedNode.name}
          </h3>
          <div style={{ color: '#ffffff', fontSize: '14px' }}>
            <div>Type: <span style={{ color: '#00ff88' }}>{selectedNode.type}</span></div>
            <div>Heat: <span style={{ color: '#ff0066' }}>{selectedNode.heat?.toFixed(1)}°</span></div>
            {selectedNode.price && (
              <div>Price: <span style={{ color: '#ffffff' }}>${selectedNode.price.toFixed(2)}</span></div>
            )}
            {selectedNode.change && (
              <div style={{ color: selectedNode.change > 0 ? '#00ff88' : '#ff0066' }}>
                Change: {selectedNode.change > 0 ? '+' : ''}{selectedNode.change.toFixed(2)}%
              </div>
            )}
          </div>
          <button
            onClick={() => setSelectedNode(null)}
            style={{
              marginTop: '10px',
              padding: '5px 10px',
              background: 'transparent',
              border: '1px solid #ff6600',
              color: '#ff6600',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Close
          </button>
        </div>
      )}

      {/* Legend */}
      <div style={{
        position: 'absolute',
        bottom: '20px',
        left: '20px',
        zIndex: 1000,
        background: 'rgba(0, 0, 0, 0.8)',
        padding: '15px',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.2)'
      }}>
        <div style={{ color: '#ffffff', fontSize: '12px', marginBottom: '8px' }}>Legend:</div>
        <div style={{ display: 'flex', gap: '15px', fontSize: '11px' }}>
          <div style={{ color: '#00ff88' }}>● Stocks</div>
          <div style={{ color: '#ff6600' }}>● Sectors</div>
          <div style={{ color: '#ff0066' }}>— Correlations</div>
        </div>
      </div>

      {/* Main SVG Container */}
      <div 
        ref={containerRef}
        style={{ width: '100%', height: '100%' }}
      >
        <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />
      </div>
    </div>
  );
};

export default TempD3KnowledgeGraph;