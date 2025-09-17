import React, { useState, useEffect, useRef, useCallback } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import Cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import cola from 'cytoscape-cola';
import fcose from 'cytoscape-fcose';
import io from 'socket.io-client';
import './EnhancedKnowledgeGraph.css';

// Register Cytoscape extensions
Cytoscape.use(dagre);
Cytoscape.use(cola);
Cytoscape.use(fcose);

const EnhancedKnowledgeGraph = ({ apiUrl = 'http://localhost:8001' }) => {
  const cyRef = useRef();
  const socketRef = useRef();
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeDetails, setNodeDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [heatMapEnabled, setHeatMapEnabled] = useState(true);
  const [layout, setLayout] = useState('concentric');
  const [isLiveData, setIsLiveData] = useState(false);
  const [marketStatus, setMarketStatus] = useState('closed');
  const [expandedNodes, setExpandedNodes] = useState(new Set());
  const [animationEnabled, setAnimationEnabled] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Professional financial color schemes
  const colorSchemes = {
    nodes: {
      market: '#4338CA',          // Deep indigo for market root
      sector: {
        technology: '#8B5CF6',    // Purple
        financials: '#059669',    // Emerald
        healthcare: '#DC2626',    // Red
        energy: '#F59E0B',        // Amber
        consumer_discretionary: '#EC4899',  // Pink
        consumer_staples: '#10B981',        // Emerald
        industrials: '#6B7280',   // Gray
        materials: '#F97316',     // Orange
        utilities: '#14B8A6',     // Teal
        real_estate: '#7C3AED',   // Violet
        communication: '#3B82F6'  // Blue
      },
      stock: {
        hot: '#DC2626',           // Bright red (heat > 0.7)
        warm: '#F59E0B',          // Orange (heat 0.4-0.7)
        neutral: '#6B7280',       // Gray (heat 0.2-0.4)
        cool: '#10B981',          // Green (heat < 0.2)
        frozen: '#3B82F6'         // Blue (heat < 0.1)
      }
    },
    edges: {
      contains: '#4338CA',
      belongs_to: '#8B5CF6',
      correlates_with: '#EC4899',
      influences: '#F59E0B',
      competes_with: '#DC2626'
    }
  };

  // Market hours detection
  const isMarketOpen = () => {
    const now = new Date();
    const currentHour = now.getHours();
    const currentDay = now.getDay();
    
    // Market is open Mon-Fri, 9:30 AM - 4:00 PM EST
    const isWeekday = currentDay >= 1 && currentDay <= 5;
    const isMarketHours = currentHour >= 9 && currentHour < 16;
    
    return isWeekday && isMarketHours;
  };

  // Initialize WebSocket connection for real-time data
  useEffect(() => {
    if (isMarketOpen()) {
      setMarketStatus('open');
      setIsLiveData(true);
      
      // Connect to live data stream
      socketRef.current = io(`${apiUrl}/live-graph`);
      
      socketRef.current.on('connect', () => {
        console.log('🔴 Connected to live graph stream');
      });

      socketRef.current.on('graph_update', (data) => {
        console.log('📊 Received live graph update:', data);
        updateGraphWithLiveData(data);
        setLastUpdate(new Date());
      });

      socketRef.current.on('heat_update', (data) => {
        console.log('🔥 Received heat update:', data);
        updateNodeHeat(data);
      });

      socketRef.current.on('disconnect', () => {
        console.log('❌ Disconnected from live stream');
      });

      return () => {
        socketRef.current?.disconnect();
      };
    } else {
      setMarketStatus('closed');
      setIsLiveData(false);
    }
  }, [apiUrl]);

  // Professional Cytoscape styles using backend-provided styling
  const cytoscapeStyles = [
    // Base node styles - use backend-provided styling
    {
      selector: 'node',
      style: {
        'width': 'data(size)',
        'height': 'data(size)',
        'background-color': 'data(color)',
        'border-width': 'data(border_width)',
        'border-color': 'data(border_color)',
        'opacity': 'data(opacity)',
        'label': 'data(label)',
        'text-valign': 'center',
        'text-halign': 'center',
        'color': '#FFFFFF',
        'font-family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
        'font-size': '10px',
        'font-weight': '600',
        'text-outline-color': '#000000',
        'text-outline-width': 1,
        'text-wrap': 'wrap',
        'text-max-width': '80px',
        'transition-property': 'background-color, border-color, width, height, opacity',
        'transition-duration': '0.4s',
        'transition-timing-function': 'ease-out'
      }
    },
    
    // Sector nodes - larger and more prominent
    {
      selector: 'node[type*="sector"]',
      style: {
        'font-size': '12px',
        'font-weight': '700',
        'text-outline-width': 2,
        'border-width': 4
      }
    },
    
    // Stock nodes
    {
      selector: 'node[type="technology"], node[type="financial_services"], node[type="healthcare"], node[type="consumer_discretionary"], node[type="consumer_staples"]',
      style: {
        'font-size': '9px',
        'border-width': 2
      }
    },
    
    // Stock nodes with heat visualization
    {
      selector: 'node[type="stock"]',
      style: {
        'width': 'mapData(heat, 0, 1, 25, 50)',
        'height': 'mapData(heat, 0, 1, 25, 50)',
        'font-size': '10px'
      }
    },
    
    // Hot stocks (pulsing effect)
    {
      selector: 'node[heat > 0.7]',
      style: {
        'border-color': '#FEE2E2',
        'border-width': 4,
        'background-color': colorSchemes.nodes.stock.hot
      }
    },
    
    // Warm stocks
    {
      selector: 'node[heat > 0.4][heat <= 0.7]',
      style: {
        'background-color': colorSchemes.nodes.stock.warm
      }
    },
    
    // Cool stocks
    {
      selector: 'node[heat <= 0.4]',
      style: {
        'background-color': colorSchemes.nodes.stock.cool
      }
    },
    
    // Selected node
    {
      selector: 'node:selected',
      style: {
        'border-color': '#FCD34D',
        'border-width': 5,
        'shadow-blur': 15,
        'shadow-color': 'rgba(252, 211, 77, 0.6)'
      }
    },
    
    // Edge styles - use backend-provided styling
    {
      selector: 'edge',
      style: {
        'width': 'data(width)',
        'line-color': 'data(color)',
        'line-style': 'data(style)',
        'opacity': 'data(opacity)',
        'curve-style': 'bezier',
        'control-point-step-size': 40,
        'transition-property': 'width, line-color, opacity',
        'transition-duration': '0.3s'
      }
    },
    
    // Sector relationship edges (belongs_to)
    {
      selector: 'edge[type="belongs_to"]',
      style: {
        'target-arrow-shape': 'triangle',
        'target-arrow-color': 'data(color)',
        'arrow-scale': 0.8
      }
    },
    
    // Correlation edges
    {
      selector: 'edge[type="correlated"]',
      style: {
        'line-style': 'dashed',
        'target-arrow-shape': 'none'
      }
    },
    
    {
      selector: 'edge[relationship="CORRELATES_WITH"]',
      style: {
        'line-color': colorSchemes.edges.correlates_with,
        'target-arrow-color': colorSchemes.edges.correlates_with,
        'width': 2,
        'line-style': 'dashed'
      }
    },
    
    // Hover effects
    {
      selector: 'node:active',
      style: {
        'overlay-color': 'rgba(252, 211, 77, 0.3)',
        'overlay-padding': 10
      }
    }
  ];

  // Layout configurations for different visualization modes
  const layoutConfigs = {
    fcose: {
      name: 'fcose',
      quality: 'default',
      randomize: true,
      animate: animationEnabled,
      animationDuration: 2000,
      fit: true,
      padding: 80,
      nodeSeparation: 200,
      idealEdgeLength: 250,
      nodeRepulsion: 20000,
      edgeElasticity: 0.1,
      nestingFactor: 0.1,
      numIter: 2000,
      tile: false,
      uniformNodeDimensions: false,
      packComponents: true,
      step: 'all',
      samplingType: false,
      sampleSize: 25,
      nodeDimensionsIncludeLabels: true
    },
    
    dagre: {
      name: 'dagre',
      rankDir: 'TB',
      animate: animationEnabled,
      animationDuration: 1500,
      fit: true,
      padding: 50,
      spacingFactor: 1.5,
      nodeSeparation: 50,
      rankSeparation: 100
    },
    
    cola: {
      name: 'cola',
      animate: animationEnabled,
      animationDuration: 2000,
      maxSimulationTime: 5000,
      unconstrIter: 1000,
      userConstIter: 0,
      allConstIter: 1000,
      fit: true,
      padding: 50,
      edgeLength: 120,
      nodeSpacing: 30,
      flow: { axis: 'y', minSeparation: 50 }
    },
    
    concentric: {
      name: 'concentric',
      fit: true,
      padding: 100,
      animate: animationEnabled,
      animationDuration: 1500,
      concentric: function(node) {
        // Sectors in center (highest value = innermost)
        if (node.data('type').includes('sector')) {
          return 100;
        } else {
          // Hot stocks in inner ring, cool stocks in outer ring
          const heat = node.data('heat_level') || 0;
          return Math.floor(heat / 10) + 1; // Heat levels 0-9
        }
      },
      levelWidth: function(nodes) {
        return 3; // More nodes per ring
      },
      minNodeSpacing: 120,
      spacingFactor: 2.0,
      startAngle: 0,
      sweep: 6.28 // Full circle
    },
    
    circle: {
      name: 'circle',
      fit: true,
      padding: 80,
      animate: animationEnabled,
      animationDuration: 1500,
      radius: 200,
      spacingFactor: 1.2,
      transform: function(node, position) {
        return position;
      }
    },
    
    // Custom grid layout that prevents linear arrangement
    grid: {
      name: 'grid',
      fit: true,
      padding: 80,
      animate: animationEnabled,
      animationDuration: 1500,
      rows: 4,
      cols: 5,
      position: function(node) {
        // Custom positioning to distribute nodes in a grid pattern
        const index = node.id().replace(/[^0-9]/g, '') || node.id().length;
        const row = Math.floor(index / 5);
        const col = index % 5;
        return { row, col };
      },
      sort: function(a, b) {
        // Sort sectors first, then by heat level
        const aIsSector = a.data('type').includes('sector');
        const bIsSector = b.data('type').includes('sector');
        
        if (aIsSector && !bIsSector) return -1;
        if (!aIsSector && bIsSector) return 1;
        
        return (b.data('heat_level') || 0) - (a.data('heat_level') || 0);
      }
    }
  };

  // Calculate node color based on type and heat
  const getNodeColor = (node) => {
    if (node.type === 'market') {
      return colorSchemes.nodes.market;
    }
    
    if (node.type === 'sector') {
      const sectorKey = node.sector_key?.toLowerCase() || 'technology';
      return colorSchemes.nodes.sector[sectorKey] || colorSchemes.nodes.sector.technology;
    }
    
    if (node.type === 'stock') {
      const heat = node.heat || 0;
      if (heat > 0.7) return colorSchemes.nodes.stock.hot;
      if (heat > 0.4) return colorSchemes.nodes.stock.warm;
      if (heat > 0.2) return colorSchemes.nodes.stock.neutral;
      if (heat > 0.1) return colorSchemes.nodes.stock.cool;
      return colorSchemes.nodes.stock.frozen;
    }
    
    return '#6B7280'; // Default gray
  };

  // Fetch graph data from API
  const fetchGraphData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Determine data source based on market status
      const endpoint = isLiveData 
        ? `${apiUrl}/api/graph/live-data`
        : `${apiUrl}/api/graph/synthetic-data`;

      console.log(`📡 Fetching ${isLiveData ? 'live' : 'synthetic'} graph data from:`, endpoint);

      const response = await fetch(endpoint);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch graph data');
      }

      const { nodes = [], relationships = [] } = result.data;

      // Transform nodes for Cytoscape
      const cytoscapeNodes = nodes.map(node => ({
        data: {
          id: node.id,
          label: node.label || node.id,
          type: node.type,
          heat: node.heat_level || node.heat || Math.random() * 0.8 + 0.1,
          size: node.type === 'market' ? 80 : node.type === 'sector' ? 60 : 40,
          color: getNodeColor(node),
          sector_key: node.sector || node.sector_key,
          price: node.price,
          change: node.change_percent,
          volume: node.volume,
          market_cap: node.market_cap,
          performance: node.performance,
          volatility: node.volatility,
          ai_prediction: node.ai_prediction
        }
      }));

      // Transform edges for Cytoscape
      const cytoscapeEdges = relationships.map((rel, index) => ({
        data: {
          id: `edge-${index}`,
          source: rel.source,
          target: rel.target,
          relationship: rel.type,
          weight: rel.weight || 2,
          strength: rel.strength || 0.5,
          color: colorSchemes.edges[rel.type?.toLowerCase()] || '#64748B',
          value: rel.value || rel.properties?.value
        }
      }));

      setGraphData({
        nodes: cytoscapeNodes,
        edges: cytoscapeEdges
      });

      console.log(`✅ Loaded ${cytoscapeNodes.length} nodes and ${cytoscapeEdges.length} edges`);
      setLoading(false);

    } catch (err) {
      console.error('❌ Error fetching graph data:', err);
      setError(err.message);
      setLoading(false);
      
      // Fallback to synthetic data generation
      if (isLiveData) {
        generateSyntheticData();
      }
    }
  }, [apiUrl, isLiveData]);

  // Generate synthetic data when live data is unavailable
  const generateSyntheticData = () => {
    console.log('🔄 Generating synthetic graph data...');
    
    const sectors = [
      { id: 'TECHNOLOGY', name: 'Technology', stocks: ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'] },
      { id: 'FINANCIALS', name: 'Financials', stocks: ['JPM', 'BAC', 'WFC', 'GS'] },
      { id: 'HEALTHCARE', name: 'Healthcare', stocks: ['JNJ', 'PFE', 'UNH', 'ABBV'] },
      { id: 'ENERGY', name: 'Energy', stocks: ['XOM', 'CVX', 'COP', 'SLB'] },
      { id: 'CONSUMER_DISCRETIONARY', name: 'Consumer Discretionary', stocks: ['AMZN', 'TSLA', 'HD', 'MCD'] }
    ];

    const nodes = [
      // Market root
      {
        data: {
          id: 'MARKET',
          label: 'Global Market',
          type: 'market',
          heat: 0.6,
          size: 80,
          color: colorSchemes.nodes.market
        }
      }
    ];

    const edges = [];

    // Add sectors and stocks
    sectors.forEach((sector, sectorIndex) => {
      const sectorHeat = Math.random() * 0.6 + 0.2;
      
      // Add sector node
      nodes.push({
        data: {
          id: sector.id,
          label: sector.name,
          type: 'sector',
          heat: sectorHeat,
          size: 60,
          color: colorSchemes.nodes.sector[sector.id.toLowerCase()] || colorSchemes.nodes.sector.technology,
          sector_key: sector.id.toLowerCase()
        }
      });

      // Connect sector to market
      edges.push({
        data: {
          id: `market-${sector.id}`,
          source: 'MARKET',
          target: sector.id,
          relationship: 'CONTAINS',
          weight: 4,
          color: colorSchemes.edges.contains
        }
      });

      // Add stocks in sector
      sector.stocks.forEach((stock, stockIndex) => {
        const stockHeat = Math.random() * 0.9 + 0.05;
        const price = 50 + Math.random() * 500;
        const change = (Math.random() - 0.5) * 10;
        
        nodes.push({
          data: {
            id: stock,
            label: stock,
            type: 'stock',
            heat: stockHeat,
            size: Math.max(25, 25 + stockHeat * 20),
            color: getNodeColor({ type: 'stock', heat: stockHeat }),
            sector_key: sector.id.toLowerCase(),
            price: price,
            change: change,
            volume: Math.floor(Math.random() * 50000000) + 1000000,
            market_cap: price * (Math.random() * 1000 + 100) * 1000000
          }
        });

        // Connect stock to sector
        edges.push({
          data: {
            id: `${sector.id}-${stock}`,
            source: sector.id,
            target: stock,
            relationship: 'BELONGS_TO',
            weight: 3,
            color: colorSchemes.edges.belongs_to
          }
        });
      });
    });

    // Add some correlation edges between similar stocks
    const techStocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'];
    for (let i = 0; i < techStocks.length - 1; i++) {
      if (Math.random() < 0.3) {
        edges.push({
          data: {
            id: `corr-${techStocks[i]}-${techStocks[i + 1]}`,
            source: techStocks[i],
            target: techStocks[i + 1],
            relationship: 'CORRELATES_WITH',
            weight: 2,
            color: colorSchemes.edges.correlates_with,
            value: `${Math.floor(Math.random() * 30 + 70)}% corr`
          }
        });
      }
    }

    setGraphData({ nodes, edges });
    setLoading(false);
    console.log('✅ Generated synthetic data with', nodes.length, 'nodes and', edges.length, 'edges');
  };

  // Update graph with live data
  const updateGraphWithLiveData = (liveData) => {
    if (!cyRef.current) return;

    const cy = cyRef.current;
    
    // Update node data
    Object.entries(liveData.nodes || {}).forEach(([nodeId, nodeData]) => {
      const node = cy.getElementById(nodeId);
      if (node.length > 0) {
        // Update heat and other properties
        const newHeat = nodeData.heat_level || nodeData.heat || node.data('heat');
        const newColor = getNodeColor({ type: node.data('type'), heat: newHeat });
        
        node.data({
          heat: newHeat,
          color: newColor,
          price: nodeData.price || node.data('price'),
          change: nodeData.change_percent || node.data('change'),
          volume: nodeData.volume || node.data('volume')
        });

        // Animate hot nodes
        if (newHeat > 0.7 && animationEnabled) {
          node.animate({
            style: { 'border-width': 6 },
            duration: 300
          }).animate({
            style: { 'border-width': 3 },
            duration: 300
          });
        }
      }
    });

    // Update selected node details if it was updated
    if (selectedNode && liveData.nodes[selectedNode.data('id')]) {
      const updatedNode = cy.getElementById(selectedNode.data('id'));
      if (updatedNode.length > 0) {
        setSelectedNode(updatedNode);
      }
    }
  };

  // Update node heat specifically
  const updateNodeHeat = (heatData) => {
    if (!cyRef.current) return;

    const cy = cyRef.current;
    
    Object.entries(heatData).forEach(([nodeId, heat]) => {
      const node = cy.getElementById(nodeId);
      if (node.length > 0) {
        const newColor = getNodeColor({ type: node.data('type'), heat });
        node.data({ heat, color: newColor });
        
        // Pulse effect for very hot nodes
        if (heat > 0.8 && animationEnabled) {
          node.style('border-color', '#FEE2E2');
          setTimeout(() => {
            node.style('border-color', '#FFFFFF');
          }, 500);
        }
      }
    });
  };

  // Fetch detailed node analysis
  const fetchNodeDetails = useCallback(async (nodeId, nodeType) => {
    if (nodeType !== 'stock') return null;
    
    try {
      const response = await fetch(`${apiUrl}/api/stock/${nodeId}/analysis`);
      if (!response.ok) return null;
      
      const result = await response.json();
      return result.data;
    } catch (error) {
      console.error('Error fetching node details:', error);
      return null;
    }
  }, [apiUrl]);

  // Handle node selection
  const handleNodeSelect = useCallback(async (event) => {
    const node = event.target;
    setSelectedNode(node);
    
    if (node.data('type') === 'stock') {
      setNodeDetails(null);
      const details = await fetchNodeDetails(node.data('id'), node.data('type'));
      setNodeDetails(details);
    } else {
      setNodeDetails(null);
    }
  }, [fetchNodeDetails]);

  // Apply layout
  const applyLayout = (layoutName) => {
    if (!cyRef.current) return;
    
    const cy = cyRef.current;
    const layoutConfig = layoutConfigs[layoutName];
    
    if (layoutConfig) {
      const layout = cy.layout(layoutConfig);
      layout.run();
    }
  };

  // Filter graph data
  const getFilteredData = () => {
    let filteredNodes = graphData.nodes;
    let filteredEdges = graphData.edges;

    // Apply search filter
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      filteredNodes = filteredNodes.filter(node =>
        node.data.label.toLowerCase().includes(searchLower) ||
        node.data.id.toLowerCase().includes(searchLower)
      );
      
      const nodeIds = new Set(filteredNodes.map(n => n.data.id));
      filteredEdges = filteredEdges.filter(edge =>
        nodeIds.has(edge.data.source) && nodeIds.has(edge.data.target)
      );
    }

    // Apply type filter
    if (filterType !== 'all') {
      filteredNodes = filteredNodes.filter(node => node.data.type === filterType);
      
      const nodeIds = new Set(filteredNodes.map(n => n.data.id));
      filteredEdges = filteredEdges.filter(edge =>
        nodeIds.has(edge.data.source) && nodeIds.has(edge.data.target)
      );
    }

    return [...filteredNodes, ...filteredEdges];
  };

  // Initialize data fetching
  useEffect(() => {
    fetchGraphData();
  }, [fetchGraphData]);

  // Auto-refresh data periodically
  useEffect(() => {
    if (isLiveData) {
      const interval = setInterval(fetchGraphData, 30000); // Every 30 seconds
      return () => clearInterval(interval);
    }
  }, [fetchGraphData, isLiveData]);

  // Apply layout when layout changes
  useEffect(() => {
    if (cyRef.current && graphData.nodes.length > 0) {
      setTimeout(() => applyLayout(layout), 500);
    }
  }, [layout, graphData]);

  if (loading) {
    return (
      <div className="enhanced-graph-container">
        <div className="graph-loading">
          <div className="loading-spinner"></div>
          <div className="loading-text">
            <h3>Loading Knowledge Graph...</h3>
            <p>Initializing {isLiveData ? 'live data' : 'synthetic data'} pipeline</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="enhanced-graph-container">
        <div className="graph-error">
          <h3>🚨 Graph Loading Error</h3>
          <p>{error}</p>
          <div className="error-actions">
            <button onClick={fetchGraphData} className="retry-btn">
              🔄 Retry Connection
            </button>
            <button onClick={generateSyntheticData} className="synthetic-btn">
              🎲 Use Synthetic Data
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="enhanced-graph-container">
      {/* Professional toolbar */}
      <div className="graph-toolbar">
        <div className="toolbar-section">
          <div className="market-status">
            <div className={`status-indicator ${marketStatus}`}></div>
            <span className="status-text">
              {marketStatus === 'open' ? '🟢 Market Open' : '🔴 Market Closed'}
            </span>
            <span className="data-source">
              {isLiveData ? '📡 Live Data' : '🎲 Synthetic Data'}
            </span>
          </div>
          
          {lastUpdate && (
            <div className="last-update">
              Last Update: {lastUpdate.toLocaleTimeString()}
            </div>
          )}
        </div>

        <div className="toolbar-section">
          <input
            type="text"
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Nodes</option>
            <option value="market">Market</option>
            <option value="sector">Sectors</option>
            <option value="stock">Stocks</option>
          </select>
        </div>

        <div className="toolbar-section">
          <select
            value={layout}
            onChange={(e) => setLayout(e.target.value)}
            className="layout-select"
          >
            <option value="concentric">Concentric (Recommended)</option>
            <option value="fcose">Force-Directed (fCoSE)</option>
            <option value="dagre">Hierarchical (Dagre)</option>
            <option value="cola">Constraint-Based (Cola)</option>
            <option value="circle">Circle Layout</option>
            <option value="grid">Grid Layout</option>
          </select>
          
          <button
            onClick={() => applyLayout(layout)}
            className="layout-btn"
            title="Re-apply Layout"
          >
            🔄
          </button>
        </div>

        <div className="toolbar-section">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={heatMapEnabled}
              onChange={(e) => setHeatMapEnabled(e.target.checked)}
            />
            Heat Map
          </label>
          
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={animationEnabled}
              onChange={(e) => setAnimationEnabled(e.target.checked)}
            />
            Animations
          </label>
        </div>

        <div className="toolbar-section">
          <div className="node-count">
            {graphData.nodes.length} nodes • {graphData.edges.length} edges
          </div>
        </div>
      </div>

      {/* Main graph area */}
      <div className="graph-main">
        <div className="cytoscape-container">
          <CytoscapeComponent
            elements={getFilteredData()}
            style={{ width: '100%', height: '100%' }}
            stylesheet={cytoscapeStyles}
            layout={layoutConfigs[layout]}
            cy={(cy) => {
              cyRef.current = cy;
              
              // Set up event listeners
              cy.on('tap', 'node', handleNodeSelect);
              
              cy.on('mouseover', 'node', (event) => {
                const node = event.target;
                node.style({
                  'border-width': 5,
                  'shadow-blur': 12
                });
              });
              
              cy.on('mouseout', 'node', (event) => {
                const node = event.target;
                node.style({
                  'border-width': 3,
                  'shadow-blur': 8
                });
              });
            }}
            wheelSensitivity={0.2}
            minZoom={0.1}
            maxZoom={3.0}
          />
        </div>

        {/* Node details panel */}
        {selectedNode && (
          <div className="node-details-panel">
            <div className="details-header">
              <h3>
                <span className="node-icon">
                  {selectedNode.data('type') === 'market' ? '🌐' :
                   selectedNode.data('type') === 'sector' ? '🏢' : '📊'}
                </span>
                {selectedNode.data('label')}
              </h3>
              <button 
                onClick={() => setSelectedNode(null)}
                className="close-btn"
              >
                ✕
              </button>
            </div>

            <div className="details-content">
              <div className="node-properties">
                <div className="property">
                  <span className="prop-label">Type:</span>
                  <span className="prop-value">{selectedNode.data('type')}</span>
                </div>
                
                <div className="property">
                  <span className="prop-label">Heat Level:</span>
                  <span className={`prop-value heat-${
                    selectedNode.data('heat') > 0.7 ? 'hot' :
                    selectedNode.data('heat') > 0.4 ? 'warm' : 'cool'
                  }`}>
                    {(selectedNode.data('heat') * 100).toFixed(1)}%
                    {selectedNode.data('heat') > 0.7 && ' 🔥'}
                  </span>
                </div>

                {selectedNode.data('type') === 'stock' && (
                  <>
                    <div className="property">
                      <span className="prop-label">Price:</span>
                      <span className="prop-value">
                        ${selectedNode.data('price')?.toFixed(2) || 'N/A'}
                      </span>
                    </div>
                    
                    <div className="property">
                      <span className="prop-label">Change:</span>
                      <span className={`prop-value ${
                        (selectedNode.data('change') || 0) >= 0 ? 'positive' : 'negative'
                      }`}>
                        {(selectedNode.data('change') || 0) >= 0 ? '+' : ''}
                        {selectedNode.data('change')?.toFixed(2) || '0.00'}%
                      </span>
                    </div>
                    
                    <div className="property">
                      <span className="prop-label">Volume:</span>
                      <span className="prop-value">
                        {selectedNode.data('volume')?.toLocaleString() || 'N/A'}
                      </span>
                    </div>
                  </>
                )}
              </div>

              {/* Live analysis data */}
              {selectedNode.data('type') === 'stock' && nodeDetails && (
                <div className="analysis-section">
                  <h4>🧠 AI Analysis</h4>
                  
                  <div className="recommendation">
                    <div className="rec-header">
                      <span className={`rec-action ${nodeDetails.recommendation?.action?.toLowerCase()}`}>
                        {nodeDetails.recommendation?.action || 'HOLD'}
                      </span>
                      <span className="rec-confidence">
                        {Math.round((nodeDetails.recommendation?.confidence || 0.5) * 100)}%
                      </span>
                    </div>
                    
                    {nodeDetails.recommendation?.price_target && (
                      <div className="price-target">
                        Target: ${nodeDetails.recommendation.price_target.toFixed(2)}
                      </div>
                    )}
                  </div>

                  {nodeDetails.recommendation?.explanation && (
                    <div className="explanation">
                      <div className="reasoning">
                        <h5>📈 Key Factors</h5>
                        <ul>
                          {nodeDetails.recommendation.explanation.reasoning_path?.slice(0, 3).map((reason, i) => (
                            <li key={i}>{reason}</li>
                          ))}
                        </ul>
                      </div>
                      
                      <div className="risks">
                        <h5>⚠️ Risk Factors</h5>
                        <ul>
                          {nodeDetails.recommendation.explanation.risk_factors?.slice(0, 2).map((risk, i) => (
                            <li key={i}>{risk}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {selectedNode.data('type') === 'stock' && !nodeDetails && (
                <div className="loading-analysis">
                  <div className="loading-spinner-small"></div>
                  <span>Loading AI analysis...</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Professional legend */}
      <div className="graph-legend">
        <h4>Graph Legend</h4>
        
        <div className="legend-section">
          <h5>Heat Levels</h5>
          <div className="heat-legend">
            <div className="heat-item">
              <div className="heat-indicator hot"></div>
              <span>Hot (&gt;70%)</span>
            </div>
            <div className="heat-item">
              <div className="heat-indicator warm"></div>
              <span>Warm (40-70%)</span>
            </div>
            <div className="heat-item">
              <div className="heat-indicator cool"></div>
              <span>Cool (&lt;40%)</span>
            </div>
          </div>
        </div>
        
        <div className="legend-section">
          <h5>Relationships</h5>
          <div className="rel-legend">
            <div className="rel-item">
              <div className="rel-line contains"></div>
              <span>Contains</span>
            </div>
            <div className="rel-item">
              <div className="rel-line belongs-to"></div>
              <span>Belongs To</span>
            </div>
            <div className="rel-item">
              <div className="rel-line correlates"></div>
              <span>Correlates</span>
            </div>
          </div>
        </div>
        
        <div className="legend-section">
          <h5>Controls</h5>
          <ul className="controls-list">
            <li>Click nodes for details</li>
            <li>Drag to reposition</li>
            <li>Scroll to zoom</li>
            <li>Search to filter</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default EnhancedKnowledgeGraph;