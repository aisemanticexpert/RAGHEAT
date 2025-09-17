/**
 * Advanced Ontology-Aware Knowledge Graph Visualization
 * Professional implementation using Cytoscape.js with semantic layouts
 * Inspired by yFiles and Tom Sawyer Perspectives approaches
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import coseBilkent from 'cytoscape-cose-bilkent';
import cola from 'cytoscape-cola';
import euler from 'cytoscape-euler';
import spread from 'cytoscape-spread';
import './AdvancedOntologyGraph.css';

// Register layout extensions
cytoscape.use(dagre);
cytoscape.use(coseBilkent);
cytoscape.use(cola);
cytoscape.use(euler);
cytoscape.use(spread);

const AdvancedOntologyGraph = ({ apiUrl = 'http://localhost:8001' }) => {
  const cyRef = useRef();
  const containerRef = useRef();
  
  // State management
  const [currentLevel, setCurrentLevel] = useState(3);
  const [currentLayout, setCurrentLayout] = useState('cose-bilkent');
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [marketOverview, setMarketOverview] = useState(null);
  const [error, setError] = useState(null);
  const [filterOptions, setFilterOptions] = useState({
    nodeTypes: new Set(),
    relationships: new Set(),
    sectors: new Set()
  });
  const [activeFilters, setActiveFilters] = useState({
    nodeTypes: new Set(),
    relationships: new Set(),
    sectors: new Set()
  });

  // Professional ontology-aware layouts
  const layoutConfigs = {
    'dagre': {
      name: 'dagre',
      rankDir: 'TB',
      nodeSep: 80,
      edgeSep: 40,
      rankSep: 100,
      animate: true,
      animationDuration: 1000
    },
    'cose-bilkent': {
      name: 'cose-bilkent',
      quality: 'proof',
      nodeRepulsion: 4500,
      idealEdgeLength: 120,
      edgeElasticity: 0.45,
      nestingFactor: 0.1,
      gravity: 0.4,
      numIter: 2500,
      tile: true,
      animate: 'end',
      animationDuration: 1000
    },
    'cola': {
      name: 'cola',
      animate: true,
      refresh: 1,
      maxSimulationTime: 4000,
      ungrabifyWhileSimulating: false,
      fit: true,
      padding: 30,
      nodeDimensionsIncludeLabels: false,
      randomize: false,
      avoidOverlap: true,
      handleDisconnected: true,
      convergenceThreshold: 0.01,
      nodeSpacing: 100,
      flow: undefined,
      alignment: undefined,
      gapInequalities: undefined
    },
    'euler': {
      name: 'euler',
      springLength: 80,
      springCoeff: 0.0008,
      mass: 4,
      gravity: -1.2,
      pull: 0.001,
      theta: 0.666,
      dragCoeff: 0.02,
      movementThreshold: 1,
      timeStep: 20,
      refresh: 10,
      animate: true,
      animationDuration: 2000
    },
    'spread': {
      name: 'spread',
      animate: true,
      ready: undefined,
      stop: undefined,
      fit: true,
      minDist: 20,
      padding: 20,
      expandingFactor: -1.0,
      layoutBy: undefined,
      prelayout: { name: 'cose' },
      maxExpandIterations: 4,
      boundingBox: undefined,
      randomize: false
    }
  };

  // Ontology-aware node styles
  const getNodeStyle = (node) => {
    const type = node.type || node.data?.type;
    const level = node.level || node.data?.level || currentLevel;
    
    const baseStyle = {
      'width': 'data(size)',
      'height': 'data(size)',
      'background-color': 'data(color)',
      'border-width': 3,
      'border-color': '#ffffff',
      'border-opacity': 0.8,
      'label': 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      'font-size': '12px',
      'font-weight': 'bold',
      'color': '#ffffff',
      'text-outline-width': 2,
      'text-outline-color': '#000000',
      'text-outline-opacity': 0.7,
      'shadow-blur': 6,
      'shadow-color': '#000000',
      'shadow-opacity': 0.3,
      'shadow-offset-x': 2,
      'shadow-offset-y': 2
    };

    // Ontology-specific styling
    switch (type) {
      case 'Sector':
        return {
          ...baseStyle,
          'shape': 'round-rectangle',
          'background-gradient-direction': 'to-bottom',
          'background-gradient-stop-colors': 'data(color) #ffffff',
          'background-gradient-stop-positions': '0% 100%'
        };
      
      case 'Stock':
        return {
          ...baseStyle,
          'shape': 'ellipse',
          'background-gradient-direction': 'to-bottom-right',
          'background-gradient-stop-colors': 'data(color) #333333',
          'background-gradient-stop-positions': '0% 100%'
        };
      
      case 'BuySignal':
        return {
          ...baseStyle,
          'shape': 'triangle',
          'background-color': '#00FF88',
          'border-color': '#00CC66'
        };
      
      case 'SellSignal':
        return {
          ...baseStyle,
          'shape': 'triangle',
          'background-color': '#FF6B35',
          'border-color': '#CC4422',
          'transform': 'rotate(180deg)'
        };
      
      case 'HoldSignal':
        return {
          ...baseStyle,
          'shape': 'square',
          'background-color': '#FFA726',
          'border-color': '#FF8800'
        };
      
      case 'TechnicalIndicator':
        return {
          ...baseStyle,
          'shape': 'diamond',
          'background-gradient-direction': 'to-top',
          'background-gradient-stop-colors': 'data(color) #666666',
          'background-gradient-stop-positions': '0% 100%'
        };
      
      case 'Corporation':
        return {
          ...baseStyle,
          'shape': 'round-rectangle',
          'background-opacity': 0.8
        };

      default:
        return baseStyle;
    }
  };

  // Ontology-aware edge styles
  const getEdgeStyle = (relationship) => {
    const baseStyle = {
      'width': 'data(strength)',
      'line-color': 'data(color)',
      'target-arrow-color': 'data(color)',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'opacity': 0.8,
      'label': 'data(label)',
      'font-size': '10px',
      'color': '#ffffff',
      'text-outline-width': 1,
      'text-outline-color': '#000000',
      'text-rotation': 'autorotate'
    };

    switch (relationship) {
      case 'belongs_to_sector':
      case 'same_sector':
        return {
          ...baseStyle,
          'line-style': 'solid',
          'line-color': '#00D4FF',
          'target-arrow-color': '#00D4FF'
        };
      
      case 'correlates_with':
        return {
          ...baseStyle,
          'line-style': 'dashed',
          'line-color': '#FF6600',
          'target-arrow-color': '#FF6600',
          'line-dash-pattern': [5, 5]
        };
      
      case 'generates_signal':
        return {
          ...baseStyle,
          'line-style': 'solid',
          'line-color': '#00FF88',
          'target-arrow-color': '#00FF88',
          'width': 4
        };
      
      case 'calculates_for':
        return {
          ...baseStyle,
          'line-style': 'dotted',
          'line-color': '#9966FF',
          'target-arrow-color': '#9966FF'
        };
      
      case 'heat_flow':
        return {
          ...baseStyle,
          'line-style': 'solid',
          'line-color': '#FF3366',
          'target-arrow-color': '#FF3366',
          'width': 'mapData(heat_flow, 0, 1, 2, 8)',
          'source-arrow-shape': 'circle'
        };
      
      default:
        return baseStyle;
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
      
      // Extract filter options
      const nodeTypes = new Set(data.nodes?.map(n => n.type) || []);
      const relationships = new Set(data.links?.map(l => l.relationship) || []);
      const sectors = new Set(data.nodes?.filter(n => n.sector).map(n => n.sector) || []);
      
      setFilterOptions({ nodeTypes, relationships, sectors });
      
      // Also fetch market overview
      const overviewResponse = await fetch(`${apiUrl}/api/ontology/market-overview`);
      const overview = await overviewResponse.json();
      setMarketOverview(overview);
      
    } catch (error) {
      console.error('Error fetching graph data:', error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  }, [apiUrl]);

  // Convert API data to Cytoscape format
  const convertToCytoscapeFormat = useCallback((data) => {
    if (!data.nodes || !data.links) return [];

    const nodes = data.nodes
      .filter(node => {
        if (activeFilters.nodeTypes.size > 0 && !activeFilters.nodeTypes.has(node.type)) return false;
        if (activeFilters.sectors.size > 0 && node.sector && !activeFilters.sectors.has(node.sector)) return false;
        return true;
      })
      .map(node => ({
        data: {
          id: node.id,
          label: node.name || node.symbol || node.id,
          type: node.type,
          level: node.level || currentLevel,
          size: Math.max(20, Math.min(80, (node.size || 30))),
          color: getNodeColor(node),
          sector: node.sector,
          price: node.price,
          market_cap: node.market_cap,
          heat_score: node.heat_score,
          strength: node.strength,
          confidence: node.confidence
        }
      }));

    const nodeIds = new Set(nodes.map(n => n.data.id));
    const edges = data.links
      .filter(link => {
        if (!nodeIds.has(link.source) || !nodeIds.has(link.target)) return false;
        if (activeFilters.relationships.size > 0 && !activeFilters.relationships.has(link.relationship)) return false;
        return true;
      })
      .map((link, index) => ({
        data: {
          id: `edge-${index}`,
          source: link.source,
          target: link.target,
          label: link.relationship?.replace(/_/g, ' '),
          relationship: link.relationship,
          strength: Math.max(1, Math.min(8, (link.strength || link.correlation || link.heat_flow || 1) * 5)),
          color: getEdgeColor(link),
          heat_flow: link.heat_flow,
          correlation: link.correlation
        }
      }));

    return [...nodes, ...edges];
  }, [activeFilters, currentLevel]);

  // Get node color based on ontology type and properties
  const getNodeColor = (node) => {
    if (node.heat_score !== undefined) {
      const heat = node.heat_score;
      if (heat > 0.8) return '#FF0033';
      if (heat > 0.6) return '#FF6600';
      if (heat > 0.4) return '#FFAA00';
      if (heat > 0.2) return '#00AAFF';
      return '#0066CC';
    }

    switch (node.type) {
      case 'Sector': return '#00D4FF';
      case 'Stock': return '#00FF88';
      case 'BuySignal': return '#00FF88';
      case 'SellSignal': return '#FF6B35';
      case 'HoldSignal': return '#FFA726';
      case 'TechnicalIndicator': return '#9966FF';
      case 'Corporation': return '#66CCFF';
      case 'Market': return '#FFD700';
      default: return '#CCCCCC';
    }
  };

  // Get edge color based on relationship type
  const getEdgeColor = (link) => {
    switch (link.relationship) {
      case 'belongs_to_sector':
      case 'same_sector': return '#00D4FF';
      case 'correlates_with': return '#FF6600';
      case 'generates_signal': return '#00FF88';
      case 'calculates_for': return '#9966FF';
      case 'heat_flow': return '#FF3366';
      default: return '#999999';
    }
  };

  // Initialize and render the graph
  const renderAdvancedGraph = useCallback(() => {
    if (!containerRef.current || !graphData.nodes?.length) return;

    // Destroy existing instance
    if (cyRef.current) {
      cyRef.current.destroy();
    }

    const elements = convertToCytoscapeFormat(graphData);
    
    // Initialize Cytoscape with professional configuration
    cyRef.current = cytoscape({
      container: containerRef.current,
      elements: elements,
      style: [
        {
          selector: 'node',
          style: getNodeStyle({})
        },
        {
          selector: 'edge',
          style: getEdgeStyle('')
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 6,
            'border-color': '#FFD700',
            'shadow-blur': 12,
            'shadow-color': '#FFD700',
            'shadow-opacity': 0.8
          }
        },
        {
          selector: 'edge:selected',
          style: {
            'width': '+=4',
            'line-color': '#FFD700',
            'target-arrow-color': '#FFD700',
            'shadow-blur': 8,
            'shadow-color': '#FFD700'
          }
        }
      ],
      layout: layoutConfigs[currentLayout],
      wheelSensitivity: 0.2,
      minZoom: 0.2,
      maxZoom: 3,
      zoomingEnabled: true,
      userZoomingEnabled: true,
      panningEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: true,
      selectionType: 'single',
      autoungrabify: false,
      autounselectify: false
    });

    // Advanced interaction handlers
    cyRef.current.on('tap', 'node', (event) => {
      const node = event.target;
      const nodeData = node.data();
      setSelectedNode({
        id: nodeData.id,
        label: nodeData.label,
        type: nodeData.type,
        properties: {
          sector: nodeData.sector,
          price: nodeData.price,
          market_cap: nodeData.market_cap,
          heat_score: nodeData.heat_score,
          strength: nodeData.strength,
          confidence: nodeData.confidence
        }
      });
      
      // Highlight connected elements
      const connectedEdges = node.connectedEdges();
      const connectedNodes = connectedEdges.connectedNodes();
      
      cyRef.current.elements().removeClass('highlighted dimmed');
      node.addClass('highlighted');
      connectedEdges.addClass('highlighted');
      connectedNodes.addClass('highlighted');
      cyRef.current.elements().not(node.union(connectedEdges).union(connectedNodes)).addClass('dimmed');
    });

    cyRef.current.on('tap', (event) => {
      if (event.target === cyRef.current) {
        setSelectedNode(null);
        cyRef.current.elements().removeClass('highlighted dimmed');
      }
    });

    // Professional hover effects
    cyRef.current.on('mouseover', 'node', (event) => {
      const node = event.target;
      node.style('transform', 'scale(1.2)');
      containerRef.current.style.cursor = 'pointer';
    });

    cyRef.current.on('mouseout', 'node', (event) => {
      const node = event.target;
      node.style('transform', 'scale(1)');
      containerRef.current.style.cursor = 'default';
    });

    // Fit graph to container
    setTimeout(() => {
      if (cyRef.current) {
        cyRef.current.fit(undefined, 50);
        cyRef.current.center();
      }
    }, 100);

  }, [graphData, currentLayout, convertToCytoscapeFormat]);

  // Handle layout change
  const changeLayout = (layoutName) => {
    setCurrentLayout(layoutName);
    if (cyRef.current) {
      const layout = cyRef.current.layout(layoutConfigs[layoutName]);
      layout.run();
    }
  };

  // Handle filter changes
  const toggleFilter = (filterType, value) => {
    setActiveFilters(prev => {
      const newFilters = { ...prev };
      if (newFilters[filterType].has(value)) {
        newFilters[filterType].delete(value);
      } else {
        newFilters[filterType].add(value);
      }
      return newFilters;
    });
  };

  // Clear all filters
  const clearFilters = () => {
    setActiveFilters({
      nodeTypes: new Set(),
      relationships: new Set(),
      sectors: new Set()
    });
  };

  // Export graph
  const exportGraph = (format = 'png') => {
    if (!cyRef.current) return;
    
    const options = {
      output: 'blob-promise',
      format: format,
      quality: 1.0,
      scale: 2,
      full: true,
      maxWidth: 4000,
      maxHeight: 4000
    };
    
    cyRef.current[format](options).then(blob => {
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.download = `ontology-graph-level-${currentLevel}.${format}`;
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);
    });
  };

  // Effects
  useEffect(() => {
    fetchGraphData(currentLevel);
  }, [currentLevel, fetchGraphData]);

  useEffect(() => {
    renderAdvancedGraph();
  }, [renderAdvancedGraph]);

  useEffect(() => {
    const handleResize = () => {
      if (cyRef.current) {
        cyRef.current.resize();
        cyRef.current.fit(undefined, 50);
      }
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const levelInfo = {
    1: { name: "Market Structure", icon: "🏛️" },
    2: { name: "Sector Classification", icon: "🏢" },
    3: { name: "Financial Instruments", icon: "📈" },
    4: { name: "Corporate Relations", icon: "🔗" },
    5: { name: "Trading Signals", icon: "📊" },
    6: { name: "Technical Indicators", icon: "📉" },
    7: { name: "Heat Propagation", icon: "🔥" }
  };

  return (
    <div className="advanced-ontology-graph">
      {/* Professional Control Panel */}
      <div className="control-panel">
        <div className="level-controls">
          <label>Ontology Level:</label>
          <select 
            value={currentLevel} 
            onChange={(e) => setCurrentLevel(parseInt(e.target.value))}
            className="level-selector"
          >
            {Object.entries(levelInfo).map(([level, info]) => (
              <option key={level} value={level}>
                {info.icon} Level {level}: {info.name}
              </option>
            ))}
          </select>
        </div>

        <div className="layout-controls">
          <label>Layout Algorithm:</label>
          <select 
            value={currentLayout} 
            onChange={(e) => changeLayout(e.target.value)}
            className="layout-selector"
          >
            <option value="cose-bilkent">🔬 COSE Bilkent (Best)</option>
            <option value="dagre">📊 Dagre Hierarchical</option>
            <option value="cola">🔗 COLA Force</option>
            <option value="euler">🌀 Euler Physics</option>
            <option value="spread">📡 Spread Layout</option>
          </select>
        </div>

        <div className="action-controls">
          <button onClick={() => exportGraph('png')} className="export-btn">
            📸 Export PNG
          </button>
          <button onClick={() => exportGraph('jpg')} className="export-btn">
            🖼️ Export JPG
          </button>
          <button onClick={clearFilters} className="clear-filters-btn">
            🗑️ Clear Filters
          </button>
        </div>
      </div>

      {/* Advanced Filter Panel */}
      <div className="filter-panel">
        <h4>🔍 Ontology Filters</h4>
        
        {filterOptions.nodeTypes.size > 0 && (
          <div className="filter-group">
            <label>Node Types:</label>
            <div className="filter-options">
              {Array.from(filterOptions.nodeTypes).map(type => (
                <label key={type} className="filter-checkbox">
                  <input
                    type="checkbox"
                    checked={activeFilters.nodeTypes.has(type)}
                    onChange={() => toggleFilter('nodeTypes', type)}
                  />
                  <span className={`filter-label type-${type.toLowerCase()}`}>
                    {type}
                  </span>
                </label>
              ))}
            </div>
          </div>
        )}

        {filterOptions.relationships.size > 0 && (
          <div className="filter-group">
            <label>Relationships:</label>
            <div className="filter-options">
              {Array.from(filterOptions.relationships).map(rel => (
                <label key={rel} className="filter-checkbox">
                  <input
                    type="checkbox"
                    checked={activeFilters.relationships.has(rel)}
                    onChange={() => toggleFilter('relationships', rel)}
                  />
                  <span className="filter-label">
                    {rel?.replace(/_/g, ' ')}
                  </span>
                </label>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Main Graph Container */}
      <div className="graph-container" ref={containerRef}>
        {isLoading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Loading ontology level {currentLevel}...</p>
          </div>
        )}
        
        {error && (
          <div className="error-overlay">
            <h3>⚠️ Error Loading Graph</h3>
            <p>{error}</p>
            <button onClick={() => fetchGraphData(currentLevel)}>Retry</button>
          </div>
        )}
      </div>

      {/* Node Details Panel */}
      {selectedNode && (
        <div className="node-details-panel">
          <div className="panel-header">
            <h4>{selectedNode.label}</h4>
            <button onClick={() => setSelectedNode(null)}>✕</button>
          </div>
          <div className="node-properties">
            <div className="property-row">
              <span>Type:</span>
              <span className={`node-type type-${selectedNode.type?.toLowerCase()}`}>
                {selectedNode.type}
              </span>
            </div>
            {Object.entries(selectedNode.properties || {}).map(([key, value]) => 
              value !== undefined && value !== null && (
                <div key={key} className="property-row">
                  <span>{key.replace(/_/g, ' ')}:</span>
                  <span className="property-value">
                    {typeof value === 'number' ? 
                      (value < 1 ? (value * 100).toFixed(1) + '%' : value.toLocaleString()) 
                      : value}
                  </span>
                </div>
              )
            )}
          </div>
        </div>
      )}

      {/* Market Overview Stats */}
      {marketOverview?.data && (
        <div className="market-overview-panel">
          <h4>📈 Market Overview</h4>
          <div className="market-stats">
            <div className="stat-item">
              <span className="stat-label">Total Stocks:</span>
              <span className="stat-value">{marketOverview.data.total_stocks}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Avg Heat:</span>
              <span className="stat-value heat-score">{marketOverview.data.average_heat}%</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Sentiment:</span>
              <span className={`stat-value sentiment-${marketOverview.data.market_sentiment?.toLowerCase().replace(' ', '-')}`}>
                {marketOverview.data.market_sentiment}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvancedOntologyGraph;