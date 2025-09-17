import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import * as d3 from 'd3';
import './ProfessionalNeo4jGraph.css';

const ProfessionalNeo4jGraph = ({ apiUrl = 'http://localhost:8001' }) => {
  const svgRef = useRef();
  const containerRef = useRef();
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeDetails, setNodeDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [showRelationships, setShowRelationships] = useState(true);
  const [expandedNodes, setExpandedNodes] = useState(new Set());
  const [zoomLevel, setZoomLevel] = useState(1);
  const [simulationRunning, setSimulationRunning] = useState(true);

  // Professional Neo4j color scheme
  const colorScheme = {
    market: '#6366F1',      // Royal Purple
    sector: '#3B82F6',      // Royal Blue
    technology: '#8B5CF6',  // Purple for tech stocks
    financial: '#10B981',   // Emerald for financial
    healthcare: '#F59E0B',  // Amber for healthcare
    energy: '#EF4444',      // Red for energy
    industrial: '#6B7280',  // Gray for industrial
    consumer: '#EC4899',    // Pink for consumer
    utilities: '#14B8A6',   // Teal for utilities
    materials: '#F97316',   // Orange for materials
    default: '#94A3B8',     // Slate for unknown
    hot: '#DC2626',         // Bright red for hot stocks
    warm: '#F59E0B',        // Orange for warm
    cool: '#22C55E',        // Green for cool
    edge: {
      contains: '#6366F1',
      belongsTo: '#3B82F6',
      correlatedWith: '#8B5CF6',
      default: '#64748B'
    }
  };

  // Node embedding simulation for clustering
  const generateNodeEmbedding = (node) => {
    const embeddings = {
      'Technology': [0.8, 0.2, 0.9, 0.1],
      'Financial': [0.2, 0.8, 0.1, 0.9],
      'Healthcare': [0.5, 0.6, 0.8, 0.3],
      'Energy': [0.9, 0.1, 0.2, 0.8],
      'Industrial': [0.4, 0.7, 0.3, 0.6],
      'Consumer_Discretionary': [0.7, 0.3, 0.6, 0.4],
      'Communication': [0.6, 0.4, 0.7, 0.5],
      'Consumer_Staples': [0.3, 0.6, 0.4, 0.7],
      'Utilities': [0.1, 0.9, 0.2, 0.8],
      'Real_Estate': [0.5, 0.5, 0.5, 0.5]
    };
    
    if (node.type === 'stock') {
      return embeddings[node.sector] || [0.5, 0.5, 0.5, 0.5];
    }
    return embeddings[node.id] || [0.5, 0.5, 0.5, 0.5];
  };

  // Calculate node color based on embedding similarity
  const getNodeColor = (node) => {
    if (node.type === 'market') return colorScheme.market;
    
    if (node.type === 'sector') {
      return colorScheme[node.id.toLowerCase()] || colorScheme.sector;
    }
    
    if (node.type === 'stock') {
      const sectorColor = colorScheme[node.sector?.toLowerCase()] || colorScheme.default;
      
      // Adjust brightness based on heat
      if (node.heat > 0.7) return colorScheme.hot;
      if (node.heat > 0.4) return colorScheme.warm;
      if (node.heat < 0.3) return colorScheme.cool;
      
      return sectorColor;
    }
    
    return colorScheme.default;
  };

  // Fetch detailed node information
  const fetchNodeDetails = useCallback(async (nodeId, nodeType) => {
    if (nodeType !== 'stock') return null;
    
    try {
      const response = await fetch(`${apiUrl}/api/stock/${nodeId}/analysis`);
      if (!response.ok) throw new Error('Failed to fetch');
      
      const result = await response.json();
      return result.data;
    } catch (error) {
      console.error('Error fetching node details:', error);
      return null;
    }
  }, [apiUrl]);

  // Fetch graph data
  const fetchGraphData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Mock heat data with embeddings
      const heatData = {
        nodes: [
          { id: 'AAPL', heat_level: 0.85, embedding: [0.8, 0.2, 0.9, 0.1] },
          { id: 'MSFT', heat_level: 0.72, embedding: [0.75, 0.25, 0.85, 0.15] },
          { id: 'NVDA', heat_level: 0.93, embedding: [0.9, 0.1, 0.95, 0.05] },
          { id: 'GOOGL', heat_level: 0.68, embedding: [0.7, 0.3, 0.8, 0.2] },
          { id: 'TSLA', heat_level: 0.89, embedding: [0.6, 0.4, 0.7, 0.3] },
          { id: 'AMZN', heat_level: 0.76, embedding: [0.65, 0.35, 0.75, 0.25] },
          { id: 'META', heat_level: 0.64, embedding: [0.55, 0.45, 0.65, 0.35] },
          { id: 'JPM', heat_level: 0.42, embedding: [0.2, 0.8, 0.1, 0.9] },
          { id: 'JNJ', heat_level: 0.38, embedding: [0.5, 0.6, 0.8, 0.3] },
          { id: 'XOM', heat_level: 0.55, embedding: [0.9, 0.1, 0.2, 0.8] }
        ]
      };

      // Fetch sector analysis
      const sectorResponse = await fetch(`${apiUrl}/api/sectors/analysis`);
      let sectorData = [];
      if (sectorResponse.ok) {
        const result = await sectorResponse.json();
        sectorData = result.data;
      }

      // Generate professional graph structure
      const nodes = [];
      const links = [];

      // Add root market node
      nodes.push({
        id: 'MARKET',
        label: 'Global Market',
        type: 'market',
        heat: 0.6,
        radius: 35,
        group: 'root',
        embedding: [0.5, 0.5, 0.5, 0.5],
        expanded: true,
        level: 0,
        strength: 1.0
      });

      // Add sector nodes
      sectorData.forEach((sector, index) => {
        const baseHeat = Math.abs(sector.performance || 0) / 10;
        const volatilityHeat = (sector.volatility || 0.2) * 2;
        const volumeHeat = Math.abs(sector.volume_change || 0) / 20;
        const combinedHeat = Math.min(0.9, Math.max(0.1, (baseHeat + volatilityHeat + volumeHeat) / 3));

        nodes.push({
          id: sector.key,
          label: sector.name,
          type: 'sector',
          heat: combinedHeat,
          performance: sector.performance || 0,
          radius: 25,
          group: 'sector',
          volume_change: sector.volume_change || 0,
          volatility: sector.volatility || 0.2,
          ai_prediction: sector.ai_prediction || 0,
          embedding: generateNodeEmbedding({ type: 'sector', id: sector.key }),
          expanded: false,
          level: 1,
          strength: combinedHeat,
          stockCount: sector.top_stocks?.length || 0
        });

        // Link sector to market
        links.push({
          source: 'MARKET',
          target: sector.key,
          type: 'CONTAINS',
          strength: combinedHeat,
          weight: 3,
          value: `${sector.top_stocks?.length || 0} stocks`,
          color: colorScheme.edge.contains
        });

        // Add stocks for expanded sectors or high-heat sectors
        if (expandedNodes.has(sector.key) || combinedHeat > 0.5) {
          sector.top_stocks?.slice(0, 6).forEach((stock, stockIndex) => {
            const stockHeat = heatData.nodes?.find(n => n.id === stock)?.heat_level || Math.random() * 0.6 + 0.2;
            const stockEmbedding = heatData.nodes?.find(n => n.id === stock)?.embedding || generateNodeEmbedding({ type: 'stock', sector: sector.key });
            
            nodes.push({
              id: stock,
              label: stock,
              type: 'stock',
              sector: sector.key,
              heat: stockHeat,
              radius: Math.max(8, 12 + stockHeat * 8),
              group: 'stock',
              price: 100 + Math.random() * 300,
              change: (Math.random() - 0.5) * 10,
              volume: Math.floor(Math.random() * 50000000) + 5000000,
              embedding: stockEmbedding,
              expanded: false,
              level: 2,
              strength: stockHeat
            });

            // Link stock to sector
            links.push({
              source: sector.key,
              target: stock,
              type: 'BELONGS_TO',
              strength: stockHeat,
              weight: 2,
              value: `${(stockHeat * 100).toFixed(0)}% heat`,
              color: colorScheme.edge.belongsTo
            });
          });
        }
      });

      // Add correlation links between similar stocks
      const stockNodes = nodes.filter(n => n.type === 'stock');
      for (let i = 0; i < stockNodes.length; i++) {
        for (let j = i + 1; j < stockNodes.length; j++) {
          const node1 = stockNodes[i];
          const node2 = stockNodes[j];
          
          // Calculate embedding similarity
          const similarity = calculateEmbeddingSimilarity(node1.embedding, node2.embedding);
          
          if (similarity > 0.8 && Math.random() < 0.3) {
            const correlationValue = similarity.toFixed(2);
            links.push({
              source: node1.id,
              target: node2.id,
              type: 'CORRELATED_WITH',
              strength: similarity,
              weight: 1,
              value: `${(similarity * 100).toFixed(0)}% corr`,
              color: colorScheme.edge.correlatedWith
            });
          }
        }
      }

      setGraphData({ nodes, links });
      setLoading(false);
    } catch (err) {
      console.error('Error fetching graph data:', err);
      setError(err.message);
      setLoading(false);
    }
  }, [apiUrl, expandedNodes]);

  // Calculate embedding similarity (cosine similarity)
  const calculateEmbeddingSimilarity = (emb1, emb2) => {
    const dotProduct = emb1.reduce((sum, a, i) => sum + a * emb2[i], 0);
    const magnitude1 = Math.sqrt(emb1.reduce((sum, a) => sum + a * a, 0));
    const magnitude2 = Math.sqrt(emb2.reduce((sum, a) => sum + a * a, 0));
    return dotProduct / (magnitude1 * magnitude2);
  };

  // Toggle node expansion
  const toggleNodeExpansion = (nodeId) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  };

  // Filter nodes based on search and type
  const filteredData = useMemo(() => {
    let filteredNodes = graphData.nodes;
    let filteredLinks = graphData.links;

    // Apply search filter
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      filteredNodes = filteredNodes.filter(node =>
        node.label.toLowerCase().includes(searchLower) ||
        node.id.toLowerCase().includes(searchLower)
      );
      
      const nodeIds = new Set(filteredNodes.map(n => n.id));
      filteredLinks = filteredLinks.filter(link =>
        nodeIds.has(link.source.id || link.source) &&
        nodeIds.has(link.target.id || link.target)
      );
    }

    // Apply type filter
    if (filterType !== 'all') {
      filteredNodes = filteredNodes.filter(node => node.type === filterType);
      
      const nodeIds = new Set(filteredNodes.map(n => n.id));
      filteredLinks = filteredLinks.filter(link =>
        nodeIds.has(link.source.id || link.source) &&
        nodeIds.has(link.target.id || link.target)
      );
    }

    return { nodes: filteredNodes, links: showRelationships ? filteredLinks : [] };
  }, [graphData, searchTerm, filterType, showRelationships]);

  // Initialize D3 visualization with professional Neo4j styling
  useEffect(() => {
    if (!filteredData.nodes.length || loading) return;

    const svg = d3.select(svgRef.current);
    const container = d3.select(containerRef.current);
    svg.selectAll("*").remove();

    const width = container.node()?.getBoundingClientRect().width || 1200;
    const height = container.node()?.getBoundingClientRect().height || 800;

    svg.attr("width", width).attr("height", height);

    // Create zoom behavior
    const zoomBehavior = d3.zoom()
      .scaleExtent([0.1, 10])
      .on("zoom", (event) => {
        mainGroup.attr("transform", event.transform);
        setZoomLevel(event.transform.k);
      });

    svg.call(zoomBehavior);
    
    // Add professional gradient background
    const defs = svg.append("defs");
    
    const bgGradient = defs.append("radialGradient")
      .attr("id", "bgGradient")
      .attr("cx", "50%")
      .attr("cy", "50%");
    
    bgGradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#1E293B");
    
    bgGradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#0F172A");

    svg.append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "url(#bgGradient)");

    const mainGroup = svg.append("g");

    // Create arrow markers for relationships
    const arrowMarker = defs.append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "-0 -5 10 10")
      .attr("refX", 20)
      .attr("refY", 0)
      .attr("orient", "auto")
      .attr("markerWidth", 8)
      .attr("markerHeight", 8);
    
    arrowMarker.append("path")
      .attr("d", "M 0,-5 L 10 ,0 L 0,5")
      .attr("fill", "#64748B")
      .style("stroke", "none");

    // Create simulation
    const simulation = d3.forceSimulation(filteredData.nodes)
      .force("link", d3.forceLink(filteredData.links)
        .id(d => d.id)
        .distance(d => {
          if (d.type === 'CONTAINS') return 100;
          if (d.type === 'BELONGS_TO') return 60;
          if (d.type === 'CORRELATED_WITH') return 150;
          return 80;
        })
        .strength(d => d.strength * 0.5)
      )
      .force("charge", d3.forceManyBody()
        .strength(d => {
          if (d.type === 'market') return -2000;
          if (d.type === 'sector') return -800;
          return -300;
        })
      )
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide()
        .radius(d => d.radius + 10)
      );

    // Create links with professional styling
    const link = mainGroup.append("g")
      .selectAll("line")
      .data(filteredData.links)
      .enter().append("line")
      .attr("class", "neo4j-edge")
      .attr("stroke", d => d.color)
      .attr("stroke-width", d => Math.max(1, d.weight))
      .attr("stroke-opacity", 0.8)
      .attr("marker-end", "url(#arrowhead)");

    // Create edge labels with values
    const edgeLabel = mainGroup.append("g")
      .selectAll("text")
      .data(filteredData.links)
      .enter().append("text")
      .attr("class", "edge-label")
      .attr("text-anchor", "middle")
      .attr("fill", "#E2E8F0")
      .attr("font-size", "10px")
      .attr("font-family", "Monaco, monospace")
      .attr("dy", -3)
      .style("opacity", d => zoomLevel > 1.5 ? 1 : 0)
      .text(d => d.value);

    // Create node groups
    const node = mainGroup.append("g")
      .selectAll("g")
      .data(filteredData.nodes)
      .enter().append("g")
      .attr("class", "neo4j-node")
      .style("cursor", "pointer")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Add node backgrounds (glow effect for important nodes)
    node.filter(d => d.type === 'market' || d.heat > 0.7)
      .append("circle")
      .attr("r", d => d.radius + 8)
      .attr("fill", d => getNodeColor(d))
      .attr("opacity", 0.3)
      .style("filter", "blur(4px)");

    // Add main node circles
    node.append("circle")
      .attr("r", d => d.radius)
      .attr("fill", d => getNodeColor(d))
      .attr("stroke", "#FFFFFF")
      .attr("stroke-width", 2)
      .style("filter", "drop-shadow(2px 2px 4px rgba(0,0,0,0.5))")
      .style("transition", "all 0.3s ease");

    // Add expansion indicators for expandable nodes
    node.filter(d => d.type === 'sector' && d.stockCount > 0)
      .append("circle")
      .attr("r", 6)
      .attr("cx", d => d.radius - 8)
      .attr("cy", d => -d.radius + 8)
      .attr("fill", "#10B981")
      .attr("stroke", "#FFFFFF")
      .attr("stroke-width", 1)
      .style("cursor", "pointer");

    // Add expansion icons
    node.filter(d => d.type === 'sector' && d.stockCount > 0)
      .append("text")
      .attr("x", d => d.radius - 8)
      .attr("y", d => -d.radius + 8)
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("fill", "#FFFFFF")
      .attr("font-size", "8px")
      .attr("font-weight", "bold")
      .text(d => expandedNodes.has(d.id) ? "−" : "+")
      .style("pointer-events", "none");

    // Add node labels (dynamic based on zoom)
    node.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", d => d.type === 'stock' ? "0.35em" : "0.5em")
      .attr("fill", "#FFFFFF")
      .attr("font-size", d => {
        if (d.type === 'market') return "14px";
        if (d.type === 'sector') return "12px";
        return "10px";
      })
      .attr("font-weight", d => d.type === 'market' ? "bold" : "normal")
      .attr("font-family", "Inter, sans-serif")
      .style("text-shadow", "1px 1px 2px rgba(0,0,0,0.8)")
      .style("opacity", d => {
        if (zoomLevel < 0.5) return d.type === 'market' ? 1 : 0;
        if (zoomLevel < 1.0) return d.type !== 'stock' ? 1 : 0;
        return 1;
      })
      .text(d => {
        const maxLength = zoomLevel > 1.5 ? 15 : 8;
        if (d.label.length > maxLength) {
          return d.label.substring(0, maxLength - 2) + "...";
        }
        return d.label;
      })
      .style("pointer-events", "none");

    // Add heat indicators for stocks
    node.filter(d => d.type === 'stock' && zoomLevel > 1.0)
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "25px")
      .attr("fill", "#F59E0B")
      .attr("font-size", "8px")
      .attr("font-family", "Monaco, monospace")
      .text(d => `🔥${Math.round(d.heat * 100)}%`)
      .style("pointer-events", "none");

    // Node click handlers
    node.on("click", async (event, d) => {
      event.stopPropagation();
      
      // Handle expansion for sectors
      if (d.type === 'sector' && d.stockCount > 0) {
        const clickX = event.offsetX;
        const clickY = event.offsetY;
        const nodeX = d.x;
        const nodeY = d.y;
        const expansionButtonX = nodeX + d.radius - 8;
        const expansionButtonY = nodeY - d.radius + 8;
        
        // Check if click is on expansion button
        const distance = Math.sqrt(Math.pow(clickX - expansionButtonX, 2) + Math.pow(clickY - expansionButtonY, 2));
        if (distance < 10) {
          toggleNodeExpansion(d.id);
          return;
        }
      }
      
      setSelectedNode(d);
      
      if (d.type === 'stock') {
        setNodeDetails(null);
        const details = await fetchNodeDetails(d.id, d.type);
        setNodeDetails(details);
      } else {
        setNodeDetails(null);
      }
    });

    // Node hover effects
    node.on("mouseover", function(event, d) {
      d3.select(this).select("circle")
        .transition()
        .duration(200)
        .attr("r", d.radius * 1.2)
        .attr("stroke-width", 3);
    })
    .on("mouseout", function(event, d) {
      d3.select(this).select("circle")
        .transition()
        .duration(200)
        .attr("r", d.radius)
        .attr("stroke-width", 2);
    });

    // Update positions on simulation tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      edgeLabel
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2);

      node
        .attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // Zoom control functions
    window.zoomIn = () => {
      svg.transition().duration(300).call(
        zoomBehavior.scaleBy, 1.5
      );
    };

    window.zoomOut = () => {
      svg.transition().duration(300).call(
        zoomBehavior.scaleBy, 1 / 1.5
      );
    };

    window.fitToScreen = () => {
      const bounds = mainGroup.node().getBBox();
      const fullWidth = width;
      const fullHeight = height;
      const widthScale = fullWidth / bounds.width;
      const heightScale = fullHeight / bounds.height;
      const scale = Math.min(widthScale, heightScale) * 0.8;
      const centerX = bounds.x + bounds.width / 2;
      const centerY = bounds.y + bounds.height / 2;
      
      svg.transition().duration(750).call(
        zoomBehavior.transform,
        d3.zoomIdentity
          .translate(fullWidth / 2, fullHeight / 2)
          .scale(scale)
          .translate(-centerX, -centerY)
      );
    };

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

    // Control simulation
    if (!simulationRunning) {
      simulation.stop();
    }

    // Cleanup
    return () => {
      simulation.stop();
    };

  }, [filteredData, fetchNodeDetails, expandedNodes, zoomLevel, simulationRunning]);

  // Auto-refresh data when expanded nodes change
  useEffect(() => {
    fetchGraphData();
  }, [fetchGraphData]);

  if (loading) {
    return (
      <div className="professional-neo4j-container">
        <div className="neo4j-loading">
          <div className="neo4j-spinner"></div>
          <p>Loading Professional Graph...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="professional-neo4j-container">
        <div className="neo4j-error">
          <h3>Error loading graph</h3>
          <p>{error}</p>
          <button onClick={fetchGraphData} className="neo4j-retry-btn">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="professional-neo4j-container">
      {/* Professional Neo4j-style toolbar */}
      <div className="neo4j-professional-toolbar">
        <div className="neo4j-search-section">
          <input
            type="text"
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="neo4j-search-input"
          />
        </div>
        
        <div className="neo4j-filter-section">
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="neo4j-select"
          >
            <option value="all">All Nodes</option>
            <option value="market">Market</option>
            <option value="sector">Sectors</option>
            <option value="stock">Stocks</option>
          </select>
        </div>

        <div className="neo4j-control-section">
          <button
            onClick={() => window.zoomIn()}
            className="neo4j-zoom-btn"
            title="Zoom In"
          >
            🔍+
          </button>
          <button
            onClick={() => window.zoomOut()}
            className="neo4j-zoom-btn"
            title="Zoom Out"
          >
            🔍−
          </button>
          <button
            onClick={() => window.fitToScreen()}
            className="neo4j-zoom-btn"
            title="Fit to Screen"
          >
            ⊞
          </button>
        </div>

        <div className="neo4j-toggle-section">
          <label className="neo4j-toggle">
            <input
              type="checkbox"
              checked={showRelationships}
              onChange={(e) => setShowRelationships(e.target.checked)}
            />
            Relationships
          </label>
          <label className="neo4j-toggle">
            <input
              type="checkbox"
              checked={simulationRunning}
              onChange={(e) => setSimulationRunning(e.target.checked)}
            />
            Physics
          </label>
        </div>

        <div className="neo4j-info-section">
          <span className="zoom-indicator">Zoom: {(zoomLevel * 100).toFixed(0)}%</span>
          <span className="node-count">{filteredData.nodes.length} nodes</span>
        </div>
      </div>

      <div className="neo4j-main-content">
        {/* Graph visualization */}
        <div className="neo4j-graph-container" ref={containerRef}>
          <svg ref={svgRef}></svg>
        </div>

        {/* Node details panel */}
        {selectedNode && (
          <div className="neo4j-details-panel">
            <div className="neo4j-details-header">
              <h3>{selectedNode.label}</h3>
              <button
                onClick={() => {
                  setSelectedNode(null);
                  setNodeDetails(null);
                }}
                className="neo4j-close-btn"
              >
                ×
              </button>
            </div>

            <div className="neo4j-details-content">
              {/* Basic node info */}
              <div className="neo4j-node-info">
                <div className="neo4j-property">
                  <span className="neo4j-key">Type:</span>
                  <span className="neo4j-value">{selectedNode.type}</span>
                </div>
                <div className="neo4j-property">
                  <span className="neo4j-key">Heat Level:</span>
                  <span className="neo4j-value">{(selectedNode.heat * 100).toFixed(1)}%</span>
                </div>
                
                {selectedNode.type === 'stock' && (
                  <>
                    <div className="neo4j-property">
                      <span className="neo4j-key">Sector:</span>
                      <span className="neo4j-value">{selectedNode.sector}</span>
                    </div>
                    <div className="neo4j-property">
                      <span className="neo4j-key">Price:</span>
                      <span className="neo4j-value">${selectedNode.price?.toFixed(2)}</span>
                    </div>
                    <div className="neo4j-property">
                      <span className="neo4j-key">Change:</span>
                      <span className={`neo4j-value ${selectedNode.change > 0 ? 'positive' : 'negative'}`}>
                        {selectedNode.change > 0 ? '+' : ''}{selectedNode.change?.toFixed(2)}%
                      </span>
                    </div>
                  </>
                )}

                {selectedNode.type === 'sector' && (
                  <>
                    <div className="neo4j-property">
                      <span className="neo4j-key">Performance:</span>
                      <span className={`neo4j-value ${selectedNode.performance > 0 ? 'positive' : 'negative'}`}>
                        {selectedNode.performance?.toFixed(2)}%
                      </span>
                    </div>
                    <div className="neo4j-property">
                      <span className="neo4j-key">Stock Count:</span>
                      <span className="neo4j-value">{selectedNode.stockCount}</span>
                    </div>
                    <div className="neo4j-property">
                      <span className="neo4j-key">Expanded:</span>
                      <span className="neo4j-value">{expandedNodes.has(selectedNode.id) ? 'Yes' : 'No'}</span>
                    </div>
                  </>
                )}
              </div>

              {/* Stock analysis details */}
              {selectedNode.type === 'stock' && nodeDetails && (
                <div className="neo4j-analysis">
                  <h4>Investment Analysis</h4>
                  
                  <div className="neo4j-recommendation">
                    <div className="neo4j-action">
                      <span className={`neo4j-action-badge ${nodeDetails.recommendation?.action.toLowerCase()}`}>
                        {nodeDetails.recommendation?.action}
                      </span>
                      <span className="neo4j-confidence">
                        {(nodeDetails.recommendation?.confidence * 100)?.toFixed(0)}%
                      </span>
                    </div>
                    
                    <div className="neo4j-price-target">
                      <span className="neo4j-key">Target:</span>
                      <span className="neo4j-value">${nodeDetails.recommendation?.price_target?.toFixed(2)}</span>
                    </div>
                  </div>

                  {/* RAG Explanation */}
                  {nodeDetails.recommendation?.explanation && (
                    <div className="neo4j-rag-explanation">
                      <h5>📊 Analysis Path</h5>
                      <ul className="neo4j-reasoning">
                        {nodeDetails.recommendation.explanation.reasoning_path?.slice(0, 3).map((reason, index) => (
                          <li key={index}>{reason}</li>
                        ))}
                      </ul>

                      <h5>✅ Key Evidence</h5>
                      <ul className="neo4j-evidence">
                        {nodeDetails.recommendation.explanation.supporting_evidence?.slice(0, 2).map((evidence, index) => (
                          <li key={index}>{evidence}</li>
                        ))}
                      </ul>

                      <h5>⚠️ Risks</h5>
                      <ul className="neo4j-risks">
                        {nodeDetails.recommendation.explanation.risk_factors?.slice(0, 2).map((risk, index) => (
                          <li key={index}>{risk}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {selectedNode.type === 'stock' && !nodeDetails && (
                <div className="neo4j-loading-details">
                  <div className="neo4j-spinner-small"></div>
                  <p>Loading analysis...</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Professional legend */}
      <div className="neo4j-professional-legend">
        <h4>Graph Legend</h4>
        <div className="neo4j-legend-section">
          <h5>Node Types</h5>
          <div className="neo4j-legend-items">
            <div className="neo4j-legend-item">
              <div className="neo4j-legend-node" style={{backgroundColor: colorScheme.market}}></div>
              <span>Market Root</span>
            </div>
            <div className="neo4j-legend-item">
              <div className="neo4j-legend-node" style={{backgroundColor: colorScheme.sector}}></div>
              <span>Sector</span>
            </div>
            <div className="neo4j-legend-item">
              <div className="neo4j-legend-node" style={{backgroundColor: colorScheme.hot}}></div>
              <span>Hot Stock (70%+)</span>
            </div>
            <div className="neo4j-legend-item">
              <div className="neo4j-legend-node" style={{backgroundColor: colorScheme.warm}}></div>
              <span>Warm Stock (40-70%)</span>
            </div>
            <div className="neo4j-legend-item">
              <div className="neo4j-legend-node" style={{backgroundColor: colorScheme.cool}}></div>
              <span>Cool Stock (&lt;40%)</span>
            </div>
          </div>
        </div>
        
        <div className="neo4j-legend-section">
          <h5>Relationships</h5>
          <div className="neo4j-legend-items">
            <div className="neo4j-legend-item">
              <div className="neo4j-rel-line" style={{backgroundColor: colorScheme.edge.contains}}></div>
              <span>CONTAINS</span>
            </div>
            <div className="neo4j-legend-item">
              <div className="neo4j-rel-line" style={{backgroundColor: colorScheme.edge.belongsTo}}></div>
              <span>BELONGS_TO</span>
            </div>
            <div className="neo4j-legend-item">
              <div className="neo4j-rel-line" style={{backgroundColor: colorScheme.edge.correlatedWith}}></div>
              <span>CORRELATED_WITH</span>
            </div>
          </div>
        </div>

        <div className="neo4j-instructions">
          <h5>Controls</h5>
          <ul>
            <li>Click + on sectors to expand/collapse</li>
            <li>Drag nodes to reposition</li>
            <li>Use zoom controls for navigation</li>
            <li>Click stocks for analysis</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ProfessionalNeo4jGraph;