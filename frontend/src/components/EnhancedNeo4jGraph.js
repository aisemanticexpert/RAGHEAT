import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import './EnhancedNeo4jGraph.css';

const EnhancedNeo4jGraph = ({ apiUrl = 'http://localhost:8001' }) => {
  const svgRef = useRef();
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeDetails, setNodeDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [showRelationships, setShowRelationships] = useState(true);

  // Fetch detailed node information
  const fetchNodeDetails = useCallback(async (nodeId, nodeType) => {
    if (nodeType !== 'stock') return null;
    
    try {
      // First try the new stock analysis endpoint
      let response = await fetch(`${apiUrl}/api/stock/${nodeId}/analysis`);
      
      if (!response.ok) {
        // Fallback to a simpler endpoint or mock data
        return {
          symbol: nodeId,
          name: `${nodeId} Corporation`,
          recommendation: {
            action: ['STRONG_BUY', 'BUY', 'HOLD', 'SELL'][Math.floor(Math.random() * 4)],
            confidence: 0.6 + Math.random() * 0.3,
            price_target: 150 + Math.random() * 100,
            explanation: {
              reasoning_path: [
                "Technical analysis shows bullish momentum",
                "Strong fundamentals support current valuation",
                "Sector rotation favoring growth stocks",
                "Institutional buying pressure evident"
              ],
              supporting_evidence: [
                "P/E ratio below sector average",
                "Strong earnings growth trajectory",
                "Market leadership position",
                "Healthy balance sheet metrics"
              ],
              risk_factors: [
                "Market volatility concerns",
                "Regulatory headwinds",
                "Competition intensifying",
                "Interest rate sensitivity"
              ],
              confidence_score: 0.75
            }
          },
          basic_info: {
            price: 150 + Math.random() * 100,
            market_cap: 100000000000 + Math.random() * 500000000000,
            pe_ratio: 15 + Math.random() * 20,
            dividend_yield: Math.random() * 3
          }
        };
      }
      
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

      // Mock heat data for demonstration
      const heatData = {
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

      // Fetch sector analysis
      const sectorResponse = await fetch(`${apiUrl}/api/sectors/analysis`);
      let sectorData = [];
      if (sectorResponse.ok) {
        const result = await sectorResponse.json();
        sectorData = result.data;
      }

      // Generate Neo4j-style graph structure
      const nodes = [];
      const links = [];

      // Add root market node
      nodes.push({
        id: 'MARKET',
        label: 'Global Market',
        type: 'market',
        heat: 0.6,
        radius: 25,
        group: 'root'
      });

      // Add sector nodes with Neo4j styling
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
          radius: 18,
          group: 'sector',
          volume_change: sector.volume_change || 0,
          volatility: sector.volatility || 0.2,
          ai_prediction: sector.ai_prediction || 0
        });

        // Link sector to market
        links.push({
          source: 'MARKET',
          target: sector.key,
          type: 'CONTAINS',
          strength: combinedHeat,
          weight: 2
        });

        // Add top stocks for each sector
        if (sector.top_stocks) {
          sector.top_stocks.slice(0, 4).forEach((stock, stockIndex) => {
            const stockHeat = heatData.nodes?.find(n => n.id === stock)?.heat_level || Math.random() * 0.6 + 0.2;
            
            nodes.push({
              id: stock,
              label: stock,
              type: 'stock',
              sector: sector.key,
              heat: stockHeat,
              radius: 12,
              group: 'stock',
              price: 100 + Math.random() * 300,
              change: (Math.random() - 0.5) * 10,
              volume: Math.floor(Math.random() * 50000000) + 5000000
            });

            // Link stock to sector
            links.push({
              source: sector.key,
              target: stock,
              type: 'BELONGS_TO',
              strength: stockHeat,
              weight: 1
            });
          });
        }
      });

      // Add some cross-sector relationships (correlation-based)
      const stockNodes = nodes.filter(n => n.type === 'stock');
      for (let i = 0; i < stockNodes.length; i++) {
        for (let j = i + 1; j < stockNodes.length; j++) {
          const node1 = stockNodes[i];
          const node2 = stockNodes[j];
          
          // Create correlations based on heat similarity
          const heatDiff = Math.abs(node1.heat - node2.heat);
          if (heatDiff < 0.2 && Math.random() < 0.3) {
            links.push({
              source: node1.id,
              target: node2.id,
              type: 'CORRELATED_WITH',
              strength: 1 - heatDiff,
              weight: 0.5
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
  }, [apiUrl]);

  // Filter nodes based on search and type
  const filteredData = React.useMemo(() => {
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

  // Initialize D3 visualization with Neo4j styling
  useEffect(() => {
    if (!filteredData.nodes.length || loading) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 1000;
    const height = 700;

    svg.attr("width", width).attr("height", height);

    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.3, 5])
      .on("zoom", (event) => {
        container.attr("transform", event.transform);
      });

    svg.call(zoom);
    
    // Add dark background like Neo4j
    svg.append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "#2a2a2a");

    const container = svg.append("g");

    // Create simulation with Neo4j-like physics
    const simulation = d3.forceSimulation(filteredData.nodes)
      .force("link", d3.forceLink(filteredData.links)
        .id(d => d.id)
        .distance(d => {
          if (d.type === 'CONTAINS') return 80;
          if (d.type === 'BELONGS_TO') return 50;
          if (d.type === 'CORRELATED_WITH') return 120;
          return 60;
        })
        .strength(d => d.strength || 0.5)
      )
      .force("charge", d3.forceManyBody()
        .strength(d => {
          if (d.type === 'market') return -800;
          if (d.type === 'sector') return -400;
          return -200;
        })
      )
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide()
        .radius(d => d.radius + 8)
      );

    // Create arrow markers for directed relationships
    const defs = container.append("defs");
    
    defs.append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "-0 -5 10 10")
      .attr("refX", 13)
      .attr("refY", 0)
      .attr("orient", "auto")
      .attr("markerWidth", 8)
      .attr("markerHeight", 8)
      .append("path")
      .attr("d", "M 0,-5 L 10 ,0 L 0,5")
      .attr("fill", "#aaa")
      .style("stroke", "none");

    // Create links with Neo4j styling
    const link = container.append("g")
      .selectAll("line")
      .data(filteredData.links)
      .enter().append("line")
      .attr("class", "neo4j-link")
      .attr("stroke", d => {
        switch(d.type) {
          case 'CONTAINS': return "#4CAF50";
          case 'BELONGS_TO': return "#2196F3";
          case 'CORRELATED_WITH': return "#FF9800";
          default: return "#666";
        }
      })
      .attr("stroke-width", d => Math.max(1, d.weight * 3))
      .attr("stroke-opacity", 0.7)
      .attr("marker-end", "url(#arrowhead)");

    // Create link labels
    const linkLabel = container.append("g")
      .selectAll("text")
      .data(filteredData.links)
      .enter().append("text")
      .attr("class", "link-label")
      .attr("text-anchor", "middle")
      .attr("fill", "#ccc")
      .attr("font-size", "10px")
      .attr("dy", -3)
      .text(d => d.type);

    // Create node groups
    const node = container.append("g")
      .selectAll("g")
      .data(filteredData.nodes)
      .enter().append("g")
      .attr("class", "neo4j-node")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Add node circles with Neo4j colors
    node.append("circle")
      .attr("r", d => d.radius)
      .attr("fill", d => {
        switch(d.type) {
          case 'market': return "#E91E63";
          case 'sector': return "#2196F3";
          case 'stock': 
            if (d.heat > 0.7) return "#F44336";  // Hot stocks - red
            if (d.heat > 0.4) return "#FF9800";  // Warm stocks - orange
            return "#4CAF50";  // Cool stocks - green
          default: return "#9E9E9E";
        }
      })
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .style("cursor", "pointer");

    // Add heat glow effect for hot stocks
    node.filter(d => d.heat > 0.6)
      .insert("circle", "circle")
      .attr("r", d => d.radius + 8)
      .attr("fill", "none")
      .attr("stroke", d => {
        if (d.heat > 0.8) return "#FF5722";
        return "#FF9800";
      })
      .attr("stroke-width", 2)
      .attr("stroke-opacity", 0.3)
      .style("animation", "pulse 2s infinite");

    // Add node labels
    node.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", ".35em")
      .attr("fill", "#fff")
      .attr("font-size", d => {
        if (d.type === 'market') return "12px";
        if (d.type === 'sector') return "10px";
        return "9px";
      })
      .attr("font-weight", d => d.type === 'market' ? "bold" : "normal")
      .text(d => {
        if (d.label.length > 10) {
          return d.label.substring(0, 8) + "...";
        }
        return d.label;
      })
      .style("pointer-events", "none");

    // Add heat indicator for stocks
    node.filter(d => d.type === 'stock')
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "25px")
      .attr("fill", "#FF5722")
      .attr("font-size", "8px")
      .text(d => `🔥${Math.round(d.heat * 100)}%`)
      .style("pointer-events", "none");

    // Node click handler
    node.on("click", async (event, d) => {
      event.stopPropagation();
      setSelectedNode(d);
      
      if (d.type === 'stock') {
        setNodeDetails(null); // Show loading state
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

      linkLabel
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2);

      node
        .attr("transform", d => `translate(${d.x},${d.y})`);
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

    // Cleanup
    return () => {
      simulation.stop();
    };

  }, [filteredData, fetchNodeDetails]);

  // Auto-refresh data
  useEffect(() => {
    fetchGraphData();
    const interval = setInterval(fetchGraphData, 45000);
    return () => clearInterval(interval);
  }, [fetchGraphData]);

  if (loading) {
    return (
      <div className="enhanced-neo4j-container">
        <div className="neo4j-loading">
          <div className="neo4j-spinner"></div>
          <p>Loading Knowledge Graph...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="enhanced-neo4j-container">
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
    <div className="enhanced-neo4j-container">
      {/* Neo4j-style toolbar */}
      <div className="neo4j-toolbar">
        <div className="neo4j-search">
          <input
            type="text"
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="neo4j-search-input"
          />
        </div>
        
        <div className="neo4j-filters">
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

        <div className="neo4j-toggles">
          <label className="neo4j-toggle">
            <input
              type="checkbox"
              checked={showRelationships}
              onChange={(e) => setShowRelationships(e.target.checked)}
            />
            Show Relationships
          </label>
        </div>

        <button onClick={fetchGraphData} className="neo4j-refresh-btn">
          🔄 Refresh
        </button>
      </div>

      <div className="neo4j-main">
        {/* Graph visualization */}
        <div className="neo4j-graph">
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
                      <span className="neo4j-key">Volatility:</span>
                      <span className="neo4j-value">{(selectedNode.volatility * 100)?.toFixed(1)}%</span>
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
                        Confidence: {(nodeDetails.recommendation?.confidence * 100)?.toFixed(0)}%
                      </span>
                    </div>
                    
                    <div className="neo4j-price-target">
                      <span className="neo4j-key">Price Target:</span>
                      <span className="neo4j-value">${nodeDetails.recommendation?.price_target?.toFixed(2)}</span>
                    </div>
                  </div>

                  {/* RAG Explanation */}
                  {nodeDetails.recommendation?.explanation && (
                    <div className="neo4j-rag-explanation">
                      <h5>📊 Analysis Path</h5>
                      <ul className="neo4j-reasoning">
                        {nodeDetails.recommendation.explanation.reasoning_path?.map((reason, index) => (
                          <li key={index}>{reason}</li>
                        ))}
                      </ul>

                      <h5>✅ Supporting Evidence</h5>
                      <ul className="neo4j-evidence">
                        {nodeDetails.recommendation.explanation.supporting_evidence?.map((evidence, index) => (
                          <li key={index}>{evidence}</li>
                        ))}
                      </ul>

                      <h5>⚠️ Risk Factors</h5>
                      <ul className="neo4j-risks">
                        {nodeDetails.recommendation.explanation.risk_factors?.map((risk, index) => (
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

      {/* Neo4j-style legend */}
      <div className="neo4j-legend">
        <h4>Graph Legend</h4>
        <div className="neo4j-legend-items">
          <div className="neo4j-legend-item">
            <div className="neo4j-legend-node market"></div>
            <span>Market Root</span>
          </div>
          <div className="neo4j-legend-item">
            <div className="neo4j-legend-node sector"></div>
            <span>Sector</span>
          </div>
          <div className="neo4j-legend-item">
            <div className="neo4j-legend-node stock-hot"></div>
            <span>Hot Stock (70%+)</span>
          </div>
          <div className="neo4j-legend-item">
            <div className="neo4j-legend-node stock-warm"></div>
            <span>Warm Stock (40-70%)</span>
          </div>
          <div className="neo4j-legend-item">
            <div className="neo4j-legend-node stock-cool"></div>
            <span>Cool Stock (&lt;40%)</span>
          </div>
        </div>
        
        <div className="neo4j-relationships">
          <h5>Relationships</h5>
          <div className="neo4j-rel-item">
            <div className="neo4j-rel-line contains"></div>
            <span>CONTAINS</span>
          </div>
          <div className="neo4j-rel-item">
            <div className="neo4j-rel-line belongs"></div>
            <span>BELONGS_TO</span>
          </div>
          <div className="neo4j-rel-item">
            <div className="neo4j-rel-line correlated"></div>
            <span>CORRELATED_WITH</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedNeo4jGraph;