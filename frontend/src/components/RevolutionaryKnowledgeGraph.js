/**
 * Revolutionary Knowledge Graph Visualization
 * Advanced Sigma.js WebGL rendering with physics-inspired heat propagation
 * Professional financial platform interface
 */
import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
// For now, let's create a simplified version without Sigma.js to avoid module resolution issues
// import { Sigma } from 'sigma/dist/sigma.esm.js';
import Graph from 'graphology';
import { circular, random } from 'graphology-layout';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import louvain from 'graphology-communities-louvain';
import * as d3 from 'd3';
import io from 'socket.io-client';
import './RevolutionaryKnowledgeGraph.css';

const RevolutionaryKnowledgeGraph = ({ apiUrl = 'http://localhost:8001' }) => {
  // Core refs and state
  const containerRef = useRef(null);
  const sigmaRef = useRef(null);
  const graphRef = useRef(new Graph());
  const socketRef = useRef(null);
  const animationRef = useRef(null);
  
  // UI State
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [graphStats, setGraphStats] = useState({ nodes: 0, edges: 0 });
  const [heatPulses, setHeatPulses] = useState([]);
  
  // Controls State
  const [layoutAlgorithm, setLayoutAlgorithm] = useState('forceAtlas2');
  const [showHeatMap, setShowHeatMap] = useState(true);
  const [showSectorClusters, setShowSectorClusters] = useState(true);
  const [nodeFilterMin, setNodeFilterMin] = useState(0);
  const [edgeFilterMin, setEdgeFilterMin] = useState(0);
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Physics and Heat State
  const [heatSources, setHeatSources] = useState(new Set());
  const [temperatureField, setTemperatureField] = useState(new Map());
  const [isSimulating, setIsSimulating] = useState(false);
  
  // Advanced color scales for heat visualization
  const heatColorScale = useMemo(() => 
    d3.scaleSequential(d3.interpolateInferno).domain([0, 1]), []);
    
  const sectorColorScale = useMemo(() => ({
    'Technology': '#6366F1',           // Indigo
    'Financial': '#10B981',            // Emerald
    'Healthcare': '#EF4444',           // Red
    'Consumer': '#F59E0B',             // Amber
    'Energy': '#8B5CF6',               // Purple
    'Industrials': '#6B7280',          // Gray
    'Materials': '#F97316',            // Orange
    'Utilities': '#14B8A6',            // Teal
    'Real_Estate': '#84CC16',          // Lime
    'Communication': '#3B82F6'         // Blue
  }), []);

  // Initialize Sigma.js with WebGL renderer
  const initializeSigma = useCallback(() => {
    if (!containerRef.current) return;

    try {
      // Advanced Sigma configuration
      const sigmaInstance = new Sigma(graphRef.current, containerRef.current, {
        // WebGL renderer for high performance
        renderer: {
          type: 'webgl',
          antialias: true,
          preserveDrawingBuffer: true
        },
        
        // Advanced settings for large graphs
        settings: {
          // Node rendering
          defaultNodeType: 'circle',
          nodeProgramClasses: {
            circle: require('sigma/rendering/webgl/programs/node.circle').default,
            'bordered-circle': require('sigma/rendering/webgl/programs/node.bordered-circle').default
          },
          
          // Edge rendering  
          defaultEdgeType: 'line',
          edgeProgramClasses: {
            line: require('sigma/rendering/webgl/programs/edge.line').default,
            arrow: require('sigma/rendering/webgl/programs/edge.arrow').default
          },
          
          // Performance optimizations
          hideEdgesOnMove: true,
          hideLabelsOnMove: true,
          renderLabels: true,
          renderEdgeLabels: false,
          
          // Label settings
          labelFont: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
          labelSize: 12,
          labelWeight: '500',
          labelColor: { color: '#FFFFFF' },
          
          // Mouse/touch interactions
          enableEdgeClickEvents: true,
          enableEdgeWheelEvents: false,
          enableEdgeHoverEvents: true,
          
          // Animation settings
          animationsTime: 300,
          
          // Zoom settings
          minCameraRatio: 0.1,
          maxCameraRatio: 10,
          
          // Stage settings  
          stagePadding: 20,
          
          // Advanced rendering
          zIndex: true,
          
          // Node sizing
          nodeBorderSize: 2,
          defaultNodeBorderColor: '#FFFFFF',
          
          // Edge styling
          defaultEdgeColor: '#94A3B8',
          edgeReducer: null,
          
          // Heat map specific
          nodeReducer: (node, data) => {
            const temperature = temperatureField.get(node) || 0;
            const sector = data.sector || 'Other';
            
            return {
              ...data,
              color: showHeatMap ? 
                heatColorScale(temperature) : 
                sectorColorScale[sector] || '#6B7280',
              size: Math.max(8, Math.min(40, 15 + temperature * 25)),
              borderColor: temperature > 0.7 ? '#FBBF24' : '#FFFFFF',
              borderWidth: temperature > 0.7 ? 3 : 1,
              zIndex: Math.floor(temperature * 10)
            };
          }
        }
      });
      
      // Set up event handlers
      setupSigmaEventHandlers(sigmaInstance);
      
      sigmaRef.current = sigmaInstance;
      setLoading(false);
      
      console.log('🚀 Revolutionary Sigma.js initialized with WebGL rendering');
      
    } catch (error) {
      console.error('❌ Failed to initialize Sigma.js:', error);
      setError(`Failed to initialize visualization: ${error.message}`);
      setLoading(false);
    }
  }, [heatColorScale, sectorColorScale, temperatureField, showHeatMap]);

  // Setup comprehensive event handlers
  const setupSigmaEventHandlers = useCallback((sigmaInstance) => {
    // Node interaction events
    sigmaInstance.on('clickNode', (event) => {
      const nodeId = event.node;
      const nodeData = graphRef.current.getNodeAttributes(nodeId);
      setSelectedNode({ id: nodeId, ...nodeData });
      
      // Highlight connected nodes
      highlightConnectedNodes(nodeId);
      
      // Create heat pulse from selected node
      if (showHeatMap) {
        createHeatPulse(nodeId);
      }
    });

    sigmaInstance.on('enterNode', (event) => {
      const nodeId = event.node;
      const nodeData = graphRef.current.getNodeAttributes(nodeId);
      setHoveredNode({ id: nodeId, ...nodeData });
      
      // Update cursor
      containerRef.current.style.cursor = 'pointer';
      
      // Temporary highlight
      graphRef.current.setNodeAttribute(nodeId, 'highlighted', true);
      sigmaInstance.refresh();
    });

    sigmaInstance.on('leaveNode', (event) => {
      const nodeId = event.node;
      setHoveredNode(null);
      
      // Reset cursor
      containerRef.current.style.cursor = 'default';
      
      // Remove temporary highlight
      graphRef.current.setNodeAttribute(nodeId, 'highlighted', false);
      sigmaInstance.refresh();
    });

    // Stage events
    sigmaInstance.on('clickStage', () => {
      setSelectedNode(null);
      clearHighlights();
    });
    
    // Camera events for performance optimization
    sigmaInstance.getCamera().on('updated', () => {
      const ratio = sigmaInstance.getCamera().ratio;
      
      // Adaptive LOD (Level of Detail)
      if (ratio > 2) {
        // Far zoom - hide labels, simplify rendering
        sigmaInstance.setSetting('renderLabels', false);
        sigmaInstance.setSetting('hideEdgesOnMove', true);
      } else {
        // Close zoom - show details
        sigmaInstance.setSetting('renderLabels', true);
        sigmaInstance.setSetting('hideEdgesOnMove', false);
      }
    });
    
  }, [showHeatMap]);

  // Advanced node highlighting system
  const highlightConnectedNodes = useCallback((centralNodeId) => {
    const graph = graphRef.current;
    const sigma = sigmaRef.current;
    
    if (!graph || !sigma) return;
    
    // Clear previous highlights
    clearHighlights();
    
    // Get connected nodes
    const connectedNodes = new Set([centralNodeId]);
    const connectedEdges = new Set();
    
    // First-degree connections
    graph.forEachNeighbor(centralNodeId, (neighborId) => {
      connectedNodes.add(neighborId);
    });
    
    // Get connecting edges
    graph.forEachEdge(centralNodeId, (edgeId, attributes, source, target) => {
      if (connectedNodes.has(source) && connectedNodes.has(target)) {
        connectedEdges.add(edgeId);
      }
    });
    
    // Apply highlighting
    connectedNodes.forEach(nodeId => {
      graph.setNodeAttribute(nodeId, 'highlighted', true);
      if (nodeId === centralNodeId) {
        graph.setNodeAttribute(nodeId, 'selected', true);
      }
    });
    
    connectedEdges.forEach(edgeId => {
      graph.setEdgeAttribute(edgeId, 'highlighted', true);
    });
    
    // Dim non-connected elements
    graph.forEachNode((nodeId, attributes) => {
      if (!connectedNodes.has(nodeId)) {
        graph.setNodeAttribute(nodeId, 'dimmed', true);
      }
    });
    
    graph.forEachEdge((edgeId, attributes) => {
      if (!connectedEdges.has(edgeId)) {
        graph.setEdgeAttribute(edgeId, 'dimmed', true);
      }
    });
    
    sigma.refresh();
  }, []);

  // Clear all highlights
  const clearHighlights = useCallback(() => {
    const graph = graphRef.current;
    const sigma = sigmaRef.current;
    
    if (!graph || !sigma) return;
    
    graph.forEachNode((nodeId) => {
      graph.removeNodeAttribute(nodeId, 'highlighted');
      graph.removeNodeAttribute(nodeId, 'selected');
      graph.removeNodeAttribute(nodeId, 'dimmed');
    });
    
    graph.forEachEdge((edgeId) => {
      graph.removeEdgeAttribute(edgeId, 'highlighted');
      graph.removeEdgeAttribute(edgeId, 'dimmed');
    });
    
    sigma.refresh();
  }, []);

  // Advanced heat pulse visualization
  const createHeatPulse = useCallback((sourceNodeId) => {
    const graph = graphRef.current;
    const sigma = sigmaRef.current;
    
    if (!graph || !sigma || !sourceNodeId) return;
    
    const sourceTemp = temperatureField.get(sourceNodeId) || 0;
    if (sourceTemp < 0.1) return; // Only pulse from hot nodes
    
    // Create expanding heat wave
    const pulseId = `pulse_${Date.now()}`;
    const connectedNodes = [];
    
    graph.forEachNeighbor(sourceNodeId, (nodeId) => {
      connectedNodes.push(nodeId);
    });
    
    // Animate heat propagation
    const startTime = Date.now();
    const duration = 2000; // 2 seconds
    
    const animateHeatPulse = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      if (progress < 1) {
        // Update connected node temperatures
        connectedNodes.forEach((nodeId, index) => {
          const delay = index * 100; // Stagger the pulse
          if (elapsed > delay) {
            const nodeProgress = Math.min((elapsed - delay) / (duration - delay), 1);
            const heatTransfer = sourceTemp * 0.3 * (1 - nodeProgress);
            
            const currentTemp = temperatureField.get(nodeId) || 0;
            const newTemp = Math.min(1, currentTemp + heatTransfer * 0.1);
            
            temperatureField.set(nodeId, newTemp);
            
            // Visual pulse effect
            const pulseIntensity = Math.sin(nodeProgress * Math.PI);
            graph.setNodeAttribute(nodeId, 'pulse', pulseIntensity);
          }
        });
        
        sigma.refresh();
        requestAnimationFrame(animateHeatPulse);
      } else {
        // Clean up pulse effects
        connectedNodes.forEach(nodeId => {
          graph.removeNodeAttribute(nodeId, 'pulse');
        });
        sigma.refresh();
        
        setHeatPulses(prev => prev.filter(p => p.id !== pulseId));
      }
    };
    
    // Record pulse for tracking
    const pulse = {
      id: pulseId,
      source: sourceNodeId,
      targets: connectedNodes,
      startTime,
      duration
    };
    
    setHeatPulses(prev => [...prev, pulse]);
    requestAnimationFrame(animateHeatPulse);
    
  }, [temperatureField]);

  // Advanced layout algorithms
  const applyLayout = useCallback(async (algorithm) => {
    const graph = graphRef.current;
    const sigma = sigmaRef.current;
    
    if (!graph || !sigma) return;
    
    setLoading(true);
    
    try {
      switch (algorithm) {
        case 'forceAtlas2':
          // Advanced ForceAtlas2 with sector-based attraction
          const fa2Settings = {
            iterations: 300,
            settings: {
              barnesHutOptimize: true,
              barnesHutTheta: 0.5,
              edgeWeightInfluence: 1,
              scalingRatio: 2,
              strongGravityMode: false,
              gravity: 0.05,
              slowDown: 2,
              linLogMode: false,
              outboundAttractionDistribution: true,
              adjustSizes: false
            }
          };
          
          forceAtlas2.assign(graph, fa2Settings);
          break;
          
        case 'physics_heat':
          // Custom physics-based layout using heat equation
          await applyPhysicsHeatLayout(graph);
          break;
          
        case 'sector_clusters':
          // Sector-based clustering layout
          await applySectorClusterLayout(graph);
          break;
          
        case 'circular_sectors':
          // Circular layout with sector grouping
          await applyCircularSectorLayout(graph);
          break;
          
        case 'force_directed':
          // Custom force-directed with heat influences
          await applyHeatInfluencedForceLayout(graph);
          break;
          
        default:
          circular.assign(graph);
      }
      
      // Apply noverlap to prevent node overlapping
      noverlap.assign(graph, {
        maxIterations: 150,
        settings: {
          margin: 5,
          expansion: 1.1,
          gridSize: 20,
          speed: 3
        }
      });
      
      sigma.refresh();
      setLayoutAlgorithm(algorithm);
      
    } catch (error) {
      console.error(`❌ Layout ${algorithm} failed:`, error);
      setError(`Layout failed: ${error.message}`);
    }
    
    setLoading(false);
  }, []);

  // Custom physics-based heat layout
  const applyPhysicsHeatLayout = useCallback(async (graph) => {
    const nodes = graph.nodes();
    const nodePositions = new Map();
    
    // Initialize positions based on sector and temperature
    nodes.forEach(nodeId => {
      const attrs = graph.getNodeAttributes(nodeId);
      const temp = temperatureField.get(nodeId) || 0;
      const sector = attrs.sector || 'Other';
      
      // Sector-based base positioning
      const sectorAngle = getSectorAngle(sector);
      const sectorRadius = 200 + temp * 100; // Heat influences distance from center
      
      // Add some randomness
      const angle = sectorAngle + (Math.random() - 0.5) * 0.5;
      const radius = sectorRadius * (0.8 + Math.random() * 0.4);
      
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      
      nodePositions.set(nodeId, { x, y });
      graph.setNodeAttribute(nodeId, 'x', x);
      graph.setNodeAttribute(nodeId, 'y', y);
    });
    
    // Simulate heat diffusion for positioning
    for (let iteration = 0; iteration < 100; iteration++) {
      const forces = new Map();
      
      // Initialize forces
      nodes.forEach(nodeId => {
        forces.set(nodeId, { fx: 0, fy: 0 });
      });
      
      // Heat-based repulsion/attraction
      nodes.forEach(nodeId1 => {
        const pos1 = nodePositions.get(nodeId1);
        const temp1 = temperatureField.get(nodeId1) || 0;
        
        nodes.forEach(nodeId2 => {
          if (nodeId1 === nodeId2) return;
          
          const pos2 = nodePositions.get(nodeId2);
          const temp2 = temperatureField.get(nodeId2) || 0;
          
          const dx = pos2.x - pos1.x;
          const dy = pos2.y - pos1.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance < 1) return; // Avoid division by zero
          
          // Heat-based force calculation
          const tempDiff = temp1 - temp2;
          const forceMagnitude = tempDiff * 10 / (distance * distance);
          
          const force1 = forces.get(nodeId1);
          force1.fx += (dx / distance) * forceMagnitude;
          force1.fy += (dy / distance) * forceMagnitude;
        });
      });
      
      // Apply forces
      nodes.forEach(nodeId => {
        const pos = nodePositions.get(nodeId);
        const force = forces.get(nodeId);
        
        pos.x += force.fx * 0.1;
        pos.y += force.fy * 0.1;
        
        graph.setNodeAttribute(nodeId, 'x', pos.x);
        graph.setNodeAttribute(nodeId, 'y', pos.y);
      });
    }
    
  }, [temperatureField]);

  // Sector-based clustering layout
  const applySectorClusterLayout = useCallback(async (graph) => {
    // Detect communities using Louvain algorithm
    const communities = louvain(graph);
    
    // Position nodes based on community and sector
    const sectorCenters = {};
    const sectorNodes = {};
    
    // Group nodes by sector
    graph.forEachNode((nodeId, attrs) => {
      const sector = attrs.sector || 'Other';
      if (!sectorNodes[sector]) {
        sectorNodes[sector] = [];
        sectorCenters[sector] = {
          x: Math.cos(getSectorAngle(sector)) * 300,
          y: Math.sin(getSectorAngle(sector)) * 300
        };
      }
      sectorNodes[sector].push(nodeId);
    });
    
    // Position nodes within each sector cluster
    Object.entries(sectorNodes).forEach(([sector, nodes]) => {
      const center = sectorCenters[sector];
      const clusterRadius = Math.sqrt(nodes.length) * 20;
      
      nodes.forEach((nodeId, index) => {
        const angle = (index / nodes.length) * 2 * Math.PI;
        const radius = clusterRadius * (0.5 + Math.random() * 0.5);
        
        const x = center.x + Math.cos(angle) * radius;
        const y = center.y + Math.sin(angle) * radius;
        
        graph.setNodeAttribute(nodeId, 'x', x);
        graph.setNodeAttribute(nodeId, 'y', y);
      });
    });
  }, []);

  // Helper function to get sector angle
  const getSectorAngle = useCallback((sector) => {
    const sectorAngles = {
      'Technology': 0,
      'Financial': Math.PI / 4,
      'Healthcare': Math.PI / 2,
      'Consumer': 3 * Math.PI / 4,
      'Energy': Math.PI,
      'Industrials': 5 * Math.PI / 4,
      'Materials': 3 * Math.PI / 2,
      'Utilities': 7 * Math.PI / 4
    };
    
    return sectorAngles[sector] || Math.random() * 2 * Math.PI;
  }, []);

  // Load graph data with advanced processing
  const loadGraphData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${apiUrl}/api/graph/synthetic-data`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error('API returned unsuccessful response');
      }
      
      await processGraphData(data.data);
      
    } catch (error) {
      console.error('❌ Failed to load graph data:', error);
      setError(`Failed to load data: ${error.message}`);
    }
    
    setLoading(false);
  }, [apiUrl]);

  // Advanced graph data processing
  const processGraphData = useCallback(async (data) => {
    const graph = graphRef.current;
    
    if (!data.nodes || !data.relationships) {
      throw new Error('Invalid graph data structure');
    }
    
    // Clear existing graph
    graph.clear();
    
    // Process and add nodes with advanced attributes
    const processedNodes = data.nodes.map(node => ({
      ...node,
      // Normalize coordinates to viewport
      x: (node.x || Math.random()) * 800 - 400,
      y: (node.y || Math.random()) * 600 - 300,
      
      // Enhanced visual properties
      size: Math.max(8, Math.min(40, (node.size || 15))),
      temperature: node.heat_level / 100,
      sector: node.type.replace('_', ' ').split(' ').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
      
      // Color based on heat or sector
      color: showHeatMap ? 
        heatColorScale(node.heat_level / 100) : 
        sectorColorScale[node.type] || '#6B7280',
      
      // Labels
      label: node.label || node.id,
      
      // Z-index for layering
      zIndex: Math.floor((node.heat_level || 0) / 10)
    }));
    
    // Add nodes to graph
    processedNodes.forEach(node => {
      graph.addNode(node.id, node);
      
      // Update temperature field
      setTemperatureField(prev => {
        const newField = new Map(prev);
        newField.set(node.id, node.temperature);
        return newField;
      });
    });
    
    // Process and add edges with advanced styling
    if (data.relationships) {
      data.relationships.forEach(edge => {
        if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
          const edgeId = `${edge.source}-${edge.target}`;
          
          graph.addEdge(edgeId, edge.source, edge.target, {
            size: Math.max(0.5, Math.min(6, (edge.width || 1))),
            color: edge.color || '#94A3B8',
            type: edge.style === 'dashed' ? 'dashed' : 'line',
            weight: edge.correlation || edge.weight || 1,
            label: edge.type || '',
            opacity: edge.opacity || 0.6,
            
            // Heat conductance for physics
            conductance: Math.abs(edge.correlation || 0.5),
            
            // Z-index
            zIndex: 1
          });
        }
      });
    }
    
    // Update graph statistics
    setGraphStats({
      nodes: graph.order,
      edges: graph.size
    });
    
    // Apply initial layout
    await applyLayout(layoutAlgorithm);
    
    console.log(`🎯 Processed graph: ${graph.order} nodes, ${graph.size} edges`);
    
  }, [showHeatMap, heatColorScale, sectorColorScale, layoutAlgorithm, applyLayout]);

  // Real-time updates via WebSocket
  const setupRealTimeUpdates = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
    }
    
    const socket = io(apiUrl);
    socketRef.current = socket;
    
    socket.on('connect', () => {
      console.log('🔌 WebSocket connected for real-time updates');
    });
    
    socket.on('graph_update', (data) => {
      // Apply incremental updates
      updateGraphRealTime(data);
    });
    
    socket.on('heat_pulse', (pulseData) => {
      // Handle real-time heat pulses
      if (pulseData.source) {
        createHeatPulse(pulseData.source);
      }
    });
    
    socket.on('market_event', (event) => {
      // Handle market events that affect heat
      if (event.symbol && event.heat_impact > 0.1) {
        updateNodeTemperature(event.symbol, event.heat_impact);
      }
    });
    
    socket.on('disconnect', () => {
      console.log('🔌 WebSocket disconnected');
    });
    
  }, [apiUrl]);

  // Real-time graph updates
  const updateGraphRealTime = useCallback((data) => {
    const graph = graphRef.current;
    const sigma = sigmaRef.current;
    
    if (!graph || !sigma) return;
    
    // Update node temperatures
    if (data.nodes) {
      data.nodes.forEach(nodeUpdate => {
        if (graph.hasNode(nodeUpdate.id)) {
          const temperature = nodeUpdate.heat_level / 100;
          
          // Update temperature field
          setTemperatureField(prev => {
            const newField = new Map(prev);
            newField.set(nodeUpdate.id, temperature);
            return newField;
          });
          
          // Update visual properties
          const color = showHeatMap ? 
            heatColorScale(temperature) : 
            sectorColorScale[nodeUpdate.type] || '#6B7280';
            
          graph.mergeNodeAttributes(nodeUpdate.id, {
            color,
            size: Math.max(8, Math.min(40, 15 + temperature * 25)),
            temperature
          });
        }
      });
      
      sigma.refresh();
    }
  }, [showHeatMap, heatColorScale, sectorColorScale]);

  // Update individual node temperature
  const updateNodeTemperature = useCallback((nodeId, heatImpact) => {
    const graph = graphRef.current;
    const sigma = sigmaRef.current;
    
    if (!graph || !sigma || !graph.hasNode(nodeId)) return;
    
    const currentTemp = temperatureField.get(nodeId) || 0;
    const newTemp = Math.min(1, currentTemp + heatImpact * 0.1);
    
    setTemperatureField(prev => {
      const newField = new Map(prev);
      newField.set(nodeId, newTemp);
      return newField;
    });
    
    // Visual update
    const color = showHeatMap ? heatColorScale(newTemp) : graph.getNodeAttribute(nodeId, 'color');
    
    graph.mergeNodeAttributes(nodeId, {
      color,
      size: Math.max(8, Math.min(40, 15 + newTemp * 25)),
      temperature: newTemp
    });
    
    sigma.refresh();
    
    // Create heat pulse effect
    createHeatPulse(nodeId);
    
  }, [temperatureField, showHeatMap, heatColorScale, createHeatPulse]);

  // Search functionality
  const searchNodes = useCallback((query) => {
    const graph = graphRef.current;
    const sigma = sigmaRef.current;
    
    if (!graph || !sigma || !query) {
      clearHighlights();
      return;
    }
    
    const matchedNodes = [];
    const queryLower = query.toLowerCase();
    
    graph.forEachNode((nodeId, attributes) => {
      const label = attributes.label?.toLowerCase() || '';
      const sector = attributes.sector?.toLowerCase() || '';
      
      if (label.includes(queryLower) || sector.includes(queryLower) || nodeId.toLowerCase().includes(queryLower)) {
        matchedNodes.push(nodeId);
      }
    });
    
    if (matchedNodes.length > 0) {
      // Highlight matched nodes
      clearHighlights();
      
      matchedNodes.forEach(nodeId => {
        graph.setNodeAttribute(nodeId, 'highlighted', true);
      });
      
      // Dim non-matched nodes
      graph.forEachNode((nodeId, attributes) => {
        if (!matchedNodes.includes(nodeId)) {
          graph.setNodeAttribute(nodeId, 'dimmed', true);
        }
      });
      
      sigma.refresh();
      
      // Focus on first match
      if (matchedNodes.length === 1) {
        const nodeId = matchedNodes[0];
        const nodeAttrs = graph.getNodeAttributes(nodeId);
        sigma.getCamera().animate({ x: nodeAttrs.x, y: nodeAttrs.y, ratio: 0.5 }, { duration: 500 });
      }
    }
    
    setSearchQuery(query);
  }, []);

  // Initialize everything
  useEffect(() => {
    initializeSigma();
    loadGraphData();
    setupRealTimeUpdates();
    
    // Cleanup
    return () => {
      if (sigmaRef.current) {
        sigmaRef.current.kill();
      }
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [initializeSigma, loadGraphData, setupRealTimeUpdates]);

  // Auto-refresh data
  useEffect(() => {
    const interval = setInterval(() => {
      if (!loading && !error) {
        loadGraphData();
      }
    }, 5000); // Refresh every 5 seconds
    
    return () => clearInterval(interval);
  }, [loading, error, loadGraphData]);

  // Heat simulation toggle
  const toggleHeatSimulation = useCallback(() => {
    setIsSimulating(!isSimulating);
    
    if (!isSimulating) {
      // Start continuous heat simulation
      const simulateHeat = () => {
        if (!isSimulating) return;
        
        // Cool down all nodes slightly
        setTemperatureField(prev => {
          const newField = new Map();
          prev.forEach((temp, nodeId) => {
            newField.set(nodeId, Math.max(0, temp * 0.995)); // Gradual cooling
          });
          return newField;
        });
        
        // Add random heat sources
        const graph = graphRef.current;
        if (graph && Math.random() < 0.1) { // 10% chance per frame
          const nodes = graph.nodes();
          const randomNode = nodes[Math.floor(Math.random() * nodes.length)];
          updateNodeTemperature(randomNode, Math.random() * 0.5);
        }
        
        animationRef.current = requestAnimationFrame(simulateHeat);
      };
      
      animationRef.current = requestAnimationFrame(simulateHeat);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }
  }, [isSimulating, updateNodeTemperature]);

  return (
    <div className="revolutionary-knowledge-graph">
      {/* Professional Header */}
      <div className="graph-header">
        <div className="header-title">
          <h2>🔥 Revolutionary Knowledge Graph</h2>
          <span className="physics-badge">Physics-Informed Neural Network</span>
        </div>
        
        <div className="graph-stats">
          <div className="stat-item">
            <span className="stat-label">Nodes</span>
            <span className="stat-value">{graphStats.nodes}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Connections</span>
            <span className="stat-value">{graphStats.edges}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Heat Pulses</span>
            <span className="stat-value">{heatPulses.length}</span>
          </div>
        </div>
      </div>

      {/* Advanced Controls Panel */}
      <div className="graph-controls">
        {/* Search Bar */}
        <div className="control-group">
          <input
            type="text"
            placeholder="Search nodes, sectors, or symbols..."
            value={searchQuery}
            onChange={(e) => searchNodes(e.target.value)}
            className="search-input"
          />
        </div>

        {/* Layout Controls */}
        <div className="control-group">
          <label>Layout Algorithm:</label>
          <select 
            value={layoutAlgorithm} 
            onChange={(e) => applyLayout(e.target.value)}
            className="layout-select"
          >
            <option value="forceAtlas2">ForceAtlas2 (Recommended)</option>
            <option value="physics_heat">Physics Heat Diffusion</option>
            <option value="sector_clusters">Sector Clustering</option>
            <option value="circular_sectors">Circular Sectors</option>
            <option value="force_directed">Heat-Influenced Force</option>
          </select>
        </div>

        {/* Visualization Toggles */}
        <div className="control-group">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showHeatMap}
              onChange={(e) => setShowHeatMap(e.target.checked)}
            />
            Heat Map Visualization
          </label>
          
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showSectorClusters}
              onChange={(e) => setShowSectorClusters(e.target.checked)}
            />
            Sector Clustering
          </label>
        </div>

        {/* Heat Simulation */}
        <div className="control-group">
          <button 
            className={`simulation-button ${isSimulating ? 'active' : ''}`}
            onClick={toggleHeatSimulation}
          >
            {isSimulating ? '⏸️ Pause Heat Simulation' : '▶️ Start Heat Simulation'}
          </button>
        </div>

        {/* Filters */}
        <div className="control-group">
          <label>Node Filter (Min Heat): {nodeFilterMin}%</label>
          <input
            type="range"
            min="0"
            max="100"
            value={nodeFilterMin}
            onChange={(e) => setNodeFilterMin(parseInt(e.target.value))}
            className="slider"
          />
          
          <label>Edge Filter (Min Weight): {edgeFilterMin}</label>
          <input
            type="range"
            min="0"
            max="100"
            value={edgeFilterMin}
            onChange={(e) => setEdgeFilterMin(parseInt(e.target.value))}
            className="slider"
          />
        </div>
      </div>

      {/* Main Visualization Container */}
      <div className="graph-container">
        <div ref={containerRef} className="sigma-container" />
        
        {/* Loading Overlay */}
        {loading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <div className="loading-text">Loading Revolutionary Graph...</div>
          </div>
        )}
        
        {/* Error Overlay */}
        {error && (
          <div className="error-overlay">
            <div className="error-message">
              <h3>⚠️ Error</h3>
              <p>{error}</p>
              <button onClick={() => {setError(null); loadGraphData();}}>
                Retry
              </button>
            </div>
          </div>
        )}
        
        {/* Node Info Panel */}
        {selectedNode && (
          <div className="node-info-panel">
            <div className="panel-header">
              <h3>{selectedNode.label || selectedNode.id}</h3>
              <button onClick={() => setSelectedNode(null)}>×</button>
            </div>
            <div className="panel-content">
              <div className="info-item">
                <span className="info-label">Sector:</span>
                <span className="info-value">{selectedNode.sector}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Temperature:</span>
                <span className="info-value">
                  {((selectedNode.temperature || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="info-item">
                <span className="info-label">Heat Level:</span>
                <div className="heat-bar">
                  <div 
                    className="heat-fill" 
                    style={{width: `${(selectedNode.temperature || 0) * 100}%`}}
                  ></div>
                </div>
              </div>
              {selectedNode.price && (
                <div className="info-item">
                  <span className="info-label">Price:</span>
                  <span className="info-value">${selectedNode.price}</span>
                </div>
              )}
              {selectedNode.volume && (
                <div className="info-item">
                  <span className="info-label">Volume:</span>
                  <span className="info-value">{selectedNode.volume.toLocaleString()}</span>
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Heat Pulse Indicators */}
        {heatPulses.length > 0 && (
          <div className="heat-pulse-indicator">
            <div className="pulse-icon">🔥</div>
            <div className="pulse-count">{heatPulses.length}</div>
          </div>
        )}
      </div>

      {/* Professional Legend */}
      <div className="graph-legend">
        <div className="legend-section">
          <h4>Heat Scale</h4>
          <div className="heat-scale">
            {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map(temp => (
              <div key={temp} className="heat-sample">
                <div 
                  className="heat-color" 
                  style={{backgroundColor: heatColorScale(temp)}}
                ></div>
                <span>{(temp * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
        
        <div className="legend-section">
          <h4>Sectors</h4>
          <div className="sector-legend">
            {Object.entries(sectorColorScale).map(([sector, color]) => (
              <div key={sector} className="sector-item">
                <div className="sector-color" style={{backgroundColor: color}}></div>
                <span>{sector.replace('_', ' ')}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RevolutionaryKnowledgeGraph;