/**
 * Advanced Trading Dashboard
 * Revolutionary UI for the RAGHeat trading system with viral heat propagation
 * 
 * Features:
 * - Real-time heat propagation visualization
 * - Interactive knowledge graph
 * - Advanced hierarchical sector analysis
 * - Machine learning prediction displays
 * - Option pricing predictions
 * - Live signal streaming
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell, PieChart, Pie, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  Treemap, Sankey
} from 'recharts';
import { 
  TrendingUp, TrendingDown, Activity, Target, Brain,
  Network, Zap, Eye, Settings, RefreshCw, Play, Pause,
  AlertTriangle, CheckCircle, Info, Star, Flame
} from 'lucide-react';
import * as d3 from 'd3';
import './AdvancedTradingDashboard.css';

// Simple React component replacements for UI elements
const Card = ({ children, className = '' }) => <div className={`card ${className}`}>{children}</div>;
const CardContent = ({ children, className = '' }) => <div className={`card-content ${className}`}>{children}</div>;
const CardHeader = ({ children, className = '' }) => <div className={`card-header ${className}`}>{children}</div>;
const CardTitle = ({ children, className = '' }) => <h3 className={`card-title ${className}`}>{children}</h3>;
const Badge = ({ children, variant = '', className = '' }) => <span className={`badge ${variant} ${className}`}>{children}</span>;
const Button = ({ children, onClick, className = '' }) => <button onClick={onClick} className={`button ${className}`}>{children}</button>;
const Progress = ({ value, className = '' }) => <div className={`progress ${className}`}><div className="progress-bar" style={{width: `${value}%`}}></div></div>;
const Avatar = ({ children, className = '' }) => <div className={`avatar ${className}`}>{children}</div>;
const AvatarFallback = ({ children }) => <span>{children}</span>;
const Tabs = ({ children, value, onValueChange, className = '' }) => (
  <div className={`tabs ${className}`} data-value={value}>
    {React.Children.map(children, child => 
      React.cloneElement(child, { activeValue: value, onValueChange })
    )}
  </div>
);
const TabsList = ({ children, className = '', activeValue, onValueChange }) => (
  <div className={`tabs-list ${className}`}>
    {React.Children.map(children, child => 
      React.cloneElement(child, { activeValue, onValueChange })
    )}
  </div>
);
const TabsTrigger = ({ children, value, className = '', activeValue, onValueChange }) => (
  <button 
    className={`tabs-trigger ${className}`} 
    data-state={activeValue === value ? "active" : "inactive"}
    onClick={() => onValueChange && onValueChange(value)}
  >
    {children}
  </button>
);
const TabsContent = ({ children, value, className = '', activeValue }) => 
  activeValue === value ? <div className={`tabs-content ${className}`}>{children}</div> : null;
const Table = ({ children, className = '' }) => <table className={`table ${className}`}>{children}</table>;
const TableHeader = ({ children }) => <thead>{children}</thead>;
const TableBody = ({ children }) => <tbody>{children}</tbody>;
const TableRow = ({ children, onClick, className = '' }) => <tr onClick={onClick} className={`table-row ${className}`}>{children}</tr>;
const TableHead = ({ children, className = '' }) => <th className={`table-head ${className}`}>{children}</th>;
const TableCell = ({ children, className = '' }) => <td className={`table-cell ${className}`}>{children}</td>;
const Switch = ({ checked, onCheckedChange = () => {}, className = '' }) => <input type="checkbox" checked={checked} onChange={(e) => onCheckedChange && onCheckedChange(e.target.checked)} className={`switch ${className}`} />;
const Slider = ({ value, onValueChange = () => {}, min = 0, max = 100, step = 1, className = '' }) => <input type="range" value={value} onChange={(e) => onValueChange && onValueChange([parseInt(e.target.value)])} min={min} max={max} step={step} className={`slider ${className}`} />;

// Custom hooks for real-time data
const useWebSocket = (url) => {
  const [data, setData] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const ws = useRef(null);

  useEffect(() => {
    try {
      ws.current = new WebSocket(url);
      
      ws.current.onopen = () => {
        setConnectionStatus('Connected');
        console.log('WebSocket connected');
      };
      
      ws.current.onmessage = (event) => {
        const newData = JSON.parse(event.data);
        setData(newData);
      };
      
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('Error');
      };
      
      ws.current.onclose = () => {
        setConnectionStatus('Disconnected');
        console.log('WebSocket disconnected');
      };
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      setConnectionStatus('Failed');
    }

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [url]);

  return { data, connectionStatus };
};

// Heat propagation visualization component
const HeatPropagationChart = ({ data, onNodeClick }) => {
  const svgRef = useRef();
  const [selectedNode, setSelectedNode] = useState(null);

  useEffect(() => {
    if (!data || !data.nodes || !data.links) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };

    // Create heat color scale
    const heatScale = d3.scaleSequential(d3.interpolateInferno)
      .domain([0, d3.max(data.nodes, d => d.heat_level)]);

    // Create force simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Create SVG container
    const container = svg
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('background', 'linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%)');

    // Add glow filter
    const defs = container.append('defs');
    const filter = defs.append('filter')
      .attr('id', 'glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');

    filter.append('feGaussianBlur')
      .attr('stdDeviation', 3)
      .attr('result', 'coloredBlur');

    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Create links
    const link = container.append('g')
      .selectAll('line')
      .data(data.links)
      .enter().append('line')
      .attr('stroke', d => d.heat_flow > 0.5 ? '#ff6b35' : '#4CAF50')
      .attr('stroke-width', d => Math.max(1, d.heat_flow * 5))
      .attr('stroke-opacity', 0.6)
      .style('filter', 'url(#glow)');

    // Create nodes
    const node = container.append('g')
      .selectAll('circle')
      .data(data.nodes)
      .enter().append('circle')
      .attr('r', d => 10 + d.market_cap_billions * 2)
      .attr('fill', d => heatScale(d.heat_level))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', d => selectedNode?.id === d.id ? 3 : 1)
      .style('filter', 'url(#glow)')
      .style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended))
      .on('click', (event, d) => {
        setSelectedNode(d);
        onNodeClick && onNodeClick(d);
      })
      .on('mouseover', function(event, d) {
        d3.select(this).transition()
          .duration(200)
          .attr('r', 15 + d.market_cap_billions * 2);
        
        // Show tooltip
        showTooltip(event, d);
      })
      .on('mouseout', function(event, d) {
        d3.select(this).transition()
          .duration(200)
          .attr('r', 10 + d.market_cap_billions * 2);
        
        hideTooltip();
      });

    // Add labels
    const label = container.append('g')
      .selectAll('text')
      .data(data.nodes)
      .enter().append('text')
      .text(d => d.symbol)
      .attr('font-size', 12)
      .attr('font-family', 'Arial, sans-serif')
      .attr('fill', '#ffffff')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central')
      .style('pointer-events', 'none')
      .style('font-weight', 'bold');

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);

      label
        .attr('x', d => d.x)
        .attr('y', d => d.y);
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

    // Tooltip functions
    function showTooltip(event, d) {
      const tooltip = d3.select('body').append('div')
        .attr('class', 'heat-tooltip')
        .style('opacity', 0)
        .style('position', 'absolute')
        .style('background', 'rgba(0, 0, 0, 0.9)')
        .style('color', 'white')
        .style('padding', '10px')
        .style('border-radius', '5px')
        .style('font-size', '12px')
        .style('pointer-events', 'none');

      tooltip.transition()
        .duration(200)
        .style('opacity', 1);

      tooltip.html(`
        <strong>${d.symbol}</strong><br/>
        Heat Level: ${d.heat_level.toFixed(2)}<br/>
        Market Cap: $${d.market_cap_billions}B<br/>
        Sector: ${d.sector}<br/>
        Prediction: ${d.prediction > 0 ? '+' : ''}${(d.prediction * 100).toFixed(1)}%
      `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    }

    function hideTooltip() {
      d3.selectAll('.heat-tooltip').remove();
    }

  }, [data, selectedNode, onNodeClick]);

  return (
    <div className="heat-propagation-container">
      <svg ref={svgRef}></svg>
      {selectedNode && (
        <div className="selected-node-info">
          <h3>{selectedNode.symbol}</h3>
          <p>Heat Level: {selectedNode.heat_level.toFixed(2)}</p>
          <p>Prediction: {(selectedNode.prediction * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
};

// Advanced sector analysis component
const SectorAnalysisGrid = ({ sectorData, onSectorClick }) => {
  const [sortBy, setSortBy] = useState('performance');
  const [sortOrder, setSortOrder] = useState('desc');

  const sortedSectors = useMemo(() => {
    if (!sectorData) return [];
    
    return [...sectorData].sort((a, b) => {
      const aVal = a[sortBy];
      const bVal = b[sortBy];
      const multiplier = sortOrder === 'desc' ? -1 : 1;
      return (aVal - bVal) * multiplier;
    });
  }, [sectorData, sortBy, sortOrder]);

  const getSectorIcon = (sector) => {
    const icons = {
      'Technology': '💻',
      'Healthcare': '🏥',
      'Financial': '🏦',
      'Energy': '⚡',
      'Consumer_Discretionary': '🛒',
      'Communication': '📡',
      'Industrial': '🏭',
      'Consumer_Staples': '🥫',
      'Utilities': '💡',
      'Real_Estate': '🏢'
    };
    return icons[sector] || '📈';
  };

  const getPerformanceColor = (performance) => {
    if (performance > 5) return 'text-green-500';
    if (performance > 0) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getHeatLevel = (heatScore) => {
    if (heatScore > 0.8) return 'HIGH';
    if (heatScore > 0.5) return 'MEDIUM';
    return 'LOW';
  };

  return (
    <div className="sector-analysis-grid">
      <div className="controls mb-4">
        <div className="flex gap-4 items-center">
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
            className="px-3 py-2 border rounded-md"
          >
            <option value="performance">Performance</option>
            <option value="heat_score">Heat Score</option>
            <option value="volume_change">Volume Change</option>
            <option value="volatility">Volatility</option>
          </select>
          <button 
            onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
            className="px-3 py-2 bg-blue-500 text-white rounded-md"
          >
            {sortOrder === 'desc' ? '↓' : '↑'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sortedSectors.map((sector, index) => (
          <Card 
            key={sector.name}
            className={`sector-card cursor-pointer transition-all duration-200 hover:shadow-lg ${
              sector.heat_score > 0.7 ? 'border-orange-400 shadow-orange-100' : ''
            }`}
            onClick={() => onSectorClick && onSectorClick(sector)}
          >
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{getSectorIcon(sector.key)}</span>
                  <CardTitle className="text-lg">{sector.name}</CardTitle>
                </div>
                <Badge variant={sector.heat_score > 0.7 ? 'destructive' : 'secondary'}>
                  {getHeatLevel(sector.heat_score)}
                </Badge>
              </div>
            </CardHeader>
            
            <CardContent>
              <div className="space-y-3">
                {/* Performance */}
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Performance</span>
                  <span className={`font-bold ${getPerformanceColor(sector.performance)}`}>
                    {sector.performance > 0 ? '+' : ''}{sector.performance.toFixed(2)}%
                  </span>
                </div>

                {/* Heat Score */}
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Heat Score</span>
                  <div className="flex items-center gap-2">
                    <Progress value={sector.heat_score * 100} className="w-16" />
                    <span className="text-sm font-medium">{(sector.heat_score * 100).toFixed(0)}%</span>
                  </div>
                </div>

                {/* Volume Change */}
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Volume</span>
                  <span className={`text-sm ${sector.volume_change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {sector.volume_change > 0 ? '+' : ''}{sector.volume_change.toFixed(1)}%
                  </span>
                </div>

                {/* Top Stocks Preview */}
                <div className="mt-3">
                  <span className="text-xs text-gray-500">Top Stocks:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {sector.top_stocks?.slice(0, 4).map(stock => (
                      <Badge key={stock} variant="outline" className="text-xs">
                        {stock}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* AI Prediction */}
                <div className="flex items-center gap-2 p-2 bg-gray-50 rounded-md">
                  <Brain className="w-4 h-4 text-purple-600" />
                  <span className="text-xs text-gray-600">AI Prediction:</span>
                  <Badge variant={sector.ai_prediction > 0 ? 'default' : 'secondary'}>
                    {sector.ai_prediction > 0 ? 'BULLISH' : 'BEARISH'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

// Real-time signals component
const LiveSignalsPanel = ({ signals, onSignalClick }) => {
  const [filter, setFilter] = useState('ALL');
  const [autoScroll, setAutoScroll] = useState(true);
  const signalsEndRef = useRef(null);

  useEffect(() => {
    if (autoScroll && signalsEndRef.current) {
      signalsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [signals, autoScroll]);

  const filteredSignals = useMemo(() => {
    if (!signals) return [];
    if (filter === 'ALL') return signals;
    return signals.filter(signal => signal.signal_type === filter);
  }, [signals, filter]);

  const getSignalIcon = (signalType) => {
    switch (signalType) {
      case 'STRONG_BUY': return <TrendingUp className="w-4 h-4 text-green-600" />;
      case 'BUY': return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'HOLD': return <Activity className="w-4 h-4 text-yellow-500" />;
      case 'SELL': return <TrendingDown className="w-4 h-4 text-red-400" />;
      case 'STRONG_SELL': return <TrendingDown className="w-4 h-4 text-red-600" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const getSignalBadgeVariant = (signalType) => {
    switch (signalType) {
      case 'STRONG_BUY': return 'default';
      case 'BUY': return 'secondary';
      case 'HOLD': return 'outline';
      case 'SELL': return 'destructive';
      case 'STRONG_SELL': return 'destructive';
      default: return 'outline';
    }
  };

  return (
    <Card className="live-signals-panel h-full">
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5" />
            Live Trading Signals
          </CardTitle>
          <div className="flex items-center gap-2">
            <Switch 
              checked={autoScroll}
              onCheckedChange={setAutoScroll}
              id="auto-scroll"
            />
            <label htmlFor="auto-scroll" className="text-sm">Auto-scroll</label>
          </div>
        </div>
        
        <div className="flex gap-2 mt-2">
          {['ALL', 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'].map(filterType => (
            <Button
              key={filterType}
              variant={filter === filterType ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilter(filterType)}
            >
              {filterType.replace('_', ' ')}
            </Button>
          ))}
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        <div className="max-h-96 overflow-y-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Signal</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Price Target</TableHead>
                <TableHead>Time</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredSignals.map((signal, index) => (
                <TableRow 
                  key={signal.id || index}
                  className="cursor-pointer hover:bg-gray-50"
                  onClick={() => onSignalClick && onSignalClick(signal)}
                >
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-2">
                      <Avatar className="w-8 h-8">
                        <AvatarFallback>{signal.symbol}</AvatarFallback>
                      </Avatar>
                      {signal.symbol}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      {getSignalIcon(signal.signal_type)}
                      <Badge variant={getSignalBadgeVariant(signal.signal_type)}>
                        {signal.signal_type.replace('_', ' ')}
                      </Badge>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Progress value={signal.confidence * 100} className="w-16" />
                      <span className="text-sm">{(signal.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <span className={signal.expected_return > 0 ? 'text-green-600' : 'text-red-600'}>
                      ${signal.price_target?.toFixed(2) || 'N/A'}
                    </span>
                  </TableCell>
                  <TableCell className="text-sm text-gray-500">
                    {new Date(signal.timestamp).toLocaleTimeString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <div ref={signalsEndRef} />
        </div>
      </CardContent>
    </Card>
  );
};

// ML Model Performance Dashboard
const MLModelDashboard = ({ modelData, predictions }) => {
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [timeHorizon, setTimeHorizon] = useState('20d');

  const modelPerformanceData = useMemo(() => {
    if (!modelData?.performance_metrics) return [];
    
    return Object.entries(modelData.performance_metrics).map(([model, metrics]) => ({
      model,
      accuracy: metrics.accuracy || 0,
      precision: metrics.precision || 0,
      recall: metrics.recall || 0,
      f1_score: metrics.f1_score || 0,
      sharpe_ratio: metrics.sharpe_ratio || 0
    }));
  }, [modelData]);

  const predictionData = useMemo(() => {
    if (!predictions) return [];
    
    return predictions.map(pred => ({
      symbol: pred.symbol,
      predicted_return: pred.predicted_return * 100,
      confidence: pred.confidence * 100,
      model_used: pred.model_name,
      time_horizon: pred.prediction_horizon
    }));
  }, [predictions]);

  return (
    <div className="ml-model-dashboard">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Performance Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Model Performance Comparison
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={modelPerformanceData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="model" />
                <PolarRadiusAxis angle={30} domain={[0, 1]} />
                <Radar
                  name="Accuracy"
                  dataKey="accuracy"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.3}
                />
                <Radar
                  name="Sharpe Ratio"
                  dataKey="sharpe_ratio"
                  stroke="#82ca9d"
                  fill="#82ca9d"
                  fillOpacity={0.3}
                />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Prediction Confidence Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Prediction Confidence Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid />
                <XAxis 
                  type="number" 
                  dataKey="predicted_return" 
                  name="Predicted Return %"
                  domain={[-20, 20]}
                />
                <YAxis 
                  type="number" 
                  dataKey="confidence" 
                  name="Confidence %"
                  domain={[0, 100]}
                />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter data={predictionData} fill="#8884d8">
                  {predictionData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.predicted_return > 0 ? '#00C49F' : '#FF8042'} 
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Model Selection and Controls */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Model Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Active Model</label>
              <select 
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-3 py-2 border rounded-md"
              >
                <option value="ensemble">Ensemble Model</option>
                <option value="random_forest">Random Forest</option>
                <option value="xgboost">XGBoost</option>
                <option value="lstm">LSTM Neural Network</option>
                <option value="garch">GARCH Model</option>
                <option value="hmm">Hidden Markov Model</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Prediction Horizon</label>
              <select 
                value={timeHorizon}
                onChange={(e) => setTimeHorizon(e.target.value)}
                className="w-full px-3 py-2 border rounded-md"
              >
                <option value="1d">1 Day</option>
                <option value="5d">5 Days</option>
                <option value="20d">20 Days</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Confidence Threshold</label>
              <Slider
                value={[0.6]}
                max={1}
                min={0}
                step={0.1}
                className="w-full"
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Main Advanced Trading Dashboard
const AdvancedTradingDashboard = ({ apiUrl = 'http://localhost:8002' }) => {
  // State management
  const [activeTab, setActiveTab] = useState('overview');
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedStock, setSelectedStock] = useState(null);
  const [refreshInterval, setRefreshInterval] = useState(5000);

  // Helper functions
  const getSignalIcon = (signalType) => {
    switch (signalType) {
      case 'STRONG_BUY': return <TrendingUp className="w-4 h-4 text-green-600" />;
      case 'BUY': return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'HOLD': return <Activity className="w-4 h-4 text-yellow-500" />;
      case 'SELL': return <TrendingDown className="w-4 h-4 text-red-400" />;
      case 'STRONG_SELL': return <TrendingDown className="w-4 h-4 text-red-600" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const getSignalBadgeVariant = (signalType) => {
    switch (signalType) {
      case 'STRONG_BUY': return 'default';
      case 'BUY': return 'secondary';
      case 'HOLD': return 'outline';
      case 'SELL': return 'destructive';
      case 'STRONG_SELL': return 'destructive';
      default: return 'outline';
    }
  };

  // State for real data from API
  const [heatData, setHeatData] = useState(null);
  const [signalsData, setSignalsData] = useState(null);
  const [sectorData, setSectorData] = useState(null);
  const [heatStatus, setHeatStatus] = useState('disconnected');
  const [signalsStatus, setSignalsStatus] = useState('disconnected');
  const [sectorStatus, setSectorStatus] = useState('disconnected');

  // Fetch real market data from API
  const fetchMarketData = useCallback(async () => {
    if (!isStreaming) return;
    
    try {
      setHeatStatus('connecting');
      setSignalsStatus('connecting');
      setSectorStatus('connecting');
      
      // Fetch market data from our API
      const marketResponse = await fetch(`${apiUrl}/api/market-data`);
      const marketData = await marketResponse.json();
      
      if (marketData.status === 'success' && marketData.data) {
        const stocks = marketData.data.stocks;
        
        // Convert to heat data format
        const heatNodes = Object.values(stocks).map(stock => ({
          id: stock.symbol,
          symbol: stock.symbol,
          heat_level: stock.heat_score / 100,
          market_cap_billions: (stock.market_cap || 0) / 1000000000,
          sector: stock.sector,
          prediction: stock.change_percent / 100
        }));
        
        setHeatData({ nodes: heatNodes });
        setSignalsData({ signals: heatNodes });
        setSectorData({ sectors: heatNodes });
        
        setHeatStatus('connected');
        setSignalsStatus('connected');
        setSectorStatus('connected');
      }
    } catch (error) {
      console.error('Error fetching market data:', error);
      setHeatStatus('error');
      setSignalsStatus('error');
      setSectorStatus('error');
    }
  }, [apiUrl, isStreaming]);

  // Auto-fetch data when streaming is enabled
  useEffect(() => {
    if (isStreaming) {
      fetchMarketData();
      const interval = setInterval(fetchMarketData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [isStreaming, fetchMarketData, refreshInterval]);

  // Mock data for demonstration
  const mockHeatData = {
    nodes: [
      { id: 'AAPL', symbol: 'AAPL', heat_level: 0.85, market_cap_billions: 2800, sector: 'Technology', prediction: 0.12 },
      { id: 'TSLA', symbol: 'TSLA', heat_level: 0.92, market_cap_billions: 800, sector: 'Consumer_Discretionary', prediction: 0.08 },
      { id: 'GOOGL', symbol: 'GOOGL', heat_level: 0.78, market_cap_billions: 1600, sector: 'Technology', prediction: 0.15 },
      { id: 'NVDA', symbol: 'NVDA', heat_level: 0.95, market_cap_billions: 1000, sector: 'Technology', prediction: 0.20 },
      { id: 'META', symbol: 'META', heat_level: 0.72, market_cap_billions: 750, sector: 'Communication', prediction: 0.06 }
    ],
    links: [
      { source: 'AAPL', target: 'GOOGL', heat_flow: 0.7 },
      { source: 'NVDA', target: 'TSLA', heat_flow: 0.8 },
      { source: 'AAPL', target: 'NVDA', heat_flow: 0.6 },
      { source: 'GOOGL', target: 'META', heat_flow: 0.5 }
    ]
  };

  const mockSectorData = [
    { key: 'Technology', name: 'Technology', performance: 8.5, heat_score: 0.85, volume_change: 15.2, volatility: 0.25, ai_prediction: 1, top_stocks: ['AAPL', 'GOOGL', 'NVDA', 'MSFT'] },
    { key: 'Healthcare', name: 'Healthcare', performance: 3.2, heat_score: 0.65, volume_change: 8.1, volatility: 0.18, ai_prediction: 1, top_stocks: ['JNJ', 'PFE', 'UNH', 'ABBV'] },
    { key: 'Financial', name: 'Financial Services', performance: -1.8, heat_score: 0.45, volume_change: -5.3, volatility: 0.32, ai_prediction: -1, top_stocks: ['JPM', 'BAC', 'WFC', 'GS'] },
    { key: 'Energy', name: 'Energy', performance: 12.7, heat_score: 0.78, volume_change: 22.4, volatility: 0.45, ai_prediction: 1, top_stocks: ['XOM', 'CVX', 'COP', 'EOG'] }
  ];

  const mockSignalsData = [
    { id: 1, symbol: 'NVDA', signal_type: 'STRONG_BUY', confidence: 0.92, expected_return: 0.15, price_target: 485.50, timestamp: new Date() },
    { id: 2, symbol: 'TSLA', signal_type: 'BUY', confidence: 0.78, expected_return: 0.08, price_target: 245.30, timestamp: new Date() },
    { id: 3, symbol: 'AAPL', signal_type: 'HOLD', confidence: 0.65, expected_return: 0.02, price_target: 185.75, timestamp: new Date() },
    { id: 4, symbol: 'META', signal_type: 'SELL', confidence: 0.72, expected_return: -0.05, price_target: 295.20, timestamp: new Date() }
  ];

  // Event handlers
  const handleNodeClick = useCallback((node) => {
    setSelectedStock(node);
    console.log('Selected node:', node);
  }, []);

  const handleSectorClick = useCallback((sector) => {
    console.log('Selected sector:', sector);
    setActiveTab('stocks');
  }, []);

  const handleSignalClick = useCallback((signal) => {
    setSelectedStock({ symbol: signal.symbol });
    console.log('Selected signal:', signal);
  }, []);

  const toggleStreaming = useCallback(() => {
    setIsStreaming(!isStreaming);
  }, [isStreaming]);

  return (
    <div className="advanced-trading-dashboard min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      {/* Header */}
      <header className="bg-black/50 backdrop-blur-sm border-b border-gray-700 p-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Flame className="w-8 h-8 text-orange-500" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-orange-400 to-red-600 bg-clip-text text-transparent">
                RAGHeat Pro
              </h1>
            </div>
            <Badge variant="secondary" className="bg-purple-600/20 text-purple-300">
              AI-Powered Trading Platform
            </Badge>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isStreaming ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
              <span className="text-sm">{isStreaming ? 'Live' : 'Paused'}</span>
            </div>
            
            <Button onClick={toggleStreaming} variant={isStreaming ? 'destructive' : 'default'}>
              {isStreaming ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isStreaming ? 'Pause' : 'Start'} Stream
            </Button>
            
            <Button variant="outline">
              <Settings className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-5 bg-gray-800/50">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="heat-map">Heat Propagation</TabsTrigger>
            <TabsTrigger value="sectors">Sector Analysis</TabsTrigger>
            <TabsTrigger value="signals">Live Signals</TabsTrigger>
            <TabsTrigger value="ml-models">AI Models</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Key Metrics */}
              <Card className="bg-gradient-to-br from-green-900/20 to-green-800/20 border-green-500/30">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-green-400">
                    <TrendingUp className="w-5 h-5" />
                    Portfolio Performance
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-green-400">+24.7%</div>
                  <p className="text-sm text-gray-400">This Month</p>
                  <div className="mt-4 space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Total Return</span>
                      <span className="text-sm text-green-400">+$47,892</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Sharpe Ratio</span>
                      <span className="text-sm">2.43</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 border-blue-500/30">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-blue-400">
                    <Brain className="w-5 h-5" />
                    AI Confidence
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-blue-400">87%</div>
                  <p className="text-sm text-gray-400">Model Accuracy</p>
                  <div className="mt-4">
                    <Progress value={87} className="w-full" />
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-orange-900/20 to-orange-800/20 border-orange-500/30">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-orange-400">
                    <Flame className="w-5 h-5" />
                    Heat Index
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-orange-400">9.2</div>
                  <p className="text-sm text-gray-400">Market Temperature</p>
                  <Badge variant="destructive" className="mt-2">
                    VERY HOT
                  </Badge>
                </CardContent>
              </Card>
            </div>

            {/* Quick Overview Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-gray-800/50">
                <CardHeader>
                  <CardTitle>Top Performers Today</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'META'].map((symbol, index) => (
                      <div key={symbol} className="flex items-center justify-between p-2 rounded bg-gray-700/30">
                        <div className="flex items-center gap-3">
                          <span className="text-sm font-medium">{index + 1}</span>
                          <span className="font-medium">{symbol}</span>
                        </div>
                        <span className="text-green-400 font-medium">+{(Math.random() * 10 + 1).toFixed(2)}%</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-800/50">
                <CardHeader>
                  <CardTitle>Recent AI Predictions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {mockSignalsData.slice(0, 4).map((signal, index) => (
                      <div key={index} className="flex items-center justify-between p-2 rounded bg-gray-700/30">
                        <div className="flex items-center gap-3">
                          {getSignalIcon(signal.signal_type)}
                          <span className="font-medium">{signal.symbol}</span>
                        </div>
                        <Badge variant={getSignalBadgeVariant(signal.signal_type)}>
                          {signal.signal_type.replace('_', ' ')}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Heat Propagation Tab */}
          <TabsContent value="heat-map" className="space-y-6">
            <Card className="bg-gray-800/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Network className="w-5 h-5" />
                  Viral Heat Propagation Network
                </CardTitle>
                <p className="text-sm text-gray-400">
                  Real-time visualization of heat propagation through stock networks using viral spread algorithms
                </p>
              </CardHeader>
              <CardContent>
                <HeatPropagationChart 
                  data={heatData || mockHeatData} 
                  onNodeClick={handleNodeClick}
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Sector Analysis Tab */}
          <TabsContent value="sectors" className="space-y-6">
            <SectorAnalysisGrid 
              sectorData={sectorData || mockSectorData}
              onSectorClick={handleSectorClick}
            />
          </TabsContent>

          {/* Live Signals Tab */}
          <TabsContent value="signals" className="space-y-6">
            <LiveSignalsPanel 
              signals={signalsData || mockSignalsData}
              onSignalClick={handleSignalClick}
            />
          </TabsContent>

          {/* ML Models Tab */}
          <TabsContent value="ml-models" className="space-y-6">
            <MLModelDashboard 
              modelData={{}}
              predictions={[]}
            />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default AdvancedTradingDashboard;