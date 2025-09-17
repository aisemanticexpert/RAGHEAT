import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { motion } from 'framer-motion';

const HeatMapChart = ({ data }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!data?.cells) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 900;
    const height = 650;
    const margin = { top: 40, right: 40, bottom: 80, left: 100 };

    // Use the cells data from the heat distribution API
    const heatMapData = data.cells;
    const gridCols = 6; // Reduced columns for larger cells
    const gridRows = Math.ceil(heatMapData.length / gridCols);

    // Calculate cell dimensions
    const cellWidth = (width - margin.left - margin.right) / gridCols;
    const cellHeight = (height - margin.top - margin.bottom) / gridRows;

    // Color scale based on heat value (0-100)
    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, 100]);

    // Signal-based color override
    const getSignalColor = (cell) => {
      if (cell.signal === 'buy' && cell.confidence > 0.7) return '#00ff88';
      if (cell.signal === 'buy' && cell.confidence > 0.5) return '#88ff44';  
      if (cell.signal === 'sell' && cell.confidence > 0.7) return '#ff3366';
      if (cell.signal === 'sell' && cell.confidence > 0.5) return '#ff6b35';
      return cell.color || colorScale(cell.value);
    };

    // Create tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'heat-tooltip')
      .style('opacity', 0);

    // Draw heat map cells
    svg.selectAll('.heat-cell')
      .data(heatMapData)
      .enter().append('g')
      .attr('class', 'heat-cell')
      .attr('transform', d => `translate(${margin.left + d.x * cellWidth}, ${margin.top + d.y * cellHeight})`)
      .each(function(d) {
        const cell = d3.select(this);
        
        // Main rectangle
        cell.append('rect')
          .attr('width', cellWidth - 2)
          .attr('height', cellHeight - 2)
          .attr('fill', getSignalColor(d))
          .attr('stroke', '#1a1a3a')
          .attr('stroke-width', 1)
          .attr('opacity', 0)
          .transition()
          .duration(800)
          .delay((d, i) => i * 30)
          .attr('opacity', 0.8);
        
        // Stock symbol text
        cell.append('text')
          .attr('x', cellWidth / 2)
          .attr('y', cellHeight / 2 - 18)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .style('fill', 'white')
          .style('font-size', '16px')
          .style('font-weight', 'bold')
          .style('text-shadow', '1px 1px 2px rgba(0,0,0,0.8)')
          .text(d.symbol || 'N/A');
        
        // Signal indicator
        cell.append('text')
          .attr('x', cellWidth / 2)
          .attr('y', cellHeight / 2 + 18)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .style('fill', 'white')
          .style('font-size', '11px')
          .style('font-weight', '600')
          .style('text-shadow', '1px 1px 2px rgba(0,0,0,0.8)')
          .text(`${(d.signal || 'HOLD').toUpperCase()} ${((d.confidence || 0) * 100).toFixed(0)}%`);
      })
      .on('mouseover', function(event, d) {
        d3.select(this).select('rect').style('opacity', 1);
        tooltip.transition().duration(200).style('opacity', .9);
        tooltip.html(`
          <strong>${d.symbol || 'N/A'}</strong> (${d.sector || 'Unknown'})<br/>
          Signal: <span style="color: ${getSignalColor(d)}">${(d.signal || 'hold').toUpperCase()}</span><br/>
          Confidence: ${(d.confidence * 100).toFixed(1)}%<br/>
          Change: ${(d.change || d.change_percent || 0).toFixed(2)}%<br/>
          Heat: ${(d.value || 0).toFixed(1)}<br/>
          Volume: ${(d.volume || 0).toLocaleString()}
        `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', function(d) {
        d3.select(this).select('rect').style('opacity', 0.8);
        tooltip.transition().duration(500).style('opacity', 0);
      });

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .style('fill', '#00d4ff')
      .style('font-size', '20px')
      .style('font-weight', 'bold')
      .text('Market Heat Map - Live Trading Signals');

    // Add legend
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - 180}, 50)`);
    
    const legendData = [
      { label: 'Strong Buy', color: '#00ff88' },
      { label: 'Buy', color: '#88ff44' },
      { label: 'Hold', color: '#ffaa00' },
      { label: 'Sell', color: '#ff6b35' },
      { label: 'Strong Sell', color: '#ff3366' }
    ];
    
    legend.selectAll('.legend-item')
      .data(legendData)
      .enter().append('g')
      .attr('class', 'legend-item')
      .attr('transform', (d, i) => `translate(0, ${i * 25})`)
      .each(function(d) {
        const item = d3.select(this);
        item.append('rect')
          .attr('width', 15)
          .attr('height', 15)
          .attr('fill', d.color);
        item.append('text')
          .attr('x', 20)
          .attr('y', 12)
          .style('fill', '#a0a0c0')
          .style('font-size', '12px')
          .text(d.label);
      });

    return () => {
      tooltip.remove();
    };
  }, [data]);

  return (
    <motion.div 
      className="heatmap-container"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6 }}
    >
      <svg 
        ref={svgRef} 
        width="100%" 
        height="650" 
        viewBox="0 0 900 650"
      />
      <style jsx>{`
        .heat-tooltip {
          position: absolute;
          text-align: left;
          padding: 10px;
          font-size: 12px;
          background: rgba(15, 15, 35, 0.95);
          color: white;
          border: 1px solid #00d4ff;
          border-radius: 8px;
          pointer-events: none;
          backdrop-filter: blur(10px);
        }
        .heatmap-container {
          width: 100%;
          overflow: visible;
        }
      `}</style>
    </motion.div>
  );
};

export default HeatMapChart;