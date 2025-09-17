"""
Revolutionary Viral Heat Propagation Engine for Stock Market Prediction
Based on Social Network Viral Spread Algorithms Applied to Financial Markets

This engine calculates the probability of a stock performing well/poorly based on:
1. Neighbor stock performance (sector/industry connections)
2. Network topology (market relationships)
3. Information cascades (news, earnings, sentiment)
4. Heat diffusion through market networks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HeatNode:
    """Represents a node in the heat propagation network"""
    symbol: str
    node_type: str  # 'stock', 'sector', 'industry', 'market'
    current_heat: float
    heat_capacity: float  # How much heat the node can absorb
    heat_conductivity: float  # How well heat flows through this node
    connections: List[str]  # Connected node symbols
    influence_weights: Dict[str, float]  # Influence on each connection
    metadata: Dict[str, any]


@dataclass
class HeatPropagationResult:
    """Result of heat propagation analysis"""
    symbol: str
    initial_heat: float
    final_heat: float
    heat_change: float
    propagation_steps: int
    influence_sources: Dict[str, float]  # Which nodes influenced this one
    viral_coefficient: float  # How viral this node's heat is
    heat_trajectory: List[float]  # Heat over time
    convergence_reached: bool
    prediction_confidence: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'initial_heat': self.initial_heat,
            'final_heat': self.final_heat,
            'heat_change': self.heat_change,
            'propagation_steps': self.propagation_steps,
            'influence_sources': self.influence_sources,
            'viral_coefficient': self.viral_coefficient,
            'heat_trajectory': self.heat_trajectory,
            'convergence_reached': self.convergence_reached,
            'prediction_confidence': self.prediction_confidence,
            'timestamp': self.timestamp.isoformat()
        }


class ViralHeatEngine:
    """
    Advanced Viral Heat Propagation Engine for Market Prediction
    
    Implements algorithms inspired by:
    - Independent Cascade Model (ICM)
    - Linear Threshold Model (LTM)
    - Heat Diffusion on Networks
    - PageRank-style Authority Propagation
    - Epidemic Spread Models (SIR/SEIR)
    """
    
    def __init__(self,
                 propagation_model: str = "hybrid",  # "icm", "ltm", "heat_diffusion", "hybrid"
                 convergence_threshold: float = 0.001,
                 max_iterations: int = 100,
                 damping_factor: float = 0.85,
                 heat_decay: float = 0.95):
        """
        Initialize Viral Heat Engine
        
        Args:
            propagation_model: Type of propagation algorithm
            convergence_threshold: Threshold for convergence detection
            max_iterations: Maximum propagation iterations
            damping_factor: Heat propagation damping (like PageRank)
            heat_decay: Heat decay rate over time
        """
        self.propagation_model = propagation_model
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.damping_factor = damping_factor
        self.heat_decay = heat_decay
        
        # Network components
        self.graph = nx.DiGraph()
        self.heat_nodes = {}
        self.adjacency_matrix = None
        self.transition_matrix = None
        
        # Heat propagation state
        self.current_heat_vector = None
        self.heat_history = []
        self.iteration_count = 0
        
        # Performance tracking
        self.propagation_stats = {
            'total_propagations': 0,
            'avg_iterations': 0,
            'convergence_rate': 0
        }
    
    def build_market_network(self, 
                           stocks_data: Dict[str, any],
                           sector_correlations: Optional[pd.DataFrame] = None,
                           custom_connections: Optional[Dict[str, List[str]]] = None) -> 'ViralHeatEngine':
        """
        Build the market network graph for heat propagation
        
        Args:
            stocks_data: Dictionary with stock info {symbol: {sector, industry, market_cap, etc}}
            sector_correlations: Correlation matrix between sectors
            custom_connections: Custom defined connections between stocks
            
        Returns:
            Self for method chaining
        """
        # Clear existing network
        self.graph.clear()
        self.heat_nodes.clear()
        
        # Create nodes for each stock
        for symbol, data in stocks_data.items():
            heat_node = HeatNode(
                symbol=symbol,
                node_type='stock',
                current_heat=0.0,
                heat_capacity=self._calculate_heat_capacity(data),
                heat_conductivity=self._calculate_heat_conductivity(data),
                connections=[],
                influence_weights={},
                metadata=data
            )
            
            self.heat_nodes[symbol] = heat_node
            self.graph.add_node(symbol, **heat_node.__dict__)
        
        # Add sector and industry nodes
        self._add_hierarchical_nodes(stocks_data)
        
        # Create connections based on multiple criteria
        self._create_sector_connections(stocks_data)
        self._create_industry_connections(stocks_data)
        self._create_correlation_connections(sector_correlations)
        self._create_market_cap_connections(stocks_data)
        self._create_custom_connections(custom_connections)
        
        # Build adjacency and transition matrices
        self._build_matrices()
        
        return self
    
    def _calculate_heat_capacity(self, stock_data: Dict) -> float:
        """Calculate how much heat a stock can absorb"""
        # Based on market cap, volatility, liquidity
        market_cap = stock_data.get('market_cap', 1e9)
        volatility = stock_data.get('volatility', 0.3)
        volume = stock_data.get('avg_volume', 1e6)
        
        # Larger, less volatile, more liquid stocks have higher capacity
        capacity = np.log(market_cap / 1e6) / (1 + volatility) * np.log(volume / 1e3)
        return max(0.1, min(10.0, capacity))
    
    def _calculate_heat_conductivity(self, stock_data: Dict) -> float:
        """Calculate how well heat flows through a stock"""
        # Based on beta, correlation with market, news coverage
        beta = stock_data.get('beta', 1.0)
        correlation = stock_data.get('market_correlation', 0.7)
        news_coverage = stock_data.get('news_mentions', 10)
        
        # Higher beta, correlation, and news coverage = higher conductivity
        conductivity = (abs(beta) + correlation) * np.log(news_coverage + 1) / 3
        return max(0.1, min(5.0, conductivity))
    
    def _add_hierarchical_nodes(self, stocks_data: Dict[str, any]):
        """Add sector and industry level nodes"""
        sectors = set()
        industries = set()
        
        for symbol, data in stocks_data.items():
            sector = data.get('sector', 'Unknown')
            industry = data.get('industry', 'Unknown')
            sectors.add(sector)
            industries.add(industry)
        
        # Add sector nodes
        for sector in sectors:
            if sector != 'Unknown':
                sector_node = HeatNode(
                    symbol=f"SECTOR_{sector}",
                    node_type='sector',
                    current_heat=0.0,
                    heat_capacity=5.0,  # Sectors have high capacity
                    heat_conductivity=3.0,  # Good conductivity
                    connections=[],
                    influence_weights={},
                    metadata={'name': sector, 'type': 'sector'}
                )
                
                self.heat_nodes[sector_node.symbol] = sector_node
                self.graph.add_node(sector_node.symbol, **sector_node.__dict__)
        
        # Add industry nodes
        for industry in industries:
            if industry != 'Unknown':
                industry_node = HeatNode(
                    symbol=f"INDUSTRY_{industry}",
                    node_type='industry',
                    current_heat=0.0,
                    heat_capacity=3.0,
                    heat_conductivity=2.0,
                    connections=[],
                    influence_weights={},
                    metadata={'name': industry, 'type': 'industry'}
                )
                
                self.heat_nodes[industry_node.symbol] = industry_node
                self.graph.add_node(industry_node.symbol, **industry_node.__dict__)
    
    def _create_sector_connections(self, stocks_data: Dict[str, any]):
        """Create connections between stocks and their sectors"""
        for symbol, data in stocks_data.items():
            sector = data.get('sector', 'Unknown')
            if sector != 'Unknown':
                sector_symbol = f"SECTOR_{sector}"
                
                # Bidirectional connection with different weights
                influence_to_sector = 0.3  # Stock influences sector
                influence_from_sector = 0.7  # Sector influences stock more
                
                self.graph.add_edge(symbol, sector_symbol, weight=influence_to_sector)
                self.graph.add_edge(sector_symbol, symbol, weight=influence_from_sector)
                
                self.heat_nodes[symbol].connections.append(sector_symbol)
                self.heat_nodes[symbol].influence_weights[sector_symbol] = influence_to_sector
    
    def _create_industry_connections(self, stocks_data: Dict[str, any]):
        """Create connections between stocks and their industries"""
        for symbol, data in stocks_data.items():
            industry = data.get('industry', 'Unknown')
            if industry != 'Unknown':
                industry_symbol = f"INDUSTRY_{industry}"
                
                influence_to_industry = 0.2
                influence_from_industry = 0.5
                
                self.graph.add_edge(symbol, industry_symbol, weight=influence_to_industry)
                self.graph.add_edge(industry_symbol, symbol, weight=influence_from_industry)
    
    def _create_correlation_connections(self, correlations: Optional[pd.DataFrame]):
        """Create connections based on correlation strength"""
        if correlations is None:
            return
        
        # Connect highly correlated stocks
        for i, stock1 in enumerate(correlations.index):
            for j, stock2 in enumerate(correlations.columns):
                if i != j and stock1 in self.heat_nodes and stock2 in self.heat_nodes:
                    correlation = correlations.loc[stock1, stock2]
                    
                    if abs(correlation) > 0.5:  # Strong correlation threshold
                        weight = abs(correlation) * 0.4  # Scale correlation to weight
                        self.graph.add_edge(stock1, stock2, weight=weight)
                        self.graph.add_edge(stock2, stock1, weight=weight)
    
    def _create_market_cap_connections(self, stocks_data: Dict[str, any]):
        """Create connections between stocks of similar market cap"""
        # Group stocks by market cap buckets
        cap_buckets = {'mega': [], 'large': [], 'mid': [], 'small': []}
        
        for symbol, data in stocks_data.items():
            market_cap = data.get('market_cap', 1e9)
            
            if market_cap > 200e9:
                cap_buckets['mega'].append(symbol)
            elif market_cap > 10e9:
                cap_buckets['large'].append(symbol)
            elif market_cap > 2e9:
                cap_buckets['mid'].append(symbol)
            else:
                cap_buckets['small'].append(symbol)
        
        # Connect stocks within each bucket with weak connections
        for bucket, stocks in cap_buckets.items():
            for i, stock1 in enumerate(stocks):
                for j, stock2 in enumerate(stocks):
                    if i != j:
                        weight = 0.1  # Weak connection
                        self.graph.add_edge(stock1, stock2, weight=weight)
    
    def _create_custom_connections(self, custom_connections: Optional[Dict[str, List[str]]]):
        """Add custom defined connections"""
        if custom_connections is None:
            return
        
        for source, targets in custom_connections.items():
            if source in self.heat_nodes:
                for target in targets:
                    if target in self.heat_nodes:
                        weight = 0.6  # Strong custom connection
                        self.graph.add_edge(source, target, weight=weight)
    
    def _build_matrices(self):
        """Build adjacency and transition matrices for efficient computation"""
        nodes = list(self.graph.nodes())
        n = len(nodes)
        
        # Create adjacency matrix
        adjacency = np.zeros((n, n))
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if self.graph.has_edge(node1, node2):
                    weight = self.graph[node1][node2]['weight']
                    adjacency[i][j] = weight
        
        self.adjacency_matrix = csr_matrix(adjacency)
        
        # Create transition matrix (row-normalized)
        row_sums = np.array(adjacency.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        
        transition = adjacency / row_sums[:, np.newaxis]
        self.transition_matrix = csr_matrix(transition)
        
        # Initialize heat vector
        self.current_heat_vector = np.zeros(n)
    
    def set_initial_heat(self, heat_sources: Dict[str, float]) -> 'ViralHeatEngine':
        """
        Set initial heat values for specified nodes
        
        Args:
            heat_sources: Dictionary {symbol: heat_value}
            
        Returns:
            Self for method chaining
        """
        nodes = list(self.graph.nodes())
        
        for symbol, heat in heat_sources.items():
            if symbol in nodes:
                idx = nodes.index(symbol)
                self.current_heat_vector[idx] = heat
                self.heat_nodes[symbol].current_heat = heat
        
        # Store initial state
        self.heat_history = [self.current_heat_vector.copy()]
        
        return self
    
    def propagate_heat(self, steps: Optional[int] = None) -> Dict[str, HeatPropagationResult]:
        """
        Execute heat propagation algorithm
        
        Args:
            steps: Number of propagation steps (None for convergence)
            
        Returns:
            Dictionary of propagation results for each node
        """
        if self.current_heat_vector is None:
            raise ValueError("Must set initial heat before propagation")
        
        max_steps = steps or self.max_iterations
        nodes = list(self.graph.nodes())
        
        # Store initial heat for comparison
        initial_heat = self.current_heat_vector.copy()
        
        # Propagation loop
        for iteration in range(max_steps):
            previous_heat = self.current_heat_vector.copy()
            
            if self.propagation_model == "icm":
                self._independent_cascade_step()
            elif self.propagation_model == "ltm":
                self._linear_threshold_step()
            elif self.propagation_model == "heat_diffusion":
                self._heat_diffusion_step()
            else:  # hybrid
                self._hybrid_propagation_step()
            
            # Apply heat decay
            self.current_heat_vector *= self.heat_decay
            
            # Store heat state
            self.heat_history.append(self.current_heat_vector.copy())
            
            # Check convergence
            heat_change = np.linalg.norm(self.current_heat_vector - previous_heat)
            if heat_change < self.convergence_threshold:
                self.iteration_count = iteration + 1
                break
        else:
            self.iteration_count = max_steps
        
        # Generate results
        results = {}
        for i, symbol in enumerate(nodes):
            if symbol in self.heat_nodes:
                # Calculate influence sources
                influence_sources = self._calculate_influence_sources(symbol, i)
                
                # Calculate viral coefficient
                viral_coefficient = self._calculate_viral_coefficient(symbol, i)
                
                # Extract heat trajectory
                heat_trajectory = [heat_vec[i] for heat_vec in self.heat_history]
                
                result = HeatPropagationResult(
                    symbol=symbol,
                    initial_heat=initial_heat[i],
                    final_heat=self.current_heat_vector[i],
                    heat_change=self.current_heat_vector[i] - initial_heat[i],
                    propagation_steps=self.iteration_count,
                    influence_sources=influence_sources,
                    viral_coefficient=viral_coefficient,
                    heat_trajectory=heat_trajectory,
                    convergence_reached=self.iteration_count < max_steps,
                    prediction_confidence=self._calculate_prediction_confidence(symbol, i),
                    timestamp=datetime.now()
                )
                
                results[symbol] = result
                
                # Update node heat
                self.heat_nodes[symbol].current_heat = self.current_heat_vector[i]
        
        # Update performance stats
        self._update_performance_stats()
        
        return results
    
    def _independent_cascade_step(self):
        """Independent Cascade Model propagation step"""
        nodes = list(self.graph.nodes())
        new_heat = self.current_heat_vector.copy()
        
        for i, node in enumerate(nodes):
            if self.current_heat_vector[i] > 0:
                # This node can spread heat to neighbors
                for neighbor in self.graph.successors(node):
                    j = nodes.index(neighbor)
                    edge_data = self.graph[node][neighbor]
                    weight = edge_data['weight']
                    
                    # Probability of activation/heat transfer
                    transfer_prob = weight * self.current_heat_vector[i]
                    
                    # Random activation with probability
                    if np.random.random() < transfer_prob:
                        heat_transfer = self.current_heat_vector[i] * weight * 0.5
                        new_heat[j] += heat_transfer
        
        self.current_heat_vector = new_heat
    
    def _linear_threshold_step(self):
        """Linear Threshold Model propagation step"""
        nodes = list(self.graph.nodes())
        new_heat = np.zeros_like(self.current_heat_vector)
        
        for i, node in enumerate(nodes):
            # Calculate weighted influence from neighbors
            total_influence = 0
            for predecessor in self.graph.predecessors(node):
                j = nodes.index(predecessor)
                edge_data = self.graph[predecessor][node]
                weight = edge_data['weight']
                total_influence += weight * self.current_heat_vector[j]
            
            # Heat capacity acts as threshold
            heat_capacity = self.heat_nodes[node].heat_capacity
            threshold = heat_capacity * 0.5
            
            if total_influence > threshold:
                new_heat[i] = min(total_influence, heat_capacity)
            else:
                new_heat[i] = self.current_heat_vector[i] * 0.9  # Decay
        
        self.current_heat_vector = new_heat
    
    def _heat_diffusion_step(self):
        """Heat diffusion propagation step"""
        # Simple heat diffusion: H(t+1) = α * A * H(t) + (1-α) * H(0)
        alpha = self.damping_factor
        
        # Matrix multiplication for heat diffusion
        diffused_heat = self.transition_matrix.dot(self.current_heat_vector)
        
        # Combine with original heat (random walk with restart)
        initial_heat = self.heat_history[0] if self.heat_history else self.current_heat_vector
        self.current_heat_vector = alpha * diffused_heat + (1 - alpha) * initial_heat
    
    def _hybrid_propagation_step(self):
        """Hybrid propagation combining multiple models"""
        # Store current state
        original_heat = self.current_heat_vector.copy()
        
        # Apply each model with different weights
        # Heat diffusion (40%)
        self._heat_diffusion_step()
        diffusion_result = self.current_heat_vector.copy()
        
        # ICM (30%)
        self.current_heat_vector = original_heat.copy()
        self._independent_cascade_step()
        icm_result = self.current_heat_vector.copy()
        
        # LTM (30%)
        self.current_heat_vector = original_heat.copy()
        self._linear_threshold_step()
        ltm_result = self.current_heat_vector.copy()
        
        # Weighted combination
        self.current_heat_vector = (
            0.4 * diffusion_result +
            0.3 * icm_result +
            0.3 * ltm_result
        )
    
    def _calculate_influence_sources(self, symbol: str, node_idx: int) -> Dict[str, float]:
        """Calculate which nodes influenced this node's heat"""
        influences = {}
        nodes = list(self.graph.nodes())
        
        for predecessor in self.graph.predecessors(symbol):
            pred_idx = nodes.index(predecessor)
            edge_data = self.graph[predecessor][symbol]
            weight = edge_data['weight']
            
            # Influence = weight * predecessor_heat
            influence = weight * self.current_heat_vector[pred_idx]
            influences[predecessor] = influence
        
        return influences
    
    def _calculate_viral_coefficient(self, symbol: str, node_idx: int) -> float:
        """Calculate how viral this node's heat propagation is"""
        # Viral coefficient based on:
        # 1. Number of connections (degree centrality)
        # 2. Strength of connections (weighted degree)
        # 3. Position in network (betweenness centrality)
        
        try:
            degree_centrality = nx.degree_centrality(self.graph)[symbol]
            betweenness_centrality = nx.betweenness_centrality(self.graph).get(symbol, 0)
            
            # Weighted degree
            out_degree = self.graph.out_degree(symbol, weight='weight')
            
            # Combine metrics
            viral_coefficient = (
                0.4 * degree_centrality +
                0.3 * (out_degree / 10) +  # Normalize weighted degree
                0.3 * betweenness_centrality
            )
            
            return min(1.0, viral_coefficient)
            
        except Exception:
            return 0.5  # Default moderate virality
    
    def _calculate_prediction_confidence(self, symbol: str, node_idx: int) -> float:
        """Calculate prediction confidence for this node"""
        # Confidence based on:
        # 1. Network connectivity
        # 2. Heat stability
        # 3. Model convergence
        
        # Connectivity factor
        degree = self.graph.degree(symbol)
        connectivity_factor = min(1.0, degree / 10)
        
        # Heat stability (less variance = higher confidence)
        if len(self.heat_history) > 1:
            heat_trajectory = [heat_vec[node_idx] for heat_vec in self.heat_history]
            heat_variance = np.var(heat_trajectory[-5:])  # Last 5 steps
            stability_factor = 1.0 / (1.0 + heat_variance * 10)
        else:
            stability_factor = 0.5
        
        # Convergence factor
        convergence_factor = 1.0 if self.iteration_count < self.max_iterations else 0.7
        
        confidence = (
            0.4 * connectivity_factor +
            0.4 * stability_factor +
            0.2 * convergence_factor
        )
        
        return max(0.1, min(1.0, confidence))
    
    def _update_performance_stats(self):
        """Update performance tracking statistics"""
        self.propagation_stats['total_propagations'] += 1
        
        # Update average iterations
        total_iterations = (
            self.propagation_stats['avg_iterations'] * 
            (self.propagation_stats['total_propagations'] - 1) +
            self.iteration_count
        )
        self.propagation_stats['avg_iterations'] = total_iterations / self.propagation_stats['total_propagations']
        
        # Update convergence rate
        if self.iteration_count < self.max_iterations:
            converged = 1
        else:
            converged = 0
        
        total_convergence = (
            self.propagation_stats['convergence_rate'] * 
            (self.propagation_stats['total_propagations'] - 1) +
            converged
        )
        self.propagation_stats['convergence_rate'] = total_convergence / self.propagation_stats['total_propagations']
    
    def get_top_heat_nodes(self, n: int = 10, node_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get top N nodes by heat value"""
        nodes = list(self.graph.nodes())
        
        if self.current_heat_vector is None:
            return []
        
        node_heats = []
        for i, symbol in enumerate(nodes):
            if symbol in self.heat_nodes:
                node = self.heat_nodes[symbol]
                if node_type is None or node.node_type == node_type:
                    node_heats.append((symbol, self.current_heat_vector[i]))
        
        # Sort by heat value descending
        node_heats.sort(key=lambda x: x[1], reverse=True)
        
        return node_heats[:n]
    
    def analyze_heat_clusters(self) -> Dict[str, List[str]]:
        """Identify clusters of high-heat nodes"""
        if self.current_heat_vector is None:
            return {}
        
        nodes = list(self.graph.nodes())
        heat_threshold = np.percentile(self.current_heat_vector, 75)  # Top 25%
        
        high_heat_nodes = []
        for i, symbol in enumerate(nodes):
            if self.current_heat_vector[i] > heat_threshold:
                high_heat_nodes.append(symbol)
        
        # Find connected components among high-heat nodes
        subgraph = self.graph.subgraph(high_heat_nodes)
        clusters = {}
        
        for i, component in enumerate(nx.connected_components(subgraph.to_undirected())):
            clusters[f"cluster_{i}"] = list(component)
        
        return clusters
    
    def get_heat_flow_paths(self, source: str, target: str, max_paths: int = 3) -> List[List[str]]:
        """Find heat flow paths between two nodes"""
        try:
            paths = list(nx.shortest_simple_paths(self.graph, source, target))
            return paths[:max_paths]
        except nx.NetworkXNoPath:
            return []
    
    def export_network_data(self) -> Dict[str, any]:
        """Export network data for visualization"""
        nodes_data = []
        edges_data = []
        
        # Export nodes
        for symbol in self.graph.nodes():
            node_data = {
                'id': symbol,
                'label': symbol,
                'heat': self.current_heat_vector[list(self.graph.nodes()).index(symbol)] if self.current_heat_vector is not None else 0,
                'type': self.heat_nodes[symbol].node_type if symbol in self.heat_nodes else 'unknown',
                'capacity': self.heat_nodes[symbol].heat_capacity if symbol in self.heat_nodes else 1.0,
                'conductivity': self.heat_nodes[symbol].heat_conductivity if symbol in self.heat_nodes else 1.0
            }
            nodes_data.append(node_data)
        
        # Export edges
        for source, target, data in self.graph.edges(data=True):
            edge_data = {
                'source': source,
                'target': target,
                'weight': data.get('weight', 0.1),
                'type': 'heat_flow'
            }
            edges_data.append(edge_data)
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'stats': self.propagation_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_network(self):
        """Reset network to initial state"""
        self.current_heat_vector = None
        self.heat_history = []
        self.iteration_count = 0
        
        # Reset node heat values
        for node in self.heat_nodes.values():
            node.current_heat = 0.0
    
    def get_system_diagnostics(self) -> Dict[str, any]:
        """Get comprehensive system diagnostics"""
        return {
            'network_stats': {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
                'density': nx.density(self.graph),
                'connected_components': nx.number_connected_components(self.graph.to_undirected())
            },
            'propagation_stats': self.propagation_stats,
            'heat_stats': {
                'total_heat': float(np.sum(self.current_heat_vector)) if self.current_heat_vector is not None else 0,
                'max_heat': float(np.max(self.current_heat_vector)) if self.current_heat_vector is not None else 0,
                'avg_heat': float(np.mean(self.current_heat_vector)) if self.current_heat_vector is not None else 0,
                'heat_std': float(np.std(self.current_heat_vector)) if self.current_heat_vector is not None else 0
            },
            'algorithm_config': {
                'propagation_model': self.propagation_model,
                'convergence_threshold': self.convergence_threshold,
                'max_iterations': self.max_iterations,
                'damping_factor': self.damping_factor,
                'heat_decay': self.heat_decay
            }
        }