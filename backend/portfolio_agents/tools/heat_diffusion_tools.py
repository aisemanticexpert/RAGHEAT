"""
Heat Diffusion Tools for Portfolio Construction
=============================================

Tools for modeling heat diffusion and influence propagation in financial networks.
Based on physics-inspired heat equation models applied to financial graphs.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crewai_tools import BaseTool
from loguru import logger
import networkx as nx
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

class HeatEquationSolverTool(BaseTool):
    """Tool for solving heat diffusion equations on financial graphs."""
    
    name: str = "heat_equation_solver"
    description: str = "Solve heat diffusion equations to model influence propagation in financial networks"
    
    def __init__(self):
        super().__init__()
        self.default_diffusion_coefficient = 0.1
        self.default_time_steps = 50
        self.convergence_threshold = 1e-6
    
    def _run(self, graph: Dict[str, Any], initial_heat: Dict[str, float], 
             diffusion_coefficient: float = None, time_steps: int = None) -> Dict[str, Any]:
        """Solve heat diffusion equation on the given graph."""
        try:
            # Set default parameters
            beta = diffusion_coefficient or self.default_diffusion_coefficient
            steps = time_steps or self.default_time_steps
            
            # Build graph from input data
            nx_graph = self._build_networkx_graph(graph)
            
            if nx_graph.number_of_nodes() == 0:
                return {'error': 'Empty graph provided'}
            
            # Initialize heat values
            heat_vector = self._initialize_heat_vector(nx_graph, initial_heat)
            
            # Solve heat equation
            heat_evolution = self._solve_heat_diffusion(nx_graph, heat_vector, beta, steps)
            
            # Calculate final heat distribution
            final_heat = heat_evolution[-1]
            
            # Analyze heat propagation patterns
            propagation_analysis = self._analyze_heat_propagation(
                nx_graph, heat_evolution, initial_heat
            )
            
            return {
                'diffusion_results': {
                    'initial_heat': dict(zip(list(nx_graph.nodes()), heat_vector)),
                    'final_heat': dict(zip(list(nx_graph.nodes()), final_heat)),
                    'heat_evolution': self._format_heat_evolution(nx_graph, heat_evolution),
                    'propagation_analysis': propagation_analysis,
                    'parameters': {
                        'diffusion_coefficient': beta,
                        'time_steps': steps,
                        'convergence_achieved': propagation_analysis.get('converged', False)
                    }
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error solving heat equation: {e}")
            return {'error': str(e)}
    
    def _build_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """Build NetworkX graph from input data."""
        G = nx.Graph()
        
        # Add nodes
        if 'nodes' in graph_data:
            for node in graph_data['nodes']:
                node_id = node.get('id')
                if node_id:
                    G.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})
        
        # Add edges
        if 'edges' in graph_data:
            for edge in graph_data['edges']:
                source = edge.get('source')
                target = edge.get('target')
                weight = edge.get('weight', 1.0)
                
                if source and target:
                    G.add_edge(source, target, weight=weight)
        
        return G
    
    def _initialize_heat_vector(self, graph: nx.Graph, initial_heat: Dict[str, float]) -> np.ndarray:
        """Initialize heat vector for all nodes."""
        nodes = list(graph.nodes())
        heat_vector = np.zeros(len(nodes))
        
        for i, node in enumerate(nodes):
            heat_vector[i] = initial_heat.get(node, 0.0)
        
        return heat_vector
    
    def _solve_heat_diffusion(self, graph: nx.Graph, initial_heat: np.ndarray, 
                            beta: float, time_steps: int) -> List[np.ndarray]:
        """Solve heat diffusion using finite difference method."""
        # Get graph Laplacian
        laplacian = self._compute_graph_laplacian(graph)
        
        # Initialize heat evolution storage
        heat_evolution = [initial_heat.copy()]
        current_heat = initial_heat.copy()
        
        # Time step size
        dt = 0.01
        
        # Diffusion matrix (I - beta * dt * L)
        n = len(current_heat)
        identity = sparse.eye(n)
        diffusion_matrix = identity - beta * dt * laplacian
        
        # Solve heat equation iteratively
        for step in range(time_steps):
            # Implicit Euler method: (I - beta * dt * L) * heat_new = heat_old
            try:
                current_heat = spsolve(diffusion_matrix, current_heat)
                heat_evolution.append(current_heat.copy())
                
                # Check for convergence
                if step > 10:
                    heat_diff = np.linalg.norm(heat_evolution[-1] - heat_evolution[-2])
                    if heat_diff < self.convergence_threshold:
                        logger.info(f"Heat diffusion converged at step {step}")
                        break
                        
            except Exception as e:
                logger.warning(f"Error in diffusion step {step}: {e}")
                break
        
        return heat_evolution
    
    def _compute_graph_laplacian(self, graph: nx.Graph) -> sparse.csc_matrix:
        """Compute the graph Laplacian matrix."""
        try:
            # Get adjacency matrix with weights
            adjacency = nx.adjacency_matrix(graph, weight='weight')
            
            # Compute degree matrix
            degrees = np.array(adjacency.sum(axis=1)).flatten()
            degree_matrix = sparse.diags(degrees)
            
            # Laplacian = Degree - Adjacency
            laplacian = degree_matrix - adjacency
            
            return laplacian.tocsc()
            
        except Exception as e:
            logger.error(f"Error computing Laplacian: {e}")
            # Return identity matrix as fallback
            n = graph.number_of_nodes()
            return sparse.eye(n).tocsc()
    
    def _analyze_heat_propagation(self, graph: nx.Graph, heat_evolution: List[np.ndarray], 
                                initial_heat: Dict[str, float]) -> Dict[str, Any]:
        """Analyze heat propagation patterns."""
        nodes = list(graph.nodes())
        
        # Identify heat sources (initial non-zero heat)
        heat_sources = [node for node, heat in initial_heat.items() if heat > 0]
        
        # Calculate heat diffusion metrics
        initial_state = heat_evolution[0]
        final_state = heat_evolution[-1]
        
        # Heat conservation
        total_initial_heat = np.sum(initial_state)
        total_final_heat = np.sum(final_state)
        heat_conservation = abs(total_final_heat - total_initial_heat) / max(total_initial_heat, 1e-10)
        
        # Identify most heated nodes
        final_heat_ranking = sorted(
            [(nodes[i], final_state[i]) for i in range(len(nodes))],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate heat spread metrics
        non_zero_initial = np.count_nonzero(initial_state)
        non_zero_final = np.count_nonzero(final_state > 1e-10)
        
        # Check convergence
        converged = len(heat_evolution) < self.default_time_steps
        
        return {
            'heat_sources': heat_sources,
            'total_heat_conservation': float(heat_conservation),
            'heat_spread': {
                'initial_heated_nodes': int(non_zero_initial),
                'final_heated_nodes': int(non_zero_final),
                'spread_factor': float(non_zero_final / max(non_zero_initial, 1))
            },
            'top_heated_nodes': [(node, float(heat)) for node, heat in final_heat_ranking[:10]],
            'converged': converged,
            'convergence_iterations': len(heat_evolution) - 1
        }
    
    def _format_heat_evolution(self, graph: nx.Graph, heat_evolution: List[np.ndarray]) -> Dict[str, List[float]]:
        """Format heat evolution for output."""
        nodes = list(graph.nodes())
        formatted_evolution = {}
        
        # Sample every 5th step to reduce output size
        sample_steps = list(range(0, len(heat_evolution), max(1, len(heat_evolution) // 10)))
        
        for node_idx, node in enumerate(nodes):
            formatted_evolution[node] = [
                float(heat_evolution[step][node_idx]) for step in sample_steps
            ]
        
        return formatted_evolution

class GraphLaplacianCalculatorTool(BaseTool):
    """Tool for calculating graph Laplacian matrices and their properties."""
    
    name: str = "graph_laplacian_calculator"
    description: str = "Calculate graph Laplacian matrices and analyze their spectral properties"
    
    def _run(self, graph: Dict[str, Any], laplacian_type: str = "combinatorial") -> Dict[str, Any]:
        """Calculate graph Laplacian and its properties."""
        try:
            # Build NetworkX graph
            nx_graph = self._build_networkx_graph(graph)
            
            if nx_graph.number_of_nodes() == 0:
                return {'error': 'Empty graph provided'}
            
            # Calculate different types of Laplacians
            laplacian_matrices = self._calculate_laplacian_matrices(nx_graph, laplacian_type)
            
            # Analyze spectral properties
            spectral_analysis = self._analyze_spectral_properties(laplacian_matrices['main_laplacian'])
            
            # Calculate connectivity metrics
            connectivity_metrics = self._calculate_connectivity_metrics(nx_graph)
            
            return {
                'laplacian_analysis': {
                    'graph_size': {
                        'nodes': nx_graph.number_of_nodes(),
                        'edges': nx_graph.number_of_edges()
                    },
                    'laplacian_properties': laplacian_matrices,
                    'spectral_analysis': spectral_analysis,
                    'connectivity_metrics': connectivity_metrics
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating graph Laplacian: {e}")
            return {'error': str(e)}
    
    def _build_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """Build NetworkX graph from input data."""
        G = nx.Graph()
        
        # Add nodes
        if 'nodes' in graph_data:
            for node in graph_data['nodes']:
                node_id = node.get('id')
                if node_id:
                    G.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})
        
        # Add edges
        if 'edges' in graph_data:
            for edge in graph_data['edges']:
                source = edge.get('source')
                target = edge.get('target')
                weight = edge.get('weight', 1.0)
                
                if source and target and weight > 0:
                    G.add_edge(source, target, weight=weight)
        
        return G
    
    def _calculate_laplacian_matrices(self, graph: nx.Graph, laplacian_type: str) -> Dict[str, Any]:
        """Calculate different types of Laplacian matrices."""
        try:
            # Combinatorial Laplacian (L = D - A)
            combinatorial_laplacian = nx.laplacian_matrix(graph, weight='weight').toarray()
            
            # Normalized Laplacian (L_norm = D^(-1/2) * L * D^(-1/2))
            normalized_laplacian = nx.normalized_laplacian_matrix(graph, weight='weight').toarray()
            
            # Random walk Laplacian (L_rw = D^(-1) * L)
            try:
                adjacency = nx.adjacency_matrix(graph, weight='weight')
                degrees = np.array(adjacency.sum(axis=1)).flatten()
                # Avoid division by zero
                degrees[degrees == 0] = 1
                degree_inv = sparse.diags(1.0 / degrees)
                random_walk_laplacian = (degree_inv @ (sparse.diags(degrees) - adjacency)).toarray()
            except:
                random_walk_laplacian = combinatorial_laplacian
            
            # Select main Laplacian based on type
            if laplacian_type == "normalized":
                main_laplacian = normalized_laplacian
            elif laplacian_type == "random_walk":
                main_laplacian = random_walk_laplacian
            else:
                main_laplacian = combinatorial_laplacian
            
            return {
                'main_laplacian': main_laplacian,
                'combinatorial_laplacian': combinatorial_laplacian,
                'normalized_laplacian': normalized_laplacian,
                'random_walk_laplacian': random_walk_laplacian,
                'laplacian_type_used': laplacian_type
            }
            
        except Exception as e:
            logger.error(f"Error calculating Laplacian matrices: {e}")
            return {'error': str(e)}
    
    def _analyze_spectral_properties(self, laplacian: np.ndarray) -> Dict[str, Any]:
        """Analyze spectral properties of the Laplacian matrix."""
        try:
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            
            # Sort eigenvalues
            sorted_indices = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            # Analyze eigenvalue distribution
            spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
            algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0
            
            return {
                'eigenvalues': eigenvalues.tolist()[:10],  # First 10 eigenvalues
                'spectral_gap': float(spectral_gap),
                'algebraic_connectivity': float(algebraic_connectivity),
                'condition_number': float(eigenvalues[-1] / max(eigenvalues[1], 1e-10)) if len(eigenvalues) > 1 else 1.0,
                'rank': int(np.linalg.matrix_rank(laplacian)),
                'trace': float(np.trace(laplacian)),
                'determinant': float(np.linalg.det(laplacian)) if laplacian.shape[0] <= 100 else 0.0  # Avoid computation for large matrices
            }
            
        except Exception as e:
            logger.error(f"Error in spectral analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_connectivity_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate graph connectivity metrics."""
        try:
            # Basic connectivity
            is_connected = nx.is_connected(graph)
            num_components = nx.number_connected_components(graph)
            
            # Connectivity measures
            if is_connected and graph.number_of_nodes() > 1:
                node_connectivity = nx.node_connectivity(graph)
                edge_connectivity = nx.edge_connectivity(graph)
            else:
                node_connectivity = 0
                edge_connectivity = 0
            
            # Diameter and average path length
            if is_connected:
                try:
                    diameter = nx.diameter(graph)
                    avg_path_length = nx.average_shortest_path_length(graph, weight='weight')
                except:
                    diameter = 0
                    avg_path_length = 0
            else:
                diameter = float('inf')
                avg_path_length = float('inf')
            
            return {
                'is_connected': is_connected,
                'number_of_components': num_components,
                'node_connectivity': node_connectivity,
                'edge_connectivity': edge_connectivity,
                'diameter': diameter if diameter != float('inf') else -1,
                'average_path_length': avg_path_length if avg_path_length != float('inf') else -1,
                'density': nx.density(graph),
                'clustering_coefficient': nx.average_clustering(graph, weight='weight')
            }
            
        except Exception as e:
            logger.error(f"Error calculating connectivity metrics: {e}")
            return {'error': str(e)}

class DiffusionSimulatorTool(BaseTool):
    """Tool for simulating various diffusion processes on financial networks."""
    
    name: str = "diffusion_simulator"
    description: str = "Simulate different diffusion processes including heat, influence, and information propagation"
    
    def _run(self, graph: Dict[str, Any], simulation_type: str = "heat", 
             parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run diffusion simulation."""
        try:
            # Build graph
            nx_graph = self._build_networkx_graph(graph)
            
            if nx_graph.number_of_nodes() == 0:
                return {'error': 'Empty graph provided'}
            
            # Set default parameters
            params = parameters or {}
            
            # Run appropriate simulation
            if simulation_type == "heat":
                results = self._simulate_heat_diffusion(nx_graph, params)
            elif simulation_type == "influence":
                results = self._simulate_influence_propagation(nx_graph, params)
            elif simulation_type == "information":
                results = self._simulate_information_diffusion(nx_graph, params)
            else:
                return {'error': f'Unknown simulation type: {simulation_type}'}
            
            return {
                'simulation_results': results,
                'simulation_type': simulation_type,
                'graph_properties': {
                    'nodes': nx_graph.number_of_nodes(),
                    'edges': nx_graph.number_of_edges()
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in diffusion simulation: {e}")
            return {'error': str(e)}
    
    def _build_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """Build NetworkX graph from input data."""
        G = nx.Graph()
        
        # Add nodes with attributes
        if 'nodes' in graph_data:
            for node in graph_data['nodes']:
                node_id = node.get('id')
                if node_id:
                    G.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})
        
        # Add edges with weights
        if 'edges' in graph_data:
            for edge in graph_data['edges']:
                source = edge.get('source')
                target = edge.get('target')
                weight = edge.get('weight', 1.0)
                
                if source and target:
                    G.add_edge(source, target, weight=weight)
        
        return G
    
    def _simulate_heat_diffusion(self, graph: nx.Graph, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate heat diffusion process."""
        # Parameters
        initial_sources = parameters.get('heat_sources', {})
        diffusion_rate = parameters.get('diffusion_rate', 0.1)
        time_steps = parameters.get('time_steps', 100)
        cooling_rate = parameters.get('cooling_rate', 0.0)
        
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        
        # Initialize heat
        heat = np.zeros(n_nodes)
        for i, node in enumerate(nodes):
            heat[i] = initial_sources.get(node, 0.0)
        
        # Adjacency matrix
        adjacency = nx.adjacency_matrix(graph, nodelist=nodes, weight='weight').toarray()
        
        # Degree matrix
        degrees = np.sum(adjacency, axis=1)
        degrees[degrees == 0] = 1  # Avoid division by zero
        
        # Simulation
        heat_history = [heat.copy()]
        
        for step in range(time_steps):
            new_heat = heat.copy()
            
            # Heat diffusion
            for i in range(n_nodes):
                if degrees[i] > 0:
                    # Heat flows to neighbors
                    outflow = diffusion_rate * heat[i]
                    new_heat[i] -= outflow
                    
                    # Distribute heat to neighbors
                    for j in range(n_nodes):
                        if adjacency[i, j] > 0:
                            flow = outflow * (adjacency[i, j] / degrees[i])
                            new_heat[j] += flow
            
            # Apply cooling
            new_heat *= (1 - cooling_rate)
            
            heat = new_heat
            heat_history.append(heat.copy())
            
            # Check for convergence
            if step > 10:
                change = np.linalg.norm(heat_history[-1] - heat_history[-2])
                if change < 1e-6:
                    break
        
        # Calculate results
        final_heat = dict(zip(nodes, heat))
        heat_ranking = sorted(final_heat.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'final_heat_distribution': final_heat,
            'heat_ranking': heat_ranking[:20],  # Top 20
            'simulation_steps': len(heat_history) - 1,
            'total_heat_remaining': float(np.sum(heat)),
            'heat_spread_nodes': int(np.sum(heat > 1e-6)),
            'parameters_used': {
                'diffusion_rate': diffusion_rate,
                'cooling_rate': cooling_rate,
                'time_steps': time_steps
            }
        }
    
    def _simulate_influence_propagation(self, graph: nx.Graph, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate influence propagation (cascade model)."""
        # Parameters
        initial_influencers = parameters.get('influencers', {})
        influence_threshold = parameters.get('threshold', 0.5)
        influence_decay = parameters.get('decay', 0.1)
        max_steps = parameters.get('max_steps', 50)
        
        nodes = list(graph.nodes())
        
        # Initialize influence
        influence = {}
        activated = set()
        
        for node in nodes:
            influence[node] = initial_influencers.get(node, 0.0)
            if influence[node] > influence_threshold:
                activated.add(node)
        
        # Simulation steps
        activation_history = []
        
        for step in range(max_steps):
            step_activations = set()
            new_influence = influence.copy()
            
            # Propagate influence from activated nodes
            for node in activated:
                for neighbor in graph.neighbors(node):
                    if neighbor not in activated:
                        # Calculate influence received
                        edge_weight = graph[node][neighbor].get('weight', 1.0)
                        influence_received = influence[node] * edge_weight * (1 - influence_decay)
                        
                        new_influence[neighbor] += influence_received
                        
                        # Check activation threshold
                        if new_influence[neighbor] > influence_threshold:
                            step_activations.add(neighbor)
            
            # Update activated set
            activated.update(step_activations)
            influence = new_influence
            activation_history.append(len(activated))
            
            # Stop if no new activations
            if len(step_activations) == 0:
                break
        
        return {
            'final_influence': influence,
            'activated_nodes': list(activated),
            'activation_count': len(activated),
            'activation_rate': len(activated) / len(nodes),
            'activation_history': activation_history,
            'simulation_steps': len(activation_history),
            'influence_ranking': sorted(influence.items(), key=lambda x: x[1], reverse=True)[:20]
        }
    
    def _simulate_information_diffusion(self, graph: nx.Graph, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate information diffusion (epidemic model)."""
        # Parameters (SIR model)
        initial_informed = parameters.get('informed', {})
        transmission_rate = parameters.get('transmission_rate', 0.3)
        recovery_rate = parameters.get('recovery_rate', 0.1)
        max_steps = parameters.get('max_steps', 100)
        
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        
        # Initialize states: 0=Susceptible, 1=Informed, 2=Recovered
        states = np.zeros(n_nodes, dtype=int)
        
        for i, node in enumerate(nodes):
            if node in initial_informed:
                states[i] = 1
        
        # Simulation
        history = {'susceptible': [], 'informed': [], 'recovered': []}
        
        for step in range(max_steps):
            new_states = states.copy()
            
            # Information transmission
            for i in range(n_nodes):
                if states[i] == 0:  # Susceptible
                    # Check informed neighbors
                    informed_neighbors = 0
                    for j in graph.neighbors(nodes[i]):
                        j_idx = nodes.index(j)
                        if states[j_idx] == 1:
                            informed_neighbors += 1
                    
                    # Probability of getting informed
                    if informed_neighbors > 0:
                        infection_prob = 1 - (1 - transmission_rate) ** informed_neighbors
                        if np.random.random() < infection_prob:
                            new_states[i] = 1
                
                elif states[i] == 1:  # Informed
                    # Recovery (stop spreading)
                    if np.random.random() < recovery_rate:
                        new_states[i] = 2
            
            states = new_states
            
            # Record history
            history['susceptible'].append(np.sum(states == 0))
            history['informed'].append(np.sum(states == 1))
            history['recovered'].append(np.sum(states == 2))
            
            # Stop if no informed nodes
            if np.sum(states == 1) == 0:
                break
        
        return {
            'final_states': {
                'susceptible': int(np.sum(states == 0)),
                'informed': int(np.sum(states == 1)),
                'recovered': int(np.sum(states == 2))
            },
            'evolution_history': history,
            'total_informed': int(np.sum(states >= 1)),
            'information_reach': float(np.sum(states >= 1) / n_nodes),
            'simulation_steps': len(history['informed']),
            'parameters_used': {
                'transmission_rate': transmission_rate,
                'recovery_rate': recovery_rate
            }
        }

class InfluencePropagatorTool(BaseTool):
    """Tool for modeling influence propagation in financial networks."""
    
    name: str = "influence_propagator"
    description: str = "Model how financial events and shocks influence other entities in the network"
    
    def _run(self, graph: Dict[str, Any], events: List[Dict[str, Any]], 
             propagation_model: str = "linear") -> Dict[str, Any]:
        """Propagate influence from events through the network."""
        try:
            # Build graph
            nx_graph = self._build_networkx_graph(graph)
            
            if nx_graph.number_of_nodes() == 0:
                return {'error': 'Empty graph provided'}
            
            # Process events and calculate influence
            influence_results = []
            
            for event in events:
                event_result = self._propagate_single_event(nx_graph, event, propagation_model)
                influence_results.append(event_result)
            
            # Combine influence from multiple events
            combined_influence = self._combine_influences(nx_graph, influence_results)
            
            return {
                'influence_propagation': {
                    'individual_events': influence_results,
                    'combined_influence': combined_influence,
                    'propagation_summary': self._summarize_propagation(combined_influence)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in influence propagation: {e}")
            return {'error': str(e)}
    
    def _build_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """Build NetworkX graph from input data."""
        G = nx.Graph()
        
        if 'nodes' in graph_data:
            for node in graph_data['nodes']:
                node_id = node.get('id')
                if node_id:
                    G.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})
        
        if 'edges' in graph_data:
            for edge in graph_data['edges']:
                source = edge.get('source')
                target = edge.get('target')
                weight = edge.get('weight', 1.0)
                
                if source and target:
                    G.add_edge(source, target, weight=weight)
        
        return G
    
    def _propagate_single_event(self, graph: nx.Graph, event: Dict[str, Any], 
                               propagation_model: str) -> Dict[str, Any]:
        """Propagate influence from a single event."""
        event_id = event.get('id', 'unknown_event')
        event_impact = event.get('impact', 1.0)
        affected_entities = event.get('affected_entities', [])
        
        # Initialize influence
        nodes = list(graph.nodes())
        influence = {node: 0.0 for node in nodes}
        
        # Set initial influence on directly affected entities
        for entity in affected_entities:
            if entity in influence:
                influence[entity] = event_impact
        
        # Propagate based on model
        if propagation_model == "linear":
            final_influence = self._linear_propagation(graph, influence)
        elif propagation_model == "exponential_decay":
            final_influence = self._exponential_decay_propagation(graph, influence)
        elif propagation_model == "threshold":
            final_influence = self._threshold_propagation(graph, influence, event.get('threshold', 0.5))
        else:
            final_influence = influence
        
        # Calculate propagation metrics
        propagation_metrics = self._calculate_propagation_metrics(influence, final_influence)
        
        return {
            'event_id': event_id,
            'initial_influence': influence,
            'final_influence': final_influence,
            'propagation_metrics': propagation_metrics,
            'model_used': propagation_model
        }
    
    def _linear_propagation(self, graph: nx.Graph, initial_influence: Dict[str, float], 
                          decay_factor: float = 0.8, max_hops: int = 3) -> Dict[str, float]:
        """Linear influence propagation with distance decay."""
        current_influence = initial_influence.copy()
        
        for hop in range(max_hops):
            new_influence = current_influence.copy()
            
            for node in graph.nodes():
                if current_influence[node] > 0:
                    # Propagate to neighbors
                    for neighbor in graph.neighbors(node):
                        edge_weight = graph[node][neighbor].get('weight', 1.0)
                        influence_transfer = current_influence[node] * edge_weight * (decay_factor ** (hop + 1))
                        new_influence[neighbor] = max(new_influence[neighbor], influence_transfer)
            
            current_influence = new_influence
        
        return current_influence
    
    def _exponential_decay_propagation(self, graph: nx.Graph, initial_influence: Dict[str, float],
                                     decay_rate: float = 0.2) -> Dict[str, float]:
        """Exponential decay propagation."""
        # Calculate shortest paths from all influenced nodes
        influenced_nodes = [node for node, inf in initial_influence.items() if inf > 0]
        final_influence = initial_influence.copy()
        
        for source in influenced_nodes:
            if source in graph:
                try:
                    paths = nx.single_source_shortest_path_length(graph, source)
                    
                    for target, distance in paths.items():
                        if distance > 0:  # Don't update source
                            influence_at_distance = initial_influence[source] * np.exp(-decay_rate * distance)
                            final_influence[target] = max(final_influence[target], influence_at_distance)
                except:
                    continue
        
        return final_influence
    
    def _threshold_propagation(self, graph: nx.Graph, initial_influence: Dict[str, float],
                             threshold: float = 0.5) -> Dict[str, float]:
        """Threshold-based propagation (cascade model)."""
        current_influence = initial_influence.copy()
        activated = set(node for node, inf in initial_influence.items() if inf >= threshold)
        
        changed = True
        while changed:
            changed = False
            new_activated = set()
            
            for node in graph.nodes():
                if node not in activated:
                    # Calculate influence from activated neighbors
                    neighbor_influence = 0
                    for neighbor in graph.neighbors(node):
                        if neighbor in activated:
                            edge_weight = graph[node][neighbor].get('weight', 1.0)
                            neighbor_influence += current_influence[neighbor] * edge_weight
                    
                    # Check threshold
                    if neighbor_influence >= threshold:
                        new_activated.add(node)
                        current_influence[node] = neighbor_influence
                        changed = True
            
            activated.update(new_activated)
        
        return current_influence
    
    def _calculate_propagation_metrics(self, initial: Dict[str, float], 
                                     final: Dict[str, float]) -> Dict[str, Any]:
        """Calculate metrics for influence propagation."""
        initial_influenced = sum(1 for v in initial.values() if v > 0)
        final_influenced = sum(1 for v in final.values() if v > 0)
        
        total_initial_influence = sum(initial.values())
        total_final_influence = sum(final.values())
        
        return {
            'reach_expansion': final_influenced - initial_influenced,
            'influence_amplification': total_final_influence / max(total_initial_influence, 1e-10),
            'propagation_efficiency': (final_influenced / len(initial)) if len(initial) > 0 else 0,
            'top_influenced_entities': sorted(
                [(k, v) for k, v in final.items()], 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    def _combine_influences(self, graph: nx.Graph, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine influences from multiple events."""
        nodes = list(graph.nodes())
        combined_influence = {node: 0.0 for node in nodes}
        
        # Combine influences (additive model)
        for result in individual_results:
            final_influence = result['final_influence']
            for node in nodes:
                combined_influence[node] += final_influence.get(node, 0.0)
        
        return {
            'combined_influence': combined_influence,
            'top_influenced': sorted(
                combined_influence.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20],
            'total_system_influence': sum(combined_influence.values()),
            'influenced_entity_count': sum(1 for v in combined_influence.values() if v > 0.01)
        }
    
    def _summarize_propagation(self, combined_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate summary of propagation results."""
        top_influenced = combined_result['top_influenced']
        total_influence = combined_result['total_system_influence']
        
        if not top_influenced:
            return {'summary': 'No significant influence propagation detected.'}
        
        most_affected = top_influenced[0][0]
        most_affected_score = top_influenced[0][1]
        
        summary = f"Most affected entity: {most_affected} (influence: {most_affected_score:.3f}). "
        summary += f"Total system influence: {total_influence:.3f}. "
        summary += f"Number of significantly influenced entities: {combined_result['influenced_entity_count']}."
        
        return {'summary': summary}

class HeatKernelCalculatorTool(BaseTool):
    """Tool for calculating heat kernel on graphs for diffusion analysis."""
    
    name: str = "heat_kernel_calculator"
    description: str = "Calculate heat kernel for analyzing diffusion patterns and node similarities"
    
    def _run(self, graph: Dict[str, Any], time_parameter: float = 1.0, 
             kernel_type: str = "exponential") -> Dict[str, Any]:
        """Calculate heat kernel on the graph."""
        try:
            # Build NetworkX graph
            nx_graph = self._build_networkx_graph(graph)
            
            if nx_graph.number_of_nodes() == 0:
                return {'error': 'Empty graph provided'}
            
            # Calculate Laplacian matrix
            laplacian = nx.laplacian_matrix(nx_graph, weight='weight').toarray()
            
            # Calculate heat kernel
            heat_kernel = self._calculate_heat_kernel(laplacian, time_parameter, kernel_type)
            
            # Analyze kernel properties
            kernel_analysis = self._analyze_heat_kernel(heat_kernel, list(nx_graph.nodes()))
            
            return {
                'heat_kernel_analysis': {
                    'time_parameter': time_parameter,
                    'kernel_type': kernel_type,
                    'graph_size': nx_graph.number_of_nodes(),
                    'kernel_matrix_properties': kernel_analysis,
                    'node_similarities': self._calculate_node_similarities(heat_kernel, list(nx_graph.nodes()))
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating heat kernel: {e}")
            return {'error': str(e)}
    
    def _build_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """Build NetworkX graph from input data."""
        G = nx.Graph()
        
        if 'nodes' in graph_data:
            for node in graph_data['nodes']:
                node_id = node.get('id')
                if node_id:
                    G.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})
        
        if 'edges' in graph_data:
            for edge in graph_data['edges']:
                source = edge.get('source')
                target = edge.get('target')
                weight = edge.get('weight', 1.0)
                
                if source and target:
                    G.add_edge(source, target, weight=weight)
        
        return G
    
    def _calculate_heat_kernel(self, laplacian: np.ndarray, t: float, kernel_type: str) -> np.ndarray:
        """Calculate the heat kernel matrix."""
        try:
            if kernel_type == "exponential":
                # Heat kernel: K(t) = exp(-t * L)
                heat_kernel = scipy.linalg.expm(-t * laplacian)
            
            elif kernel_type == "polynomial":
                # Polynomial approximation: K(t) ≈ (I + t*L)^(-1)
                identity = np.eye(laplacian.shape[0])
                heat_kernel = np.linalg.inv(identity + t * laplacian)
            
            elif kernel_type == "diffusion":
                # Diffusion kernel: K(t) = (I - t*L)^(-1) for small t
                identity = np.eye(laplacian.shape[0])
                if t < 1.0:  # Ensure stability
                    heat_kernel = np.linalg.inv(identity - t * laplacian)
                else:
                    heat_kernel = scipy.linalg.expm(-t * laplacian)
            
            else:
                # Default to exponential
                heat_kernel = scipy.linalg.expm(-t * laplacian)
            
            return heat_kernel
            
        except Exception as e:
            logger.error(f"Error computing heat kernel: {e}")
            # Return identity matrix as fallback
            return np.eye(laplacian.shape[0])
    
    def _analyze_heat_kernel(self, kernel: np.ndarray, nodes: List[str]) -> Dict[str, Any]:
        """Analyze properties of the heat kernel."""
        try:
            n = kernel.shape[0]
            
            # Basic properties
            trace = np.trace(kernel)
            frobenius_norm = np.linalg.norm(kernel, 'fro')
            
            # Diagonal elements (self-diffusion)
            diagonal = np.diag(kernel)
            
            # Symmetry check
            is_symmetric = np.allclose(kernel, kernel.T)
            
            # Positive definiteness check
            eigenvalues = np.linalg.eigvals(kernel)
            is_positive_definite = np.all(eigenvalues > -1e-10)
            
            return {
                'trace': float(trace),
                'frobenius_norm': float(frobenius_norm),
                'is_symmetric': is_symmetric,
                'is_positive_definite': is_positive_definite,
                'min_eigenvalue': float(np.min(eigenvalues)),
                'max_eigenvalue': float(np.max(eigenvalues)),
                'condition_number': float(np.max(eigenvalues) / max(np.min(eigenvalues), 1e-10)),
                'diagonal_statistics': {
                    'mean': float(np.mean(diagonal)),
                    'std': float(np.std(diagonal)),
                    'min': float(np.min(diagonal)),
                    'max': float(np.max(diagonal))
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing heat kernel: {e}")
            return {'error': str(e)}
    
    def _calculate_node_similarities(self, kernel: np.ndarray, nodes: List[str]) -> Dict[str, Any]:
        """Calculate node similarities based on heat kernel."""
        try:
            n = len(nodes)
            similarities = {}
            
            # Calculate pairwise similarities
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    # Cosine similarity of heat kernel rows
                    if i != j:
                        row_i = kernel[i, :]
                        row_j = kernel[j, :]
                        
                        norm_i = np.linalg.norm(row_i)
                        norm_j = np.linalg.norm(row_j)
                        
                        if norm_i > 0 and norm_j > 0:
                            similarity = np.dot(row_i, row_j) / (norm_i * norm_j)
                            similarity_matrix[i, j] = similarity
            
            # Find most similar pairs
            similar_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    similarity = similarity_matrix[i, j]
                    similar_pairs.append((nodes[i], nodes[j], float(similarity)))
            
            # Sort by similarity
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            
            return {
                'most_similar_pairs': similar_pairs[:10],
                'average_similarity': float(np.mean(similarity_matrix[similarity_matrix > 0])) if np.any(similarity_matrix > 0) else 0.0,
                'similarity_distribution': {
                    'mean': float(np.mean(similarity_matrix)),
                    'std': float(np.std(similarity_matrix)),
                    'min': float(np.min(similarity_matrix)),
                    'max': float(np.max(similarity_matrix))
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating node similarities: {e}")
            return {'error': str(e)}

# Initialize tools
heat_equation_solver = HeatEquationSolverTool()
graph_laplacian_calculator = GraphLaplacianCalculatorTool()
diffusion_simulator = DiffusionSimulatorTool()
influence_propagator = InfluencePropagatorTool()
heat_kernel_calculator = HeatKernelCalculatorTool()