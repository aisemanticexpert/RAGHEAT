import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy import sparse
import logging

logger = logging.getLogger(__name__)

class HeatDiffusionEngine:
    '''
    Implements heat diffusion over the financial knowledge graph
    Based on the heat equation: dh(t)/dt = -βL·h(t)
    '''

    def __init__(self, graph: nx.DiGraph, beta: float = 0.85):
        self.graph = graph
        self.beta = beta  # Diffusion coefficient
        self.heat_values = {}
        self.node_list = list(graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}

    def compute_laplacian(self) -> np.ndarray:
        '''Compute the graph Laplacian matrix'''
        n = len(self.node_list)
        A = nx.adjacency_matrix(self.graph, nodelist=self.node_list)
        D = sparse.diags(A.sum(axis=1).A1)
        L = D - A
        return L

    def apply_heat_source(self, source_nodes: Dict[str, float]):
        '''
        Apply heat to source nodes (e.g., stocks affected by events)
        source_nodes: {node_id: heat_value}
        '''
        n = len(self.node_list)
        heat_vector = np.zeros(n)

        for node_id, heat_value in source_nodes.items():
            if node_id in self.node_to_idx:
                idx = self.node_to_idx[node_id]
                heat_vector[idx] = heat_value

        return heat_vector

    def propagate_heat(self, initial_heat: np.ndarray, time_steps: int = 5) -> np.ndarray:
        '''
        Propagate heat through the graph using iterative method
        '''
        L = self.compute_laplacian()

        # For numerical stability, use iterative approach
        heat = initial_heat.copy()
        dt = 0.1  # Time step

        for _ in range(time_steps):
            # Euler method: h(t+dt) = h(t) - β·dt·L·h(t)
            heat_change = -self.beta * dt * L.dot(heat)
            heat = heat + heat_change

            # Ensure non-negative heat values
            heat = np.maximum(heat, 0)

            # Normalize to prevent explosion
            if heat.max() > 0:
                heat = heat / heat.max()

        return heat

    def calculate_heat_distribution(
        self, 
        event_impacts: Dict[str, float],
        propagation_steps: int = 5
    ) -> Dict[str, float]:
        '''
        Calculate heat distribution across the graph given event impacts
        '''
        # Apply heat sources
        initial_heat = self.apply_heat_source(event_impacts)

        # Propagate heat
        final_heat = self.propagate_heat(initial_heat, propagation_steps)

        # Convert back to dictionary
        heat_distribution = {}
        for idx, node_id in enumerate(self.node_list):
            heat_distribution[node_id] = float(final_heat[idx])

        self.heat_values = heat_distribution
        return heat_distribution

    def get_heated_sectors(self, top_k: int = 3) -> List[Tuple[str, float]]:
        '''Get the most heated sectors'''
        sector_heats = {}

        for node_id, heat in self.heat_values.items():
            if node_id.startswith("SECTOR_"):
                sector_heats[node_id] = heat

        # Sort by heat score
        sorted_sectors = sorted(
            sector_heats.items(), 
            key=lambda x: x[1], 
            reverse=True
        )

        return sorted_sectors[:top_k]

    def get_heated_stocks_in_sector(
        self, 
        sector_id: str, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        '''Get the most heated stocks within a specific sector'''
        stock_heats = {}

        # Get all stocks in the sector
        sector_stocks = [
            n for n in self.graph.neighbors(sector_id)
            if n.startswith("STOCK_")
        ]

        for stock_id in sector_stocks:
            if stock_id in self.heat_values:
                stock_heats[stock_id] = self.heat_values[stock_id]

        # Sort by heat score
        sorted_stocks = sorted(
            stock_heats.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_stocks[:top_k]