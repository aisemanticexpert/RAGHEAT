"""
Physics-Informed Neural Network Heat Equation Engine
Revolutionary heat diffusion model for financial knowledge graphs
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import networkx as nx
from datetime import datetime
import logging
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HeatNode:
    """Advanced node representation for heat equation modeling"""
    id: str
    position: Tuple[float, float]
    temperature: float
    conductivity: float
    capacity: float
    market_value: float
    volatility: float
    sector: str
    connections: List[str]
    timestamp: datetime

@dataclass
class HeatEdge:
    """Heat conduction edge between nodes"""
    source: str
    target: str
    conductance: float
    resistance: float
    correlation: float
    information_flow: float
    timestamp: datetime

class PhysicsInformedHeatNetwork:
    """
    Revolutionary Physics-Informed Neural Network for Heat Diffusion
    Models market heat propagation using actual physics equations
    """
    
    def __init__(self, spatial_dim: int = 2, temporal_steps: int = 100):
        self.spatial_dim = spatial_dim
        self.temporal_steps = temporal_steps
        self.nodes: Dict[str, HeatNode] = {}
        self.edges: Dict[str, HeatEdge] = {}
        self.adjacency_matrix = None
        self.laplacian_matrix = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Physics constants for financial heat equation
        self.thermal_diffusivity = 0.1  # Market heat diffusion rate
        self.dt = 0.01  # Time step
        self.dx = 1.0   # Spatial discretization
        
        # Initialize neural network components
        self.heat_predictor = self._build_heat_predictor()
        self.optimizer = optim.Adam(self.heat_predictor.parameters(), lr=0.001)
        
        logger.info(f"🔥 Physics-Informed Heat Network initialized on {self.device}")

    def _build_heat_predictor(self) -> nn.Module:
        """Build physics-informed neural network for heat prediction"""
        
        class HeatEquationPINN(nn.Module):
            def __init__(self, input_dim=4, hidden_dim=128, output_dim=1):
                super().__init__()
                
                # Network architecture inspired by physics
                self.spatial_encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim//2),
                    nn.Tanh()
                )
                
                self.temporal_encoder = nn.Sequential(
                    nn.Linear(hidden_dim//2, hidden_dim//2),
                    nn.Tanh(),
                    nn.Linear(hidden_dim//2, hidden_dim//4),
                    nn.Tanh()
                )
                
                self.heat_predictor = nn.Sequential(
                    nn.Linear(hidden_dim//4, hidden_dim//4),
                    nn.Tanh(),
                    nn.Linear(hidden_dim//4, output_dim),
                    nn.Sigmoid()  # Temperature bounded [0,1]
                )
                
            def forward(self, x):
                # x: [batch_size, 4] -> [position_x, position_y, time, current_temp]
                spatial_features = self.spatial_encoder(x)
                temporal_features = self.temporal_encoder(spatial_features)
                temperature = self.heat_predictor(temporal_features)
                return temperature
                
            def physics_loss(self, x, y_pred):
                """Enforce physics constraints using heat equation"""
                x.requires_grad_(True)
                
                # Compute gradients for physics constraints
                u_t = torch.autograd.grad(y_pred.sum(), x, create_graph=True)[0][:, 2:3]  # ∂u/∂t
                
                u_x = torch.autograd.grad(y_pred.sum(), x, create_graph=True)[0][:, 0:1]   # ∂u/∂x
                u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]     # ∂²u/∂x²
                
                u_y = torch.autograd.grad(y_pred.sum(), x, create_graph=True)[0][:, 1:2]   # ∂u/∂y  
                u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1:2]     # ∂²u/∂y²
                
                # Heat equation: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
                alpha = 0.1  # Thermal diffusivity
                heat_equation = u_t - alpha * (u_xx + u_yy)
                
                return torch.mean(heat_equation**2)
                
        return HeatEquationPINN().to(self.device)

    def add_financial_node(self, symbol: str, market_data: Dict) -> None:
        """Add financial instrument as heat node"""
        
        # Physics-based positioning using market properties
        position = self._calculate_physics_position(market_data)
        
        # Calculate thermal properties from financial data
        temperature = self._market_to_temperature(market_data)
        conductivity = self._calculate_thermal_conductivity(market_data)
        capacity = self._calculate_heat_capacity(market_data)
        
        node = HeatNode(
            id=symbol,
            position=position,
            temperature=temperature,
            conductivity=conductivity,
            capacity=capacity,
            market_value=market_data.get('market_value', 0),
            volatility=market_data.get('volatility', 0),
            sector=market_data.get('sector', 'Unknown'),
            connections=[],
            timestamp=datetime.now()
        )
        
        self.nodes[symbol] = node
        logger.info(f"🌡️ Added thermal node {symbol} at T={temperature:.3f}")

    def _calculate_physics_position(self, market_data: Dict) -> Tuple[float, float]:
        """Calculate node position based on financial physics"""
        
        # Market cap influences x-position (larger = further right)
        market_cap = market_data.get('market_cap', 1e9)
        x = np.log10(market_cap) / 12.0  # Normalize to [0,1]
        
        # Volatility influences y-position (higher = further up)
        volatility = market_data.get('volatility', 0.2)
        y = min(volatility / 0.5, 1.0)  # Normalize to [0,1]
        
        # Add sector-based clustering
        sector = market_data.get('sector', 'Unknown')
        sector_offsets = {
            'Technology': (0.1, 0.8),
            'Financial': (0.8, 0.2),
            'Healthcare': (0.2, 0.9),
            'Energy': (0.9, 0.1),
            'Consumer': (0.5, 0.5)
        }
        
        if sector in sector_offsets:
            offset_x, offset_y = sector_offsets[sector]
            x = 0.7 * x + 0.3 * offset_x  # Blend individual and sector positioning
            y = 0.7 * y + 0.3 * offset_y
            
        return (float(x), float(y))

    def _market_to_temperature(self, market_data: Dict) -> float:
        """Convert market data to thermal temperature"""
        
        # Heat sources: volume, price change, news sentiment
        volume = market_data.get('volume', 0)
        price_change = abs(market_data.get('price_change', 0))
        sentiment = market_data.get('sentiment', 0)
        volatility = market_data.get('volatility', 0)
        
        # Normalize volume (log scale)
        volume_factor = min(np.log10(volume + 1) / 10.0, 1.0)
        
        # Price change contribution
        price_factor = min(price_change / 0.1, 1.0)  # 10% change = max heat
        
        # Sentiment contribution
        sentiment_factor = (sentiment + 1) / 2.0  # [-1,1] -> [0,1]
        
        # Volatility contribution  
        volatility_factor = min(volatility / 0.5, 1.0)
        
        # Combined temperature calculation
        temperature = (
            0.4 * volume_factor +
            0.3 * price_factor +
            0.2 * sentiment_factor + 
            0.1 * volatility_factor
        )
        
        return max(0.0, min(1.0, temperature))

    def _calculate_thermal_conductivity(self, market_data: Dict) -> float:
        """Calculate thermal conductivity from market liquidity"""
        
        volume = market_data.get('volume', 0)
        market_cap = market_data.get('market_cap', 1e9)
        
        # High volume + large cap = high conductivity
        liquidity_factor = np.log10(volume + 1) / 10.0
        size_factor = np.log10(market_cap) / 12.0
        
        conductivity = 0.6 * liquidity_factor + 0.4 * size_factor
        return max(0.01, min(1.0, conductivity))

    def _calculate_heat_capacity(self, market_data: Dict) -> float:
        """Calculate heat capacity from market stability"""
        
        volatility = market_data.get('volatility', 0.2)
        market_cap = market_data.get('market_cap', 1e9)
        
        # Large, stable stocks have high heat capacity
        stability_factor = max(0, 1.0 - volatility / 0.5)
        size_factor = np.log10(market_cap) / 12.0
        
        capacity = 0.7 * stability_factor + 0.3 * size_factor
        return max(0.1, min(2.0, capacity))

    def add_heat_connection(self, source: str, target: str, correlation: float) -> None:
        """Add thermal connection between nodes"""
        
        if source not in self.nodes or target not in self.nodes:
            logger.warning(f"Cannot connect {source}-{target}: nodes not found")
            return
            
        # Calculate thermal conductance from correlation
        conductance = self._correlation_to_conductance(correlation)
        resistance = 1.0 / (conductance + 1e-6)
        
        # Information flow based on market dynamics
        source_node = self.nodes[source]
        target_node = self.nodes[target]
        
        info_flow = self._calculate_information_flow(source_node, target_node, correlation)
        
        edge_id = f"{source}_{target}"
        edge = HeatEdge(
            source=source,
            target=target,
            conductance=conductance,
            resistance=resistance,
            correlation=correlation,
            information_flow=info_flow,
            timestamp=datetime.now()
        )
        
        self.edges[edge_id] = edge
        
        # Update node connections
        self.nodes[source].connections.append(target)
        self.nodes[target].connections.append(source)
        
        logger.info(f"🔗 Connected {source}-{target} with conductance {conductance:.3f}")

    def _correlation_to_conductance(self, correlation: float) -> float:
        """Convert financial correlation to thermal conductance"""
        
        # High correlation = high heat conductance
        abs_corr = abs(correlation)
        
        if abs_corr > 0.8:
            return 0.9 + 0.1 * abs_corr  # Very high conductance
        elif abs_corr > 0.5:
            return 0.5 + 0.4 * abs_corr  # Medium conductance
        else:
            return 0.1 + 0.4 * abs_corr  # Low conductance
            
    def _calculate_information_flow(self, source: HeatNode, target: HeatNode, correlation: float) -> float:
        """Calculate information flow rate between nodes"""
        
        # Flow depends on temperature difference and conductance
        temp_diff = abs(source.temperature - target.temperature)
        conductance = self._correlation_to_conductance(correlation)
        
        # Market size influences flow capacity
        size_factor = min(source.market_value, target.market_value) / 1e12  # Normalize to trillions
        
        flow = temp_diff * conductance * (1.0 + size_factor)
        return min(1.0, flow)

    def simulate_heat_diffusion(self, time_steps: int = 100) -> Dict:
        """Simulate heat diffusion across the financial network"""
        
        if not self.nodes:
            logger.warning("No nodes in network for simulation")
            return {}
            
        logger.info(f"🔬 Starting heat diffusion simulation ({time_steps} steps)")
        
        # Build network matrices
        self._build_network_matrices()
        
        # Initialize temperature vector
        n_nodes = len(self.nodes)
        node_ids = list(self.nodes.keys())
        temperatures = np.array([self.nodes[nid].temperature for nid in node_ids])
        
        # Store evolution
        evolution = {
            'timestamps': [],
            'temperatures': [],
            'heat_flows': [],
            'entropy': []
        }
        
        for step in range(time_steps):
            # Heat equation: dT/dt = α∇²T + sources - sinks
            laplacian_term = self.laplacian_matrix @ temperatures
            
            # Add external heat sources (market events)
            sources = self._calculate_heat_sources(node_ids, step)
            
            # Heat capacity affects temperature change rate
            capacities = np.array([self.nodes[nid].capacity for nid in node_ids])
            
            # Update temperatures using explicit Euler method
            dt = self.dt
            dT_dt = self.thermal_diffusivity * laplacian_term + sources
            temperatures += dt * dT_dt / capacities
            
            # Ensure physical bounds
            temperatures = np.clip(temperatures, 0.0, 1.0)
            
            # Update node temperatures
            for i, node_id in enumerate(node_ids):
                self.nodes[node_id].temperature = temperatures[i]
            
            # Record evolution
            if step % (time_steps // 20) == 0:  # Store every 5%
                evolution['timestamps'].append(step * dt)
                evolution['temperatures'].append(temperatures.copy())
                evolution['heat_flows'].append(self._calculate_heat_flows(node_ids))
                evolution['entropy'].append(self._calculate_system_entropy(temperatures))
                
        logger.info(f"✅ Heat diffusion complete. Final entropy: {evolution['entropy'][-1]:.3f}")
        return evolution

    def _build_network_matrices(self) -> None:
        """Build adjacency and Laplacian matrices for heat diffusion"""
        
        n_nodes = len(self.nodes)
        node_ids = list(self.nodes.keys())
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # Build adjacency matrix with conductances
        adjacency = np.zeros((n_nodes, n_nodes))
        
        for edge_id, edge in self.edges.items():
            i = node_to_idx[edge.source]
            j = node_to_idx[edge.target]
            
            # Conductance determines heat flow rate
            adjacency[i, j] = edge.conductance
            adjacency[j, i] = edge.conductance  # Symmetric
            
        # Build Laplacian matrix for diffusion
        degree = np.sum(adjacency, axis=1)
        laplacian = np.diag(degree) - adjacency
        
        self.adjacency_matrix = adjacency
        self.laplacian_matrix = laplacian
        
        logger.info(f"📊 Built network matrices ({n_nodes} nodes, {len(self.edges)} edges)")

    def _calculate_heat_sources(self, node_ids: List[str], time_step: int) -> np.ndarray:
        """Calculate external heat sources (market events)"""
        
        sources = np.zeros(len(node_ids))
        
        # Simulate market events as heat pulses
        if time_step % 20 == 0:  # Market events every 20 steps
            # Random market shock
            shock_node = np.random.choice(len(node_ids))
            shock_intensity = np.random.uniform(0.1, 0.3)
            sources[shock_node] = shock_intensity
            
            logger.debug(f"💥 Market shock at {node_ids[shock_node]} (intensity: {shock_intensity:.3f})")
            
        return sources

    def _calculate_heat_flows(self, node_ids: List[str]) -> Dict:
        """Calculate current heat flows between nodes"""
        
        flows = {}
        
        for edge_id, edge in self.edges.items():
            source_temp = self.nodes[edge.source].temperature
            target_temp = self.nodes[edge.target].temperature
            
            # Heat flow: q = conductance × ΔT
            flow = edge.conductance * (source_temp - target_temp)
            flows[edge_id] = flow
            
        return flows

    def _calculate_system_entropy(self, temperatures: np.ndarray) -> float:
        """Calculate thermodynamic entropy of the system"""
        
        # Shannon entropy of temperature distribution
        # Higher entropy = more uniform heat distribution
        temp_bins = np.histogram(temperatures, bins=10, range=(0, 1), density=True)[0]
        temp_bins = temp_bins + 1e-10  # Avoid log(0)
        
        entropy = -np.sum(temp_bins * np.log(temp_bins))
        return entropy

    def predict_future_temperatures(self, forecast_steps: int = 50) -> Dict:
        """Use PINN to predict future temperature evolution"""
        
        logger.info(f"🔮 Predicting temperature evolution ({forecast_steps} steps)")
        
        if not self.nodes:
            return {}
            
        node_ids = list(self.nodes.keys())
        current_temps = np.array([self.nodes[nid].temperature for nid in node_ids])
        positions = np.array([self.nodes[nid].position for nid in node_ids])
        
        predictions = []
        
        for step in range(forecast_steps):
            # Prepare input for neural network
            time_val = step * self.dt
            inputs = []
            
            for i, node_id in enumerate(node_ids):
                x, y = positions[i]
                temp = current_temps[i] 
                input_vec = [x, y, time_val, temp]
                inputs.append(input_vec)
            
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            
            # Predict next temperatures
            with torch.no_grad():
                next_temps = self.heat_predictor(inputs_tensor).cpu().numpy().flatten()
                
            predictions.append(next_temps.copy())
            current_temps = next_temps  # Update for next iteration
            
        return {
            'node_ids': node_ids,
            'predictions': predictions,
            'time_points': np.arange(forecast_steps) * self.dt
        }

    def get_network_visualization_data(self) -> Dict:
        """Export data for advanced visualization"""
        
        # Nodes with physics properties
        nodes = []
        for node_id, node in self.nodes.items():
            node_data = {
                'id': node_id,
                'x': node.position[0],
                'y': node.position[1],
                'temperature': node.temperature,
                'conductivity': node.conductivity,
                'capacity': node.capacity,
                'size': 10 + node.temperature * 30,  # Size based on heat
                'color': self._temperature_to_color(node.temperature),
                'sector': node.sector,
                'market_value': node.market_value,
                'volatility': node.volatility,
                'label': node_id
            }
            nodes.append(node_data)
            
        # Edges with heat flow information
        edges = []
        for edge_id, edge in self.edges.items():
            edge_data = {
                'id': edge_id,
                'source': edge.source,
                'target': edge.target,
                'conductance': edge.conductance,
                'resistance': edge.resistance,
                'correlation': edge.correlation,
                'information_flow': edge.information_flow,
                'width': 1 + edge.conductance * 5,  # Width based on conductance
                'color': self._flow_to_color(edge.information_flow),
                'opacity': 0.3 + edge.conductance * 0.7
            }
            edges.append(edge_data)
            
        return {
            'nodes': nodes,
            'edges': edges,
            'physics_properties': {
                'thermal_diffusivity': self.thermal_diffusivity,
                'time_step': self.dt,
                'system_entropy': self._calculate_system_entropy(
                    np.array([n.temperature for n in self.nodes.values()])
                )
            },
            'timestamp': datetime.now().isoformat()
        }

    def _temperature_to_color(self, temperature: float) -> str:
        """Convert temperature to heat map color"""
        
        # Physics-inspired color mapping
        if temperature > 0.8:
            return '#FF0000'  # Hot - Red
        elif temperature > 0.6:
            return '#FF4500'  # Orange-Red
        elif temperature > 0.4:
            return '#FF8C00'  # Orange
        elif temperature > 0.2:
            return '#FFD700'  # Yellow
        else:
            return '#00BFFF'  # Cool - Blue
            
    def _flow_to_color(self, flow: float) -> str:
        """Convert information flow to edge color"""
        
        if flow > 0.8:
            return '#FF1493'  # High flow - Pink
        elif flow > 0.5:
            return '#9370DB'  # Medium flow - Purple
        else:
            return '#4682B4'  # Low flow - Steel Blue

    def train_heat_predictor(self, training_data: List[Dict], epochs: int = 1000) -> None:
        """Train the physics-informed neural network"""
        
        logger.info(f"🎓 Training heat predictor for {epochs} epochs")
        
        # Convert training data to tensors
        inputs = []
        targets = []
        
        for data_point in training_data:
            x, y = data_point['position']
            time = data_point['time']
            current_temp = data_point['current_temperature']
            target_temp = data_point['target_temperature']
            
            inputs.append([x, y, time, current_temp])
            targets.append([target_temp])
            
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        targets_tensor = torch.tensor(targets, dtype=torch.float32).to(self.device)
        
        # Training loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.heat_predictor(inputs_tensor)
            
            # Data loss
            data_loss = nn.MSELoss()(predictions, targets_tensor)
            
            # Physics loss (enforce heat equation)
            physics_loss = self.heat_predictor.physics_loss(inputs_tensor, predictions)
            
            # Total loss
            total_loss = data_loss + 0.1 * physics_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Data Loss={data_loss:.4f}, Physics Loss={physics_loss:.4f}")
                
        logger.info("✅ Heat predictor training complete")

# Example usage and testing
def create_sample_heat_network():
    """Create a sample financial heat network for demonstration"""
    
    network = PhysicsInformedHeatNetwork()
    
    # Add major financial instruments
    stocks = {
        'AAPL': {
            'market_cap': 3e12, 
            'volatility': 0.25, 
            'volume': 80e6, 
            'price_change': 0.02, 
            'sentiment': 0.3,
            'sector': 'Technology'
        },
        'GOOGL': {
            'market_cap': 2e12, 
            'volatility': 0.28, 
            'volume': 25e6, 
            'price_change': -0.01, 
            'sentiment': 0.1,
            'sector': 'Technology'  
        },
        'JPM': {
            'market_cap': 500e9, 
            'volatility': 0.35, 
            'volume': 15e6, 
            'price_change': 0.015, 
            'sentiment': -0.1,
            'sector': 'Financial'
        },
        'TSLA': {
            'market_cap': 800e9, 
            'volatility': 0.45, 
            'volume': 100e6, 
            'price_change': 0.05, 
            'sentiment': 0.6,
            'sector': 'Consumer'
        }
    }
    
    # Add nodes
    for symbol, data in stocks.items():
        network.add_financial_node(symbol, data)
        
    # Add correlations (heat connections)
    correlations = [
        ('AAPL', 'GOOGL', 0.75),
        ('AAPL', 'TSLA', 0.45),
        ('GOOGL', 'TSLA', 0.38),
        ('JPM', 'AAPL', 0.25),
        ('JPM', 'TSLA', 0.15)
    ]
    
    for source, target, corr in correlations:
        network.add_heat_connection(source, target, corr)
        
    return network

if __name__ == "__main__":
    # Demonstration
    network = create_sample_heat_network()
    
    # Run heat diffusion simulation
    evolution = network.simulate_heat_diffusion(time_steps=100)
    
    # Get visualization data
    viz_data = network.get_network_visualization_data()
    
    # Predict future temperatures
    predictions = network.predict_future_temperatures(forecast_steps=50)
    
    print("🔥 Physics-Informed Heat Network demonstration complete!")
    print(f"System entropy: {viz_data['physics_properties']['system_entropy']:.3f}")
    print(f"Nodes: {len(viz_data['nodes'])}")
    print(f"Thermal connections: {len(viz_data['edges'])}")