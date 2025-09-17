"""
Advanced Heat Equation Engine for Market Analysis
Implements sophisticated heat diffusion models to identify heated sectors and predict propagation
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class HeatSourceType(Enum):
    NEWS_EVENT = "news_event"
    EARNINGS_SURPRISE = "earnings_surprise"
    VOLUME_SPIKE = "volume_spike"
    MOMENTUM_BREAKOUT = "momentum_breakout"
    SECTOR_ROTATION = "sector_rotation"
    MARKET_SHOCK = "market_shock"


class BoundaryCondition(Enum):
    ZERO_FLUX = "zero_flux"  # No heat flows out of boundaries
    FIXED_TEMPERATURE = "fixed_temperature"  # Fixed heat at boundaries
    CONVECTIVE = "convective"  # Heat exchange with environment


@dataclass
class HeatSource:
    """A heat source in the market"""
    entity_id: str
    entity_type: str  # "stock", "sector", "market"
    source_type: HeatSourceType
    intensity: float  # Heat generation rate
    location: Tuple[float, float]  # Spatial coordinates in market space
    radius: float  # Influence radius
    duration: float  # How long the source lasts (in time units)
    start_time: datetime
    decay_rate: float = 0.1  # Exponential decay rate
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeatFieldState:
    """State of the heat field at a given time"""
    timestamp: datetime
    heat_values: np.ndarray  # Heat at each node
    temperature_gradient: np.ndarray  # Gradient field
    heat_flux: np.ndarray  # Heat flux field
    active_sources: List[HeatSource]
    total_energy: float
    max_temperature: float
    heated_entities: List[str]  # Entities above threshold


@dataclass
class HeatPropagationResult:
    """Result of heat propagation simulation"""
    initial_state: HeatFieldState
    final_state: HeatFieldState
    time_series: List[HeatFieldState]
    propagation_metrics: Dict[str, Any]
    heated_sectors: List[Dict[str, Any]]
    heat_flow_analysis: Dict[str, Any]
    predictions: Dict[str, Any]


class AdvancedHeatEngine:
    """
    Advanced heat diffusion engine using finite difference methods
    """
    
    def __init__(self):
        self.thermal_diffusivity = 0.1  # α in heat equation
        self.grid_size = 50  # Spatial discretization
        self.time_step = 0.01  # Temporal discretization
        self.heat_threshold = 0.3  # Threshold for "heated" classification
        
        # Spatial grid for heat diffusion
        self.x_grid = np.linspace(0, 1, self.grid_size)
        self.y_grid = np.linspace(0, 1, self.grid_size)
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]
        
        # Entity positioning in market space
        self.entity_positions = {}
        self.entity_heat_map = {}
        self.correlation_matrix = None
        
        # Heat sources and field history
        self.active_sources = []
        self.field_history = []
        self.current_field = None
        
        # Initialize market topology
        self.initialize_market_topology()
    
    def initialize_market_topology(self):
        """Initialize the spatial topology of market entities"""
        # Position major sectors in market space
        sector_positions = {
            "TECHNOLOGY": (0.2, 0.8),  # Growth quadrant
            "HEALTHCARE": (0.8, 0.8),  # Defensive growth
            "FINANCIAL": (0.2, 0.2),   # Cyclical value
            "ENERGY": (0.5, 0.2),      # Commodity cyclical
            "CONSUMER": (0.8, 0.5),    # Consumer defensive
            "US_MARKET": (0.5, 0.5)    # Market center
        }
        
        # Add individual stocks around their sectors with some noise
        stock_positions = {
            # Technology stocks around tech sector
            "STOCK_AAPL": (0.18, 0.82),
            "STOCK_MSFT": (0.22, 0.78),
            "STOCK_GOOGL": (0.16, 0.85),
            "STOCK_NVDA": (0.25, 0.83),
            "STOCK_META": (0.19, 0.75),
            "STOCK_TSLA": (0.15, 0.88),
            
            # Financial stocks
            "STOCK_JPM": (0.18, 0.22),
            "STOCK_BAC": (0.22, 0.18),
            
            # Other sectors
            "STOCK_JNJ": (0.82, 0.78),  # Healthcare
            "STOCK_XOM": (0.48, 0.18),  # Energy
            "STOCK_WMT": (0.78, 0.52)   # Consumer
        }
        
        self.entity_positions = {**sector_positions, **stock_positions}
        
        # Initialize heat field
        self.current_field = np.zeros((self.grid_size, self.grid_size))
        
        # Create correlation matrix based on distances
        self.create_correlation_matrix()
    
    def create_correlation_matrix(self):
        """Create correlation matrix based on spatial distances and sector relationships"""
        entities = list(self.entity_positions.keys())
        n = len(entities)
        self.correlation_matrix = np.zeros((n, n))
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i == j:
                    self.correlation_matrix[i, j] = 1.0
                else:
                    # Distance-based correlation
                    pos1 = self.entity_positions[entity1]
                    pos2 = self.entity_positions[entity2]
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    # Base correlation from distance (closer = more correlated)
                    base_correlation = np.exp(-distance * 5)  # Exponential decay
                    
                    # Sector relationship bonus
                    sector_bonus = self.get_sector_relationship_bonus(entity1, entity2)
                    
                    self.correlation_matrix[i, j] = np.clip(base_correlation + sector_bonus, 0, 1)
    
    def get_sector_relationship_bonus(self, entity1: str, entity2: str) -> float:
        """Get correlation bonus based on sector relationships"""
        # Same sector stocks have higher correlation
        if entity1.startswith("STOCK_") and entity2.startswith("STOCK_"):
            # Get sectors for stocks (simplified mapping)
            sector_mapping = {
                "AAPL": "TECHNOLOGY", "MSFT": "TECHNOLOGY", "GOOGL": "TECHNOLOGY",
                "NVDA": "TECHNOLOGY", "META": "TECHNOLOGY", "TSLA": "TECHNOLOGY",
                "JPM": "FINANCIAL", "BAC": "FINANCIAL",
                "JNJ": "HEALTHCARE", "XOM": "ENERGY", "WMT": "CONSUMER"
            }
            
            stock1 = entity1.replace("STOCK_", "")
            stock2 = entity2.replace("STOCK_", "")
            
            if (stock1 in sector_mapping and stock2 in sector_mapping and
                sector_mapping[stock1] == sector_mapping[stock2]):
                return 0.3  # Same sector bonus
        
        # Sector to its stocks
        if (entity1 in ["TECHNOLOGY", "HEALTHCARE", "FINANCIAL", "ENERGY", "CONSUMER"] and
            entity2.startswith("STOCK_")):
            return 0.4  # Sector-stock relationship
        
        return 0.0
    
    def add_heat_source(self, source: HeatSource):
        """Add a heat source to the simulation"""
        self.active_sources.append(source)
        logger.info(f"🔥 Added heat source: {source.entity_id} ({source.source_type.value})")
    
    def create_heat_source_from_market_event(self, entity_id: str, event_data: Dict[str, Any]) -> HeatSource:
        """Create heat source from market event data"""
        # Determine source type based on event characteristics
        if "earnings" in event_data.get("type", "").lower():
            source_type = HeatSourceType.EARNINGS_SURPRISE
            base_intensity = 0.8
        elif "news" in event_data.get("type", "").lower():
            source_type = HeatSourceType.NEWS_EVENT
            base_intensity = 0.6
        elif event_data.get("volume_spike", False):
            source_type = HeatSourceType.VOLUME_SPIKE
            base_intensity = 0.7
        else:
            source_type = HeatSourceType.MOMENTUM_BREAKOUT
            base_intensity = 0.5
        
        # Get entity position
        position = self.entity_positions.get(entity_id, (0.5, 0.5))
        
        # Calculate intensity based on event magnitude
        event_impact = event_data.get("impact", 0.5)
        intensity = base_intensity * abs(event_impact)
        
        return HeatSource(
            entity_id=entity_id,
            entity_type="stock" if entity_id.startswith("STOCK_") else "sector",
            source_type=source_type,
            intensity=intensity,
            location=position,
            radius=0.1 + 0.1 * intensity,  # Larger impact = larger radius
            duration=3600,  # 1 hour default
            start_time=datetime.now(),
            decay_rate=0.05,
            metadata=event_data
        )
    
    def solve_heat_equation_2d(self, time_span: float, num_steps: int = 100) -> List[HeatFieldState]:
        """Solve 2D heat equation using finite difference method"""
        dt = time_span / num_steps
        states = []
        
        # Initialize field
        current_heat = self.current_field.copy()
        
        # Create finite difference operators
        laplacian_operator = self.create_2d_laplacian_operator()
        
        for step in range(num_steps):
            current_time = datetime.now() + timedelta(seconds=step * dt)
            
            # Add heat sources
            source_term = self.compute_source_term(current_time)
            
            # Apply heat equation: ∂u/∂t = α∇²u + S(x,y,t)
            heat_laplacian = laplacian_operator @ current_heat.flatten()
            heat_laplacian = heat_laplacian.reshape(self.grid_size, self.grid_size)
            
            # Euler forward step
            heat_change = self.thermal_diffusivity * heat_laplacian + source_term
            current_heat += dt * heat_change
            
            # Apply boundary conditions
            current_heat = self.apply_boundary_conditions(current_heat)
            
            # Create state snapshot
            if step % (num_steps // 20) == 0:  # Save 20 snapshots
                state = self.create_field_state(current_time, current_heat)
                states.append(state)
        
        return states
    
    def create_2d_laplacian_operator(self) -> sp.csr_matrix:
        """Create 2D Laplacian operator matrix for finite differences"""
        n = self.grid_size
        N = n * n
        
        # Create sparse matrix for 2D Laplacian
        diagonals = []
        offsets = []
        
        # Main diagonal (central difference)
        main_diag = -4 * np.ones(N)
        diagonals.append(main_diag)
        offsets.append(0)
        
        # x-direction neighbors
        x_neighbors = np.ones(N)
        # Handle boundaries
        for i in range(n):
            x_neighbors[i * n - 1] = 0  # Right boundary
            x_neighbors[i * n] = 0      # Left boundary of next row
        
        diagonals.extend([x_neighbors[:-1], x_neighbors[1:]])
        offsets.extend([-1, 1])
        
        # y-direction neighbors
        y_neighbors = np.ones(N)
        diagonals.extend([y_neighbors[:-n], y_neighbors[n:]])
        offsets.extend([-n, n])
        
        # Create sparse matrix
        laplacian = sp.diags(diagonals, offsets, shape=(N, N), format='csr')
        
        # Scale by grid spacing
        laplacian = laplacian / (self.dx**2)
        
        return laplacian
    
    def compute_source_term(self, current_time: datetime) -> np.ndarray:
        """Compute heat source term S(x,y,t) for current time"""
        source_field = np.zeros((self.grid_size, self.grid_size))
        
        for source in self.active_sources:
            # Check if source is still active
            elapsed = (current_time - source.start_time).total_seconds()
            if elapsed > source.duration:
                continue
            
            # Exponential decay over time
            time_factor = np.exp(-source.decay_rate * elapsed / 3600)  # Decay per hour
            
            # Spatial distribution (Gaussian around source location)
            x_center, y_center = source.location
            
            # Convert to grid indices
            x_idx = int(x_center * (self.grid_size - 1))
            y_idx = int(y_center * (self.grid_size - 1))
            
            # Create Gaussian source
            sigma = source.radius * self.grid_size  # Convert radius to grid units
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    distance_sq = (i - x_idx)**2 + (j - y_idx)**2
                    gaussian = np.exp(-distance_sq / (2 * sigma**2))
                    source_field[i, j] += source.intensity * time_factor * gaussian
        
        return source_field
    
    def apply_boundary_conditions(self, heat_field: np.ndarray) -> np.ndarray:
        """Apply boundary conditions to heat field"""
        # Zero flux boundary conditions (Neumann)
        heat_field[0, :] = heat_field[1, :]      # Top
        heat_field[-1, :] = heat_field[-2, :]    # Bottom
        heat_field[:, 0] = heat_field[:, 1]      # Left
        heat_field[:, -1] = heat_field[:, -2]    # Right
        
        return heat_field
    
    def create_field_state(self, timestamp: datetime, heat_field: np.ndarray) -> HeatFieldState:
        """Create a field state snapshot"""
        # Calculate gradients
        grad_x, grad_y = np.gradient(heat_field, self.dx, self.dy)
        temperature_gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate heat flux (Fourier's law: q = -k∇T)
        heat_flux = self.thermal_diffusivity * temperature_gradient
        
        # Find heated entities
        heated_entities = []
        for entity_id, position in self.entity_positions.items():
            x_idx = int(position[0] * (self.grid_size - 1))
            y_idx = int(position[1] * (self.grid_size - 1))
            
            entity_heat = heat_field[x_idx, y_idx]
            if abs(entity_heat) > self.heat_threshold:
                heated_entities.append(entity_id)
        
        # Calculate total energy
        total_energy = np.sum(heat_field**2) * self.dx * self.dy
        
        return HeatFieldState(
            timestamp=timestamp,
            heat_values=heat_field.copy(),
            temperature_gradient=temperature_gradient,
            heat_flux=heat_flux,
            active_sources=[s for s in self.active_sources if 
                          (timestamp - s.start_time).total_seconds() < s.duration],
            total_energy=total_energy,
            max_temperature=np.max(np.abs(heat_field)),
            heated_entities=heated_entities
        )
    
    def analyze_heat_propagation(self, market_data: Dict[str, Any], 
                               time_horizon: float = 3600) -> HeatPropagationResult:
        """Analyze heat propagation for given market data"""
        try:
            # Clear old sources and reset field
            self.active_sources = []
            self.current_field = np.zeros((self.grid_size, self.grid_size))
            
            # Create heat sources from market data
            self.create_heat_sources_from_data(market_data)
            
            # Get initial state
            initial_state = self.create_field_state(datetime.now(), self.current_field)
            
            # Solve heat equation
            logger.info(f"🔥 Solving heat diffusion equation for {time_horizon}s")
            time_series = self.solve_heat_equation_2d(time_horizon, num_steps=200)
            
            # Get final state
            final_state = time_series[-1] if time_series else initial_state
            
            # Analyze propagation metrics
            propagation_metrics = self.calculate_propagation_metrics(initial_state, final_state, time_series)
            
            # Identify heated sectors
            heated_sectors = self.identify_heated_sectors(final_state, market_data)
            
            # Analyze heat flow patterns
            heat_flow_analysis = self.analyze_heat_flow_patterns(time_series)
            
            # Generate predictions
            predictions = self.generate_heat_predictions(final_state, time_series)
            
            return HeatPropagationResult(
                initial_state=initial_state,
                final_state=final_state,
                time_series=time_series,
                propagation_metrics=propagation_metrics,
                heated_sectors=heated_sectors,
                heat_flow_analysis=heat_flow_analysis,
                predictions=predictions
            )
            
        except Exception as e:
            logger.error(f"❌ Error in heat propagation analysis: {e}")
            # Return empty result on error
            return HeatPropagationResult(
                initial_state=self.create_field_state(datetime.now(), np.zeros((self.grid_size, self.grid_size))),
                final_state=self.create_field_state(datetime.now(), np.zeros((self.grid_size, self.grid_size))),
                time_series=[],
                propagation_metrics={"error": str(e)},
                heated_sectors=[],
                heat_flow_analysis={"error": str(e)},
                predictions={"error": str(e)}
            )
    
    def create_heat_sources_from_data(self, market_data: Dict[str, Any]):
        """Create heat sources from market data"""
        # Process heated sectors
        for heated_sector in market_data.get("heated_sectors", []):
            sector_id = heated_sector["sector"].upper()
            if sector_id in self.entity_positions:
                
                source = HeatSource(
                    entity_id=sector_id,
                    entity_type="sector",
                    source_type=HeatSourceType.SECTOR_ROTATION,
                    intensity=abs(heated_sector["heat_level"]) * 2,  # Amplify for visualization
                    location=self.entity_positions[sector_id],
                    radius=0.15,
                    duration=7200,  # 2 hours
                    start_time=datetime.now(),
                    decay_rate=0.03,
                    metadata={"reason": heated_sector.get("reason", "")}
                )
                self.add_heat_source(source)
        
        # Process individual stock heat
        for symbol, stock_data in market_data.get("stocks", {}).items():
            entity_id = f"STOCK_{symbol}"
            if entity_id in self.entity_positions:
                stock_heat = stock_data.get("heat_level", 0)
                
                if abs(stock_heat) > 0.3:  # Only significant heat
                    # Determine source type based on stock characteristics
                    volume = stock_data.get("volume", 0)
                    change_percent = stock_data.get("change_percent", 0)
                    
                    if volume > 2000000 and abs(change_percent) > 5:
                        source_type = HeatSourceType.VOLUME_SPIKE
                    elif abs(change_percent) > 10:
                        source_type = HeatSourceType.MOMENTUM_BREAKOUT
                    else:
                        source_type = HeatSourceType.NEWS_EVENT
                    
                    source = HeatSource(
                        entity_id=entity_id,
                        entity_type="stock",
                        source_type=source_type,
                        intensity=abs(stock_heat) * 1.5,
                        location=self.entity_positions[entity_id],
                        radius=0.05 + 0.05 * abs(stock_heat),
                        duration=1800,  # 30 minutes for stocks
                        start_time=datetime.now(),
                        decay_rate=0.1,
                        metadata={
                            "symbol": symbol,
                            "price": stock_data.get("price", 0),
                            "change_percent": change_percent,
                            "volume": volume
                        }
                    )
                    self.add_heat_source(source)
    
    def calculate_propagation_metrics(self, initial: HeatFieldState, final: HeatFieldState,
                                    time_series: List[HeatFieldState]) -> Dict[str, Any]:
        """Calculate heat propagation metrics"""
        metrics = {}
        
        # Energy conservation
        initial_energy = initial.total_energy
        final_energy = final.total_energy
        energy_change = (final_energy - initial_energy) / max(initial_energy, 1e-6)
        
        metrics["energy_conservation"] = {
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change_percent": energy_change * 100
        }
        
        # Heat diffusion rate
        max_temps = [state.max_temperature for state in time_series]
        if len(max_temps) > 1:
            diffusion_rate = (max_temps[-1] - max_temps[0]) / len(max_temps)
            metrics["diffusion_rate"] = diffusion_rate
        
        # Heated entity evolution
        heated_counts = [len(state.heated_entities) for state in time_series]
        metrics["heated_entity_evolution"] = {
            "initial_count": heated_counts[0] if heated_counts else 0,
            "final_count": heated_counts[-1] if heated_counts else 0,
            "peak_count": max(heated_counts) if heated_counts else 0,
            "average_count": np.mean(heated_counts) if heated_counts else 0
        }
        
        # Propagation speed (how fast heat spreads)
        if len(time_series) > 1:
            time_to_peak = np.argmax(max_temps) if max_temps else 0
            metrics["propagation_speed"] = {
                "time_to_peak_temperature": time_to_peak,
                "average_spread_rate": final.max_temperature / len(time_series)
            }
        
        return metrics
    
    def identify_heated_sectors(self, final_state: HeatFieldState, 
                              market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify sectors that are heated based on final field state"""
        heated_sectors = []
        
        sector_entities = [entity for entity in self.entity_positions.keys() 
                         if not entity.startswith("STOCK_") and entity != "US_MARKET"]
        
        for sector_id in sector_entities:
            position = self.entity_positions[sector_id]
            x_idx = int(position[0] * (self.grid_size - 1))
            y_idx = int(position[1] * (self.grid_size - 1))
            
            sector_heat = final_state.heat_values[x_idx, y_idx]
            
            # Calculate surrounding heat (average in local neighborhood)
            neighborhood_size = 3
            x_start = max(0, x_idx - neighborhood_size)
            x_end = min(self.grid_size, x_idx + neighborhood_size + 1)
            y_start = max(0, y_idx - neighborhood_size)
            y_end = min(self.grid_size, y_idx + neighborhood_size + 1)
            
            neighborhood_heat = np.mean(final_state.heat_values[x_start:x_end, y_start:y_end])
            
            if abs(sector_heat) > self.heat_threshold or abs(neighborhood_heat) > self.heat_threshold:
                # Find reason for heating
                reasons = []
                for source in final_state.active_sources:
                    if source.entity_id == sector_id:
                        reasons.append(f"{source.source_type.value}: {source.metadata.get('reason', '')}")
                
                # Check if stocks in sector are heated
                sector_stocks = [entity for entity in final_state.heated_entities 
                               if entity.startswith("STOCK_") and self.get_stock_sector(entity) == sector_id]
                
                heated_sectors.append({
                    "sector": sector_id,
                    "heat_level": float(sector_heat),
                    "neighborhood_heat": float(neighborhood_heat),
                    "heated_stocks": sector_stocks,
                    "heated_stocks_count": len(sector_stocks),
                    "heat_intensity": "high" if abs(sector_heat) > 0.7 else "medium" if abs(sector_heat) > 0.4 else "low",
                    "reasons": reasons if reasons else ["Heat diffusion from market dynamics"],
                    "recommendation": self.generate_sector_recommendation(sector_heat, neighborhood_heat),
                    "confidence": min(abs(sector_heat) + 0.3, 1.0)
                })
        
        # Sort by heat level
        heated_sectors.sort(key=lambda x: abs(x["heat_level"]), reverse=True)
        
        return heated_sectors
    
    def get_stock_sector(self, stock_entity_id: str) -> str:
        """Get sector for stock entity (simplified mapping)"""
        stock_symbol = stock_entity_id.replace("STOCK_", "")
        
        sector_mapping = {
            "AAPL": "TECHNOLOGY", "MSFT": "TECHNOLOGY", "GOOGL": "TECHNOLOGY",
            "NVDA": "TECHNOLOGY", "META": "TECHNOLOGY", "TSLA": "TECHNOLOGY",
            "JPM": "FINANCIAL", "BAC": "FINANCIAL",
            "JNJ": "HEALTHCARE", "XOM": "ENERGY", "WMT": "CONSUMER"
        }
        
        return sector_mapping.get(stock_symbol, "UNKNOWN")
    
    def generate_sector_recommendation(self, sector_heat: float, neighborhood_heat: float) -> str:
        """Generate trading recommendation based on sector heat"""
        avg_heat = (sector_heat + neighborhood_heat) / 2
        
        if avg_heat > 0.7:
            return "STRONG_BUY - Sector showing strong bullish heat momentum"
        elif avg_heat > 0.4:
            return "BUY - Moderate bullish heat in sector"
        elif avg_heat > 0.2:
            return "WATCH - Emerging heat pattern"
        elif avg_heat < -0.7:
            return "STRONG_SELL - Sector showing strong bearish heat"
        elif avg_heat < -0.4:
            return "SELL - Moderate bearish heat in sector"
        elif avg_heat < -0.2:
            return "WATCH - Emerging bearish pattern"
        else:
            return "HOLD - Neutral heat levels"
    
    def analyze_heat_flow_patterns(self, time_series: List[HeatFieldState]) -> Dict[str, Any]:
        """Analyze heat flow patterns over time"""
        if len(time_series) < 2:
            return {"error": "Insufficient time series data"}
        
        flow_analysis = {
            "dominant_flow_direction": self.find_dominant_flow_direction(time_series),
            "heat_convergence_points": self.find_heat_convergence_points(time_series[-1]),
            "heat_sources_strength": self.analyze_source_strength(time_series),
            "diffusion_efficiency": self.calculate_diffusion_efficiency(time_series)
        }
        
        return flow_analysis
    
    def find_dominant_flow_direction(self, time_series: List[HeatFieldState]) -> Dict[str, float]:
        """Find the dominant direction of heat flow"""
        if len(time_series) < 2:
            return {"x": 0.0, "y": 0.0}
        
        # Calculate average flow direction from gradient fields
        total_flow_x = 0.0
        total_flow_y = 0.0
        
        for state in time_series[-5:]:  # Use last 5 states
            grad_x, grad_y = np.gradient(state.heat_values, self.dx, self.dy)
            
            # Weight by magnitude
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            weighted_x = np.sum(grad_x * magnitude)
            weighted_y = np.sum(grad_y * magnitude)
            
            total_flow_x += weighted_x
            total_flow_y += weighted_y
        
        # Normalize
        total_magnitude = np.sqrt(total_flow_x**2 + total_flow_y**2)
        if total_magnitude > 1e-6:
            return {
                "x": total_flow_x / total_magnitude,
                "y": total_flow_y / total_magnitude,
                "magnitude": total_magnitude
            }
        
        return {"x": 0.0, "y": 0.0, "magnitude": 0.0}
    
    def find_heat_convergence_points(self, final_state: HeatFieldState) -> List[Dict[str, Any]]:
        """Find points where heat converges (local maxima)"""
        from scipy.ndimage import maximum_filter
        
        heat_field = final_state.heat_values
        
        # Find local maxima
        local_maxima = maximum_filter(heat_field, size=3) == heat_field
        
        # Filter significant maxima
        significant_maxima = local_maxima & (np.abs(heat_field) > self.heat_threshold)
        
        convergence_points = []
        max_coords = np.where(significant_maxima)
        
        for i, j in zip(max_coords[0], max_coords[1]):
            # Convert grid coordinates to spatial coordinates
            x_coord = i / (self.grid_size - 1)
            y_coord = j / (self.grid_size - 1)
            
            # Find nearest entity
            nearest_entity = self.find_nearest_entity(x_coord, y_coord)
            
            convergence_points.append({
                "grid_position": (i, j),
                "spatial_position": (x_coord, y_coord),
                "heat_value": float(heat_field[i, j]),
                "nearest_entity": nearest_entity,
                "strength": "high" if abs(heat_field[i, j]) > 0.7 else "medium"
            })
        
        return convergence_points
    
    def find_nearest_entity(self, x: float, y: float) -> str:
        """Find nearest entity to given coordinates"""
        min_distance = float('inf')
        nearest_entity = "unknown"
        
        for entity_id, position in self.entity_positions.items():
            distance = np.sqrt((x - position[0])**2 + (y - position[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_entity = entity_id
        
        return nearest_entity
    
    def analyze_source_strength(self, time_series: List[HeatFieldState]) -> Dict[str, Any]:
        """Analyze the strength of different heat sources"""
        source_analysis = {}
        
        if not time_series:
            return source_analysis
        
        # Track source effectiveness over time
        for state in time_series:
            for source in state.active_sources:
                if source.entity_id not in source_analysis:
                    source_analysis[source.entity_id] = {
                        "source_type": source.source_type.value,
                        "total_intensity": 0.0,
                        "peak_intensity": 0.0,
                        "duration_active": 0,
                        "effectiveness_score": 0.0
                    }
                
                source_analysis[source.entity_id]["total_intensity"] += source.intensity
                source_analysis[source.entity_id]["peak_intensity"] = max(
                    source_analysis[source.entity_id]["peak_intensity"], source.intensity
                )
                source_analysis[source.entity_id]["duration_active"] += 1
        
        # Calculate effectiveness scores
        for entity_id, analysis in source_analysis.items():
            analysis["effectiveness_score"] = (
                analysis["peak_intensity"] * analysis["duration_active"] / len(time_series)
            )
        
        return source_analysis
    
    def calculate_diffusion_efficiency(self, time_series: List[HeatFieldState]) -> float:
        """Calculate how efficiently heat diffuses through the system"""
        if len(time_series) < 2:
            return 0.0
        
        initial_max = time_series[0].max_temperature
        final_max = time_series[-1].max_temperature
        
        if initial_max < 1e-6:
            return 0.0
        
        # Efficiency is how well heat spreads vs concentrates
        efficiency = (final_max / initial_max) if final_max > 0 else 0.0
        
        return min(efficiency, 1.0)
    
    def generate_heat_predictions(self, final_state: HeatFieldState, 
                                time_series: List[HeatFieldState]) -> Dict[str, Any]:
        """Generate predictions about future heat evolution"""
        predictions = {}
        
        try:
            # Predict next heated entities
            current_heated = set(final_state.heated_entities)
            
            # Entities near current heated ones are likely to heat up next
            potential_next_heated = []
            for entity_id, position in self.entity_positions.items():
                if entity_id not in current_heated:
                    # Check proximity to heated entities
                    min_distance = float('inf')
                    for heated_entity in current_heated:
                        if heated_entity in self.entity_positions:
                            heated_pos = self.entity_positions[heated_entity]
                            distance = np.sqrt((position[0] - heated_pos[0])**2 + 
                                             (position[1] - heated_pos[1])**2)
                            min_distance = min(min_distance, distance)
                    
                    if min_distance < 0.2:  # Close proximity threshold
                        potential_next_heated.append({
                            "entity": entity_id,
                            "probability": 1.0 - min_distance * 5,  # Closer = higher probability
                            "expected_time_to_heat": min_distance * 3600  # Distance-based time estimate
                        })
            
            predictions["potential_next_heated"] = sorted(
                potential_next_heated, key=lambda x: x["probability"], reverse=True
            )[:5]
            
            # Predict heat evolution trend
            if len(time_series) > 5:
                recent_max_temps = [state.max_temperature for state in time_series[-5:]]
                trend = np.polyfit(range(len(recent_max_temps)), recent_max_temps, 1)[0]
                
                predictions["heat_trend"] = {
                    "direction": "increasing" if trend > 0 else "decreasing",
                    "rate": abs(trend),
                    "confidence": 0.7 if abs(trend) > 0.01 else 0.3
                }
            
            # Predict when current heat will dissipate
            if final_state.max_temperature > 0:
                # Simple exponential decay model
                decay_time = -np.log(0.1) / 0.1  # Time to reach 10% of current heat
                predictions["heat_dissipation"] = {
                    "estimated_time_to_cool": decay_time,
                    "entities_to_cool_first": final_state.heated_entities[:3]
                }
            
        except Exception as e:
            predictions["error"] = f"Prediction generation failed: {e}"
        
        return predictions


# Global heat engine instance
advanced_heat_engine = AdvancedHeatEngine()


# Convenience functions
def analyze_market_heat(market_data: Dict[str, Any], time_horizon: float = 3600) -> HeatPropagationResult:
    """Analyze market heat using advanced heat equation model"""
    return advanced_heat_engine.analyze_heat_propagation(market_data, time_horizon)


def get_heated_sectors_prediction(market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get heated sectors prediction using physics-based model"""
    result = analyze_market_heat(market_data)
    return result.heated_sectors


def get_heat_engine_status() -> Dict[str, Any]:
    """Get status of the heat engine"""
    return {
        "timestamp": datetime.now().isoformat(),
        "thermal_diffusivity": advanced_heat_engine.thermal_diffusivity,
        "grid_size": advanced_heat_engine.grid_size,
        "active_sources": len(advanced_heat_engine.active_sources),
        "entities_tracked": len(advanced_heat_engine.entity_positions),
        "heat_threshold": advanced_heat_engine.heat_threshold,
        "field_initialized": advanced_heat_engine.current_field is not None
    }


if __name__ == "__main__":
    # Test the advanced heat engine
    print("🔥 Testing Advanced Heat Equation Engine")
    print("=" * 50)
    
    # Mock market data for testing
    test_data = {
        "heated_sectors": [
            {"sector": "technology", "heat_level": 0.8, "reason": "AI breakthrough news"},
            {"sector": "energy", "heat_level": -0.6, "reason": "Oil price decline"}
        ],
        "stocks": {
            "AAPL": {"heat_level": 0.7, "price": 180, "change_percent": 5.2, "volume": 3000000},
            "NVDA": {"heat_level": 0.9, "price": 450, "change_percent": 8.1, "volume": 5000000},
            "XOM": {"heat_level": -0.5, "price": 105, "change_percent": -3.2, "volume": 2000000}
        }
    }
    
    # Analyze heat propagation
    result = analyze_market_heat(test_data)
    
    print(f"Heated sectors found: {len(result.heated_sectors)}")
    for sector in result.heated_sectors[:3]:
        print(f"  - {sector['sector']}: {sector['heat_level']:.2f} ({sector['heat_intensity']})")
        print(f"    Recommendation: {sector['recommendation']}")
    
    print(f"\nHeat flow analysis:")
    flow_direction = result.heat_flow_analysis.get("dominant_flow_direction", {})
    print(f"  - Flow direction: ({flow_direction.get('x', 0):.2f}, {flow_direction.get('y', 0):.2f})")
    
    print(f"\nPredictions:")
    predictions = result.predictions
    if "potential_next_heated" in predictions:
        print(f"  - Next entities likely to heat: {len(predictions['potential_next_heated'])}")
        for pred in predictions["potential_next_heated"][:2]:
            print(f"    • {pred['entity']}: {pred['probability']:.0%} probability")
    
    print(f"\nEngine status: {get_heat_engine_status()}")