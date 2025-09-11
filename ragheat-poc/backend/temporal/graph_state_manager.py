"""
Temporal Graph State Manager
Implements bi-temporal data model with dynamic graph state evolution inspired by Graphiti
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class GraphStateType(Enum):
    """Graph state transition types"""
    CREATION = "creation"
    UPDATE = "update"
    MERGE = "merge"
    SPLIT = "split"
    DECAY = "decay"
    INVALIDATION = "invalidation"

class EdgeValidityState(Enum):
    """Edge validity states over time"""
    ACTIVE = "active"
    WEAKENING = "weakening"
    DORMANT = "dormant"
    INVALID = "invalid"

@dataclass
class TemporalPoint:
    """Represents a point in bi-temporal space"""
    event_time: datetime  # When the event actually occurred
    ingestion_time: datetime  # When we learned about the event
    valid_from: datetime  # When this data becomes valid
    valid_to: Optional[datetime] = None  # When this data becomes invalid

@dataclass
class GraphSnapshot:
    """Complete graph state at a specific temporal point"""
    snapshot_id: str
    temporal_point: TemporalPoint
    nodes: Dict[str, Dict] = field(default_factory=dict)
    edges: Dict[str, Dict] = field(default_factory=dict)
    global_heat: float = 0.0
    market_sentiment: float = 0.0
    validity_score: float = 1.0

@dataclass
class EdgeEvolution:
    """Tracks how an edge evolves over time"""
    edge_id: str
    source_id: str
    target_id: str
    strength_history: List[Tuple[datetime, float]] = field(default_factory=list)
    validity_state: EdgeValidityState = EdgeValidityState.ACTIVE
    last_activity: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1  # Per hour
    creation_time: datetime = field(default_factory=datetime.now)

@dataclass
class NodeEvolution:
    """Tracks how a node evolves over time"""
    node_id: str
    heat_history: List[Tuple[datetime, float]] = field(default_factory=list)
    weight_history: List[Tuple[datetime, Dict[str, float]]] = field(default_factory=list)
    activity_bursts: List[Tuple[datetime, float]] = field(default_factory=list)
    influence_radius: float = 1.0
    temporal_importance: float = 0.5

class TemporalGraphStateManager:
    """
    Manages dynamic graph states with temporal validity and decay
    Implements bi-temporal data model for point-in-time queries
    """
    
    def __init__(self,
                 max_snapshots: int = 100,
                 snapshot_interval: timedelta = timedelta(minutes=5),
                 edge_decay_half_life: timedelta = timedelta(hours=2),
                 heat_propagation_speed: float = 0.1):
        
        self.max_snapshots = max_snapshots
        self.snapshot_interval = snapshot_interval
        self.edge_decay_half_life = edge_decay_half_life
        self.heat_propagation_speed = heat_propagation_speed
        
        # Temporal storage
        self.snapshots: List[GraphSnapshot] = []
        self.node_evolutions: Dict[str, NodeEvolution] = {}
        self.edge_evolutions: Dict[str, EdgeEvolution] = {}
        
        # Current state
        self.current_snapshot: Optional[GraphSnapshot] = None
        self.last_snapshot_time = datetime.now()
        
        # Concurrency control
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.query_cache: Dict[str, Tuple[datetime, Dict]] = {}
        self.cache_ttl = timedelta(seconds=30)
        
        logger.info("Temporal Graph State Manager initialized")
    
    def create_snapshot(self, 
                       nodes: Dict[str, Dict],
                       edges: Dict[str, Dict],
                       global_heat: float = 0.0,
                       market_sentiment: float = 0.0) -> GraphSnapshot:
        """Create a new temporal snapshot of the graph state"""
        
        current_time = datetime.now()
        
        snapshot = GraphSnapshot(
            snapshot_id=f"snapshot_{current_time.timestamp()}",
            temporal_point=TemporalPoint(
                event_time=current_time,
                ingestion_time=current_time,
                valid_from=current_time
            ),
            nodes=nodes.copy(),
            edges=edges.copy(),
            global_heat=global_heat,
            market_sentiment=market_sentiment,
            validity_score=self._calculate_validity_score(nodes, edges)
        )
        
        with self.lock:
            # Add to snapshot history
            self.snapshots.append(snapshot)
            self.current_snapshot = snapshot
            self.last_snapshot_time = current_time
            
            # Maintain snapshot limit
            if len(self.snapshots) > self.max_snapshots:
                old_snapshot = self.snapshots.pop(0)
                logger.debug(f"Removed old snapshot: {old_snapshot.snapshot_id}")
        
        logger.info(f"Created snapshot: {snapshot.snapshot_id}")
        return snapshot
    
    def update_node_evolution(self, 
                            node_id: str,
                            heat_value: float,
                            weights: Dict[str, float],
                            activity_burst: float = 0.0):
        """Update the evolutionary state of a node"""
        
        current_time = datetime.now()
        
        with self.lock:
            if node_id not in self.node_evolutions:
                self.node_evolutions[node_id] = NodeEvolution(
                    node_id=node_id,
                    influence_radius=self._calculate_influence_radius(weights)
                )
            
            evolution = self.node_evolutions[node_id]
            
            # Update histories
            evolution.heat_history.append((current_time, heat_value))
            evolution.weight_history.append((current_time, weights.copy()))
            
            if activity_burst > 0.1:
                evolution.activity_bursts.append((current_time, activity_burst))
            
            # Update temporal importance based on recent activity
            evolution.temporal_importance = self._calculate_temporal_importance(evolution)
            
            # Maintain history limits
            max_history = 200
            if len(evolution.heat_history) > max_history:
                evolution.heat_history = evolution.heat_history[-max_history:]
            if len(evolution.weight_history) > max_history:
                evolution.weight_history = evolution.weight_history[-max_history:]
    
    def update_edge_evolution(self,
                            edge_id: str,
                            source_id: str,
                            target_id: str,
                            strength: float):
        """Update the evolutionary state of an edge"""
        
        current_time = datetime.now()
        
        with self.lock:
            if edge_id not in self.edge_evolutions:
                self.edge_evolutions[edge_id] = EdgeEvolution(
                    edge_id=edge_id,
                    source_id=source_id,
                    target_id=target_id
                )
            
            evolution = self.edge_evolutions[edge_id]
            evolution.strength_history.append((current_time, strength))
            evolution.last_activity = current_time
            
            # Update validity state based on strength and activity
            evolution.validity_state = self._determine_edge_validity(evolution, current_time)
            
            # Maintain history limit
            if len(evolution.strength_history) > 100:
                evolution.strength_history = evolution.strength_history[-100:]
    
    def get_graph_state_at_time(self, 
                              target_time: datetime,
                              include_predictions: bool = False) -> Optional[GraphSnapshot]:
        """
        Get graph state at a specific point in time
        Implements point-in-time query functionality
        """
        
        # Check cache first
        cache_key = f"state_{target_time.timestamp()}_{include_predictions}"
        if cache_key in self.query_cache:
            cached_time, cached_result = self.query_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_result
        
        with self.lock:
            # Find the most appropriate snapshot
            best_snapshot = None
            min_time_diff = float('inf')
            
            for snapshot in self.snapshots:
                time_diff = abs((snapshot.temporal_point.event_time - target_time).total_seconds())
                
                # Must be valid at target time
                if (snapshot.temporal_point.valid_from <= target_time and 
                    (snapshot.temporal_point.valid_to is None or 
                     snapshot.temporal_point.valid_to > target_time)):
                    
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_snapshot = snapshot
            
            if best_snapshot is None:
                return None
            
            # Apply temporal corrections to get accurate state at target_time
            corrected_snapshot = self._apply_temporal_corrections(
                best_snapshot, target_time, include_predictions
            )
            
            # Cache result
            self.query_cache[cache_key] = (datetime.now(), corrected_snapshot)
            
            return corrected_snapshot
    
    def calculate_heat_propagation(self,
                                 source_node: str,
                                 initial_heat: float,
                                 max_hops: int = 3) -> Dict[str, float]:
        """
        Calculate how heat propagates through the graph from a source node
        Uses diffusion equation with temporal decay
        """
        
        if not self.current_snapshot:
            return {}
        
        propagated_heat = {source_node: initial_heat}
        visited = {source_node}
        current_wave = {source_node: initial_heat}
        
        for hop in range(max_hops):
            next_wave = {}
            wave_intensity = initial_heat * (self.heat_propagation_speed ** hop)
            
            if wave_intensity < 0.01:  # Heat too weak to propagate
                break
            
            for node_id, heat in current_wave.items():
                # Find neighboring nodes
                neighbors = self._get_node_neighbors(node_id)
                
                for neighbor_id, edge_strength in neighbors.items():
                    if neighbor_id not in visited:
                        # Calculate propagated heat with decay
                        propagated = heat * edge_strength * self.heat_propagation_speed
                        
                        # Apply temporal decay based on edge age
                        edge_id = f"{node_id}_{neighbor_id}"
                        if edge_id in self.edge_evolutions:
                            time_factor = self._calculate_edge_time_factor(
                                self.edge_evolutions[edge_id]
                            )
                            propagated *= time_factor
                        
                        if propagated > 0.01:
                            next_wave[neighbor_id] = propagated
                            propagated_heat[neighbor_id] = propagated
                            visited.add(neighbor_id)
            
            current_wave = next_wave
            
            if not current_wave:  # No more propagation
                break
        
        return propagated_heat
    
    def detect_graph_anomalies(self, 
                             time_window: timedelta = timedelta(minutes=15)) -> List[Dict]:
        """
        Detect anomalies in graph evolution patterns
        """
        
        anomalies = []
        current_time = datetime.now()
        start_time = current_time - time_window
        
        # 1. Sudden heat spikes
        for node_id, evolution in self.node_evolutions.items():
            heat_spike = self._detect_heat_spike(evolution, start_time, current_time)
            if heat_spike:
                anomalies.append({
                    'type': 'heat_spike',
                    'node_id': node_id,
                    'severity': heat_spike['severity'],
                    'timestamp': heat_spike['timestamp'],
                    'description': f"Unusual heat spike detected in {node_id}"
                })
        
        # 2. Edge strength volatility
        for edge_id, evolution in self.edge_evolutions.items():
            volatility = self._calculate_edge_volatility(evolution, start_time, current_time)
            if volatility > 0.7:  # High volatility threshold
                anomalies.append({
                    'type': 'edge_volatility',
                    'edge_id': edge_id,
                    'severity': volatility,
                    'timestamp': current_time,
                    'description': f"High volatility in edge {edge_id}"
                })
        
        # 3. Rapid topology changes
        topology_change_rate = self._calculate_topology_change_rate(time_window)
        if topology_change_rate > 0.5:  # 50% of edges changed
            anomalies.append({
                'type': 'topology_instability',
                'severity': topology_change_rate,
                'timestamp': current_time,
                'description': "Rapid graph topology changes detected"
            })
        
        return anomalies
    
    def predict_future_state(self, 
                           prediction_horizon: timedelta = timedelta(minutes=10),
                           confidence_threshold: float = 0.6) -> Optional[GraphSnapshot]:
        """
        Predict future graph state based on current trends
        """
        
        if not self.current_snapshot:
            return None
        
        future_time = datetime.now() + prediction_horizon
        
        # Analyze trends for each node
        predicted_nodes = {}
        for node_id, node_data in self.current_snapshot.nodes.items():
            if node_id in self.node_evolutions:
                predicted_heat = self._predict_node_heat(
                    self.node_evolutions[node_id], 
                    prediction_horizon
                )
                
                predicted_nodes[node_id] = node_data.copy()
                predicted_nodes[node_id]['predicted_heat'] = predicted_heat
                predicted_nodes[node_id]['prediction_confidence'] = min(
                    self.node_evolutions[node_id].temporal_importance, 
                    confidence_threshold
                )
        
        # Analyze trends for edges
        predicted_edges = {}
        for edge_id, edge_data in self.current_snapshot.edges.items():
            if edge_id in self.edge_evolutions:
                predicted_strength = self._predict_edge_strength(
                    self.edge_evolutions[edge_id],
                    prediction_horizon
                )
                
                predicted_edges[edge_id] = edge_data.copy()
                predicted_edges[edge_id]['predicted_strength'] = predicted_strength
        
        # Create prediction snapshot
        prediction_snapshot = GraphSnapshot(
            snapshot_id=f"prediction_{future_time.timestamp()}",
            temporal_point=TemporalPoint(
                event_time=future_time,
                ingestion_time=datetime.now(),
                valid_from=future_time
            ),
            nodes=predicted_nodes,
            edges=predicted_edges,
            global_heat=self._predict_global_heat(prediction_horizon),
            market_sentiment=self.current_snapshot.market_sentiment,  # Assume stable
            validity_score=confidence_threshold
        )
        
        return prediction_snapshot
    
    def _calculate_validity_score(self, nodes: Dict, edges: Dict) -> float:
        """Calculate how valid/reliable the current graph state is"""
        
        node_count = len(nodes)
        edge_count = len(edges)
        
        if node_count == 0:
            return 0.0
        
        # Base score from graph connectivity
        connectivity_score = min(edge_count / max(node_count - 1, 1), 1.0)
        
        # Penalty for stale data
        staleness_penalty = 0.0
        current_time = datetime.now()
        
        for evolution in self.node_evolutions.values():
            if evolution.heat_history:
                last_update = evolution.heat_history[-1][0]
                minutes_stale = (current_time - last_update).total_seconds() / 60
                staleness_penalty += min(minutes_stale / 30, 0.2)  # Max 20% penalty per node
        
        staleness_penalty = min(staleness_penalty / node_count, 0.5)  # Cap at 50%
        
        return max(0.1, connectivity_score - staleness_penalty)
    
    def _calculate_influence_radius(self, weights: Dict[str, float]) -> float:
        """Calculate how far a node's influence spreads"""
        
        composite_weight = weights.get('composite_weight', 0.0)
        heat_intensity = weights.get('heat_intensity', 0.0)
        
        # Base radius from composite weight
        base_radius = composite_weight * 2.0
        
        # Boost from heat intensity
        heat_boost = heat_intensity * 0.5
        
        return min(base_radius + heat_boost, 3.0)  # Max radius of 3 hops
    
    def _calculate_temporal_importance(self, evolution: NodeEvolution) -> float:
        """Calculate temporal importance based on recent activity patterns"""
        
        if not evolution.heat_history:
            return 0.1
        
        current_time = datetime.now()
        recent_threshold = current_time - timedelta(minutes=30)
        
        # Count recent heat updates
        recent_updates = sum(1 for time, _ in evolution.heat_history 
                           if time > recent_threshold)
        
        # Calculate heat variance (higher = more dynamic)
        recent_heats = [heat for time, heat in evolution.heat_history 
                       if time > recent_threshold]
        
        heat_variance = np.var(recent_heats) if recent_heats else 0.0
        
        # Activity burst impact
        burst_impact = sum(burst for time, burst in evolution.activity_bursts
                          if time > recent_threshold)
        
        # Combine factors
        importance = (recent_updates / 10.0 + heat_variance + burst_impact / 5.0) / 3.0
        
        return min(importance, 1.0)
    
    def _determine_edge_validity(self, evolution: EdgeEvolution, current_time: datetime) -> EdgeValidityState:
        """Determine edge validity state based on activity and decay"""
        
        if not evolution.strength_history:
            return EdgeValidityState.INVALID
        
        # Get recent activity
        time_since_activity = (current_time - evolution.last_activity).total_seconds() / 3600  # hours
        
        # Get current strength
        current_strength = evolution.strength_history[-1][1]
        
        # Calculate decay
        decay_factor = np.exp(-evolution.decay_rate * time_since_activity)
        effective_strength = current_strength * decay_factor
        
        # Determine state
        if effective_strength > 0.7:
            return EdgeValidityState.ACTIVE
        elif effective_strength > 0.3:
            return EdgeValidityState.WEAKENING
        elif effective_strength > 0.1:
            return EdgeValidityState.DORMANT
        else:
            return EdgeValidityState.INVALID
    
    def _apply_temporal_corrections(self, 
                                  snapshot: GraphSnapshot,
                                  target_time: datetime,
                                  include_predictions: bool) -> GraphSnapshot:
        """Apply temporal corrections to adjust snapshot to exact target time"""
        
        time_diff = (target_time - snapshot.temporal_point.event_time).total_seconds()
        
        corrected_nodes = {}
        for node_id, node_data in snapshot.nodes.items():
            corrected_node = node_data.copy()
            
            # Apply heat decay over time
            if 'heat_score' in corrected_node and node_id in self.node_evolutions:
                original_heat = corrected_node['heat_score']
                decay_rate = 0.001  # Per second
                corrected_heat = original_heat * np.exp(-decay_rate * abs(time_diff))
                corrected_node['heat_score'] = corrected_heat
            
            corrected_nodes[node_id] = corrected_node
        
        corrected_edges = {}
        for edge_id, edge_data in snapshot.edges.items():
            corrected_edge = edge_data.copy()
            
            # Apply edge strength decay
            if 'strength' in corrected_edge and edge_id in self.edge_evolutions:
                evolution = self.edge_evolutions[edge_id]
                original_strength = corrected_edge['strength']
                decay_factor = np.exp(-evolution.decay_rate * abs(time_diff) / 3600)
                corrected_edge['strength'] = original_strength * decay_factor
            
            corrected_edges[edge_id] = corrected_edge
        
        # Create corrected snapshot
        corrected_snapshot = GraphSnapshot(
            snapshot_id=f"corrected_{snapshot.snapshot_id}_{target_time.timestamp()}",
            temporal_point=TemporalPoint(
                event_time=target_time,
                ingestion_time=datetime.now(),
                valid_from=target_time
            ),
            nodes=corrected_nodes,
            edges=corrected_edges,
            global_heat=snapshot.global_heat * np.exp(-0.0001 * abs(time_diff)),
            market_sentiment=snapshot.market_sentiment,
            validity_score=snapshot.validity_score * (0.9 if abs(time_diff) > 300 else 1.0)
        )
        
        return corrected_snapshot
    
    def _get_node_neighbors(self, node_id: str) -> Dict[str, float]:
        """Get neighboring nodes and their edge strengths"""
        
        neighbors = {}
        
        if not self.current_snapshot:
            return neighbors
        
        for edge_id, edge_data in self.current_snapshot.edges.items():
            if edge_data.get('source') == node_id:
                target = edge_data.get('target')
                strength = edge_data.get('strength', 1.0)
                if target:
                    neighbors[target] = strength
            elif edge_data.get('target') == node_id:
                source = edge_data.get('source')
                strength = edge_data.get('strength', 1.0)
                if source:
                    neighbors[source] = strength
        
        return neighbors
    
    def _calculate_edge_time_factor(self, evolution: EdgeEvolution) -> float:
        """Calculate time-based factor for edge strength"""
        
        current_time = datetime.now()
        edge_age_hours = (current_time - evolution.creation_time).total_seconds() / 3600
        
        # Newer edges have higher time factor
        if edge_age_hours < 1:
            return 1.0
        elif edge_age_hours < 24:
            return 0.9
        elif edge_age_hours < 168:  # 1 week
            return 0.7
        else:
            return 0.5
    
    def _detect_heat_spike(self, evolution: NodeEvolution, start_time: datetime, end_time: datetime) -> Optional[Dict]:
        """Detect sudden heat spikes in node evolution"""
        
        recent_heats = [(time, heat) for time, heat in evolution.heat_history
                       if start_time <= time <= end_time]
        
        if len(recent_heats) < 3:
            return None
        
        # Calculate moving average
        heats = [heat for _, heat in recent_heats]
        window_size = min(5, len(heats) // 2)
        
        if window_size < 2:
            return None
        
        moving_avg = []
        for i in range(window_size, len(heats)):
            avg = sum(heats[i-window_size:i]) / window_size
            moving_avg.append(avg)
        
        # Check for spikes
        current_heat = heats[-1]
        recent_avg = sum(moving_avg[-3:]) / len(moving_avg[-3:]) if moving_avg else current_heat
        
        # Spike detection
        if current_heat > recent_avg * 2.0 and current_heat > 0.5:
            return {
                'severity': min((current_heat / recent_avg - 1.0), 2.0),
                'timestamp': recent_heats[-1][0]
            }
        
        return None
    
    def _calculate_edge_volatility(self, evolution: EdgeEvolution, start_time: datetime, end_time: datetime) -> float:
        """Calculate edge strength volatility in time window"""
        
        recent_strengths = [strength for time, strength in evolution.strength_history
                          if start_time <= time <= end_time]
        
        if len(recent_strengths) < 3:
            return 0.0
        
        return float(np.std(recent_strengths))
    
    def _calculate_topology_change_rate(self, time_window: timedelta) -> float:
        """Calculate rate of topology changes"""
        
        if len(self.snapshots) < 2:
            return 0.0
        
        current_time = datetime.now()
        start_time = current_time - time_window
        
        # Find snapshots in window
        window_snapshots = [s for s in self.snapshots 
                          if start_time <= s.temporal_point.event_time <= current_time]
        
        if len(window_snapshots) < 2:
            return 0.0
        
        # Calculate edge changes between snapshots
        total_changes = 0
        total_comparisons = 0
        
        for i in range(1, len(window_snapshots)):
            prev_edges = set(window_snapshots[i-1].edges.keys())
            curr_edges = set(window_snapshots[i].edges.keys())
            
            changes = len(prev_edges.symmetric_difference(curr_edges))
            total_edges = len(prev_edges.union(curr_edges))
            
            if total_edges > 0:
                total_changes += changes
                total_comparisons += total_edges
        
        return total_changes / total_comparisons if total_comparisons > 0 else 0.0
    
    def _predict_node_heat(self, evolution: NodeEvolution, horizon: timedelta) -> float:
        """Predict future node heat based on historical trends"""
        
        if len(evolution.heat_history) < 3:
            return evolution.heat_history[-1][1] if evolution.heat_history else 0.0
        
        # Get recent heat values and timestamps
        recent_data = evolution.heat_history[-10:]  # Last 10 points
        times = np.array([(t - evolution.heat_history[0][0]).total_seconds() 
                         for t, _ in recent_data])
        heats = np.array([h for _, h in recent_data])
        
        # Simple linear trend extrapolation
        if len(times) >= 2:
            slope, intercept = np.polyfit(times, heats, 1)
            future_seconds = (datetime.now() + horizon - evolution.heat_history[0][0]).total_seconds()
            predicted_heat = slope * future_seconds + intercept
            
            # Apply bounds and decay
            predicted_heat = max(0.0, min(predicted_heat, 1.0))
            
            # Apply time decay
            decay_factor = np.exp(-0.0001 * horizon.total_seconds())
            predicted_heat *= decay_factor
            
            return predicted_heat
        
        return evolution.heat_history[-1][1]
    
    def _predict_edge_strength(self, evolution: EdgeEvolution, horizon: timedelta) -> float:
        """Predict future edge strength"""
        
        if not evolution.strength_history:
            return 0.0
        
        current_strength = evolution.strength_history[-1][1]
        
        # Apply decay over prediction horizon
        decay_time_hours = horizon.total_seconds() / 3600
        decay_factor = np.exp(-evolution.decay_rate * decay_time_hours)
        
        return current_strength * decay_factor
    
    def _predict_global_heat(self, horizon: timedelta) -> float:
        """Predict global heat level"""
        
        if not self.current_snapshot:
            return 0.0
        
        current_global = self.current_snapshot.global_heat
        
        # Simple decay model for global heat
        decay_factor = np.exp(-0.0002 * horizon.total_seconds())
        
        return current_global * decay_factor