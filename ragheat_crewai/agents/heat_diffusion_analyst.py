"""
Heat Diffusion Analyst Agent for RAGHeat CrewAI System
=====================================================

This agent specializes in modeling influence propagation using heat diffusion equations
on the financial knowledge graph to understand how events cascade through markets.
"""

from typing import Dict, Any, List
from .base_agent import RAGHeatBaseAgent
import logging

logger = logging.getLogger(__name__)

class HeatDiffusionAnalystAgent(RAGHeatBaseAgent):
    """
    Heat Diffusion Analyst Agent for influence propagation modeling.
    
    Specializes in:
    - Heat equation modeling on graphs
    - Influence propagation simulation
    - Event impact analysis
    - Cascade risk assessment
    - Diffusion-based predictions
    """
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Model influence propagation using heat diffusion equations.
        
        Args:
            input_data: Dictionary containing:
                - source_events: List of shock events to model
                - target_entities: Entities to analyze impact on
                - diffusion_params: Diffusion parameters (coefficient, iterations)
                - time_horizon: Time horizon for simulation
        
        Returns:
            Heat diffusion analysis results with influence maps
        """
        try:
            source_events = input_data.get("source_events", [])
            target_entities = input_data.get("target_entities", [])
            diffusion_params = input_data.get("diffusion_params", {"coefficient": 0.1, "iterations": 50})
            time_horizon = input_data.get("time_horizon", "30d")
            
            if not source_events and not target_entities:
                return {"error": "No source events or target entities provided"}
            
            logger.info(f"Heat diffusion analysis starting for {len(source_events)} events")
            
            # Prepare analysis context
            analysis_context = {
                "source_events": source_events,
                "target_entities": target_entities,
                "diffusion_params": diffusion_params,
                "time_horizon": time_horizon,
                "analysis_type": "heat_diffusion",
                "focus_areas": [
                    "shock_identification",
                    "diffusion_simulation",
                    "impact_assessment",
                    "cascade_analysis",
                    "risk_quantification"
                ]
            }
            
            # Execute heat diffusion analysis task
            task_description = f"""
            Model influence propagation using heat diffusion equations with the following specifications:
            
            Source Events: {', '.join(source_events) if source_events else 'Auto-detect recent shocks'}
            Target Entities: {', '.join(target_entities) if target_entities else 'All graph entities'}
            Diffusion Coefficient (β): {diffusion_params.get('coefficient', 0.1)}
            Simulation Iterations: {diffusion_params.get('iterations', 50)}
            Time Horizon: {time_horizon}
            
            HEAT DIFFUSION MODELING FRAMEWORK:
            
            1. SHOCK EVENT IDENTIFICATION AND QUANTIFICATION:
               - Event Impact Scoring: Quantify initial shock magnitude (0-1 scale)
                 * Federal Reserve rate decisions: |rate_change| / max_historical_change
                 * Earnings surprises: |actual - expected| / stock_price_volatility
                 * Geopolitical events: Market reaction magnitude / historical volatility
                 * Economic data releases: |actual - consensus| / historical_standard_deviation
               
               - Event Classification:
                 * Systematic shocks: Affect entire market (Fed decisions, GDP, wars)
                 * Sector-specific shocks: Industry regulation, commodity price changes
                 * Company-specific shocks: Earnings, M&A, management changes
                 * Cross-asset shocks: Currency moves, bond yield changes
               
               - Temporal Decay Modeling:
                 * Immediate impact (t=0): Full shock intensity
                 * Short-term decay (t=1-7d): Exponential decay with half-life
                 * Medium-term effects (t=1-4w): Power law decay
                 * Long-term equilibrium: Permanent vs temporary effect assessment
            
            2. GRAPH LAPLACIAN CONSTRUCTION:
               - Adjacency Matrix (A): Weighted connections between entities
                 * Correlation-based weights: w_ij = |correlation(i,j)|^α
                 * Fundamental linkages: Supply chain, ownership, sector membership
                 * Information flow: News co-mention frequency, analyst coverage overlap
                 * Trading relationships: Common institutional ownership, ETF composition
               
               - Degree Matrix (D): Node connectivity strength
                 * D_ii = Σ_j A_ij (sum of connection weights for node i)
                 * Normalization options: Symmetric, random walk, unnormalized
               
               - Laplacian Matrix (L): L = D - A
                 * Captures local diffusion dynamics
                 * Eigenvalue decomposition for diffusion analysis
                 * Spectral properties determine diffusion speed and patterns
            
            3. HEAT DIFFUSION EQUATION SETUP:
               - Continuous Heat Equation: ∂u/∂t = β∇²u = β·L·u
                 * u(x,t): Heat intensity at node x and time t
                 * β: Diffusion coefficient (controls propagation speed)
                 * L: Graph Laplacian operator
               
               - Discrete Time Solution: u(t+1) = u(t) + β·L·u(t)
                 * Euler method for numerical integration
                 * Stability condition: β < 1/λ_max(L) for convergence
               
               - Initial Conditions: u(x,0) = shock_intensity if x ∈ shock_sources, else 0
               - Boundary Conditions: Consider market structure constraints
            
            4. SIMULATION EXECUTION:
               - Iterative Diffusion Process:
                 * Initialize heat values at shock source nodes
                 * Apply diffusion operator iteratively
                 * Monitor convergence criteria (||u(t+1) - u(t)|| < ε)
                 * Track heat evolution over time steps
               
               - Multi-Event Superposition:
                 * Linear combination of multiple simultaneous shocks
                 * Non-linear interaction effects for large shocks
                 * Temporal sequencing of events
               
               - Parameter Sensitivity Analysis:
                 * Vary diffusion coefficient β
                 * Test different graph structures
                 * Analyze robustness to model assumptions
            
            5. IMPACT ANALYSIS AND INTERPRETATION:
               - Heat Intensity Rankings:
                 * Rank all entities by final heat intensity
                 * Identify most/least affected entities
                 * Compare to historical shock responses
               
               - Propagation Path Analysis:
                 * Trace influence paths from sources to targets
                 * Identify key transmission channels
                 * Measure path length and efficiency
                 * Find bottleneck nodes and amplification points
               
               - Temporal Dynamics:
                 * Heat evolution timelines for key entities
                 * Peak impact timing and duration
                 * Oscillation and damping patterns
                 * Equilibrium analysis
            
            6. RISK ASSESSMENT AND PREDICTION:
               - Cascade Risk Metrics:
                 * Systemic Risk Score: Average heat intensity across system
                 * Concentration Risk: Heat distribution inequality (Gini coefficient)
                 * Contagion Potential: Secondary shock amplification factors
               
               - Entity-Specific Risk Scores:
                 * Direct Exposure: Direct heat from source events
                 * Indirect Exposure: Heat received through network effects
                 * Amplification Factor: Heat intensity / baseline volatility
                 * Recovery Time: Expected time to return to baseline
               
               - Predictive Modeling:
                 * Use heat intensity to predict price movements
                 * Correlation between heat scores and future returns
                 * Risk-adjusted return expectations
                 * Optimal timing for contrarian positions
            
            DELIVERABLES:
            
            - Heat Map Visualization: Interactive graph showing heat propagation
            - Entity Heat Rankings: Sorted list of all entities by impact intensity
            - Propagation Timeline: Heat evolution over simulation time steps
            - Path Analysis: Key influence transmission routes
            - Risk Dashboard: Systemic and entity-specific risk metrics
            - Sensitivity Analysis: Robustness testing results
            - Prediction Models: Heat-based forecasting models
            - Cascade Scenarios: Stress testing under extreme shock scenarios
            
            Focus on identifying non-obvious indirect effects and providing
            actionable insights for risk management and investment timing.
            """
            
            result = self.execute_task(task_description, analysis_context)
            
            # Post-process results to ensure structured output
            processed_result = self._structure_diffusion_analysis(result, source_events, target_entities)
            
            logger.info(f"Heat diffusion analysis completed")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error in heat diffusion analysis: {e}")
            return {
                "error": str(e),
                "agent": "heat_diffusion_analyst",
                "analysis_type": "heat_diffusion"
            }
    
    def _structure_diffusion_analysis(self, raw_result: Dict[str, Any], source_events: List[str], target_entities: List[str]) -> Dict[str, Any]:
        """Structure the heat diffusion analysis results."""
        
        structured_result = {
            "analysis_type": "heat_diffusion",
            "agent": "heat_diffusion_analyst",
            "timestamp": self._get_current_timestamp(),
            "source_events": source_events,
            "target_entities": target_entities,
            "overall_analysis": raw_result.get("result", ""),
            "heat_rankings": {},
            "propagation_paths": [],
            "risk_metrics": {},
            "cascade_analysis": {},
            "predictions": {},
            "visualization_data": {}
        }
        
        # Extract structured data from result text
        result_text = str(raw_result.get("result", ""))
        
        # Extract heat rankings
        rankings = self._extract_heat_rankings(result_text)
        structured_result["heat_rankings"] = rankings
        
        # Extract risk metrics
        risk_metrics = self._extract_risk_metrics(result_text)
        structured_result["risk_metrics"] = risk_metrics
        
        # Extract propagation paths
        paths = self._extract_propagation_paths(result_text)
        structured_result["propagation_paths"] = paths
        
        return structured_result
    
    def _extract_heat_rankings(self, text: str) -> Dict[str, Any]:
        """Extract heat intensity rankings from result text."""
        rankings = {
            "most_affected": [],
            "least_affected": [],
            "heat_scores": {}
        }
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if 'most affected' in line.lower() or 'highest heat' in line.lower():
                # Extract entities with high heat
                words = line.split()
                for word in words:
                    if word.isupper() and len(word) <= 5:  # Likely a ticker
                        rankings["most_affected"].append(word)
            elif 'least affected' in line.lower() or 'lowest heat' in line.lower():
                # Extract entities with low heat
                words = line.split()
                for word in words:
                    if word.isupper() and len(word) <= 5:  # Likely a ticker
                        rankings["least_affected"].append(word)
        
        return rankings
    
    def _extract_risk_metrics(self, text: str) -> Dict[str, Any]:
        """Extract risk metrics from result text."""
        risk_metrics = {
            "systemic_risk_score": 0.5,  # Default values
            "concentration_risk": 0.4,
            "contagion_potential": 0.3,
            "cascade_probability": 0.2
        }
        
        # Simple extraction - would be enhanced with proper parsing
        text_lower = text.lower()
        if 'high risk' in text_lower or 'severe' in text_lower:
            for key in risk_metrics:
                risk_metrics[key] = min(risk_metrics[key] + 0.3, 1.0)
        elif 'low risk' in text_lower or 'minimal' in text_lower:
            for key in risk_metrics:
                risk_metrics[key] = max(risk_metrics[key] - 0.2, 0.0)
        
        return risk_metrics
    
    def _extract_propagation_paths(self, text: str) -> List[Dict[str, Any]]:
        """Extract propagation paths from result text."""
        paths = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if '->' in line or '→' in line:
                # This looks like a propagation path
                path_entities = line.replace('->', '→').split('→')
                if len(path_entities) >= 2:
                    paths.append({
                        "path": [entity.strip() for entity in path_entities],
                        "strength": 0.7,  # Default
                        "description": line
                    })
        
        return paths[:10]  # Limit to top 10 paths
    
    def simulate_shock_event(self, event_description: str, initial_impact: float, affected_entities: List[str] = None) -> Dict[str, Any]:
        """
        Simulate a specific shock event and its propagation.
        
        Args:
            event_description: Description of the shock event
            initial_impact: Initial impact magnitude (0-1)
            affected_entities: Initially affected entities
            
        Returns:
            Shock simulation results
        """
        task_description = f"""
        Simulate the heat diffusion effects of the following shock event:
        
        Event: {event_description}
        Initial Impact Magnitude: {initial_impact}
        Initially Affected Entities: {', '.join(affected_entities) if affected_entities else 'Auto-determine'}
        
        Simulation Requirements:
        
        1. EVENT CHARACTERIZATION:
           - Classify event type (monetary policy, earnings, geopolitical, etc.)
           - Determine direct impact entities based on event nature
           - Estimate initial shock intensity and duration
           - Model temporal decay pattern
        
        2. DIFFUSION SIMULATION:
           - Set initial heat values at directly affected entities
           - Run heat diffusion simulation for specified iterations
           - Track heat propagation through network connections
           - Monitor convergence and stability
        
        3. IMPACT ANALYSIS:
           - Identify entities with highest indirect heat exposure
           - Calculate propagation efficiency and transmission delays
           - Analyze amplification and dampening effects
           - Compare to historical similar events
        
        4. RISK ASSESSMENT:
           - Quantify cascade risk and systemic exposure
           - Identify potential secondary shock sources
           - Assess market stability and resilience
           - Recommend risk mitigation strategies
        
        Provide detailed timeline of heat propagation and entity-specific impact scores.
        """
        
        context = {
            "event_description": event_description,
            "initial_impact": initial_impact,
            "affected_entities": affected_entities or [],
            "analysis_type": "shock_simulation"
        }
        
        return self.execute_task(task_description, context)
    
    def analyze_cascade_risk(self, entities: List[str], stress_scenarios: List[str] = None) -> Dict[str, Any]:
        """
        Analyze cascade risk for specific entities under stress scenarios.
        
        Args:
            entities: Entities to analyze cascade risk for
            stress_scenarios: Stress scenarios to test
            
        Returns:
            Cascade risk analysis results
        """
        stress_scenarios = stress_scenarios or ["market_crash", "rate_shock", "geopolitical_crisis"]
        
        task_description = f"""
        Analyze cascade risk for entities: {', '.join(entities)}
        
        Stress Scenarios: {', '.join(stress_scenarios)}
        
        For each scenario and entity:
        
        1. DIRECT IMPACT ASSESSMENT:
           - Initial shock exposure based on entity characteristics
           - Fundamental vulnerability factors
           - Historical stress response patterns
           - Liquidity and solvency considerations
        
        2. INDIRECT IMPACT MODELING:
           - Network-based contagion effects
           - Supply chain disruption propagation
           - Funding market stress transmission
           - Cross-asset correlation effects
        
        3. AMPLIFICATION FACTORS:
           - Leverage and financial stability
           - Market concentration and interconnectedness
           - Behavioral factors (panic selling, margin calls)
           - Regulatory and policy responses
        
        4. SYSTEMIC RISK CONTRIBUTION:
           - Entity's contribution to overall system risk
           - Too-big-to-fail considerations
           - Critical infrastructure dependencies
           - Market making and liquidity provision roles
        
        5. RESILIENCE ANALYSIS:
           - Recovery capacity and time estimates
           - Support mechanisms and backstops
           - Adaptation and restructuring capabilities
           - Historical crisis navigation performance
        
        Provide entity-specific risk scores and system-wide stability assessment.
        """
        
        context = {
            "entities": entities,
            "stress_scenarios": stress_scenarios,
            "analysis_type": "cascade_risk"
        }
        
        return self.execute_task(task_description, context)
    
    def optimize_diffusion_parameters(self, historical_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize diffusion parameters based on historical event analysis.
        
        Args:
            historical_events: List of historical events with outcomes
            
        Returns:
            Optimized parameter recommendations
        """
        task_description = f"""
        Optimize heat diffusion parameters using historical event analysis:
        
        Historical Events: {len(historical_events)} events provided
        
        Optimization Process:
        
        1. HISTORICAL CALIBRATION:
           - Analyze actual market responses to historical shocks
           - Map observed price movements to heat diffusion patterns
           - Calculate optimal diffusion coefficients for different event types
           - Validate model accuracy against historical data
        
        2. PARAMETER SENSITIVITY:
           - Test diffusion coefficient ranges (0.01 - 0.5)
           - Analyze iteration count requirements for convergence
           - Evaluate graph structure variations
           - Assess temporal decay parameter optimization
        
        3. MODEL VALIDATION:
           - Cross-validation using held-out historical events
           - Calculate prediction accuracy metrics
           - Assess statistical significance of improvements
           - Compare to baseline models and benchmarks
        
        4. REGIME-SPECIFIC PARAMETERS:
           - Identify different market regimes (bull, bear, crisis)
           - Optimize parameters for each regime separately
           - Model regime transition effects
           - Adaptive parameter adjustment mechanisms
        
        5. ROBUSTNESS TESTING:
           - Stress test parameters under extreme scenarios
           - Analyze stability across different market conditions
           - Evaluate sensitivity to graph structure changes
           - Test computational efficiency and scalability
        
        Provide optimized parameter sets and usage guidelines.
        """
        
        context = {
            "historical_events": historical_events,
            "analysis_type": "parameter_optimization"
        }
        
        return self.execute_task(task_description, context)