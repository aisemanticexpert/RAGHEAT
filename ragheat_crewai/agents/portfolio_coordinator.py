"""
Portfolio Coordinator Agent for RAGHeat CrewAI System
====================================================

This agent coordinates multiple specialist agents through structured debate and
consensus-building to construct optimal risk-adjusted portfolios.
"""

from typing import Dict, Any, List
from .base_agent import RAGHeatBaseAgent
import logging

logger = logging.getLogger(__name__)

class PortfolioCoordinatorAgent(RAGHeatBaseAgent):
    """
    Portfolio Coordinator Agent for multi-agent coordination and consensus building.
    
    Specializes in:
    - Multi-agent debate facilitation
    - Consensus building and conflict resolution
    - Portfolio construction and optimization
    - Risk management and allocation
    - Investment decision coordination
    """
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate multi-agent analysis and build portfolio consensus.
        
        Args:
            input_data: Dictionary containing:
                - agent_analyses: Results from specialist agents
                - portfolio_constraints: Risk and allocation constraints
                - investment_objectives: Investment goals and preferences
                - debate_parameters: Debate structure and rules
        
        Returns:
            Portfolio coordination results with final recommendations
        """
        try:
            agent_analyses = input_data.get("agent_analyses", {})
            portfolio_constraints = input_data.get("portfolio_constraints", {})
            investment_objectives = input_data.get("investment_objectives", {})
            debate_parameters = input_data.get("debate_parameters", {"rounds": 3, "consensus_threshold": 0.7})
            
            if not agent_analyses:
                return {"error": "No agent analyses provided for coordination"}
            
            logger.info(f"Portfolio coordination starting with {len(agent_analyses)} agent inputs")
            
            # Prepare coordination context
            coordination_context = {
                "agent_analyses": agent_analyses,
                "portfolio_constraints": portfolio_constraints,
                "investment_objectives": investment_objectives,
                "debate_parameters": debate_parameters,
                "analysis_type": "portfolio_coordination",
                "focus_areas": [
                    "consensus_building",
                    "conflict_resolution",
                    "portfolio_optimization",
                    "risk_management",
                    "decision_integration"
                ]
            }
            
            # Execute portfolio coordination task
            task_description = f"""
            Coordinate multi-agent portfolio construction with the following specifications:
            
            Agent Analyses Available: {', '.join(agent_analyses.keys())}
            Portfolio Constraints: {portfolio_constraints}
            Investment Objectives: {investment_objectives}
            Debate Rounds: {debate_parameters.get('rounds', 3)}
            Consensus Threshold: {debate_parameters.get('consensus_threshold', 0.7)}
            
            MULTI-AGENT COORDINATION FRAMEWORK:
            
            1. AGENT ANALYSIS SYNTHESIS:
               - Fundamental Analysis Integration:
                 * Extract BUY/HOLD/SELL recommendations with confidence scores
                 * Consolidate financial health and growth assessments
                 * Identify valuation discrepancies and opportunities
                 * Synthesize long-term investment themes and catalysts
               
               - Sentiment Analysis Integration:
                 * Aggregate sentiment scores across news, social media, analysts
                 * Identify sentiment-momentum misalignments
                 * Assess short-term vs long-term sentiment divergences
                 * Incorporate contrarian opportunity identification
               
               - Technical/Valuation Analysis Integration:
                 * Combine technical signals with entry/exit timing
                 * Integrate risk-adjusted return expectations
                 * Synthesize volatility and correlation assessments
                 * Consolidate position sizing recommendations
               
               - Knowledge Graph Insights Integration:
                 * Incorporate structural relationship analysis
                 * Utilize sector and cross-asset correlations
                 * Leverage entity importance and centrality metrics
                 * Apply graph-based diversification insights
               
               - Heat Diffusion Impact Integration:
                 * Factor in cascade risk and systemic exposure
                 * Consider indirect shock transmission effects
                 * Incorporate influence propagation timing
                 * Assess portfolio resilience to market shocks
            
            2. STRUCTURED DEBATE FACILITATION:
               - Round 1 - Initial Position Presentation:
                 * Each agent presents core recommendations with evidence
                 * Identify areas of agreement and disagreement
                 * Highlight conflicting signals and their sources
                 * Establish debate priorities and key decision points
               
               - Round 2 - Evidence-Based Challenge:
                 * Agents challenge each other's assumptions and methodologies
                 * Present counter-evidence and alternative interpretations
                 * Defend recommendations with additional supporting data
                 * Identify gaps in analysis and missing considerations
               
               - Round 3 - Consensus Building:
                 * Focus on areas of strong agreement for immediate decisions
                 * Negotiate compromise positions for disputed recommendations
                 * Weight agent opinions by confidence levels and track records
                 * Establish conditions for position changes and updates
               
               - Final Synthesis:
                 * Integrate all perspectives into coherent investment strategy
                 * Document remaining disagreements and monitoring triggers
                 * Establish position confidence levels and review schedules
            
            3. CONFLICT RESOLUTION MECHANISMS:
               - Weighted Voting System:
                 * Weight agent votes by historical accuracy and confidence
                 * Apply domain expertise weighting (fundamental for long-term, sentiment for short-term)
                 * Use Bayesian updating based on agent track records
                 * Implement dynamic reweighting based on market conditions
               
               - Evidence Strength Assessment:
                 * Evaluate quality and quantity of supporting evidence
                 * Assess statistical significance and sample sizes
                 * Consider data freshness and relevance
                 * Weight quantitative vs qualitative evidence appropriately
               
               - Scenario Analysis Integration:
                 * Test recommendations under different market scenarios
                 * Assess robustness to assumption changes
                 * Evaluate sensitivity to parameter variations
                 * Consider tail risk and stress test outcomes
               
               - Temporal Reconciliation:
                 * Align short-term tactical with long-term strategic views
                 * Establish timing for position initiation and adjustment
                 * Balance momentum signals with value opportunities
                 * Coordinate entry/exit timing across positions
            
            4. PORTFOLIO CONSTRUCTION AND OPTIMIZATION:
               - Asset Selection Process:
                 * Rank all analyzed assets by consensus recommendation strength
                 * Apply minimum conviction thresholds for inclusion
                 * Consider correlation effects and diversification benefits
                 * Balance growth vs value vs momentum factors
               
               - Position Sizing Methodology:
                 * Base sizing on conviction levels and risk assessments
                 * Apply Kelly criterion or similar optimal sizing frameworks
                 * Consider liquidity constraints and market impact
                 * Implement maximum position limits and concentration controls
               
               - Risk Management Integration:
                 * Calculate portfolio Value-at-Risk and Expected Shortfall
                 * Assess correlation risk and concentration exposure
                 * Implement sector and factor diversification requirements
                 * Establish stop-loss and rebalancing triggers
               
               - Constraint Satisfaction:
                 * Ensure compliance with investment mandate and guidelines
                 * Respect ESG constraints and exclusion criteria
                 * Maintain liquidity requirements and cash buffers
                 * Balance active vs passive allocation targets
            
            5. RISK-ADJUSTED OPTIMIZATION:
               - Multi-Objective Optimization:
                 * Maximize expected return subject to risk constraints
                 * Optimize Sharpe ratio while maintaining diversification
                 * Balance growth potential with downside protection
                 * Consider transaction costs and implementation constraints
               
               - Dynamic Risk Modeling:
                 * Use time-varying volatility and correlation models
                 * Incorporate regime-switching and fat-tail distributions
                 * Model liquidity risk and market stress scenarios
                 * Account for model uncertainty and parameter estimation error
               
               - Behavioral Risk Considerations:
                 * Factor in investor psychology and behavioral biases
                 * Consider market sentiment and crowding effects
                 * Account for momentum and mean reversion cycles
                 * Integrate contrarian positioning opportunities
            
            6. IMPLEMENTATION AND MONITORING:
               - Execution Strategy:
                 * Optimize trade timing and market impact minimization
                 * Consider pre-market vs market hours execution
                 * Plan for partial fills and order management
                 * Coordinate simultaneous position adjustments
               
               - Monitoring Framework:
                 * Establish performance attribution methodology
                 * Define trigger conditions for position reviews
                 * Set up real-time risk monitoring and alerts
                 * Plan regular strategy review and rebalancing schedule
               
               - Adaptive Management:
                 * Update positions based on new information arrival
                 * Rebalance based on risk budget consumption
                 * Adjust sizing based on realized vs expected volatility
                 * Incorporate lessons learned and strategy refinements
            
            DELIVERABLES:
            
            - Portfolio Composition: Final asset allocation with position sizes
            - Consensus Report: Agreement levels and remaining disagreements
            - Risk Assessment: Comprehensive portfolio risk analysis
            - Implementation Plan: Step-by-step execution strategy
            - Monitoring Dashboard: KPIs and trigger conditions
            - Debate Transcript: Record of agent discussions and rationale
            - Sensitivity Analysis: Robustness testing results
            - Performance Expectations: Return and risk forecasts
            
            Focus on creating a robust, well-reasoned portfolio that balances
            multiple perspectives while maintaining clear risk management.
            """
            
            result = self.execute_task(task_description, coordination_context)
            
            # Post-process results to ensure structured output
            processed_result = self._structure_coordination_result(result, agent_analyses)
            
            logger.info(f"Portfolio coordination completed")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error in portfolio coordination: {e}")
            return {
                "error": str(e),
                "agent": "portfolio_coordinator",
                "analysis_type": "portfolio_coordination"
            }
    
    def _structure_coordination_result(self, raw_result: Dict[str, Any], agent_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the portfolio coordination results."""
        
        structured_result = {
            "analysis_type": "portfolio_coordination",
            "agent": "portfolio_coordinator",
            "timestamp": self._get_current_timestamp(),
            "input_analyses": list(agent_analyses.keys()),
            "overall_coordination": raw_result.get("result", ""),
            "portfolio_recommendations": {},
            "consensus_scores": {},
            "conflict_resolutions": [],
            "risk_assessment": {},
            "implementation_plan": {},
            "monitoring_framework": {}
        }
        
        # Extract structured data from result text
        result_text = str(raw_result.get("result", ""))
        
        # Extract portfolio recommendations
        recommendations = self._extract_portfolio_recommendations(result_text)
        structured_result["portfolio_recommendations"] = recommendations
        
        # Extract consensus scores
        consensus = self._extract_consensus_scores(result_text)
        structured_result["consensus_scores"] = consensus
        
        # Extract conflict resolutions
        conflicts = self._extract_conflict_resolutions(result_text)
        structured_result["conflict_resolutions"] = conflicts
        
        return structured_result
    
    def _extract_portfolio_recommendations(self, text: str) -> Dict[str, Any]:
        """Extract portfolio recommendations from coordination text."""
        recommendations = {
            "selected_assets": [],
            "position_weights": {},
            "total_positions": 0,
            "risk_level": "Moderate",
            "expected_return": 0.08,
            "expected_volatility": 0.15
        }
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if 'buy' in line.lower() or 'long' in line.lower():
                words = line.split()
                for word in words:
                    if word.isupper() and len(word) <= 5:  # Likely a ticker
                        recommendations["selected_assets"].append(word)
                        recommendations["position_weights"][word] = 0.05  # Default 5%
        
        recommendations["total_positions"] = len(recommendations["selected_assets"])
        return recommendations
    
    def _extract_consensus_scores(self, text: str) -> Dict[str, Any]:
        """Extract consensus scores from coordination text."""
        consensus = {
            "overall_consensus": 0.7,  # Default
            "high_consensus_items": [],
            "low_consensus_items": [],
            "agreement_level": "Moderate"
        }
        
        text_lower = text.lower()
        if 'strong consensus' in text_lower or 'high agreement' in text_lower:
            consensus["overall_consensus"] = 0.85
            consensus["agreement_level"] = "High"
        elif 'weak consensus' in text_lower or 'disagreement' in text_lower:
            consensus["overall_consensus"] = 0.5
            consensus["agreement_level"] = "Low"
        
        return consensus
    
    def _extract_conflict_resolutions(self, text: str) -> List[Dict[str, Any]]:
        """Extract conflict resolutions from coordination text."""
        conflicts = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if 'conflict' in line.lower() or 'disagreement' in line.lower():
                conflicts.append({
                    "conflict_description": line,
                    "resolution_method": "Weighted voting",
                    "final_decision": "Compromise position",
                    "confidence": 0.6
                })
        
        return conflicts[:5]  # Limit to top 5 conflicts
    
    def facilitate_agent_debate(self, agent_analyses: Dict[str, Any], debate_rounds: int = 3) -> Dict[str, Any]:
        """
        Facilitate structured debate between agents.
        
        Args:
            agent_analyses: Analyses from different agents
            debate_rounds: Number of debate rounds
            
        Returns:
            Debate facilitation results
        """
        task_description = f"""
        Facilitate structured debate between agents for {debate_rounds} rounds:
        
        Agent Analyses: {', '.join(agent_analyses.keys())}
        
        Debate Structure:
        
        ROUND 1 - POSITION PRESENTATION:
        - Each agent presents their core analysis and recommendations
        - Highlight key evidence and reasoning
        - Identify confidence levels and uncertainties
        - Present risk assessments and assumptions
        
        ROUND 2 - CHALLENGE AND DEFENSE:
        - Agents challenge each other's methodologies and conclusions
        - Present alternative interpretations of the same data
        - Defend positions with additional evidence
        - Identify methodological strengths and weaknesses
        
        ROUND 3 - CONSENSUS BUILDING:
        - Focus on areas of agreement
        - Negotiate compromise positions
        - Establish weighted recommendations
        - Define conditions for position changes
        
        Debate Rules:
        - Evidence-based arguments only
        - Respectful challenge of ideas, not agents
        - Quantitative support for claims when possible
        - Clear documentation of reasoning chains
        
        Output structured debate transcript with decision rationale.
        """
        
        context = {
            "agent_analyses": agent_analyses,
            "debate_rounds": debate_rounds,
            "analysis_type": "agent_debate"
        }
        
        return self.execute_task(task_description, context)
    
    def optimize_portfolio_allocation(self, asset_recommendations: Dict[str, Any], constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize portfolio allocation based on agent recommendations.
        
        Args:
            asset_recommendations: Recommendations from all agents
            constraints: Portfolio constraints and objectives
            
        Returns:
            Optimized portfolio allocation
        """
        constraints = constraints or {"max_position": 0.1, "min_position": 0.01, "max_sector": 0.3}
        
        task_description = f"""
        Optimize portfolio allocation based on multi-agent recommendations:
        
        Asset Recommendations: {len(asset_recommendations)} assets analyzed
        Constraints: {constraints}
        
        Optimization Framework:
        
        1. RECOMMENDATION SYNTHESIS:
           - Aggregate BUY/SELL signals across agents
           - Weight recommendations by agent confidence and track record
           - Resolve conflicting recommendations through evidence analysis
           - Establish conviction-weighted asset rankings
        
        2. PORTFOLIO CONSTRUCTION:
           - Select assets meeting minimum conviction thresholds
           - Apply risk-based position sizing (Kelly criterion or similar)
           - Ensure diversification across sectors and factors
           - Respect position size and concentration limits
        
        3. RISK OPTIMIZATION:
           - Minimize portfolio volatility for given expected return
           - Optimize risk-adjusted returns (Sharpe ratio)
           - Consider correlation structure and tail risks
           - Apply stress testing and scenario analysis
        
        4. CONSTRAINT SATISFACTION:
           - Enforce position size limits and concentration rules
           - Maintain sector and geographic diversification
           - Respect liquidity and trading volume constraints
           - Consider transaction costs and market impact
        
        5. SENSITIVITY ANALYSIS:
           - Test allocation sensitivity to input assumptions
           - Analyze impact of removing/adding positions
           - Evaluate robustness to correlation changes
           - Assess performance under different market regimes
        
        Provide optimal weights, risk metrics, and implementation guidance.
        """
        
        context = {
            "asset_recommendations": asset_recommendations,
            "constraints": constraints,
            "analysis_type": "portfolio_optimization"
        }
        
        return self.execute_task(task_description, context)
    
    def assess_portfolio_risk(self, portfolio_composition: Dict[str, float], market_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assess comprehensive portfolio risk metrics.
        
        Args:
            portfolio_composition: Portfolio weights by asset
            market_conditions: Current market environment
            
        Returns:
            Portfolio risk assessment
        """
        market_conditions = market_conditions or {"volatility_regime": "normal", "correlation_regime": "normal"}
        
        task_description = f"""
        Assess comprehensive portfolio risk for the given composition:
        
        Portfolio Assets: {len(portfolio_composition)} positions
        Market Conditions: {market_conditions}
        
        Risk Assessment Framework:
        
        1. TRADITIONAL RISK METRICS:
           - Portfolio volatility and Value-at-Risk (VaR)
           - Expected Shortfall (Conditional VaR)
           - Maximum drawdown estimates
           - Beta and correlation with benchmarks
        
        2. CONCENTRATION RISK:
           - Position concentration analysis
           - Sector and geographic concentration
           - Factor concentration (growth, value, momentum)
           - Single name concentration limits
        
        3. LIQUIDITY RISK:
           - Portfolio liquidity assessment
           - Market impact estimates for rebalancing
           - Liquidity stress testing
           - Days-to-liquidate analysis
        
        4. CORRELATION RISK:
           - Correlation structure analysis
           - Correlation breakdown scenarios
           - Cross-asset correlation effects
           - Regime-dependent correlation modeling
        
        5. TAIL RISK ANALYSIS:
           - Extreme scenario stress testing
           - Fat-tail distribution modeling
           - Copula-based dependency modeling
           - Black swan event preparation
        
        6. MODEL RISK:
           - Parameter estimation uncertainty
           - Model specification risk
           - Backtesting validation
           - Robustness to assumption changes
        
        Provide actionable risk management recommendations.
        """
        
        context = {
            "portfolio_composition": portfolio_composition,
            "market_conditions": market_conditions,
            "analysis_type": "portfolio_risk_assessment"
        }
        
        return self.execute_task(task_description, context)