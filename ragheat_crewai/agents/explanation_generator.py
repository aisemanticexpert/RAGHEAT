"""
Explanation Generator Agent for RAGHeat CrewAI System
====================================================

This agent generates clear, traceable explanations for investment decisions using
chain-of-thought reasoning and visual representations to ensure transparency.
"""

from typing import Dict, Any, List
from .base_agent import RAGHeatBaseAgent
import logging

logger = logging.getLogger(__name__)

class ExplanationGeneratorAgent(RAGHeatBaseAgent):
    """
    Explanation Generator Agent for investment rationale and transparency.
    
    Specializes in:
    - Investment decision explanation
    - Chain-of-thought reasoning
    - Visual representation creation
    - Regulatory compliance documentation
    - Multi-audience communication
    """
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive explanations for investment decisions.
        
        Args:
            input_data: Dictionary containing:
                - investment_decisions: Final portfolio decisions
                - agent_analyses: Supporting analyses from all agents
                - decision_process: Step-by-step decision process
                - target_audience: Audience for explanations
        
        Returns:
            Comprehensive explanation package
        """
        try:
            investment_decisions = input_data.get("investment_decisions", {})
            agent_analyses = input_data.get("agent_analyses", {})
            decision_process = input_data.get("decision_process", {})
            target_audience = input_data.get("target_audience", ["retail", "institutional"])
            
            if not investment_decisions:
                return {"error": "No investment decisions provided for explanation"}
            
            logger.info(f"Explanation generation starting for {len(investment_decisions)} decisions")
            
            # Prepare explanation context
            explanation_context = {
                "investment_decisions": investment_decisions,
                "agent_analyses": agent_analyses,
                "decision_process": decision_process,
                "target_audience": target_audience,
                "analysis_type": "explanation_generation",
                "focus_areas": [
                    "decision_rationale",
                    "evidence_trail",
                    "risk_disclosure",
                    "visual_representation",
                    "regulatory_compliance"
                ]
            }
            
            # Execute explanation generation task
            task_description = f"""
            Generate comprehensive investment decision explanations with the following specifications:
            
            Investment Decisions: {len(investment_decisions)} positions/recommendations
            Supporting Analyses: {', '.join(agent_analyses.keys()) if agent_analyses else 'None provided'}
            Target Audiences: {', '.join(target_audience)}
            
            EXPLANATION GENERATION FRAMEWORK:
            
            1. EXECUTIVE SUMMARY CREATION:
               - Investment Strategy Overview:
                 * Core investment thesis and philosophy
                 * Portfolio objectives and time horizon
                 * Risk tolerance and return expectations
                 * Key market assumptions and outlook
               
               - Portfolio Composition Summary:
                 * Asset allocation breakdown by sector/geography/style
                 * Number of positions and concentration levels
                 * Expected return and risk characteristics
                 * Benchmark comparison and tracking error
               
               - Key Investment Themes:
                 * Top 3-5 investment themes driving decisions
                 * Macro-economic factors and market trends
                 * Sector rotation and style preferences
                 * Contrarian vs momentum positioning
               
               - Risk Summary:
                 * Primary risk factors and mitigation strategies
                 * Stress testing results and tail risk assessment
                 * Liquidity considerations and market impact
                 * Regulatory and compliance considerations
            
            2. DECISION RATIONALE DOCUMENTATION:
               - Chain-of-Thought Reasoning:
                 * Step-by-step decision logic for each position
                 * Evidence evaluation and weight assignment
                 * Alternative scenarios considered and rejected
                 * Confidence levels and uncertainty quantification
               
               - Multi-Agent Analysis Integration:
                 * Fundamental analysis contribution and key insights
                 * Sentiment analysis impact on timing and sizing
                 * Technical analysis entry/exit point determination
                 * Knowledge graph relationship insights
                 * Heat diffusion risk and opportunity identification
                 * Portfolio coordination consensus building process
               
               - Evidence Hierarchy and Weighting:
                 * Primary evidence sources and their reliability
                 * Quantitative vs qualitative evidence balance
                 * Conflicting evidence resolution methodology
                 * Statistical significance and confidence intervals
               
               - Decision Trees and Branching Logic:
                 * Conditional decision pathways
                 * Scenario-dependent position adjustments
                 * Trigger conditions for position changes
                 * Exit strategy and stop-loss protocols
            
            3. INDIVIDUAL POSITION EXPLANATIONS:
               For each investment position, provide:
               
               - Investment Thesis:
                 * Why this specific asset was selected
                 * Key value drivers and growth catalysts
                 * Competitive advantages and moat analysis
                 * Valuation methodology and price targets
               
               - Supporting Evidence:
                 * Fundamental analysis highlights (financial health, growth)
                 * Sentiment analysis insights (momentum, contrarian signals)
                 * Technical analysis confirmation (entry points, risk levels)
                 * Heat diffusion analysis (systemic risk, cascade effects)
                 * Knowledge graph insights (relationships, correlations)
               
               - Risk Assessment:
                 * Position-specific risk factors and probability assessment
                 * Downside scenarios and loss limitation strategies
                 * Correlation risks and portfolio impact
                 * Liquidity risk and market impact considerations
               
               - Position Sizing Rationale:
                 * Conviction level and confidence-based sizing
                 * Risk-adjusted return expectations
                 * Correlation effects and diversification benefits
                 * Kelly criterion or alternative sizing methodology
            
            4. VISUAL REPRESENTATION DESIGN:
               - Portfolio Visualization Specifications:
                 * Interactive portfolio composition charts (pie/treemap)
                 * Risk-return scatter plots with efficient frontier
                 * Correlation matrix heatmaps
                 * Sector and geographic allocation maps
               
               - Heat Diffusion Visualizations:
                 * Network graphs showing entity relationships
                 * Heat propagation animation sequences
                 * Influence pathway diagrams
                 * Cascade risk scenario visualizations
               
               - Performance Attribution Charts:
                 * Return decomposition by factor/sector/position
                 * Risk contribution analysis
                 * Active vs passive return attribution
                 * Benchmark comparison waterfall charts
               
               - Decision Process Flowcharts:
                 * Multi-agent analysis integration workflow
                 * Consensus building and conflict resolution
                 * Risk management and position sizing process
                 * Monitoring and rebalancing triggers
            
            5. MULTI-AUDIENCE COMMUNICATION:
               - Retail Investor Version:
                 * Plain language explanations without jargon
                 * Focus on key investment themes and rationale
                 * Clear risk warnings and suitability considerations
                 * Visual emphasis on portfolio composition and expected outcomes
               
               - Institutional Investor Version:
                 * Technical methodology details and assumptions
                 * Statistical measures and confidence intervals
                 * Factor exposure analysis and style attribution
                 * Risk management framework and stress testing results
               
               - Regulatory/Compliance Version:
                 * Formal investment process documentation
                 * Fiduciary duty and suitability analysis
                 * Risk disclosure and limitation statements
                 * Audit trail and decision documentation
               
               - Internal/Management Version:
                 * Agent performance attribution and accuracy metrics
                 * Model validation and backtesting results
                 * Process improvement recommendations
                 * Technology and infrastructure requirements
            
            6. TRANSPARENCY AND AUDITABILITY:
               - Evidence Trail Documentation:
                 * Complete data source inventory and access timestamps
                 * Analysis methodology version control
                 * Agent decision weights and influence tracking
                 * Assumption documentation and sensitivity analysis
               
               - Decision Audit Log:
                 * Chronological decision timeline with rationale
                 * Alternative scenarios considered and rejection reasons
                 * Stakeholder input and consultation records
                 * Model output validation and override documentation
               
               - Conflict of Interest Disclosure:
                 * Potential conflicts identification and mitigation
                 * Data provider relationships and dependencies
                 * Algorithm bias assessment and correction
                 * Third-party influence and independence verification
            
            7. RISK DISCLOSURE AND LIMITATIONS:
               - Model Limitations:
                 * Statistical model assumptions and validity conditions
                 * Historical data limitations and regime changes
                 * Prediction uncertainty and confidence intervals
                 * Black swan events and tail risk considerations
               
               - Implementation Risks:
                 * Market impact and liquidity constraints
                 * Execution timing and slippage expectations
                 * Operational risk and technology dependencies
                 * Regulatory changes and compliance risks
               
               - Performance Disclaimers:
                 * Past performance and future results limitations
                 * Benchmark selection and comparison validity
                 * Market environment dependencies
                 * Rebalancing frequency and cost considerations
            
            DELIVERABLE SPECIFICATIONS:
            
            - Executive Summary Report (2-3 pages):
              * Investment strategy overview and key themes
              * Portfolio composition and risk characteristics
              * Expected outcomes and success metrics
            
            - Detailed Analysis Report (10-15 pages):
              * Complete position-by-position rationale
              * Multi-agent analysis integration
              * Risk assessment and mitigation strategies
              * Implementation and monitoring framework
            
            - Visual Dashboard Package:
              * Interactive portfolio visualization
              * Heat diffusion network diagrams
              * Risk-return analysis charts
              * Performance attribution graphics
            
            - Regulatory Documentation:
              * Compliance checklist and verification
              * Fiduciary standard documentation
              * Risk disclosure statements
              * Audit trail and evidence log
            
            - Plain Language Summary (1 page):
              * Key investment decisions in simple terms
              * Risk warnings and suitability guidance
              * Expected outcomes and timeline
              * Contact information and support resources
            
            Focus on creating clear, actionable explanations that build confidence
            and trust while maintaining full transparency and regulatory compliance.
            """
            
            result = self.execute_task(task_description, explanation_context)
            
            # Post-process results to ensure structured output
            processed_result = self._structure_explanation_result(result, investment_decisions)
            
            logger.info(f"Explanation generation completed")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error in explanation generation: {e}")
            return {
                "error": str(e),
                "agent": "explanation_generator",
                "analysis_type": "explanation_generation"
            }
    
    def _structure_explanation_result(self, raw_result: Dict[str, Any], investment_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the explanation generation results."""
        
        structured_result = {
            "analysis_type": "explanation_generation",
            "agent": "explanation_generator",
            "timestamp": self._get_current_timestamp(),
            "decisions_explained": len(investment_decisions),
            "overall_explanation": raw_result.get("result", ""),
            "executive_summary": {},
            "position_explanations": {},
            "visual_specifications": {},
            "risk_disclosures": [],
            "evidence_trail": {},
            "audience_versions": {}
        }
        
        # Extract structured data from result text
        result_text = str(raw_result.get("result", ""))
        
        # Extract executive summary
        summary = self._extract_executive_summary(result_text)
        structured_result["executive_summary"] = summary
        
        # Extract risk disclosures
        risks = self._extract_risk_disclosures(result_text)
        structured_result["risk_disclosures"] = risks
        
        # Extract visual specifications
        visuals = self._extract_visual_specifications(result_text)
        structured_result["visual_specifications"] = visuals
        
        return structured_result
    
    def _extract_executive_summary(self, text: str) -> Dict[str, Any]:
        """Extract executive summary from explanation text."""
        summary = {
            "investment_thesis": "",
            "portfolio_composition": {},
            "key_themes": [],
            "risk_summary": "",
            "expected_outcomes": {}
        }
        
        # Extract key themes
        lines = text.split('\n')
        themes = []
        for line in lines:
            line = line.strip()
            if 'theme:' in line.lower() or 'strategy:' in line.lower():
                theme = line.split(':', 1)[-1].strip()
                if theme and len(theme) > 10:
                    themes.append(theme)
        
        summary["key_themes"] = themes[:5]  # Limit to top 5 themes
        return summary
    
    def _extract_risk_disclosures(self, text: str) -> List[str]:
        """Extract risk disclosures from explanation text."""
        disclosures = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['risk:', 'limitation:', 'warning:', 'disclaimer:']):
                disclosure = line.split(':', 1)[-1].strip() if ':' in line else line
                if disclosure and len(disclosure) > 20:
                    disclosures.append(disclosure)
        
        return disclosures[:10]  # Limit to top 10 disclosures
    
    def _extract_visual_specifications(self, text: str) -> Dict[str, Any]:
        """Extract visual specifications from explanation text."""
        visuals = {
            "charts_recommended": [],
            "visualization_types": [],
            "interactive_elements": [],
            "dashboard_components": []
        }
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip().lower()
            if 'chart' in line or 'graph' in line or 'visualization' in line:
                if 'pie' in line:
                    visuals["charts_recommended"].append("pie_chart")
                elif 'heatmap' in line:
                    visuals["charts_recommended"].append("heatmap")
                elif 'network' in line:
                    visuals["charts_recommended"].append("network_graph")
                elif 'scatter' in line:
                    visuals["charts_recommended"].append("scatter_plot")
        
        return visuals
    
    def generate_executive_summary(self, portfolio_decisions: Dict[str, Any], key_insights: List[str]) -> Dict[str, Any]:
        """
        Generate executive summary for investment decisions.
        
        Args:
            portfolio_decisions: Portfolio composition and decisions
            key_insights: Key insights from analysis
            
        Returns:
            Executive summary
        """
        task_description = f"""
        Generate executive summary for portfolio decisions:
        
        Portfolio Decisions: {len(portfolio_decisions)} positions
        Key Insights: {len(key_insights)} insights provided
        
        Executive Summary Requirements:
        
        1. INVESTMENT STRATEGY OVERVIEW (2-3 sentences):
           - Core investment philosophy and approach
           - Market outlook and positioning
           - Risk-return objectives
        
        2. PORTFOLIO HIGHLIGHTS:
           - Number of positions and concentration
           - Sector and geographic allocation
           - Expected return and risk metrics
           - Benchmark comparison
        
        3. KEY INVESTMENT THEMES (Top 3):
           - Primary themes driving investment decisions
           - Supporting rationale and evidence
           - Expected impact and timeline
        
        4. RISK SUMMARY:
           - Main risk factors and mitigation
           - Stress testing results
           - Monitoring and adjustment triggers
        
        5. EXPECTED OUTCOMES:
           - Performance expectations
           - Success metrics and timeline
           - Review and rebalancing schedule
        
        Keep summary concise, clear, and action-oriented.
        """
        
        context = {
            "portfolio_decisions": portfolio_decisions,
            "key_insights": key_insights,
            "analysis_type": "executive_summary"
        }
        
        return self.execute_task(task_description, context)
    
    def create_position_explanations(self, positions: Dict[str, Any], supporting_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create detailed explanations for individual positions.
        
        Args:
            positions: Individual position details
            supporting_analyses: Supporting analysis from agents
            
        Returns:
            Position-by-position explanations
        """
        task_description = f"""
        Create detailed explanations for {len(positions)} individual positions:
        
        Positions: {', '.join(positions.keys())}
        Supporting Analyses: {', '.join(supporting_analyses.keys())}
        
        For each position, explain:
        
        1. INVESTMENT THESIS:
           - Why this specific asset was selected
           - Key value drivers and catalysts
           - Competitive advantages and moat
           - Valuation methodology
        
        2. SUPPORTING EVIDENCE:
           - Fundamental analysis highlights
           - Sentiment analysis insights
           - Technical analysis confirmation
           - Heat diffusion risk assessment
           - Knowledge graph relationships
        
        3. POSITION SIZING RATIONALE:
           - Conviction level and confidence
           - Risk-adjusted sizing methodology
           - Correlation and diversification effects
           - Maximum loss and stop-loss levels
        
        4. RISK ASSESSMENT:
           - Position-specific risks
           - Downside scenarios
           - Correlation risks
           - Liquidity considerations
        
        5. MONITORING PLAN:
           - Key metrics to track
           - Trigger conditions for changes
           - Review frequency and criteria
           - Exit strategy
        
        Provide clear, evidence-based rationale for each decision.
        """
        
        context = {
            "positions": positions,
            "supporting_analyses": supporting_analyses,
            "analysis_type": "position_explanations"
        }
        
        return self.execute_task(task_description, context)
    
    def design_visualization_package(self, explanation_data: Dict[str, Any], target_audience: str = "institutional") -> Dict[str, Any]:
        """
        Design comprehensive visualization package.
        
        Args:
            explanation_data: Data to visualize
            target_audience: Target audience for visualizations
            
        Returns:
            Visualization design specifications
        """
        task_description = f"""
        Design visualization package for {target_audience} audience:
        
        Data Elements: {len(explanation_data)} components to visualize
        Target Audience: {target_audience}
        
        Visualization Requirements:
        
        1. PORTFOLIO OVERVIEW VISUALS:
           - Interactive portfolio composition (pie/treemap)
           - Asset allocation breakdown by multiple dimensions
           - Risk-return scatter plot with efficient frontier
           - Sector and geographic heat maps
        
        2. HEAT DIFFUSION VISUALIZATIONS:
           - Network graph of entity relationships
           - Heat propagation animation over time
           - Influence pathway flow diagrams
           - Cascade risk scenario comparisons
        
        3. PERFORMANCE ANALYTICS:
           - Return attribution waterfall charts
           - Risk contribution analysis
           - Factor exposure spider charts
           - Benchmark comparison dashboards
        
        4. DECISION PROCESS FLOWS:
           - Multi-agent analysis integration
           - Consensus building visualization
           - Conflict resolution mapping
           - Evidence weighting displays
        
        5. INTERACTIVE ELEMENTS:
           - Drill-down capabilities
           - Scenario analysis sliders
           - Time-series animation controls
           - Filter and search functionality
        
        Specify chart types, data requirements, and interaction design.
        """
        
        context = {
            "explanation_data": explanation_data,
            "target_audience": target_audience,
            "analysis_type": "visualization_design"
        }
        
        return self.execute_task(task_description, context)