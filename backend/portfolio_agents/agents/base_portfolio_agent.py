"""
Base Portfolio Agent Class
=========================

Base class providing common functionality for all portfolio construction agents.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from loguru import logger
from crewai import Agent

class BasePortfolioAgent(ABC):
    """Base class for all portfolio construction agents."""
    
    def __init__(self, agent_type: str, crewai_agent: Agent):
        """Initialize base portfolio agent."""
        self.agent_type = agent_type
        self.crewai_agent = crewai_agent
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
    @abstractmethod
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis specific to this agent type."""
        pass
    
    def execute_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a task using the CrewAI agent."""
        try:
            start_time = datetime.now()
            
            # Prepare context for the agent
            formatted_context = self._format_context_for_agent(context or {})
            
            # Create full prompt with context
            full_prompt = self._create_agent_prompt(task_description, formatted_context)
            
            # Execute using CrewAI agent
            result = self.crewai_agent.execute_task(full_prompt)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log execution
            execution_record = {
                'timestamp': start_time.isoformat(),
                'task_description': task_description,
                'execution_time': execution_time,
                'result_length': len(str(result)) if result else 0,
                'success': result is not None
            }
            self.execution_history.append(execution_record)
            
            # Process and structure result
            structured_result = self._structure_agent_result(result, task_description)
            
            return {
                'agent_type': self.agent_type,
                'task_description': task_description,
                'result': structured_result,
                'execution_metadata': execution_record
            }
            
        except Exception as e:
            logger.error(f"Error executing task for {self.agent_type}: {e}")
            
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'task_description': task_description,
                'error': str(e),
                'success': False
            }
            self.execution_history.append(error_record)
            
            return {
                'agent_type': self.agent_type,
                'task_description': task_description,
                'error': str(e),
                'execution_metadata': error_record
            }
    
    def _format_context_for_agent(self, context: Dict[str, Any]) -> str:
        """Format context data for agent consumption."""
        if not context:
            return "No additional context provided."
        
        formatted_parts = []
        
        # Format different types of context
        for key, value in context.items():
            if key == 'stocks':
                formatted_parts.append(f"Target stocks for analysis: {', '.join(value)}")
            elif key == 'market_data':
                formatted_parts.append(f"Market data available for: {len(value)} securities")
            elif key == 'previous_analysis':
                formatted_parts.append(f"Previous analysis results available from: {', '.join(value.keys())}")
            elif isinstance(value, (dict, list)):
                formatted_parts.append(f"{key.replace('_', ' ').title()}: {self._summarize_complex_data(value)}")
            else:
                formatted_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted_parts)
    
    def _summarize_complex_data(self, data: Any) -> str:
        """Summarize complex data structures."""
        if isinstance(data, dict):
            return f"Dictionary with {len(data)} keys: {', '.join(list(data.keys())[:5])}" + ("..." if len(data) > 5 else "")
        elif isinstance(data, list):
            return f"List with {len(data)} items"
        else:
            return str(data)[:100] + ("..." if len(str(data)) > 100 else "")
    
    def _create_agent_prompt(self, task_description: str, context: str) -> str:
        """Create comprehensive prompt for the agent."""
        prompt = f"""
        ROLE: {self.crewai_agent.role}
        
        GOAL: {self.crewai_agent.goal}
        
        TASK: {task_description}
        
        CONTEXT:
        {context}
        
        INSTRUCTIONS:
        1. Analyze the provided information using your specialized expertise
        2. Apply your analytical framework to generate insights
        3. Provide specific, actionable recommendations
        4. Include confidence levels and supporting evidence
        5. Identify any limitations or assumptions in your analysis
        
        Please provide a structured analysis with clear reasoning and conclusions.
        """
        
        return prompt.strip()
    
    def _structure_agent_result(self, raw_result: Any, task_description: str) -> Dict[str, Any]:
        """Structure the raw agent result into a consistent format."""
        if isinstance(raw_result, dict):
            return raw_result
        
        # Convert string/other results to structured format
        structured = {
            'analysis': str(raw_result) if raw_result else "No analysis provided",
            'agent_type': self.agent_type,
            'task': task_description,
            'timestamp': datetime.now().isoformat(),
            'confidence': self._extract_confidence_from_result(raw_result),
            'key_insights': self._extract_key_insights(raw_result),
            'recommendations': self._extract_recommendations(raw_result)
        }
        
        return structured
    
    def _extract_confidence_from_result(self, result: Any) -> float:
        """Extract confidence level from agent result."""
        result_str = str(result).lower()
        
        # Look for confidence indicators
        confidence_keywords = {
            'very confident': 0.9,
            'confident': 0.8,
            'moderately confident': 0.7,
            'somewhat confident': 0.6,
            'uncertain': 0.4,
            'very uncertain': 0.2
        }
        
        for keyword, score in confidence_keywords.items():
            if keyword in result_str:
                return score
        
        # Default confidence
        return 0.6
    
    def _extract_key_insights(self, result: Any) -> List[str]:
        """Extract key insights from agent result."""
        result_str = str(result)
        insights = []
        
        # Look for bullet points or numbered items
        lines = result_str.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith('•') or line.startswith('-') or 
                line.startswith('*') or any(line.startswith(f'{i}.') for i in range(1, 10))):
                insight = line.lstrip('•-*0123456789. ').strip()
                if insight and len(insight) > 10:  # Filter out short/empty insights
                    insights.append(insight)
        
        return insights[:5]  # Limit to top 5 insights
    
    def _extract_recommendations(self, result: Any) -> List[str]:
        """Extract recommendations from agent result."""
        result_str = str(result).lower()
        recommendations = []
        
        # Look for recommendation keywords
        recommendation_indicators = [
            'recommend', 'suggest', 'advise', 'should', 'consider',
            'buy', 'sell', 'hold', 'avoid', 'increase', 'decrease'
        ]
        
        lines = result_str.split('\n')
        for line in lines:
            if any(indicator in line for indicator in recommendation_indicators):
                clean_line = line.strip()
                if len(clean_line) > 15:  # Filter out short lines
                    recommendations.append(clean_line.capitalize())
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this agent."""
        if not self.execution_history:
            return {'message': 'No execution history available'}
        
        successful_executions = [ex for ex in self.execution_history if ex.get('success', False)]
        total_executions = len(self.execution_history)
        
        avg_execution_time = sum(ex.get('execution_time', 0) for ex in successful_executions) / max(len(successful_executions), 1)
        
        return {
            'agent_type': self.agent_type,
            'total_executions': total_executions,
            'successful_executions': len(successful_executions),
            'success_rate': len(successful_executions) / total_executions if total_executions > 0 else 0,
            'average_execution_time': avg_execution_time,
            'last_execution': self.execution_history[-1]['timestamp'] if self.execution_history else None
        }
    
    def reset_history(self):
        """Reset execution history."""
        self.execution_history = []
        self.performance_metrics = {}
        logger.info(f"Reset history for agent: {self.agent_type}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of the agent."""
        return {
            'agent_type': self.agent_type,
            'status': 'active',
            'total_executions': len(self.execution_history),
            'performance_summary': self.get_performance_summary(),
            'capabilities': {
                'role': self.crewai_agent.role,
                'goal': self.crewai_agent.goal,
                'tools_count': len(self.crewai_agent.tools) if hasattr(self.crewai_agent, 'tools') else 0
            }
        }

class SpecializedPortfolioAgent(BasePortfolioAgent):
    """Specialized implementation of portfolio agent with enhanced capabilities."""
    
    def __init__(self, agent_type: str, crewai_agent: Agent, specialization_config: Optional[Dict[str, Any]] = None):
        """Initialize specialized portfolio agent."""
        super().__init__(agent_type, crewai_agent)
        self.specialization_config = specialization_config or {}
        self.domain_knowledge = self._load_domain_knowledge()
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform specialized analysis based on agent type."""
        analysis_method = getattr(self, f'_analyze_{self.agent_type}', self._default_analysis)
        return analysis_method(input_data)
    
    def _load_domain_knowledge(self) -> Dict[str, Any]:
        """Load domain-specific knowledge for the agent."""
        domain_knowledge = {
            'fundamental_analyst': {
                'key_metrics': ['P/E', 'P/B', 'ROE', 'Debt/Equity', 'Free Cash Flow'],
                'analysis_frameworks': ['DCF', 'Comparable Company Analysis', 'Precedent Transactions']
            },
            'sentiment_analyst': {
                'data_sources': ['News', 'Social Media', 'Analyst Reports'],
                'sentiment_models': ['VADER', 'FinBERT', 'TextBlob']
            },
            'valuation_analyst': {
                'technical_indicators': ['RSI', 'MACD', 'Bollinger Bands', 'Moving Averages'],
                'chart_patterns': ['Head and Shoulders', 'Double Top/Bottom', 'Triangles']
            }
        }
        
        return domain_knowledge.get(self.agent_type, {})
    
    def _default_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default analysis method."""
        return {
            'agent_type': self.agent_type,
            'analysis': 'Default analysis performed',
            'input_summary': f"Analyzed {len(input_data)} data points",
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_fundamental_analyst(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized analysis for fundamental analyst."""
        stocks = input_data.get('stocks', [])
        market_data = input_data.get('market_data', {})
        
        task_description = f"""
        Perform fundamental analysis on the following stocks: {', '.join(stocks)}
        
        Focus on:
        1. Financial health assessment using key ratios
        2. Growth prospects and competitive positioning
        3. Valuation relative to intrinsic value
        4. Risk factors and potential catalysts
        
        Provide specific BUY/HOLD/SELL recommendations with confidence levels.
        """
        
        return self.execute_task(task_description, input_data)
    
    def _analyze_sentiment_analyst(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized analysis for sentiment analyst."""
        stocks = input_data.get('stocks', [])
        
        task_description = f"""
        Analyze market sentiment for: {', '.join(stocks)}
        
        Evaluate:
        1. News sentiment and media coverage tone
        2. Social media buzz and retail investor sentiment
        3. Analyst opinion changes and institutional sentiment
        4. Momentum and behavioral factors
        
        Provide sentiment scores and short-term price direction predictions.
        """
        
        return self.execute_task(task_description, input_data)
    
    def _analyze_valuation_analyst(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized analysis for valuation analyst."""
        stocks = input_data.get('stocks', [])
        
        task_description = f"""
        Conduct technical and quantitative valuation analysis for: {', '.join(stocks)}
        
        Analyze:
        1. Technical indicators and chart patterns
        2. Price and volume trends
        3. Risk-adjusted return metrics
        4. Relative valuation vs peers and market
        
        Provide entry/exit points and risk management recommendations.
        """
        
        return self.execute_task(task_description, input_data)