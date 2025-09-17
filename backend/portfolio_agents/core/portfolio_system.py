"""
RAGHeat Portfolio Construction System
===================================

Main orchestration system for multi-agent portfolio construction using heat diffusion
and consensus-building approaches.
"""

from typing import Dict, Any, List, Optional
import yaml
import os
from datetime import datetime
from loguru import logger
from crewai import Crew, Task, Process

from ..agents.agent_factory import AgentFactory
from ..agents.base_portfolio_agent import SpecializedPortfolioAgent
from ..config.settings import settings
from .task_orchestrator import TaskOrchestrator
from .workflow_manager import WorkflowManager

class RAGHeatPortfolioSystem:
    """Main system for RAGHeat portfolio construction."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the RAGHeat portfolio system."""
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), '..', 'config')
        
        # Initialize components
        self.agent_factory = AgentFactory()
        self.task_orchestrator = TaskOrchestrator(self.config_dir)
        self.workflow_manager = WorkflowManager()
        
        # System state
        self.agents: Dict[str, Any] = {}
        self.crew: Optional[Crew] = None
        self.execution_history: List[Dict[str, Any]] = []
        self.system_status = "initialized"
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the complete system."""
        try:
            logger.info("Initializing RAGHeat Portfolio System...")
            
            # Validate configuration
            validation_result = self.agent_factory.validate_configuration()
            if not validation_result['valid']:
                logger.error(f"Configuration validation failed: {validation_result['errors']}")
                self.system_status = "configuration_error"
                return
            
            # Create agents
            self.agents = self._create_specialized_agents()
            
            # Create crew
            self.crew = self._create_crew()
            
            # Initialize workflow
            self.workflow_manager.initialize()
            
            self.system_status = "ready"
            logger.info("RAGHeat Portfolio System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            self.system_status = "initialization_error"
    
    def _create_specialized_agents(self) -> Dict[str, SpecializedPortfolioAgent]:
        """Create specialized portfolio agents."""
        specialized_agents = {}
        
        # Get all agent types from configuration
        agent_types = self.agent_factory.list_available_agents()
        
        for agent_type in agent_types:
            try:
                # Create CrewAI agent
                crewai_agent = self.agent_factory.create_agent(agent_type)
                
                if crewai_agent:
                    # Wrap in specialized agent
                    specialized_agent = SpecializedPortfolioAgent(
                        agent_type=agent_type,
                        crewai_agent=crewai_agent
                    )
                    specialized_agents[agent_type] = specialized_agent
                    logger.info(f"Created specialized agent: {agent_type}")
                
            except Exception as e:
                logger.error(f"Error creating specialized agent {agent_type}: {e}")
        
        return specialized_agents
    
    def _create_crew(self) -> Crew:
        """Create CrewAI crew from agents."""
        try:
            # Get CrewAI agent instances
            crewai_agents = [agent.crewai_agent for agent in self.agents.values()]
            
            # Get tasks from orchestrator
            tasks = self.task_orchestrator.create_all_tasks(crewai_agents)
            
            # Create crew
            crew = Crew(
                agents=crewai_agents,
                tasks=tasks,
                process=Process.hierarchical,  # Use hierarchical process for coordination
                verbose=settings.VERBOSE_LOGGING,
                memory=settings.ENABLE_MEMORY
            )
            
            logger.info(f"Created crew with {len(crewai_agents)} agents and {len(tasks)} tasks")
            return crew
            
        except Exception as e:
            logger.error(f"Error creating crew: {e}")
            return None
    
    def construct_portfolio(self, 
                          stocks: List[str], 
                          market_data: Optional[Dict[str, Any]] = None,
                          risk_profile: str = "moderate",
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main method to construct portfolio using multi-agent system."""
        try:
            if self.system_status != "ready":
                return {
                    'error': f'System not ready: {self.system_status}',
                    'status': self.system_status
                }
            
            logger.info(f"Starting portfolio construction for {len(stocks)} stocks")
            
            # Prepare input data
            input_data = {
                'stocks': stocks,
                'market_data': market_data or {},
                'risk_profile': risk_profile,
                'constraints': constraints or {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Execute workflow
            workflow_result = self.workflow_manager.execute_workflow(
                workflow_type="portfolio_construction",
                input_data=input_data,
                agents=self.agents
            )
            
            # If workflow execution is successful, run CrewAI crew
            if workflow_result.get('status') == 'success':
                crew_result = self._execute_crew_workflow(input_data)
                workflow_result['crew_execution'] = crew_result
            
            # Record execution
            execution_record = {
                'timestamp': datetime.now().isoformat(),
                'input_data': input_data,
                'result': workflow_result,
                'success': workflow_result.get('status') == 'success'
            }
            self.execution_history.append(execution_record)
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Error in portfolio construction: {e}")
            return {
                'error': str(e),
                'status': 'execution_error',
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_crew_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CrewAI crew workflow."""
        try:
            if not self.crew:
                return {'error': 'Crew not initialized'}
            
            # Create main portfolio construction task
            main_task_input = self._format_crew_input(input_data)
            
            # Execute crew
            result = self.crew.kickoff(inputs=main_task_input)
            
            return {
                'crew_result': result,
                'execution_time': None,  # CrewAI doesn't provide timing by default
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error executing crew workflow: {e}")
            return {
                'error': str(e),
                'status': 'crew_execution_error'
            }
    
    def _format_crew_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format input data for CrewAI crew."""
        return {
            'target_stocks': ', '.join(input_data['stocks']),
            'risk_profile': input_data['risk_profile'],
            'market_context': self._summarize_market_data(input_data.get('market_data', {})),
            'constraints': str(input_data.get('constraints', {})),
            'analysis_timestamp': input_data['timestamp']
        }
    
    def _summarize_market_data(self, market_data: Dict[str, Any]) -> str:
        """Summarize market data for agents."""
        if not market_data:
            return "No specific market data provided"
        
        summary_parts = []
        for key, value in market_data.items():
            if isinstance(value, dict):
                summary_parts.append(f"{key}: {len(value)} data points")
            else:
                summary_parts.append(f"{key}: {value}")
        
        return "; ".join(summary_parts)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        agent_statuses = {}
        for agent_type, agent in self.agents.items():
            agent_statuses[agent_type] = agent.get_agent_status()
        
        return {
            'system_status': self.system_status,
            'agents': agent_statuses,
            'total_executions': len(self.execution_history),
            'last_execution': self.execution_history[-1]['timestamp'] if self.execution_history else None,
            'configuration_valid': self.agent_factory.validate_configuration()['valid'],
            'crew_ready': self.crew is not None,
            'workflow_status': self.workflow_manager.get_status()
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        return self.execution_history[-limit:] if limit > 0 else self.execution_history
    
    def reset_system(self):
        """Reset the entire system."""
        try:
            logger.info("Resetting RAGHeat Portfolio System...")
            
            # Reset agents
            for agent in self.agents.values():
                agent.reset_history()
            
            # Reset execution history
            self.execution_history = []
            
            # Reinitialize workflow manager
            self.workflow_manager.reset()
            
            self.system_status = "ready"
            logger.info("System reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting system: {e}")
            self.system_status = "reset_error"
    
    def run_individual_analysis(self, 
                               agent_type: str, 
                               stocks: List[str], 
                               analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Run analysis using a specific agent."""
        try:
            if agent_type not in self.agents:
                return {'error': f'Agent type {agent_type} not available'}
            
            agent = self.agents[agent_type]
            
            input_data = {
                'stocks': stocks,
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat()
            }
            
            result = agent.analyze(input_data)
            
            return {
                'agent_type': agent_type,
                'analysis_result': result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error running individual analysis: {e}")
            return {
                'error': str(e),
                'agent_type': agent_type,
                'status': 'error'
            }
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents."""
        performance_summary = {}
        
        for agent_type, agent in self.agents.items():
            performance_summary[agent_type] = agent.get_performance_summary()
        
        return {
            'agent_performance': performance_summary,
            'system_executions': len(self.execution_history),
            'timestamp': datetime.now().isoformat()
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current system configuration."""
        return {
            'agents_config': self.agent_factory.agents_config,
            'tasks_config': self.task_orchestrator.tasks_config,
            'settings': {
                'max_iterations': settings.MAX_ITERATIONS,
                'verbose_logging': settings.VERBOSE_LOGGING,
                'enable_memory': settings.ENABLE_MEMORY,
                'risk_free_rate': settings.RISK_FREE_RATE
            },
            'export_timestamp': datetime.now().isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            'overall_health': 'healthy',
            'issues': [],
            'warnings': []
        }
        
        # Check system status
        if self.system_status != "ready":
            health_status['issues'].append(f"System status: {self.system_status}")
            health_status['overall_health'] = 'unhealthy'
        
        # Check agents
        if not self.agents:
            health_status['issues'].append("No agents initialized")
            health_status['overall_health'] = 'unhealthy'
        
        # Check crew
        if not self.crew:
            health_status['warnings'].append("Crew not initialized")
        
        # Check recent execution success rate
        if self.execution_history:
            recent_executions = self.execution_history[-10:]
            success_rate = sum(1 for ex in recent_executions if ex.get('success', False)) / len(recent_executions)
            
            if success_rate < 0.5:
                health_status['warnings'].append(f"Low success rate: {success_rate:.1%}")
        
        return health_status