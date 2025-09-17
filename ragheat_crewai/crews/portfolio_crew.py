"""
Portfolio Construction Crew for RAGHeat CrewAI System
=====================================================

Main orchestrator that coordinates all agents and tasks for portfolio construction.
"""

from typing import Dict, Any, List, Optional
from crewai import Crew, Task, Process
import yaml
import logging
from datetime import datetime
from pathlib import Path

from ..config.settings import settings
from ..agents import (
    FundamentalAnalystAgent,
    SentimentAnalystAgent,
    ValuationAnalystAgent,
    KnowledgeGraphEngineerAgent,
    HeatDiffusionAnalystAgent,
    PortfolioCoordinatorAgent,
    ExplanationGeneratorAgent
)
from ..tasks import (
    ConstructKnowledgeGraphTask,
    AnalyzeFundamentalsTask,
    AssessMarketSentimentTask,
    CalculateValuationsTask,
    SimulateHeatDiffusionTask,
    FacilitateAgentDebateTask,
    ConstructPortfolioTask,
    GenerateInvestmentRationaleTask
)

logger = logging.getLogger(__name__)

class PortfolioConstructionCrew:
    """
    Main crew orchestrator for portfolio construction using RAGHeat methodology.
    
    This crew coordinates multiple specialist agents through structured workflows
    to build optimal portfolios with full explainability and risk management.
    """
    
    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the portfolio construction crew.
        
        Args:
            custom_config: Optional custom configuration overrides
        """
        self.config = custom_config or {}
        self.agents = {}
        self.tasks = {}
        self.crew = None
        self.execution_history = []
        
        # Load configurations
        self._load_agent_configs()
        self._load_task_configs()
        
        # Initialize agents
        self._initialize_agents()
        
        # Create tasks
        self._create_tasks()
        
        # Build crew
        self._build_crew()
        
        logger.info("Portfolio Construction Crew initialized successfully")
    
    def _load_agent_configs(self):
        """Load agent configurations from YAML file."""
        try:
            with open(settings.agents_config_path, 'r') as f:
                self.agent_configs = yaml.safe_load(f)['agents']
            logger.info("Loaded agent configurations")
        except Exception as e:
            logger.error(f"Error loading agent configs: {e}")
            self.agent_configs = {}
    
    def _load_task_configs(self):
        """Load task configurations from YAML file."""
        try:
            with open(settings.tasks_config_path, 'r') as f:
                self.task_configs = yaml.safe_load(f)['tasks']
            logger.info("Loaded task configurations")
        except Exception as e:
            logger.error(f"Error loading task configs: {e}")
            self.task_configs = {}
    
    def _initialize_agents(self):
        """Initialize all specialist agents."""
        try:
            # Fundamental Analyst
            if 'fundamental_analyst' in self.agent_configs:
                self.agents['fundamental_analyst'] = FundamentalAnalystAgent(
                    self.agent_configs['fundamental_analyst']
                )
            
            # Sentiment Analyst
            if 'sentiment_analyst' in self.agent_configs:
                self.agents['sentiment_analyst'] = SentimentAnalystAgent(
                    self.agent_configs['sentiment_analyst']
                )
            
            # Valuation Analyst
            if 'valuation_analyst' in self.agent_configs:
                self.agents['valuation_analyst'] = ValuationAnalystAgent(
                    self.agent_configs['valuation_analyst']
                )
            
            # Knowledge Graph Engineer
            if 'knowledge_graph_engineer' in self.agent_configs:
                self.agents['knowledge_graph_engineer'] = KnowledgeGraphEngineerAgent(
                    self.agent_configs['knowledge_graph_engineer']
                )
            
            # Heat Diffusion Analyst
            if 'heat_diffusion_analyst' in self.agent_configs:
                self.agents['heat_diffusion_analyst'] = HeatDiffusionAnalystAgent(
                    self.agent_configs['heat_diffusion_analyst']
                )
            
            # Portfolio Coordinator
            if 'portfolio_coordinator' in self.agent_configs:
                self.agents['portfolio_coordinator'] = PortfolioCoordinatorAgent(
                    self.agent_configs['portfolio_coordinator']
                )
            
            # Explanation Generator
            if 'explanation_generator' in self.agent_configs:
                self.agents['explanation_generator'] = ExplanationGeneratorAgent(
                    self.agent_configs['explanation_generator']
                )
            
            logger.info(f"Initialized {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    def _create_tasks(self):
        """Create CrewAI tasks using task classes."""
        try:
            # Knowledge Graph Construction Task
            if 'knowledge_graph_engineer' in self.agents:
                self.tasks['construct_knowledge_graph'] = ConstructKnowledgeGraphTask.create(
                    agent=self.agents['knowledge_graph_engineer'].crew_agent
                )
            
            # Fundamental Analysis Task
            if 'fundamental_analyst' in self.agents:
                context_tasks = [self.tasks['construct_knowledge_graph']] if 'construct_knowledge_graph' in self.tasks else []
                self.tasks['analyze_fundamentals'] = AnalyzeFundamentalsTask.create(
                    agent=self.agents['fundamental_analyst'].crew_agent,
                    context_tasks=context_tasks
                )
            
            # Sentiment Analysis Task
            if 'sentiment_analyst' in self.agents:
                context_tasks = [self.tasks['construct_knowledge_graph']] if 'construct_knowledge_graph' in self.tasks else []
                self.tasks['assess_market_sentiment'] = AssessMarketSentimentTask.create(
                    agent=self.agents['sentiment_analyst'].crew_agent,
                    context_tasks=context_tasks
                )
            
            # Valuation Analysis Task
            if 'valuation_analyst' in self.agents:
                self.tasks['calculate_valuations'] = CalculateValuationsTask.create(
                    agent=self.agents['valuation_analyst'].crew_agent
                )
            
            # Heat Diffusion Task
            if 'heat_diffusion_analyst' in self.agents:
                context_tasks = []
                for task_name in ['construct_knowledge_graph', 'analyze_fundamentals', 'assess_market_sentiment']:
                    if task_name in self.tasks:
                        context_tasks.append(self.tasks[task_name])
                
                self.tasks['simulate_heat_diffusion'] = SimulateHeatDiffusionTask.create(
                    agent=self.agents['heat_diffusion_analyst'].crew_agent,
                    context_tasks=context_tasks
                )
            
            # Agent Debate Task
            if 'portfolio_coordinator' in self.agents:
                context_tasks = []
                for task_name in ['analyze_fundamentals', 'assess_market_sentiment', 'calculate_valuations', 'simulate_heat_diffusion']:
                    if task_name in self.tasks:
                        context_tasks.append(self.tasks[task_name])
                
                self.tasks['facilitate_agent_debate'] = FacilitateAgentDebateTask.create(
                    agent=self.agents['portfolio_coordinator'].crew_agent,
                    context_tasks=context_tasks
                )
            
            # Portfolio Construction Task
            if 'portfolio_coordinator' in self.agents:
                context_tasks = [self.tasks['facilitate_agent_debate']] if 'facilitate_agent_debate' in self.tasks else []
                self.tasks['construct_portfolio'] = ConstructPortfolioTask.create(
                    agent=self.agents['portfolio_coordinator'].crew_agent,
                    context_tasks=context_tasks
                )
            
            # Explanation Generation Task
            if 'explanation_generator' in self.agents:
                context_tasks = []
                for task_name in ['facilitate_agent_debate', 'construct_portfolio', 'simulate_heat_diffusion']:
                    if task_name in self.tasks:
                        context_tasks.append(self.tasks[task_name])
                
                self.tasks['generate_investment_rationale'] = GenerateInvestmentRationaleTask.create(
                    agent=self.agents['explanation_generator'].crew_agent,
                    context_tasks=context_tasks
                )
            
            logger.info(f"Created {len(self.tasks)} tasks")
            
        except Exception as e:
            logger.error(f"Error creating tasks: {e}")
            raise
    
    def _build_crew(self):
        """Build the CrewAI crew with all agents and tasks."""
        try:
            # Get all agent instances
            crew_agents = [agent.crew_agent for agent in self.agents.values()]
            
            # Get all task instances in execution order
            task_order = [
                'construct_knowledge_graph',
                'analyze_fundamentals',
                'assess_market_sentiment', 
                'calculate_valuations',
                'simulate_heat_diffusion',
                'facilitate_agent_debate',
                'construct_portfolio',
                'generate_investment_rationale'
            ]
            
            crew_tasks = []
            for task_name in task_order:
                if task_name in self.tasks:
                    crew_tasks.append(self.tasks[task_name])
            
            # Create the crew
            self.crew = Crew(
                agents=crew_agents,
                tasks=crew_tasks,
                process=Process.sequential,  # Sequential execution for dependencies
                verbose=True,
                memory=settings.enable_agent_memory,
                max_rpm=10  # Rate limiting
            )
            
            logger.info("Crew built successfully")
            
        except Exception as e:
            logger.error(f"Error building crew: {e}")
            raise
    
    def construct_portfolio(
        self,
        target_stocks: List[str],
        investment_objective: str = "balanced_growth",
        risk_tolerance: str = "moderate",
        time_horizon: str = "1y",
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete portfolio construction process.
        
        Args:
            target_stocks: List of stock tickers to analyze
            investment_objective: Investment objective (growth, income, balanced_growth)
            risk_tolerance: Risk tolerance (conservative, moderate, aggressive)
            time_horizon: Investment time horizon (3m, 6m, 1y, 2y, 5y)
            custom_constraints: Optional custom constraints
            
        Returns:
            Complete portfolio construction results
        """
        try:
            logger.info(f"Starting portfolio construction for {len(target_stocks)} stocks")
            
            # Prepare input data
            crew_inputs = {
                "target_stocks": target_stocks,
                "investment_objective": investment_objective,
                "risk_tolerance": risk_tolerance,
                "time_horizon": time_horizon,
                "constraints": custom_constraints or {},
                "execution_timestamp": datetime.now().isoformat()
            }
            
            # Execute the crew
            logger.info("Executing crew workflow...")
            result = self.crew.kickoff(inputs=crew_inputs)
            
            # Process and structure results
            processed_result = self._process_crew_results(result, crew_inputs)
            
            # Store execution history
            self.execution_history.append({
                "timestamp": datetime.now().isoformat(),
                "inputs": crew_inputs,
                "result_summary": str(processed_result)[:500] + "...",
                "success": True
            })
            
            logger.info("Portfolio construction completed successfully")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error in portfolio construction: {e}")
            
            # Store failed execution
            self.execution_history.append({
                "timestamp": datetime.now().isoformat(),
                "inputs": crew_inputs if 'crew_inputs' in locals() else {},
                "error": str(e),
                "success": False
            })
            
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "inputs": crew_inputs if 'crew_inputs' in locals() else {}
            }
    
    def _process_crew_results(self, raw_result: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure the crew execution results."""
        
        processed_result = {
            "execution_metadata": {
                "timestamp": datetime.now().isoformat(),
                "target_stocks": inputs["target_stocks"],
                "investment_objective": inputs["investment_objective"],
                "risk_tolerance": inputs["risk_tolerance"],
                "time_horizon": inputs["time_horizon"],
                "agents_executed": list(self.agents.keys()),
                "tasks_completed": list(self.tasks.keys())
            },
            "portfolio_recommendations": {},
            "agent_analyses": {},
            "risk_assessment": {},
            "explanation_package": {},
            "implementation_guidance": {},
            "monitoring_framework": {},
            "raw_result": str(raw_result)
        }
        
        # Extract specific results from each agent if available
        # This would be enhanced based on the actual result structure from CrewAI
        
        try:
            # Parse crew result if it's structured
            if hasattr(raw_result, 'tasks_output'):
                for task_output in raw_result.tasks_output:
                    task_name = getattr(task_output, 'task', 'unknown')
                    task_result = getattr(task_output, 'output', '')
                    processed_result["agent_analyses"][task_name] = task_result
            
            # Generate summary
            processed_result["execution_summary"] = self._generate_execution_summary(processed_result)
            
        except Exception as e:
            logger.warning(f"Error processing crew results: {e}")
            processed_result["processing_warning"] = str(e)
        
        return processed_result
    
    def _generate_execution_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of the execution results."""
        metadata = results["execution_metadata"]
        stocks_count = len(metadata["target_stocks"])
        agents_count = len(metadata["agents_executed"])
        
        summary = f"""
        Portfolio Construction Execution Summary:
        - Analyzed {stocks_count} target stocks
        - Executed {agents_count} specialist agents
        - Investment objective: {metadata["investment_objective"]}
        - Risk tolerance: {metadata["risk_tolerance"]}
        - Time horizon: {metadata["time_horizon"]}
        - Execution completed at: {metadata["timestamp"]}
        """
        
        return summary.strip()
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all agents."""
        performance = {}
        
        for agent_name, agent in self.agents.items():
            performance[agent_name] = agent.get_performance_summary()
        
        return {
            "individual_performance": performance,
            "crew_execution_history": self.execution_history,
            "overall_success_rate": self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        if not self.execution_history:
            return 0.0
        
        successful_executions = sum(1 for execution in self.execution_history if execution.get("success", False))
        return successful_executions / len(self.execution_history)
    
    def reset_crew_state(self):
        """Reset crew state and agent memories."""
        self.execution_history.clear()
        
        for agent in self.agents.values():
            agent.reset_memory()
        
        logger.info("Crew state reset successfully")
    
    def get_crew_status(self) -> Dict[str, Any]:
        """Get current crew status and configuration."""
        return {
            "crew_id": id(self.crew),
            "agents_count": len(self.agents),
            "tasks_count": len(self.tasks),
            "execution_count": len(self.execution_history),
            "success_rate": self._calculate_success_rate(),
            "agent_status": {name: agent.get_agent_status() for name, agent in self.agents.items()},
            "configuration": {
                "process_type": "sequential",
                "memory_enabled": settings.enable_agent_memory,
                "max_iterations": settings.max_iterations
            }
        }