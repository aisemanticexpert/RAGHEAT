"""
Base Agent for RAGHeat CrewAI System
====================================

This module provides the base agent class that all specialized agents inherit from.
It includes common functionality for agent initialization, tool management, and interaction.
"""

from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
from crewai import Agent
from crewai.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from ..config.settings import settings
from ..tools import get_tools_for_agent
import logging

logger = logging.getLogger(__name__)

class RAGHeatBaseAgent(ABC):
    """
    Base class for all RAGHeat agents providing common functionality.
    
    This class handles:
    - Agent initialization with proper LLM configuration
    - Tool loading and management
    - Common agent behaviors and utilities
    - Standardized logging and error handling
    """
    
    def __init__(
        self,
        agent_config: Dict[str, Any],
        custom_tools: Optional[List[BaseTool]] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_config: Agent configuration from YAML file
            custom_tools: Optional list of custom tools to add
            llm_config: Optional LLM configuration override
        """
        self.agent_config = agent_config
        self.agent_name = agent_config.get("role", "Unknown Agent")
        
        # Initialize LLM
        self.llm = self._initialize_llm(llm_config)
        
        # Load tools
        self.tools = self._load_tools(custom_tools)
        
        # Create CrewAI agent
        self.crew_agent = self._create_crew_agent()
        
        # Initialize agent state
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        self.memory: Dict[str, Any] = {}
        
        logger.info(f"Initialized {self.agent_name} with {len(self.tools)} tools")
    
    def _initialize_llm(self, llm_config: Optional[Dict[str, Any]] = None) -> ChatAnthropic:
        """Initialize the LLM for this agent."""
        config = llm_config or settings.anthropic_config
        
        return ChatAnthropic(
            api_key=config["api_key"],
            model=config["model"],
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.1)
        )
    
    def _load_tools(self, custom_tools: Optional[List[BaseTool]] = None) -> List[BaseTool]:
        """Load tools for this agent based on configuration."""
        # Get tools from configuration
        tool_names = self.agent_config.get("tools", [])
        tools = get_tools_for_agent(self.agent_name.lower().replace(" ", "_"), tool_names)
        
        # Add custom tools if provided
        if custom_tools:
            tools.extend(custom_tools)
        
        return tools
    
    def _create_crew_agent(self) -> Agent:
        """Create the CrewAI agent instance."""
        return Agent(
            role=self.agent_config["role"],
            goal=self.agent_config["goal"],
            backstory=self.agent_config["backstory"],
            tools=self.tools,
            llm=self.llm,
            verbose=self.agent_config.get("verbose", True),
            allow_delegation=self.agent_config.get("allow_delegation", False),
            max_iter=self.agent_config.get("max_iter", 5),
            memory=settings.enable_agent_memory
        )
    
    @abstractmethod
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform the agent's specialized analysis.
        
        This method must be implemented by each specialized agent.
        
        Args:
            input_data: Input data for analysis
            
        Returns:
            Analysis results
        """
        pass
    
    def execute_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a task using the CrewAI agent.
        
        Args:
            task_description: Description of the task to execute
            context: Optional context data
            
        Returns:
            Task execution results
        """
        try:
            logger.info(f"{self.agent_name} executing task: {task_description[:100]}...")
            
            # Prepare context
            formatted_context = self._format_context(context or {})
            
            # Create full prompt
            full_prompt = self._create_prompt(task_description, formatted_context)
            
            # Execute task
            result = self.crew_agent.execute_task(full_prompt)
            
            # Process and store result
            processed_result = self._process_result(result, task_description)
            
            # Update execution history
            self._update_execution_history(task_description, processed_result, True)
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Error executing task for {self.agent_name}: {e}")
            error_result = {"error": str(e), "agent": self.agent_name}
            self._update_execution_history(task_description, error_result, False)
            return error_result
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context data for agent consumption."""
        if not context:
            return "No additional context provided."
        
        formatted_parts = []
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                formatted_parts.append(f"{key}: {self._summarize_complex_data(value)}")
            else:
                formatted_parts.append(f"{key}: {value}")
        
        return "\n".join(formatted_parts)
    
    def _summarize_complex_data(self, data: Any) -> str:
        """Summarize complex data structures."""
        if isinstance(data, dict):
            return f"Dictionary with {len(data)} keys"
        elif isinstance(data, list):
            return f"List with {len(data)} items"
        else:
            return str(data)[:100] + ("..." if len(str(data)) > 100 else "")
    
    def _create_prompt(self, task_description: str, context: str) -> str:
        """Create a comprehensive prompt for the agent."""
        return f"""
        ROLE: {self.agent_config['role']}
        
        GOAL: {self.agent_config['goal']}
        
        TASK: {task_description}
        
        CONTEXT:
        {context}
        
        INSTRUCTIONS:
        1. Apply your specialized expertise to analyze the provided information
        2. Use available tools to gather additional data if needed
        3. Provide specific, actionable insights and recommendations
        4. Include confidence levels and supporting evidence
        5. Identify any limitations or assumptions in your analysis
        6. Structure your response clearly with key findings highlighted
        
        Please provide a thorough analysis with clear reasoning and actionable conclusions.
        """
    
    def _process_result(self, result: Any, task_description: str) -> Dict[str, Any]:
        """Process and structure the agent's result."""
        return {
            "agent": self.agent_name,
            "task": task_description,
            "result": result,
            "timestamp": self._get_current_timestamp(),
            "tools_used": [tool.__class__.__name__ for tool in self.tools],
            "confidence": self._extract_confidence(result)
        }
    
    def _extract_confidence(self, result: Any) -> float:
        """Extract confidence level from result."""
        # Simple confidence extraction - can be enhanced
        result_str = str(result).lower()
        if "very confident" in result_str:
            return 0.9
        elif "confident" in result_str:
            return 0.8
        elif "uncertain" in result_str:
            return 0.4
        else:
            return 0.6
    
    def _update_execution_history(self, task: str, result: Dict[str, Any], success: bool):
        """Update the agent's execution history."""
        self.execution_history.append({
            "timestamp": self._get_current_timestamp(),
            "task": task,
            "success": success,
            "result_summary": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
        })
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this agent."""
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        successful_tasks = [h for h in self.execution_history if h["success"]]
        
        return {
            "agent": self.agent_name,
            "total_tasks": len(self.execution_history),
            "successful_tasks": len(successful_tasks),
            "success_rate": len(successful_tasks) / len(self.execution_history),
            "last_execution": self.execution_history[-1]["timestamp"]
        }
    
    def update_memory(self, key: str, value: Any):
        """Update agent memory."""
        self.memory[key] = value
        logger.debug(f"{self.agent_name} updated memory key: {key}")
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Get value from agent memory."""
        return self.memory.get(key, default)
    
    def reset_memory(self):
        """Reset agent memory."""
        self.memory.clear()
        logger.info(f"{self.agent_name} memory reset")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "name": self.agent_name,
            "role": self.agent_config["role"],
            "status": "active",
            "tools_count": len(self.tools),
            "memory_items": len(self.memory),
            "execution_history_count": len(self.execution_history),
            "performance_summary": self.get_performance_summary()
        }