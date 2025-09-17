"""
Tool Registry for RAGHeat CrewAI System
=======================================

Central registry for managing tools and their assignment to agents.
"""

from typing import Dict, List, Type, Optional
from crewai.tools import BaseTool
import logging

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Central registry for managing CrewAI tools."""
    
    def __init__(self):
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._agent_tool_mappings: Dict[str, List[str]] = {}
        self._initialize_default_mappings()
    
    def _initialize_default_mappings(self):
        """Initialize default tool mappings for agents."""
        self._agent_tool_mappings = {
            "fundamental_analyst": [
                "fundamental_report_pull",
                "financial_report_rag", 
                "sec_filing_analyzer",
                "financial_ratio_calculator"
            ],
            "sentiment_analyst": [
                "news_aggregator",
                "sentiment_analyzer",
                "social_media_monitor",
                "analyst_rating_tracker",
                "finbert_sentiment"
            ],
            "valuation_analyst": [
                "price_volume_analyzer",
                "technical_indicator_calculator",
                "volatility_calculator",
                "sharpe_ratio_calculator",
                "correlation_analyzer"
            ],
            "knowledge_graph_engineer": [
                "graph_constructor",
                "sparql_query_engine",
                "neo4j_interface",
                "ontology_mapper",
                "triple_extractor"
            ],
            "heat_diffusion_analyst": [
                "heat_equation_solver",
                "graph_laplacian_calculator",
                "diffusion_simulator",
                "influence_propagator",
                "heat_kernel_calculator"
            ],
            "portfolio_coordinator": [
                "consensus_builder",
                "debate_moderator",
                "portfolio_optimizer",
                "risk_assessor",
                "weight_allocator"
            ],
            "explanation_generator": [
                "chain_of_thought_generator",
                "visualization_creator",
                "report_generator",
                "langchain_rag",
                "gpt4_interface"
            ]
        }
    
    def register_tool(self, name: str, tool_class: Type[BaseTool]):
        """Register a tool in the registry."""
        self._tools[name] = tool_class
        logger.debug(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[Type[BaseTool]]:
        """Get a tool class by name."""
        return self._tools.get(name)
    
    def get_tools_for_agent(self, agent_name: str, tool_names: List[str] = None) -> List[BaseTool]:
        """
        Get instantiated tools for a specific agent.
        
        Args:
            agent_name: Name of the agent
            tool_names: Optional list of specific tool names
            
        Returns:
            List of instantiated tool objects
        """
        if tool_names:
            # Use provided tool names
            requested_tools = tool_names
        else:
            # Use default mapping for agent
            requested_tools = self._agent_tool_mappings.get(agent_name, [])
        
        tools = []
        for tool_name in requested_tools:
            tool_class = self._tools.get(tool_name)
            if tool_class:
                try:
                    tool_instance = tool_class()
                    tools.append(tool_instance)
                except Exception as e:
                    logger.warning(f"Failed to instantiate tool {tool_name}: {e}")
            else:
                logger.warning(f"Tool {tool_name} not found in registry")
        
        logger.info(f"Loaded {len(tools)} tools for agent {agent_name}")
        return tools
    
    def get_all_tools(self) -> Dict[str, Type[BaseTool]]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def list_agent_tools(self, agent_name: str) -> List[str]:
        """List tool names for a specific agent."""
        return self._agent_tool_mappings.get(agent_name, [])
    
    def add_tool_to_agent(self, agent_name: str, tool_name: str):
        """Add a tool to an agent's tool list."""
        if agent_name not in self._agent_tool_mappings:
            self._agent_tool_mappings[agent_name] = []
        
        if tool_name not in self._agent_tool_mappings[agent_name]:
            self._agent_tool_mappings[agent_name].append(tool_name)
            logger.info(f"Added tool {tool_name} to agent {agent_name}")
    
    def remove_tool_from_agent(self, agent_name: str, tool_name: str):
        """Remove a tool from an agent's tool list."""
        if agent_name in self._agent_tool_mappings:
            if tool_name in self._agent_tool_mappings[agent_name]:
                self._agent_tool_mappings[agent_name].remove(tool_name)
                logger.info(f"Removed tool {tool_name} from agent {agent_name}")

# Global registry instance
_registry = ToolRegistry()

def register_tool(name: str, tool_class: Type[BaseTool]):
    """Register a tool globally."""
    _registry.register_tool(name, tool_class)

def get_tools_for_agent(agent_name: str, tool_names: List[str] = None) -> List[BaseTool]:
    """Get tools for an agent globally."""
    return _registry.get_tools_for_agent(agent_name, tool_names)

def get_all_tools() -> Dict[str, Type[BaseTool]]:
    """Get all tools globally."""
    return _registry.get_all_tools()

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _registry