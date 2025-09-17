"""
Tool Registry for Managing All Portfolio Construction Tools
"""

from typing import Dict, List, Any, Type
from crewai_tools import BaseTool
import importlib
import inspect
from loguru import logger

class ToolRegistry:
    """Registry for managing all tools used by portfolio agents."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_categories: Dict[str, List[str]] = {
            'fundamental': [],
            'sentiment': [],
            'valuation': [],
            'graph': [],
            'heat_diffusion': [],
            'portfolio': [],
            'explanation': []
        }
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize all tools from different modules."""
        try:
            # Import all tool modules
            tool_modules = [
                'fundamental_tools',
                'sentiment_tools', 
                'valuation_tools',
                'graph_tools',
                'heat_diffusion_tools',
                'portfolio_tools',
                'explanation_tools'
            ]
            
            for module_name in tool_modules:
                try:
                    module = importlib.import_module(f'.{module_name}', 'backend.portfolio_agents.tools')
                    self._register_tools_from_module(module, module_name.replace('_tools', ''))
                except ImportError as e:
                    logger.warning(f"Could not import {module_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error initializing tools: {e}")
    
    def _register_tools_from_module(self, module, category: str):
        """Register all tools from a module."""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseTool) and 
                obj != BaseTool):
                
                tool_instance = obj()
                tool_name = getattr(tool_instance, 'name', name.lower())
                self._tools[tool_name] = tool_instance
                self._tool_categories[category].append(tool_name)
                logger.info(f"Registered tool: {tool_name} in category: {category}")
    
    def get_tool(self, name: str) -> BaseTool:
        """Get a specific tool by name."""
        if name in self._tools:
            return self._tools[name]
        else:
            raise ValueError(f"Tool '{name}' not found in registry")
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a specific category."""
        if category in self._tool_categories:
            return [self._tools[name] for name in self._tool_categories[category]]
        else:
            raise ValueError(f"Category '{category}' not found")
    
    def get_agent_tools(self, agent_type: str) -> List[BaseTool]:
        """Get tools required for a specific agent type."""
        tool_mapping = {
            'fundamental_analyst': [
                'fundamental_report_pull',
                'financial_report_rag', 
                'sec_filing_analyzer',
                'financial_ratio_calculator'
            ],
            'sentiment_analyst': [
                'news_aggregator',
                'sentiment_analyzer',
                'social_media_monitor',
                'analyst_rating_tracker',
                'finbert_sentiment'
            ],
            'valuation_analyst': [
                'price_volume_analyzer',
                'technical_indicator_calculator',
                'volatility_calculator',
                'sharpe_ratio_calculator',
                'correlation_analyzer'
            ],
            'knowledge_graph_engineer': [
                'graph_constructor',
                'sparql_query_engine',
                'neo4j_interface',
                'ontology_mapper',
                'triple_extractor'
            ],
            'heat_diffusion_analyst': [
                'heat_equation_solver',
                'graph_laplacian_calculator',
                'diffusion_simulator',
                'influence_propagator',
                'heat_kernel_calculator'
            ],
            'portfolio_coordinator': [
                'consensus_builder',
                'debate_moderator',
                'portfolio_optimizer',
                'risk_assessor',
                'weight_allocator'
            ],
            'explanation_generator': [
                'chain_of_thought_generator',
                'visualization_creator',
                'report_generator',
                'langchain_rag',
                'gpt4_interface'
            ]
        }
        
        if agent_type not in tool_mapping:
            raise ValueError(f"Agent type '{agent_type}' not supported")
        
        tools = []
        for tool_name in tool_mapping[agent_type]:
            try:
                tools.append(self.get_tool(tool_name))
            except ValueError as e:
                logger.warning(f"Tool {tool_name} not available for agent {agent_type}: {e}")
        
        return tools
    
    def list_all_tools(self) -> Dict[str, List[str]]:
        """List all available tools by category."""
        return self._tool_categories.copy()
    
    def register_custom_tool(self, tool: BaseTool, category: str = 'custom'):
        """Register a custom tool."""
        tool_name = getattr(tool, 'name', tool.__class__.__name__.lower())
        self._tools[tool_name] = tool
        
        if category not in self._tool_categories:
            self._tool_categories[category] = []
        self._tool_categories[category].append(tool_name)
        
        logger.info(f"Registered custom tool: {tool_name} in category: {category}")

# Global tool registry instance
tool_registry = ToolRegistry()