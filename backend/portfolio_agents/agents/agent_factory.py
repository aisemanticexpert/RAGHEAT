"""
Agent Factory for Creating Portfolio Construction Agents
======================================================

Factory class for creating and configuring CrewAI agents based on YAML configuration.
"""

from typing import Dict, List, Any, Optional
import yaml
import os
from crewai import Agent
from loguru import logger
from ..tools.tool_registry import tool_registry
from ..config.settings import settings

class AgentFactory:
    """Factory for creating portfolio construction agents."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the agent factory."""
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), '..', 'config', 'agents.yaml'
        )
        self.agents_config = self._load_agents_config()
        self._created_agents: Dict[str, Agent] = {}
    
    def _load_agents_config(self) -> Dict[str, Any]:
        """Load agents configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config.get('agents', {})
        except FileNotFoundError:
            logger.error(f"Agent configuration file not found: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing agent configuration: {e}")
            return {}
    
    def create_agent(self, agent_type: str) -> Optional[Agent]:
        """Create a specific agent by type."""
        if agent_type in self._created_agents:
            return self._created_agents[agent_type]
        
        if agent_type not in self.agents_config:
            logger.error(f"Agent type '{agent_type}' not found in configuration")
            return None
        
        try:
            agent_config = self.agents_config[agent_type]
            
            # Get tools for this agent
            agent_tools = self._get_agent_tools(agent_type, agent_config.get('tools', []))
            
            # Create CrewAI agent
            agent = Agent(
                role=agent_config['role'],
                goal=agent_config['goal'],
                backstory=agent_config['backstory'],
                tools=agent_tools,
                verbose=agent_config.get('verbose', settings.VERBOSE_LOGGING),
                allow_delegation=agent_config.get('allow_delegation', False),
                max_iter=agent_config.get('max_iter', settings.MAX_ITERATIONS),
                memory=settings.ENABLE_MEMORY,
                llm=self._get_llm_for_agent(agent_type)
            )
            
            self._created_agents[agent_type] = agent
            logger.info(f"Created agent: {agent_type}")
            
            return agent
            
        except Exception as e:
            logger.error(f"Error creating agent {agent_type}: {e}")
            return None
    
    def create_all_agents(self) -> Dict[str, Agent]:
        """Create all agents defined in configuration."""
        agents = {}
        
        for agent_type in self.agents_config.keys():
            agent = self.create_agent(agent_type)
            if agent:
                agents[agent_type] = agent
        
        logger.info(f"Created {len(agents)} agents successfully")
        return agents
    
    def _get_agent_tools(self, agent_type: str, tool_names: List[str]) -> List:
        """Get tools for a specific agent."""
        agent_tools = []
        
        # Get tools from registry
        try:
            registry_tools = tool_registry.get_agent_tools(agent_type)
            agent_tools.extend(registry_tools)
        except ValueError as e:
            logger.warning(f"Could not get tools from registry for {agent_type}: {e}")
        
        # Get individual tools by name
        for tool_name in tool_names:
            try:
                tool = tool_registry.get_tool(tool_name)
                if tool not in agent_tools:  # Avoid duplicates
                    agent_tools.append(tool)
            except ValueError as e:
                logger.warning(f"Tool '{tool_name}' not found for agent {agent_type}: {e}")
        
        logger.info(f"Loaded {len(agent_tools)} tools for agent {agent_type}")
        return agent_tools
    
    def _get_llm_for_agent(self, agent_type: str) -> Optional[Any]:
        """Get LLM configuration for specific agent."""
        try:
            # Use Anthropic Claude for all agents
            from langchain_anthropic import ChatAnthropic
            
            llm = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                max_tokens=2000,
                temperature=0.1,  # Low temperature for consistent analysis
                timeout=settings.AGENT_TIMEOUT
            )
            
            return llm
            
        except ImportError:
            logger.warning("Anthropic LangChain integration not available, using default LLM")
            return None
        except Exception as e:
            logger.error(f"Error configuring LLM for {agent_type}: {e}")
            return None
    
    def get_agent_info(self, agent_type: str) -> Dict[str, Any]:
        """Get information about a specific agent type."""
        if agent_type not in self.agents_config:
            return {'error': f'Agent type {agent_type} not found'}
        
        config = self.agents_config[agent_type]
        return {
            'agent_type': agent_type,
            'role': config['role'],
            'goal': config['goal'],
            'backstory': config['backstory'],
            'tools': config.get('tools', []),
            'max_iterations': config.get('max_iter', settings.MAX_ITERATIONS),
            'allow_delegation': config.get('allow_delegation', False),
            'verbose': config.get('verbose', settings.VERBOSE_LOGGING)
        }
    
    def list_available_agents(self) -> List[str]:
        """List all available agent types."""
        return list(self.agents_config.keys())
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the agent configuration."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'agent_count': len(self.agents_config)
        }
        
        required_fields = ['role', 'goal', 'backstory']
        
        for agent_type, config in self.agents_config.items():
            # Check required fields
            for field in required_fields:
                if field not in config:
                    validation_results['errors'].append(
                        f"Agent '{agent_type}' missing required field: {field}"
                    )
                    validation_results['valid'] = False
            
            # Check tools existence
            tools = config.get('tools', [])
            for tool_name in tools:
                try:
                    tool_registry.get_tool(tool_name)
                except ValueError:
                    validation_results['warnings'].append(
                        f"Tool '{tool_name}' for agent '{agent_type}' not found in registry"
                    )
        
        return validation_results
    
    def get_agent_dependencies(self) -> Dict[str, List[str]]:
        """Get dependencies between agents (delegation relationships)."""
        dependencies = {}
        
        for agent_type, config in self.agents_config.items():
            dependencies[agent_type] = []
            
            # Agents that allow delegation might depend on others
            if config.get('allow_delegation', False):
                # In a real implementation, this would analyze the backstory and goals
                # to determine which agents this one might delegate to
                dependencies[agent_type] = ['portfolio_coordinator']
        
        return dependencies
    
    def create_agent_network_graph(self) -> Dict[str, Any]:
        """Create a network representation of agent relationships."""
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            
            # Add nodes (agents)
            for agent_type in self.agents_config.keys():
                G.add_node(agent_type, **self.get_agent_info(agent_type))
            
            # Add edges (delegation relationships)
            dependencies = self.get_agent_dependencies()
            for agent, deps in dependencies.items():
                for dep in deps:
                    if dep in self.agents_config:
                        G.add_edge(agent, dep, relationship='delegates_to')
            
            # Create serializable representation
            graph_data = {
                'nodes': [
                    {
                        'id': node,
                        'type': 'agent',
                        **G.nodes[node]
                    }
                    for node in G.nodes()
                ],
                'edges': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'type': G.edges[edge].get('relationship', 'unknown')
                    }
                    for edge in G.edges()
                ]
            }
            
            return graph_data
            
        except ImportError:
            return {'error': 'NetworkX not available for graph creation'}
        except Exception as e:
            logger.error(f"Error creating agent network graph: {e}")
            return {'error': str(e)}