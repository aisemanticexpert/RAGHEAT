"""
Task Orchestrator for Portfolio Construction
==========================================

Manages task creation, scheduling, and execution coordination based on tasks.yaml configuration.
"""

from typing import Dict, Any, List, Optional
import yaml
import os
from datetime import datetime
from loguru import logger
from crewai import Task, Agent

class TaskOrchestrator:
    """Orchestrates task creation and execution for portfolio construction."""
    
    def __init__(self, config_dir: str):
        """Initialize task orchestrator."""
        self.config_dir = config_dir
        self.tasks_config = self._load_tasks_config()
        self.created_tasks: Dict[str, Task] = {}
        self.task_dependencies = self._build_dependency_graph()
        
    def _load_tasks_config(self) -> Dict[str, Any]:
        """Load tasks configuration from YAML file."""
        tasks_path = os.path.join(self.config_dir, 'tasks.yaml')
        
        try:
            with open(tasks_path, 'r') as file:
                config = yaml.safe_load(file)
                return config.get('tasks', {})
        except FileNotFoundError:
            logger.error(f"Tasks configuration file not found: {tasks_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing tasks configuration: {e}")
            return {}
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build task dependency graph from configuration."""
        dependencies = {}
        
        for task_name, task_config in self.tasks_config.items():
            context_tasks = task_config.get('context', [])
            dependencies[task_name] = context_tasks
            
        return dependencies
    
    def create_task(self, task_name: str, agents: List[Agent]) -> Optional[Task]:
        """Create a specific task by name."""
        if task_name in self.created_tasks:
            return self.created_tasks[task_name]
        
        if task_name not in self.tasks_config:
            logger.error(f"Task '{task_name}' not found in configuration")
            return None
        
        try:
            task_config = self.tasks_config[task_name]
            
            # Find the appropriate agent
            agent_name = task_config.get('agent', '')
            assigned_agent = self._find_agent_by_name(agent_name, agents)
            
            if not assigned_agent:
                logger.error(f"No suitable agent found for task '{task_name}'")
                return None
            
            # Create task with enhanced description
            enhanced_description = self._enhance_task_description(task_config)
            
            task = Task(
                description=enhanced_description,
                agent=assigned_agent,
                expected_output=task_config.get('expected_output', ''),
                tools=self._get_task_tools(task_config),
                async_execution=False,  # Keep synchronous for now
                context=self._get_context_tasks(task_name)
            )
            
            self.created_tasks[task_name] = task
            logger.info(f"Created task: {task_name}")
            
            return task
            
        except Exception as e:
            logger.error(f"Error creating task {task_name}: {e}")
            return None
    
    def create_all_tasks(self, agents: List[Agent]) -> List[Task]:
        """Create all tasks defined in configuration."""
        created_tasks = []
        
        # Create tasks in dependency order
        task_order = self._get_task_execution_order()
        
        for task_name in task_order:
            task = self.create_task(task_name, agents)
            if task:
                created_tasks.append(task)
        
        logger.info(f"Created {len(created_tasks)} tasks successfully")
        return created_tasks
    
    def _find_agent_by_name(self, agent_name: str, agents: List[Agent]) -> Optional[Agent]:
        """Find agent by name from available agents."""
        for agent in agents:
            # Check if agent role matches or contains the agent name
            if (agent_name.lower() in agent.role.lower() or 
                any(keyword in agent.role.lower() for keyword in agent_name.split('_'))):
                return agent
        
        # Fallback: return first agent if no specific match
        if agents:
            logger.warning(f"No specific agent found for '{agent_name}', using first available agent")
            return agents[0]
        
        return None
    
    def _enhance_task_description(self, task_config: Dict[str, Any]) -> str:
        """Enhance task description with additional context and formatting."""
        base_description = task_config.get('description', '')
        expected_output = task_config.get('expected_output', '')
        tools = task_config.get('tools', [])
        
        enhanced_description = f"""
        TASK OBJECTIVE:
        {base_description}
        
        EXPECTED OUTPUT:
        {expected_output}
        
        AVAILABLE TOOLS:
        {', '.join(tools) if tools else 'Standard analysis tools'}
        
        INSTRUCTIONS:
        1. Use your specialized expertise to analyze the provided data
        2. Apply relevant tools and methodologies for thorough analysis
        3. Provide specific, actionable insights and recommendations
        4. Include confidence levels and supporting evidence
        5. Structure your output according to the expected format
        6. Consider risk factors and limitations in your analysis
        
        Please ensure your analysis is comprehensive and well-reasoned.
        """
        
        return enhanced_description.strip()
    
    def _get_task_tools(self, task_config: Dict[str, Any]) -> List[str]:
        """Get tools specified for the task."""
        return task_config.get('tools', [])
    
    def _get_context_tasks(self, task_name: str) -> List[Task]:
        """Get context tasks that this task depends on."""
        context_task_names = self.task_dependencies.get(task_name, [])
        context_tasks = []
        
        for context_name in context_task_names:
            if context_name in self.created_tasks:
                context_tasks.append(self.created_tasks[context_name])
        
        return context_tasks
    
    def _get_task_execution_order(self) -> List[str]:
        """Determine optimal task execution order based on dependencies."""
        # Topological sort of task dependencies
        visited = set()
        temp_visited = set()
        result = []
        
        def dfs(task_name: str):
            if task_name in temp_visited:
                # Circular dependency detected, skip
                logger.warning(f"Circular dependency detected involving task: {task_name}")
                return
            
            if task_name in visited:
                return
            
            temp_visited.add(task_name)
            
            # Visit dependencies first
            for dependency in self.task_dependencies.get(task_name, []):
                if dependency in self.tasks_config:
                    dfs(dependency)
            
            temp_visited.remove(task_name)
            visited.add(task_name)
            result.append(task_name)
        
        # Process all tasks
        for task_name in self.tasks_config.keys():
            if task_name not in visited:
                dfs(task_name)
        
        return result
    
    def get_task_status(self, task_name: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        if task_name not in self.tasks_config:
            return {'error': f'Task {task_name} not found'}
        
        task_config = self.tasks_config[task_name]
        
        status = {
            'task_name': task_name,
            'agent': task_config.get('agent', ''),
            'dependencies': self.task_dependencies.get(task_name, []),
            'tools': task_config.get('tools', []),
            'created': task_name in self.created_tasks,
            'description_length': len(task_config.get('description', '')),
            'expected_output_defined': bool(task_config.get('expected_output', ''))
        }
        
        return status
    
    def get_all_tasks_status(self) -> Dict[str, Any]:
        """Get status of all tasks."""
        tasks_status = {}
        
        for task_name in self.tasks_config.keys():
            tasks_status[task_name] = self.get_task_status(task_name)
        
        return {
            'tasks': tasks_status,
            'total_tasks': len(self.tasks_config),
            'created_tasks': len(self.created_tasks),
            'execution_order': self._get_task_execution_order()
        }
    
    def validate_tasks_configuration(self) -> Dict[str, Any]:
        """Validate tasks configuration."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'task_count': len(self.tasks_config)
        }
        
        required_fields = ['description', 'agent', 'expected_output']
        
        for task_name, task_config in self.tasks_config.items():
            # Check required fields
            for field in required_fields:
                if field not in task_config or not task_config[field]:
                    validation_results['errors'].append(
                        f"Task '{task_name}' missing or empty required field: {field}"
                    )
                    validation_results['valid'] = False
            
            # Check for circular dependencies
            if self._has_circular_dependency(task_name):
                validation_results['warnings'].append(
                    f"Task '{task_name}' may have circular dependencies"
                )
            
            # Check if context tasks exist
            context_tasks = task_config.get('context', [])
            for context_task in context_tasks:
                if context_task not in self.tasks_config:
                    validation_results['warnings'].append(
                        f"Task '{task_name}' references non-existent context task: '{context_task}'"
                    )
        
        return validation_results
    
    def _has_circular_dependency(self, task_name: str, visited: Optional[set] = None) -> bool:
        """Check if a task has circular dependencies."""
        if visited is None:
            visited = set()
        
        if task_name in visited:
            return True
        
        visited.add(task_name)
        
        dependencies = self.task_dependencies.get(task_name, [])
        for dependency in dependencies:
            if self._has_circular_dependency(dependency, visited.copy()):
                return True
        
        return False
    
    def create_task_execution_plan(self) -> Dict[str, Any]:
        """Create a detailed execution plan for all tasks."""
        execution_order = self._get_task_execution_order()
        
        execution_plan = {
            'total_tasks': len(execution_order),
            'execution_phases': [],
            'estimated_duration': self._estimate_execution_duration(),
            'critical_path': self._identify_critical_path(),
            'parallel_opportunities': self._identify_parallel_tasks()
        }
        
        # Group tasks into phases based on dependencies
        phases = []
        remaining_tasks = execution_order.copy()
        
        while remaining_tasks:
            current_phase = []
            ready_tasks = []
            
            for task in remaining_tasks:
                dependencies = self.task_dependencies.get(task, [])
                if all(dep not in remaining_tasks for dep in dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Break circular dependencies
                ready_tasks = [remaining_tasks[0]]
            
            current_phase.extend(ready_tasks)
            phases.append(current_phase)
            
            for task in ready_tasks:
                remaining_tasks.remove(task)
        
        # Format phases
        for i, phase in enumerate(phases):
            phase_info = {
                'phase_number': i + 1,
                'tasks': phase,
                'can_execute_parallel': len(phase) > 1,
                'estimated_time': max(self._estimate_task_duration(task) for task in phase)
            }
            execution_plan['execution_phases'].append(phase_info)
        
        return execution_plan
    
    def _estimate_execution_duration(self) -> float:
        """Estimate total execution duration in minutes."""
        # Simple estimation based on task complexity
        base_time_per_task = 2.0  # 2 minutes per task
        return len(self.tasks_config) * base_time_per_task
    
    def _estimate_task_duration(self, task_name: str) -> float:
        """Estimate duration for a specific task."""
        if task_name not in self.tasks_config:
            return 2.0
        
        task_config = self.tasks_config[task_name]
        
        # Estimate based on complexity factors
        base_time = 1.0
        
        # More tools = more complex
        tools_factor = len(task_config.get('tools', [])) * 0.5
        
        # Dependencies add complexity
        deps_factor = len(self.task_dependencies.get(task_name, [])) * 0.3
        
        # Description length as complexity indicator
        desc_factor = len(task_config.get('description', '')) / 1000
        
        estimated_time = base_time + tools_factor + deps_factor + desc_factor
        return min(estimated_time, 10.0)  # Cap at 10 minutes
    
    def _identify_critical_path(self) -> List[str]:
        """Identify the critical path of task execution."""
        # Simple critical path: tasks with most dependencies
        tasks_with_deps = [
            (task, len(self.task_dependencies.get(task, [])))
            for task in self.tasks_config.keys()
        ]
        
        # Sort by dependency count (descending)
        tasks_with_deps.sort(key=lambda x: x[1], reverse=True)
        
        return [task for task, _ in tasks_with_deps[:3]]  # Top 3 critical tasks
    
    def _identify_parallel_tasks(self) -> List[List[str]]:
        """Identify tasks that can be executed in parallel."""
        parallel_groups = []
        execution_order = self._get_task_execution_order()
        
        # Find tasks with no interdependencies
        independent_tasks = []
        for task in execution_order:
            deps = self.task_dependencies.get(task, [])
            if not deps:
                independent_tasks.append(task)
        
        if len(independent_tasks) > 1:
            parallel_groups.append(independent_tasks)
        
        return parallel_groups
    
    def reset(self):
        """Reset the task orchestrator."""
        self.created_tasks = {}
        logger.info("Task orchestrator reset completed")