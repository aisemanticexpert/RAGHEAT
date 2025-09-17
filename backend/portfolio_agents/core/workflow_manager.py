"""
Workflow Manager for Portfolio Construction
==========================================

Manages high-level workflows and orchestrates agent interactions.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

class WorkflowStatus(Enum):
    """Workflow execution status."""
    INITIALIZED = "initialized"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class WorkflowStep:
    """Individual workflow step."""
    
    def __init__(self, 
                 step_id: str,
                 description: str, 
                 agent_type: str,
                 function: Callable,
                 dependencies: List[str] = None,
                 timeout: int = 300):
        self.step_id = step_id
        self.description = description
        self.agent_type = agent_type
        self.function = function
        self.dependencies = dependencies or []
        self.timeout = timeout
        self.status = WorkflowStatus.INITIALIZED
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None

class WorkflowManager:
    """Manages portfolio construction workflows."""
    
    def __init__(self):
        """Initialize workflow manager."""
        self.workflows = {}
        self.current_workflow = None
        self.execution_history = []
        self.status = WorkflowStatus.INITIALIZED
        
    def initialize(self):
        """Initialize workflow manager."""
        self._register_default_workflows()
        self.status = WorkflowStatus.INITIALIZED
        logger.info("Workflow manager initialized")
    
    def _register_default_workflows(self):
        """Register default workflow templates."""
        # Portfolio Construction Workflow
        self.workflows['portfolio_construction'] = {
            'description': 'Complete portfolio construction using multi-agent analysis',
            'steps': [
                'construct_knowledge_graph',
                'analyze_fundamentals', 
                'assess_market_sentiment',
                'calculate_valuations',
                'simulate_heat_diffusion',
                'facilitate_agent_debate',
                'construct_portfolio',
                'generate_investment_rationale'
            ],
            'parallel_steps': [
                ['analyze_fundamentals', 'assess_market_sentiment', 'calculate_valuations'],
                ['facilitate_agent_debate', 'simulate_heat_diffusion']
            ],
            'timeout': 1800  # 30 minutes
        }
        
        # Quick Analysis Workflow
        self.workflows['quick_analysis'] = {
            'description': 'Quick analysis for immediate insights',
            'steps': [
                'analyze_fundamentals',
                'assess_market_sentiment', 
                'generate_investment_rationale'
            ],
            'parallel_steps': [
                ['analyze_fundamentals', 'assess_market_sentiment']
            ],
            'timeout': 600  # 10 minutes
        }
        
        # Risk Assessment Workflow
        self.workflows['risk_assessment'] = {
            'description': 'Comprehensive risk assessment and management',
            'steps': [
                'construct_knowledge_graph',
                'simulate_heat_diffusion',
                'assess_portfolio_risks',
                'generate_risk_report'
            ],
            'parallel_steps': [],
            'timeout': 900  # 15 minutes
        }
    
    def execute_workflow(self, 
                        workflow_type: str,
                        input_data: Dict[str, Any],
                        agents: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific workflow."""
        try:
            if workflow_type not in self.workflows:
                return {
                    'status': 'error',
                    'error': f'Workflow type {workflow_type} not found'
                }
            
            logger.info(f"Starting workflow: {workflow_type}")
            self.status = WorkflowStatus.RUNNING
            self.current_workflow = workflow_type
            
            workflow_config = self.workflows[workflow_type]
            execution_start = datetime.now()
            
            # Execute workflow steps
            workflow_results = self._execute_workflow_steps(
                workflow_config, input_data, agents
            )
            
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            # Determine overall status
            overall_status = 'success' if all(
                step_result.get('status') == 'success' 
                for step_result in workflow_results.values()
            ) else 'partial_failure'
            
            result = {
                'workflow_type': workflow_type,
                'status': overall_status,
                'execution_time': execution_time,
                'step_results': workflow_results,
                'summary': self._generate_workflow_summary(workflow_results),
                'timestamp': datetime.now().isoformat()
            }
            
            # Record execution
            self.execution_history.append(result)
            self.status = WorkflowStatus.COMPLETED
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_type}: {e}")
            self.status = WorkflowStatus.FAILED
            
            return {
                'workflow_type': workflow_type,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_workflow_steps(self,
                               workflow_config: Dict[str, Any],
                               input_data: Dict[str, Any], 
                               agents: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual workflow steps."""
        steps = workflow_config['steps']
        parallel_steps = workflow_config.get('parallel_steps', [])
        results = {}
        
        # Track completed steps
        completed_steps = set()
        
        # Process parallel steps first
        for parallel_group in parallel_steps:
            parallel_results = self._execute_parallel_steps(
                parallel_group, input_data, agents, completed_steps
            )
            results.update(parallel_results)
            completed_steps.update(parallel_group)
        
        # Process remaining steps sequentially
        remaining_steps = [step for step in steps if step not in completed_steps]
        
        for step in remaining_steps:
            step_result = self._execute_single_step(step, input_data, agents, results)
            results[step] = step_result
            
            # Update input data with step result for next steps
            if step_result.get('status') == 'success':
                input_data[f'{step}_result'] = step_result.get('result', {})
        
        return results
    
    def _execute_parallel_steps(self,
                               step_group: List[str],
                               input_data: Dict[str, Any],
                               agents: Dict[str, Any],
                               completed_steps: set) -> Dict[str, Any]:
        """Execute a group of steps in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(step_group)) as executor:
            # Submit all steps
            future_to_step = {
                executor.submit(
                    self._execute_single_step, 
                    step, input_data, agents, {}
                ): step
                for step in step_group
            }
            
            # Collect results
            for future in as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    step_result = future.result(timeout=300)  # 5 minute timeout per step
                    results[step] = step_result
                    
                    # Update input data with step result
                    if step_result.get('status') == 'success':
                        input_data[f'{step}_result'] = step_result.get('result', {})
                        
                except Exception as e:
                    logger.error(f"Error in parallel step {step}: {e}")
                    results[step] = {
                        'status': 'error',
                        'error': str(e),
                        'step': step
                    }
        
        return results
    
    def _execute_single_step(self,
                            step: str,
                            input_data: Dict[str, Any],
                            agents: Dict[str, Any],
                            previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        try:
            start_time = datetime.now()
            
            # Map step to agent and method
            step_mapping = self._get_step_agent_mapping()
            
            if step not in step_mapping:
                return {
                    'status': 'error',
                    'error': f'Step {step} not implemented',
                    'step': step
                }
            
            agent_type, method_name = step_mapping[step]
            
            if agent_type not in agents:
                return {
                    'status': 'error', 
                    'error': f'Agent {agent_type} not available',
                    'step': step
                }
            
            agent = agents[agent_type]
            
            # Prepare step input
            step_input = {
                'input_data': input_data,
                'previous_results': previous_results,
                'step_name': step
            }
            
            # Execute step
            if hasattr(agent, method_name):
                method = getattr(agent, method_name)
                result = method(step_input)
            else:
                # Fallback to generic analyze method
                result = agent.analyze(step_input)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'status': 'success',
                'result': result,
                'execution_time': execution_time,
                'step': step,
                'agent_type': agent_type
            }
            
        except Exception as e:
            logger.error(f"Error executing step {step}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'step': step,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _get_step_agent_mapping(self) -> Dict[str, tuple]:
        """Map workflow steps to agents and methods."""
        return {
            'construct_knowledge_graph': ('knowledge_graph_engineer', 'construct_graph'),
            'analyze_fundamentals': ('fundamental_analyst', 'analyze'),
            'assess_market_sentiment': ('sentiment_analyst', 'analyze'),
            'calculate_valuations': ('valuation_analyst', 'analyze'),
            'simulate_heat_diffusion': ('heat_diffusion_analyst', 'simulate_diffusion'),
            'facilitate_agent_debate': ('portfolio_coordinator', 'facilitate_debate'),
            'construct_portfolio': ('portfolio_coordinator', 'construct_portfolio'),
            'generate_investment_rationale': ('explanation_generator', 'generate_explanation'),
            'assess_portfolio_risks': ('portfolio_coordinator', 'assess_risks'),
            'generate_risk_report': ('explanation_generator', 'generate_risk_report')
        }
    
    def _generate_workflow_summary(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of workflow execution."""
        total_steps = len(step_results)
        successful_steps = sum(1 for result in step_results.values() 
                              if result.get('status') == 'success')
        failed_steps = total_steps - successful_steps
        
        total_execution_time = sum(
            result.get('execution_time', 0) 
            for result in step_results.values()
        )
        
        # Extract key insights
        key_insights = []
        recommendations = []
        
        for step_name, result in step_results.values():
            if result.get('status') == 'success' and 'result' in result:
                step_result = result['result']
                
                # Extract insights
                if isinstance(step_result, dict):
                    insights = step_result.get('key_insights', [])
                    if insights:
                        key_insights.extend(insights[:2])  # Limit per step
                    
                    recs = step_result.get('recommendations', [])
                    if recs:
                        recommendations.extend(recs[:2])  # Limit per step
        
        return {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'success_rate': successful_steps / total_steps if total_steps > 0 else 0,
            'total_execution_time': total_execution_time,
            'key_insights': key_insights[:5],  # Top 5 insights
            'recommendations': recommendations[:5]  # Top 5 recommendations
        }
    
    def get_workflow_status(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of current or specific workflow."""
        return {
            'manager_status': self.status.value,
            'current_workflow': self.current_workflow,
            'available_workflows': list(self.workflows.keys()),
            'execution_history_count': len(self.execution_history),
            'last_execution': self.execution_history[-1]['timestamp'] if self.execution_history else None
        }
    
    def get_workflow_templates(self) -> Dict[str, Any]:
        """Get available workflow templates."""
        templates = {}
        
        for workflow_name, config in self.workflows.items():
            templates[workflow_name] = {
                'description': config['description'],
                'steps': config['steps'],
                'parallel_steps': config.get('parallel_steps', []),
                'estimated_duration': config.get('timeout', 1800) / 60,  # Convert to minutes
                'step_count': len(config['steps'])
            }
        
        return templates
    
    def create_custom_workflow(self,
                              workflow_name: str,
                              description: str,
                              steps: List[str],
                              parallel_steps: List[List[str]] = None,
                              timeout: int = 1800) -> Dict[str, Any]:
        """Create a custom workflow template."""
        try:
            # Validate steps
            valid_steps = set(self._get_step_agent_mapping().keys())
            invalid_steps = [step for step in steps if step not in valid_steps]
            
            if invalid_steps:
                return {
                    'status': 'error',
                    'error': f'Invalid steps: {invalid_steps}',
                    'valid_steps': list(valid_steps)
                }
            
            # Create workflow
            self.workflows[workflow_name] = {
                'description': description,
                'steps': steps,
                'parallel_steps': parallel_steps or [],
                'timeout': timeout,
                'custom': True,
                'created': datetime.now().isoformat()
            }
            
            logger.info(f"Created custom workflow: {workflow_name}")
            
            return {
                'status': 'success',
                'workflow_name': workflow_name,
                'message': 'Custom workflow created successfully'
            }
            
        except Exception as e:
            logger.error(f"Error creating custom workflow: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def pause_workflow(self):
        """Pause current workflow execution."""
        if self.status == WorkflowStatus.RUNNING:
            self.status = WorkflowStatus.PAUSED
            logger.info("Workflow paused")
    
    def resume_workflow(self):
        """Resume paused workflow execution."""
        if self.status == WorkflowStatus.PAUSED:
            self.status = WorkflowStatus.RUNNING
            logger.info("Workflow resumed")
    
    def cancel_workflow(self):
        """Cancel current workflow execution."""
        self.status = WorkflowStatus.FAILED
        self.current_workflow = None
        logger.info("Workflow cancelled")
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get workflow execution history."""
        return self.execution_history[-limit:] if limit > 0 else self.execution_history
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get workflow performance metrics."""
        if not self.execution_history:
            return {'message': 'No execution history available'}
        
        # Calculate metrics
        total_executions = len(self.execution_history)
        successful_executions = sum(
            1 for exec in self.execution_history 
            if exec.get('status') == 'success'
        )
        
        avg_execution_time = sum(
            exec.get('execution_time', 0) 
            for exec in self.execution_history
        ) / total_executions
        
        workflow_counts = {}
        for exec in self.execution_history:
            workflow_type = exec.get('workflow_type', 'unknown')
            workflow_counts[workflow_type] = workflow_counts.get(workflow_type, 0) + 1
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions,
            'average_execution_time': avg_execution_time,
            'workflow_usage': workflow_counts,
            'performance_period': {
                'start': self.execution_history[0]['timestamp'],
                'end': self.execution_history[-1]['timestamp']
            }
        }
    
    def reset(self):
        """Reset workflow manager."""
        self.current_workflow = None
        self.execution_history = []
        self.status = WorkflowStatus.INITIALIZED
        logger.info("Workflow manager reset")
    
    def get_status(self) -> str:
        """Get current status."""
        return self.status.value