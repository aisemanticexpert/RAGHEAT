"""
Portfolio Construction Tools
==========================

Tools for portfolio optimization, consensus building, and risk management.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crewai_tools import BaseTool
from loguru import logger

try:
    import cvxpy as cp
    from scipy.optimize import minimize
    import scipy.stats as stats
except ImportError:
    logger.warning("Some optimization libraries not available")

class ConsensusBuilderTool(BaseTool):
    """Tool for building consensus among agent recommendations."""
    
    name: str = "consensus_builder"
    description: str = "Build consensus from multiple agent analyses and recommendations"
    
    def _run(self, agent_analyses: List[Dict[str, Any]], consensus_method: str = "weighted_voting") -> Dict[str, Any]:
        """Build consensus from multiple agent analyses."""
        try:
            if not agent_analyses:
                return {'error': 'No agent analyses provided'}
            
            # Extract recommendations from analyses
            recommendations = self._extract_recommendations(agent_analyses)
            
            # Build consensus based on method
            if consensus_method == "weighted_voting":
                consensus = self._weighted_voting_consensus(recommendations)
            elif consensus_method == "delphi":
                consensus = self._delphi_consensus(recommendations)
            elif consensus_method == "bayesian":
                consensus = self._bayesian_consensus(recommendations)
            else:
                consensus = self._simple_majority_consensus(recommendations)
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(recommendations, consensus)
            
            # Generate consensus explanation
            explanation = self._generate_consensus_explanation(agent_analyses, consensus, confidence_metrics)
            
            return {
                'consensus_results': {
                    'consensus_recommendations': consensus,
                    'confidence_metrics': confidence_metrics,
                    'individual_analyses': agent_analyses,
                    'consensus_method': consensus_method,
                    'explanation': explanation
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error building consensus: {e}")
            return {'error': str(e)}
    
    def _extract_recommendations(self, analyses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract recommendations by stock from agent analyses."""
        recommendations = {}
        
        for analysis in analyses:
            agent_type = analysis.get('agent_type', 'unknown')
            agent_weight = analysis.get('weight', 1.0)
            
            if 'recommendations' in analysis:
                for rec in analysis['recommendations']:
                    symbol = rec.get('symbol')
                    if symbol:
                        if symbol not in recommendations:
                            recommendations[symbol] = []
                        
                        recommendations[symbol].append({
                            'agent': agent_type,
                            'weight': agent_weight,
                            'recommendation': rec.get('action', 'HOLD'),
                            'confidence': rec.get('confidence', 0.5),
                            'score': rec.get('score', 0.5),
                            'reasoning': rec.get('reasoning', '')
                        })
            
            # Handle single recommendation format
            elif 'recommendation' in analysis:
                rec = analysis['recommendation']
                symbol = rec.get('symbol')
                if symbol:
                    if symbol not in recommendations:
                        recommendations[symbol] = []
                    
                    recommendations[symbol].append({
                        'agent': agent_type,
                        'weight': agent_weight,
                        'recommendation': rec.get('action', 'HOLD'),
                        'confidence': rec.get('confidence', 0.5),
                        'score': rec.get('score', 0.5),
                        'reasoning': rec.get('reasoning', '')
                    })
        
        return recommendations
    
    def _weighted_voting_consensus(self, recommendations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Build consensus using weighted voting."""
        consensus = {}
        
        for symbol, recs in recommendations.items():
            # Convert recommendations to numerical scores
            weighted_scores = []
            action_votes = {'STRONG_BUY': 0, 'BUY': 0, 'HOLD': 0, 'SELL': 0, 'STRONG_SELL': 0}
            total_weight = 0
            
            for rec in recs:
                weight = rec['weight'] * rec['confidence']
                total_weight += weight
                
                # Numerical score (1-5 scale)
                score_map = {
                    'STRONG_SELL': 1, 'SELL': 2, 'HOLD': 3, 'BUY': 4, 'STRONG_BUY': 5
                }
                numerical_score = score_map.get(rec['recommendation'], 3)
                weighted_scores.append(numerical_score * weight)
                
                # Count votes
                action_votes[rec['recommendation']] += weight
            
            if total_weight > 0:
                # Calculate weighted average score
                avg_score = sum(weighted_scores) / total_weight
                
                # Determine consensus action
                consensus_action = max(action_votes, key=action_votes.get)
                
                # Calculate agreement level
                agreement_level = action_votes[consensus_action] / total_weight
                
                consensus[symbol] = {
                    'consensus_action': consensus_action,
                    'consensus_score': float(avg_score),
                    'agreement_level': float(agreement_level),
                    'vote_distribution': {k: float(v/total_weight) for k, v in action_votes.items()},
                    'num_agents': len(recs)
                }
        
        return consensus
    
    def _delphi_consensus(self, recommendations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Build consensus using Delphi method (simplified)."""
        # For simplicity, using a modified weighted voting with iterative refinement
        consensus = {}
        
        for symbol, recs in recommendations.items():
            # Calculate initial consensus
            scores = [self._action_to_score(rec['recommendation']) for rec in recs]
            weights = [rec['weight'] * rec['confidence'] for rec in recs]
            
            # Iterative refinement (simplified Delphi)
            for iteration in range(3):
                if len(scores) > 1:
                    median_score = np.median(scores)
                    
                    # Adjust scores toward median (consensus building)
                    adjusted_scores = []
                    for i, score in enumerate(scores):
                        adjustment_factor = 0.3 * weights[i]  # Weighted adjustment
                        adjusted_score = score + adjustment_factor * (median_score - score)
                        adjusted_scores.append(adjusted_score)
                    
                    scores = adjusted_scores
            
            # Final consensus
            final_score = np.average(scores, weights=weights) if len(scores) > 0 else 3.0
            consensus_action = self._score_to_action(final_score)
            
            # Calculate convergence (how much scores changed)
            initial_scores = [self._action_to_score(rec['recommendation']) for rec in recs]
            convergence = 1.0 - (np.std(scores) / max(np.std(initial_scores), 1e-6))
            
            consensus[symbol] = {
                'consensus_action': consensus_action,
                'consensus_score': float(final_score),
                'convergence': float(max(0, min(1, convergence))),
                'delphi_iterations': 3,
                'num_agents': len(recs)
            }
        
        return consensus
    
    def _bayesian_consensus(self, recommendations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Build consensus using Bayesian approach."""
        consensus = {}
        
        for symbol, recs in recommendations.items():
            # Prior belief (neutral)
            prior_score = 3.0  # HOLD
            prior_confidence = 0.1
            
            # Bayesian updating
            posterior_numerator = prior_score * prior_confidence
            posterior_denominator = prior_confidence
            
            for rec in recs:
                agent_score = self._action_to_score(rec['recommendation'])
                agent_confidence = rec['confidence'] * rec['weight']
                
                posterior_numerator += agent_score * agent_confidence
                posterior_denominator += agent_confidence
            
            # Posterior belief
            if posterior_denominator > 0:
                posterior_score = posterior_numerator / posterior_denominator
                posterior_confidence = posterior_denominator / (len(recs) + 1)
            else:
                posterior_score = prior_score
                posterior_confidence = prior_confidence
            
            consensus_action = self._score_to_action(posterior_score)
            
            consensus[symbol] = {
                'consensus_action': consensus_action,
                'consensus_score': float(posterior_score),
                'posterior_confidence': float(posterior_confidence),
                'bayesian_update': True,
                'num_agents': len(recs)
            }
        
        return consensus
    
    def _simple_majority_consensus(self, recommendations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Build consensus using simple majority voting."""
        consensus = {}
        
        for symbol, recs in recommendations.items():
            # Count votes for each action
            votes = {}
            for rec in recs:
                action = rec['recommendation']
                votes[action] = votes.get(action, 0) + 1
            
            # Find majority
            consensus_action = max(votes, key=votes.get) if votes else 'HOLD'
            agreement_level = votes[consensus_action] / len(recs) if recs else 0
            
            consensus[symbol] = {
                'consensus_action': consensus_action,
                'agreement_level': float(agreement_level),
                'vote_counts': votes,
                'num_agents': len(recs)
            }
        
        return consensus
    
    def _action_to_score(self, action: str) -> float:
        """Convert action to numerical score."""
        score_map = {
            'STRONG_SELL': 1.0, 'SELL': 2.0, 'HOLD': 3.0, 'BUY': 4.0, 'STRONG_BUY': 5.0
        }
        return score_map.get(action, 3.0)
    
    def _score_to_action(self, score: float) -> str:
        """Convert numerical score to action."""
        if score >= 4.5:
            return 'STRONG_BUY'
        elif score >= 3.5:
            return 'BUY'
        elif score >= 2.5:
            return 'HOLD'
        elif score >= 1.5:
            return 'SELL'
        else:
            return 'STRONG_SELL'
    
    def _calculate_confidence_metrics(self, recommendations: Dict[str, List[Dict[str, Any]]], 
                                    consensus: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence metrics for the consensus."""
        total_agreements = 0
        total_recommendations = 0
        agreement_scores = []
        
        for symbol in consensus:
            if symbol in recommendations:
                recs = recommendations[symbol]
                consensus_action = consensus[symbol]['consensus_action']
                
                # Count agreements with consensus
                agreements = sum(1 for rec in recs if rec['recommendation'] == consensus_action)
                total_agreements += agreements
                total_recommendations += len(recs)
                
                if len(recs) > 0:
                    agreement_scores.append(agreements / len(recs))
        
        return {
            'overall_agreement_rate': float(total_agreements / max(total_recommendations, 1)),
            'average_agreement_per_stock': float(np.mean(agreement_scores)) if agreement_scores else 0.0,
            'consensus_strength': float(np.std(agreement_scores)) if len(agreement_scores) > 1 else 1.0,
            'number_of_stocks': len(consensus),
            'total_recommendations_processed': total_recommendations
        }
    
    def _generate_consensus_explanation(self, analyses: List[Dict[str, Any]], 
                                       consensus: Dict[str, Dict[str, Any]],
                                       confidence_metrics: Dict[str, Any]) -> str:
        """Generate explanation of consensus building process."""
        num_agents = len(analyses)
        num_stocks = len(consensus)
        agreement_rate = confidence_metrics['overall_agreement_rate']
        
        explanation = f"Consensus built from {num_agents} agents analyzing {num_stocks} stocks. "
        
        if agreement_rate > 0.7:
            explanation += "Strong consensus achieved with high agent agreement. "
        elif agreement_rate > 0.5:
            explanation += "Moderate consensus with some disagreement among agents. "
        else:
            explanation += "Weak consensus due to significant agent disagreement. "
        
        # Highlight highly recommended stocks
        strong_buys = [symbol for symbol, data in consensus.items() 
                      if data['consensus_action'] in ['STRONG_BUY', 'BUY']]
        
        if strong_buys:
            explanation += f"Agents recommend buying: {', '.join(strong_buys[:5])}. "
        
        return explanation

class DebateModerator(BaseTool):
    """Tool for facilitating structured debates between agents."""
    
    name: str = "debate_moderator"
    description: str = "Moderate structured debates between agents to resolve disagreements"
    
    def _run(self, agent_positions: List[Dict[str, Any]], debate_topic: str = "portfolio_construction") -> Dict[str, Any]:
        """Moderate a debate between agents."""
        try:
            if len(agent_positions) < 2:
                return {'error': 'Need at least 2 agents for debate'}
            
            # Initialize debate
            debate_rounds = []
            current_positions = agent_positions.copy()
            
            # Run debate rounds
            for round_num in range(3):  # 3 rounds of debate
                round_result = self._run_debate_round(current_positions, round_num + 1)
                debate_rounds.append(round_result)
                
                # Update positions based on debate
                current_positions = self._update_positions_after_round(current_positions, round_result)
            
            # Final resolution
            resolution = self._resolve_debate(current_positions, debate_rounds)
            
            return {
                'debate_results': {
                    'topic': debate_topic,
                    'initial_positions': agent_positions,
                    'debate_rounds': debate_rounds,
                    'final_positions': current_positions,
                    'resolution': resolution,
                    'debate_summary': self._summarize_debate(debate_rounds, resolution)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in debate moderation: {e}")
            return {'error': str(e)}
    
    def _run_debate_round(self, positions: List[Dict[str, Any]], round_num: int) -> Dict[str, Any]:
        """Run a single round of debate."""
        round_exchanges = []
        
        # Each agent presents their position and challenges others
        for i, agent in enumerate(positions):
            agent_id = agent.get('agent_id', f'agent_{i}')
            agent_position = agent.get('position', {})
            
            # Generate arguments
            arguments = self._generate_arguments(agent, positions)
            
            # Generate challenges to other agents
            challenges = self._generate_challenges(agent, positions)
            
            round_exchanges.append({
                'agent_id': agent_id,
                'arguments': arguments,
                'challenges': challenges,
                'confidence_level': agent.get('confidence', 0.5)
            })
        
        # Analyze round dynamics
        round_analysis = self._analyze_round(round_exchanges)
        
        return {
            'round_number': round_num,
            'exchanges': round_exchanges,
            'round_analysis': round_analysis
        }
    
    def _generate_arguments(self, agent: Dict[str, Any], all_positions: List[Dict[str, Any]]) -> List[str]:
        """Generate arguments for an agent's position."""
        agent_type = agent.get('agent_type', 'unknown')
        position = agent.get('position', {})
        
        arguments = []
        
        # Generate arguments based on agent type
        if agent_type == 'fundamental_analyst':
            if 'strong_fundamentals' in str(position):
                arguments.append("Strong financial metrics support long-term value creation")
                arguments.append("Balance sheet strength provides downside protection")
        
        elif agent_type == 'sentiment_analyst':
            if 'positive_sentiment' in str(position):
                arguments.append("Market sentiment and news flow strongly favor this position")
                arguments.append("Social media buzz indicates growing investor interest")
        
        elif agent_type == 'valuation_analyst':
            if 'undervalued' in str(position):
                arguments.append("Technical indicators suggest the stock is undervalued")
                arguments.append("Risk-adjusted returns are attractive at current levels")
        
        elif agent_type == 'heat_diffusion_analyst':
            arguments.append("Heat diffusion models show strong influence propagation")
            arguments.append("Network effects support this investment thesis")
        
        # Generic arguments if no specific ones
        if not arguments:
            arguments.append(f"Analysis from {agent_type} perspective strongly supports this position")
            arguments.append("Risk-reward profile is favorable based on our methodology")
        
        return arguments
    
    def _generate_challenges(self, agent: Dict[str, Any], all_positions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate challenges to other agents' positions."""
        challenges = []
        agent_id = agent.get('agent_id', 'unknown')
        
        for other_agent in all_positions:
            other_id = other_agent.get('agent_id', 'unknown')
            if other_id != agent_id:
                # Find disagreements
                disagreements = self._find_disagreements(agent, other_agent)
                
                for disagreement in disagreements:
                    challenges.append({
                        'target_agent': other_id,
                        'challenge': disagreement
                    })
        
        return challenges
    
    def _find_disagreements(self, agent1: Dict[str, Any], agent2: Dict[str, Any]) -> List[str]:
        """Find areas of disagreement between two agents."""
        disagreements = []
        
        pos1 = agent1.get('position', {})
        pos2 = agent2.get('position', {})
        
        # Compare recommendations
        rec1 = pos1.get('recommendation', 'HOLD')
        rec2 = pos2.get('recommendation', 'HOLD')
        
        if rec1 != rec2:
            disagreements.append(f"I recommend {rec1} while you recommend {rec2} - explain your reasoning")
        
        # Compare confidence levels
        conf1 = agent1.get('confidence', 0.5)
        conf2 = agent2.get('confidence', 0.5)
        
        if abs(conf1 - conf2) > 0.3:
            disagreements.append("Our confidence levels differ significantly - what drives your conviction?")
        
        # Generic disagreement if specific ones not found
        if not disagreements and rec1 != rec2:
            disagreements.append("I challenge your methodology and conclusions")
        
        return disagreements
    
    def _analyze_round(self, exchanges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the dynamics of a debate round."""
        total_arguments = sum(len(ex['arguments']) for ex in exchanges)
        total_challenges = sum(len(ex['challenges']) for ex in exchanges)
        avg_confidence = np.mean([ex['confidence_level'] for ex in exchanges])
        
        # Find most active participants
        most_active = max(exchanges, key=lambda x: len(x['arguments']) + len(x['challenges']))
        
        return {
            'total_arguments': total_arguments,
            'total_challenges': total_challenges,
            'average_confidence': float(avg_confidence),
            'most_active_agent': most_active['agent_id'],
            'debate_intensity': min(1.0, (total_arguments + total_challenges) / (len(exchanges) * 5)),
            'consensus_potential': self._assess_consensus_potential(exchanges)
        }
    
    def _assess_consensus_potential(self, exchanges: List[Dict[str, Any]]) -> float:
        """Assess potential for consensus based on round dynamics."""
        # Simple heuristic: fewer challenges and higher confidence suggest consensus potential
        total_challenges = sum(len(ex['challenges']) for ex in exchanges)
        avg_confidence = np.mean([ex['confidence_level'] for ex in exchanges])
        
        # Lower challenges and higher confidence = higher consensus potential
        challenge_factor = max(0, 1 - (total_challenges / (len(exchanges) * 3)))
        confidence_factor = avg_confidence
        
        return float((challenge_factor + confidence_factor) / 2)
    
    def _update_positions_after_round(self, positions: List[Dict[str, Any]], 
                                    round_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update agent positions based on debate round results."""
        updated_positions = []
        
        for position in positions:
            updated_position = position.copy()
            
            # Slightly adjust confidence based on debate dynamics
            current_confidence = position.get('confidence', 0.5)
            
            # If consensus potential is high, agents become more confident
            consensus_potential = round_result['round_analysis']['consensus_potential']
            confidence_adjustment = (consensus_potential - 0.5) * 0.1
            
            new_confidence = max(0.1, min(0.9, current_confidence + confidence_adjustment))
            updated_position['confidence'] = new_confidence
            
            updated_positions.append(updated_position)
        
        return updated_positions
    
    def _resolve_debate(self, final_positions: List[Dict[str, Any]], 
                       debate_rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve the debate and determine final outcome."""
        # Calculate final consensus metrics
        final_consensus_potential = np.mean([
            round_data['round_analysis']['consensus_potential'] 
            for round_data in debate_rounds
        ])
        
        # Determine winning arguments
        all_arguments = []
        for round_data in debate_rounds:
            for exchange in round_data['exchanges']:
                all_arguments.extend(exchange['arguments'])
        
        # Simple resolution based on final positions
        recommendations = [pos.get('position', {}).get('recommendation', 'HOLD') 
                         for pos in final_positions]
        
        from collections import Counter
        recommendation_counts = Counter(recommendations)
        winning_recommendation = recommendation_counts.most_common(1)[0][0] if recommendations else 'HOLD'
        
        return {
            'final_recommendation': winning_recommendation,
            'consensus_strength': float(final_consensus_potential),
            'agreement_level': float(recommendation_counts[winning_recommendation] / len(recommendations)) if recommendations else 0.0,
            'debate_effectiveness': float(np.mean([r['round_analysis']['debate_intensity'] for r in debate_rounds])),
            'total_arguments_presented': len(all_arguments),
            'resolution_summary': f"After {len(debate_rounds)} rounds of debate, consensus reached on {winning_recommendation}"
        }
    
    def _summarize_debate(self, rounds: List[Dict[str, Any]], resolution: Dict[str, Any]) -> str:
        """Generate a summary of the entire debate."""
        num_rounds = len(rounds)
        final_rec = resolution['final_recommendation']
        consensus_strength = resolution['consensus_strength']
        
        summary = f"Debate concluded after {num_rounds} rounds with recommendation: {final_rec}. "
        
        if consensus_strength > 0.7:
            summary += "Strong consensus achieved through structured argumentation. "
        elif consensus_strength > 0.5:
            summary += "Moderate consensus reached with some remaining disagreements. "
        else:
            summary += "Weak consensus - significant disagreements persist. "
        
        total_args = sum(r['round_analysis']['total_arguments'] for r in rounds)
        summary += f"Total of {total_args} arguments presented across all rounds."
        
        return summary

class PortfolioOptimizerTool(BaseTool):
    """Tool for optimizing portfolio allocation."""
    
    name: str = "portfolio_optimizer"
    description: str = "Optimize portfolio weights using modern portfolio theory and advanced techniques"
    
    def _run(self, stocks: List[str], expected_returns: Dict[str, float], 
             risk_matrix: Dict[str, Dict[str, float]], constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize portfolio allocation."""
        try:
            # Prepare data
            n_assets = len(stocks)
            if n_assets == 0:
                return {'error': 'No stocks provided for optimization'}
            
            # Convert to arrays
            returns_array = np.array([expected_returns.get(stock, 0.0) for stock in stocks])
            
            # Build covariance matrix
            cov_matrix = self._build_covariance_matrix(stocks, risk_matrix)
            
            # Set constraints
            constraints_dict = constraints or {}
            
            # Optimize portfolio
            optimization_results = {}
            
            # Mean-Variance Optimization
            mv_result = self._mean_variance_optimization(returns_array, cov_matrix, constraints_dict)
            optimization_results['mean_variance'] = mv_result
            
            # Risk Parity
            rp_result = self._risk_parity_optimization(cov_matrix, constraints_dict)
            optimization_results['risk_parity'] = rp_result
            
            # Maximum Sharpe Ratio
            sharpe_result = self._max_sharpe_optimization(returns_array, cov_matrix, constraints_dict)
            optimization_results['max_sharpe'] = sharpe_result
            
            # Minimum Variance
            min_var_result = self._min_variance_optimization(cov_matrix, constraints_dict)
            optimization_results['min_variance'] = min_var_result
            
            # Select best portfolio based on Sharpe ratio
            best_portfolio = self._select_best_portfolio(optimization_results)
            
            return {
                'portfolio_optimization': {
                    'stocks': stocks,
                    'optimization_results': optimization_results,
                    'recommended_portfolio': best_portfolio,
                    'portfolio_metrics': self._calculate_portfolio_metrics(
                        best_portfolio['weights'], returns_array, cov_matrix
                    )
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {'error': str(e)}
    
    def _build_covariance_matrix(self, stocks: List[str], risk_matrix: Dict[str, Dict[str, float]]) -> np.ndarray:
        """Build covariance matrix from risk data."""
        n = len(stocks)
        cov_matrix = np.eye(n) * 0.01  # Default small variance
        
        for i, stock1 in enumerate(stocks):
            for j, stock2 in enumerate(stocks):
                if stock1 in risk_matrix and stock2 in risk_matrix[stock1]:
                    cov_matrix[i, j] = risk_matrix[stock1][stock2]
                elif i == j:
                    # Use provided variance or default
                    variance = risk_matrix.get(stock1, {}).get(stock1, 0.01)
                    cov_matrix[i, j] = variance
        
        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return cov_matrix
    
    def _mean_variance_optimization(self, returns: np.ndarray, cov_matrix: np.ndarray, 
                                   constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Mean-variance optimization."""
        try:
            n = len(returns)
            
            # Define optimization variables
            weights = cp.Variable(n)
            
            # Objective: maximize expected return - lambda * risk
            risk_aversion = constraints.get('risk_aversion', 1.0)
            portfolio_return = returns.T @ weights
            portfolio_risk = cp.quad_form(weights, cov_matrix)
            objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
            
            # Constraints
            constraints_list = [
                cp.sum(weights) == 1,  # Fully invested
                weights >= 0  # Long-only (can be modified)
            ]
            
            # Add custom constraints
            if 'max_weight' in constraints:
                constraints_list.append(weights <= constraints['max_weight'])
            
            if 'min_weight' in constraints:
                constraints_list.append(weights >= constraints['min_weight'])
            
            # Solve optimization
            problem = cp.Problem(objective, constraints_list)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                return {
                    'status': 'optimal',
                    'weights': optimal_weights.tolist(),
                    'expected_return': float(returns.T @ optimal_weights),
                    'risk': float(np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)),
                    'sharpe_ratio': self._calculate_sharpe_ratio(optimal_weights, returns, cov_matrix)
                }
            else:
                return {'status': 'failed', 'error': f'Optimization failed: {problem.status}'}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _risk_parity_optimization(self, cov_matrix: np.ndarray, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Risk parity optimization."""
        try:
            n = cov_matrix.shape[0]
            
            def risk_parity_objective(weights):
                """Risk parity objective function."""
                weights = np.array(weights)
                portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                
                # Marginal risk contributions
                marginal_contrib = cov_matrix @ weights / portfolio_vol
                risk_contrib = weights * marginal_contrib
                
                # Target equal risk contribution
                target_contrib = portfolio_vol / n
                
                # Sum of squared deviations from equal risk contribution
                return np.sum((risk_contrib - target_contrib) ** 2)
            
            # Constraints
            cons = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
            ]
            
            bounds = [(0, 1) for _ in range(n)]  # Long-only
            
            # Initial guess (equal weights)
            x0 = np.ones(n) / n
            
            # Optimize
            result = minimize(risk_parity_objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=cons)
            
            if result.success:
                optimal_weights = result.x
                return {
                    'status': 'optimal',
                    'weights': optimal_weights.tolist(),
                    'risk': float(np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)),
                    'risk_contributions': self._calculate_risk_contributions(optimal_weights, cov_matrix).tolist()
                }
            else:
                return {'status': 'failed', 'error': result.message}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _max_sharpe_optimization(self, returns: np.ndarray, cov_matrix: np.ndarray, 
                               constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Maximum Sharpe ratio optimization."""
        try:
            n = len(returns)
            risk_free_rate = constraints.get('risk_free_rate', 0.02)
            
            # Excess returns
            excess_returns = returns - risk_free_rate
            
            def negative_sharpe(weights):
                """Negative Sharpe ratio for minimization."""
                portfolio_return = weights.T @ excess_returns
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                
                if portfolio_risk == 0:
                    return -np.inf
                
                return -(portfolio_return / portfolio_risk)
            
            # Constraints
            cons = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            ]
            
            bounds = [(0, 1) for _ in range(n)]
            x0 = np.ones(n) / n
            
            # Optimize
            result = minimize(negative_sharpe, x0, method='SLSQP', 
                            bounds=bounds, constraints=cons)
            
            if result.success:
                optimal_weights = result.x
                return {
                    'status': 'optimal',
                    'weights': optimal_weights.tolist(),
                    'expected_return': float(returns.T @ optimal_weights),
                    'risk': float(np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)),
                    'sharpe_ratio': -result.fun  # Convert back to positive
                }
            else:
                return {'status': 'failed', 'error': result.message}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _min_variance_optimization(self, cov_matrix: np.ndarray, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Minimum variance optimization."""
        try:
            n = cov_matrix.shape[0]
            
            # Define optimization variables
            weights = cp.Variable(n)
            
            # Objective: minimize portfolio variance
            objective = cp.Minimize(cp.quad_form(weights, cov_matrix))
            
            # Constraints
            constraints_list = [
                cp.sum(weights) == 1,
                weights >= 0
            ]
            
            # Solve
            problem = cp.Problem(objective, constraints_list)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                return {
                    'status': 'optimal',
                    'weights': optimal_weights.tolist(),
                    'risk': float(np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights))
                }
            else:
                return {'status': 'failed', 'error': f'Optimization failed: {problem.status}'}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_sharpe_ratio(self, weights: np.ndarray, returns: np.ndarray, 
                               cov_matrix: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        portfolio_return = weights.T @ returns
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        
        if portfolio_risk == 0:
            return 0.0
        
        return float((portfolio_return - risk_free_rate) / portfolio_risk)
    
    def _calculate_risk_contributions(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk contributions."""
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights / portfolio_vol
        return weights * marginal_contrib
    
    def _select_best_portfolio(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Select best portfolio from optimization results."""
        best_portfolio = None
        best_sharpe = -np.inf
        
        for strategy, result in results.items():
            if result.get('status') == 'optimal':
                sharpe = result.get('sharpe_ratio', 0)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_portfolio = result.copy()
                    best_portfolio['strategy'] = strategy
        
        if best_portfolio is None:
            # Fallback to equal weights
            n = len(results['mean_variance'].get('weights', [1]))
            best_portfolio = {
                'strategy': 'equal_weight',
                'weights': [1/n] * n,
                'status': 'fallback'
            }
        
        return best_portfolio
    
    def _calculate_portfolio_metrics(self, weights: List[float], returns: np.ndarray, 
                                   cov_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""
        weights_array = np.array(weights)
        
        portfolio_return = float(weights_array.T @ returns)
        portfolio_risk = float(np.sqrt(weights_array.T @ cov_matrix @ weights_array))
        sharpe_ratio = self._calculate_sharpe_ratio(weights_array, returns, cov_matrix)
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'max_weight': float(np.max(weights_array)),
            'min_weight': float(np.min(weights_array)),
            'concentration': float(np.sum(weights_array ** 2))  # Herfindahl index
        }

class RiskAssessorTool(BaseTool):
    """Tool for comprehensive risk assessment."""
    
    name: str = "risk_assessor"
    description: str = "Assess various risk factors and calculate risk metrics for portfolios"
    
    def _run(self, portfolio: Dict[str, Any], market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess portfolio risks."""
        try:
            weights = portfolio.get('weights', [])
            stocks = portfolio.get('stocks', [])
            
            if len(weights) != len(stocks) or not weights:
                return {'error': 'Invalid portfolio specification'}
            
            # Calculate different risk metrics
            risk_metrics = {
                'market_risk': self._assess_market_risk(weights, stocks, market_data),
                'concentration_risk': self._assess_concentration_risk(weights),
                'sector_risk': self._assess_sector_risk(weights, stocks, market_data),
                'liquidity_risk': self._assess_liquidity_risk(weights, stocks, market_data),
                'correlation_risk': self._assess_correlation_risk(weights, stocks, market_data)
            }
            
            # Overall risk score
            overall_risk = self._calculate_overall_risk_score(risk_metrics)
            
            # Risk recommendations
            recommendations = self._generate_risk_recommendations(risk_metrics)
            
            return {
                'risk_assessment': {
                    'individual_risks': risk_metrics,
                    'overall_risk_score': overall_risk,
                    'risk_level': self._categorize_risk_level(overall_risk),
                    'recommendations': recommendations
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {'error': str(e)}
    
    def _assess_market_risk(self, weights: List[float], stocks: List[str], 
                           market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess market risk (beta exposure)."""
        if not market_data:
            return {'score': 0.5, 'note': 'No market data available'}
        
        # Calculate portfolio beta
        portfolio_beta = 0.0
        for i, stock in enumerate(stocks):
            stock_beta = market_data.get(stock, {}).get('beta', 1.0)
            portfolio_beta += weights[i] * stock_beta
        
        # Market risk score (higher beta = higher risk)
        risk_score = min(1.0, abs(portfolio_beta - 1.0) + max(0, portfolio_beta - 1.5) * 0.5)
        
        return {
            'portfolio_beta': float(portfolio_beta),
            'risk_score': float(risk_score),
            'interpretation': self._interpret_beta(portfolio_beta)
        }
    
    def _assess_concentration_risk(self, weights: List[float]) -> Dict[str, Any]:
        """Assess concentration risk."""
        weights_array = np.array(weights)
        
        # Herfindahl-Hirschman Index
        hhi = np.sum(weights_array ** 2)
        
        # Maximum weight
        max_weight = np.max(weights_array)
        
        # Number of significant positions (>5%)
        significant_positions = np.sum(weights_array > 0.05)
        
        # Risk score based on concentration
        concentration_risk = hhi * 2  # Scale to 0-1
        
        return {
            'hhi': float(hhi),
            'max_weight': float(max_weight),
            'significant_positions': int(significant_positions),
            'risk_score': float(min(1.0, concentration_risk)),
            'diversification_ratio': float(1 / (hhi * len(weights)))
        }
    
    def _assess_sector_risk(self, weights: List[float], stocks: List[str], 
                           market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess sector concentration risk."""
        if not market_data:
            return {'score': 0.3, 'note': 'No sector data available'}
        
        # Calculate sector allocations
        sector_weights = {}
        for i, stock in enumerate(stocks):
            sector = market_data.get(stock, {}).get('sector', 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + weights[i]
        
        # Calculate sector HHI
        sector_hhi = sum(w**2 for w in sector_weights.values())
        
        # Maximum sector allocation
        max_sector_weight = max(sector_weights.values()) if sector_weights else 0
        
        return {
            'sector_allocations': sector_weights,
            'sector_hhi': float(sector_hhi),
            'max_sector_weight': float(max_sector_weight),
            'risk_score': float(min(1.0, sector_hhi * 1.5)),
            'num_sectors': len(sector_weights)
        }
    
    def _assess_liquidity_risk(self, weights: List[float], stocks: List[str], 
                              market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess liquidity risk."""
        if not market_data:
            return {'score': 0.3, 'note': 'No liquidity data available'}
        
        # Calculate weighted average liquidity metrics
        total_liquidity_score = 0
        for i, stock in enumerate(stocks):
            market_cap = market_data.get(stock, {}).get('market_cap', 1e9)
            avg_volume = market_data.get(stock, {}).get('avg_volume', 1e6)
            
            # Simple liquidity score (higher is better)
            liquidity_score = min(1.0, (market_cap / 1e9) * 0.5 + (avg_volume / 1e6) * 0.5)
            total_liquidity_score += weights[i] * liquidity_score
        
        # Risk score (inverse of liquidity)
        risk_score = 1.0 - total_liquidity_score
        
        return {
            'portfolio_liquidity_score': float(total_liquidity_score),
            'risk_score': float(risk_score),
            'liquidity_level': 'high' if total_liquidity_score > 0.7 else 'medium' if total_liquidity_score > 0.4 else 'low'
        }
    
    def _assess_correlation_risk(self, weights: List[float], stocks: List[str], 
                                market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess correlation risk."""
        if not market_data or len(stocks) < 2:
            return {'score': 0.3, 'note': 'Insufficient data for correlation analysis'}
        
        # Get correlation matrix if available
        correlations = market_data.get('correlations', {})
        
        if not correlations:
            return {'score': 0.5, 'note': 'No correlation data available'}
        
        # Calculate average correlation weighted by portfolio weights
        weighted_correlation = 0
        total_weight_pairs = 0
        
        for i, stock1 in enumerate(stocks):
            for j, stock2 in enumerate(stocks):
                if i != j and stock1 in correlations and stock2 in correlations.get(stock1, {}):
                    correlation = correlations[stock1][stock2]
                    weight_product = weights[i] * weights[j]
                    weighted_correlation += correlation * weight_product
                    total_weight_pairs += weight_product
        
        if total_weight_pairs > 0:
            avg_correlation = weighted_correlation / total_weight_pairs
        else:
            avg_correlation = 0.5  # Default assumption
        
        # Risk score (higher correlation = higher risk)
        risk_score = max(0, avg_correlation)
        
        return {
            'average_correlation': float(avg_correlation),
            'risk_score': float(risk_score),
            'diversification_benefit': float(1 - abs(avg_correlation))
        }
    
    def _calculate_overall_risk_score(self, risk_metrics: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall risk score."""
        # Weight different risk factors
        weights = {
            'market_risk': 0.25,
            'concentration_risk': 0.30,
            'sector_risk': 0.20,
            'liquidity_risk': 0.15,
            'correlation_risk': 0.10
        }
        
        overall_score = 0
        total_weight = 0
        
        for risk_type, risk_data in risk_metrics.items():
            if 'risk_score' in risk_data and risk_type in weights:
                overall_score += risk_data['risk_score'] * weights[risk_type]
                total_weight += weights[risk_type]
        
        return float(overall_score / total_weight) if total_weight > 0 else 0.5
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize overall risk level."""
        if risk_score < 0.3:
            return 'Low'
        elif risk_score < 0.6:
            return 'Medium'
        elif risk_score < 0.8:
            return 'High'
        else:
            return 'Very High'
    
    def _generate_risk_recommendations(self, risk_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        # Concentration risk
        if risk_metrics.get('concentration_risk', {}).get('risk_score', 0) > 0.6:
            recommendations.append("Reduce concentration risk by limiting individual position sizes")
        
        # Sector risk
        if risk_metrics.get('sector_risk', {}).get('risk_score', 0) > 0.7:
            recommendations.append("Diversify across more sectors to reduce sector concentration")
        
        # Market risk
        market_beta = risk_metrics.get('market_risk', {}).get('portfolio_beta', 1.0)
        if market_beta > 1.5:
            recommendations.append("Consider adding defensive assets to reduce market sensitivity")
        elif market_beta < 0.5:
            recommendations.append("Portfolio may underperform in bull markets due to low beta")
        
        # Liquidity risk
        if risk_metrics.get('liquidity_risk', {}).get('risk_score', 0) > 0.6:
            recommendations.append("Improve liquidity by focusing on larger, more liquid stocks")
        
        # Correlation risk
        if risk_metrics.get('correlation_risk', {}).get('risk_score', 0) > 0.7:
            recommendations.append("Add uncorrelated assets to improve diversification benefits")
        
        if not recommendations:
            recommendations.append("Portfolio risk profile appears well-balanced")
        
        return recommendations
    
    def _interpret_beta(self, beta: float) -> str:
        """Interpret portfolio beta."""
        if beta > 1.3:
            return "High market sensitivity - amplifies market movements"
        elif beta > 1.1:
            return "Moderate market sensitivity - slightly more volatile than market"
        elif beta < 0.8:
            return "Low market sensitivity - less volatile than market"
        else:
            return "Market-like sensitivity - moves with overall market"

class WeightAllocatorTool(BaseTool):
    """Tool for allocating weights to portfolio positions."""
    
    name: str = "weight_allocator"
    description: str = "Allocate portfolio weights based on various allocation strategies"
    
    def _run(self, stocks: List[str], scores: Dict[str, float], 
             allocation_method: str = "score_weighted", constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Allocate weights to portfolio positions."""
        try:
            if not stocks or not scores:
                return {'error': 'Stocks and scores required for allocation'}
            
            constraints_dict = constraints or {}
            
            # Apply allocation method
            if allocation_method == "equal_weight":
                weights = self._equal_weight_allocation(stocks)
            elif allocation_method == "score_weighted":
                weights = self._score_weighted_allocation(stocks, scores)
            elif allocation_method == "rank_weighted":
                weights = self._rank_weighted_allocation(stocks, scores)
            elif allocation_method == "kelly_criterion":
                weights = self._kelly_criterion_allocation(stocks, scores, constraints_dict)
            elif allocation_method == "risk_budgeted":
                weights = self._risk_budgeted_allocation(stocks, scores, constraints_dict)
            else:
                weights = self._score_weighted_allocation(stocks, scores)
            
            # Apply constraints
            final_weights = self._apply_constraints(weights, stocks, constraints_dict)
            
            # Calculate allocation metrics
            allocation_metrics = self._calculate_allocation_metrics(final_weights, stocks, scores)
            
            return {
                'weight_allocation': {
                    'stocks': stocks,
                    'weights': final_weights,
                    'allocation_method': allocation_method,
                    'allocation_metrics': allocation_metrics,
                    'constraints_applied': list(constraints_dict.keys())
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in weight allocation: {e}")
            return {'error': str(e)}
    
    def _equal_weight_allocation(self, stocks: List[str]) -> Dict[str, float]:
        """Equal weight allocation."""
        weight = 1.0 / len(stocks)
        return {stock: weight for stock in stocks}
    
    def _score_weighted_allocation(self, stocks: List[str], scores: Dict[str, float]) -> Dict[str, float]:
        """Score-weighted allocation."""
        # Ensure all scores are positive for weighting
        min_score = min(scores.values())
        adjusted_scores = {stock: scores.get(stock, 0) - min_score + 0.1 for stock in stocks}
        
        total_score = sum(adjusted_scores.values())
        
        if total_score == 0:
            return self._equal_weight_allocation(stocks)
        
        return {stock: adjusted_scores[stock] / total_score for stock in stocks}
    
    def _rank_weighted_allocation(self, stocks: List[str], scores: Dict[str, float]) -> Dict[str, float]:
        """Rank-weighted allocation (higher rank gets more weight)."""
        # Sort stocks by score
        sorted_stocks = sorted(stocks, key=lambda x: scores.get(x, 0), reverse=True)
        
        # Assign weights based on rank (1/rank weighting)
        rank_weights = {}
        total_weight = 0
        
        for i, stock in enumerate(sorted_stocks):
            rank = i + 1
            weight = 1.0 / rank
            rank_weights[stock] = weight
            total_weight += weight
        
        # Normalize weights
        return {stock: weight / total_weight for stock, weight in rank_weights.items()}
    
    def _kelly_criterion_allocation(self, stocks: List[str], scores: Dict[str, float], 
                                   constraints: Dict[str, Any]) -> Dict[str, float]:
        """Kelly Criterion allocation (simplified)."""
        # For Kelly criterion, we need expected returns and win probability
        # Using scores as proxy for expected returns
        
        total_capital = 1.0
        weights = {}
        
        for stock in stocks:
            expected_return = scores.get(stock, 0)
            win_probability = constraints.get('win_probability', {}).get(stock, 0.6)
            loss_probability = 1 - win_probability
            
            # Simplified Kelly formula
            if loss_probability > 0:
                kelly_fraction = (win_probability * expected_return - loss_probability) / expected_return
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            else:
                kelly_fraction = 0.1  # Default small allocation
            
            weights[stock] = kelly_fraction
        
        # Normalize if total exceeds 1
        total_weight = sum(weights.values())
        if total_weight > 1:
            weights = {stock: weight / total_weight for stock, weight in weights.items()}
        elif total_weight < 1:
            # Distribute remaining weight proportionally
            remaining = 1 - total_weight
            for stock in weights:
                weights[stock] += remaining / len(stocks)
        
        return weights
    
    def _risk_budgeted_allocation(self, stocks: List[str], scores: Dict[str, float], 
                                 constraints: Dict[str, Any]) -> Dict[str, float]:
        """Risk-budgeted allocation."""
        # Allocate based on inverse volatility weighting
        volatilities = constraints.get('volatilities', {})
        
        if not volatilities:
            # Fallback to score weighting
            return self._score_weighted_allocation(stocks, scores)
        
        # Calculate inverse volatility weights
        inv_vol_weights = {}
        for stock in stocks:
            vol = volatilities.get(stock, 0.2)  # Default 20% volatility
            inv_vol_weights[stock] = 1.0 / vol
        
        # Normalize
        total_inv_vol = sum(inv_vol_weights.values())
        weights = {stock: weight / total_inv_vol for stock, weight in inv_vol_weights.items()}
        
        # Adjust by scores
        for stock in stocks:
            score_multiplier = max(0.1, scores.get(stock, 0.5))
            weights[stock] *= score_multiplier
        
        # Re-normalize
        total_weight = sum(weights.values())
        return {stock: weight / total_weight for stock, weight in weights.items()}
    
    def _apply_constraints(self, weights: Dict[str, float], stocks: List[str], 
                          constraints: Dict[str, Any]) -> Dict[str, float]:
        """Apply allocation constraints."""
        constrained_weights = weights.copy()
        
        # Maximum weight constraint
        max_weight = constraints.get('max_weight', 1.0)
        for stock in stocks:
            if constrained_weights[stock] > max_weight:
                constrained_weights[stock] = max_weight
        
        # Minimum weight constraint
        min_weight = constraints.get('min_weight', 0.0)
        for stock in stocks:
            if constrained_weights[stock] < min_weight:
                constrained_weights[stock] = min_weight
        
        # Re-normalize to ensure sum equals 1
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {
                stock: weight / total_weight 
                for stock, weight in constrained_weights.items()
            }
        
        # Exclude positions below threshold
        min_position_size = constraints.get('min_position_size', 0.01)
        significant_positions = {
            stock: weight for stock, weight in constrained_weights.items() 
            if weight >= min_position_size
        }
        
        if significant_positions:
            # Re-normalize significant positions
            total_significant = sum(significant_positions.values())
            constrained_weights = {
                stock: weight / total_significant 
                for stock, weight in significant_positions.items()
            }
        
        return constrained_weights
    
    def _calculate_allocation_metrics(self, weights: Dict[str, float], stocks: List[str], 
                                    scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate allocation performance metrics."""
        weight_values = list(weights.values())
        
        metrics = {
            'number_of_positions': len([w for w in weight_values if w > 0.001]),
            'max_weight': float(max(weight_values)),
            'min_weight': float(min(weight_values)),
            'weight_concentration': float(sum(w**2 for w in weight_values)),  # HHI
            'effective_number_of_stocks': float(1 / sum(w**2 for w in weight_values)),
            'diversification_ratio': float(len(stocks) / (1 + sum(w**2 for w in weight_values) * len(stocks)))
        }
        
        # Score-weighted metrics
        if scores:
            weighted_score = sum(weights.get(stock, 0) * scores.get(stock, 0) for stock in stocks)
            metrics['portfolio_weighted_score'] = float(weighted_score)
        
        return metrics

# Initialize tools
consensus_builder = ConsensusBuilderTool()
debate_moderator = DebateModerator()
portfolio_optimizer = PortfolioOptimizerTool()
risk_assessor = RiskAssessorTool()
weight_allocator = WeightAllocatorTool()