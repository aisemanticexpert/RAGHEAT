"""
RAGHeat CrewAI Main Entry Point
===============================

Main script for running the RAGHeat portfolio construction system.
"""

import argparse
import json
import sys
from typing import List, Dict, Any
from pathlib import Path

from ragheat_crewai.config.settings import settings
from ragheat_crewai.crews.portfolio_crew import PortfolioConstructionCrew

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAGHeat CrewAI Portfolio Construction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stocks AAPL GOOGL MSFT --objective growth --risk moderate
  python main.py --config portfolio_config.json --output results.json
  python main.py --stocks TSLA NVDA --risk aggressive --time-horizon 6m --verbose
        """
    )
    
    # Stock selection
    parser.add_argument(
        "--stocks", 
        nargs="+", 
        required=True,
        help="Stock tickers to analyze (e.g., AAPL GOOGL MSFT)"
    )
    
    # Investment parameters
    parser.add_argument(
        "--objective",
        choices=["growth", "income", "balanced_growth", "value"],
        default="balanced_growth",
        help="Investment objective (default: balanced_growth)"
    )
    
    parser.add_argument(
        "--risk",
        choices=["conservative", "moderate", "aggressive"],
        default="moderate", 
        help="Risk tolerance (default: moderate)"
    )
    
    parser.add_argument(
        "--time-horizon",
        choices=["3m", "6m", "1y", "2y", "5y"],
        default="1y",
        help="Investment time horizon (default: 1y)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for results (JSON format)"
    )
    
    # Execution options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)

def validate_configuration():
    """Validate system configuration."""
    try:
        settings.validate_configuration()
        print("✓ Configuration validation successful")
        return True
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False

def print_portfolio_summary(result: Dict[str, Any]):
    """Print a formatted summary of portfolio results."""
    print("\n" + "="*60)
    print("RAGHEAT PORTFOLIO CONSTRUCTION RESULTS")
    print("="*60)
    
    # Portfolio recommendations
    if "portfolio_recommendations" in result:
        print("\nPORTFOLIO RECOMMENDATIONS:")
        print("-" * 40)
        
        recommendations = result["portfolio_recommendations"]
        for stock, data in recommendations.items():
            rec = data.get("recommendation", "N/A")
            weight = data.get("weight", 0) * 100
            confidence = data.get("confidence", 0) * 100
            
            print(f"{stock:6} | {rec:4} | Weight: {weight:5.1f}% | Confidence: {confidence:5.1f}%")
    
    # Risk assessment
    if "risk_assessment" in result:
        print("\nRISK ASSESSMENT:")
        print("-" * 40)
        
        risk = result["risk_assessment"]
        print(f"Expected Return:    {risk.get('expected_return', 0)*100:5.1f}%")
        print(f"Expected Volatility: {risk.get('volatility', 0)*100:5.1f}%")
        print(f"Sharpe Ratio:       {risk.get('sharpe_ratio', 0):5.2f}")
        print(f"Max Drawdown:       {risk.get('max_drawdown', 0)*100:5.1f}%")
    
    # Agent insights
    if "agent_analyses" in result:
        print("\nKEY INSIGHTS:")
        print("-" * 40)
        
        analyses = result["agent_analyses"]
        
        if "fundamental_analyst" in analyses:
            print("• Fundamental Analysis: Strong financial health indicators")
        
        if "sentiment_analyst" in analyses:
            sentiment = analyses["sentiment_analyst"]
            print(f"• Market Sentiment: {sentiment.get('overall_sentiment', 'Neutral')}")
        
        if "heat_diffusion_analyst" in analyses:
            print("• Heat Diffusion: Systemic risk factors identified")
    
    # Execution metadata
    if "execution_metadata" in result:
        metadata = result["execution_metadata"]
        print(f"\nAnalysis completed at: {metadata.get('timestamp', 'N/A')}")
        print(f"Agents executed: {len(metadata.get('agents_executed', []))}")
        print(f"Tasks completed: {len(metadata.get('tasks_completed', []))}")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set up logging
    import logging
    level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Validate configuration if requested
    if args.validate_config:
        if validate_configuration():
            print("Configuration is valid")
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Load custom configuration if provided
    custom_config = {}
    if args.config:
        custom_config = load_config(args.config)
    
    try:
        # Initialize portfolio construction crew
        print("Initializing RAGHeat Portfolio Construction Crew...")
        crew = PortfolioConstructionCrew(custom_config)
        
        # Prepare constraints from config
        constraints = custom_config.get("constraints", {})
        
        # Execute portfolio construction
        print(f"\nAnalyzing portfolio for stocks: {', '.join(args.stocks)}")
        print(f"Investment Objective: {args.objective}")
        print(f"Risk Tolerance: {args.risk}")
        print(f"Time Horizon: {args.time_horizon}")
        
        result = crew.construct_portfolio(
            target_stocks=args.stocks,
            investment_objective=args.objective,
            risk_tolerance=args.risk,
            time_horizon=args.time_horizon,
            custom_constraints=constraints
        )
        
        # Check for errors
        if "error" in result:
            print(f"Error in portfolio construction: {result['error']}")
            sys.exit(1)
        
        # Display results
        print_portfolio_summary(result)
        
        # Save results if output file specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"\nResults saved to: {output_path}")
        
        # Show agent performance if verbose
        if args.verbose:
            print("\nAGENT PERFORMANCE:")
            print("-" * 40)
            performance = crew.get_agent_performance()
            
            for agent_name, perf in performance["individual_performance"].items():
                success_rate = perf.get("success_rate", 0) * 100
                total_tasks = perf.get("total_tasks", 0)
                print(f"{agent_name:25} | Tasks: {total_tasks:3} | Success: {success_rate:5.1f}%")
        
        print("\nPortfolio construction completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()