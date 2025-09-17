"""
Test script for RAGHeat CrewAI System
=====================================

Simple test to verify the system can initialize and run basic operations.
"""

import sys
import os
import logging
from typing import List

# Add the crewai module to the path
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test configuration
        from config.settings import settings
        print("✓ Settings imported successfully")
        
        # Test agents
        from agents.fundamental_analyst import FundamentalAnalystAgent
        from agents.sentiment_analyst import SentimentAnalystAgent
        from agents.valuation_analyst import ValuationAnalystAgent
        print("✓ Agent classes imported successfully")
        
        # Test tools
        from tools.fundamental_tools import FundamentalReportPull, FinancialRatioCalculator
        from tools.sentiment_tools import NewsAggregator, SentimentAnalyzer
        print("✓ Tool classes imported successfully")
        
        # Test tasks
        from tasks.portfolio_tasks import ConstructKnowledgeGraphTask
        print("✓ Task classes imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_tool_functionality():
    """Test basic tool functionality."""
    print("\nTesting tool functionality...")
    
    try:
        from tools.fundamental_tools import FundamentalReportPull
        
        # Test fundamental tool
        tool = FundamentalReportPull()
        result = tool._run(ticker="AAPL", report_types=["10-K"], lookback_days=30)
        
        if "ticker" in result and result["ticker"] == "AAPL":
            print("✓ Fundamental analysis tool working")
            return True
        else:
            print("✗ Fundamental analysis tool returned unexpected result")
            return False
            
    except Exception as e:
        print(f"✗ Tool test failed: {e}")
        return False

def test_agent_initialization():
    """Test agent initialization."""
    print("\nTesting agent initialization...")
    
    try:
        from agents.fundamental_analyst import FundamentalAnalystAgent
        
        # Create test agent config
        agent_config = {
            "role": "Test Fundamental Analyst",
            "goal": "Test fundamental analysis",
            "backstory": "Test backstory",
            "tools": ["fundamental_report_pull"],
            "verbose": True,
            "allow_delegation": False,
            "max_iter": 3
        }
        
        # Initialize agent
        agent = FundamentalAnalystAgent(agent_config)
        
        if agent.agent_name == "Test Fundamental Analyst":
            print("✓ Agent initialized successfully")
            return True
        else:
            print("✗ Agent initialization failed")
            return False
            
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        return False

def test_simplified_portfolio_construction():
    """Test simplified portfolio construction without full crew."""
    print("\nTesting simplified portfolio construction...")
    
    try:
        from agents.fundamental_analyst import FundamentalAnalystAgent
        
        # Simple agent config
        agent_config = {
            "role": "Fundamental Analyst",
            "goal": "Analyze stock fundamentals",
            "backstory": "Financial analyst with expertise in fundamental analysis",
            "tools": ["fundamental_report_pull"],
            "verbose": False,
            "allow_delegation": False,
            "max_iter": 2
        }
        
        # Initialize agent
        agent = FundamentalAnalystAgent(agent_config)
        
        # Test analysis
        result = agent.analyze({
            "stocks": ["AAPL", "GOOGL"],
            "time_horizon": "1y"
        })
        
        if "analysis_type" in result and result["analysis_type"] == "fundamental":
            print("✓ Simplified portfolio analysis working")
            return True
        else:
            print("✗ Simplified analysis failed")
            return False
            
    except Exception as e:
        print(f"✗ Simplified analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("RAGHeat CrewAI System Test Suite")
    print("="*60)
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    tests = [
        test_imports,
        test_tool_functionality,
        test_agent_initialization,
        test_simplified_portfolio_construction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! System is ready.")
        print("\nNext steps:")
        print("1. Run: python main.py --stocks AAPL GOOGL --objective growth --risk moderate")
        print("2. Or import and use in Python:")
        print("   from crews.portfolio_crew import PortfolioConstructionCrew")
        print("   crew = PortfolioConstructionCrew()")
    else:
        print("✗ Some tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())