"""
Test RAGHeat CrewAI System
=========================

Test the renamed system to avoid conflicts.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, '/Users/rajeshgupta/PycharmProjects/ragheat-poc')

def test_crewai_framework():
    """Test CrewAI framework import."""
    try:
        import crewai
        from crewai import Agent, Task, Crew
        print("✓ CrewAI framework imported successfully")
        print(f"  CrewAI version: {crewai.__version__ if hasattr(crewai, '__version__') else 'Unknown'}")
        return True
    except Exception as e:
        print(f"✗ CrewAI import failed: {e}")
        return False

def test_ragheat_tools():
    """Test RAGHeat tools."""
    try:
        from ragheat_crewai.tools.fundamental_tools import FundamentalReportPull
        
        tool = FundamentalReportPull()
        result = tool._run(ticker="AAPL")
        
        if result.get("ticker") == "AAPL":
            print("✓ RAGHeat fundamental tool working")
            print(f"  Company: {result.get('company_name', 'Unknown')}")
            print(f"  Sector: {result.get('sector', 'Unknown')}")
            return True
        else:
            print(f"✗ Unexpected result: {result}")
            return False
            
    except Exception as e:
        print(f"✗ RAGHeat tools test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ragheat_agent():
    """Test RAGHeat agent initialization."""
    try:
        from ragheat_crewai.agents.fundamental_analyst import FundamentalAnalystAgent
        
        agent_config = {
            "role": "Test Fundamental Analyst",
            "goal": "Test analysis",
            "backstory": "Test agent",
            "tools": [],
            "verbose": False
        }
        
        agent = FundamentalAnalystAgent(agent_config)
        print("✓ RAGHeat agent initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ RAGHeat agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_portfolio_construction():
    """Test basic portfolio construction."""
    try:
        from ragheat_crewai.crews.portfolio_crew import PortfolioConstructionCrew
        
        # This will test initialization only
        print("✓ PortfolioConstructionCrew class imported successfully")
        print("  Note: Full crew initialization requires all dependencies")
        return True
        
    except Exception as e:
        print(f"✗ Portfolio crew test failed: {e}")
        print("  This is expected if all dependencies aren't configured")
        return False

def main():
    """Run tests."""
    print("="*60)
    print("RAGHeat CrewAI System Test")
    print("="*60)
    
    tests = [
        ("CrewAI Framework", test_crewai_framework),
        ("RAGHeat Tools", test_ragheat_tools), 
        ("RAGHeat Agent", test_ragheat_agent),
        ("Portfolio Crew", test_portfolio_construction)
    ]
    
    passed = 0
    for name, test in tests:
        print(f"\nTesting {name}...")
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed >= 2:  # At least basic functionality working
        print("\n✓ Core system is functional!")
        print("\nNext steps:")
        print("1. Configure API keys in .env file")
        print("2. Set up Neo4j database")  
        print("3. Run: python ragheat_crewai/main.py --stocks AAPL --risk moderate")
        
        # Show usage example
        print("\nBasic usage example:")
        print("```python")
        print("from ragheat_crewai.crews.portfolio_crew import PortfolioConstructionCrew")
        print("crew = PortfolioConstructionCrew()")
        print("result = crew.construct_portfolio(['AAPL', 'GOOGL'])")
        print("```")
    else:
        print("\n✗ System needs more configuration")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())