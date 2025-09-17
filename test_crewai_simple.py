"""
Simple test for CrewAI system
============================

Test the CrewAI implementation with absolute imports.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, '/Users/rajeshgupta/PycharmProjects/ragheat-poc')

def test_basic_functionality():
    """Test basic functionality without relative imports."""
    print("Testing basic CrewAI functionality...")
    
    try:
        # Test tools directly
        from crewai.tools.fundamental_tools import FundamentalReportPull
        
        tool = FundamentalReportPull()
        result = tool._run(ticker="AAPL")
        
        print(f"✓ Tool test successful: {result['ticker']}")
        
        # Test simplified analysis
        if "company_name" in result:
            print(f"✓ Got company data for {result['company_name']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_crewai_import():
    """Test CrewAI framework import."""
    try:
        import crewai
        from crewai import Agent, Task, Crew
        print("✓ CrewAI framework imported successfully")
        return True
    except Exception as e:
        print(f"✗ CrewAI import failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("="*50)
    print("Simple CrewAI Test")
    print("="*50)
    
    tests = [
        test_crewai_import,
        test_basic_functionality
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ Basic functionality working!")
        print("\nTo use the full system:")
        print("1. cd crewai")
        print("2. python -c \"from crews.portfolio_crew import PortfolioConstructionCrew; print('Success!')\"")
    
    return 0 if passed == len(tests) else 1

if __name__ == "__main__":
    sys.exit(main())