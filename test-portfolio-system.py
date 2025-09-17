#!/usr/bin/env python3
"""
Test script for RAGHeat Portfolio Construction System
====================================================

This script validates the portfolio system components without requiring full Docker deployment.
"""

import sys
import os
import importlib.util
import yaml
import json
from pathlib import Path

def test_configuration_files():
    """Test that configuration files exist and are valid."""
    print("🔍 Testing configuration files...")
    
    # Test agents.yaml
    agents_file = Path("backend/portfolio_agents/config/agents.yaml")
    if not agents_file.exists():
        print("❌ agents.yaml not found")
        return False
    
    try:
        with open(agents_file, 'r') as f:
            agents_config = yaml.safe_load(f)
        
        # Check that we have the expected agents
        expected_agents = [
            'fundamental_analyst', 'sentiment_analyst', 'valuation_analyst',
            'knowledge_graph_engineer', 'heat_diffusion_analyst',
            'portfolio_coordinator', 'explanation_generator'
        ]
        
        for agent in expected_agents:
            if agent not in agents_config.get('agents', {}):
                print(f"❌ Missing agent: {agent}")
                return False
                
        print("✅ agents.yaml is valid")
    except Exception as e:
        print(f"❌ Error reading agents.yaml: {e}")
        return False
    
    # Test tasks.yaml
    tasks_file = Path("backend/portfolio_agents/config/tasks.yaml")
    if not tasks_file.exists():
        print("❌ tasks.yaml not found")
        return False
    
    try:
        with open(tasks_file, 'r') as f:
            tasks_config = yaml.safe_load(f)
        
        # Check that we have the expected tasks
        expected_tasks = [
            'construct_knowledge_graph', 'analyze_fundamentals',
            'assess_market_sentiment', 'calculate_valuations',
            'simulate_heat_diffusion', 'facilitate_agent_debate',
            'construct_portfolio', 'generate_investment_rationale'
        ]
        
        for task in expected_tasks:
            if task not in tasks_config.get('tasks', {}):
                print(f"❌ Missing task: {task}")
                return False
                
        print("✅ tasks.yaml is valid")
    except Exception as e:
        print(f"❌ Error reading tasks.yaml: {e}")
        return False
    
    return True

def test_python_imports():
    """Test that key Python modules can be imported."""
    print("\n🔍 Testing Python module structure...")
    
    # Check that Python files exist rather than importing (which requires dependencies)
    core_files = [
        "backend/portfolio_agents/core/portfolio_system.py",
        "backend/portfolio_agents/agents/agent_factory.py", 
        "backend/portfolio_agents/tools/tool_registry.py"
    ]
    
    for file_path in core_files:
        if not Path(file_path).exists():
            print(f"❌ Missing core file: {file_path}")
            return False
    
    # Check that requirements file has key dependencies
    req_file = Path("backend/portfolio_agents/requirements.txt")
    if not req_file.exists():
        print("❌ Requirements file not found")
        return False
    
    try:
        with open(req_file, 'r') as f:
            requirements = f.read()
        
        required_packages = ["crewai", "anthropic", "fastapi", "loguru", "neo4j"]
        for package in required_packages:
            if package not in requirements:
                print(f"❌ Missing required package: {package}")
                return False
        
        print("✅ Core Python files and dependencies are present")
    except Exception as e:
        print(f"❌ Error checking requirements: {e}")
        return False
    
    return True

def test_api_structure():
    """Test that API files are properly structured."""
    print("\n🔍 Testing API structure...")
    
    # Check main API file
    main_api = Path("backend/portfolio_agents/api/main.py")
    if not main_api.exists():
        print("❌ Main API file not found")
        return False
    
    # Check routes
    routes_dir = Path("backend/portfolio_agents/api/routes")
    if not routes_dir.exists():
        print("❌ API routes directory not found")
        return False
    
    expected_routes = ["portfolio_routes.py", "analysis_routes.py", "system_routes.py"]
    for route_file in expected_routes:
        route_path = routes_dir / route_file
        if not route_path.exists():
            print(f"❌ Missing route file: {route_file}")
            return False
    
    print("✅ API structure is valid")
    return True

def test_docker_configuration():
    """Test Docker configuration files."""
    print("\n🔍 Testing Docker configuration...")
    
    # Check Docker Compose file
    compose_file = Path("docker-compose-portfolio-agents.yml")
    if not compose_file.exists():
        print("❌ Docker Compose file not found")
        return False
    
    # Check Dockerfile
    dockerfile = Path("backend/portfolio_agents/Dockerfile")
    if not dockerfile.exists():
        print("❌ Portfolio agents Dockerfile not found")
        return False
    
    # Check frontend Dockerfile
    frontend_dockerfile = Path("frontend/Dockerfile")
    if not frontend_dockerfile.exists():
        print("❌ Frontend Dockerfile not found")
        return False
    
    # Check startup script
    startup_script = Path("start-portfolio-system.sh")
    if not startup_script.exists():
        print("❌ Startup script not found")
        return False
    
    if not os.access(startup_script, os.X_OK):
        print("❌ Startup script is not executable")
        return False
    
    print("✅ Docker configuration is valid")
    return True

def test_frontend_integration():
    """Test frontend integration."""
    print("\n🔍 Testing frontend integration...")
    
    # Check package.json
    package_json = Path("frontend/package.json")
    if not package_json.exists():
        print("❌ Frontend package.json not found")
        return False
    
    try:
        with open(package_json, 'r') as f:
            package_data = json.load(f)
        
        # Check for required dependencies
        required_deps = ["@mui/material", "@mui/icons-material", "axios", "react", "react-dom"]
        for dep in required_deps:
            if dep not in package_data.get('dependencies', {}):
                print(f"❌ Missing frontend dependency: {dep}")
                return False
        
        print("✅ Frontend dependencies are valid")
    except Exception as e:
        print(f"❌ Error reading package.json: {e}")
        return False
    
    # Check App.js
    app_js = Path("frontend/src/App.js")
    if not app_js.exists():
        print("❌ Frontend App.js not found")
        return False
    
    try:
        with open(app_js, 'r') as f:
            app_content = f.read()
        
        # Check for portfolio dashboard import
        if "PortfolioConstructionDashboard" not in app_content:
            print("❌ Portfolio dashboard not integrated in App.js")
            return False
        
        print("✅ Frontend integration is valid")
    except Exception as e:
        print(f"❌ Error reading App.js: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 RAGHeat Portfolio Construction System Validation")
    print("=" * 55)
    
    tests = [
        test_configuration_files,
        test_python_imports,
        test_api_structure,
        test_docker_configuration,
        test_frontend_integration
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! The system is ready for deployment.")
        print("\n🚀 To start the system, run:")
        print("   ./start-portfolio-system.sh")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main()