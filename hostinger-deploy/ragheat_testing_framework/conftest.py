#!/usr/bin/env python3
"""
Pytest configuration and fixtures for RAGHeat testing
"""

import pytest
import os
import logging
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pytest_configure(config):
    """Configure pytest"""
    # Create directories
    os.makedirs('reports', exist_ok=True)
    os.makedirs('screenshots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Log test configuration
    logger.info("🚀 Starting RAGHeat test suite")
    logger.info(f"Frontend URL: {os.getenv('RAGHEAT_FRONTEND_URL', 'http://localhost:3000')}")
    logger.info(f"API URL: {os.getenv('RAGHEAT_API_URL', 'http://localhost:8000')}")
    logger.info(f"Headless mode: {os.getenv('HEADLESS', 'true')}")

def pytest_sessionstart(session):
    """Called after the Session object has been created"""
    print("\n" + "="*80)
    print("🚀 RAGHeat Testing Framework - Session Started")
    print("="*80)

def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished"""
    print("\n" + "="*80)
    print(f"🏁 RAGHeat Testing Framework - Session Finished (Exit: {exitstatus})")
    print("="*80)

@pytest.fixture(scope="session")
def driver():
    """Selenium WebDriver fixture"""
    print("🔧 Setting up WebDriver...")
    
    options = Options()
    
    # Headless mode
    if os.getenv("HEADLESS", "true").lower() == "true":
        options.add_argument("--headless")
        print("👤 Running in headless mode")
    else:
        print("🖥️ Running with visible browser")
    
    # Chrome options for testing
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-images")  # Faster loading
    options.add_argument("--disable-javascript")  # Will be enabled when needed
    
    # Performance options
    options.add_argument("--memory-pressure-off")
    options.add_argument("--max_old_space_size=4096")
    
    # Create service with automatically managed ChromeDriver
    service = Service(ChromeDriverManager().install())
    
    try:
        driver = webdriver.Chrome(service=service, options=options)
        driver.implicitly_wait(10)
        print("✅ WebDriver initialized successfully")
        yield driver
    except Exception as e:
        print(f"❌ Failed to initialize WebDriver: {e}")
        raise
    finally:
        try:
            driver.quit()
            print("✅ WebDriver closed successfully")
        except Exception as e:
            print(f"⚠️ Error closing WebDriver: {e}")

@pytest.fixture(scope="session")
def test_data():
    """Test data fixture"""
    return {
        'basic_stocks': ["AAPL", "GOOGL", "MSFT"],
        'extended_stocks': ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        'large_portfolio': ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "BRK.B"],
        'sector_tech': ["AAPL", "GOOGL", "MSFT", "NVDA", "AMD", "INTC"],
        'sector_finance': ["JPM", "BAC", "WFC", "GS", "C"],
        'invalid_stocks': ["INVALID123", "BADSTOCK", ""],
        'single_stock': ["AAPL"],
        'empty_portfolio': []
    }

@pytest.fixture(scope="session")
def api_endpoints():
    """API endpoints configuration"""
    base_url = os.getenv('RAGHEAT_API_URL', 'http://localhost:8000')
    return {
        'health': f"{base_url}/api/health",
        'system_status': f"{base_url}/api/system/status",
        'portfolio_construct': f"{base_url}/api/portfolio/construct",
        'fundamental_analysis': f"{base_url}/api/analysis/fundamental",
        'sentiment_analysis': f"{base_url}/api/analysis/sentiment",
        'technical_analysis': f"{base_url}/api/analysis/technical",
        'heat_diffusion': f"{base_url}/api/analysis/heat-diffusion",
        'analysis_tools': f"{base_url}/api/analysis/tools",
        'root': f"{base_url}/"
    }

def pytest_runtest_makereport(item, call):
    """Custom test reporting"""
    if call.when == "call":
        # Log test execution
        test_name = item.nodeid
        duration = call.duration
        
        if call.excinfo is None:
            # Test passed
            print(f"✅ {test_name} - PASSED ({duration:.2f}s)")
        else:
            # Test failed
            print(f"❌ {test_name} - FAILED ({duration:.2f}s)")
            
            # Take screenshot if it's a UI test and driver is available
            if hasattr(item.instance, 'driver') and hasattr(item.instance, 'take_screenshot'):
                try:
                    screenshot_name = f"{item.nodeid.replace('::', '_').replace('/', '_')}_failure"
                    item.instance.take_screenshot(screenshot_name)
                except Exception as e:
                    print(f"⚠️ Failed to take screenshot: {e}")

def pytest_html_report_title(report):
    """Custom HTML report title"""
    report.title = "RAGHeat Multi-Agent Portfolio System - Test Execution Report"

def pytest_html_results_summary(prefix, summary, postfix):
    """Custom HTML report summary"""
    prefix.extend([
        "<h2>🤖 RAGHeat Multi-Agent Portfolio Construction System</h2>",
        "<p><strong>Test Environment:</strong></p>",
        "<ul>",
        f"<li>Frontend URL: {os.getenv('RAGHEAT_FRONTEND_URL', 'http://localhost:3000')}</li>",
        f"<li>API URL: {os.getenv('RAGHEAT_API_URL', 'http://localhost:8000')}</li>",
        f"<li>Headless Mode: {os.getenv('HEADLESS', 'true')}</li>",
        f"<li>Test Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}</li>",
        "</ul>"
    ])

@pytest.fixture
def performance_tracker():
    """Performance tracking fixture"""
    start_time = time.time()
    metrics = {'start_time': start_time}
    
    yield metrics
    
    end_time = time.time()
    metrics['end_time'] = end_time
    metrics['duration'] = end_time - start_time
    
    print(f"⏱️ Test completed in {metrics['duration']:.2f}s")

# Hooks for test collection
def pytest_collection_modifyitems(config, items):
    """Modify test items during collection"""
    # Add markers based on test file location
    for item in items:
        # Add markers based on file path
        if "smoke" in str(item.fspath):
            item.add_marker(pytest.mark.smoke)
            item.add_marker(pytest.mark.critical)
        elif "sanity" in str(item.fspath):
            item.add_marker(pytest.mark.sanity)
        elif "regression" in str(item.fspath):
            item.add_marker(pytest.mark.regression)
        elif "api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

# Custom command line options
def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--env", action="store", default="local",
        help="Test environment: local, staging, production"
    )
    parser.addoption(
        "--browser", action="store", default="chrome",
        help="Browser to use: chrome, firefox"
    )
    parser.addoption(
        "--headless", action="store", default="true",
        help="Run browser in headless mode: true, false"
    )

@pytest.fixture(scope="session")
def test_config(request):
    """Test configuration based on command line options"""
    return {
        'environment': request.config.getoption("--env"),
        'browser': request.config.getoption("--browser"),
        'headless': request.config.getoption("--headless").lower() == "true"
    }