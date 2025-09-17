#!/usr/bin/env python3
"""
Base test class for RAGHeat testing framework
"""

import os
import time
import pytest
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class BaseTest:
    """Base class for all RAGHeat tests - Each test gets its own browser instance"""
    
    @classmethod 
    def setup_class(cls):
        """Setup test environment (shared across all tests in class)"""
        cls.base_url = os.getenv("RAGHEAT_FRONTEND_URL", "http://localhost:3000")
        cls.api_url = os.getenv("RAGHEAT_API_URL", "http://localhost:8000") 
        cls.setup_api_client()
        cls.test_start_time = time.time()
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
    
    def setup_webdriver(self):
        """Setup Chrome WebDriver for individual test (called per test)"""
        chrome_options = Options()
        
        # Add headless mode (can be disabled by setting environment variable)
        if os.getenv("HEADLESS", "true").lower() == "true":
            chrome_options.add_argument("--headless")
        
        # Standard Chrome options for testing
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        # Enable images and javascript for better testing
        # chrome_options.add_argument("--disable-images")
        # chrome_options.add_argument("--disable-javascript")
        
        # Create service with automatically managed ChromeDriver
        service = Service(ChromeDriverManager().install())
        
        try:
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            self.wait = WebDriverWait(self.driver, 15)
            print(f"✅ Chrome WebDriver initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Chrome WebDriver: {e}")
            raise
    
    @classmethod
    def setup_api_client(cls):
        """Setup API client"""
        cls.session = requests.Session()
        cls.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RAGHeat-TestFramework/1.0',
            'Accept': 'application/json'
        })
        cls.session.timeout = 30
        
        # Test API connectivity
        try:
            response = cls.session.get(f"{cls.api_url}/health", timeout=5)
            print(f"✅ API client initialized - Health check: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠️ API connectivity issue: {e}")
    
    def take_screenshot(self, name):
        """Take screenshot on test failure"""
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/{name}_{timestamp}.png"
        try:
            self.driver.save_screenshot(filename)
            print(f"📸 Screenshot saved: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Failed to take screenshot: {e}")
            return None
    
    def log_performance_metric(self, operation, duration):
        """Log performance metrics"""
        with open('logs/performance.log', 'a') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | {operation} | {duration:.3f}s\n")
    
    def log_test_result(self, test_name, result, details=""):
        """Log test results"""
        with open('logs/test_results.log', 'a') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | {test_name} | {result} | {details}\n")
    
    def wait_for_element(self, locator, timeout=10):
        """Wait for element to be present"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located(locator)
            )
            return element
        except Exception as e:
            print(f"❌ Element not found: {locator}, Error: {e}")
            return None
    
    def wait_for_clickable_element(self, locator, timeout=10):
        """Wait for element to be clickable"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable(locator)
            )
            return element
        except Exception as e:
            print(f"❌ Element not clickable: {locator}, Error: {e}")
            return None
    
    @classmethod
    def teardown_class(cls):
        """Cleanup after tests"""
        if hasattr(cls, 'driver'):
            try:
                cls.driver.quit()
                print("✅ WebDriver closed successfully")
            except Exception as e:
                print(f"⚠️ Error closing WebDriver: {e}")
        
        if hasattr(cls, 'session'):
            try:
                cls.session.close()
                print("✅ API session closed successfully")
            except Exception as e:
                print(f"⚠️ Error closing API session: {e}")
        
        # Log total test time
        if hasattr(cls, 'test_start_time'):
            total_time = time.time() - cls.test_start_time
            print(f"📊 Total test execution time: {total_time:.2f}s")
    
    def setup_method(self, method):
        """Setup before each test method"""
        self.method_start_time = time.time()
        # Create new browser instance for this test
        self.setup_webdriver()
        print(f"🚀 Starting test: {method.__name__}")
    
    def teardown_method(self, method):
        """Cleanup after each test method"""
        test_name = method.__name__
        
        # Take screenshot before closing
        if hasattr(self, 'driver') and self.driver:
            try:
                self.capture_screenshot(f"{test_name}_completed")
            except:
                pass
        
        # Close the browser for this test
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
                print(f"🧹 Browser closed for test: {test_name}")
            except Exception as e:
                print(f"⚠️ Error closing browser for {test_name}: {e}")
        
        # Log performance
        if hasattr(self, 'method_start_time'):
            method_time = time.time() - self.method_start_time
            self.log_performance_metric(f"Test: {test_name}", method_time)
            print(f"⏱️ Test {test_name} completed in {method_time:.2f}s")
    
    def capture_screenshot(self, name):
        """Capture screenshot with improved naming"""
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/{name}_{timestamp}.png"
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.save_screenshot(filename)
                print(f"📸 Screenshot saved: {filename}")
                return filename
        except Exception as e:
            print(f"❌ Failed to take screenshot: {e}")
        return None
    
    def _get_test_name(self):
        """Get current test method name"""
        import inspect
        return inspect.stack()[2].function