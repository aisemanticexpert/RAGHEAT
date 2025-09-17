#!/usr/bin/env python3
"""
Smoke Test 1: Application Startup Test
Critical path test to verify the application starts and loads correctly
"""

import pytest
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from framework.base_test import BaseTest
from framework.page_objects.portfolio_dashboard import PortfolioDashboard
from framework.utilities.api_client import RAGHeatAPIClient


class TestApplicationStartup(BaseTest):
    """Test suite for application startup verification"""
    
    def setup_method(self, method):
        """Setup for each test method"""
        super().setup_method(method)
        self.dashboard = PortfolioDashboard(self.driver)
        self.api_client = RAGHeatAPIClient('http://localhost:8001')
        
    def teardown_method(self, method):
        """Teardown after each test method"""
        super().teardown_method(method)
    
    @pytest.mark.smoke
    def test_frontend_application_loads(self):
        """
        Test: Verify frontend application loads successfully
        Pre: Application should be running on localhost:3000
        Test: Navigate to frontend and verify page loads
        Post: Capture screenshot and close browser
        """
        print("🔥 SMOKE TEST 1.1: Frontend Application Load")
        
        # Pre-condition check
        start_time = time.time()
        
        # Test execution
        success = self.dashboard.navigate_to_dashboard('http://localhost:3000')
        load_time = time.time() - start_time
        
        # Assertions
        assert success, "Frontend application failed to load"
        assert load_time < 10, f"Application took too long to load: {load_time}s"
        
        # Verify page title contains expected text
        page_title = self.driver.title
        assert "RAGHeat" in page_title or "Portfolio" in page_title, f"Unexpected page title: {page_title}"
        
        # Take screenshot for verification
        self.capture_screenshot("frontend_application_loaded")
        
        print(f"✅ Frontend loaded in {load_time:.2f}s with title: {page_title}")
    
    @pytest.mark.smoke  
    def test_api_health_check(self):
        """
        Test: Verify API health endpoint responds
        Pre: Backend should be running on localhost:8001
        Test: Call health endpoint and verify 200 response
        Post: Verify response contains expected data
        """
        print("🔥 SMOKE TEST 1.2: API Health Check")
        
        # Test execution
        start_time = time.time()
        response = self.api_client.health_check()
        response_time = time.time() - start_time
        
        # Assertions
        assert response.status_code == 200, f"API health check failed: {response.status_code}"
        assert response_time < 5, f"API response too slow: {response_time}s"
        
        # Verify response data
        data = response.json()
        assert "status" in data, "Health response missing status field"
        assert data["status"] in ["healthy", "ok"], f"Unexpected health status: {data['status']}"
        
        print(f"✅ API health check passed in {response_time:.3f}s")
    
    @pytest.mark.smoke
    def test_main_page_components_present(self):
        """
        Test: Verify main page UI components are present
        Pre: Navigate to main application page
        Test: Check for presence of key UI elements
        Post: Capture screenshot of loaded page
        """
        print("🔥 SMOKE TEST 1.3: Main Page Components")
        
        # Pre-condition
        self.dashboard.navigate_to_dashboard('http://localhost:3000')
        time.sleep(2)  # Allow page to fully load
        
        # Test execution - Check for main container
        main_container = self.driver.find_elements(By.XPATH, "//div[@id='root']")
        assert len(main_container) > 0, "Main container (#root) not found"
        
        # Check for header/navigation
        header_elements = self.driver.find_elements(By.XPATH, "//header | //div[contains(@class, 'header')]")
        nav_elements = self.driver.find_elements(By.XPATH, "//nav | //div[contains(@class, 'nav')]")
        
        # At least one navigation structure should exist
        assert len(header_elements) > 0 or len(nav_elements) > 0, "No header or navigation elements found"
        
        # Check for any interactive buttons
        buttons = self.driver.find_elements(By.XPATH, "//button")
        assert len(buttons) > 0, "No interactive buttons found on main page"
        
        # Capture screenshot
        self.capture_screenshot("main_page_components")
        
        print(f"✅ Found {len(buttons)} buttons, header/nav elements present")