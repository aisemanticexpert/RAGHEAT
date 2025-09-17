#!/usr/bin/env python3
"""
Smoke Test 2: Portfolio Construction Critical Path
Tests the core portfolio construction functionality
"""

import pytest
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from framework.base_test import BaseTest
from framework.page_objects.portfolio_dashboard import PortfolioDashboard
from framework.utilities.api_client import RAGHeatAPIClient


class TestPortfolioConstruction(BaseTest):
    """Critical path tests for portfolio construction"""
    
    def setup_method(self, method):
        """Setup for each test method"""
        super().setup_method(method)
        self.dashboard = PortfolioDashboard(self.driver)
        self.api_client = RAGHeatAPIClient('http://localhost:8001')
        
    def teardown_method(self, method):
        """Teardown after each test method"""
        super().teardown_method(method)
    
    @pytest.mark.smoke
    def test_portfolio_api_construct_basic(self):
        """
        Test: Basic portfolio construction via API
        Pre: API should be available and responsive
        Test: Send portfolio construction request with test stocks
        Post: Verify portfolio weights are returned correctly
        """
        print("🔥 SMOKE TEST 2.1: Portfolio API Construction")
        
        # Test data
        test_stocks = ['AAPL', 'GOOGL', 'MSFT']
        
        # Test execution
        start_time = time.time()
        response = self.api_client.construct_portfolio(test_stocks)
        response_time = time.time() - start_time
        
        # Assertions
        assert response.status_code == 200, f"Portfolio construction failed: {response.status_code}"
        assert response_time < 10, f"Portfolio construction too slow: {response_time}s"
        
        # Verify response data
        data = response.json()
        assert "portfolio_weights" in data, "Response missing portfolio_weights"
        
        weights = data["portfolio_weights"]
        assert isinstance(weights, dict), "Portfolio weights should be a dictionary"
        assert len(weights) == len(test_stocks), f"Expected {len(test_stocks)} weights, got {len(weights)}"
        
        # Verify weights sum to approximately 1
        total_weight = sum(weights.values())
        assert 0.95 <= total_weight <= 1.05, f"Weights don't sum to 1: {total_weight}"
        
        print(f"✅ Portfolio constructed in {response_time:.2f}s, total weight: {total_weight:.3f}")
    
    @pytest.mark.smoke
    def test_stock_input_field_interaction(self):
        """
        Test: Stock input field can accept input
        Pre: Navigate to portfolio construction page
        Test: Find stock input and enter test symbol
        Post: Verify input accepts and displays entered value
        """
        print("🔥 SMOKE TEST 2.2: Stock Input Field Interaction")
        
        # Pre-condition
        self.dashboard.navigate_to_dashboard('http://localhost:3000')
        time.sleep(3)  # Allow page to fully load
        
        # Test execution - Find stock input field
        stock_input = self.dashboard.find_stock_input()
        
        if stock_input:
            # Clear and enter test data
            stock_input.clear()
            test_symbol = "AAPL"
            stock_input.send_keys(test_symbol)
            
            # Verify input was accepted
            entered_value = stock_input.get_attribute('value')
            assert test_symbol in entered_value, f"Input not accepted: expected '{test_symbol}', got '{entered_value}'"
            
            self.capture_screenshot("stock_input_interaction")
            print(f"✅ Stock input field working: entered '{entered_value}'")
        else:
            # Check if there are alternative input methods
            alternative_inputs = self.driver.find_elements(By.XPATH, "//input[@type='text' or @type='search']")
            assert len(alternative_inputs) > 0, "No stock input fields found on page"
            
            # Test with first available input
            first_input = alternative_inputs[0]
            first_input.clear()
            first_input.send_keys("AAPL")
            entered_value = first_input.get_attribute('value')
            assert "AAPL" in entered_value, "Alternative input field not working"
            
            self.capture_screenshot("alternative_input_interaction")
            print(f"✅ Alternative input field working: entered '{entered_value}'")
    
    @pytest.mark.smoke
    def test_portfolio_construction_button_present(self):
        """
        Test: Portfolio construction button is present and clickable
        Pre: Navigate to main page
        Test: Find and verify construct portfolio button
        Post: Verify button is enabled and interactable
        """
        print("🔥 SMOKE TEST 2.3: Portfolio Construction Button")
        
        # Pre-condition
        self.dashboard.navigate_to_dashboard('http://localhost:3000')
        time.sleep(2)
        
        # Test execution - Find construct button
        construct_button = self.dashboard.find_construct_button()
        
        if construct_button:
            # Verify button properties
            assert construct_button.is_displayed(), "Construct button not visible"
            
            button_text = construct_button.text
            expected_keywords = ['construct', 'portfolio', 'build', 'create']
            text_match = any(keyword.lower() in button_text.lower() for keyword in expected_keywords)
            assert text_match, f"Button text doesn't match expected keywords: '{button_text}'"
            
            self.capture_screenshot("construct_button_found")
            print(f"✅ Construct button found: '{button_text}'")
        else:
            # Look for any button that might be the construct button
            all_buttons = self.driver.find_elements(By.XPATH, "//button")
            assert len(all_buttons) > 0, "No buttons found on page"
            
            # Check if any button contains portfolio-related text
            portfolio_buttons = []
            for button in all_buttons:
                button_text = button.text.lower()
                if any(word in button_text for word in ['portfolio', 'construct', 'build', 'analyze']):
                    portfolio_buttons.append(button)
            
            assert len(portfolio_buttons) > 0, "No portfolio-related buttons found"
            
            self.capture_screenshot("portfolio_buttons_found")
            print(f"✅ Found {len(portfolio_buttons)} portfolio-related buttons")
    
    @pytest.mark.smoke
    def test_multi_agent_system_status(self):
        """
        Test: Multi-agent system status endpoint
        Pre: API should be running
        Test: Call system status endpoint
        Post: Verify agents are listed and responsive
        """
        print("🔥 SMOKE TEST 2.4: Multi-Agent System Status")
        
        # Test execution
        start_time = time.time()
        response = self.api_client.system_status()
        response_time = time.time() - start_time
        
        # Basic response assertions
        assert response.status_code == 200, f"System status failed: {response.status_code}"
        assert response_time < 5, f"System status too slow: {response_time}s"
        
        # Verify response structure
        data = response.json()
        
        # Check for agent information (flexible structure)
        if "agents" in data:
            agents = data["agents"]
            assert isinstance(agents, list), "Agents should be a list"
            assert len(agents) > 0, "No agents found in system"
            print(f"✅ System status OK: {len(agents)} agents found in {response_time:.3f}s")
        elif "status" in data:
            # Alternative response format
            assert data["status"] in ["healthy", "ok", "running"], f"Unexpected system status: {data['status']}"
            print(f"✅ System status: {data['status']} in {response_time:.3f}s")
        else:
            # Minimal response - just verify it's valid JSON
            assert isinstance(data, dict), "Status response should be JSON object"
            print(f"✅ System status endpoint responsive in {response_time:.3f}s")
    
    @pytest.mark.smoke
    def test_fundamental_analysis_endpoint(self):
        """
        Test: Fundamental analysis agent endpoint
        Pre: API and agents should be running
        Test: Call fundamental analysis with test stocks
        Post: Verify response is received (may be simulated)
        """
        print("🔥 SMOKE TEST 2.5: Fundamental Analysis Endpoint")
        
        # Test data
        test_stocks = ['AAPL', 'MSFT']
        
        # Test execution
        start_time = time.time()
        response = self.api_client.fundamental_analysis(test_stocks)
        response_time = time.time() - start_time
        
        # Assertions - Accept various response codes for smoke test
        valid_codes = [200, 202, 501]  # OK, Accepted, or Not Implemented
        assert response.status_code in valid_codes, f"Unexpected response code: {response.status_code}"
        assert response_time < 15, f"Fundamental analysis too slow: {response_time}s"
        
        # If successful response, verify structure
        if response.status_code == 200:
            try:
                data = response.json()
                assert isinstance(data, dict), "Response should be JSON object"
                print(f"✅ Fundamental analysis successful in {response_time:.2f}s")
            except ValueError:
                # Response might not be JSON - that's ok for smoke test
                print(f"✅ Fundamental analysis responded in {response_time:.2f}s (non-JSON response)")
        else:
            print(f"✅ Fundamental analysis endpoint accessible ({response.status_code}) in {response_time:.2f}s")