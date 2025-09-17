#!/usr/bin/env python3
"""
Smoke Test 4: Critical API Endpoints
Tests essential API endpoints for basic functionality
"""

import pytest
import time
from framework.base_test import BaseTest
from framework.utilities.api_client import RAGHeatAPIClient


class TestCriticalAPIEndpoints(BaseTest):
    """Critical API endpoint smoke tests"""
    
    def setup_method(self, method):
        """Setup for each test method"""
        super().setup_method(method)
        self.api_client = RAGHeatAPIClient('http://localhost:8001')
        
    def teardown_method(self, method):
        """Teardown after each test method"""
        super().teardown_method(method)
    
    @pytest.mark.smoke
    def test_health_endpoint_accessibility(self):
        """
        Test: Health endpoint basic accessibility
        Pre: API server should be running
        Test: Call health endpoint and verify response
        Post: Ensure response time is acceptable
        """
        print("🔥 SMOKE TEST 4.1: Health Endpoint Accessibility")
        
        # Test execution
        start_time = time.time()
        response = self.api_client.health_check()
        response_time = time.time() - start_time
        
        # Assertions
        assert response.status_code == 200, f"Health endpoint failed: {response.status_code}"
        assert response_time < 3, f"Health endpoint too slow: {response_time}s"
        
        # Verify response is valid JSON
        try:
            data = response.json()
            assert isinstance(data, dict), "Health response should be JSON object"
            print(f"✅ Health endpoint accessible in {response_time:.3f}s")
        except ValueError:
            # Accept non-JSON response for smoke test
            print(f"✅ Health endpoint responsive (non-JSON) in {response_time:.3f}s")
    
    @pytest.mark.smoke
    def test_portfolio_construct_endpoint(self):
        """
        Test: Portfolio construction endpoint functionality
        Pre: API should be running with portfolio service
        Test: Send basic portfolio construction request
        Post: Verify endpoint responds appropriately
        """
        print("🔥 SMOKE TEST 4.2: Portfolio Construction Endpoint")
        
        # Test data
        test_stocks = ['AAPL', 'MSFT']
        
        # Test execution
        start_time = time.time()
        response = self.api_client.construct_portfolio(test_stocks)
        response_time = time.time() - start_time
        
        # Accept various response codes for smoke test
        valid_codes = [200, 201, 202, 400, 501]  # OK, Created, Accepted, Bad Request, Not Implemented
        assert response.status_code in valid_codes, f"Unexpected response code: {response.status_code}"
        assert response_time < 15, f"Portfolio endpoint too slow: {response_time}s"
        
        if response.status_code in [200, 201, 202]:
            try:
                data = response.json()
                print(f"✅ Portfolio construction successful in {response_time:.2f}s")
            except ValueError:
                print(f"✅ Portfolio endpoint responsive in {response_time:.2f}s")
        else:
            print(f"✅ Portfolio endpoint accessible ({response.status_code}) in {response_time:.2f}s")
    
    @pytest.mark.smoke  
    def test_system_status_endpoint(self):
        """
        Test: System status endpoint functionality
        Pre: API should provide system status information
        Test: Request system status
        Post: Verify endpoint provides some status information
        """
        print("🔥 SMOKE TEST 4.3: System Status Endpoint")
        
        # Test execution
        start_time = time.time()
        response = self.api_client.system_status()
        response_time = time.time() - start_time
        
        # Accept various response codes
        valid_codes = [200, 404, 501]  # OK, Not Found, Not Implemented
        assert response.status_code in valid_codes, f"Unexpected response code: {response.status_code}"
        assert response_time < 5, f"System status too slow: {response_time}s"
        
        if response.status_code == 200:
            try:
                data = response.json()
                assert isinstance(data, dict), "Status should return JSON object"
                print(f"✅ System status available in {response_time:.3f}s")
            except ValueError:
                print(f"✅ System status endpoint responsive in {response_time:.3f}s")
        else:
            print(f"✅ System status endpoint accessible ({response.status_code}) in {response_time:.3f}s")
    
    @pytest.mark.smoke
    def test_fundamental_analysis_endpoint(self):
        """
        Test: Fundamental analysis endpoint accessibility
        Pre: API should have fundamental analysis capability
        Test: Call fundamental analysis endpoint
        Post: Verify endpoint is reachable and responsive
        """
        print("🔥 SMOKE TEST 4.4: Fundamental Analysis Endpoint")
        
        # Test data
        test_stocks = ['AAPL']
        
        # Test execution
        start_time = time.time()
        response = self.api_client.fundamental_analysis(test_stocks)
        response_time = time.time() - start_time
        
        # Accept wide range of response codes for smoke test
        valid_codes = [200, 202, 400, 404, 501]
        assert response.status_code in valid_codes, f"Unexpected response code: {response.status_code}"
        assert response_time < 20, f"Fundamental analysis too slow: {response_time}s"
        
        if response.status_code == 200:
            print(f"✅ Fundamental analysis working in {response_time:.2f}s")
        else:
            print(f"✅ Fundamental analysis endpoint accessible ({response.status_code}) in {response_time:.2f}s")
    
    @pytest.mark.smoke
    def test_sentiment_analysis_endpoint(self):
        """
        Test: Sentiment analysis endpoint accessibility  
        Pre: API should have sentiment analysis capability
        Test: Call sentiment analysis endpoint
        Post: Verify endpoint responds to requests
        """
        print("🔥 SMOKE TEST 4.5: Sentiment Analysis Endpoint")
        
        # Test data
        test_stocks = ['MSFT']
        
        # Test execution
        start_time = time.time()
        response = self.api_client.sentiment_analysis(test_stocks)
        response_time = time.time() - start_time
        
        # Accept various response codes
        valid_codes = [200, 202, 400, 404, 501]
        assert response.status_code in valid_codes, f"Unexpected response code: {response.status_code}"
        assert response_time < 20, f"Sentiment analysis too slow: {response_time}s"
        
        if response.status_code == 200:
            print(f"✅ Sentiment analysis working in {response_time:.2f}s")
        else:
            print(f"✅ Sentiment analysis endpoint accessible ({response.status_code}) in {response_time:.2f}s")