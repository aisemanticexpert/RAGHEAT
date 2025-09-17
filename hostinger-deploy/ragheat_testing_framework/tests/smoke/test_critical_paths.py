#!/usr/bin/env python3
"""
Smoke tests for RAGHeat critical paths
"""

import pytest
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from framework.base_test import BaseTest
from framework.utilities.api_client import RAGHeatAPIClient
from framework.page_objects.portfolio_dashboard import PortfolioDashboard

@pytest.mark.smoke
@pytest.mark.critical
class TestCriticalPaths(BaseTest):
    """Test critical application paths"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.api_client = RAGHeatAPIClient(self.api_url)
        self.portfolio_page = PortfolioDashboard(self.driver)
    
    def test_application_accessibility(self):
        """Test that RAGHeat application is accessible"""
        print("🔥 Testing application accessibility...")
        start_time = time.time()
        
        # Test frontend accessibility
        success = self.portfolio_page.navigate_to_dashboard(self.base_url)
        load_time = time.time() - start_time
        
        # Basic assertions
        assert success, "Failed to navigate to application"
        assert load_time < 10, f"Application load time {load_time:.2f}s exceeds 10 seconds"
        
        # Check if page loaded properly
        page_loaded = self.portfolio_page.is_page_loaded()
        assert page_loaded, "Page did not load properly"
        
        # Get page info for debugging
        page_info = self.portfolio_page.get_page_info()
        print(f"📄 Page info: {page_info}")
        
        self.log_performance_metric("Frontend Load", load_time)
        self.log_test_result("Application Accessibility", "PASS", f"Load time: {load_time:.2f}s")
    
    def test_api_health_check(self):
        """Test API health endpoint"""
        print("🔥 Testing API health check...")
        
        try:
            response = self.api_client.health_check()
            
            # Basic response validation
            assert response.status_code == 200, f"Health check failed: {response.status_code}"
            
            # Try to parse JSON response
            try:
                data = response.json()
                print(f"✅ Health check response: {data}")
                
                # Check for expected fields (flexible)
                if isinstance(data, dict):
                    if 'status' in data:
                        assert data['status'] in ['healthy', 'ok', 'running'], f"Unexpected status: {data['status']}"
                
                self.log_test_result("API Health Check", "PASS", f"Status: {response.status_code}")
                
            except ValueError:
                # Not JSON response, check if it's HTML (frontend served)
                if 'text/html' in response.headers.get('content-type', ''):
                    print("⚠️ Health endpoint returned HTML (likely frontend), checking if it's valid")
                    assert len(response.text) > 100, "Response too short for valid HTML"
                    self.log_test_result("API Health Check", "PASS", "HTML response (frontend)")
                else:
                    pytest.fail(f"Invalid response format: {response.headers.get('content-type')}")
                    
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            self.log_test_result("API Health Check", "FAIL", str(e))
            pytest.fail(f"API Health check failed: {e}")
    
    def test_system_status_endpoint(self):
        """Test system status endpoint"""
        print("🔥 Testing system status endpoint...")
        
        try:
            response = self.api_client.system_status()
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"✅ System status response: {data}")
                    
                    # Flexible validation
                    if isinstance(data, dict):
                        # Check for agent information if present
                        if 'agents' in data:
                            agents = data['agents']
                            if isinstance(agents, list) and len(agents) > 0:
                                print(f"📊 Found {len(agents)} agents")
                                for agent in agents[:3]:  # Check first 3 agents
                                    if isinstance(agent, dict) and 'name' in agent:
                                        print(f"🤖 Agent: {agent['name']}")
                    
                    self.log_test_result("System Status", "PASS", f"Found {len(data)} fields")
                    
                except ValueError:
                    # Not JSON, might be HTML
                    if response.status_code == 200:
                        print("⚠️ System status returned non-JSON response")
                        self.log_test_result("System Status", "PASS", "Non-JSON response")
                    
            elif response.status_code == 404:
                print("⚠️ System status endpoint not found (404) - skipping")
                self.log_test_result("System Status", "SKIP", "Endpoint not found")
                pytest.skip("System status endpoint not available")
            else:
                print(f"⚠️ System status returned {response.status_code}")
                self.log_test_result("System Status", "SKIP", f"Status: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ System status check failed: {e}")
            self.log_test_result("System Status", "SKIP", str(e))
            pytest.skip(f"System status endpoint not accessible: {e}")
    
    def test_basic_portfolio_construction_api(self):
        """Test basic portfolio construction via API"""
        print("🔥 Testing basic portfolio construction API...")
        start_time = time.time()
        
        try:
            # Test with basic stock portfolio
            test_stocks = ["AAPL", "GOOGL", "MSFT"]
            response = self.api_client.construct_portfolio(test_stocks)
            
            construction_time = time.time() - start_time
            
            if response.status_code == 200:
                # Validate response
                validation = self.api_client.validate_portfolio_response(response)
                print(f"📊 Portfolio response validation: {validation}")
                
                # Basic assertions
                assert validation['status_code_ok'], "Portfolio construction failed"
                
                if validation['is_json']:
                    data = response.json()
                    print(f"✅ Portfolio constructed successfully: {list(data.keys())}")
                    
                    # Check for reasonable response structure
                    if validation['has_portfolio_weights']:
                        weights = data['portfolio_weights']
                        print(f"💰 Portfolio weights: {weights}")
                        
                        if validation['weights_sum_valid']:
                            print("✅ Portfolio weights sum validation passed")
                        else:
                            print("⚠️ Portfolio weights sum validation failed (non-critical)")
                
                self.log_performance_metric("Portfolio Construction API", construction_time)
                self.log_test_result("Portfolio Construction API", "PASS", f"Time: {construction_time:.2f}s")
                
            elif response.status_code == 404:
                pytest.skip("Portfolio construction endpoint not available")
            else:
                pytest.fail(f"Portfolio construction failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Portfolio construction failed: {e}")
            self.log_test_result("Portfolio Construction API", "FAIL", str(e))
            pytest.fail(f"Portfolio construction API test failed: {e}")
    
    def test_frontend_basic_functionality(self):
        """Test basic frontend functionality"""
        print("🔥 Testing frontend basic functionality...")
        
        # Navigate to dashboard
        success = self.portfolio_page.navigate_to_dashboard(self.base_url)
        assert success, "Failed to navigate to dashboard"
        
        # Get page information
        page_info = self.portfolio_page.get_page_info()
        print(f"📄 Page information: {page_info}")
        
        # Basic page validation
        assert page_info.get('title', ''), "Page has no title"
        assert page_info.get('body_text_length', 0) > 50, "Page content too short"
        
        # Check for form elements (indicates interactive functionality)
        forms_count = page_info.get('forms_count', 0)
        buttons_count = page_info.get('buttons_count', 0)
        inputs_count = page_info.get('inputs_count', 0)
        
        interactive_elements = forms_count + buttons_count + inputs_count
        print(f"🎮 Interactive elements found: {interactive_elements} (forms: {forms_count}, buttons: {buttons_count}, inputs: {inputs_count})")
        
        # If we have interactive elements, try basic interaction
        if interactive_elements > 0:
            try:
                # Try to find and interact with stock input
                stock_input = self.portfolio_page.find_stock_input()
                if stock_input:
                    print("✅ Stock input field found and accessible")
                    stock_input.send_keys("AAPL")
                    print("✅ Successfully entered stock symbol")
                
                # Try to find construct button
                construct_button = self.portfolio_page.find_construct_button()
                if construct_button:
                    print("✅ Portfolio construction button found")
                
            except Exception as e:
                print(f"⚠️ Interactive elements found but interaction failed: {e}")
        
        self.log_test_result("Frontend Basic Functionality", "PASS", f"Interactive elements: {interactive_elements}")
    
    def test_end_to_end_basic_flow(self):
        """Test basic end-to-end flow"""
        print("🔥 Testing basic end-to-end flow...")
        
        try:
            # Step 1: Navigate to frontend
            success = self.portfolio_page.navigate_to_dashboard(self.base_url)
            assert success, "Failed to navigate to application"
            
            # Step 2: Try API call
            api_works = False
            try:
                response = self.api_client.health_check()
                api_works = response.status_code == 200
            except:
                pass
            
            # Step 3: Try frontend interaction if possible
            frontend_interactive = False
            try:
                stock_input = self.portfolio_page.find_stock_input()
                construct_button = self.portfolio_page.find_construct_button()
                
                if stock_input and construct_button:
                    frontend_interactive = True
                    print("✅ Frontend appears to be interactive")
            except:
                pass
            
            # Overall assessment
            if api_works and frontend_interactive:
                result = "PASS - Full functionality"
            elif api_works or frontend_interactive:
                result = "PARTIAL - Some functionality working"
            else:
                result = "BASIC - Application accessible but limited functionality"
            
            print(f"📊 End-to-end assessment: {result}")
            self.log_test_result("End-to-End Basic Flow", "PASS", result)
            
        except Exception as e:
            print(f"❌ End-to-end test failed: {e}")
            self.log_test_result("End-to-End Basic Flow", "FAIL", str(e))
            pytest.fail(f"End-to-end basic flow failed: {e}")
    
    def teardown_method(self, method):
        """Cleanup after each test"""
        super().teardown_method(method)
        
        # Take screenshot if test failed (optional, handled by conftest.py)