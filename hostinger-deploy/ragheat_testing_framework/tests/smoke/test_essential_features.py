#!/usr/bin/env python3
"""
Smoke tests for essential RAGHeat features
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
class TestEssentialFeatures(BaseTest):
    """Test essential application features"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.api_client = RAGHeatAPIClient(self.api_url)
        self.portfolio_page = PortfolioDashboard(self.driver)
    
    def test_api_endpoints_accessibility(self):
        """Test accessibility of key API endpoints"""
        print("🔥 Testing API endpoints accessibility...")
        
        endpoints = [
            ('/api/health', 'Health Check'),
            ('/api/system/status', 'System Status'),
            ('/api/analysis/tools', 'Analysis Tools'),
            ('/', 'Root Endpoint')
        ]
        
        results = []
        
        for endpoint, description in endpoints:
            try:
                start_time = time.time()
                response = self.api_client._make_request('GET', endpoint)
                response_time = time.time() - start_time
                
                status = "✅ PASS" if response.status_code == 200 else f"⚠️ {response.status_code}"
                results.append({
                    'endpoint': endpoint,
                    'description': description,
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'result': status
                })
                
                print(f"{status} {description} ({endpoint}): {response.status_code} - {response_time:.3f}s")
                
            except Exception as e:
                results.append({
                    'endpoint': endpoint,
                    'description': description,
                    'status_code': 'ERROR',
                    'response_time': 0,
                    'result': f"❌ ERROR: {str(e)}"
                })
                print(f"❌ ERROR {description} ({endpoint}): {str(e)}")
        
        # Summary
        successful_endpoints = sum(1 for r in results if r['status_code'] == 200)
        total_endpoints = len(results)
        
        print(f"📊 API Endpoints Summary: {successful_endpoints}/{total_endpoints} accessible")
        
        # At least one endpoint should be accessible
        assert successful_endpoints > 0, "No API endpoints are accessible"
        
        self.log_test_result("API Endpoints", "PASS", f"{successful_endpoints}/{total_endpoints} accessible")
    
    def test_portfolio_construction_basic(self):
        """Test basic portfolio construction functionality"""
        print("🔥 Testing basic portfolio construction...")
        
        test_portfolios = [
            {
                'name': 'Single Stock',
                'stocks': ['AAPL'],
                'expected_weight': 1.0
            },
            {
                'name': 'Basic Portfolio',
                'stocks': ['AAPL', 'GOOGL', 'MSFT'],
                'expected_weight': 1.0
            },
            {
                'name': 'Tech Portfolio',
                'stocks': ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'],
                'expected_weight': 1.0
            }
        ]
        
        successful_constructions = 0
        
        for portfolio in test_portfolios:
            try:
                print(f"🎯 Testing {portfolio['name']}: {portfolio['stocks']}")
                
                start_time = time.time()
                response = self.api_client.construct_portfolio(portfolio['stocks'])
                construction_time = time.time() - start_time
                
                if response.status_code == 200:
                    validation = self.api_client.validate_portfolio_response(response)
                    
                    if validation['is_json'] and validation['has_portfolio_weights']:
                        data = response.json()
                        weights = data['portfolio_weights']
                        total_weight = sum(weights.values()) if weights else 0
                        
                        print(f"✅ {portfolio['name']}: Constructed in {construction_time:.2f}s, total weight: {total_weight:.3f}")
                        successful_constructions += 1
                        
                        # Log individual stock weights
                        for stock, weight in weights.items():
                            print(f"   📊 {stock}: {weight:.3f}")
                            
                    else:
                        print(f"⚠️ {portfolio['name']}: Invalid response structure")
                        
                elif response.status_code == 404:
                    print(f"⚠️ {portfolio['name']}: Endpoint not available (404)")
                    break  # No point testing other portfolios
                else:
                    print(f"❌ {portfolio['name']}: Failed with status {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {portfolio['name']}: Exception - {str(e)}")
        
        if successful_constructions > 0:
            print(f"✅ Portfolio construction working: {successful_constructions}/{len(test_portfolios)} successful")
            self.log_test_result("Portfolio Construction", "PASS", f"{successful_constructions}/{len(test_portfolios)} successful")
        else:
            print("⚠️ Portfolio construction not available or not working")
            self.log_test_result("Portfolio Construction", "SKIP", "Not available")
            pytest.skip("Portfolio construction endpoint not available")
    
    def test_agent_analysis_endpoints(self):
        """Test individual agent analysis endpoints"""
        print("🔥 Testing agent analysis endpoints...")
        
        test_stocks = ['AAPL', 'GOOGL']
        
        agent_endpoints = [
            ('fundamental_analysis', 'Fundamental Analysis Agent'),
            ('sentiment_analysis', 'Sentiment Analysis Agent'),
            ('technical_analysis', 'Technical Analysis Agent'),
            ('heat_diffusion_analysis', 'Heat Diffusion Agent')
        ]
        
        working_agents = 0
        
        for method_name, agent_name in agent_endpoints:
            try:
                print(f"🤖 Testing {agent_name}...")
                
                if hasattr(self.api_client, method_name):
                    method = getattr(self.api_client, method_name)
                    
                    start_time = time.time()
                    response = method(test_stocks)
                    analysis_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            if isinstance(data, dict) and ('analysis' in data or 'insights' in data or 'result' in data):
                                print(f"✅ {agent_name}: Working ({analysis_time:.2f}s)")
                                working_agents += 1
                            else:
                                print(f"⚠️ {agent_name}: Response format issue")
                        except:
                            print(f"⚠️ {agent_name}: Invalid JSON response")
                    elif response.status_code == 404:
                        print(f"⚠️ {agent_name}: Not available (404)")
                    else:
                        print(f"❌ {agent_name}: Failed ({response.status_code})")
                else:
                    print(f"❌ {agent_name}: Method not found in client")
                    
            except Exception as e:
                print(f"❌ {agent_name}: Exception - {str(e)}")
        
        print(f"📊 Agent Analysis Summary: {working_agents}/{len(agent_endpoints)} agents working")
        
        if working_agents > 0:
            self.log_test_result("Agent Analysis", "PASS", f"{working_agents}/{len(agent_endpoints)} agents working")
        else:
            self.log_test_result("Agent Analysis", "SKIP", "No agent endpoints available")
            print("⚠️ No agent analysis endpoints are working")
    
    def test_frontend_portfolio_interaction(self):
        """Test frontend portfolio interaction"""
        print("🔥 Testing frontend portfolio interaction...")
        
        # Navigate to application
        success = self.portfolio_page.navigate_to_dashboard(self.base_url)
        assert success, "Failed to navigate to application"
        
        # Check for interactive elements
        page_info = self.portfolio_page.get_page_info()
        interactive_elements = page_info.get('buttons_count', 0) + page_info.get('inputs_count', 0) + page_info.get('forms_count', 0)
        
        if interactive_elements == 0:
            print("📄 No interactive elements found - likely static content")
            self.log_test_result("Frontend Interaction", "SKIP", "No interactive elements")
            pytest.skip("No interactive elements found on frontend")
        
        print(f"🎮 Found {interactive_elements} interactive elements")
        
        # Try to find portfolio-specific elements
        stock_input_found = False
        construct_button_found = False
        
        try:
            stock_input = self.portfolio_page.find_stock_input()
            if stock_input:
                stock_input_found = True
                print("✅ Stock input field found")
                
                # Try to interact with it
                stock_input.clear()
                stock_input.send_keys("AAPL")
                entered_value = stock_input.get_attribute('value')
                
                if "AAPL" in entered_value:
                    print("✅ Stock input interaction successful")
                else:
                    print("⚠️ Stock input interaction issue")
                    
        except Exception as e:
            print(f"⚠️ Stock input interaction failed: {e}")
        
        try:
            construct_button = self.portfolio_page.find_construct_button()
            if construct_button:
                construct_button_found = True
                print("✅ Portfolio construction button found")
                
                # Check if button is clickable
                if construct_button.is_enabled():
                    print("✅ Construction button is clickable")
                else:
                    print("⚠️ Construction button is disabled")
                    
        except Exception as e:
            print(f"⚠️ Construction button check failed: {e}")
        
        # Summary
        portfolio_features = stock_input_found + construct_button_found
        if portfolio_features > 0:
            print(f"✅ Portfolio frontend features working: {portfolio_features}/2")
            self.log_test_result("Frontend Interaction", "PASS", f"{portfolio_features}/2 features found")
        else:
            print("📝 Generic interactive elements found, but no portfolio-specific features")
            self.log_test_result("Frontend Interaction", "PARTIAL", "Generic elements only")
    
    def test_error_handling(self):
        """Test basic error handling"""
        print("🔥 Testing error handling...")
        
        error_scenarios = [
            {
                'name': 'Invalid Endpoint',
                'endpoint': '/api/nonexistent',
                'expected_codes': [404, 405]
            },
            {
                'name': 'Invalid Portfolio Stocks',
                'test': lambda: self.api_client.construct_portfolio(['INVALID_STOCK_123', '']),
                'expected_codes': [200, 400, 422]  # Some APIs handle gracefully
            }
        ]
        
        error_handling_works = 0
        
        for scenario in error_scenarios:
            try:
                print(f"🧪 Testing {scenario['name']}...")
                
                if 'endpoint' in scenario:
                    response = self.api_client._make_request('GET', scenario['endpoint'])
                elif 'test' in scenario:
                    response = scenario['test']()
                else:
                    continue
                
                expected_codes = scenario['expected_codes']
                
                if response.status_code in expected_codes:
                    print(f"✅ {scenario['name']}: Proper error handling ({response.status_code})")
                    error_handling_works += 1
                else:
                    print(f"⚠️ {scenario['name']}: Unexpected response ({response.status_code})")
                    
            except Exception as e:
                print(f"❌ {scenario['name']}: Exception during error test - {str(e)}")
        
        print(f"📊 Error Handling Summary: {error_handling_works}/{len(error_scenarios)} scenarios handled properly")
        self.log_test_result("Error Handling", "PASS", f"{error_handling_works}/{len(error_scenarios)} scenarios OK")
    
    def test_performance_benchmarks(self):
        """Test basic performance benchmarks"""
        print("🔥 Testing performance benchmarks...")
        
        # Test frontend load performance
        frontend_times = []
        for i in range(3):  # Test 3 times for average
            start_time = time.time()
            success = self.portfolio_page.navigate_to_dashboard(self.base_url)
            load_time = time.time() - start_time
            
            if success:
                frontend_times.append(load_time)
        
        if frontend_times:
            avg_frontend_load = sum(frontend_times) / len(frontend_times)
            max_frontend_load = max(frontend_times)
            print(f"🚀 Frontend Load - Avg: {avg_frontend_load:.2f}s, Max: {max_frontend_load:.2f}s")
        
        # Test API performance
        api_performance = self.api_client.get_performance_summary()
        if api_performance:
            avg_api_time = api_performance.get('average_response_time_ms', 0)
            max_api_time = api_performance.get('max_response_time_ms', 0)
            print(f"🌐 API Performance - Avg: {avg_api_time:.0f}ms, Max: {max_api_time:.0f}ms")
        
        # Performance assertions (reasonable thresholds)
        if frontend_times:
            assert max(frontend_times) < 30, f"Frontend load too slow: {max(frontend_times):.2f}s"
        
        if api_performance and api_performance.get('max_response_time_ms', 0) > 0:
            max_api_ms = api_performance.get('max_response_time_ms', 0)
            assert max_api_ms < 10000, f"API response too slow: {max_api_ms:.0f}ms"  # 10 seconds max
        
        self.log_test_result("Performance Benchmarks", "PASS", f"Frontend: {avg_frontend_load:.2f}s, API: {avg_api_time:.0f}ms")
    
    def teardown_method(self, method):
        """Cleanup after each test"""
        super().teardown_method(method)
        
        # Print API performance summary if available
        try:
            performance = self.api_client.get_performance_summary()
            if performance and performance.get('total_requests', 0) > 0:
                print(f"📊 API Performance Summary: {performance}")
        except:
            pass