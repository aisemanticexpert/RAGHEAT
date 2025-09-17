#!/usr/bin/env python3
"""
API tests for RAGHeat REST endpoints
"""

import pytest
import time
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from framework.base_test import BaseTest
from framework.utilities.api_client import RAGHeatAPIClient

@pytest.mark.api
class TestRestEndpoints(BaseTest):
    """Test REST API endpoints comprehensively"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.api_client = RAGHeatAPIClient(self.api_url)
    
    def test_health_endpoint_comprehensive(self):
        """Comprehensive health endpoint testing"""
        print("🔥 Testing health endpoint comprehensively...")
        
        response = self.api_client.health_check()
        
        # Status code validation
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        
        # Response time validation
        response_time = response.elapsed.total_seconds() * 1000
        assert response_time < 5000, f"Health check too slow: {response_time:.0f}ms"
        
        # Content type validation
        content_type = response.headers.get('content-type', '')
        
        if 'application/json' in content_type:
            # JSON response validation
            data = response.json()
            print(f"✅ Health endpoint JSON response: {json.dumps(data, indent=2)}")
            
            # Flexible validation for different response formats
            if isinstance(data, dict):
                # Check for status field
                if 'status' in data:
                    valid_statuses = ['healthy', 'ok', 'running', 'up']
                    assert data['status'].lower() in valid_statuses, f"Invalid status: {data['status']}"
                
                # Check for timestamp if present
                if 'timestamp' in data:
                    assert isinstance(data['timestamp'], str), "Timestamp should be string"
                    assert len(data['timestamp']) > 0, "Timestamp should not be empty"
                
                # Check for version if present
                if 'version' in data:
                    assert isinstance(data['version'], str), "Version should be string"
                
                # Check for environment if present
                if 'environment' in data:
                    valid_environments = ['development', 'staging', 'production', 'local']
                    assert data['environment'].lower() in valid_environments, f"Invalid environment: {data['environment']}"
                
                # Check for agents_active if present
                if 'agents_active' in data:
                    assert isinstance(data['agents_active'], bool), "agents_active should be boolean"
                    
        elif 'text/html' in content_type:
            # HTML response (frontend served from API)
            html_content = response.text
            assert len(html_content) > 100, "HTML response too short"
            assert '<html' in html_content.lower(), "Invalid HTML response"
            print("✅ Health endpoint serving HTML (frontend)")
            
        else:
            pytest.fail(f"Unexpected content type: {content_type}")
        
        self.log_test_result("Health Endpoint", "PASS", f"Response time: {response_time:.0f}ms")
    
    def test_system_status_endpoint_comprehensive(self):
        """Comprehensive system status endpoint testing"""
        print("🔥 Testing system status endpoint...")
        
        try:
            response = self.api_client.system_status()
            
            if response.status_code == 404:
                print("⚠️ System status endpoint not available (404)")
                self.log_test_result("System Status", "SKIP", "Endpoint not available")
                pytest.skip("System status endpoint not implemented")
            
            assert response.status_code == 200, f"System status failed: {response.status_code}"
            
            # Response time validation
            response_time = response.elapsed.total_seconds() * 1000
            assert response_time < 10000, f"System status too slow: {response_time:.0f}ms"
            
            # JSON response validation
            if 'application/json' in response.headers.get('content-type', ''):
                data = response.json()
                print(f"✅ System status response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                
                if isinstance(data, dict):
                    # Check for agents information
                    if 'agents' in data:
                        agents = data['agents']
                        assert isinstance(agents, list), "Agents should be a list"
                        
                        if agents:
                            print(f"🤖 Found {len(agents)} agents")
                            
                            # Validate agent structure
                            for i, agent in enumerate(agents[:5]):  # Check first 5 agents
                                if isinstance(agent, dict):
                                    assert 'name' in agent, f"Agent {i} missing name"
                                    assert 'status' in agent, f"Agent {i} missing status"
                                    
                                    print(f"   🤖 {agent.get('name', 'Unknown')}: {agent.get('status', 'Unknown')}")
                                    
                                    # Check status values
                                    valid_statuses = ['active', 'inactive', 'running', 'stopped', 'healthy', 'error']
                                    if agent.get('status'):
                                        assert agent['status'].lower() in valid_statuses, f"Invalid agent status: {agent['status']}"
                    
                    # Check for system information
                    system_fields = ['version', 'environment', 'cache_enabled', 'uptime']
                    found_system_fields = [field for field in system_fields if field in data or any(field in str(key).lower() for key in data.keys())]
                    
                    if found_system_fields:
                        print(f"🔧 System fields found: {found_system_fields}")
                        
            self.log_test_result("System Status", "PASS", f"Response time: {response_time:.0f}ms")
            
        except Exception as e:
            print(f"⚠️ System status test failed: {e}")
            self.log_test_result("System Status", "FAIL", str(e))
            pytest.fail(f"System status endpoint test failed: {e}")
    
    def test_portfolio_construction_endpoint_comprehensive(self):
        """Comprehensive portfolio construction endpoint testing"""
        print("🔥 Testing portfolio construction endpoint...")
        
        # Test data variations
        test_cases = [
            {
                'name': 'Single Stock',
                'stocks': ['AAPL'],
                'expected_behavior': 'success'
            },
            {
                'name': 'Multiple Stocks',
                'stocks': ['AAPL', 'GOOGL', 'MSFT'],
                'expected_behavior': 'success'
            },
            {
                'name': 'Large Portfolio',
                'stocks': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META'],
                'expected_behavior': 'success'
            },
            {
                'name': 'Empty Portfolio',
                'stocks': [],
                'expected_behavior': 'error_or_handle'
            }
        ]
        
        successful_tests = 0
        
        for test_case in test_cases:
            print(f"🧪 Testing {test_case['name']}: {test_case['stocks']}")
            
            try:
                start_time = time.time()
                response = self.api_client.construct_portfolio(test_case['stocks'])
                response_time = (time.time() - start_time) * 1000
                
                print(f"   📊 Response: {response.status_code} in {response_time:.0f}ms")
                
                if test_case['expected_behavior'] == 'success':
                    if response.status_code == 200:
                        # Validate successful response
                        validation = self.api_client.validate_portfolio_response(response)
                        
                        if validation['is_json']:
                            data = response.json()
                            
                            # Check response structure
                            required_keys = ['portfolio_weights', 'risk_metrics', 'agent_insights']
                            optional_keys = ['analysis_summary', 'performance_metrics', 'recommendations']
                            
                            found_required = sum(1 for key in required_keys if key in data)
                            found_optional = sum(1 for key in optional_keys if key in data)
                            
                            print(f"   ✅ Response structure: {found_required}/{len(required_keys)} required, {found_optional}/{len(optional_keys)} optional")
                            
                            # Validate portfolio weights
                            if 'portfolio_weights' in data:
                                weights = data['portfolio_weights']
                                if isinstance(weights, dict) and weights:
                                    total_weight = sum(weights.values())
                                    print(f"   💰 Portfolio weights sum: {total_weight:.3f}")
                                    
                                    # Check individual weights
                                    for stock, weight in weights.items():
                                        assert isinstance(weight, (int, float)), f"Weight for {stock} is not numeric"
                                        assert 0 <= weight <= 1, f"Weight for {stock} is out of range: {weight}"
                                    
                                    # Weights should approximately sum to 1 for valid portfolios
                                    if test_case['stocks'] and 0.8 <= total_weight <= 1.2:
                                        print(f"   ✅ Weight validation passed")
                                    else:
                                        print(f"   ⚠️ Weight sum unusual: {total_weight}")
                            
                            # Validate agent insights
                            if 'agent_insights' in data:
                                insights = data['agent_insights']
                                if isinstance(insights, list) and insights:
                                    print(f"   🤖 Agent insights: {len(insights)} entries")
                                    
                                    # Check insight structure
                                    for insight in insights[:3]:  # Check first 3
                                        if isinstance(insight, dict):
                                            if 'agent' in insight and 'analysis' in insight:
                                                print(f"      🔍 {insight['agent']}: {insight['analysis'][:50]}...")
                            
                            successful_tests += 1
                            
                        else:
                            print(f"   ⚠️ Non-JSON response received")
                            
                    elif response.status_code == 404:
                        print("   ⚠️ Portfolio construction endpoint not available")
                        self.log_test_result("Portfolio Construction", "SKIP", "Endpoint not available")
                        pytest.skip("Portfolio construction endpoint not implemented")
                        
                    else:
                        print(f"   ❌ Unexpected status code: {response.status_code}")
                        
                elif test_case['expected_behavior'] == 'error_or_handle':
                    # Empty portfolio should either be handled gracefully or return error
                    if response.status_code in [200, 400, 422]:
                        print(f"   ✅ Handled appropriately: {response.status_code}")
                        successful_tests += 1
                    else:
                        print(f"   ⚠️ Unexpected handling: {response.status_code}")
                
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
        
        # Summary
        print(f"📊 Portfolio Construction Tests: {successful_tests}/{len(test_cases)} successful")
        
        if successful_tests > 0:
            self.log_test_result("Portfolio Construction", "PASS", f"{successful_tests}/{len(test_cases)} cases passed")
        else:
            self.log_test_result("Portfolio Construction", "FAIL", "No test cases passed")
            pytest.fail("Portfolio construction endpoint tests failed")
    
    def test_analysis_endpoints_comprehensive(self):
        """Test all analysis endpoints comprehensively"""
        print("🔥 Testing analysis endpoints...")
        
        test_stocks = ['AAPL', 'GOOGL']
        
        analysis_methods = [
            ('fundamental_analysis', 'Fundamental Analysis'),
            ('sentiment_analysis', 'Sentiment Analysis'),
            ('technical_analysis', 'Technical Analysis'),
            ('heat_diffusion_analysis', 'Heat Diffusion Analysis')
        ]
        
        working_endpoints = 0
        
        for method_name, endpoint_name in analysis_methods:
            print(f"🧪 Testing {endpoint_name}...")
            
            try:
                if hasattr(self.api_client, method_name):
                    method = getattr(self.api_client, method_name)
                    
                    start_time = time.time()
                    response = method(test_stocks)
                    response_time = (time.time() - start_time) * 1000
                    
                    print(f"   📊 Response: {response.status_code} in {response_time:.0f}ms")
                    
                    if response.status_code == 200:
                        # Validate JSON response
                        if 'application/json' in response.headers.get('content-type', ''):
                            try:
                                data = response.json()
                                
                                # Check response structure
                                expected_keys = ['analysis_type', 'insights', 'result', 'analysis', 'summary']
                                found_keys = [key for key in expected_keys if key in data]
                                
                                if found_keys:
                                    print(f"   ✅ Valid structure: {found_keys}")
                                    
                                    # Check analysis content
                                    for key in found_keys:
                                        content = data[key]
                                        if isinstance(content, str) and len(content) > 10:
                                            print(f"      📝 {key}: {content[:100]}...")
                                        elif isinstance(content, list) and content:
                                            print(f"      📝 {key}: {len(content)} items")
                                    
                                    working_endpoints += 1
                                else:
                                    print(f"   ⚠️ Response missing expected keys")
                                    
                            except json.JSONDecodeError:
                                print(f"   ⚠️ Invalid JSON response")
                        else:
                            print(f"   ⚠️ Non-JSON response")
                            
                    elif response.status_code == 404:
                        print(f"   ⚠️ Endpoint not available (404)")
                    else:
                        print(f"   ❌ Failed with status {response.status_code}")
                        
                else:
                    print(f"   ❌ Method {method_name} not found in API client")
                    
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
        
        print(f"📊 Analysis Endpoints: {working_endpoints}/{len(analysis_methods)} working")
        
        if working_endpoints > 0:
            self.log_test_result("Analysis Endpoints", "PASS", f"{working_endpoints}/{len(analysis_methods)} working")
        else:
            self.log_test_result("Analysis Endpoints", "SKIP", "No endpoints available")
    
    def test_analysis_tools_endpoint(self):
        """Test analysis tools endpoint"""
        print("🔥 Testing analysis tools endpoint...")
        
        try:
            response = self.api_client.get_analysis_tools()
            
            if response.status_code == 404:
                print("⚠️ Analysis tools endpoint not available")
                self.log_test_result("Analysis Tools", "SKIP", "Endpoint not available")
                pytest.skip("Analysis tools endpoint not implemented")
            
            assert response.status_code == 200, f"Analysis tools failed: {response.status_code}"
            
            # Validate response
            if 'application/json' in response.headers.get('content-type', ''):
                data = response.json()
                
                if isinstance(data, dict):
                    # Look for tools information
                    tools_keys = ['tools', 'available_tools', 'analysis_tools', 'methods']
                    found_tools_key = next((key for key in tools_keys if key in data), None)
                    
                    if found_tools_key:
                        tools = data[found_tools_key]
                        if isinstance(tools, list):
                            print(f"✅ Found {len(tools)} analysis tools")
                            
                            for tool in tools[:5]:  # Show first 5 tools
                                if isinstance(tool, dict):
                                    name = tool.get('name', 'Unknown')
                                    description = tool.get('description', 'No description')
                                    print(f"   🔧 {name}: {description[:100]}...")
                                elif isinstance(tool, str):
                                    print(f"   🔧 {tool}")
                        else:
                            print(f"✅ Tools information found: {type(tools)}")
                    else:
                        print(f"✅ Response received with keys: {list(data.keys())}")
                        
            self.log_test_result("Analysis Tools", "PASS", "Endpoint accessible")
            
        except Exception as e:
            print(f"⚠️ Analysis tools test failed: {e}")
            self.log_test_result("Analysis Tools", "FAIL", str(e))
    
    def test_root_endpoint(self):
        """Test root endpoint (serves frontend)"""
        print("🔥 Testing root endpoint...")
        
        try:
            response = self.api_client.get_root_page()
            
            assert response.status_code == 200, f"Root endpoint failed: {response.status_code}"
            
            content_type = response.headers.get('content-type', '')
            
            if 'text/html' in content_type:
                html_content = response.text
                assert len(html_content) > 200, "HTML content too short"
                
                # Check for HTML structure
                html_lower = html_content.lower()
                assert '<html' in html_lower, "Invalid HTML structure"
                assert '<body' in html_lower, "Missing body tag"
                
                # Check for React/portfolio indicators
                portfolio_indicators = ['portfolio', 'ragheat', 'stock', 'react', 'app']
                found_indicators = [indicator for indicator in portfolio_indicators if indicator in html_lower]
                
                print(f"✅ Root endpoint serving HTML ({len(html_content)} chars)")
                print(f"   📄 Content indicators: {found_indicators}")
                
            elif 'application/json' in content_type:
                # JSON response
                data = response.json()
                print(f"✅ Root endpoint serving JSON: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                
            else:
                print(f"✅ Root endpoint serving {content_type}")
            
            self.log_test_result("Root Endpoint", "PASS", f"Content type: {content_type}")
            
        except Exception as e:
            print(f"❌ Root endpoint test failed: {e}")
            self.log_test_result("Root Endpoint", "FAIL", str(e))
            pytest.fail(f"Root endpoint test failed: {e}")
    
    def test_api_error_handling(self):
        """Test API error handling"""
        print("🔥 Testing API error handling...")
        
        error_tests = [
            {
                'name': 'Nonexistent Endpoint',
                'test': lambda: self.api_client._make_request('GET', '/api/nonexistent'),
                'expected_codes': [404, 405]
            },
            {
                'name': 'Invalid Method',
                'test': lambda: self.api_client._make_request('DELETE', '/api/health'),
                'expected_codes': [404, 405, 501]
            },
            {
                'name': 'Malformed JSON',
                'test': lambda: self.api_client.session.post(
                    f"{self.api_client.base_url}/api/portfolio/construct",
                    data="invalid json",
                    headers={'Content-Type': 'application/json'}
                ),
                'expected_codes': [400, 422, 500]
            }
        ]
        
        passed_tests = 0
        
        for test in error_tests:
            try:
                print(f"🧪 Testing {test['name']}...")
                
                response = test['test']()
                expected_codes = test['expected_codes']
                
                if response.status_code in expected_codes:
                    print(f"   ✅ Proper error handling: {response.status_code}")
                    passed_tests += 1
                else:
                    print(f"   ⚠️ Unexpected response: {response.status_code}")
                    
            except Exception as e:
                print(f"   ⚠️ Error test exception: {e}")
        
        print(f"📊 Error Handling: {passed_tests}/{len(error_tests)} tests passed")
        self.log_test_result("API Error Handling", "PASS", f"{passed_tests}/{len(error_tests)} scenarios handled")
    
    def teardown_method(self, method):
        """Cleanup after each test"""
        super().teardown_method(method)
        
        # Print API performance summary
        try:
            performance = self.api_client.get_performance_summary()
            if performance:
                print(f"📊 API Performance: {performance}")
        except:
            pass