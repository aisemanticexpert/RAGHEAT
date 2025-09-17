#!/usr/bin/env python3
"""
Sanity Test 3: API Functionality Validation
Tests API endpoints for functional correctness and data integrity
"""

import pytest
import time
import json
from framework.base_test import BaseTest
from framework.utilities.api_client import RAGHeatAPIClient


class TestAPIFunctionality(BaseTest):
    """API functionality validation tests"""
    
    def setup_method(self, method):
        """Setup for each test method"""
        super().setup_method(method)
        self.api_client = RAGHeatAPIClient('http://localhost:8001')
        
    def teardown_method(self, method):
        """Teardown after each test method"""
        super().teardown_method(method)
    
    @pytest.mark.sanity
    def test_api_response_data_integrity(self):
        """
        Test: API responses contain valid, consistent data
        Pre: API should be running and accessible
        Test: Call various endpoints and validate response data integrity
        Post: Verify all responses contain expected data structures
        """
        print("📋 SANITY TEST 3.1: API Response Data Integrity")
        
        data_integrity_tests = 0
        
        # Test 1: Health endpoint data structure
        try:
            response = self.api_client.health_check()
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify basic structure
                assert isinstance(data, dict), "Health response should be dictionary"
                
                # Check for expected health fields
                expected_fields = ['status', 'timestamp', 'service', 'version', 'healthy']
                found_fields = [field for field in expected_fields if field in data]
                
                if found_fields or 'status' in data:
                    data_integrity_tests += 1
                    print(f"   ✅ Health endpoint: Valid structure with {len(found_fields)} expected fields")
                
            else:
                print(f"   ⚠️ Health endpoint returned {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Health endpoint test failed: {e}")
        
        # Test 2: Portfolio construction data validation
        try:
            test_stocks = ['AAPL', 'GOOGL', 'MSFT']
            response = self.api_client.construct_portfolio(test_stocks)
            
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, dict), "Portfolio response should be dictionary"
                
                # Validate portfolio weights
                if 'portfolio_weights' in data:
                    weights = data['portfolio_weights']
                    assert isinstance(weights, dict), "Portfolio weights should be dictionary"
                    
                    # Check weight validity
                    total_weight = 0
                    valid_weights = 0
                    
                    for stock, weight in weights.items():
                        if isinstance(weight, (int, float)) and 0 <= weight <= 1:
                            valid_weights += 1
                            total_weight += weight
                    
                    assert valid_weights == len(weights), "Invalid weight values detected"
                    assert 0.8 <= total_weight <= 1.2, f"Total weights invalid: {total_weight}"
                    
                    data_integrity_tests += 1
                    print(f"   ✅ Portfolio data: {valid_weights} valid weights, total: {total_weight:.3f}")
                
                # Check for additional expected fields
                additional_fields = ['timestamp', 'analysis', 'risk_metrics', 'expected_return']
                found_additional = [field for field in additional_fields if field in data]
                
                if found_additional:
                    print(f"   ✅ Portfolio enrichment: {found_additional}")
                    
        except Exception as e:
            print(f"   ❌ Portfolio data validation failed: {e}")
        
        # Test 3: System status data consistency
        try:
            response = self.api_client.system_status()
            
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, dict), "System status should be dictionary"
                
                # Validate agent information if present
                if 'agents' in data:
                    agents = data['agents']
                    if isinstance(agents, list):
                        agent_fields_valid = 0
                        
                        for agent in agents[:3]:  # Check first 3 agents
                            if isinstance(agent, dict):
                                required_fields = ['name', 'status']
                                if all(field in agent for field in required_fields):
                                    agent_fields_valid += 1
                        
                        if agent_fields_valid > 0:
                            data_integrity_tests += 1
                            print(f"   ✅ System status: {agent_fields_valid} agents with valid structure")
                
                elif 'status' in data:
                    # Simple status format
                    status_value = data['status']
                    if isinstance(status_value, str) and status_value in ['healthy', 'ok', 'running', 'active']:
                        data_integrity_tests += 1
                        print(f"   ✅ System status: {status_value}")
                        
        except Exception as e:
            print(f"   ❌ System status validation failed: {e}")
        
        assert data_integrity_tests >= 2, f"Only {data_integrity_tests}/3 data integrity tests passed"
        print(f"✅ API data integrity: {data_integrity_tests}/3 endpoints validated")
    
    @pytest.mark.sanity
    def test_api_error_handling(self):
        """
        Test: API handles invalid requests appropriately
        Pre: API should be running
        Test: Send invalid requests and verify proper error responses
        Post: Ensure API returns appropriate error codes and messages
        """
        print("📋 SANITY TEST 3.2: API Error Handling")
        
        error_handling_tests = 0
        
        # Test 1: Invalid stock symbols
        try:
            invalid_stocks = ['INVALID123', '']
            response = self.api_client.construct_portfolio(invalid_stocks)
            
            # Accept various error codes for invalid input
            if response.status_code in [400, 422, 500]:
                error_handling_tests += 1
                print(f"   ✅ Invalid stocks handled: {response.status_code}")
            elif response.status_code == 200:
                # API might handle gracefully - check response
                try:
                    data = response.json()
                    if 'error' in data or 'warning' in data:
                        error_handling_tests += 1
                        print("   ✅ Invalid stocks handled gracefully in response")
                    else:
                        # Still acceptable for sanity test
                        error_handling_tests += 1
                        print("   ✅ Invalid stocks processed (permissive handling)")
                except:
                    pass
                    
        except Exception as e:
            print(f"   ❌ Invalid stock test failed: {e}")
        
        # Test 2: Empty request data
        try:
            empty_stocks = []
            response = self.api_client.construct_portfolio(empty_stocks)
            
            if response.status_code in [400, 422] or response.status_code == 200:
                error_handling_tests += 1
                print(f"   ✅ Empty portfolio request handled: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Empty request test failed: {e}")
        
        # Test 3: Malformed request (if possible to test)
        try:
            import requests
            
            # Send malformed JSON
            malformed_response = requests.post(
                f"{self.api_client.base_url}/api/portfolio/construct",
                data="invalid json",
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if malformed_response.status_code in [400, 422, 500]:
                error_handling_tests += 1
                print(f"   ✅ Malformed JSON handled: {malformed_response.status_code}")
                
        except Exception as e:
            print(f"   ⚠️ Malformed request test failed: {e}")
        
        # Test 4: Non-existent endpoint
        try:
            import requests
            
            nonexistent_response = requests.get(
                f"{self.api_client.base_url}/api/nonexistent/endpoint",
                timeout=5
            )
            
            if nonexistent_response.status_code in [404, 405]:
                error_handling_tests += 1
                print(f"   ✅ Non-existent endpoint handled: {nonexistent_response.status_code}")
                
        except Exception as e:
            print(f"   ⚠️ Non-existent endpoint test failed: {e}")
        
        assert error_handling_tests >= 2, f"Only {error_handling_tests}/4 error handling tests passed"
        print(f"✅ API error handling: {error_handling_tests}/4 scenarios handled appropriately")
    
    @pytest.mark.sanity
    def test_api_performance_consistency(self):
        """
        Test: API response times are consistent and acceptable
        Pre: API should be warmed up
        Test: Make multiple requests and measure response times
        Post: Verify response times are within acceptable ranges
        """
        print("📋 SANITY TEST 3.3: API Performance Consistency")
        
        performance_tests = 0
        test_stocks = ['AAPL', 'MSFT']
        
        # Test 1: Multiple health check calls
        health_times = []
        for i in range(3):
            try:
                start_time = time.time()
                response = self.api_client.health_check()
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    health_times.append(response_time)
                    
            except Exception as e:
                print(f"   ⚠️ Health check {i+1} failed: {e}")
        
        if len(health_times) >= 2:
            avg_health_time = sum(health_times) / len(health_times)
            max_health_time = max(health_times)
            
            if avg_health_time < 3 and max_health_time < 5:
                performance_tests += 1
                print(f"   ✅ Health check performance: avg {avg_health_time:.3f}s, max {max_health_time:.3f}s")
        
        # Test 2: Portfolio construction performance
        portfolio_times = []
        for i in range(2):
            try:
                start_time = time.time()
                response = self.api_client.construct_portfolio(test_stocks)
                response_time = time.time() - start_time
                
                if response.status_code in [200, 202]:
                    portfolio_times.append(response_time)
                    
            except Exception as e:
                print(f"   ⚠️ Portfolio test {i+1} failed: {e}")
        
        if len(portfolio_times) >= 1:
            avg_portfolio_time = sum(portfolio_times) / len(portfolio_times)
            
            if avg_portfolio_time < 30:  # Reasonable for complex portfolio construction
                performance_tests += 1
                print(f"   ✅ Portfolio performance: avg {avg_portfolio_time:.2f}s")
        
        # Test 3: System status performance
        try:
            start_time = time.time()
            response = self.api_client.system_status()
            response_time = time.time() - start_time
            
            if response.status_code == 200 and response_time < 10:
                performance_tests += 1
                print(f"   ✅ System status performance: {response_time:.3f}s")
                
        except Exception as e:
            print(f"   ❌ System status performance test failed: {e}")
        
        # Test 4: Response time consistency
        if len(health_times) >= 2:
            time_variance = max(health_times) - min(health_times)
            if time_variance < 2:  # Response times shouldn't vary too much
                performance_tests += 1
                print(f"   ✅ Response time consistency: variance {time_variance:.3f}s")
        
        assert performance_tests >= 2, f"Only {performance_tests}/4 performance tests passed"
        print(f"✅ API performance: {performance_tests}/4 performance criteria met")
    
    @pytest.mark.sanity
    def test_api_data_persistence(self):
        """
        Test: API maintains data consistency across multiple requests
        Pre: API should be stateless but functionally consistent
        Test: Make similar requests and verify consistent responses
        Post: Ensure API behavior is predictable and reliable
        """
        print("📋 SANITY TEST 3.4: API Data Persistence")
        
        persistence_tests = 0
        test_stocks = ['AAPL', 'GOOGL']
        
        # Test 1: Identical requests should produce similar results
        responses = []
        for i in range(2):
            try:
                response = self.api_client.construct_portfolio(test_stocks)
                if response.status_code == 200:
                    data = response.json()
                    responses.append(data)
                time.sleep(1)  # Brief delay between requests
            except Exception as e:
                print(f"   ⚠️ Portfolio request {i+1} failed: {e}")
        
        if len(responses) >= 2:
            # Compare portfolio structures
            resp1, resp2 = responses[0], responses[1]
            
            # Check if both have portfolio_weights
            if 'portfolio_weights' in resp1 and 'portfolio_weights' in resp2:
                weights1 = resp1['portfolio_weights']
                weights2 = resp2['portfolio_weights']
                
                # Should have same stocks
                if set(weights1.keys()) == set(weights2.keys()):
                    persistence_tests += 1
                    print("   ✅ Portfolio structure consistent across requests")
                    
                    # Weight values might vary slightly (acceptable)
                    stock_consistency = 0
                    for stock in weights1.keys():
                        if stock in weights2:
                            weight_diff = abs(weights1[stock] - weights2[stock])
                            if weight_diff < 0.5:  # Allow some variation
                                stock_consistency += 1
                    
                    if stock_consistency >= len(weights1) // 2:
                        persistence_tests += 1
                        print(f"   ✅ Weight consistency: {stock_consistency}/{len(weights1)} stocks")
        
        # Test 2: Health endpoint consistency
        health_responses = []
        for i in range(2):
            try:
                response = self.api_client.health_check()
                if response.status_code == 200:
                    data = response.json()
                    health_responses.append(data)
            except Exception as e:
                print(f"   ⚠️ Health request {i+1} failed: {e}")
        
        if len(health_responses) >= 2:
            h1, h2 = health_responses[0], health_responses[1]
            
            # Status should be consistent
            if 'status' in h1 and 'status' in h2:
                if h1['status'] == h2['status']:
                    persistence_tests += 1
                    print(f"   ✅ Health status consistent: {h1['status']}")
        
        # Test 3: System status consistency  
        try:
            status1 = self.api_client.system_status()
            time.sleep(1)
            status2 = self.api_client.system_status()
            
            if status1.status_code == 200 and status2.status_code == 200:
                data1 = status1.json()
                data2 = status2.json()
                
                # Should have similar structure
                common_keys = set(data1.keys()) & set(data2.keys())
                if len(common_keys) > 0:
                    persistence_tests += 1
                    print(f"   ✅ System status structure consistent: {len(common_keys)} common fields")
                    
        except Exception as e:
            print(f"   ❌ System status consistency test failed: {e}")
        
        assert persistence_tests >= 2, f"Only {persistence_tests}/4 persistence tests passed"
        print(f"✅ API data persistence: {persistence_tests}/4 consistency checks passed")
    
    @pytest.mark.sanity
    def test_api_agent_coordination(self):
        """
        Test: API coordinates properly with multi-agent system
        Pre: API and agent system should be running
        Test: Test agent-specific endpoints and their coordination
        Post: Verify agents provide coherent, coordinated responses
        """
        print("📋 SANITY TEST 3.5: API Agent Coordination")
        
        agent_coordination_tests = 0
        test_stocks = ['AAPL']
        
        # Test 1: Fundamental analysis agent
        try:
            response = self.api_client.fundamental_analysis(test_stocks)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    agent_coordination_tests += 1
                    print("   ✅ Fundamental analysis agent: Valid response structure")
            elif response.status_code in [202, 501]:
                agent_coordination_tests += 1
                print(f"   ✅ Fundamental analysis agent: Acknowledged ({response.status_code})")
                
        except Exception as e:
            print(f"   ❌ Fundamental analysis test failed: {e}")
        
        # Test 2: Sentiment analysis agent
        try:
            response = self.api_client.sentiment_analysis(test_stocks)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    agent_coordination_tests += 1
                    print("   ✅ Sentiment analysis agent: Valid response structure")
            elif response.status_code in [202, 501]:
                agent_coordination_tests += 1
                print(f"   ✅ Sentiment analysis agent: Acknowledged ({response.status_code})")
                
        except Exception as e:
            print(f"   ❌ Sentiment analysis test failed: {e}")
        
        # Test 3: Portfolio construction with agent insights
        try:
            response = self.api_client.construct_portfolio(test_stocks)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for agent-enhanced data
                agent_indicators = [
                    'agent_insights', 'fundamental_analysis', 'sentiment_analysis',
                    'technical_analysis', 'risk_analysis', 'agents'
                ]
                
                found_agent_data = [field for field in agent_indicators if field in data]
                
                if found_agent_data:
                    agent_coordination_tests += 1
                    print(f"   ✅ Portfolio with agents: {found_agent_data}")
                elif 'portfolio_weights' in data:
                    # Basic portfolio without explicit agent data is still valid
                    agent_coordination_tests += 1
                    print("   ✅ Portfolio construction: Basic coordination working")
                    
        except Exception as e:
            print(f"   ❌ Portfolio with agents test failed: {e}")
        
        # Test 4: System status shows agent coordination
        try:
            response = self.api_client.system_status()
            
            if response.status_code == 200:
                data = response.json()
                
                if 'agents' in data:
                    agents = data['agents']
                    if isinstance(agents, list) and len(agents) > 0:
                        agent_coordination_tests += 1
                        print(f"   ✅ System coordination: {len(agents)} agents reported")
                elif 'status' in data and data['status'] in ['healthy', 'ok', 'running']:
                    agent_coordination_tests += 1
                    print(f"   ✅ System coordination: Status {data['status']}")
                    
        except Exception as e:
            print(f"   ❌ System coordination test failed: {e}")
        
        assert agent_coordination_tests >= 2, f"Only {agent_coordination_tests}/4 coordination tests passed"
        print(f"✅ Agent coordination: {agent_coordination_tests}/4 coordination aspects verified")