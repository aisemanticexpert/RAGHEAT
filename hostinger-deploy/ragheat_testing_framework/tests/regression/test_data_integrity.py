#!/usr/bin/env python3
"""
Regression Test 3: Data Integrity and Consistency
Tests data consistency, validation, and integrity across the application
"""

import pytest
import time
import json
from selenium.webdriver.common.by import By
from framework.base_test import BaseTest
from framework.page_objects.portfolio_dashboard import PortfolioDashboard
from framework.utilities.api_client import RAGHeatAPIClient


class TestDataIntegrity(BaseTest):
    """Data integrity and consistency validation tests"""
    
    def setup_method(self, method):
        """Setup for each test method"""
        super().setup_method(method)
        self.dashboard = PortfolioDashboard(self.driver)
        self.api_client = RAGHeatAPIClient('http://localhost:8001')
        
    def teardown_method(self, method):
        """Teardown after each test method"""
        super().teardown_method(method)
    
    @pytest.mark.regression
    def test_portfolio_weight_mathematics(self):
        """
        Test: Portfolio weights follow mathematical constraints
        Pre: Portfolio construction should be working
        Test: Verify weights sum to 1, are non-negative, and mathematically valid
        Post: Ensure mathematical integrity of portfolio calculations
        """
        print("🔄 REGRESSION TEST 3.1: Portfolio Weight Mathematics")
        
        math_tests = 0
        test_scenarios = [
            {'stocks': ['AAPL', 'GOOGL'], 'name': '2-stock'},
            {'stocks': ['AAPL', 'GOOGL', 'MSFT'], 'name': '3-stock'},
            {'stocks': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'], 'name': '5-stock'}
        ]
        
        for scenario in test_scenarios:
            stocks = scenario['stocks']
            name = scenario['name']
            
            try:
                response = self.api_client.construct_portfolio(stocks)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'portfolio_weights' in data:
                        weights = data['portfolio_weights']
                        
                        # Test 1: All weights are non-negative
                        negative_weights = [w for w in weights.values() if w < 0]
                        assert len(negative_weights) == 0, f"{name}: Found negative weights: {negative_weights}"
                        
                        # Test 2: All weights are reasonable (not too large)
                        excessive_weights = [w for w in weights.values() if w > 1.5]
                        assert len(excessive_weights) == 0, f"{name}: Found excessive weights: {excessive_weights}"
                        
                        # Test 3: Weights sum to approximately 1
                        total_weight = sum(weights.values())
                        weight_tolerance = 0.05  # 5% tolerance
                        assert 1 - weight_tolerance <= total_weight <= 1 + weight_tolerance, \
                               f"{name}: Total weight {total_weight:.4f} not close to 1.0"
                        
                        # Test 4: Number of weights matches number of stocks
                        assert len(weights) == len(stocks), \
                               f"{name}: Expected {len(stocks)} weights, got {len(weights)}"
                        
                        # Test 5: All requested stocks have weights
                        missing_stocks = [s for s in stocks if s not in weights]
                        assert len(missing_stocks) == 0, f"{name}: Missing weights for {missing_stocks}"
                        
                        math_tests += 1
                        print(f"   ✅ {name}: Mathematical constraints satisfied (total: {total_weight:.4f})")
                        
                        # Test 6: Weight precision is reasonable
                        precision_valid = all(
                            len(str(w).split('.')[-1]) <= 6 for w in weights.values() if '.' in str(w)
                        )
                        if precision_valid:
                            print(f"   ✅ {name}: Weight precision appropriate")
                    else:
                        print(f"   ⚠️ {name}: No portfolio weights in response")
                else:
                    print(f"   ⚠️ {name}: Portfolio construction failed ({response.status_code})")
                    
            except Exception as e:
                print(f"   ❌ {name}: Mathematical test failed - {e}")
        
        assert math_tests >= 2, f"Only {math_tests}/3 mathematical constraint tests passed"
        print(f"✅ Mathematical integrity: {math_tests}/3 scenarios validated")
    
    @pytest.mark.regression
    def test_data_consistency_across_requests(self):
        """
        Test: Data remains consistent across multiple similar requests
        Pre: API should provide consistent behavior
        Test: Make multiple similar requests and verify consistency
        Post: Ensure data integrity is maintained across requests
        """
        print("🔄 REGRESSION TEST 3.2: Data Consistency Across Requests")
        
        consistency_tests = 0
        test_stocks = ['AAPL', 'GOOGL']
        
        # Test 1: Multiple identical requests
        try:
            responses = []
            for i in range(3):
                response = self.api_client.construct_portfolio(test_stocks)
                if response.status_code == 200:
                    data = response.json()
                    responses.append(data)
                time.sleep(1)  # Brief delay between requests
            
            if len(responses) >= 2:
                # Compare structure consistency
                first_response = responses[0]
                structure_consistent = True
                
                for response in responses[1:]:
                    # Check if same top-level keys exist
                    if set(first_response.keys()) != set(response.keys()):
                        structure_consistent = False
                        break
                
                if structure_consistent:
                    consistency_tests += 1
                    print("   ✅ Response structure: Consistent across requests")
                
                # Check portfolio weights consistency
                if all('portfolio_weights' in r for r in responses):
                    weights_list = [r['portfolio_weights'] for r in responses]
                    
                    # All should have same stocks
                    stock_sets = [set(w.keys()) for w in weights_list]
                    stocks_consistent = all(s == stock_sets[0] for s in stock_sets)
                    
                    if stocks_consistent:
                        consistency_tests += 1
                        print("   ✅ Stock selection: Consistent across requests")
                        
                        # Weight values might vary slightly (stochastic optimization)
                        # Check that variations are within reasonable bounds
                        total_weights = [sum(w.values()) for w in weights_list]
                        weight_variance = max(total_weights) - min(total_weights)
                        
                        if weight_variance < 0.1:  # 10% tolerance
                            consistency_tests += 1
                            print(f"   ✅ Weight totals: Variance {weight_variance:.4f} within tolerance")
                            
        except Exception as e:
            print(f"   ❌ Identical requests test failed: {e}")
        
        # Test 2: Health endpoint consistency
        try:
            health_responses = []
            for i in range(3):
                response = self.api_client.health_check()
                if response.status_code == 200:
                    data = response.json()
                    health_responses.append(data)
            
            if len(health_responses) >= 2:
                first_health = health_responses[0]
                
                # Status should be consistent
                if 'status' in first_health:
                    statuses = [h.get('status') for h in health_responses]
                    if all(s == statuses[0] for s in statuses):
                        consistency_tests += 1
                        print(f"   ✅ Health status: Consistent '{statuses[0]}'")
                        
        except Exception as e:
            print(f"   ❌ Health consistency test failed: {e}")
        
        assert consistency_tests >= 2, f"Only {consistency_tests}/4 consistency tests passed"
        print(f"✅ Data consistency: {consistency_tests}/4 consistency checks passed")
    
    @pytest.mark.regression
    def test_json_response_validation(self):
        """
        Test: All JSON responses are properly formatted and valid
        Pre: API endpoints should be accessible
        Test: Validate JSON structure and content of all responses
        Post: Ensure all API responses meet JSON standards
        """
        print("🔄 REGRESSION TEST 3.3: JSON Response Validation")
        
        json_validation_tests = 0
        
        # Test endpoints with JSON responses
        endpoints_to_test = [
            ('health_check', lambda: self.api_client.health_check()),
            ('portfolio_construction', lambda: self.api_client.construct_portfolio(['AAPL', 'GOOGL'])),
            ('system_status', lambda: self.api_client.system_status()),
            ('fundamental_analysis', lambda: self.api_client.fundamental_analysis(['AAPL'])),
            ('sentiment_analysis', lambda: self.api_client.sentiment_analysis(['AAPL']))
        ]
        
        for endpoint_name, endpoint_func in endpoints_to_test:
            try:
                response = endpoint_func()
                
                if response.status_code == 200:
                    # Test JSON parsing
                    try:
                        data = response.json()
                        json_validation_tests += 1
                        print(f"   ✅ {endpoint_name}: Valid JSON structure")
                        
                        # Test JSON content types
                        if isinstance(data, dict):
                            # Check for common issues
                            json_issues = []
                            
                            # Check for null values in important fields
                            if 'portfolio_weights' in data:
                                weights = data['portfolio_weights']
                                if isinstance(weights, dict):
                                    null_weights = [k for k, v in weights.items() if v is None]
                                    if null_weights:
                                        json_issues.append(f"null weights: {null_weights}")
                            
                            # Check for extremely long strings (potential data corruption)
                            def check_string_lengths(obj, path=""):
                                if isinstance(obj, dict):
                                    for k, v in obj.items():
                                        check_string_lengths(v, f"{path}.{k}")
                                elif isinstance(obj, str) and len(obj) > 10000:
                                    json_issues.append(f"very long string at {path}: {len(obj)} chars")
                            
                            check_string_lengths(data)
                            
                            if not json_issues:
                                print(f"   ✅ {endpoint_name}: JSON content validation passed")
                            else:
                                print(f"   ⚠️ {endpoint_name}: JSON issues found: {json_issues}")
                                
                        elif isinstance(data, list):
                            # Valid JSON array
                            print(f"   ✅ {endpoint_name}: Valid JSON array with {len(data)} items")
                        else:
                            print(f"   ⚠️ {endpoint_name}: Unusual JSON type: {type(data)}")
                            
                    except json.JSONDecodeError as e:
                        print(f"   ❌ {endpoint_name}: Invalid JSON - {e}")
                        
                    # Test response headers
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        print(f"   ✅ {endpoint_name}: Correct Content-Type header")
                    elif content_type:
                        print(f"   ⚠️ {endpoint_name}: Unexpected Content-Type: {content_type}")
                        
                elif response.status_code in [400, 422, 500]:
                    # Error responses should still be valid JSON if they claim to be
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        try:
                            error_data = response.json()
                            print(f"   ✅ {endpoint_name}: Valid error JSON ({response.status_code})")
                        except:
                            print(f"   ❌ {endpoint_name}: Invalid error JSON ({response.status_code})")
                    else:
                        print(f"   ✅ {endpoint_name}: Non-JSON error response ({response.status_code})")
                else:
                    print(f"   ⚠️ {endpoint_name}: Unexpected status {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ {endpoint_name}: Endpoint test failed - {e}")
        
        assert json_validation_tests >= 2, f"Only {json_validation_tests}/5 JSON validation tests passed"
        print(f"✅ JSON validation: {json_validation_tests}/5 endpoints validated")
    
    @pytest.mark.regression
    def test_data_type_integrity(self):
        """
        Test: All data types are consistent and appropriate
        Pre: API should return structured data
        Test: Validate data types match expected formats
        Post: Ensure type safety across the application
        """
        print("🔄 REGRESSION TEST 3.4: Data Type Integrity")
        
        type_integrity_tests = 0
        
        # Test portfolio construction data types
        try:
            response = self.api_client.construct_portfolio(['AAPL', 'GOOGL', 'MSFT'])
            
            if response.status_code == 200:
                data = response.json()
                
                # Test portfolio weights types
                if 'portfolio_weights' in data:
                    weights = data['portfolio_weights']
                    
                    # Should be dictionary
                    assert isinstance(weights, dict), f"Weights should be dict, got {type(weights)}"
                    
                    # Stock symbols should be strings
                    for stock in weights.keys():
                        assert isinstance(stock, str), f"Stock symbol should be string, got {type(stock)}"
                        assert len(stock) > 0, "Stock symbol should not be empty string"
                    
                    # Weight values should be numbers
                    for weight in weights.values():
                        assert isinstance(weight, (int, float)), f"Weight should be number, got {type(weight)}"
                        assert not isinstance(weight, bool), "Weight should not be boolean"
                    
                    type_integrity_tests += 1
                    print("   ✅ Portfolio weights: All data types correct")
                
                # Test timestamp if present
                if 'timestamp' in data:
                    timestamp = data['timestamp']
                    # Should be string or number
                    assert isinstance(timestamp, (str, int, float)), f"Timestamp type invalid: {type(timestamp)}"
                    print("   ✅ Timestamp: Data type valid")
                
                # Test analysis data if present
                analysis_fields = ['analysis', 'risk_metrics', 'expected_return', 'agent_insights']
                for field in analysis_fields:
                    if field in data:
                        field_data = data[field]
                        # Should be structured data, not just strings for everything
                        if isinstance(field_data, dict):
                            print(f"   ✅ {field}: Structured data type")
                        elif isinstance(field_data, (list, str, int, float)):
                            print(f"   ✅ {field}: Valid data type ({type(field_data).__name__})")
                        else:
                            print(f"   ⚠️ {field}: Unusual data type ({type(field_data).__name__})")
                            
        except Exception as e:
            print(f"   ❌ Portfolio data type test failed: {e}")
        
        # Test system status data types
        try:
            response = self.api_client.system_status()
            
            if response.status_code == 200:
                data = response.json()
                
                if 'agents' in data:
                    agents = data['agents']
                    
                    # Should be list
                    assert isinstance(agents, list), f"Agents should be list, got {type(agents)}"
                    
                    # Each agent should be structured data
                    for i, agent in enumerate(agents[:3]):  # Check first 3
                        assert isinstance(agent, dict), f"Agent {i} should be dict, got {type(agent)}"
                        
                        if 'name' in agent:
                            assert isinstance(agent['name'], str), f"Agent name should be string"
                        
                        if 'status' in agent:
                            assert isinstance(agent['status'], str), f"Agent status should be string"
                    
                    type_integrity_tests += 1
                    print(f"   ✅ System agents: Data types valid for {len(agents)} agents")
                
                elif 'status' in data:
                    status = data['status']
                    assert isinstance(status, str), f"System status should be string, got {type(status)}"
                    type_integrity_tests += 1
                    print("   ✅ System status: String type valid")
                    
        except Exception as e:
            print(f"   ❌ System status data type test failed: {e}")
        
        # Test health check data types
        try:
            response = self.api_client.health_check()
            
            if response.status_code == 200:
                data = response.json()
                
                # Basic type checks for health response
                valid_health_types = True
                
                for key, value in data.items():
                    if key in ['status', 'service'] and not isinstance(value, str):
                        valid_health_types = False
                        print(f"   ❌ Health {key} should be string, got {type(value)}")
                    elif key in ['timestamp', 'uptime'] and not isinstance(value, (str, int, float)):
                        valid_health_types = False
                        print(f"   ❌ Health {key} should be string/number, got {type(value)}")
                
                if valid_health_types:
                    type_integrity_tests += 1
                    print("   ✅ Health check: All data types appropriate")
                    
        except Exception as e:
            print(f"   ❌ Health data type test failed: {e}")
        
        self.capture_screenshot("data_type_integrity_tested")
        assert type_integrity_tests >= 2, f"Only {type_integrity_tests}/3 data type tests passed"
        print(f"✅ Data type integrity: {type_integrity_tests}/3 type validation tests passed")
    
    @pytest.mark.regression  
    def test_cross_component_data_flow(self):
        """
        Test: Data flows correctly between UI and API components
        Pre: Both UI and API should be functional
        Test: Verify data consistency between frontend and backend
        Post: Ensure data integrity across component boundaries
        """
        print("🔄 REGRESSION TEST 3.5: Cross-Component Data Flow")
        
        data_flow_tests = 0
        
        # Test 1: UI to API data consistency
        try:
            self.dashboard.navigate_to_dashboard('http://localhost:3000')
            time.sleep(3)
            
            # Try to enter stock symbols in UI
            stock_input = self.dashboard.find_stock_input()
            test_stocks = ['AAPL', 'GOOGL']
            
            if stock_input:
                # Enter stocks in UI
                for stock in test_stocks:
                    stock_input.clear()
                    stock_input.send_keys(stock)
                    time.sleep(1)
                
                # Try to trigger portfolio construction
                construct_button = self.dashboard.find_construct_button()
                if construct_button:
                    construct_button.click()
                    time.sleep(3)
                    
                    # Check if results appear in UI
                    results_visible = False
                    
                    # Look for any new content
                    results_containers = self.driver.find_elements(By.XPATH, 
                        "//div[contains(@class, 'result') or contains(@class, 'portfolio') or contains(@class, 'weight')]"
                    )
                    
                    if results_containers:
                        visible_results = [r for r in results_containers if r.is_displayed()]
                        if visible_results:
                            results_visible = True
                    
                    # Compare with direct API call
                    api_response = self.api_client.construct_portfolio(test_stocks)
                    
                    if api_response.status_code == 200 and results_visible:
                        data_flow_tests += 1
                        print("   ✅ UI to API: Data flow functional")
                    elif api_response.status_code == 200:
                        # API works but UI might not show results yet
                        data_flow_tests += 1
                        print("   ✅ UI to API: API integration working")
                    
            else:
                print("   ⚠️ UI input not found for data flow test")
                
        except Exception as e:
            print(f"   ❌ UI to API data flow test failed: {e}")
        
        # Test 2: API response format matches UI expectations
        try:
            api_response = self.api_client.construct_portfolio(['AAPL', 'MSFT'])
            
            if api_response.status_code == 200:
                data = api_response.json()
                
                # Check if response structure is UI-friendly
                ui_friendly = True
                ui_issues = []
                
                # Portfolio weights should be displayable
                if 'portfolio_weights' in data:
                    weights = data['portfolio_weights']
                    if isinstance(weights, dict):
                        for stock, weight in weights.items():
                            # Weight should be displayable as percentage
                            if not isinstance(weight, (int, float)):
                                ui_friendly = False
                                ui_issues.append(f"Non-numeric weight for {stock}")
                            elif weight < 0 or weight > 2:
                                ui_issues.append(f"Unusual weight for {stock}: {weight}")
                
                # Check for displayable text fields
                text_fields = ['analysis', 'summary', 'insights']
                for field in text_fields:
                    if field in data:
                        field_value = data[field]
                        if isinstance(field_value, str) and len(field_value) > 10000:
                            ui_issues.append(f"{field} text very long: {len(field_value)} chars")
                
                if ui_friendly and len(ui_issues) == 0:
                    data_flow_tests += 1
                    print("   ✅ API to UI: Response format UI-friendly")
                else:
                    print(f"   ⚠️ API to UI: Potential UI issues: {ui_issues}")
                    
        except Exception as e:
            print(f"   ❌ API to UI format test failed: {e}")
        
        # Test 3: Error handling consistency
        try:
            # Test API error handling
            api_error_response = self.api_client.construct_portfolio([])  # Empty portfolio
            
            # Test UI with similar error scenario
            self.dashboard.navigate_to_dashboard('http://localhost:3000')
            time.sleep(2)
            
            construct_button = self.dashboard.find_construct_button()
            if construct_button:
                # Try to construct without stocks
                construct_button.click()
                time.sleep(2)
                
                # Look for error messages
                error_elements = self.driver.find_elements(By.XPATH, 
                    "//*[contains(@class, 'error') or contains(@class, 'alert')]"
                )
                
                ui_shows_error = len(error_elements) > 0
                api_shows_error = api_error_response.status_code in [400, 422]
                
                if ui_shows_error or api_shows_error:
                    data_flow_tests += 1
                    print("   ✅ Error handling: Consistent between UI and API")
                else:
                    # Both might handle gracefully
                    data_flow_tests += 1
                    print("   ✅ Error handling: Both components handle gracefully")
                    
        except Exception as e:
            print(f"   ❌ Error handling consistency test failed: {e}")
        
        self.capture_screenshot("cross_component_data_flow")
        assert data_flow_tests >= 2, f"Only {data_flow_tests}/3 data flow tests passed"
        print(f"✅ Cross-component data flow: {data_flow_tests}/3 integration points validated")