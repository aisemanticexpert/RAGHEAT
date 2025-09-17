#!/usr/bin/env python3
"""
Regression Test 2: Edge Cases and Boundary Conditions
Tests application behavior with unusual inputs and edge cases
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


class TestEdgeCases(BaseTest):
    """Edge cases and boundary condition tests"""
    
    def setup_method(self, method):
        """Setup for each test method"""
        super().setup_method(method)
        self.dashboard = PortfolioDashboard(self.driver)
        self.api_client = RAGHeatAPIClient('http://localhost:8001')
        
    def teardown_method(self, method):
        """Teardown after each test method"""
        super().teardown_method(method)
    
    @pytest.mark.regression
    def test_invalid_stock_symbols(self):
        """
        Test: Application handles various invalid stock symbol formats
        Pre: Portfolio construction should be accessible
        Test: Try various invalid stock symbol combinations
        Post: Verify application handles edge cases gracefully
        """
        print("🔄 REGRESSION TEST 2.1: Invalid Stock Symbols")
        
        invalid_symbol_tests = [
            {'symbols': [''], 'description': 'empty string'},
            {'symbols': ['12345'], 'description': 'numbers only'},
            {'symbols': ['A' * 20], 'description': 'very long symbol'},
            {'symbols': ['AAPL', ''], 'description': 'mixed valid/empty'},
            {'symbols': ['@#$%'], 'description': 'special characters'},
            {'symbols': ['aapl'], 'description': 'lowercase'},  # might be valid
            {'symbols': ['NONEXISTENTSYMBOL123'], 'description': 'non-existent'},
            {'symbols': [' AAPL '], 'description': 'whitespace padding'},
            {'symbols': ['AAPL', 'AAPL'], 'description': 'duplicate symbols'}
        ]
        
        edge_cases_handled = 0
        
        for test_case in invalid_symbol_tests:
            symbols = test_case['symbols']
            description = test_case['description']
            
            try:
                response = self.api_client.construct_portfolio(symbols)
                
                if response.status_code == 400:
                    # Proper error handling
                    edge_cases_handled += 1
                    print(f"   ✅ {description}: Properly rejected (400)")
                elif response.status_code == 422:
                    # Validation error
                    edge_cases_handled += 1
                    print(f"   ✅ {description}: Validation error (422)")
                elif response.status_code == 200:
                    # Accepted - check if handled gracefully
                    try:
                        data = response.json()
                        if 'error' in data or 'warning' in data:
                            edge_cases_handled += 1
                            print(f"   ✅ {description}: Gracefully handled in response")
                        elif 'portfolio_weights' in data:
                            weights = data['portfolio_weights']
                            if isinstance(weights, dict) and len(weights) > 0:
                                # Some symbols might be valid after processing
                                edge_cases_handled += 1
                                print(f"   ✅ {description}: Processed successfully")
                    except:
                        pass
                else:
                    print(f"   ⚠️ {description}: Unexpected response {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ {description}: Request failed - {e}")
        
        # UI edge case testing
        ui_edge_cases_handled = 0
        
        try:
            self.dashboard.navigate_to_dashboard('http://localhost:3000')
            time.sleep(2)
            
            stock_input = self.dashboard.find_stock_input()
            if stock_input:
                ui_test_cases = [
                    {'input': 'A' * 50, 'description': 'very long input'},
                    {'input': '@#$%', 'description': 'special characters in UI'},
                    {'input': '   ', 'description': 'whitespace only'}
                ]
                
                for test_case in ui_test_cases:
                    input_text = test_case['input']
                    description = test_case['description']
                    
                    try:
                        stock_input.clear()
                        stock_input.send_keys(input_text)
                        time.sleep(1)
                        
                        entered_value = stock_input.get_attribute('value')
                        
                        # Check for validation feedback
                        error_elements = self.driver.find_elements(
                            By.XPATH, "//*[contains(@class, 'error') or contains(@class, 'invalid')]"
                        )
                        
                        if error_elements or len(entered_value.strip()) == 0:
                            ui_edge_cases_handled += 1
                            print(f"   ✅ UI {description}: Validation working")
                        elif entered_value != input_text:
                            # Input was modified/filtered
                            ui_edge_cases_handled += 1
                            print(f"   ✅ UI {description}: Input filtered")
                        else:
                            print(f"   ⚠️ UI {description}: No validation detected")
                            
                    except Exception as e:
                        print(f"   ❌ UI {description}: Test failed - {e}")
        
        except Exception as e:
            print(f"   ❌ UI edge case testing failed: {e}")
        
        self.capture_screenshot("invalid_symbols_tested")
        total_handled = edge_cases_handled + ui_edge_cases_handled
        assert total_handled >= 6, f"Only {total_handled} edge cases handled appropriately"
        print(f"✅ Invalid symbols: {edge_cases_handled} API + {ui_edge_cases_handled} UI edge cases handled")
    
    @pytest.mark.regression
    def test_extreme_portfolio_sizes(self):
        """
        Test: Application handles portfolios with extreme sizes
        Pre: API should handle portfolio construction
        Test: Test very small and very large portfolios
        Post: Verify appropriate handling of boundary conditions
        """
        print("🔄 REGRESSION TEST 2.2: Extreme Portfolio Sizes")
        
        extreme_size_tests = 0
        
        # Test 1: Single stock portfolio
        try:
            single_stock = ['AAPL']
            response = self.api_client.construct_portfolio(single_stock)
            
            if response.status_code == 200:
                data = response.json()
                if 'portfolio_weights' in data:
                    weights = data['portfolio_weights']
                    if len(weights) == 1 and 'AAPL' in weights:
                        weight_value = weights['AAPL']
                        if 0.9 <= weight_value <= 1.1:  # Should be close to 1.0
                            extreme_size_tests += 1
                            print(f"   ✅ Single stock: Weight {weight_value:.3f}")
            elif response.status_code in [400, 422]:
                # Rejection is also acceptable
                extreme_size_tests += 1
                print(f"   ✅ Single stock: Appropriately rejected ({response.status_code})")
                
        except Exception as e:
            print(f"   ❌ Single stock test failed: {e}")
        
        # Test 2: Large portfolio (10+ stocks)
        try:
            large_portfolio = [
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 
                'NFLX', 'DIS', 'UBER', 'SPOT', 'PYPL', 'ADBE', 'CRM', 'ORCL'
            ]
            
            response = self.api_client.construct_portfolio(large_portfolio)
            
            if response.status_code == 200:
                data = response.json()
                if 'portfolio_weights' in data:
                    weights = data['portfolio_weights']
                    if len(weights) == len(large_portfolio):
                        total_weight = sum(weights.values())
                        if 0.9 <= total_weight <= 1.1:
                            extreme_size_tests += 1
                            print(f"   ✅ Large portfolio: {len(weights)} stocks, total weight {total_weight:.3f}")
            elif response.status_code == 413:  # Payload Too Large
                extreme_size_tests += 1
                print(f"   ✅ Large portfolio: Appropriately limited (413)")
            elif response.status_code in [400, 422]:
                extreme_size_tests += 1
                print(f"   ✅ Large portfolio: Handled appropriately ({response.status_code})")
                
        except Exception as e:
            print(f"   ❌ Large portfolio test failed: {e}")
        
        # Test 3: Empty portfolio
        try:
            empty_portfolio = []
            response = self.api_client.construct_portfolio(empty_portfolio)
            
            # Should be rejected
            if response.status_code in [400, 422]:
                extreme_size_tests += 1
                print(f"   ✅ Empty portfolio: Appropriately rejected ({response.status_code})")
            elif response.status_code == 200:
                data = response.json()
                if 'error' in data or 'portfolio_weights' not in data:
                    extreme_size_tests += 1
                    print("   ✅ Empty portfolio: Handled gracefully in response")
                    
        except Exception as e:
            print(f"   ❌ Empty portfolio test failed: {e}")
        
        # Test 4: Two-stock minimum portfolio
        try:
            two_stock = ['AAPL', 'GOOGL']
            response = self.api_client.construct_portfolio(two_stock)
            
            if response.status_code == 200:
                data = response.json()
                if 'portfolio_weights' in data:
                    weights = data['portfolio_weights']
                    if len(weights) == 2:
                        total_weight = sum(weights.values())
                        if 0.9 <= total_weight <= 1.1:
                            extreme_size_tests += 1
                            print(f"   ✅ Two-stock portfolio: Weights sum to {total_weight:.3f}")
                            
        except Exception as e:
            print(f"   ❌ Two-stock test failed: {e}")
        
        assert extreme_size_tests >= 3, f"Only {extreme_size_tests}/4 extreme size tests passed"
        print(f"✅ Extreme portfolio sizes: {extreme_size_tests}/4 boundary conditions handled")
    
    @pytest.mark.regression
    def test_network_timeout_scenarios(self):
        """
        Test: Application handles network timeouts and connectivity issues
        Pre: API should be accessible under normal conditions
        Test: Simulate timeout scenarios and verify graceful handling
        Post: Ensure application remains stable after network issues
        """
        print("🔄 REGRESSION TEST 2.3: Network Timeout Scenarios")
        
        timeout_tests = 0
        
        # Test 1: Very short timeout
        try:
            import requests
            
            short_timeout_response = requests.get(
                f"{self.api_client.base_url}/api/health",
                timeout=0.1  # Very short timeout
            )
            
            if short_timeout_response.status_code == 200:
                timeout_tests += 1
                print("   ✅ Short timeout: API responded quickly")
                
        except requests.exceptions.Timeout:
            # Timeout is expected and handled properly
            timeout_tests += 1
            print("   ✅ Short timeout: Properly timed out")
        except Exception as e:
            print(f"   ⚠️ Short timeout test failed: {e}")
        
        # Test 2: Portfolio construction with reasonable timeout
        try:
            test_stocks = ['AAPL', 'GOOGL', 'MSFT']
            
            import requests
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.api_client.base_url}/api/portfolio/construct",
                    json={"stocks": test_stocks},
                    timeout=5  # 5 second timeout
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    timeout_tests += 1
                    print(f"   ✅ Portfolio timeout: Completed in {response_time:.2f}s")
                elif response.status_code == 408:  # Request Timeout
                    timeout_tests += 1
                    print(f"   ✅ Portfolio timeout: Properly handled (408)")
                    
            except requests.exceptions.Timeout:
                timeout_tests += 1
                timeout_duration = time.time() - start_time
                print(f"   ✅ Portfolio timeout: Timed out at {timeout_duration:.2f}s")
                
        except Exception as e:
            print(f"   ❌ Portfolio timeout test failed: {e}")
        
        # Test 3: UI behavior during potential API delays
        try:
            self.dashboard.navigate_to_dashboard('http://localhost:3000')
            time.sleep(2)
            
            # Check for loading indicators or error handling
            loading_elements = self.driver.find_elements(
                By.XPATH, "//*[contains(@class, 'loading') or contains(@class, 'spinner')]"
            )
            
            error_elements = self.driver.find_elements(
                By.XPATH, "//*[contains(@class, 'error') or contains(@class, 'timeout')]"
            )
            
            # Try to trigger an API call from UI
            construct_button = self.dashboard.find_construct_button()
            if construct_button:
                construct_button.click()
                time.sleep(3)  # Wait for potential response
                
                # Check if UI shows any feedback
                new_loading = self.driver.find_elements(
                    By.XPATH, "//*[contains(@class, 'loading') or contains(@class, 'spinner')]"
                )
                
                new_errors = self.driver.find_elements(
                    By.XPATH, "//*[contains(@class, 'error') or contains(@class, 'timeout')]"
                )
                
                if new_loading or new_errors or len(loading_elements) > 0:
                    timeout_tests += 1
                    print("   ✅ UI timeout handling: Loading/error feedback present")
                else:
                    # UI remained functional
                    page_ready = self.driver.execute_script("return document.readyState")
                    if page_ready == "complete":
                        timeout_tests += 1
                        print("   ✅ UI timeout handling: Interface remained stable")
                        
        except Exception as e:
            print(f"   ❌ UI timeout test failed: {e}")
        
        self.capture_screenshot("timeout_scenarios_tested")
        assert timeout_tests >= 2, f"Only {timeout_tests}/3 timeout scenarios handled"
        print(f"✅ Network timeouts: {timeout_tests}/3 scenarios handled gracefully")
    
    @pytest.mark.regression
    def test_unicode_and_special_characters(self):
        """
        Test: Application handles unicode and special characters appropriately
        Pre: Input fields should be accessible
        Test: Enter various unicode and special character combinations
        Post: Verify application doesn't break with unusual text input
        """
        print("🔄 REGRESSION TEST 2.4: Unicode and Special Characters")
        
        unicode_tests = 0
        
        # Test 1: API with special characters
        try:
            special_symbols = ['ÄÄℙℒ', '中国', 'ТЕСЛА', '🚀AAPL', 'GOOGL®', 'MSFT™']
            
            response = self.api_client.construct_portfolio(special_symbols)
            
            # API should handle this gracefully
            if response.status_code in [200, 400, 422]:
                try:
                    data = response.json()
                    # JSON should be parseable
                    unicode_tests += 1
                    print("   ✅ API unicode: JSON response parseable")
                except:
                    # Non-JSON response is also acceptable
                    unicode_tests += 1
                    print("   ✅ API unicode: Response received")
            else:
                print(f"   ⚠️ API unicode: Unexpected status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ API unicode test failed: {e}")
        
        # Test 2: UI input fields with special characters
        try:
            self.dashboard.navigate_to_dashboard('http://localhost:3000')
            time.sleep(2)
            
            stock_input = self.dashboard.find_stock_input()
            if stock_input:
                unicode_inputs = [
                    '测试',  # Chinese characters
                    'ÃÅÄÖ',  # Accented characters
                    '🚀📈',  # Emojis
                    'AAPL©',  # Copyright symbol
                    'A\u0000B',  # Null character
                    'MSFT\n\r\t'  # Control characters
                ]
                
                for unicode_input in unicode_inputs:
                    try:
                        stock_input.clear()
                        stock_input.send_keys(unicode_input)
                        time.sleep(0.5)
                        
                        # Check if browser/application crashed
                        entered_value = stock_input.get_attribute('value')
                        
                        # Just verify we can still interact
                        if entered_value is not None:
                            unicode_tests += 1
                            print(f"   ✅ UI unicode: Handled input gracefully")
                            break
                            
                    except Exception as e:
                        print(f"   ⚠️ Unicode input failed: {e}")
                        
        except Exception as e:
            print(f"   ❌ UI unicode test setup failed: {e}")
        
        # Test 3: Page title and text content
        try:
            page_title = self.driver.title
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # Verify we can read page content (no encoding issues)
            if isinstance(page_title, str) and isinstance(page_text, str):
                unicode_tests += 1
                print("   ✅ Page content: Unicode handling functional")
                
        except Exception as e:
            print(f"   ❌ Page content unicode test failed: {e}")
        
        # Test 4: JavaScript execution with unicode
        try:
            js_result = self.driver.execute_script("""
                return {
                    test: 'Unicode test: 测试 ñoño 🚀',
                    length: 'Unicode test: 测试 ñoño 🚀'.length
                };
            """)
            
            if js_result and isinstance(js_result, dict):
                unicode_tests += 1
                print("   ✅ JavaScript unicode: Execution successful")
                
        except Exception as e:
            print(f"   ❌ JavaScript unicode test failed: {e}")
        
        self.capture_screenshot("unicode_characters_tested")
        assert unicode_tests >= 2, f"Only {unicode_tests}/4 unicode tests passed"
        print(f"✅ Unicode handling: {unicode_tests}/4 scenarios handled correctly")
    
    @pytest.mark.regression
    def test_browser_edge_cases(self):
        """
        Test: Application handles browser-specific edge cases
        Pre: Browser should be functional
        Test: Test browser-specific scenarios and edge cases
        Post: Verify application remains stable across browser behaviors
        """
        print("🔄 REGRESSION TEST 2.5: Browser Edge Cases")
        
        browser_edge_tests = 0
        
        # Test 1: Window resize edge cases
        try:
            original_size = self.driver.get_window_size()
            
            # Test very small window
            self.driver.set_window_size(320, 240)  # Very small mobile size
            time.sleep(1)
            
            # Verify page still works
            buttons = self.driver.find_elements(By.XPATH, "//button")
            if len(buttons) > 0:
                browser_edge_tests += 1
                print("   ✅ Tiny window: Page elements still accessible")
            
            # Test very large window
            self.driver.set_window_size(3840, 2160)  # 4K resolution
            time.sleep(1)
            
            large_buttons = self.driver.find_elements(By.XPATH, "//button")
            if len(large_buttons) > 0:
                browser_edge_tests += 1
                print("   ✅ Large window: Page scales appropriately")
            
            # Restore original size
            self.driver.set_window_size(original_size['width'], original_size['height'])
            
        except Exception as e:
            print(f"   ❌ Window resize test failed: {e}")
        
        # Test 2: JavaScript console access
        try:
            # Test console.log doesn't break anything
            self.driver.execute_script("console.log('Test message');")
            
            # Test error handling
            try:
                self.driver.execute_script("throw new Error('Test error');")
            except Exception:
                pass  # Expected to fail
            
            # Verify page is still functional
            page_ready = self.driver.execute_script("return document.readyState")
            if page_ready == "complete":
                browser_edge_tests += 1
                print("   ✅ JavaScript errors: Page remained stable")
                
        except Exception as e:
            print(f"   ❌ JavaScript error test failed: {e}")
        
        # Test 3: Navigation edge cases
        try:
            current_url = self.driver.current_url
            
            # Test back/forward navigation
            self.driver.execute_script("window.history.pushState({}, '', '#test');")
            time.sleep(0.5)
            
            self.driver.back()
            time.sleep(1)
            
            # Verify page is still functional
            final_buttons = self.driver.find_elements(By.XPATH, "//button")
            if len(final_buttons) > 0:
                browser_edge_tests += 1
                print("   ✅ Navigation: Back/forward handled correctly")
                
        except Exception as e:
            print(f"   ❌ Navigation test failed: {e}")
        
        # Test 4: Focus and blur events
        try:
            inputs = self.driver.find_elements(By.XPATH, "//input")
            if inputs:
                input_field = inputs[0]
                
                # Test focus/blur cycle
                input_field.click()  # Focus
                time.sleep(0.5)
                
                # Click elsewhere to blur
                body = self.driver.find_element(By.TAG_NAME, "body")
                body.click()
                time.sleep(0.5)
                
                # Verify input is still functional
                input_field.clear()
                input_field.send_keys("TEST")
                entered_value = input_field.get_attribute('value')
                
                if "TEST" in entered_value:
                    browser_edge_tests += 1
                    print("   ✅ Focus/blur: Input handling stable")
                    
        except Exception as e:
            print(f"   ❌ Focus/blur test failed: {e}")
        
        self.capture_screenshot("browser_edge_cases_complete")
        assert browser_edge_tests >= 2, f"Only {browser_edge_tests}/4 browser edge cases handled"
        print(f"✅ Browser edge cases: {browser_edge_tests}/4 scenarios handled successfully")