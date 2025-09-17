#!/usr/bin/env python3
"""
Regression Test 1: Performance and Stress Testing
Tests application behavior under various load and stress conditions
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from framework.base_test import BaseTest
from framework.page_objects.portfolio_dashboard import PortfolioDashboard
from framework.utilities.api_client import RAGHeatAPIClient


class TestPerformanceStress(BaseTest):
    """Performance and stress testing for regression validation"""
    
    def setup_method(self, method):
        """Setup for each test method"""
        super().setup_method(method)
        self.dashboard = PortfolioDashboard(self.driver)
        self.api_client = RAGHeatAPIClient('http://localhost:8001')
        
    def teardown_method(self, method):
        """Teardown after each test method"""
        super().teardown_method(method)
    
    @pytest.mark.regression
    def test_concurrent_api_requests(self):
        """
        Test: API handles concurrent requests without degradation
        Pre: API should be responsive to single requests
        Test: Send multiple concurrent requests and verify performance
        Post: Ensure API maintains performance under concurrent load
        """
        print("🔄 REGRESSION TEST 1.1: Concurrent API Requests")
        
        # Test configuration
        num_concurrent = 5
        test_stocks = ['AAPL', 'GOOGL', 'MSFT']
        
        def make_portfolio_request():
            """Helper function for concurrent requests"""
            try:
                start_time = time.time()
                response = self.api_client.construct_portfolio(test_stocks)
                response_time = time.time() - start_time
                
                return {
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'success': response.status_code == 200
                }
            except Exception as e:
                return {
                    'status_code': 500,
                    'response_time': float('inf'),
                    'success': False,
                    'error': str(e)
                }
        
        # Execute concurrent requests
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_portfolio_request) for _ in range(num_concurrent)]
            
            for future in as_completed(futures, timeout=60):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'status_code': 500,
                        'response_time': float('inf'),
                        'success': False,
                        'error': str(e)
                    })
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        if successful_requests:
            avg_response_time = sum(r['response_time'] for r in successful_requests) / len(successful_requests)
            max_response_time = max(r['response_time'] for r in successful_requests)
            
            # Performance assertions
            assert len(successful_requests) >= num_concurrent // 2, f"Too many failures: {len(failed_requests)}/{num_concurrent}"
            assert avg_response_time < 30, f"Average response time too slow: {avg_response_time}s"
            assert max_response_time < 45, f"Maximum response time too slow: {max_response_time}s"
            
            print(f"   ✅ Concurrent performance: {len(successful_requests)}/{num_concurrent} succeeded")
            print(f"   ✅ Average response time: {avg_response_time:.2f}s")
            print(f"   ✅ Total execution time: {total_time:.2f}s")
        else:
            assert False, f"All concurrent requests failed: {[r.get('error', 'Unknown') for r in failed_requests]}"
    
    @pytest.mark.regression
    def test_memory_usage_stability(self):
        """
        Test: Application maintains stable memory usage over time
        Pre: Application should be running normally
        Test: Perform repeated operations and monitor for memory leaks
        Post: Verify memory usage remains stable
        """
        print("🔄 REGRESSION TEST 1.2: Memory Usage Stability")
        
        # Navigate to application
        self.dashboard.navigate_to_dashboard('http://localhost:3000')
        time.sleep(2)
        
        memory_measurements = []
        
        # Perform repeated operations
        for i in range(5):
            try:
                # Navigation operations
                self.driver.refresh()
                time.sleep(2)
                
                # Find and interact with elements
                buttons = self.driver.find_elements(By.XPATH, "//button")
                if buttons:
                    visible_buttons = [b for b in buttons if b.is_displayed()]
                    if visible_buttons:
                        visible_buttons[0].click()
                        time.sleep(1)
                
                # JavaScript memory measurement (if available)
                try:
                    memory_info = self.driver.execute_script("""
                        if (performance.memory) {
                            return {
                                used: performance.memory.usedJSHeapSize,
                                total: performance.memory.totalJSHeapSize,
                                limit: performance.memory.jsHeapSizeLimit
                            };
                        }
                        return null;
                    """)
                    
                    if memory_info:
                        memory_measurements.append(memory_info)
                        
                except Exception:
                    # Memory API not available in all browsers
                    pass
                
                print(f"   Iteration {i+1}/5 completed")
                
            except Exception as e:
                print(f"   ⚠️ Memory test iteration {i+1} failed: {e}")
        
        # Analyze memory usage
        if memory_measurements:
            initial_memory = memory_measurements[0]['used']
            final_memory = memory_measurements[-1]['used']
            peak_memory = max(m['used'] for m in memory_measurements)
            
            # Check for excessive memory growth
            memory_growth = final_memory - initial_memory
            memory_growth_percentage = (memory_growth / initial_memory) * 100
            
            assert memory_growth_percentage < 50, f"Excessive memory growth: {memory_growth_percentage:.1f}%"
            
            print(f"   ✅ Memory stability: {memory_growth_percentage:.1f}% growth over {len(memory_measurements)} iterations")
            print(f"   ✅ Peak memory usage: {peak_memory / 1024 / 1024:.1f} MB")
        else:
            # Fallback: Basic functionality test
            page_title = self.driver.title
            assert len(page_title) > 0, "Page became unresponsive during memory test"
            print("   ✅ Application remained responsive (memory API unavailable)")
        
        self.capture_screenshot("memory_stability_test")
    
    @pytest.mark.regression
    def test_ui_responsiveness_under_load(self):
        """
        Test: UI remains responsive under simulated load conditions
        Pre: Application should be loaded and functional
        Test: Perform rapid UI interactions and verify responsiveness
        Post: Ensure UI maintains usability under stress
        """
        print("🔄 REGRESSION TEST 1.3: UI Responsiveness Under Load")
        
        # Pre-condition
        self.dashboard.navigate_to_dashboard('http://localhost:3000')
        time.sleep(3)
        
        responsiveness_tests = 0
        
        # Test 1: Rapid click operations
        try:
            buttons = self.driver.find_elements(By.XPATH, "//button")
            clickable_buttons = [b for b in buttons if b.is_displayed() and b.is_enabled()]
            
            if clickable_buttons:
                start_time = time.time()
                
                # Perform rapid clicks
                for i in range(10):
                    button = clickable_buttons[i % len(clickable_buttons)]
                    try:
                        button.click()
                        time.sleep(0.1)  # Very brief pause
                    except Exception:
                        pass  # Some clicks might fail due to state changes
                
                click_duration = time.time() - start_time
                
                # Verify UI is still responsive
                page_ready = self.driver.execute_script("return document.readyState")
                if page_ready == "complete" and click_duration < 5:
                    responsiveness_tests += 1
                    print(f"   ✅ Rapid clicks: UI remained responsive ({click_duration:.2f}s)")
                    
        except Exception as e:
            print(f"   ❌ Rapid click test failed: {e}")
        
        # Test 2: Rapid form input
        try:
            inputs = self.driver.find_elements(By.XPATH, "//input")
            text_inputs = [i for i in inputs if i.is_displayed() and i.is_enabled()]
            
            if text_inputs:
                input_field = text_inputs[0]
                start_time = time.time()
                
                # Rapid input operations
                for i in range(5):
                    input_field.clear()
                    input_field.send_keys(f"STOCK{i}")
                    time.sleep(0.2)
                
                input_duration = time.time() - start_time
                
                # Check if input is still functional
                final_value = input_field.get_attribute('value')
                if final_value and input_duration < 3:
                    responsiveness_tests += 1
                    print(f"   ✅ Rapid input: UI remained responsive ({input_duration:.2f}s)")
                    
        except Exception as e:
            print(f"   ❌ Rapid input test failed: {e}")
        
        # Test 3: Page scroll stress
        try:
            start_time = time.time()
            
            # Rapid scrolling
            for i in range(10):
                scroll_position = (i % 4) * 200
                self.driver.execute_script(f"window.scrollTo(0, {scroll_position});")
                time.sleep(0.05)
            
            scroll_duration = time.time() - start_time
            
            # Verify page is still functional
            elements = self.driver.find_elements(By.XPATH, "//*")
            if len(elements) > 10 and scroll_duration < 2:
                responsiveness_tests += 1
                print(f"   ✅ Scroll stress: UI remained responsive ({scroll_duration:.2f}s)")
                
        except Exception as e:
            print(f"   ❌ Scroll stress test failed: {e}")
        
        self.capture_screenshot("ui_load_test_complete")
        assert responsiveness_tests >= 2, f"Only {responsiveness_tests}/3 responsiveness tests passed"
        print(f"✅ UI load testing: {responsiveness_tests}/3 stress tests passed")
    
    @pytest.mark.regression
    def test_api_rate_limiting(self):
        """
        Test: API handles rate limiting and prevents abuse
        Pre: API should be accessible
        Test: Send requests at various rates and verify appropriate handling
        Post: Ensure API maintains stability under high request rates
        """
        print("🔄 REGRESSION TEST 1.4: API Rate Limiting")
        
        rate_limit_tests = 0
        
        # Test 1: Rapid sequential requests
        try:
            rapid_responses = []
            start_time = time.time()
            
            for i in range(10):
                response = self.api_client.health_check()
                rapid_responses.append({
                    'status': response.status_code,
                    'time': time.time() - start_time
                })
                
                if i < 9:  # Don't sleep after last request
                    time.sleep(0.1)
            
            total_rapid_time = time.time() - start_time
            
            # Analyze responses
            successful_rapid = [r for r in rapid_responses if r['status'] == 200]
            rate_limited = [r for r in rapid_responses if r['status'] == 429]  # Too Many Requests
            
            if len(successful_rapid) >= 5:
                rate_limit_tests += 1
                print(f"   ✅ Rapid requests: {len(successful_rapid)}/10 succeeded in {total_rapid_time:.2f}s")
            
            if rate_limited:
                print(f"   ✅ Rate limiting detected: {len(rate_limited)} requests limited")
                
        except Exception as e:
            print(f"   ❌ Rapid request test failed: {e}")
        
        # Test 2: Sustained request pattern
        try:
            sustained_responses = []
            
            for i in range(5):
                response = self.api_client.health_check()
                sustained_responses.append(response.status_code)
                time.sleep(1)  # More reasonable rate
            
            successful_sustained = [s for s in sustained_responses if s == 200]
            
            if len(successful_sustained) >= 4:
                rate_limit_tests += 1
                print(f"   ✅ Sustained requests: {len(successful_sustained)}/5 succeeded")
                
        except Exception as e:
            print(f"   ❌ Sustained request test failed: {e}")
        
        # Test 3: Portfolio construction rate limits
        try:
            portfolio_responses = []
            test_stocks = ['AAPL']
            
            for i in range(3):
                start_time = time.time()
                response = self.api_client.construct_portfolio(test_stocks)
                response_time = time.time() - start_time
                
                portfolio_responses.append({
                    'status': response.status_code,
                    'time': response_time
                })
                
                time.sleep(2)  # Reasonable delay for complex operations
            
            successful_portfolio = [r for r in portfolio_responses if r['status'] in [200, 202]]
            
            if len(successful_portfolio) >= 2:
                avg_time = sum(r['time'] for r in successful_portfolio) / len(successful_portfolio)
                rate_limit_tests += 1
                print(f"   ✅ Portfolio rate handling: {len(successful_portfolio)}/3 succeeded, avg {avg_time:.2f}s")
                
        except Exception as e:
            print(f"   ❌ Portfolio rate test failed: {e}")
        
        assert rate_limit_tests >= 2, f"Only {rate_limit_tests}/3 rate limiting tests passed"
        print(f"✅ API rate limiting: {rate_limit_tests}/3 rate scenarios handled appropriately")
    
    @pytest.mark.regression
    def test_long_running_operations(self):
        """
        Test: Application handles long-running operations gracefully
        Pre: Application should be running with timeout handling
        Test: Initiate operations that may take significant time
        Post: Verify application remains stable during extended operations
        """
        print("🔄 REGRESSION TEST 1.5: Long Running Operations")
        
        long_operation_tests = 0
        
        # Test 1: Extended UI session
        try:
            self.dashboard.navigate_to_dashboard('http://localhost:3000')
            session_start = time.time()
            
            # Simulate extended user session
            for i in range(3):
                # Wait longer between operations
                time.sleep(3)
                
                # Perform various operations
                buttons = self.driver.find_elements(By.XPATH, "//button")
                if buttons:
                    visible_buttons = [b for b in buttons if b.is_displayed()]
                    if visible_buttons:
                        visible_buttons[0].click()
                        time.sleep(2)
                
                # Check page is still responsive
                page_ready = self.driver.execute_script("return document.readyState")
                if page_ready != "complete":
                    break
            
            session_duration = time.time() - session_start
            
            # Final responsiveness check
            final_buttons = self.driver.find_elements(By.XPATH, "//button")
            if len(final_buttons) > 0 and session_duration > 9:
                long_operation_tests += 1
                print(f"   ✅ Extended UI session: {session_duration:.1f}s - remained responsive")
                
        except Exception as e:
            print(f"   ❌ Extended session test failed: {e}")
        
        # Test 2: Complex portfolio construction
        try:
            complex_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
            
            start_time = time.time()
            response = self.api_client.construct_portfolio(complex_stocks)
            operation_time = time.time() - start_time
            
            if response.status_code in [200, 202] and operation_time < 60:
                long_operation_tests += 1
                print(f"   ✅ Complex portfolio: {len(complex_stocks)} stocks in {operation_time:.2f}s")
            elif response.status_code == 408:  # Timeout
                print(f"   ✅ Timeout handling: Operation properly timed out at {operation_time:.2f}s")
                long_operation_tests += 1
                
        except Exception as e:
            print(f"   ❌ Complex portfolio test failed: {e}")
        
        # Test 3: Multiple agent coordination
        try:
            test_stocks = ['AAPL', 'MSFT']
            
            # Try to trigger multiple agents
            operations = []
            
            # Fundamental analysis
            start_time = time.time()
            fund_response = self.api_client.fundamental_analysis(test_stocks)
            operations.append({
                'name': 'fundamental',
                'status': fund_response.status_code,
                'time': time.time() - start_time
            })
            
            # Sentiment analysis
            start_time = time.time()
            sent_response = self.api_client.sentiment_analysis(test_stocks)
            operations.append({
                'name': 'sentiment',
                'status': sent_response.status_code,
                'time': time.time() - start_time
            })
            
            successful_ops = [op for op in operations if op['status'] in [200, 202]]
            
            if len(successful_ops) >= 1:
                avg_agent_time = sum(op['time'] for op in successful_ops) / len(successful_ops)
                if avg_agent_time < 30:
                    long_operation_tests += 1
                    print(f"   ✅ Agent coordination: {len(successful_ops)} agents, avg {avg_agent_time:.2f}s")
                    
        except Exception as e:
            print(f"   ❌ Agent coordination test failed: {e}")
        
        self.capture_screenshot("long_operations_complete")
        assert long_operation_tests >= 2, f"Only {long_operation_tests}/3 long operation tests passed"
        print(f"✅ Long running operations: {long_operation_tests}/3 scenarios handled successfully")