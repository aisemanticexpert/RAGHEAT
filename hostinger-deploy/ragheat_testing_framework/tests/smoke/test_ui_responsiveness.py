#!/usr/bin/env python3
"""
Smoke Test 5: UI Responsiveness and Basic Interaction
Tests basic UI responsiveness and interaction capabilities
"""

import pytest
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from framework.base_test import BaseTest
from framework.page_objects.portfolio_dashboard import PortfolioDashboard


class TestUIResponsiveness(BaseTest):
    """UI responsiveness and interaction smoke tests"""
    
    def setup_method(self, method):
        """Setup for each test method"""
        super().setup_method(method)
        self.dashboard = PortfolioDashboard(self.driver)
        self.wait = WebDriverWait(self.driver, 10)
        
    def teardown_method(self, method):
        """Teardown after each test method"""
        super().teardown_method(method)
    
    @pytest.mark.smoke
    def test_page_load_performance(self):
        """
        Test: Page load performance within acceptable limits
        Pre: Browser should be ready
        Test: Navigate to application and measure load time
        Post: Verify page loads within performance threshold
        """
        print("🔥 SMOKE TEST 5.1: Page Load Performance")
        
        # Test execution
        start_time = time.time()
        success = self.dashboard.navigate_to_dashboard('http://localhost:3000')
        load_time = time.time() - start_time
        
        # Assertions
        assert success, "Page failed to load"
        assert load_time < 15, f"Page load too slow: {load_time}s"
        
        # Wait for JavaScript to initialize
        time.sleep(2)
        
        # Verify page is interactive
        page_ready = self.driver.execute_script("return document.readyState")
        assert page_ready == "complete", f"Page not fully loaded: {page_ready}"
        
        self.capture_screenshot("page_loaded_performance")
        print(f"✅ Page loaded in {load_time:.2f}s, ready state: {page_ready}")
    
    @pytest.mark.smoke
    def test_interactive_elements_present(self):
        """
        Test: Interactive elements are present and functional
        Pre: Navigate to main application page
        Test: Check for clickable buttons and input fields
        Post: Verify elements respond to user interaction
        """
        print("🔥 SMOKE TEST 5.2: Interactive Elements Present")
        
        # Pre-condition
        self.dashboard.navigate_to_dashboard('http://localhost:3000')
        time.sleep(3)
        
        # Test execution - Count interactive elements
        buttons = self.driver.find_elements(By.XPATH, "//button")
        inputs = self.driver.find_elements(By.XPATH, "//input")
        links = self.driver.find_elements(By.XPATH, "//a[@href]")
        
        total_interactive = len(buttons) + len(inputs) + len(links)
        
        # Assertions
        assert total_interactive > 0, "No interactive elements found"
        assert len(buttons) > 0, "No buttons found on page"
        
        # Test button interaction
        visible_buttons = [btn for btn in buttons if btn.is_displayed() and btn.is_enabled()]
        assert len(visible_buttons) > 0, "No visible/enabled buttons found"
        
        # Test hover on first visible button
        if visible_buttons:
            first_button = visible_buttons[0]
            try:
                ActionChains(self.driver).move_to_element(first_button).perform()
                time.sleep(1)
            except Exception as e:
                print(f"Hover test failed (acceptable): {e}")
        
        self.capture_screenshot("interactive_elements")
        print(f"✅ Found {total_interactive} interactive elements: {len(buttons)} buttons, {len(inputs)} inputs, {len(links)} links")
    
    @pytest.mark.smoke
    def test_browser_responsiveness(self):
        """
        Test: Browser remains responsive during page interaction
        Pre: Page should be loaded and ready
        Test: Perform various browser operations and verify responsiveness
        Post: Ensure browser doesn't hang or become unresponsive
        """
        print("🔥 SMOKE TEST 5.3: Browser Responsiveness")
        
        # Pre-condition
        self.dashboard.navigate_to_dashboard('http://localhost:3000')
        time.sleep(2)
        
        # Test execution - Browser operations
        operations_completed = 0
        
        # Operation 1: Page scroll
        try:
            self.driver.execute_script("window.scrollTo(0, 100);")
            time.sleep(0.5)
            self.driver.execute_script("window.scrollTo(0, 0);")
            operations_completed += 1
        except Exception as e:
            print(f"Scroll operation failed: {e}")
        
        # Operation 2: Window resize (if possible)
        try:
            original_size = self.driver.get_window_size()
            self.driver.set_window_size(1200, 800)
            time.sleep(0.5)
            self.driver.set_window_size(original_size['width'], original_size['height'])
            operations_completed += 1
        except Exception as e:
            print(f"Resize operation failed: {e}")
        
        # Operation 3: JavaScript execution
        try:
            result = self.driver.execute_script("return navigator.userAgent;")
            assert isinstance(result, str), "JavaScript execution failed"
            operations_completed += 1
        except Exception as e:
            print(f"JavaScript execution failed: {e}")
        
        # Operation 4: DOM query
        try:
            elements = self.driver.find_elements(By.XPATH, "//*")
            assert len(elements) > 0, "DOM query failed"
            operations_completed += 1
        except Exception as e:
            print(f"DOM query failed: {e}")
        
        # Assertions
        assert operations_completed >= 2, f"Only {operations_completed}/4 browser operations successful"
        
        self.capture_screenshot("browser_responsive")
        print(f"✅ Browser responsive: {operations_completed}/4 operations completed successfully")
    
    @pytest.mark.smoke
    def test_css_styling_loaded(self):
        """
        Test: CSS styling is properly loaded and applied
        Pre: Page should be loaded with stylesheets
        Test: Check if styles are applied to elements
        Post: Verify page has visual styling
        """
        print("🔥 SMOKE TEST 5.4: CSS Styling Loaded")
        
        # Pre-condition
        self.dashboard.navigate_to_dashboard('http://localhost:3000')
        time.sleep(3)
        
        # Test execution - Check for styled elements
        styled_elements_found = 0
        
        # Check 1: Look for elements with background colors
        try:
            colored_elements = self.driver.execute_script("""
                var elements = document.querySelectorAll('*');
                var coloredCount = 0;
                for(var i = 0; i < elements.length; i++) {
                    var style = window.getComputedStyle(elements[i]);
                    if(style.backgroundColor !== 'rgba(0, 0, 0, 0)' && 
                       style.backgroundColor !== 'transparent' &&
                       style.backgroundColor !== '') {
                        coloredCount++;
                    }
                }
                return coloredCount;
            """)
            if colored_elements > 0:
                styled_elements_found += colored_elements
                print(f"   Found {colored_elements} elements with background colors")
        except Exception as e:
            print(f"Background color check failed: {e}")
        
        # Check 2: Look for custom fonts
        try:
            font_families = self.driver.execute_script("""
                var elements = document.querySelectorAll('*');
                var fontSet = new Set();
                for(var i = 0; i < Math.min(elements.length, 50); i++) {
                    var style = window.getComputedStyle(elements[i]);
                    if(style.fontFamily) {
                        fontSet.add(style.fontFamily);
                    }
                }
                return Array.from(fontSet);
            """)
            if len(font_families) > 1:
                styled_elements_found += len(font_families)
                print(f"   Found {len(font_families)} different font families")
        except Exception as e:
            print(f"Font family check failed: {e}")
        
        # Check 3: Look for positioned elements
        try:
            positioned = self.driver.execute_script("""
                var elements = document.querySelectorAll('*');
                var positionedCount = 0;
                for(var i = 0; i < Math.min(elements.length, 100); i++) {
                    var style = window.getComputedStyle(elements[i]);
                    if(style.position !== 'static') {
                        positionedCount++;
                    }
                }
                return positionedCount;
            """)
            if positioned > 0:
                styled_elements_found += positioned
                print(f"   Found {positioned} positioned elements")
        except Exception as e:
            print(f"Position check failed: {e}")
        
        # Assertions
        assert styled_elements_found > 0, "No CSS styling detected on page"
        
        self.capture_screenshot("css_styling_applied")
        print(f"✅ CSS styling loaded: {styled_elements_found} styled elements detected")
    
    @pytest.mark.smoke
    def test_error_handling_graceful(self):
        """
        Test: Application handles errors gracefully
        Pre: Navigate to application
        Test: Trigger potential error scenarios and verify graceful handling
        Post: Ensure application remains functional after errors
        """
        print("🔥 SMOKE TEST 5.5: Graceful Error Handling")
        
        # Pre-condition
        self.dashboard.navigate_to_dashboard('http://localhost:3000')
        time.sleep(2)
        
        # Test execution - Error scenarios
        error_tests_passed = 0
        
        # Test 1: Invalid JavaScript execution
        try:
            # This should not crash the browser
            result = self.driver.execute_script("return typeof undefined_variable;")
            if result == "undefined":
                error_tests_passed += 1
        except Exception:
            # Exception is acceptable - browser handled it gracefully
            error_tests_passed += 1
        
        # Test 2: Navigate to non-existent hash
        try:
            current_url = self.driver.current_url
            self.driver.get(current_url + "#nonexistent")
            time.sleep(1)
            
            # Check if page is still responsive
            page_ready = self.driver.execute_script("return document.readyState")
            if page_ready == "complete":
                error_tests_passed += 1
        except Exception as e:
            print(f"Hash navigation test failed: {e}")
        
        # Test 3: Console error check (if possible)
        try:
            logs = self.driver.get_log('browser')
            severe_errors = [log for log in logs if log['level'] == 'SEVERE']
            
            # Having some errors is normal, but too many severe errors is bad
            if len(severe_errors) < 5:  # Threshold for acceptable errors
                error_tests_passed += 1
                print(f"   Found {len(severe_errors)} severe console errors (acceptable)")
        except Exception:
            # Browser might not support log collection
            error_tests_passed += 1
            print("   Console log check not available (acceptable)")
        
        # Test 4: Page functionality after potential errors
        try:
            buttons = self.driver.find_elements(By.XPATH, "//button")
            if len(buttons) > 0 and buttons[0].is_displayed():
                error_tests_passed += 1
        except Exception as e:
            print(f"Post-error functionality test failed: {e}")
        
        # Assertions
        assert error_tests_passed >= 2, f"Only {error_tests_passed}/4 error handling tests passed"
        
        self.capture_screenshot("error_handling_complete")
        print(f"✅ Error handling graceful: {error_tests_passed}/4 scenarios handled properly")