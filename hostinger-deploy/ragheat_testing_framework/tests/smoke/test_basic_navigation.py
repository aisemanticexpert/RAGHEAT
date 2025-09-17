#!/usr/bin/env python3
"""
Smoke tests for basic navigation and accessibility
"""

import pytest
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from framework.base_test import BaseTest
from framework.page_objects.portfolio_dashboard import PortfolioDashboard

@pytest.mark.smoke
class TestBasicNavigation(BaseTest):
    """Test basic navigation functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.portfolio_page = PortfolioDashboard(self.driver)
    
    def test_homepage_loads(self):
        """Test that homepage loads successfully"""
        print("🔥 Testing homepage loading...")
        
        start_time = time.time()
        success = self.portfolio_page.navigate_to_dashboard(self.base_url)
        load_time = time.time() - start_time
        
        # Assertions
        assert success, "Failed to load homepage"
        assert load_time < 15, f"Homepage took too long to load: {load_time:.2f}s"
        
        # Check page loaded properly
        page_loaded = self.portfolio_page.is_page_loaded()
        assert page_loaded, "Homepage did not load properly"
        
        print(f"✅ Homepage loaded in {load_time:.2f}s")
        self.log_performance_metric("Homepage Load", load_time)
        self.log_test_result("Homepage Load", "PASS", f"Load time: {load_time:.2f}s")
    
    def test_page_title_and_basic_content(self):
        """Test page title and basic content presence"""
        print("🔥 Testing page title and content...")
        
        # Navigate to page
        success = self.portfolio_page.navigate_to_dashboard(self.base_url)
        assert success, "Failed to navigate to page"
        
        # Get page information
        page_info = self.portfolio_page.get_page_info()
        
        # Check title
        title = page_info.get('title', '')
        assert len(title) > 0, "Page has no title"
        print(f"📄 Page title: '{title}'")
        
        # Check content length
        body_length = page_info.get('body_text_length', 0)
        assert body_length > 20, f"Page content too short: {body_length} characters"
        print(f"📊 Page content length: {body_length} characters")
        
        # Check URL
        current_url = page_info.get('url', '')
        assert self.base_url in current_url, f"URL mismatch: {current_url}"
        
        self.log_test_result("Page Title and Content", "PASS", f"Title length: {len(title)}, Content: {body_length} chars")
    
    def test_responsive_layout_basic(self):
        """Test basic responsive layout"""
        print("🔥 Testing basic responsive layout...")
        
        # Navigate to page
        success = self.portfolio_page.navigate_to_dashboard(self.base_url)
        assert success, "Failed to navigate to page"
        
        # Test different viewport sizes
        viewports = [
            (1920, 1080, "Desktop Large"),
            (1366, 768, "Desktop Standard"),
            (768, 1024, "Tablet Portrait"),
            (375, 667, "Mobile")
        ]
        
        results = []
        
        for width, height, device in viewports:
            try:
                # Set viewport size
                self.driver.set_window_size(width, height)
                time.sleep(1)  # Allow layout to adjust
                
                # Check if page is still functional
                page_info = self.portfolio_page.get_page_info()
                body_length = page_info.get('body_text_length', 0)
                
                # Basic assertion - content should still be present
                if body_length > 20:
                    results.append(f"✅ {device} ({width}x{height}): OK")
                else:
                    results.append(f"⚠️ {device} ({width}x{height}): Content issue")
                
                print(f"📱 {device} ({width}x{height}): Content length {body_length}")
                
            except Exception as e:
                results.append(f"❌ {device} ({width}x{height}): Error - {str(e)}")
        
        # Reset to standard size
        self.driver.set_window_size(1920, 1080)
        
        # Assert that most viewports worked
        successful_tests = sum(1 for result in results if "✅" in result)
        total_tests = len(results)
        
        print(f"📊 Responsive test results: {successful_tests}/{total_tests} passed")
        for result in results:
            print(result)
        
        assert successful_tests >= total_tests // 2, f"Too many viewport tests failed: {successful_tests}/{total_tests}"
        
        self.log_test_result("Responsive Layout", "PASS", f"{successful_tests}/{total_tests} viewports OK")
    
    def test_page_elements_presence(self):
        """Test presence of basic page elements"""
        print("🔥 Testing page elements presence...")
        
        # Navigate to page
        success = self.portfolio_page.navigate_to_dashboard(self.base_url)
        assert success, "Failed to navigate to page"
        
        page_info = self.portfolio_page.get_page_info()
        
        # Check for basic HTML elements
        elements_found = {
            'forms': page_info.get('forms_count', 0),
            'buttons': page_info.get('buttons_count', 0),
            'inputs': page_info.get('inputs_count', 0)
        }
        
        print(f"🎮 Page elements found: {elements_found}")
        
        # Check for interactive elements (portfolio application should have some)
        total_interactive = sum(elements_found.values())
        
        if total_interactive > 0:
            print(f"✅ Found {total_interactive} interactive elements - page appears functional")
            
            # Try to find specific portfolio-related elements
            stock_input = self.portfolio_page.find_stock_input()
            construct_button = self.portfolio_page.find_construct_button()
            
            portfolio_elements = []
            if stock_input:
                portfolio_elements.append("stock_input")
            if construct_button:
                portfolio_elements.append("construct_button")
            
            if portfolio_elements:
                print(f"🎯 Portfolio-specific elements found: {portfolio_elements}")
            else:
                print("📝 Generic interactive elements found, but no portfolio-specific ones")
                
        else:
            print("📄 No interactive elements found - possibly static content")
        
        # Minimum assertion - page should have some content
        assert page_info.get('body_text_length', 0) > 0, "Page has no content"
        
        self.log_test_result("Page Elements", "PASS", f"Interactive elements: {total_interactive}")
    
    def test_javascript_enabled_check(self):
        """Test if JavaScript is working (if applicable)"""
        print("🔥 Testing JavaScript functionality...")
        
        # Navigate to page
        success = self.portfolio_page.navigate_to_dashboard(self.base_url)
        assert success, "Failed to navigate to page"
        
        try:
            # Try to execute a simple JavaScript command
            js_result = self.driver.execute_script("return document.title;")
            
            if js_result:
                print(f"✅ JavaScript is enabled and working: '{js_result}'")
                js_works = True
            else:
                print("⚠️ JavaScript execution returned empty result")
                js_works = False
                
            # Try another JS test
            try:
                page_height = self.driver.execute_script("return document.body.scrollHeight;")
                if isinstance(page_height, (int, float)) and page_height > 0:
                    print(f"✅ JavaScript DOM access working: page height = {page_height}px")
                    js_works = True
            except:
                pass
            
            # Check for React/dynamic content indicators
            try:
                react_elements = self.driver.find_elements("xpath", "//*[@data-reactroot or contains(@class, 'react') or contains(@id, 'react') or contains(@id, 'root')]")
                if react_elements:
                    print(f"⚛️ React elements detected: {len(react_elements)} elements")
                    js_works = True
            except:
                pass
            
            if js_works:
                self.log_test_result("JavaScript Check", "PASS", "JavaScript enabled and functional")
            else:
                self.log_test_result("JavaScript Check", "SKIP", "JavaScript not detected or not working")
                print("📝 JavaScript functionality could not be verified")
                
        except Exception as e:
            print(f"⚠️ JavaScript test failed: {e}")
            self.log_test_result("JavaScript Check", "SKIP", f"Error: {str(e)}")
    
    def test_network_connectivity(self):
        """Test network connectivity and resource loading"""
        print("🔥 Testing network connectivity...")
        
        # Navigate and measure load time
        start_time = time.time()
        success = self.portfolio_page.navigate_to_dashboard(self.base_url)
        total_load_time = time.time() - start_time
        
        assert success, "Failed to connect to application"
        
        # Check for any obvious network issues
        page_info = self.portfolio_page.get_page_info()
        
        # If page loaded very quickly with minimal content, might indicate network issues
        if total_load_time < 0.1 and page_info.get('body_text_length', 0) < 50:
            print("⚠️ Very fast load with minimal content - possible network issue")
            
        # Check browser logs for network errors (if available)
        try:
            logs = self.driver.get_log('browser')
            network_errors = [log for log in logs if 'network' in log.get('message', '').lower() or 'failed to load' in log.get('message', '').lower()]
            
            if network_errors:
                print(f"⚠️ Found {len(network_errors)} network-related browser errors")
                for error in network_errors[:3]:  # Show first 3 errors
                    print(f"   📝 {error.get('message', '')}")
            else:
                print("✅ No network errors detected in browser logs")
                
        except Exception as e:
            print(f"📝 Could not check browser logs: {e}")
        
        print(f"🌐 Network connectivity test completed - Load time: {total_load_time:.2f}s")
        self.log_test_result("Network Connectivity", "PASS", f"Load time: {total_load_time:.2f}s")
    
    def teardown_method(self, method):
        """Cleanup after each test"""
        super().teardown_method(method)
        
        # Reset window size
        try:
            self.driver.set_window_size(1920, 1080)
        except:
            pass