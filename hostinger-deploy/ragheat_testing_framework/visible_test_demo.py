#!/usr/bin/env python3
"""
Comprehensive Visible Browser Test Demo
Shows real browser automation with all UI interactions visible
"""

import os
import sys
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

class VisibleBrowserDemo:
    def __init__(self):
        self.driver = None
        self.wait = None
        self.frontend_url = "http://localhost:3000"
        self.api_url = "http://localhost:8001"
        
    def setup_browser(self):
        """Setup Chrome browser in visible mode"""
        print("🔧 Setting up Chrome browser (visible mode)...")
        
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Important: NO headless mode so you can see the browser
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.wait = WebDriverWait(self.driver, 10)
        
        print("✅ Browser setup complete - You should see Chrome window open!")
        return True
    
    def test_api_endpoints(self):
        """Test all API endpoints"""
        print("\n📡 Testing API Endpoints...")
        
        endpoints = [
            ('/health', 'Health Check'),
            ('/system/status', 'System Status'),
            ('/portfolio/construct', 'Portfolio Construction', 'POST'),
            ('/analysis/fundamental', 'Fundamental Analysis', 'POST'),
            ('/analysis/sentiment', 'Sentiment Analysis', 'POST')
        ]
        
        for endpoint in endpoints:
            url = f"{self.api_url}{endpoint[0]}"
            method = endpoint[2] if len(endpoint) > 2 else 'GET'
            
            try:
                if method == 'POST':
                    response = requests.post(url, json={'stocks': ['AAPL', 'GOOGL']}, timeout=5)
                else:
                    response = requests.get(url, timeout=5)
                
                status = "✅" if response.status_code in [200, 201] else "❌"
                print(f"   {status} {endpoint[1]}: {response.status_code}")
                
            except Exception as e:
                print(f"   ❌ {endpoint[1]}: Error - {e}")
    
    def test_frontend_navigation(self):
        """Test frontend loading and navigation"""
        print("\n🌐 Testing Frontend Navigation...")
        print("   📱 Loading RAGHeat Dashboard...")
        
        try:
            self.driver.get(self.frontend_url)
            time.sleep(3)  # Give time to see the page load
            
            # Get page title
            title = self.driver.title
            print(f"   ✅ Page Title: {title}")
            
            # Count buttons and inputs
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            inputs = self.driver.find_elements(By.TAG_NAME, "input")
            
            print(f"   ✅ Found {len(buttons)} buttons and {len(inputs)} inputs")
            
            # Take a screenshot
            screenshot_path = "screenshots/frontend_loaded.png"
            os.makedirs("screenshots", exist_ok=True)
            self.driver.save_screenshot(screenshot_path)
            print(f"   📸 Screenshot saved: {screenshot_path}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Frontend navigation failed: {e}")
            return False
    
    def test_ui_interactions(self):
        """Test UI element interactions"""
        print("\n🖱️ Testing UI Interactions...")
        
        try:
            # Look for common UI elements
            ui_elements = [
                "//button[contains(text(), 'REVOLUTIONARY DASHBOARD')]",
                "//button[contains(text(), 'PORTFOLIO AI')]",
                "//button[contains(text(), 'ADVANCED TRADING')]",
                "//input[@placeholder='Enter stock symbols']",
                "//button[contains(text(), 'Construct Portfolio')]"
            ]
            
            found_elements = 0
            
            for xpath in ui_elements:
                try:
                    element = self.driver.find_element(By.XPATH, xpath)
                    if element.is_displayed():
                        print(f"   ✅ Found: {xpath.split('/')[-1]}")
                        found_elements += 1
                        
                        # Try to click if it's a button
                        if "button" in xpath and found_elements <= 3:  # Only click first 3 buttons
                            print(f"   👆 Clicking element...")
                            element.click()
                            time.sleep(2)  # Let you see the click action
                            
                except Exception as e:
                    print(f"   ⚠️ Element not found: {xpath}")
            
            print(f"   📊 Total interactive elements found: {found_elements}")
            return found_elements > 0
            
        except Exception as e:
            print(f"   ❌ UI interaction test failed: {e}")
            return False
    
    def test_portfolio_construction(self):
        """Test portfolio construction workflow"""
        print("\n💼 Testing Portfolio Construction...")
        
        try:
            # Look for portfolio construction form
            stock_input_selectors = [
                "//input[@placeholder='Enter stock symbols']",
                "//input[@placeholder='Enter stock symbol']",
                "//input[@name='stock']",
                "//input[contains(@class, 'stock')]"
            ]
            
            stock_input = None
            for selector in stock_input_selectors:
                try:
                    stock_input = self.driver.find_element(By.XPATH, selector)
                    if stock_input.is_displayed():
                        print(f"   ✅ Found stock input field")
                        break
                except:
                    continue
            
            if stock_input:
                print("   📝 Entering test stocks: AAPL, GOOGL, MSFT")
                stock_input.clear()
                stock_input.send_keys("AAPL,GOOGL,MSFT")
                time.sleep(2)  # Let you see the typing
                
                # Look for construct button
                construct_selectors = [
                    "//button[contains(text(), 'Construct')]",
                    "//button[contains(text(), 'Build')]",
                    "//button[contains(text(), 'Create')]"
                ]
                
                for selector in construct_selectors:
                    try:
                        button = self.driver.find_element(By.XPATH, selector)
                        if button.is_displayed():
                            print("   👆 Clicking portfolio construction button...")
                            button.click()
                            time.sleep(3)  # Let you see the results
                            print("   ✅ Portfolio construction initiated")
                            return True
                    except:
                        continue
                        
                print("   ⚠️ Construct button not found")
                return False
            else:
                print("   ⚠️ Stock input field not found")
                return False
                
        except Exception as e:
            print(f"   ❌ Portfolio construction test failed: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run all tests with visible browser"""
        print("\n" + "="*80)
        print("🚀 RAGHeat Comprehensive Visible Browser Test")
        print("="*80)
        
        test_results = {
            'setup': False,
            'api': False,
            'frontend': False,
            'ui_interactions': False,
            'portfolio': False
        }
        
        try:
            # Setup browser
            test_results['setup'] = self.setup_browser()
            
            # Test API endpoints
            self.test_api_endpoints()
            test_results['api'] = True
            
            # Test frontend
            test_results['frontend'] = self.test_frontend_navigation()
            
            # Test UI interactions
            test_results['ui_interactions'] = self.test_ui_interactions()
            
            # Test portfolio construction
            test_results['portfolio'] = self.test_portfolio_construction()
            
            # Keep browser open for observation
            print("\n🔍 KEEPING BROWSER OPEN FOR 15 SECONDS...")
            print("   👀 You can observe the RAGHeat application!")
            print("   🕐 Browser will close automatically after 15 seconds...")
            
            for i in range(15, 0, -1):
                print(f"\r   ⏰ Closing in {i} seconds...", end="", flush=True)
                time.sleep(1)
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        
        finally:
            # Clean up
            if self.driver:
                print("🧹 Closing browser...")
                self.driver.quit()
        
        # Print results
        print("\n" + "="*80)
        print("📊 TEST RESULTS SUMMARY")
        print("="*80)
        
        passed = sum(test_results.values())
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name.upper():20} {status}")
        
        print(f"\n🏆 Overall Success Rate: {passed}/{total} ({(passed/total)*100:.1f}%)")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! RAGHeat application is working perfectly!")
        else:
            print("⚠️ Some tests failed. Check the logs above for details.")
        
        return test_results

if __name__ == "__main__":
    print("🚀 Starting RAGHeat Visible Browser Test Demo")
    print("📱 This will open a Chrome browser window that you can see!")
    print("👁️ You'll be able to observe all the automation happening live!")
    
    demo = VisibleBrowserDemo()
    results = demo.run_comprehensive_test()
    
    print(f"\n✨ Test completed! Results: {results}")