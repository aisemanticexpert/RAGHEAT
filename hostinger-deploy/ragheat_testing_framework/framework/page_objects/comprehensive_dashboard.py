#!/usr/bin/env python3
"""
Comprehensive RAGHeat Dashboard Page Object
Real XPath-based interactions with all dashboard elements
"""

import time
import json
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class ComprehensiveDashboard:
    """Complete page object for RAGHeat dashboard with real XPath interactions"""
    
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(driver, 15)
        self.actions = ActionChains(driver)
        
        # Load XPath mappings from configuration
        config_path = Path("framework/config/ui_elements.json")
        with open(config_path, 'r') as f:
            self.ui_elements = json.load(f)
    
    def navigate_to_dashboard(self, url):
        """Navigate to RAGHeat dashboard and wait for load"""
        print(f"🌐 Navigating to: {url}")
        self.driver.get(url)
        time.sleep(3)  # Initial load time
        
        # Wait for main container to load
        try:
            self.wait.until(EC.presence_of_element_located((By.XPATH, self.ui_elements["homepage"]["main_container"])))
            print("✅ Dashboard loaded successfully")
            return True
        except TimeoutException:
            print("❌ Dashboard failed to load")
            return False
    
    def click_dashboard_button(self, button_name):
        """Click specific dashboard button by name"""
        try:
            xpath = self.ui_elements["dashboard_buttons"].get(button_name)
            if not xpath:
                print(f"❌ Button '{button_name}' not found in configuration")
                return False
            
            print(f"🖱️ Looking for button: {button_name}")
            element = self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
            
            # Scroll to element if needed
            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            time.sleep(1)
            
            # Highlight element before clicking
            self.driver.execute_script("arguments[0].style.border='3px solid red'", element)
            time.sleep(2)  # Let user see the highlight
            
            element.click()
            print(f"✅ Clicked button: {button_name}")
            time.sleep(3)  # Wait for page reaction
            
            # Remove highlight
            self.driver.execute_script("arguments[0].style.border=''", element)
            return True
            
        except TimeoutException:
            print(f"❌ Button '{button_name}' not found or not clickable")
            return False
        except Exception as e:
            print(f"❌ Error clicking button '{button_name}': {e}")
            return False
    
    def test_all_dashboard_buttons(self):
        """Test clicking all dashboard buttons"""
        print("🚀 Testing ALL dashboard buttons with real clicking...")
        
        button_results = {}
        buttons_to_test = [
            "revolutionary_dashboard",
            "market_dashboard", 
            "live_options_signals",
            "live_data_stream",
            "knowledge_graph",
            "enhanced_graph",
            "ontology_graph",
            "advanced_kg",
            "portfolio_ai"
        ]
        
        for button_name in buttons_to_test:
            print(f"\\n🔥 Testing button: {button_name}")
            success = self.click_dashboard_button(button_name)
            button_results[button_name] = success
            
            if success:
                # Wait and observe the result
                time.sleep(3)
                # Take screenshot after each click
                screenshot_path = f"screenshots/button_click_{button_name}_{int(time.time())}.png"
                self.driver.save_screenshot(screenshot_path)
                print(f"📸 Screenshot saved: {screenshot_path}")
        
        return button_results
    
    def test_all_tab_navigation(self):
        """Test clicking all navigation tabs"""
        print("🚀 Testing ALL navigation tabs with real clicking...")
        
        tab_results = {}
        tabs_to_test = [
            "overview_tab",
            "heat_propagation_tab",
            "sector_analysis_tab", 
            "live_signals_tab",
            "ai_models_tab"
        ]
        
        for tab_name in tabs_to_test:
            print(f"\\n🔥 Testing tab: {tab_name}")
            success = self.click_tab(tab_name)
            tab_results[tab_name] = success
            
            if success:
                # Wait and observe the result
                time.sleep(3)
                # Take screenshot after each click
                screenshot_path = f"screenshots/tab_click_{tab_name}_{int(time.time())}.png"
                self.driver.save_screenshot(screenshot_path)
                print(f"📸 Screenshot saved: {screenshot_path}")
        
        return tab_results
    
    def click_tab(self, tab_name):
        """Click specific navigation tab by name"""
        try:
            xpath = self.ui_elements["tab_navigation"].get(tab_name)
            if not xpath:
                print(f"❌ Tab '{tab_name}' not found in configuration")
                return False
            
            print(f"🖱️ Looking for tab: {tab_name}")
            element = self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
            
            # Scroll to element if needed
            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            time.sleep(1)
            
            # Highlight element before clicking
            self.driver.execute_script("arguments[0].style.border='3px solid purple'", element)
            time.sleep(2)  # Let user see the highlight
            
            element.click()
            print(f"✅ Clicked tab: {tab_name}")
            time.sleep(3)  # Wait for tab content to load
            
            # Remove highlight
            self.driver.execute_script("arguments[0].style.border=''", element)
            return True
            
        except TimeoutException:
            print(f"❌ Tab '{tab_name}' not found or not clickable")
            return False
        except Exception as e:
            print(f"❌ Error clicking tab '{tab_name}': {e}")
            return False
    
    def test_portfolio_construction_workflow(self):
        """Test complete portfolio construction workflow with real interactions"""
        print("🚀 Testing Portfolio Construction Workflow...")
        
        workflow_results = {
            'stock_input_found': False,
            'stocks_entered': False,
            'portfolio_constructed': False,
            'results_displayed': False
        }
        
        # Step 1: Find and interact with stock input field
        stock_input_selectors = [
            self.ui_elements["portfolio_construction"]["stock_input"],
            self.ui_elements["portfolio_construction"]["stock_search"],
            "//input[contains(@placeholder, 'stock')]",
            "//input[@type='text']"
        ]
        
        stock_input_element = None
        for selector in stock_input_selectors:
            try:
                print(f"🔍 Looking for stock input with: {selector}")
                stock_input_element = self.driver.find_element(By.XPATH, selector)
                if stock_input_element.is_displayed():
                    workflow_results['stock_input_found'] = True
                    print("✅ Stock input field found!")
                    break
            except NoSuchElementException:
                continue
        
        if stock_input_element:
            # Highlight and interact with input
            self.driver.execute_script("arguments[0].style.border='3px solid green'", stock_input_element)
            time.sleep(2)
            
            # Enter test stocks
            test_stocks = "AAPL,GOOGL,MSFT,TSLA"
            stock_input_element.clear()
            stock_input_element.send_keys(test_stocks)
            workflow_results['stocks_entered'] = True
            print(f"⌨️ Entered stocks: {test_stocks}")
            
            time.sleep(3)
            screenshot_path = f"screenshots/stocks_entered_{int(time.time())}.png"
            self.driver.save_screenshot(screenshot_path)
            
            # Step 2: Click construct portfolio button
            construct_selectors = [
                self.ui_elements["portfolio_construction"]["construct_portfolio_button"],
                "//button[contains(text(), 'Construct')]",
                "//button[contains(text(), 'Build')]",
                "//button[contains(text(), 'Create')]"
            ]
            
            for selector in construct_selectors:
                try:
                    construct_btn = self.driver.find_element(By.XPATH, selector)
                    if construct_btn.is_displayed():
                        self.driver.execute_script("arguments[0].style.border='3px solid blue'", construct_btn)
                        time.sleep(2)
                        construct_btn.click()
                        workflow_results['portfolio_constructed'] = True
                        print("✅ Portfolio construction initiated!")
                        time.sleep(5)  # Wait for processing
                        break
                except NoSuchElementException:
                    continue
        
        # Step 3: Check for results
        result_selectors = [
            self.ui_elements["portfolio_results"]["results_container"],
            self.ui_elements["portfolio_results"]["weights_table"],
            "//div[contains(@class, 'result')]",
            "//table",
            "//*[contains(text(), 'weight') or contains(text(), 'Weight')]"
        ]
        
        for selector in result_selectors:
            try:
                result_element = self.driver.find_element(By.XPATH, selector)
                if result_element.is_displayed():
                    workflow_results['results_displayed'] = True
                    print("✅ Portfolio results displayed!")
                    screenshot_path = f"screenshots/portfolio_results_{int(time.time())}.png"
                    self.driver.save_screenshot(screenshot_path)
                    break
            except NoSuchElementException:
                continue
        
        return workflow_results
    
    def test_knowledge_graph_interactions(self):
        """Test knowledge graph interactions"""
        print("🚀 Testing Knowledge Graph Interactions...")
        
        # Click knowledge graph button first
        if self.click_dashboard_button("knowledge_graph"):
            time.sleep(5)  # Wait for graph to load
            
            # Look for graph elements
            graph_selectors = [
                self.ui_elements["knowledge_graph"]["graph_container"],
                self.ui_elements["data_visualization"]["charts_container"],
                "//*[name()='svg']",
                "//canvas"
            ]
            
            graph_found = False
            for selector in graph_selectors:
                try:
                    graph_element = self.driver.find_element(By.XPATH, selector)
                    if graph_element.is_displayed():
                        print("✅ Knowledge graph visualization found!")
                        screenshot_path = f"screenshots/knowledge_graph_{int(time.time())}.png"
                        self.driver.save_screenshot(screenshot_path)
                        graph_found = True
                        break
                except NoSuchElementException:
                    continue
            
            return graph_found
        
        return False
    
    def test_real_time_data_features(self):
        """Test real-time data streaming features"""
        print("🚀 Testing Real-Time Data Features...")
        
        # Click live data stream button
        if self.click_dashboard_button("live_data_stream"):
            time.sleep(5)
            
            # Look for live data indicators
            live_indicators = [
                self.ui_elements["real_time_data"]["live_feed"],
                self.ui_elements["real_time_data"]["websocket_status"],
                "//*[contains(text(), 'Live') or contains(text(), 'Connected')]",
                "//*[contains(@class, 'stream') or contains(@class, 'live')]"
            ]
            
            live_data_found = False
            for selector in live_indicators:
                try:
                    element = self.driver.find_element(By.XPATH, selector)
                    if element.is_displayed():
                        print("✅ Live data streaming interface found!")
                        screenshot_path = f"screenshots/live_data_{int(time.time())}.png"
                        self.driver.save_screenshot(screenshot_path)
                        live_data_found = True
                        break
                except NoSuchElementException:
                    continue
            
            return live_data_found
        
        return False
    
    def test_options_trading_features(self):
        """Test options trading functionality"""
        print("🚀 Testing Options Trading Features...")
        
        if self.click_dashboard_button("live_options_signals"):
            time.sleep(5)
            
            # Look for options elements
            options_indicators = [
                self.ui_elements["options_trading"]["options_panel"],
                self.ui_elements["options_trading"]["buy_signals"],
                self.ui_elements["options_trading"]["sell_signals"],
                "//*[contains(text(), 'BUY') or contains(text(), 'SELL')]",
                "//*[contains(text(), 'Option') or contains(text(), 'Signal')]"
            ]
            
            options_found = False
            signal_count = 0
            
            for selector in options_indicators:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    if elements:
                        signal_count += len([e for e in elements if e.is_displayed()])
                        options_found = True
                except NoSuchElementException:
                    continue
            
            if options_found:
                print(f"✅ Options trading interface found with {signal_count} signals!")
                screenshot_path = f"screenshots/options_trading_{int(time.time())}.png"
                self.driver.save_screenshot(screenshot_path)
            
            return options_found, signal_count
        
        return False, 0
    
    def test_multi_agent_system(self):
        """Test multi-agent system functionality"""
        print("🚀 Testing Multi-Agent System...")
        
        agent_results = {
            'fundamental_agent': False,
            'sentiment_agent': False,
            'technical_agent': False,
            'consensus_found': False
        }
        
        # Look for agent indicators
        agent_selectors = {
            'fundamental_agent': [
                self.ui_elements["multi_agent_system"]["fundamental_agent"],
                "//*[contains(text(), 'Fundamental')]"
            ],
            'sentiment_agent': [
                self.ui_elements["multi_agent_system"]["sentiment_agent"],
                "//*[contains(text(), 'Sentiment')]"
            ],
            'technical_agent': [
                self.ui_elements["multi_agent_system"]["technical_agent"],
                "//*[contains(text(), 'Technical')]"
            ]
        }
        
        for agent_name, selectors in agent_selectors.items():
            for selector in selectors:
                try:
                    element = self.driver.find_element(By.XPATH, selector)
                    if element.is_displayed():
                        agent_results[agent_name] = True
                        print(f"✅ {agent_name.replace('_', ' ').title()} found!")
                        break
                except NoSuchElementException:
                    continue
        
        # Check for consensus panel
        try:
            consensus_element = self.driver.find_element(By.XPATH, self.ui_elements["multi_agent_system"]["consensus_panel"])
            if consensus_element.is_displayed():
                agent_results['consensus_found'] = True
                print("✅ Agent consensus panel found!")
        except NoSuchElementException:
            pass
        
        if any(agent_results.values()):
            screenshot_path = f"screenshots/multi_agent_system_{int(time.time())}.png"
            self.driver.save_screenshot(screenshot_path)
        
        return agent_results
    
    def run_comprehensive_test_suite(self):
        """Run complete comprehensive test suite with real interactions"""
        print("\\n" + "="*80)
        print("🚀 COMPREHENSIVE RAGHEAT DASHBOARD TEST SUITE")
        print("👁️ Real XPath-based clicking and interactions")
        print("="*80)
        
        test_results = {
            'navigation_success': False,
            'button_tests': {},
            'portfolio_workflow': {},
            'knowledge_graph': False,
            'real_time_data': False,
            'options_trading': (False, 0),
            'multi_agent_system': {},
            'total_screenshots': 0
        }
        
        # Navigation test
        test_results['navigation_success'] = self.navigate_to_dashboard("http://localhost:3000")
        
        if test_results['navigation_success']:
            # Test all dashboard buttons
            test_results['button_tests'] = self.test_all_dashboard_buttons()
            
            # Test portfolio construction workflow
            test_results['portfolio_workflow'] = self.test_portfolio_construction_workflow()
            
            # Test knowledge graph
            test_results['knowledge_graph'] = self.test_knowledge_graph_interactions()
            
            # Test real-time data
            test_results['real_time_data'] = self.test_real_time_data_features()
            
            # Test options trading
            test_results['options_trading'] = self.test_options_trading_features()
            
            # Test multi-agent system
            test_results['multi_agent_system'] = self.test_multi_agent_system()
        
        # Count screenshots
        import os
        screenshots = [f for f in os.listdir('screenshots') if f.endswith('.png')]
        test_results['total_screenshots'] = len(screenshots)
        
        return test_results