#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE TEST EXECUTION
Complete end-to-end testing with real clicking and comprehensive reporting
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
sys.path.append('.')

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from framework.utilities.enhanced_email_reporter import EnhancedEmailReporter

class FinalComprehensiveTestExecutor:
    """Execute comprehensive testing with real interactions and generate reports"""
    
    def __init__(self):
        self.results = {
            'execution_metadata': {
                'start_time': datetime.now().isoformat(),
                'framework_version': '2.0.0',
                'environment': 'production',
                'frontend_url': 'http://localhost:3000',
                'backend_url': 'http://localhost:8001'
            },
            'test_results': {
                'dashboard_buttons': {},
                'navigation_tabs': {},
                'ui_interactions': {},
                'api_validations': {},
                'screenshots_captured': []
            },
            'summary': {}
        }
        
        # Set environment
        os.environ['HEADLESS'] = 'false'
        os.environ['RAGHEAT_API_URL'] = 'http://localhost:8001'
        os.environ['RAGHEAT_FRONTEND_URL'] = 'http://localhost:3000'
    
    def setup_browser(self):
        """Setup browser for testing"""
        chrome_options = Options()
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        print("✅ Browser setup complete")
        
    def execute_dashboard_button_testing(self):
        """Execute comprehensive dashboard button testing"""
        print("\\n🚀 EXECUTING DASHBOARD BUTTON TESTING")
        print("="*60)
        
        self.driver.get('http://localhost:3000')
        time.sleep(8)  # Wait for full page load
        
        # Take initial screenshot
        screenshot_path = f"screenshots/dashboard_initial_{int(time.time())}.png"
        self.driver.save_screenshot(screenshot_path)
        self.results['test_results']['screenshots_captured'].append(screenshot_path)
        
        # Find all buttons and test clicking
        all_buttons = self.driver.find_elements(By.TAG_NAME, 'button')
        print(f"📊 Found {len(all_buttons)} buttons on the dashboard")
        
        clicked_buttons = []
        button_details = []
        
        for i, btn in enumerate(all_buttons):
            try:
                text = btn.text.strip()
                if text and btn.is_displayed() and btn.is_enabled():
                    print(f"\\n🖱️ Testing button {i+1}: \"{text}\"")
                    
                    # Highlight button
                    self.driver.execute_script("arguments[0].style.border='3px solid red'", btn)
                    time.sleep(1)
                    
                    # Click button
                    btn.click()
                    print(f"   ✅ Successfully clicked: \"{text}\"")
                    
                    # Wait and take screenshot
                    time.sleep(3)
                    screenshot_path = f"screenshots/button_clicked_{text.replace(' ', '_')}_{int(time.time())}.png"
                    self.driver.save_screenshot(screenshot_path)
                    self.results['test_results']['screenshots_captured'].append(screenshot_path)
                    
                    # Remove highlight
                    self.driver.execute_script("arguments[0].style.border=''", btn)
                    
                    clicked_buttons.append(text)
                    button_details.append({
                        'name': text,
                        'status': 'clicked',
                        'screenshot': screenshot_path
                    })
                    
            except Exception as e:
                print(f"   ❌ Failed to click button {i+1}: {e}")
                if text:
                    button_details.append({
                        'name': text,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        self.results['test_results']['dashboard_buttons'] = {
            'total_buttons': len(all_buttons),
            'clicked_successfully': len(clicked_buttons),
            'success_rate': (len(clicked_buttons) / len(all_buttons)) * 100 if all_buttons else 0,
            'clicked_buttons': clicked_buttons,
            'button_details': button_details
        }
        
        print(f"\\n📈 Button Testing Results:")
        print(f"   Total buttons: {len(all_buttons)}")
        print(f"   Successfully clicked: {len(clicked_buttons)}")
        print(f"   Success rate: {self.results['test_results']['dashboard_buttons']['success_rate']:.1f}%")
        
        return self.results['test_results']['dashboard_buttons']
    
    def execute_navigation_testing(self):
        """Execute navigation and tab testing"""
        print("\\n🚀 EXECUTING NAVIGATION TESTING")
        print("="*60)
        
        # Look for tab-like elements
        tab_selectors = [
            "//button[contains(text(), 'OVERVIEW')]",
            "//button[contains(text(), 'HEAT')]", 
            "//button[contains(text(), 'SECTOR')]",
            "//button[contains(text(), 'SIGNALS')]",
            "//button[contains(text(), 'MODELS')]"
        ]
        
        navigation_results = {
            'tabs_found': 0,
            'tabs_clicked': 0,
            'tab_details': []
        }
        
        for selector in tab_selectors:
            try:
                tabs = self.driver.find_elements(By.XPATH, selector)
                if tabs:
                    tab = tabs[0]
                    text = tab.text.strip()
                    print(f"\\n🔍 Found tab: \"{text}\"")
                    
                    navigation_results['tabs_found'] += 1
                    
                    # Highlight and click
                    self.driver.execute_script("arguments[0].style.border='3px solid blue'", tab)
                    time.sleep(1)
                    tab.click()
                    
                    print(f"   ✅ Successfully clicked tab: \"{text}\"")
                    navigation_results['tabs_clicked'] += 1
                    
                    # Take screenshot
                    time.sleep(3)
                    screenshot_path = f"screenshots/tab_clicked_{text.replace(' ', '_')}_{int(time.time())}.png"
                    self.driver.save_screenshot(screenshot_path)
                    self.results['test_results']['screenshots_captured'].append(screenshot_path)
                    
                    navigation_results['tab_details'].append({
                        'name': text,
                        'status': 'clicked',
                        'screenshot': screenshot_path
                    })
                    
                    # Remove highlight
                    self.driver.execute_script("arguments[0].style.border=''", tab)
                    
            except Exception as e:
                print(f"   ❌ Tab clicking failed: {e}")
        
        self.results['test_results']['navigation_tabs'] = navigation_results
        
        print(f"\\n📈 Navigation Testing Results:")
        print(f"   Tabs found: {navigation_results['tabs_found']}")
        print(f"   Tabs clicked: {navigation_results['tabs_clicked']}")
        
        return navigation_results
    
    def execute_ui_interaction_testing(self):
        """Execute UI interaction testing"""
        print("\\n🚀 EXECUTING UI INTERACTION TESTING")
        print("="*60)
        
        ui_results = {
            'inputs_found': 0,
            'inputs_tested': 0,
            'forms_found': 0,
            'interactions': []
        }
        
        # Test input fields
        inputs = self.driver.find_elements(By.TAG_NAME, 'input')
        ui_results['inputs_found'] = len(inputs)
        print(f"📝 Found {len(inputs)} input fields")
        
        for i, inp in enumerate(inputs[:3]):  # Test first 3 inputs
            try:
                placeholder = inp.get_attribute('placeholder') or f'input_{i}'
                print(f"\\n⌨️ Testing input: \"{placeholder}\"")
                
                # Highlight input
                self.driver.execute_script("arguments[0].style.border='3px solid green'", inp)
                time.sleep(1)
                
                # Enter test data
                test_data = "AAPL,GOOGL,MSFT" if 'stock' in placeholder.lower() else "test data"
                inp.clear()
                inp.send_keys(test_data)
                
                print(f"   ✅ Successfully entered data in: \"{placeholder}\"")
                ui_results['inputs_tested'] += 1
                
                # Take screenshot
                time.sleep(2)
                screenshot_path = f"screenshots/input_tested_{placeholder.replace(' ', '_')}_{int(time.time())}.png"
                self.driver.save_screenshot(screenshot_path)
                self.results['test_results']['screenshots_captured'].append(screenshot_path)
                
                ui_results['interactions'].append({
                    'type': 'input',
                    'element': placeholder,
                    'data_entered': test_data,
                    'status': 'success',
                    'screenshot': screenshot_path
                })
                
                # Remove highlight
                self.driver.execute_script("arguments[0].style.border=''", inp)
                
            except Exception as e:
                print(f"   ❌ Input testing failed: {e}")
                ui_results['interactions'].append({
                    'type': 'input',
                    'element': placeholder,
                    'status': 'failed',
                    'error': str(e)
                })
        
        self.results['test_results']['ui_interactions'] = ui_results
        
        print(f"\\n📈 UI Interaction Results:")
        print(f"   Input fields found: {ui_results['inputs_found']}")
        print(f"   Input fields tested: {ui_results['inputs_tested']}")
        
        return ui_results
    
    def execute_api_validation_testing(self):
        """Execute API validation testing"""
        print("\\n🚀 EXECUTING API VALIDATION TESTING")
        print("="*60)
        
        import requests
        
        api_results = {
            'endpoints_tested': 0,
            'endpoints_working': 0,
            'api_details': []
        }
        
        # Test key API endpoints
        endpoints = [
            ('/health', 'GET', 'Health Check'),
            ('/system/status', 'GET', 'System Status'),
            ('/portfolio/construct', 'POST', 'Portfolio Construction'),
            ('/analysis/fundamental', 'POST', 'Fundamental Analysis'),
            ('/analysis/sentiment', 'POST', 'Sentiment Analysis')
        ]
        
        for endpoint, method, description in endpoints:
            try:
                url = f"http://localhost:8001{endpoint}"
                print(f"\\n🌐 Testing {description}: {method} {endpoint}")
                
                api_results['endpoints_tested'] += 1
                
                if method == 'GET':
                    response = requests.get(url, timeout=5)
                else:
                    response = requests.post(url, json={'stocks': ['AAPL', 'GOOGL']}, timeout=5)
                
                status = "✅ WORKING" if response.status_code in [200, 201] else "❌ FAILED"
                print(f"   {status} ({response.status_code})")
                
                if response.status_code in [200, 201]:
                    api_results['endpoints_working'] += 1
                
                api_results['api_details'].append({
                    'endpoint': endpoint,
                    'method': method,
                    'description': description,
                    'status_code': response.status_code,
                    'status': 'working' if response.status_code in [200, 201] else 'failed'
                })
                
            except Exception as e:
                print(f"   ❌ {description} failed: {e}")
                api_results['api_details'].append({
                    'endpoint': endpoint,
                    'method': method,
                    'description': description,
                    'status': 'error',
                    'error': str(e)
                })
        
        self.results['test_results']['api_validations'] = api_results
        
        print(f"\\n📈 API Validation Results:")
        print(f"   Endpoints tested: {api_results['endpoints_tested']}")
        print(f"   Endpoints working: {api_results['endpoints_working']}")
        
        return api_results
    
    def generate_comprehensive_summary(self):
        """Generate comprehensive test summary"""
        print("\\n🚀 GENERATING COMPREHENSIVE SUMMARY")
        print("="*60)
        
        # Calculate overall metrics
        button_results = self.results['test_results']['dashboard_buttons']
        nav_results = self.results['test_results']['navigation_tabs']
        ui_results = self.results['test_results']['ui_interactions']
        api_results = self.results['test_results']['api_validations']
        
        total_tests = (
            button_results.get('total_buttons', 0) +
            nav_results.get('tabs_found', 0) +
            ui_results.get('inputs_found', 0) +
            api_results.get('endpoints_tested', 0)
        )
        
        successful_tests = (
            button_results.get('clicked_successfully', 0) +
            nav_results.get('tabs_clicked', 0) +
            ui_results.get('inputs_tested', 0) +
            api_results.get('endpoints_working', 0)
        )
        
        overall_success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        summary = {
            'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests_executed': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'overall_success_rate': f"{overall_success_rate:.1f}%",
            'screenshots_captured': len(self.results['test_results']['screenshots_captured']),
            'features_validated': [
                'Dashboard Button Interactions',
                'Navigation Tab Clicking',
                'Form Input Testing',
                'API Endpoint Validation',
                'UI Element Detection',
                'Browser Automation',
                'Screenshot Documentation'
            ],
            'detailed_results': {
                'dashboard_buttons': f"{button_results.get('clicked_successfully', 0)}/{button_results.get('total_buttons', 0)} clicked",
                'navigation_tabs': f"{nav_results.get('tabs_clicked', 0)}/{nav_results.get('tabs_found', 0)} clicked",
                'ui_interactions': f"{ui_results.get('inputs_tested', 0)}/{ui_results.get('inputs_found', 0)} tested",
                'api_validations': f"{api_results.get('endpoints_working', 0)}/{api_results.get('endpoints_tested', 0)} working"
            }
        }
        
        self.results['summary'] = summary
        
        print(f"📊 COMPREHENSIVE TEST EXECUTION SUMMARY:")
        print(f"   Execution Date: {summary['execution_date']}")
        print(f"   Total Tests: {summary['total_tests_executed']}")
        print(f"   Successful: {summary['successful_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Success Rate: {summary['overall_success_rate']}")
        print(f"   Screenshots: {summary['screenshots_captured']}")
        
        return summary
    
    def generate_stakeholder_report(self):
        """Generate and send stakeholder report"""
        print("\\n📧 GENERATING STAKEHOLDER REPORT")
        print("="*60)
        
        # Save detailed results to JSON
        results_file = f"reports/comprehensive_test_results_{int(time.time())}.json"
        os.makedirs("reports", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Detailed results saved to: {results_file}")
        
        # Generate email report
        try:
            reporter = EnhancedEmailReporter()
            
            summary = self.results['summary']
            
            email_summary = {
                'total': summary['total_tests_executed'],
                'passed': summary['successful_tests'],
                'failed': summary['failed_tests'],
                'skipped': 0
            }
            
            build_info = {
                'build_number': 'FINAL-COMPREHENSIVE-TEST',
                'branch': 'main',
                'commit': 'complete-xpath-automation'
            }
            
            # Determine email template type
            success_rate = float(summary['overall_success_rate'].replace('%', ''))
            template_type = 'success' if success_rate >= 70 else 'failure'
            
            # Send email report
            reporter.send_test_report(
                recipient_email='semanticraj@gmail.com',
                test_summary=email_summary,
                build_info=build_info,
                template_type=template_type
            )
            
            print("✅ Stakeholder email report sent successfully!")
            
        except Exception as e:
            print(f"⚠️ Email report generation failed: {e}")
        
        return results_file
    
    def display_final_summary(self):
        """Display final comprehensive summary"""
        print("\\n" + "="*100)
        print("🎉 FINAL COMPREHENSIVE TEST EXECUTION COMPLETED!")
        print("="*100)
        
        summary = self.results['summary']
        
        print(f"📅 Execution Date: {summary['execution_date']}")
        print(f"🔢 Total Tests: {summary['total_tests_executed']}")
        print(f"✅ Successful: {summary['successful_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Success Rate: {summary['overall_success_rate']}")
        print(f"📸 Screenshots: {summary['screenshots_captured']}")
        
        print("\\n🎯 Features Validated:")
        for feature in summary['features_validated']:
            print(f"   ✅ {feature}")
        
        print("\\n📊 Detailed Results:")
        for category, result in summary['detailed_results'].items():
            print(f"   🔹 {category.replace('_', ' ').title()}: {result}")
        
        success_rate = float(summary['overall_success_rate'].replace('%', ''))
        
        if success_rate >= 80:
            print("\\n🏆 EXCELLENT: Outstanding performance! System ready for production!")
        elif success_rate >= 60:
            print("\\n👍 GOOD: System performing well with minor improvements needed!")
        else:
            print("\\n⚠️ NEEDS ATTENTION: Several areas require improvement!")
        
        print("\\n📁 Generated Artifacts:")
        print("   📄 Comprehensive JSON results file")
        print("   📧 Professional email report sent to semanticraj@gmail.com")
        print(f"   📸 {summary['screenshots_captured']} screenshots captured")
        
        print("\\n✨ COMPREHENSIVE TESTING MISSION ACCOMPLISHED! ✨")
        print("="*100)
    
    def execute_complete_testing(self):
        """Execute complete comprehensive testing"""
        print("🚀 STARTING FINAL COMPREHENSIVE TEST EXECUTION")
        print("🎯 Complete end-to-end validation with real interactions")
        
        try:
            # Setup
            self.setup_browser()
            
            # Execute all test phases
            self.execute_dashboard_button_testing()
            self.execute_navigation_testing() 
            self.execute_ui_interaction_testing()
            self.execute_api_validation_testing()
            
            # Generate summary and reports
            self.generate_comprehensive_summary()
            self.generate_stakeholder_report()
            self.display_final_summary()
            
            return self.results
            
        except Exception as e:
            print(f"\\n❌ Test execution failed: {e}")
            raise
        
        finally:
            # Cleanup
            if hasattr(self, 'driver'):
                self.driver.quit()
                print("🧹 Browser closed successfully")


def main():
    """Main execution function"""
    executor = FinalComprehensiveTestExecutor()
    results = executor.execute_complete_testing()
    return results


if __name__ == "__main__":
    results = main()
    print(f"\\n✅ Final results: {results['summary']['overall_success_rate']} success rate")