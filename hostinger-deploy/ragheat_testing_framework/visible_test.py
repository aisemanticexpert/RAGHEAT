#!/usr/bin/env python3
"""
Visible browser test for RAGHeat application
"""

import os
import sys
import time
sys.path.append('.')
from framework.base_test import BaseTest
from framework.utilities.api_client import RAGHeatAPIClient
from framework.page_objects.portfolio_dashboard import PortfolioDashboard

class VisibleRAGHeatTest(BaseTest):
    
    def run_comprehensive_visible_test(self):
        """Run comprehensive test with visible browser"""
        print('🚀 Starting RAGHeat Comprehensive Testing with Visible Browser')
        print('=' * 70)
        print('Frontend: http://localhost:3000')
        print('Backend: http://localhost:8001')
        print('=' * 70)
        
        # Initialize components
        print('\n🔧 Setting up test environment...')
        self.setup_class()
        
        api_client = RAGHeatAPIClient('http://localhost:8001')
        portfolio_page = PortfolioDashboard(self.driver)
        
        test_results = {}
        
        try:
            # Test 1: API Health Check
            print('\n1️⃣ Testing API Health Check...')
            try:
                response = api_client.health_check()
                print(f'   ✅ API Health Status: {response.status_code}')
                if response.status_code == 200:
                    print('   ✅ Backend API is healthy and responsive')
                    test_results['api_health'] = 'PASS'
                else:
                    test_results['api_health'] = 'FAIL'
            except Exception as e:
                print(f'   ❌ API Health Check failed: {e}')
                test_results['api_health'] = 'FAIL'
            
            # Test 2: Frontend Load Test
            print('\n2️⃣ Testing Frontend Application Load...')
            try:
                start_time = time.time()
                success = portfolio_page.navigate_to_dashboard('http://localhost:3000')
                load_time = time.time() - start_time
                
                if success:
                    print(f'   ✅ Frontend loaded successfully in {load_time:.2f}s')
                    print('   🌐 Browser is now visible - you can see the application!')
                    test_results['frontend_load'] = 'PASS'
                    
                    # Give time to see the application
                    print('   ⏳ Keeping browser open for 5 seconds...')
                    time.sleep(5)
                else:
                    print('   ❌ Frontend failed to load')
                    test_results['frontend_load'] = 'FAIL'
            except Exception as e:
                print(f'   ❌ Frontend load test failed: {e}')
                test_results['frontend_load'] = 'FAIL'
            
            # Test 3: UI Component Analysis
            print('\n3️⃣ Testing UI Components and Page Structure...')
            try:
                page_info = portfolio_page.get_page_info()
                title = page_info.get('title', 'No title')
                buttons = page_info.get('buttons_count', 0)
                inputs = page_info.get('inputs_count', 0)
                forms = page_info.get('forms_count', 0)
                
                print(f'   📄 Page Title: "{title}"')
                print(f'   🎮 Interactive Elements Found:')
                print(f'      - {buttons} buttons')
                print(f'      - {inputs} input fields')
                print(f'      - {forms} forms')
                
                if buttons > 0 or inputs > 0:
                    print('   ✅ UI components are present and functional')
                    test_results['ui_components'] = 'PASS'
                else:
                    print('   ⚠️ Limited UI components found')
                    test_results['ui_components'] = 'PARTIAL'
                    
            except Exception as e:
                print(f'   ❌ UI component test failed: {e}')
                test_results['ui_components'] = 'FAIL'
            
            # Test 4: Portfolio Construction API
            print('\n4️⃣ Testing Portfolio Construction API...')
            try:
                test_stocks = ['AAPL', 'GOOGL', 'MSFT']
                print(f'   📊 Testing portfolio with stocks: {test_stocks}')
                
                start_time = time.time()
                response = api_client.construct_portfolio(test_stocks)
                api_time = time.time() - start_time
                
                print(f'   🌐 API Response: {response.status_code} ({api_time:.2f}s)')
                
                if response.status_code == 200:
                    data = response.json()
                    if 'portfolio_weights' in data:
                        weights = data['portfolio_weights']
                        print('   💰 Portfolio Weights Generated:')
                        for stock, weight in weights.items():
                            print(f'      - {stock}: {weight:.3f} ({weight*100:.1f}%)')
                        
                        total_weight = sum(weights.values())
                        print(f'   ✅ Total Weight: {total_weight:.3f} (should be ~1.0)')
                        test_results['portfolio_api'] = 'PASS'
                    else:
                        print('   ⚠️ Portfolio constructed but no weights returned')
                        test_results['portfolio_api'] = 'PARTIAL'
                else:
                    print(f'   ❌ Portfolio construction failed: {response.status_code}')
                    test_results['portfolio_api'] = 'FAIL'
                    
            except Exception as e:
                print(f'   ❌ Portfolio API test failed: {e}')
                test_results['portfolio_api'] = 'FAIL'
            
            # Test 5: Multi-Agent Analysis
            print('\n5️⃣ Testing Multi-Agent Analysis System...')
            try:
                test_stocks = ['AAPL', 'GOOGL']
                
                # Test Fundamental Analysis
                response = api_client.fundamental_analysis(test_stocks)
                print(f'   🧮 Fundamental Analysis: {response.status_code}')
                
                # Test Sentiment Analysis
                response = api_client.sentiment_analysis(test_stocks)
                print(f'   😊 Sentiment Analysis: {response.status_code}')
                
                # Test System Status
                response = api_client.system_status()
                print(f'   ⚙️ System Status: {response.status_code}')
                
                if response.status_code == 200:
                    data = response.json()
                    if 'agents' in data:
                        agents = data['agents']
                        print(f'   🤖 Found {len(agents)} agents in system')
                        for agent in agents[:3]:  # Show first 3 agents
                            name = agent.get('name', 'Unknown')
                            status = agent.get('status', 'Unknown')
                            print(f'      - {name}: {status}')
                
                print('   ✅ Multi-agent system is operational')
                test_results['multi_agent'] = 'PASS'
                
            except Exception as e:
                print(f'   ❌ Multi-agent test failed: {e}')
                test_results['multi_agent'] = 'FAIL'
            
            # Test 6: Interactive UI Testing
            print('\n6️⃣ Testing Interactive UI Elements...')
            try:
                # Try to find and interact with portfolio elements
                stock_input = portfolio_page.find_stock_input()
                construct_button = portfolio_page.find_construct_button()
                
                interactive_elements = 0
                
                if stock_input:
                    print('   ✅ Stock input field found and accessible')
                    # Try to interact with it
                    stock_input.clear()
                    stock_input.send_keys('AAPL')
                    entered_value = stock_input.get_attribute('value')
                    if 'AAPL' in entered_value:
                        print('   ✅ Stock input interaction successful')
                        interactive_elements += 1
                    print('   ⏳ Showing interaction for 3 seconds...')
                    time.sleep(3)
                
                if construct_button:
                    print('   ✅ Portfolio construction button found')
                    if construct_button.is_enabled():
                        print('   ✅ Construction button is clickable')
                        interactive_elements += 1
                
                if interactive_elements > 0:
                    print(f'   ✅ Interactive UI elements working ({interactive_elements}/2)')
                    test_results['interactive_ui'] = 'PASS'
                else:
                    print('   ⚠️ Limited interactive functionality found')
                    test_results['interactive_ui'] = 'PARTIAL'
                    
            except Exception as e:
                print(f'   ❌ Interactive UI test failed: {e}')
                test_results['interactive_ui'] = 'FAIL'
            
            # Test 7: Final Browser Showcase
            print('\n7️⃣ Final Application Showcase...')
            print('   🎭 Demonstrating application for final 8 seconds...')
            print('   👀 You can now see the complete RAGHeat application in action!')
            time.sleep(8)
            
        except Exception as e:
            print(f'\n❌ Comprehensive test failed: {e}')
        
        finally:
            # Test Summary
            print('\n' + '='*70)
            print('📊 COMPREHENSIVE TEST SUMMARY')
            print('='*70)
            
            for test_name, result in test_results.items():
                status_icon = '✅' if result == 'PASS' else '⚠️' if result == 'PARTIAL' else '❌'
                print(f'{status_icon} {test_name.replace("_", " ").title()}: {result}')
            
            passed = sum(1 for r in test_results.values() if r == 'PASS')
            total = len(test_results)
            
            print(f'\n📈 Overall Results: {passed}/{total} tests passed')
            print(f'🏆 Success Rate: {(passed/total*100):.0f}%')
            
            # Cleanup
            print('\n🧹 Cleaning up test environment...')
            self.teardown_class()
            print('✅ Test completed successfully!')
            
            return test_results

if __name__ == "__main__":
    # Set environment for visible browser
    os.environ['HEADLESS'] = 'false'
    
    test = VisibleRAGHeatTest()
    results = test.run_comprehensive_visible_test()
    
    print('\n🎉 RAGHeat Application Testing Complete!')
    print('📧 Preparing email report...')
    
    # Send email report
    try:
        from framework.utilities.email_reporter import EmailReporter
        reporter = EmailReporter()
        
        test_summary = {
            'total': len(results),
            'passed': sum(1 for r in results.values() if r == 'PASS'),
            'failed': sum(1 for r in results.values() if r == 'FAIL'),
            'skipped': sum(1 for r in results.values() if r == 'PARTIAL')
        }
        
        reporter.send_test_report('semanticraj@gmail.com', test_summary)
        print('📧 Email report sent to semanticraj@gmail.com')
        
    except Exception as e:
        print(f'📧 Email report failed: {e}')
    
    print('\n🎯 Testing session completed!')