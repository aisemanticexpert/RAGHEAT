#!/usr/bin/env python3
"""
INDIVIDUAL FEATURE TEST: Sentiment Analysis
Test file for sentiment_analysis feature - runs independently on Chrome
"""

import sys
import os
import time
import requests
import json
from pathlib import Path

# Add framework to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from framework.base_test import BaseTest
from framework.page_objects.comprehensive_dashboard import ComprehensiveDashboard
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class TestSentimentanalysis(BaseTest):
    """Individual test for Sentiment Analysis feature"""
    
    def setup_method(self, method):
        """Setup for each test method"""
        super().setup_method(method)
        self.base_url = os.getenv("RAGHEAT_FRONTEND_URL", "http://localhost:3000")
        self.api_base = os.getenv("RAGHEAT_API_URL", "http://localhost:8001")
        self.dashboard = ComprehensiveDashboard(self.driver)
        self.test_results = {
            "feature_name": "Sentiment Analysis",
            "feature_key": "sentiment_analysis",
            "test_timestamp": time.time(),
            "api_test_passed": False,
            "ui_test_passed": False,
            "data_validation_passed": False,
            "issues_found": [],
            "performance_metrics": {},
            "screenshots": []
        }
    
    def test_sentiment_analysis_api_endpoint(self):
        """Test Sentiment Analysis API endpoint"""
        print(f"🧪 Testing Sentiment Analysis API endpoint...")
        
        if not "/analysis/sentiment":
            print("   ⚠️ No API endpoint defined for this feature")
            self.test_results["api_test_passed"] = True  # Skip if no API
            return
        
        try:
            start_time = time.time()
            
            # Prepare API request
            endpoint = f"{self.api_base}/analysis/sentiment"
            test_data = {'stocks': ['AAPL', 'GOOGL']}
            
            if test_data and test_data != {}:
                response = requests.post(endpoint, json=test_data, timeout=30)
            else:
                response = requests.get(endpoint, timeout=30)
            
            response_time = time.time() - start_time
            self.test_results["performance_metrics"]["api_response_time"] = response_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate expected fields
                expected_fields = ['overall_sentiment', 'news_sentiment', 'social_sentiment', 'recommendation']
                if expected_fields:
                    self._validate_response_fields(data, expected_fields)
                
                self.test_results["api_test_passed"] = True
                print(f"   ✅ API test passed ({response_time:.2f}s)")
                
            else:
                self.test_results["issues_found"].append(f"API returned status {response.status_code}")
                print(f"   ❌ API test failed: {response.status_code}")
                
        except Exception as e:
            self.test_results["issues_found"].append(f"API test error: {str(e)}")
            print(f"   ❌ API test failed: {e}")
    
    def test_sentiment_analysis_ui_interaction(self):
        """Test Sentiment Analysis UI interaction"""
        print(f"🖱️ Testing Sentiment Analysis UI interaction...")
        
        try:
            # Navigate to dashboard
            if not self.dashboard.navigate_to_dashboard(self.base_url):
                self.test_results["issues_found"].append("Dashboard failed to load")
                return
            
            # Take initial screenshot
            initial_screenshot = f"screenshots/sentiment_analysis_initial_{int(time.time())}.png"
            self.driver.save_screenshot(initial_screenshot)
            self.test_results["screenshots"].append(initial_screenshot)
            
            ui_element = "//button[contains(text(), 'Sentiment')]"
            if ui_element:
                start_time = time.time()
                
                # Find and interact with UI element
                try:
                    element = WebDriverWait(self.driver, 15).until(
                        EC.element_to_be_clickable((By.XPATH, ui_element))
                    )
                    
                    # Highlight element
                    self.driver.execute_script("arguments[0].style.border='3px solid green'", element)
                    time.sleep(1)
                    
                    # Click element
                    element.click()
                    time.sleep(3)  # Wait for response
                    
                    interaction_time = time.time() - start_time
                    self.test_results["performance_metrics"]["ui_interaction_time"] = interaction_time
                    
                    # Take post-interaction screenshot
                    post_screenshot = f"screenshots/sentiment_analysis_clicked_{int(time.time())}.png"
                    self.driver.save_screenshot(post_screenshot)
                    self.test_results["screenshots"].append(post_screenshot)
                    
                    self.test_results["ui_test_passed"] = True
                    print(f"   ✅ UI interaction passed ({interaction_time:.2f}s)")
                    
                except TimeoutException:
                    self.test_results["issues_found"].append(f"UI element not found: {ui_element}")
                    print(f"   ❌ UI element not found: {ui_element}")
                    
            else:
                self.test_results["ui_test_passed"] = True  # Skip if no UI element
                print("   ⚠️ No UI element defined for this feature")
                
        except Exception as e:
            self.test_results["issues_found"].append(f"UI test error: {str(e)}")
            print(f"   ❌ UI test failed: {e}")
    
    def test_sentiment_analysis_data_validation(self):
        """Test Sentiment Analysis data validation"""
        print(f"📊 Testing Sentiment Analysis data validation...")
        
        try:
            # Test data consistency, ranges, and completeness
            if "/analysis/sentiment":
                endpoint = f"{self.api_base}/analysis/sentiment"
                test_data = {'stocks': ['AAPL', 'GOOGL']}
                
                if test_data and test_data != {}:
                    response = requests.post(endpoint, json=test_data, timeout=30)
                else:
                    response = requests.get(endpoint, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    validation_passed = self._validate_data_quality(data)
                    self.test_results["data_validation_passed"] = validation_passed
                    
                    if validation_passed:
                        print("   ✅ Data validation passed")
                    else:
                        print("   ❌ Data validation failed")
                else:
                    self.test_results["issues_found"].append("Cannot validate data - API failed")
                    print(f"   ❌ Cannot validate data - API returned {response.status_code}")
            else:
                self.test_results["data_validation_passed"] = True  # Skip if no API
                print("   ⚠️ No API endpoint for data validation")
                
        except Exception as e:
            self.test_results["issues_found"].append(f"Data validation error: {str(e)}")
            print(f"   ❌ Data validation failed: {e}")
    
    def test_sentiment_analysis_comprehensive(self):
        """Run comprehensive test for Sentiment Analysis"""
        print(f"\n================================================================================")
        print(f"🚀 COMPREHENSIVE TEST: Sentiment Analysis")
        print(f"================================================================================")
        
        # Run all tests
        self.test_sentiment_analysis_api_endpoint()
        self.test_sentiment_analysis_ui_interaction()
        self.test_sentiment_analysis_data_validation()
        
        # Generate results
        self._generate_test_report()
    
    def _validate_response_fields(self, data: dict, expected_fields: list):
        """Validate that response contains expected fields"""
        missing_fields = []
        
        def check_nested_fields(obj, fields, path=""):
            for field in fields:
                if isinstance(obj, dict):
                    if field not in obj:
                        missing_fields.append(f"{path}.{field}" if path else field)
                elif isinstance(obj, list) and obj:
                    # Check first item in list
                    check_nested_fields(obj[0], [field], path)
        
        check_nested_fields(data, expected_fields)
        
        if missing_fields:
            self.test_results["issues_found"].append(f"Missing fields: {', '.join(missing_fields)}")
    
    def _validate_data_quality(self, data: dict) -> bool:
        """Validate data quality and consistency"""
        issues = []
        
        # Check for empty/null values in critical fields
        if isinstance(data, dict):
            for key, value in data.items():
                if value is None or value == "":
                    issues.append(f"Empty value for key: {key}")
                elif isinstance(value, list) and len(value) == 0:
                    issues.append(f"Empty list for key: {key}")
        
        # Add specific validations based on feature type
        feature_type = "sentiment_analysis".split("_")[0]
        
        if feature_type == "sentiment":
            issues.extend(self._validate_sentiment_data(data))
        elif feature_type == "portfolio":
            issues.extend(self._validate_portfolio_data(data))
        elif feature_type == "options":
            issues.extend(self._validate_options_data(data))
        
        if issues:
            self.test_results["issues_found"].extend(issues)
            return False
        
        return True
    
    def _validate_sentiment_data(self, data: dict) -> list:
        """Validate sentiment-specific data"""
        issues = []
        
        if "results" in data:
            for stock, metrics in data["results"].items():
                sentiment = metrics.get("overall_sentiment", 0)
                if not (0 <= sentiment <= 1):
                    issues.append(f"{stock} sentiment {sentiment} out of range [0,1]")
                
                if not metrics.get("recommendation"):
                    issues.append(f"{stock} missing recommendation")
        
        return issues
    
    def _validate_portfolio_data(self, data: dict) -> list:
        """Validate portfolio-specific data"""
        issues = []
        
        if "portfolio_weights" in data:
            weights = data["portfolio_weights"]
            weights_sum = sum(weights.values())
            if abs(weights_sum - 1.0) > 0.01:
                issues.append(f"Portfolio weights sum to {weights_sum:.4f}, not 1.0")
        
        if "performance_metrics" in data:
            metrics = data["performance_metrics"]
            sharpe = metrics.get("sharpe_ratio", 0)
            if not (0 < sharpe < 5):
                issues.append(f"Sharpe ratio {sharpe} unrealistic")
        
        return issues
    
    def _validate_options_data(self, data: dict) -> list:
        """Validate options-specific data"""
        issues = []
        
        # Add options-specific validations
        if "buy_signals" in data or "sell_signals" in data:
            signals = data.get("buy_signals", []) + data.get("sell_signals", [])
            if len(signals) == 0:
                issues.append("No trading signals found")
        
        return issues
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        print(f"\n============================================================")
        print(f"📋 TEST REPORT: Sentiment Analysis")
        print(f"============================================================")
        
        # Test results summary
        api_status = "✅ PASSED" if self.test_results["api_test_passed"] else "❌ FAILED"
        ui_status = "✅ PASSED" if self.test_results["ui_test_passed"] else "❌ FAILED"
        data_status = "✅ PASSED" if self.test_results["data_validation_passed"] else "❌ FAILED"
        
        print(f"🔌 API Endpoint Test: {api_status}")
        print(f"🖱️ UI Interaction Test: {ui_status}")
        print(f"📊 Data Validation Test: {data_status}")
        
        # Performance metrics
        if self.test_results["performance_metrics"]:
            print(f"\n⏱️ Performance Metrics:")
            for metric, value in self.test_results["performance_metrics"].items():
                print(f"   {metric}: {value:.2f}s")
        
        # Issues found
        if self.test_results["issues_found"]:
            print(f"\n⚠️ Issues Found ({len(self.test_results['issues_found'])})):")
            for issue in self.test_results["issues_found"]:
                print(f"   • {issue}")
        else:
            print(f"\n✅ No issues found!")
        
        # Screenshots
        if self.test_results["screenshots"]:
            print(f"\n📸 Screenshots Captured ({len(self.test_results['screenshots'])})):")
            for screenshot in self.test_results["screenshots"]:
                print(f"   📄 {screenshot}")
        
        # Overall result
        all_passed = all([
            self.test_results["api_test_passed"],
            self.test_results["ui_test_passed"], 
            self.test_results["data_validation_passed"]
        ])
        
        overall_status = "✅ PASSED" if all_passed else "❌ FAILED"
        print(f"\n🎯 Overall Result: {overall_status}")
        
        # Save detailed report
        report_file = f"reports/individual_test_sentiment_analysis_{int(time.time())}.json"
        os.makedirs("reports", exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"💾 Detailed report saved: {report_file}")
        
        return all_passed


if __name__ == "__main__":
    """Run individual feature test"""
    print(f"🧪 STARTING INDIVIDUAL FEATURE TEST")
    print(f"📋 Feature: Sentiment Analysis")
    print(f"🔑 Key: sentiment_analysis")
    
    test = TestSentimentanalysis()
    
    try:
        test.setup_method(lambda: None)
        result = test.test_sentiment_analysis_comprehensive()
        
        if result:
            print(f"\n✨ Sentiment Analysis test completed successfully!")
        else:
            print(f"\n❌ Sentiment Analysis test failed!")
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        
    finally:
        try:
            test.teardown_method(lambda: None)
        except:
            pass
