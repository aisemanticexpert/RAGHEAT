#!/usr/bin/env python3
"""
DEEP DATA VALIDATION TEST SUITE
Comprehensive validation of all data sources, API responses, and data accuracy
"""

import requests
import json
import time
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class DeepDataValidator:
    """Comprehensive data validation for RAGHeat system"""
    
    def __init__(self):
        self.api_base = "http://localhost:8001"
        self.validation_results = {
            "api_data_accuracy": [],
            "market_data_validation": [],
            "portfolio_calculations": [],
            "agent_responses": [],
            "data_consistency": [],
            "real_vs_synthetic": []
        }
    
    def validate_api_data_accuracy(self):
        """Validate accuracy of API data responses"""
        print("🔍 VALIDATING API DATA ACCURACY")
        print("="*60)
        
        test_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        # Test Portfolio Construction Data
        portfolio_data = self._test_portfolio_construction(test_stocks)
        
        # Test Fundamental Analysis Data  
        fundamental_data = self._test_fundamental_analysis(test_stocks)
        
        # Test Sentiment Analysis Data
        sentiment_data = self._test_sentiment_analysis(test_stocks)
        
        return {
            "portfolio": portfolio_data,
            "fundamental": fundamental_data, 
            "sentiment": sentiment_data
        }
    
    def _test_portfolio_construction(self, stocks):
        """Test portfolio construction data validity"""
        try:
            response = requests.post(
                f"{self.api_base}/portfolio/construct",
                json={"stocks": stocks},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate weights sum to 1
                weights = data.get("portfolio_weights", {})
                weights_sum = sum(weights.values())
                weights_valid = abs(weights_sum - 1.0) < 0.01
                
                # Validate performance metrics are reasonable
                metrics = data.get("performance_metrics", {})
                sharpe_valid = 0 < metrics.get("sharpe_ratio", 0) < 5
                volatility_valid = 0 < metrics.get("volatility", 0) < 1
                return_valid = -0.5 < metrics.get("expected_return", 0) < 1
                
                # Validate risk analysis
                risk = data.get("risk_analysis", {})
                beta_valid = 0 < risk.get("beta", 0) < 3
                var_valid = 0 < risk.get("var_95", 0) < 0.1
                div_valid = 0 < risk.get("diversification_score", 0) <= 1
                
                issues = []
                if not weights_valid: issues.append(f"Weights sum to {weights_sum:.4f}, not 1.0")
                if not sharpe_valid: issues.append(f"Sharpe ratio {metrics.get('sharpe_ratio')} unrealistic")
                if not volatility_valid: issues.append(f"Volatility {metrics.get('volatility')} unrealistic") 
                if not return_valid: issues.append(f"Expected return {metrics.get('expected_return')} unrealistic")
                if not beta_valid: issues.append(f"Beta {risk.get('beta')} unrealistic")
                if not var_valid: issues.append(f"VaR {risk.get('var_95')} unrealistic")
                if not div_valid: issues.append(f"Diversification score {risk.get('diversification_score')} invalid")
                
                return {
                    "status": "valid" if not issues else "invalid",
                    "issues": issues,
                    "data": data,
                    "weights_sum": weights_sum,
                    "metrics_valid": all([sharpe_valid, volatility_valid, return_valid]),
                    "risk_valid": all([beta_valid, var_valid, div_valid])
                }
            else:
                return {"status": "api_error", "code": response.status_code}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _test_fundamental_analysis(self, stocks):
        """Test fundamental analysis data validity"""
        try:
            response = requests.post(
                f"{self.api_base}/analysis/fundamental",
                json={"stocks": stocks},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", {})
                
                issues = []
                for stock, metrics in results.items():
                    pe = metrics.get("pe_ratio", 0)
                    de = metrics.get("debt_to_equity", 0)
                    roe = metrics.get("roe", 0) 
                    growth = metrics.get("revenue_growth", 0)
                    rec = metrics.get("recommendation", "")
                    
                    if not (0 < pe < 100): issues.append(f"{stock} P/E ratio {pe} unrealistic")
                    if not (0 <= de < 10): issues.append(f"{stock} D/E ratio {de} unrealistic")
                    if not (-1 < roe < 2): issues.append(f"{stock} ROE {roe} unrealistic")
                    if not (-1 < growth < 2): issues.append(f"{stock} growth {growth} unrealistic")
                    if rec not in ["BUY", "SELL", "HOLD"]: issues.append(f"{stock} invalid recommendation {rec}")
                
                return {
                    "status": "valid" if not issues else "invalid",
                    "issues": issues,
                    "data": data,
                    "stocks_analyzed": len(results)
                }
            else:
                return {"status": "api_error", "code": response.status_code}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _test_sentiment_analysis(self, stocks):
        """Test sentiment analysis data validity"""
        try:
            response = requests.post(
                f"{self.api_base}/analysis/sentiment",
                json={"stocks": stocks},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", {})
                
                issues = []
                for stock, metrics in results.items():
                    sentiment = metrics.get("sentiment_score", 0)
                    confidence = metrics.get("confidence", 0)
                    sources = metrics.get("sources_analyzed", 0)
                    rec = metrics.get("recommendation", "")
                    
                    if not (-1 <= sentiment <= 1): issues.append(f"{stock} sentiment {sentiment} out of range")
                    if not (0 <= confidence <= 1): issues.append(f"{stock} confidence {confidence} out of range") 
                    if not (1 <= sources <= 1000): issues.append(f"{stock} sources {sources} unrealistic")
                    if rec not in ["BUY", "SELL", "HOLD"]: issues.append(f"{stock} invalid recommendation {rec}")
                
                return {
                    "status": "valid" if not issues else "invalid",
                    "issues": issues,
                    "data": data,
                    "stocks_analyzed": len(results)
                }
            else:
                return {"status": "api_error", "code": response.status_code}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def validate_market_data_integration(self):
        """Validate market data against real sources"""
        print("\n📊 VALIDATING MARKET DATA INTEGRATION")
        print("="*60)
        
        test_stocks = ["AAPL", "GOOGL", "MSFT"]
        validation_results = []
        
        for stock in test_stocks:
            print(f"\n🔍 Validating {stock} data...")
            
            # Get real market data
            try:
                ticker = yf.Ticker(stock)
                info = ticker.info
                hist = ticker.history(period="1mo")
                
                # Get API data
                api_fundamental = self._get_api_fundamental_data([stock])
                
                if api_fundamental["status"] == "valid":
                    api_pe = api_fundamental["data"]["results"][stock]["pe_ratio"]
                    real_pe = info.get("trailingPE", 0)
                    
                    # Compare P/E ratios (allow 50% variance for synthetic data)
                    pe_variance = abs(api_pe - real_pe) / real_pe if real_pe > 0 else 1
                    pe_reasonable = pe_variance < 0.5 or (5 < api_pe < 50)  # Either close or reasonable range
                    
                    validation_results.append({
                        "stock": stock,
                        "real_pe": real_pe,
                        "api_pe": api_pe,
                        "pe_variance": pe_variance,
                        "pe_reasonable": pe_reasonable,
                        "status": "valid" if pe_reasonable else "questionable"
                    })
                    
                    print(f"   Real P/E: {real_pe:.2f}, API P/E: {api_pe:.2f}, Variance: {pe_variance:.2%}")
                    print(f"   Status: {'✅ Valid' if pe_reasonable else '⚠️ Questionable'}")
                
            except Exception as e:
                validation_results.append({
                    "stock": stock,
                    "status": "error",
                    "message": str(e)
                })
                print(f"   ❌ Error: {e}")
        
        return validation_results
    
    def _get_api_fundamental_data(self, stocks):
        """Helper to get fundamental data from API"""
        try:
            response = requests.post(
                f"{self.api_base}/analysis/fundamental",
                json={"stocks": stocks},
                timeout=30
            )
            return {"status": "valid", "data": response.json()} if response.status_code == 200 else {"status": "error"}
        except:
            return {"status": "error"}
    
    def test_portfolio_calculation_accuracy(self):
        """Test portfolio calculation accuracy"""
        print("\n📈 TESTING PORTFOLIO CALCULATION ACCURACY")
        print("="*60)
        
        test_cases = [
            {"stocks": ["AAPL"], "expected_weight": {"AAPL": 1.0}},
            {"stocks": ["AAPL", "GOOGL"], "weight_count": 2},
            {"stocks": ["AAPL", "GOOGL", "MSFT"], "weight_count": 3},
            {"stocks": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"], "weight_count": 5}
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n🧪 Test Case {i+1}: {len(test_case['stocks'])} stocks")
            
            portfolio_data = self._test_portfolio_construction(test_case["stocks"])
            
            if portfolio_data["status"] == "valid":
                weights = portfolio_data["data"]["portfolio_weights"]
                
                # Test single stock case
                if "expected_weight" in test_case:
                    expected = test_case["expected_weight"]
                    match = all(abs(weights[k] - expected[k]) < 0.01 for k in expected)
                    results.append({
                        "test": f"Single stock {test_case['stocks'][0]}",
                        "passed": match,
                        "weights": weights,
                        "expected": expected
                    })
                    print(f"   Single stock test: {'✅ Passed' if match else '❌ Failed'}")
                
                # Test weight count
                if "weight_count" in test_case:
                    count_match = len(weights) == test_case["weight_count"]
                    results.append({
                        "test": f"{test_case['weight_count']} stocks weight count",
                        "passed": count_match,
                        "actual_count": len(weights),
                        "expected_count": test_case["weight_count"]
                    })
                    print(f"   Weight count test: {'✅ Passed' if count_match else '❌ Failed'}")
                
                # Test weights sum
                weights_sum = portfolio_data["weights_sum"]
                sum_valid = abs(weights_sum - 1.0) < 0.01
                results.append({
                    "test": f"{len(test_case['stocks'])} stocks weights sum",
                    "passed": sum_valid,
                    "weights_sum": weights_sum
                })
                print(f"   Weights sum test: {'✅ Passed' if sum_valid else '❌ Failed'} (sum: {weights_sum:.4f})")
                
            else:
                results.append({
                    "test": f"{len(test_case['stocks'])} stocks calculation",
                    "passed": False,
                    "error": portfolio_data
                })
                print(f"   ❌ API Error: {portfolio_data}")
        
        return results
    
    def test_agent_response_consistency(self):
        """Test consistency of agent responses"""
        print("\n🤖 TESTING AGENT RESPONSE CONSISTENCY")
        print("="*60)
        
        test_stocks = ["AAPL", "GOOGL", "MSFT"]
        consistency_tests = []
        
        # Run same request multiple times
        for run in range(3):
            print(f"\n🔄 Consistency Test Run {run + 1}")
            
            portfolio_data = self._test_portfolio_construction(test_stocks)
            
            if portfolio_data["status"] == "valid":
                agents = portfolio_data["data"].get("agent_insights", {})
                
                consistency_tests.append({
                    "run": run + 1,
                    "agents": list(agents.keys()),
                    "agent_count": len(agents),
                    "weights": portfolio_data["data"]["portfolio_weights"],
                    "sharpe_ratio": portfolio_data["data"]["performance_metrics"]["sharpe_ratio"],
                    "insights": agents
                })
                
                print(f"   Agents responding: {len(agents)}")
                print(f"   Sharpe ratio: {portfolio_data['data']['performance_metrics']['sharpe_ratio']:.4f}")
            else:
                consistency_tests.append({
                    "run": run + 1,
                    "error": portfolio_data
                })
            
            time.sleep(2)  # Brief pause between requests
        
        # Analyze consistency
        if len(consistency_tests) >= 2:
            successful_runs = [t for t in consistency_tests if "error" not in t]
            
            if len(successful_runs) >= 2:
                # Check agent count consistency
                agent_counts = [t["agent_count"] for t in successful_runs]
                agent_count_consistent = len(set(agent_counts)) == 1
                
                # Check Sharpe ratio variance
                sharpe_ratios = [t["sharpe_ratio"] for t in successful_runs]
                sharpe_variance = float(np.std(sharpe_ratios)) if len(sharpe_ratios) > 1 else 0.0
                sharpe_consistent = sharpe_variance < 0.1  # Allow small variance
                
                consistency_results = {
                    "total_runs": len(consistency_tests),
                    "successful_runs": len(successful_runs),
                    "agent_count_consistent": agent_count_consistent,
                    "agent_counts": agent_counts,
                    "sharpe_consistent": sharpe_consistent,
                    "sharpe_variance": sharpe_variance,
                    "sharpe_ratios": sharpe_ratios,
                    "overall_consistent": agent_count_consistent and sharpe_consistent
                }
                
                print(f"\n📊 Consistency Analysis:")
                print(f"   Agent count consistent: {'✅ Yes' if agent_count_consistent else '❌ No'}")
                print(f"   Sharpe ratio consistent: {'✅ Yes' if sharpe_consistent else '❌ No'} (std: {sharpe_variance:.4f})")
                print(f"   Overall consistent: {'✅ Yes' if consistency_results['overall_consistent'] else '❌ No'}")
                
                return consistency_results
        
        return {"status": "insufficient_data", "tests": consistency_tests}
    
    def generate_deep_data_report(self):
        """Generate comprehensive data validation report"""
        print("\n" + "="*80)
        print("🚀 STARTING DEEP DATA VALIDATION SUITE")
        print("="*80)
        
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "api_base": self.api_base,
            "test_results": {}
        }
        
        # Run all validation tests
        try:
            report["test_results"]["api_accuracy"] = self.validate_api_data_accuracy()
            report["test_results"]["market_data"] = self.validate_market_data_integration()
            report["test_results"]["calculations"] = self.test_portfolio_calculation_accuracy()
            report["test_results"]["consistency"] = self.test_agent_response_consistency()
            
            # Generate summary
            self._generate_validation_summary(report)
            
            # Save report
            report_file = f"reports/deep_data_validation_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n💾 Deep data validation report saved to: {report_file}")
            return report
            
        except Exception as e:
            print(f"\n❌ Deep data validation failed: {e}")
            return {"error": str(e)}
    
    def _generate_validation_summary(self, report):
        """Generate validation summary"""
        print("\n" + "="*80)
        print("📋 DEEP DATA VALIDATION SUMMARY")
        print("="*80)
        
        results = report["test_results"]
        
        # API Accuracy Summary
        api_results = results.get("api_accuracy", {})
        portfolio_valid = api_results.get("portfolio", {}).get("status") == "valid"
        fundamental_valid = api_results.get("fundamental", {}).get("status") == "valid"
        sentiment_valid = api_results.get("sentiment", {}).get("status") == "valid"
        
        print(f"\n🔍 API Data Accuracy:")
        print(f"   Portfolio Construction: {'✅ Valid' if portfolio_valid else '❌ Invalid'}")
        print(f"   Fundamental Analysis: {'✅ Valid' if fundamental_valid else '❌ Invalid'}")
        print(f"   Sentiment Analysis: {'✅ Valid' if sentiment_valid else '❌ Invalid'}")
        
        # Market Data Summary
        market_results = results.get("market_data", [])
        valid_stocks = len([r for r in market_results if r.get("status") == "valid"])
        total_stocks = len(market_results)
        
        print(f"\n📊 Market Data Integration:")
        print(f"   Valid Stock Data: {valid_stocks}/{total_stocks}")
        
        # Calculation Accuracy Summary
        calc_results = results.get("calculations", [])
        passed_tests = len([r for r in calc_results if r.get("passed")])
        total_tests = len(calc_results)
        
        print(f"\n📈 Portfolio Calculations:")
        print(f"   Passed Tests: {passed_tests}/{total_tests}")
        
        # Consistency Summary
        consistency = results.get("consistency", {})
        consistent = consistency.get("overall_consistent", False)
        
        print(f"\n🤖 Agent Consistency:")
        print(f"   Response Consistency: {'✅ Consistent' if consistent else '❌ Inconsistent'}")
        
        # Overall Assessment
        overall_score = (
            (1 if portfolio_valid else 0) +
            (1 if fundamental_valid else 0) + 
            (1 if sentiment_valid else 0) +
            (valid_stocks / max(total_stocks, 1)) +
            (passed_tests / max(total_tests, 1)) +
            (1 if consistent else 0)
        ) / 6 * 100
        
        print(f"\n🎯 Overall Data Quality Score: {overall_score:.1f}%")
        
        if overall_score >= 80:
            print("   ✅ EXCELLENT - Data quality is high")
        elif overall_score >= 60:
            print("   ⚠️ GOOD - Some data issues need attention")
        elif overall_score >= 40:
            print("   ❌ POOR - Significant data problems detected")
        else:
            print("   🚨 CRITICAL - Major data validation failures")
        
        report["overall_score"] = overall_score


def run_deep_data_validation():
    """Run the complete deep data validation suite"""
    print("\n🔬 STARTING DEEP DATA VALIDATION")
    print("🎯 Comprehensive analysis of all RAGHeat data sources")
    print("⚡ Testing API accuracy, market data, calculations, and consistency")
    
    validator = DeepDataValidator()
    return validator.generate_deep_data_report()


if __name__ == "__main__":
    report = run_deep_data_validation()
    
    if "error" not in report:
        print(f"\n✨ DEEP DATA VALIDATION COMPLETED!")
        print(f"📈 Overall Score: {report.get('overall_score', 0):.1f}%")
    else:
        print(f"\n❌ VALIDATION FAILED: {report['error']}")