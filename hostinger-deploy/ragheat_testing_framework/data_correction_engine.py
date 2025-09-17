#!/usr/bin/env python3
"""
DATA CORRECTION ENGINE
Identifies and fixes data issues in the RAGHeat system
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

class DataCorrectionEngine:
    """Engine for identifying and correcting data issues"""
    
    def __init__(self):
        self.api_base = "http://localhost:8001"
        self.corrections_applied = []
        self.issues_detected = []
    
    def detect_data_issues(self):
        """Detect common data issues in the system"""
        print("🔍 DETECTING DATA ISSUES")
        print("="*50)
        
        issues = []
        
        # Test sentiment analysis for invalid data
        sentiment_issues = self._check_sentiment_data_issues()
        if sentiment_issues:
            issues.extend(sentiment_issues)
        
        # Test consistency issues
        consistency_issues = self._check_consistency_issues()
        if consistency_issues:
            issues.extend(consistency_issues)
        
        # Test range validation issues
        range_issues = self._check_data_range_issues()
        if range_issues:
            issues.extend(range_issues)
        
        self.issues_detected = issues
        return issues
    
    def _check_sentiment_data_issues(self):
        """Check for sentiment analysis data issues"""
        print("\n📊 Checking sentiment data issues...")
        
        try:
            response = requests.post(
                f"{self.api_base}/analysis/sentiment",
                json={"stocks": ["AAPL", "GOOGL", "MSFT"]},
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
                    
                    if not (-1 <= sentiment <= 1):
                        issues.append({
                            "type": "sentiment_range_invalid",
                            "stock": stock,
                            "value": sentiment,
                            "expected_range": "[-1, 1]",
                            "severity": "high"
                        })
                    
                    if not (0 <= confidence <= 1):
                        issues.append({
                            "type": "confidence_range_invalid", 
                            "stock": stock,
                            "value": confidence,
                            "expected_range": "[0, 1]",
                            "severity": "medium"
                        })
                    
                    if sources <= 0 or sources > 10000:
                        issues.append({
                            "type": "sources_count_unrealistic",
                            "stock": stock,
                            "value": sources,
                            "expected_range": "(0, 10000]",
                            "severity": "low"
                        })
                
                if issues:
                    print(f"   ❌ Found {len(issues)} sentiment data issues")
                else:
                    print("   ✅ Sentiment data validation passed")
                
                return issues
            else:
                return [{"type": "api_error", "endpoint": "sentiment", "code": response.status_code}]
                
        except Exception as e:
            return [{"type": "api_exception", "endpoint": "sentiment", "error": str(e)}]
    
    def _check_consistency_issues(self):
        """Check for data consistency issues"""
        print("\n🔄 Checking consistency issues...")
        
        issues = []
        test_stocks = ["AAPL", "GOOGL", "MSFT"]
        
        # Test multiple portfolio construction calls for consistency
        responses = []
        for i in range(3):
            try:
                response = requests.post(
                    f"{self.api_base}/portfolio/construct",
                    json={"stocks": test_stocks},
                    timeout=30
                )
                if response.status_code == 200:
                    responses.append(response.json())
                time.sleep(1)
            except Exception as e:
                issues.append({
                    "type": "consistency_test_failed",
                    "run": i+1,
                    "error": str(e),
                    "severity": "high"
                })
        
        if len(responses) >= 2:
            # Check Sharpe ratio variance
            sharpe_ratios = []
            for resp in responses:
                sharpe = resp.get("performance_metrics", {}).get("sharpe_ratio", 0)
                sharpe_ratios.append(sharpe)
            
            variance = np.std(sharpe_ratios) if len(sharpe_ratios) > 1 else 0
            if variance > 0.2:  # High variance indicates inconsistency
                issues.append({
                    "type": "sharpe_ratio_inconsistent",
                    "variance": float(variance),
                    "values": sharpe_ratios,
                    "severity": "medium"
                })
                print(f"   ❌ High Sharpe ratio variance: {variance:.4f}")
            else:
                print(f"   ✅ Sharpe ratio consistency OK: {variance:.4f}")
        
        return issues
    
    def _check_data_range_issues(self):
        """Check for data range validation issues"""
        print("\n📊 Checking data range issues...")
        
        issues = []
        
        try:
            # Test fundamental analysis ranges
            response = requests.post(
                f"{self.api_base}/analysis/fundamental",
                json={"stocks": ["AAPL", "GOOGL", "MSFT"]},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", {})
                
                for stock, metrics in results.items():
                    pe = metrics.get("pe_ratio", 0)
                    de = metrics.get("debt_to_equity", 0)
                    roe = metrics.get("roe", 0)
                    growth = metrics.get("revenue_growth", 0)
                    
                    if not (0 < pe < 100):
                        issues.append({
                            "type": "pe_ratio_unrealistic",
                            "stock": stock,
                            "value": pe,
                            "expected_range": "(0, 100)",
                            "severity": "medium"
                        })
                    
                    if not (0 <= de < 10):
                        issues.append({
                            "type": "debt_equity_unrealistic",
                            "stock": stock,
                            "value": de,
                            "expected_range": "[0, 10)",
                            "severity": "medium"
                        })
                    
                    if not (-1 < roe < 2):
                        issues.append({
                            "type": "roe_unrealistic",
                            "stock": stock,
                            "value": roe,
                            "expected_range": "(-1, 2)",
                            "severity": "low"
                        })
                    
                    if not (-1 < growth < 2):
                        issues.append({
                            "type": "revenue_growth_unrealistic",
                            "stock": stock,
                            "value": growth,
                            "expected_range": "(-1, 2)",
                            "severity": "low"
                        })
        
        except Exception as e:
            issues.append({
                "type": "fundamental_analysis_failed",
                "error": str(e),
                "severity": "high"
            })
        
        if issues:
            print(f"   ❌ Found {len(issues)} range validation issues")
        else:
            print("   ✅ Data range validation passed")
        
        return issues
    
    def suggest_corrections(self):
        """Suggest corrections for detected issues"""
        print("\n🔧 SUGGESTING DATA CORRECTIONS")
        print("="*50)
        
        suggestions = []
        
        for issue in self.issues_detected:
            issue_type = issue.get("type", "unknown")
            
            if issue_type == "sentiment_range_invalid":
                suggestions.append({
                    "issue": issue,
                    "correction": "Clamp sentiment score to [-1, 1] range",
                    "implementation": f"np.clip({issue['value']}, -1, 1)",
                    "corrected_value": max(-1, min(1, issue['value']))
                })
            
            elif issue_type == "confidence_range_invalid":
                suggestions.append({
                    "issue": issue,
                    "correction": "Clamp confidence to [0, 1] range", 
                    "implementation": f"np.clip({issue['value']}, 0, 1)",
                    "corrected_value": max(0, min(1, issue['value']))
                })
            
            elif issue_type == "sharpe_ratio_inconsistent":
                suggestions.append({
                    "issue": issue,
                    "correction": "Add randomization seed for consistent results",
                    "implementation": "np.random.seed(42) in portfolio construction",
                    "priority": "high"
                })
            
            elif issue_type == "pe_ratio_unrealistic":
                if issue['value'] <= 0:
                    corrected = 15.0  # Market average
                elif issue['value'] > 100:
                    corrected = 35.0  # High but reasonable
                else:
                    corrected = issue['value']
                
                suggestions.append({
                    "issue": issue,
                    "correction": "Replace unrealistic P/E with reasonable value",
                    "implementation": f"Use market average or capped value",
                    "corrected_value": corrected
                })
            
            else:
                suggestions.append({
                    "issue": issue,
                    "correction": "Manual review required",
                    "implementation": "Developer investigation needed"
                })
        
        return suggestions
    
    def generate_correction_report(self):
        """Generate comprehensive data correction report"""
        print("\n" + "="*80)
        print("🚀 STARTING DATA CORRECTION ANALYSIS")
        print("="*80)
        
        # Detect issues
        issues = self.detect_data_issues()
        
        # Generate suggestions
        suggestions = self.suggest_corrections()
        
        # Create report
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_issues_detected": len(issues),
            "issues_by_severity": {
                "high": len([i for i in issues if i.get("severity") == "high"]),
                "medium": len([i for i in issues if i.get("severity") == "medium"]),  
                "low": len([i for i in issues if i.get("severity") == "low"])
            },
            "detected_issues": issues,
            "correction_suggestions": suggestions,
            "api_endpoints_tested": [
                "/portfolio/construct",
                "/analysis/fundamental", 
                "/analysis/sentiment"
            ]
        }
        
        # Print summary
        self._print_correction_summary(report)
        
        # Save report
        report_file = f"reports/data_correction_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Data correction report saved to: {report_file}")
        return report
    
    def _print_correction_summary(self, report):
        """Print correction analysis summary"""
        print("\n" + "="*60)
        print("📋 DATA CORRECTION SUMMARY")
        print("="*60)
        
        total = report["total_issues_detected"]
        severity = report["issues_by_severity"]
        
        print(f"\n🔍 Issues Detected: {total}")
        print(f"   🚨 High Severity: {severity['high']}")
        print(f"   ⚠️ Medium Severity: {severity['medium']}")
        print(f"   💡 Low Severity: {severity['low']}")
        
        if total == 0:
            print("\n✅ NO ISSUES DETECTED - Data quality is excellent!")
            return
        
        # Group issues by type
        issue_types = {}
        for issue in report["detected_issues"]:
            issue_type = issue.get("type", "unknown")
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1
        
        print(f"\n🔧 Issue Types:")
        for issue_type, count in issue_types.items():
            print(f"   • {issue_type.replace('_', ' ').title()}: {count}")
        
        # Print top suggestions
        suggestions = report["correction_suggestions"]
        high_priority = [s for s in suggestions if s.get("priority") == "high"]
        
        if high_priority:
            print(f"\n🚨 HIGH PRIORITY CORRECTIONS NEEDED:")
            for suggestion in high_priority[:3]:  # Top 3
                issue = suggestion["issue"]
                correction = suggestion["correction"]
                print(f"   • {issue.get('type', 'Unknown')}: {correction}")
        
        # Overall assessment
        if severity['high'] > 0:
            print(f"\n❌ CRITICAL - {severity['high']} high-severity issues require immediate attention")
        elif severity['medium'] > 3:
            print(f"\n⚠️ MODERATE - {severity['medium']} medium-severity issues should be addressed")
        elif total > 0:
            print(f"\n💡 MINOR - {total} low-severity issues for optimization")
        else:
            print(f"\n✅ EXCELLENT - No data quality issues detected")


def run_data_correction_analysis():
    """Run the complete data correction analysis"""
    print("\n🔧 STARTING DATA CORRECTION ENGINE")
    print("🎯 Identifying and suggesting fixes for data quality issues")
    print("⚡ Analyzing API responses, data ranges, and consistency")
    
    engine = DataCorrectionEngine()
    return engine.generate_correction_report()


if __name__ == "__main__":
    report = run_data_correction_analysis()
    
    total_issues = report.get("total_issues_detected", 0)
    if total_issues == 0:
        print(f"\n✨ DATA CORRECTION ANALYSIS COMPLETED!")
        print(f"🎉 No issues detected - data quality is excellent!")
    else:
        print(f"\n🔧 DATA CORRECTION ANALYSIS COMPLETED!")
        print(f"📊 {total_issues} issues detected with suggested fixes")
        
        high_severity = report.get("issues_by_severity", {}).get("high", 0)
        if high_severity > 0:
            print(f"🚨 URGENT: {high_severity} high-severity issues need immediate attention!")