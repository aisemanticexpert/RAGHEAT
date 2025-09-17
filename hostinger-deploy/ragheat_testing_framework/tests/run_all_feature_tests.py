#!/usr/bin/env python3
"""
ALL FEATURES TEST RUNNER
Executes all 45 individual feature tests independently on Chrome
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


class AllFeaturesTestRunner:
    """Runner for all individual feature tests"""
    
    def __init__(self):
        self.test_dir = Path("tests/individual_features")
        self.results = {}
        self.start_time = time.time()
    
    def run_single_test(self, test_file):
        """Run a single feature test"""
        feature_name = test_file.stem.replace("test_", "")
        print(f"🧪 Running {feature_name} test...")
        
        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test
            )
            
            execution_time = time.time() - start_time
            
            success = result.returncode == 0
            
            self.results[feature_name] = {
                "success": success,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status} ({execution_time:.1f}s)")
            
            return feature_name, success
            
        except subprocess.TimeoutExpired:
            self.results[feature_name] = {
                "success": False,
                "execution_time": 300,
                "error": "Test timeout (5 minutes)",
                "return_code": -1
            }
            print(f"   ⏰ TIMEOUT")
            return feature_name, False
            
        except Exception as e:
            self.results[feature_name] = {
                "success": False,
                "execution_time": 0,
                "error": str(e),
                "return_code": -1
            }
            print(f"   💥 ERROR: {e}")
            return feature_name, False
    
    def run_all_tests_sequential(self):
        """Run all tests sequentially"""
        print("\n🔄 RUNNING ALL TESTS SEQUENTIALLY")
        print("="*60)
        
        test_files = list(self.test_dir.glob("test_*.py"))
        total_tests = len(test_files)
        
        print(f"📊 Found {total_tests} individual feature tests")
        
        passed = 0
        failed = 0
        
        for i, test_file in enumerate(test_files, 1):
            print(f"\n[{i}/{total_tests}] {test_file.stem}")
            feature_name, success = self.run_single_test(test_file)
            
            if success:
                passed += 1
            else:
                failed += 1
        
        return passed, failed, total_tests
    
    def run_all_tests_parallel(self, max_workers=3):
        """Run tests in parallel (limited workers to avoid resource conflicts)"""
        print(f"\n⚡ RUNNING ALL TESTS IN PARALLEL ({max_workers} workers)")
        print("="*60)
        
        test_files = list(self.test_dir.glob("test_*.py"))
        total_tests = len(test_files)
        
        print(f"📊 Found {total_tests} individual feature tests")
        
        passed = 0
        failed = 0
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {executor.submit(self.run_single_test, test_file): test_file 
                              for test_file in test_files}
            
            for future in as_completed(future_to_test):
                test_file = future_to_test[future]
                completed += 1
                
                try:
                    feature_name, success = future.result()
                    
                    if success:
                        passed += 1
                    else:
                        failed += 1
                    
                    print(f"[{completed}/{total_tests}] {feature_name} completed")
                    
                except Exception as e:
                    failed += 1
                    print(f"[{completed}/{total_tests}] {test_file.stem} failed with exception: {e}")
        
        return passed, failed, total_tests
    
    def generate_summary_report(self, passed, failed, total_tests):
        """Generate comprehensive summary report"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("🏆 ALL FEATURES TEST EXECUTION SUMMARY")
        print("="*80)
        
        print(f"\n📊 Test Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success Rate: {(passed/total_tests)*100:.1f}%")
        print(f"   ⏱️ Total Time: {total_time:.1f}s")
        
        # Feature category breakdown
        categories = {
            "dashboard": [],
            "analysis": [],
            "knowledge": [],
            "options": [],
            "agents": [],
            "realtime": [],
            "visualization": []
        }
        
        for feature_name, result in self.results.items():
            for category in categories.keys():
                if category in feature_name:
                    categories[category].append((feature_name, result["success"]))
                    break
            else:
                categories.setdefault("other", []).append((feature_name, result["success"]))
        
        print(f"\n📋 Results by Category:")
        for category, tests in categories.items():
            if tests:
                category_passed = sum(1 for _, success in tests if success)
                category_total = len(tests)
                category_rate = (category_passed/category_total)*100 if category_total > 0 else 0
                print(f"   {category.title()}: {category_passed}/{category_total} ({category_rate:.1f}%)")
        
        # Failed tests details
        failed_tests = [name for name, result in self.results.items() if not result["success"]]
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test_name in failed_tests:
                result = self.results[test_name]
                error_info = result.get("error", "Unknown error")
                print(f"   • {test_name}: {error_info}")
        
        # Performance insights
        avg_time = sum(r["execution_time"] for r in self.results.values()) / len(self.results)
        slowest_tests = sorted(self.results.items(), 
                              key=lambda x: x[1]["execution_time"], 
                              reverse=True)[:5]
        
        print(f"\n⏱️ Performance Insights:")
        print(f"   Average test time: {avg_time:.1f}s")
        print(f"   Slowest tests:")
        for name, result in slowest_tests:
            print(f"     • {name}: {result['execution_time']:.1f}s")
        
        # Save detailed results
        report_file = f"reports/all_features_test_results_{int(time.time())}.json"
        Path("reports").mkdir(exist_ok=True)
        
        detailed_report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "success_rate": (passed/total_tests)*100,
                "total_execution_time": total_time,
                "average_test_time": avg_time
            },
            "results": self.results,
            "failed_tests": failed_tests,
            "timestamp": time.time()
        }
        
        with open(report_file, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"\n💾 Detailed report saved: {report_file}")
        
        # Overall assessment
        if passed == total_tests:
            print("\n🎉 ALL TESTS PASSED! RAGHeat system is fully functional!")
        elif passed >= total_tests * 0.8:
            print(f"\n✅ MOSTLY SUCCESSFUL! {failed} tests need attention.")
        elif passed >= total_tests * 0.5:
            print(f"\n⚠️ PARTIALLY SUCCESSFUL! {failed} tests require fixes.")
        else:
            print(f"\n❌ SIGNIFICANT ISSUES! {failed} tests failed - major fixes needed.")
        
        return detailed_report


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all RAGHeat feature tests")
    parser.add_argument("--parallel", action="store_true", 
                       help="Run tests in parallel (default: sequential)")
    parser.add_argument("--workers", type=int, default=3,
                       help="Number of parallel workers (default: 3)")
    
    args = parser.parse_args()
    
    print("🚀 STARTING ALL FEATURES TEST EXECUTION")
    print("🎯 Testing all 45 RAGHeat features independently on Chrome")
    print("⚡ Each test runs in isolated browser instance")
    
    runner = AllFeaturesTestRunner()
    
    if args.parallel:
        passed, failed, total = runner.run_all_tests_parallel(args.workers)
    else:
        passed, failed, total = runner.run_all_tests_sequential()
    
    # Generate comprehensive report
    report = runner.generate_summary_report(passed, failed, total)
    
    # Exit with appropriate code
    if failed == 0:
        print("\n✨ ALL FEATURES TEST EXECUTION COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(f"\n⚠️ TEST EXECUTION COMPLETED WITH {failed} FAILURES!")
        sys.exit(1)


if __name__ == "__main__":
    main()
