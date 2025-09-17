#!/usr/bin/env python3
"""
RAGHeat test execution script
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import requests

class RAGHeatTestRunner:
    """Test runner for RAGHeat application"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.reports_dir = self.base_dir / 'reports'
        self.reports_dir.mkdir(exist_ok=True)
        
        # Environment configuration
        self.frontend_url = os.getenv('RAGHEAT_FRONTEND_URL', 'http://localhost:3000')
        self.api_url = os.getenv('RAGHEAT_API_URL', 'http://localhost:8000')
    
    def check_application_running(self):
        """Check if RAGHeat application is running"""
        print("🔍 Checking if RAGHeat application is running...")
        
        # Check frontend
        try:
            response = requests.get(self.frontend_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Frontend accessible at {self.frontend_url}")
                frontend_ok = True
            else:
                print(f"⚠️ Frontend returned status {response.status_code} at {self.frontend_url}")
                frontend_ok = True  # Still accessible
        except requests.RequestException as e:
            print(f"❌ Frontend not accessible at {self.frontend_url}: {e}")
            frontend_ok = False
        
        # Check API
        api_ok = False
        try:
            # Try health endpoint
            response = requests.get(f"{self.api_url}/api/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API health check passed at {self.api_url}")
                api_ok = True
            else:
                print(f"⚠️ API health returned {response.status_code} at {self.api_url}")
        except requests.RequestException:
            # Try root endpoint (might serve frontend)
            try:
                response = requests.get(f"{self.api_url}/", timeout=5)
                if response.status_code == 200:
                    print(f"✅ API root accessible at {self.api_url}")
                    api_ok = True
            except requests.RequestException as e:
                print(f"❌ API not accessible at {self.api_url}: {e}")
        
        return frontend_ok or api_ok
    
    def run_smoke_tests(self):
        """Run smoke tests"""
        print("\n🔥 Running Smoke Tests (Critical Path)...")
        cmd = [
            'python', '-m', 'pytest', 'tests/smoke', 
            '-v', '-m', 'smoke',
            '--html=reports/smoke_report.html',
            '--self-contained-html',
            '--maxfail=3',
            '--tb=short'
        ]
        return self._run_command(cmd, "Smoke Tests")
    
    def run_sanity_tests(self):
        """Run sanity tests"""
        print("\n✅ Running Sanity Tests (Functional Validation)...")
        cmd = [
            'python', '-m', 'pytest', 'tests/sanity',
            '-v', '-m', 'sanity',
            '--html=reports/sanity_report.html',
            '--self-contained-html'
        ]
        return self._run_command(cmd, "Sanity Tests")
    
    def run_api_tests(self):
        """Run API tests"""
        print("\n🌐 Running API Tests...")
        cmd = [
            'python', '-m', 'pytest', 'tests/api',
            '-v', '-m', 'api',
            '--html=reports/api_report.html',
            '--self-contained-html'
        ]
        return self._run_command(cmd, "API Tests")
    
    def run_regression_tests(self):
        """Run regression tests"""
        print("\n🔄 Running Regression Tests (Comprehensive)...")
        cmd = [
            'python', '-m', 'pytest', 'tests/regression',
            '-v', '-m', 'regression',
            '--html=reports/regression_report.html',
            '--self-contained-html',
            '--cov=framework',
            '--cov-report=html:reports/coverage'
        ]
        return self._run_command(cmd, "Regression Tests")
    
    def run_integration_tests(self):
        """Run integration tests"""
        print("\n🔗 Running Integration Tests...")
        cmd = [
            'python', '-m', 'pytest', 'tests/integration',
            '-v', '-m', 'integration',
            '--html=reports/integration_report.html',
            '--self-contained-html'
        ]
        return self._run_command(cmd, "Integration Tests")
    
    def run_specific_test(self, test_path):
        """Run specific test file or test"""
        print(f"\n🎯 Running Specific Test: {test_path}")
        cmd = [
            'python', '-m', 'pytest', test_path,
            '-v', '-s',
            '--html=reports/specific_test_report.html',
            '--self-contained-html'
        ]
        return self._run_command(cmd, f"Specific Test: {test_path}")
    
    def _run_command(self, cmd, test_name):
        """Run command and capture output"""
        start_time = time.time()
        print(f"⚡ Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            duration = time.time() - start_time
            
            print(f"\n📊 {test_name} Results:")
            print(f"   ⏱️ Duration: {duration:.2f}s")
            print(f"   🔢 Exit Code: {result.returncode}")
            
            # Print output
            if result.stdout:
                print(f"\n📝 Output:")
                print(result.stdout)
            
            if result.stderr and result.returncode != 0:
                print(f"\n⚠️ Errors:")
                print(result.stderr)
            
            return result.returncode == 0, result
            
        except Exception as e:
            print(f"❌ Failed to run {test_name}: {e}")
            return False, None
    
    def run_all_tests(self):
        """Run all test suites"""
        print("\n" + "="*80)
        print("🚀 Starting RAGHeat Complete Test Suite")
        print("="*80)
        
        start_time = time.time()
        results = {}
        
        # Run smoke tests first (critical)
        success, result = self.run_smoke_tests()
        results['smoke'] = {'success': success, 'result': result}
        
        if not success:
            print("\n❌ Smoke tests failed. Critical issues detected!")
            print("🛑 Stopping test execution due to critical failures.")
            return False
        
        print("\n✅ Smoke tests passed! Continuing with additional tests...")
        
        # Run API tests
        success, result = self.run_api_tests()
        results['api'] = {'success': success, 'result': result}
        
        # Run sanity tests
        success, result = self.run_sanity_tests()
        results['sanity'] = {'success': success, 'result': result}
        
        # Run integration tests (optional)
        if os.path.exists('tests/integration'):
            success, result = self.run_integration_tests()
            results['integration'] = {'success': success, 'result': result}
        
        # Run regression tests (optional, can take long time)
        if os.path.exists('tests/regression'):
            success, result = self.run_regression_tests()
            results['regression'] = {'success': success, 'result': result}
        
        # Summary
        total_time = time.time() - start_time
        self._print_summary(results, total_time)
        
        # Send email report
        try:
            from framework.utilities.email_reporter import EmailReporter
            test_summary = {
                'total': len(results),
                'passed': sum(1 for r in results.values() if r['success']),
                'failed': sum(1 for r in results.values() if not r['success']),
                'skipped': 0
            }
            reporter = EmailReporter()
            reporter.send_test_report('semanticraj@gmail.com', test_summary)
        except Exception as e:
            print(f"📧 Email report failed: {e}")
        
        # Return overall success
        critical_success = results.get('smoke', {}).get('success', False)
        return critical_success
    
    def _print_summary(self, results, total_time):
        """Print test execution summary"""
        print("\n" + "="*80)
        print("📊 TEST EXECUTION SUMMARY")
        print("="*80)
        
        for suite, data in results.items():
            status = "✅ PASSED" if data['success'] else "❌ FAILED"
            print(f"{status} {suite.upper()} Tests")
        
        successful_suites = sum(1 for data in results.values() if data['success'])
        total_suites = len(results)
        
        print(f"\n📈 Overall Results: {successful_suites}/{total_suites} test suites passed")
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print(f"📁 Reports available in: {self.reports_dir}")
        
        # List available reports
        html_reports = list(self.reports_dir.glob("*.html"))
        if html_reports:
            print(f"\n📋 Generated Reports:")
            for report in html_reports:
                print(f"   📄 {report.name}")
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("📦 Installing dependencies...")
        
        try:
            result = subprocess.run([
                'pip', 'install', '-r', 'requirements.txt'
            ], capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def setup_environment(self):
        """Set up test environment"""
        print("🔧 Setting up test environment...")
        
        # Create directories
        (self.base_dir / 'logs').mkdir(exist_ok=True)
        (self.base_dir / 'screenshots').mkdir(exist_ok=True)
        (self.base_dir / 'reports').mkdir(exist_ok=True)
        
        print("✅ Test environment ready")
        return True

def main():
    parser = argparse.ArgumentParser(description='RAGHeat Test Runner')
    
    parser.add_argument('--suite', 
                       choices=['smoke', 'sanity', 'api', 'regression', 'integration', 'all'], 
                       default='smoke', 
                       help='Test suite to run (default: smoke)')
    
    parser.add_argument('--test', 
                       help='Run specific test file or test (e.g., tests/smoke/test_critical_paths.py)')
    
    parser.add_argument('--check-app', action='store_true', 
                       help='Check if application is running before tests')
    
    parser.add_argument('--install-deps', action='store_true',
                       help='Install dependencies before running tests')
    
    parser.add_argument('--headless', 
                       choices=['true', 'false'], 
                       default='true',
                       help='Run browser in headless mode (default: true)')
    
    parser.add_argument('--env',
                       choices=['local', 'staging', 'production'],
                       default='local',
                       help='Test environment (default: local)')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['HEADLESS'] = args.headless
    
    if args.env == 'production':
        os.environ['RAGHEAT_FRONTEND_URL'] = 'https://www.semanticdataservices.com'
        os.environ['RAGHEAT_API_URL'] = 'https://www.semanticdataservices.com'
    elif args.env == 'staging':
        os.environ['RAGHEAT_FRONTEND_URL'] = 'https://staging.semanticdataservices.com'
        os.environ['RAGHEAT_API_URL'] = 'https://staging.semanticdataservices.com'
    # local is default (localhost:3000, localhost:8000)
    
    runner = RAGHeatTestRunner()
    
    print(f"🎯 RAGHeat Testing Framework")
    print(f"   Environment: {args.env}")
    print(f"   Frontend URL: {runner.frontend_url}")
    print(f"   API URL: {runner.api_url}")
    print(f"   Headless mode: {args.headless}")
    
    # Install dependencies if requested
    if args.install_deps:
        if not runner.install_dependencies():
            sys.exit(1)
    
    # Set up environment
    if not runner.setup_environment():
        sys.exit(1)
    
    # Check if application is running
    if args.check_app:
        if not runner.check_application_running():
            print("❌ Application not running. Please start RAGHeat before running tests.")
            print("💡 Tip: Make sure the application is running on the configured URLs")
            sys.exit(1)
    
    success = True
    
    try:
        if args.test:
            # Run specific test
            success, _ = runner.run_specific_test(args.test)
        elif args.suite == 'smoke':
            success, _ = runner.run_smoke_tests()
        elif args.suite == 'sanity':
            success, _ = runner.run_sanity_tests()
        elif args.suite == 'api':
            success, _ = runner.run_api_tests()
        elif args.suite == 'regression':
            success, _ = runner.run_regression_tests()
        elif args.suite == 'integration':
            success, _ = runner.run_integration_tests()
        elif args.suite == 'all':
            success = runner.run_all_tests()
        
        if success:
            print("\n🎉 Test execution completed successfully!")
            exit_code = 0
        else:
            print("\n⚠️ Some tests failed. Check the reports for details.")
            exit_code = 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 Test execution interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"\n❌ Test execution failed with error: {e}")
        exit_code = 1
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()