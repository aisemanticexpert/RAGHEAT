#!/usr/bin/env python3
"""
ACTUAL CLICKING DEMO
Real XPath-based clicking with corrected selectors
This demonstrates actual clicking on all buttons and tabs
"""

import os
import sys
import time
sys.path.append('.')

from framework.base_test import BaseTest
from framework.page_objects.comprehensive_dashboard import ComprehensiveDashboard


class ActualClickingDemo(BaseTest):
    """Demonstration of actual clicking with corrected XPath selectors"""
    
    def setup_method(self, method):
        """Setup for each test"""
        super().setup_method(method)
        self.base_url = os.getenv("RAGHEAT_FRONTEND_URL", "http://localhost:3000")
        self.dashboard = ComprehensiveDashboard(self.driver)
        print(f"🚀 Starting clicking demo: {method.__name__}")
    
    def test_all_dashboard_buttons_clicking(self):
        """Test clicking ALL dashboard buttons with visible interaction"""
        print("🔥 DEMO: All Dashboard Buttons Clicking")
        
        # Navigate to dashboard
        assert self.dashboard.navigate_to_dashboard(self.base_url), "Dashboard failed to load"
        
        # Test ALL dashboard buttons
        button_results = self.dashboard.test_all_dashboard_buttons()
        
        # Show results
        print("\\n📊 Dashboard Button Clicking Results:")
        clicked_count = 0
        total_count = len(button_results)
        
        for button_name, success in button_results.items():
            status = "✅ CLICKED" if success else "❌ FAILED"
            print(f"   {button_name}: {status}")
            if success:
                clicked_count += 1
        
        success_rate = (clicked_count / total_count) * 100
        print(f"\\n🎯 Button Clicking Success Rate: {success_rate:.1f}% ({clicked_count}/{total_count})")
        
        return button_results
    
    def test_all_tab_navigation_clicking(self):
        """Test clicking ALL navigation tabs with visible interaction"""
        print("🔥 DEMO: All Navigation Tabs Clicking")
        
        # Navigate to dashboard
        assert self.dashboard.navigate_to_dashboard(self.base_url), "Dashboard failed to load"
        
        # Test ALL navigation tabs
        tab_results = self.dashboard.test_all_tab_navigation()
        
        # Show results
        print("\\n📊 Tab Navigation Clicking Results:")
        clicked_count = 0
        total_count = len(tab_results)
        
        for tab_name, success in tab_results.items():
            status = "✅ CLICKED" if success else "❌ FAILED"
            print(f"   {tab_name}: {status}")
            if success:
                clicked_count += 1
        
        success_rate = (clicked_count / total_count) * 100
        print(f"\\n🎯 Tab Clicking Success Rate: {success_rate:.1f}% ({clicked_count}/{total_count})")
        
        return tab_results
    
    def test_combined_clicking_workflow(self):
        """Test combined clicking workflow - buttons + tabs"""
        print("🔥 DEMO: Combined Clicking Workflow (Buttons + Tabs)")
        
        # Navigate to dashboard
        assert self.dashboard.navigate_to_dashboard(self.base_url), "Dashboard failed to load"
        
        # First test dashboard buttons
        print("\\n1️⃣ Testing Dashboard Buttons:")
        button_results = self.dashboard.test_all_dashboard_buttons()
        
        # Then test navigation tabs
        print("\\n2️⃣ Testing Navigation Tabs:")
        tab_results = self.dashboard.test_all_tab_navigation()
        
        # Combined results
        total_interactions = len(button_results) + len(tab_results)
        successful_interactions = sum(button_results.values()) + sum(tab_results.values())
        
        overall_success_rate = (successful_interactions / total_interactions) * 100
        
        print("\\n" + "="*80)
        print("🏆 COMBINED CLICKING WORKFLOW RESULTS")
        print("="*80)
        print(f"🖱️ Dashboard Buttons: {sum(button_results.values())}/{len(button_results)} clicked")
        print(f"📋 Navigation Tabs: {sum(tab_results.values())}/{len(tab_results)} clicked")
        print(f"🎯 Overall Success Rate: {overall_success_rate:.1f}% ({successful_interactions}/{total_interactions})")
        print("="*80)
        
        return {
            'button_results': button_results,
            'tab_results': tab_results,
            'overall_success_rate': overall_success_rate,
            'total_interactions': total_interactions,
            'successful_interactions': successful_interactions
        }


def run_actual_clicking_demo():
    """Run the actual clicking demonstration"""
    print("\\n" + "="*100)
    print("🎬 STARTING ACTUAL CLICKING DEMONSTRATION")
    print("👁️ You will see real browser automation with visible clicking!")
    print("🖱️ Watch as the system clicks on buttons and tabs with highlighting!")
    print("📸 Screenshots will be captured for each successful click!")
    print("="*100)
    
    # Set environment for visible testing
    os.environ['HEADLESS'] = 'false'
    os.environ['RAGHEAT_API_URL'] = 'http://localhost:8001'
    os.environ['RAGHEAT_FRONTEND_URL'] = 'http://localhost:3000'
    
    # Create demo instance
    demo = ActualClickingDemo()
    
    try:
        print("\\n🔧 Setting up browser for demonstration...")
        demo.setup_method(lambda: None)
        
        # Run all clicking demonstrations
        print("\\n🎯 Running Combined Clicking Workflow...")
        results = demo.test_combined_clicking_workflow()
        
        print("\\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print(f"✅ {results['successful_interactions']}/{results['total_interactions']} elements clicked successfully")
        print(f"📈 Success Rate: {results['overall_success_rate']:.1f}%")
        
        # Keep browser open for final observation
        print("\\n👀 Keeping browser open for 10 seconds for final observation...")
        for i in range(10, 0, -1):
            print(f"\\r   ⏰ Closing in {i} seconds...", end="", flush=True)
            time.sleep(1)
        print("\\n")
        
        return results
        
    except Exception as e:
        print(f"\\n❌ Demo failed with error: {e}")
        return None
    
    finally:
        # Clean up
        try:
            demo.teardown_method(lambda: None)
            print("🧹 Browser closed successfully")
        except:
            pass


if __name__ == "__main__":
    results = run_actual_clicking_demo()
    
    if results:
        print("\\n✨ ACTUAL CLICKING DEMONSTRATION COMPLETED!")
        print(f"🎯 Final Results: {results['overall_success_rate']:.1f}% success rate")
    else:
        print("\\n❌ Demonstration failed")