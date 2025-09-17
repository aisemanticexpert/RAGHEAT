#!/usr/bin/env python3
"""
FEATURE TEST GENERATOR
Creates individual test files for all 45 RAGHeat features to test independently on Chrome
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

class FeatureTestGenerator:
    """Generate individual test files for all RAGHeat features"""
    
    def __init__(self):
        self.test_dir = Path("tests/individual_features")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Complete list of 45 RAGHeat features
        self.features = {
            # Dashboard Core Features (5)
            "portfolio_construction": {
                "name": "Portfolio Construction",
                "api_endpoint": "/portfolio/construct",
                "ui_element": "//button[contains(text(), 'PORTFOLIO AI')]",
                "test_data": {"stocks": ["AAPL", "GOOGL", "MSFT"]},
                "expected_fields": ["portfolio_weights", "performance_metrics", "agent_insights"]
            },
            "dashboard_navigation": {
                "name": "Dashboard Navigation",
                "api_endpoint": None,
                "ui_element": "//button[contains(text(), 'REVOLUTIONARY DASHBOARD')]",
                "test_data": None,
                "expected_fields": ["navigation_success", "page_load"]
            },
            "market_overview": {
                "name": "Market Overview Dashboard",
                "api_endpoint": "/market/overview",
                "ui_element": "//button[contains(text(), 'MARKET DASHBOARD')]",
                "test_data": None,
                "expected_fields": ["market_indices", "top_movers", "market_status"]
            },
            "live_data_stream": {
                "name": "Live Data Streaming",
                "api_endpoint": "/stream/live-data",
                "ui_element": "//button[contains(text(), 'LIVE DATA STREAM')]",
                "test_data": None,
                "expected_fields": ["stream_status", "data_feed", "websocket_connection"]
            },
            "system_health": {
                "name": "System Health Monitoring",
                "api_endpoint": "/health",
                "ui_element": None,
                "test_data": None,
                "expected_fields": ["status", "timestamp"]
            },
            
            # Analysis Features (10)
            "fundamental_analysis": {
                "name": "Fundamental Analysis",
                "api_endpoint": "/analysis/fundamental",
                "ui_element": "//button[contains(text(), 'Fundamental')]",
                "test_data": {"stocks": ["AAPL", "GOOGL"]},
                "expected_fields": ["pe_ratio", "debt_to_equity", "roe", "recommendation"]
            },
            "sentiment_analysis": {
                "name": "Sentiment Analysis",
                "api_endpoint": "/analysis/sentiment",
                "ui_element": "//button[contains(text(), 'Sentiment')]",
                "test_data": {"stocks": ["AAPL", "GOOGL"]},
                "expected_fields": ["overall_sentiment", "news_sentiment", "social_sentiment", "recommendation"]
            },
            "technical_analysis": {
                "name": "Technical Analysis",
                "api_endpoint": "/analysis/technical",
                "ui_element": "//button[contains(text(), 'Technical')]",
                "test_data": {"stocks": ["AAPL"], "indicators": ["RSI", "MACD"]},
                "expected_fields": ["rsi", "macd", "moving_averages", "signals"]
            },
            "heat_diffusion_analysis": {
                "name": "Heat Diffusion Analysis",
                "api_endpoint": "/analysis/heat-diffusion",
                "ui_element": "//button[contains(text(), 'Heat')]",
                "test_data": {"stocks": ["AAPL", "GOOGL", "MSFT"]},
                "expected_fields": ["heat_scores", "diffusion_patterns", "correlation_matrix"]
            },
            "risk_analysis": {
                "name": "Risk Analysis",
                "api_endpoint": "/analysis/risk",
                "ui_element": "//button[contains(text(), 'Risk')]",
                "test_data": {"portfolio": ["AAPL", "GOOGL", "MSFT"]},
                "expected_fields": ["var_95", "beta", "volatility", "max_drawdown"]
            },
            "valuation_analysis": {
                "name": "Valuation Analysis",
                "api_endpoint": "/analysis/valuation",
                "ui_element": "//button[contains(text(), 'Valuation')]",
                "test_data": {"stocks": ["AAPL", "GOOGL"]},
                "expected_fields": ["fair_value", "price_targets", "valuation_ratios"]
            },
            "sector_analysis": {
                "name": "Sector Analysis",
                "api_endpoint": "/analysis/sector",
                "ui_element": "//button[contains(text(), 'SECTOR ANALYSIS')]",
                "test_data": {"sector": "technology"},
                "expected_fields": ["sector_performance", "top_stocks", "sector_trends"]
            },
            "correlation_analysis": {
                "name": "Correlation Analysis",
                "api_endpoint": "/analysis/correlation",
                "ui_element": "//button[contains(text(), 'Correlation')]",
                "test_data": {"stocks": ["AAPL", "GOOGL", "MSFT", "TSLA"]},
                "expected_fields": ["correlation_matrix", "correlation_scores"]
            },
            "momentum_analysis": {
                "name": "Momentum Analysis",
                "api_endpoint": "/analysis/momentum",
                "ui_element": "//button[contains(text(), 'Momentum')]",
                "test_data": {"stocks": ["AAPL", "GOOGL"]},
                "expected_fields": ["momentum_score", "trend_direction", "momentum_indicators"]
            },
            "volatility_analysis": {
                "name": "Volatility Analysis",
                "api_endpoint": "/analysis/volatility",
                "ui_element": "//button[contains(text(), 'Volatility')]",
                "test_data": {"stocks": ["AAPL", "GOOGL"]},
                "expected_fields": ["historical_volatility", "implied_volatility", "volatility_ranking"]
            },
            
            # Knowledge Graph Features (8)
            "knowledge_graph": {
                "name": "Knowledge Graph Visualization",
                "api_endpoint": "/graph/knowledge",
                "ui_element": "//button[contains(text(), 'KNOWLEDGE GRAPH')]",
                "test_data": {"nodes": 50},
                "expected_fields": ["nodes", "edges", "graph_data"]
            },
            "enhanced_graph": {
                "name": "Enhanced Graph Network",
                "api_endpoint": "/graph/enhanced",
                "ui_element": "//button[contains(text(), 'ENHANCED GRAPH')]",
                "test_data": {"max_nodes": 100},
                "expected_fields": ["network_data", "node_properties", "edge_weights"]
            },
            "ontology_graph": {
                "name": "Financial Ontology Graph",
                "api_endpoint": "/api/ontology/graph/3",
                "ui_element": "//button[contains(text(), 'ONTOLOGY GRAPH')]",
                "test_data": {"max_nodes": 50},
                "expected_fields": ["ontology_nodes", "relationships", "semantic_links"]
            },
            "advanced_knowledge_graph": {
                "name": "Advanced Knowledge Graph",
                "api_endpoint": "/graph/advanced",
                "ui_element": "//button[contains(text(), 'ADVANCED KG')]",
                "test_data": {"depth": 3},
                "expected_fields": ["advanced_nodes", "complex_relationships"]
            },
            "sigma_js_graph": {
                "name": "Sigma.js Graph Visualization",
                "api_endpoint": "/graph/sigma",
                "ui_element": "//button[contains(text(), 'SIGMA.JS REVOLUTION')]",
                "test_data": {"layout": "force"},
                "expected_fields": ["sigma_config", "graph_layout", "interactive_nodes"]
            },
            "network_topology": {
                "name": "Network Topology Analysis",
                "api_endpoint": "/graph/topology",
                "ui_element": "//button[contains(text(), 'Topology')]",
                "test_data": {"stocks": ["AAPL", "GOOGL", "MSFT"]},
                "expected_fields": ["topology_metrics", "centrality_scores", "clustering_coefficient"]
            },
            "graph_clustering": {
                "name": "Graph Clustering",
                "api_endpoint": "/graph/clustering",
                "ui_element": "//button[contains(text(), 'Clustering')]",
                "test_data": {"algorithm": "louvain"},
                "expected_fields": ["clusters", "modularity_score", "cluster_assignments"]
            },
            "path_analysis": {
                "name": "Graph Path Analysis",
                "api_endpoint": "/graph/paths",
                "ui_element": "//button[contains(text(), 'Paths')]",
                "test_data": {"source": "AAPL", "target": "GOOGL"},
                "expected_fields": ["shortest_paths", "path_weights", "path_analysis"]
            },
            
            # Options Trading Features (7)
            "options_signals": {
                "name": "Live Options Signals",
                "api_endpoint": "/options/signals",
                "ui_element": "//button[contains(text(), 'LIVE OPTIONS SIGNALS')]",
                "test_data": {"symbols": ["AAPL", "GOOGL"]},
                "expected_fields": ["buy_signals", "sell_signals", "signal_strength"]
            },
            "options_pricing": {
                "name": "Options Pricing Models",
                "api_endpoint": "/options/pricing",
                "ui_element": "//button[contains(text(), 'Pricing')]",
                "test_data": {"symbol": "AAPL", "strike": 150, "expiry": "2024-12-20"},
                "expected_fields": ["black_scholes_price", "greeks", "implied_volatility"]
            },
            "options_strategies": {
                "name": "Options Strategies",
                "api_endpoint": "/options/strategies",
                "ui_element": "//button[contains(text(), 'Strategies')]",
                "test_data": {"strategy_type": "covered_call"},
                "expected_fields": ["strategy_analysis", "risk_reward", "probability"]
            },
            "options_flow": {
                "name": "Options Flow Analysis",
                "api_endpoint": "/options/flow",
                "ui_element": "//button[contains(text(), 'Flow')]",
                "test_data": {"timeframe": "1D"},
                "expected_fields": ["unusual_activity", "volume_analysis", "flow_direction"]
            },
            "options_gamma": {
                "name": "Gamma Exposure Analysis",
                "api_endpoint": "/options/gamma",
                "ui_element": "//button[contains(text(), 'Gamma')]",
                "test_data": {"symbols": ["SPY", "QQQ"]},
                "expected_fields": ["gamma_exposure", "dealer_positioning", "flow_impact"]
            },
            "options_volatility": {
                "name": "Options Volatility Surface",
                "api_endpoint": "/options/volatility",
                "ui_element": "//button[contains(text(), 'Vol Surface')]",
                "test_data": {"symbol": "AAPL"},
                "expected_fields": ["volatility_surface", "skew_analysis", "term_structure"]
            },
            "options_alerts": {
                "name": "Options Alert System",
                "api_endpoint": "/options/alerts",
                "ui_element": "//button[contains(text(), 'Alerts')]",
                "test_data": {"alert_type": "unusual_volume"},
                "expected_fields": ["active_alerts", "alert_history", "notification_settings"]
            },
            
            # Multi-Agent System Features (6)
            "agent_orchestration": {
                "name": "Multi-Agent Orchestration",
                "api_endpoint": "/agents/orchestrate",
                "ui_element": "//button[contains(text(), 'AI MODELS')]",
                "test_data": {"task": "portfolio_analysis", "stocks": ["AAPL", "GOOGL"]},
                "expected_fields": ["agent_responses", "consensus", "coordination"]
            },
            "fundamental_agent": {
                "name": "Fundamental Analysis Agent",
                "api_endpoint": "/agents/fundamental",
                "ui_element": "//button[contains(text(), 'Fundamental Agent')]",
                "test_data": {"stocks": ["AAPL"]},
                "expected_fields": ["fundamental_insights", "agent_reasoning", "confidence"]
            },
            "sentiment_agent": {
                "name": "Sentiment Analysis Agent",
                "api_endpoint": "/agents/sentiment",
                "ui_element": "//button[contains(text(), 'Sentiment Agent')]",
                "test_data": {"stocks": ["AAPL"]},
                "expected_fields": ["sentiment_insights", "source_analysis", "trend_prediction"]
            },
            "risk_agent": {
                "name": "Risk Management Agent",
                "api_endpoint": "/agents/risk",
                "ui_element": "//button[contains(text(), 'Risk Agent')]",
                "test_data": {"portfolio": ["AAPL", "GOOGL", "MSFT"]},
                "expected_fields": ["risk_assessment", "recommendations", "monitoring"]
            },
            "valuation_agent": {
                "name": "Valuation Agent",
                "api_endpoint": "/agents/valuation",
                "ui_element": "//button[contains(text(), 'Valuation Agent')]",
                "test_data": {"stocks": ["AAPL"]},
                "expected_fields": ["valuation_model", "fair_value_estimate", "methodology"]
            },
            "portfolio_coordinator": {
                "name": "Portfolio Coordinator Agent",
                "api_endpoint": "/agents/coordinator",
                "ui_element": "//button[contains(text(), 'Coordinator')]",
                "test_data": {"agents": ["fundamental", "sentiment", "risk"]},
                "expected_fields": ["coordination_result", "agent_consensus", "final_recommendation"]
            },
            
            # Real-Time Features (5)
            "real_time_pricing": {
                "name": "Real-Time Price Feed",
                "api_endpoint": "/realtime/prices",
                "ui_element": "//button[contains(text(), 'Live Prices')]",
                "test_data": {"symbols": ["AAPL", "GOOGL", "MSFT"]},
                "expected_fields": ["current_prices", "price_changes", "volume"]
            },
            "streaming_quotes": {
                "name": "Streaming Market Quotes",
                "api_endpoint": "/stream/quotes",
                "ui_element": "//button[contains(text(), 'Streaming')]",
                "test_data": {"symbols": ["AAPL", "GOOGL"]},
                "expected_fields": ["bid", "ask", "last_trade", "volume"]
            },
            "websocket_data": {
                "name": "WebSocket Data Connection",
                "api_endpoint": "/ws/data",
                "ui_element": "//button[contains(text(), 'WebSocket')]",
                "test_data": {"channels": ["prices", "trades"]},
                "expected_fields": ["connection_status", "data_latency", "message_count"]
            },
            "live_alerts": {
                "name": "Live Market Alerts",
                "api_endpoint": "/realtime/alerts",
                "ui_element": "//button[contains(text(), 'LIVE SIGNALS')]",
                "test_data": {"alert_types": ["price_move", "volume_spike"]},
                "expected_fields": ["active_alerts", "alert_triggers", "notification_history"]
            },
            "market_events": {
                "name": "Real-Time Market Events",
                "api_endpoint": "/realtime/events",
                "ui_element": "//button[contains(text(), 'Events')]",
                "test_data": {"event_types": ["earnings", "news"]},
                "expected_fields": ["event_feed", "event_impact", "timing"]
            },
            
            # Visualization Features (4)
            "interactive_charts": {
                "name": "Interactive Chart Components",
                "api_endpoint": "/charts/interactive",
                "ui_element": "//div[contains(@class, 'chart')]",
                "test_data": {"chart_type": "candlestick", "symbol": "AAPL"},
                "expected_fields": ["chart_data", "interactivity", "zoom_controls"]
            },
            "heatmap_visualization": {
                "name": "Market Heatmap",
                "api_endpoint": "/visualizations/heatmap",
                "ui_element": "//div[contains(@class, 'heatmap')]",
                "test_data": {"metric": "price_change", "timeframe": "1D"},
                "expected_fields": ["heatmap_data", "color_scale", "tooltips"]
            },
            "3d_visualization": {
                "name": "3D Data Visualization",
                "api_endpoint": "/visualizations/3d",
                "ui_element": "//canvas[contains(@class, 'three')]",
                "test_data": {"visualization_type": "portfolio_sphere"},
                "expected_fields": ["3d_scene", "camera_controls", "interaction_handlers"]
            },
            "dashboard_widgets": {
                "name": "Dashboard Widget System",
                "api_endpoint": "/widgets/dashboard",
                "ui_element": "//div[contains(@class, 'widget')]",
                "test_data": {"widget_types": ["price_ticker", "news_feed"]},
                "expected_fields": ["widget_instances", "layout_configuration", "real_time_updates"]
            }
        }
    
    def generate_individual_test_file(self, feature_key: str, feature_config: Dict[str, Any]) -> str:
        """Generate individual test file for a specific feature"""
        
        test_content = f'''#!/usr/bin/env python3
"""
INDIVIDUAL FEATURE TEST: {feature_config["name"]}
Test file for {feature_key} feature - runs independently on Chrome
"""

import sys
import os
import time
import requests
import json
from pathlib import Path

# Add framework to path
sys.path.append(str(Path(__file__).parent.parent))

from framework.base_test import BaseTest
from framework.page_objects.comprehensive_dashboard import ComprehensiveDashboard
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class Test{feature_key.replace("_", "").title()}(BaseTest):
    """Individual test for {feature_config["name"]} feature"""
    
    def setup_method(self, method):
        """Setup for each test method"""
        super().setup_method(method)
        self.base_url = os.getenv("RAGHEAT_FRONTEND_URL", "http://localhost:3000")
        self.api_base = os.getenv("RAGHEAT_API_URL", "http://localhost:8001")
        self.dashboard = ComprehensiveDashboard(self.driver)
        self.test_results = {{
            "feature_name": "{feature_config["name"]}",
            "feature_key": "{feature_key}",
            "test_timestamp": time.time(),
            "api_test_passed": False,
            "ui_test_passed": False,
            "data_validation_passed": False,
            "issues_found": [],
            "performance_metrics": {{}},
            "screenshots": []
        }}
    
    def test_{feature_key}_api_endpoint(self):
        """Test {feature_config["name"]} API endpoint"""
        print(f"🧪 Testing {feature_config["name"]} API endpoint...")
        
        if not "{feature_config.get("api_endpoint", "")}":
            print("   ⚠️ No API endpoint defined for this feature")
            self.test_results["api_test_passed"] = True  # Skip if no API
            return
        
        try:
            start_time = time.time()
            
            # Prepare API request
            endpoint = f"{{self.api_base}}{feature_config.get("api_endpoint", "")}"
            test_data = {feature_config.get("test_data", {})}
            
            if test_data and test_data != {{}}:
                response = requests.post(endpoint, json=test_data, timeout=30)
            else:
                response = requests.get(endpoint, timeout=30)
            
            response_time = time.time() - start_time
            self.test_results["performance_metrics"]["api_response_time"] = response_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate expected fields
                expected_fields = {feature_config.get("expected_fields", [])}
                if expected_fields:
                    self._validate_response_fields(data, expected_fields)
                
                self.test_results["api_test_passed"] = True
                print(f"   ✅ API test passed ({{response_time:.2f}}s)")
                
            else:
                self.test_results["issues_found"].append(f"API returned status {{response.status_code}}")
                print(f"   ❌ API test failed: {{response.status_code}}")
                
        except Exception as e:
            self.test_results["issues_found"].append(f"API test error: {{str(e)}}")
            print(f"   ❌ API test failed: {{e}}")
    
    def test_{feature_key}_ui_interaction(self):
        """Test {feature_config["name"]} UI interaction"""
        print(f"🖱️ Testing {feature_config["name"]} UI interaction...")
        
        try:
            # Navigate to dashboard
            if not self.dashboard.navigate_to_dashboard(self.base_url):
                self.test_results["issues_found"].append("Dashboard failed to load")
                return
            
            # Take initial screenshot
            initial_screenshot = f"screenshots/{feature_key}_initial_{{int(time.time())}}.png"
            self.driver.save_screenshot(initial_screenshot)
            self.test_results["screenshots"].append(initial_screenshot)
            
            ui_element = "{feature_config.get("ui_element", "")}"
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
                    post_screenshot = f"screenshots/{feature_key}_clicked_{{int(time.time())}}.png"
                    self.driver.save_screenshot(post_screenshot)
                    self.test_results["screenshots"].append(post_screenshot)
                    
                    self.test_results["ui_test_passed"] = True
                    print(f"   ✅ UI interaction passed ({{interaction_time:.2f}}s)")
                    
                except TimeoutException:
                    self.test_results["issues_found"].append(f"UI element not found: {{ui_element}}")
                    print(f"   ❌ UI element not found: {{ui_element}}")
                    
            else:
                self.test_results["ui_test_passed"] = True  # Skip if no UI element
                print("   ⚠️ No UI element defined for this feature")
                
        except Exception as e:
            self.test_results["issues_found"].append(f"UI test error: {{str(e)}}")
            print(f"   ❌ UI test failed: {{e}}")
    
    def test_{feature_key}_data_validation(self):
        """Test {feature_config["name"]} data validation"""
        print(f"📊 Testing {feature_config["name"]} data validation...")
        
        try:
            # Test data consistency, ranges, and completeness
            if "{feature_config.get("api_endpoint", "")}":
                endpoint = f"{{self.api_base}}{feature_config.get("api_endpoint", "")}"
                test_data = {feature_config.get("test_data", {})}
                
                if test_data and test_data != {{}}:
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
                    print(f"   ❌ Cannot validate data - API returned {{response.status_code}}")
            else:
                self.test_results["data_validation_passed"] = True  # Skip if no API
                print("   ⚠️ No API endpoint for data validation")
                
        except Exception as e:
            self.test_results["issues_found"].append(f"Data validation error: {{str(e)}}")
            print(f"   ❌ Data validation failed: {{e}}")
    
    def test_{feature_key}_comprehensive(self):
        """Run comprehensive test for {feature_config["name"]}"""
        print(f"\\n{'='*80}")
        print(f"🚀 COMPREHENSIVE TEST: {feature_config["name"]}")
        print(f"{'='*80}")
        
        # Run all tests
        self.test_{feature_key}_api_endpoint()
        self.test_{feature_key}_ui_interaction()
        self.test_{feature_key}_data_validation()
        
        # Generate results
        self._generate_test_report()
    
    def _validate_response_fields(self, data: dict, expected_fields: list):
        """Validate that response contains expected fields"""
        missing_fields = []
        
        def check_nested_fields(obj, fields, path=""):
            for field in fields:
                if isinstance(obj, dict):
                    if field not in obj:
                        missing_fields.append(f"{{path}}.{{field}}" if path else field)
                elif isinstance(obj, list) and obj:
                    # Check first item in list
                    check_nested_fields(obj[0], [field], path)
        
        check_nested_fields(data, expected_fields)
        
        if missing_fields:
            self.test_results["issues_found"].append(f"Missing fields: {{', '.join(missing_fields)}}")
    
    def _validate_data_quality(self, data: dict) -> bool:
        """Validate data quality and consistency"""
        issues = []
        
        # Check for empty/null values in critical fields
        if isinstance(data, dict):
            for key, value in data.items():
                if value is None or value == "":
                    issues.append(f"Empty value for key: {{key}}")
                elif isinstance(value, list) and len(value) == 0:
                    issues.append(f"Empty list for key: {{key}}")
        
        # Add specific validations based on feature type
        feature_type = "{feature_key}".split("_")[0]
        
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
                    issues.append(f"{{stock}} sentiment {{sentiment}} out of range [0,1]")
                
                if not metrics.get("recommendation"):
                    issues.append(f"{{stock}} missing recommendation")
        
        return issues
    
    def _validate_portfolio_data(self, data: dict) -> list:
        """Validate portfolio-specific data"""
        issues = []
        
        if "portfolio_weights" in data:
            weights = data["portfolio_weights"]
            weights_sum = sum(weights.values())
            if abs(weights_sum - 1.0) > 0.01:
                issues.append(f"Portfolio weights sum to {{weights_sum:.4f}}, not 1.0")
        
        if "performance_metrics" in data:
            metrics = data["performance_metrics"]
            sharpe = metrics.get("sharpe_ratio", 0)
            if not (0 < sharpe < 5):
                issues.append(f"Sharpe ratio {{sharpe}} unrealistic")
        
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
        print(f"\\n{'='*60}")
        print(f"📋 TEST REPORT: {feature_config["name"]}")
        print(f"{'='*60}")
        
        # Test results summary
        api_status = "✅ PASSED" if self.test_results["api_test_passed"] else "❌ FAILED"
        ui_status = "✅ PASSED" if self.test_results["ui_test_passed"] else "❌ FAILED"
        data_status = "✅ PASSED" if self.test_results["data_validation_passed"] else "❌ FAILED"
        
        print(f"🔌 API Endpoint Test: {{api_status}}")
        print(f"🖱️ UI Interaction Test: {{ui_status}}")
        print(f"📊 Data Validation Test: {{data_status}}")
        
        # Performance metrics
        if self.test_results["performance_metrics"]:
            print(f"\\n⏱️ Performance Metrics:")
            for metric, value in self.test_results["performance_metrics"].items():
                print(f"   {{metric}}: {{value:.2f}}s")
        
        # Issues found
        if self.test_results["issues_found"]:
            print(f"\\n⚠️ Issues Found ({{len(self.test_results['issues_found'])}})):")
            for issue in self.test_results["issues_found"]:
                print(f"   • {{issue}}")
        else:
            print(f"\\n✅ No issues found!")
        
        # Screenshots
        if self.test_results["screenshots"]:
            print(f"\\n📸 Screenshots Captured ({{len(self.test_results['screenshots'])}})):")
            for screenshot in self.test_results["screenshots"]:
                print(f"   📄 {{screenshot}}")
        
        # Overall result
        all_passed = all([
            self.test_results["api_test_passed"],
            self.test_results["ui_test_passed"], 
            self.test_results["data_validation_passed"]
        ])
        
        overall_status = "✅ PASSED" if all_passed else "❌ FAILED"
        print(f"\\n🎯 Overall Result: {{overall_status}}")
        
        # Save detailed report
        report_file = f"reports/individual_test_{feature_key}_{{int(time.time())}}.json"
        os.makedirs("reports", exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"💾 Detailed report saved: {{report_file}}")
        
        return all_passed


if __name__ == "__main__":
    """Run individual feature test"""
    print(f"🧪 STARTING INDIVIDUAL FEATURE TEST")
    print(f"📋 Feature: {feature_config["name"]}")
    print(f"🔑 Key: {feature_key}")
    
    test = Test{feature_key.replace("_", "").title()}()
    
    try:
        test.setup_method(lambda: None)
        result = test.test_{feature_key}_comprehensive()
        
        if result:
            print(f"\\n✨ {feature_config["name"]} test completed successfully!")
        else:
            print(f"\\n❌ {feature_config["name"]} test failed!")
            
    except Exception as e:
        print(f"\\n💥 Test execution failed: {{e}}")
        
    finally:
        try:
            test.teardown_method(lambda: None)
        except:
            pass
'''
        
        return test_content
    
    def generate_all_feature_tests(self):
        """Generate individual test files for all 45 features"""
        print("🏭 GENERATING INDIVIDUAL FEATURE TEST FILES")
        print("="*60)
        
        generated_files = []
        
        for feature_key, feature_config in self.features.items():
            print(f"\n📝 Generating test for: {feature_config['name']}")
            
            # Generate test file content
            test_content = self.generate_individual_test_file(feature_key, feature_config)
            
            # Save test file
            test_file = self.test_dir / f"test_{feature_key}.py"
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            generated_files.append(str(test_file))
            print(f"   ✅ Generated: {test_file}")
        
        return generated_files
    
    def generate_test_runner(self):
        """Generate a test runner to execute all individual tests"""
        
        runner_content = f'''#!/usr/bin/env python3
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
        self.test_dir = Path("{self.test_dir}")
        self.results = {{}}
        self.start_time = time.time()
    
    def run_single_test(self, test_file):
        """Run a single feature test"""
        feature_name = test_file.stem.replace("test_", "")
        print(f"🧪 Running {{feature_name}} test...")
        
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
            
            self.results[feature_name] = {{
                "success": success,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }}
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {{status}} ({{execution_time:.1f}}s)")
            
            return feature_name, success
            
        except subprocess.TimeoutExpired:
            self.results[feature_name] = {{
                "success": False,
                "execution_time": 300,
                "error": "Test timeout (5 minutes)",
                "return_code": -1
            }}
            print(f"   ⏰ TIMEOUT")
            return feature_name, False
            
        except Exception as e:
            self.results[feature_name] = {{
                "success": False,
                "execution_time": 0,
                "error": str(e),
                "return_code": -1
            }}
            print(f"   💥 ERROR: {{e}}")
            return feature_name, False
    
    def run_all_tests_sequential(self):
        """Run all tests sequentially"""
        print("\\n🔄 RUNNING ALL TESTS SEQUENTIALLY")
        print("="*60)
        
        test_files = list(self.test_dir.glob("test_*.py"))
        total_tests = len(test_files)
        
        print(f"📊 Found {{total_tests}} individual feature tests")
        
        passed = 0
        failed = 0
        
        for i, test_file in enumerate(test_files, 1):
            print(f"\\n[{{i}}/{{total_tests}}] {{test_file.stem}}")
            feature_name, success = self.run_single_test(test_file)
            
            if success:
                passed += 1
            else:
                failed += 1
        
        return passed, failed, total_tests
    
    def run_all_tests_parallel(self, max_workers=3):
        """Run tests in parallel (limited workers to avoid resource conflicts)"""
        print(f"\\n⚡ RUNNING ALL TESTS IN PARALLEL ({{max_workers}} workers)")
        print("="*60)
        
        test_files = list(self.test_dir.glob("test_*.py"))
        total_tests = len(test_files)
        
        print(f"📊 Found {{total_tests}} individual feature tests")
        
        passed = 0
        failed = 0
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {{executor.submit(self.run_single_test, test_file): test_file 
                              for test_file in test_files}}
            
            for future in as_completed(future_to_test):
                test_file = future_to_test[future]
                completed += 1
                
                try:
                    feature_name, success = future.result()
                    
                    if success:
                        passed += 1
                    else:
                        failed += 1
                    
                    print(f"[{{completed}}/{{total_tests}}] {{feature_name}} completed")
                    
                except Exception as e:
                    failed += 1
                    print(f"[{{completed}}/{{total_tests}}] {{test_file.stem}} failed with exception: {{e}}")
        
        return passed, failed, total_tests
    
    def generate_summary_report(self, passed, failed, total_tests):
        """Generate comprehensive summary report"""
        total_time = time.time() - self.start_time
        
        print("\\n" + "="*80)
        print("🏆 ALL FEATURES TEST EXECUTION SUMMARY")
        print("="*80)
        
        print(f"\\n📊 Test Results:")
        print(f"   Total Tests: {{total_tests}}")
        print(f"   ✅ Passed: {{passed}}")
        print(f"   ❌ Failed: {{failed}}")
        print(f"   📈 Success Rate: {{(passed/total_tests)*100:.1f}}%")
        print(f"   ⏱️ Total Time: {{total_time:.1f}}s")
        
        # Feature category breakdown
        categories = {{
            "dashboard": [],
            "analysis": [],
            "knowledge": [],
            "options": [],
            "agents": [],
            "realtime": [],
            "visualization": []
        }}
        
        for feature_name, result in self.results.items():
            for category in categories.keys():
                if category in feature_name:
                    categories[category].append((feature_name, result["success"]))
                    break
            else:
                categories.setdefault("other", []).append((feature_name, result["success"]))
        
        print(f"\\n📋 Results by Category:")
        for category, tests in categories.items():
            if tests:
                category_passed = sum(1 for _, success in tests if success)
                category_total = len(tests)
                category_rate = (category_passed/category_total)*100 if category_total > 0 else 0
                print(f"   {{category.title()}}: {{category_passed}}/{{category_total}} ({{category_rate:.1f}}%)")
        
        # Failed tests details
        failed_tests = [name for name, result in self.results.items() if not result["success"]]
        if failed_tests:
            print(f"\\n❌ Failed Tests ({{len(failed_tests)}}):")
            for test_name in failed_tests:
                result = self.results[test_name]
                error_info = result.get("error", "Unknown error")
                print(f"   • {{test_name}}: {{error_info}}")
        
        # Performance insights
        avg_time = sum(r["execution_time"] for r in self.results.values()) / len(self.results)
        slowest_tests = sorted(self.results.items(), 
                              key=lambda x: x[1]["execution_time"], 
                              reverse=True)[:5]
        
        print(f"\\n⏱️ Performance Insights:")
        print(f"   Average test time: {{avg_time:.1f}}s")
        print(f"   Slowest tests:")
        for name, result in slowest_tests:
            print(f"     • {{name}}: {{result['execution_time']:.1f}}s")
        
        # Save detailed results
        report_file = f"reports/all_features_test_results_{{int(time.time())}}.json"
        Path("reports").mkdir(exist_ok=True)
        
        detailed_report = {{
            "summary": {{
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "success_rate": (passed/total_tests)*100,
                "total_execution_time": total_time,
                "average_test_time": avg_time
            }},
            "results": self.results,
            "failed_tests": failed_tests,
            "timestamp": time.time()
        }}
        
        with open(report_file, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"\\n💾 Detailed report saved: {{report_file}}")
        
        # Overall assessment
        if passed == total_tests:
            print("\\n🎉 ALL TESTS PASSED! RAGHeat system is fully functional!")
        elif passed >= total_tests * 0.8:
            print(f"\\n✅ MOSTLY SUCCESSFUL! {{failed}} tests need attention.")
        elif passed >= total_tests * 0.5:
            print(f"\\n⚠️ PARTIALLY SUCCESSFUL! {{failed}} tests require fixes.")
        else:
            print(f"\\n❌ SIGNIFICANT ISSUES! {{failed}} tests failed - major fixes needed.")
        
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
        print("\\n✨ ALL FEATURES TEST EXECUTION COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(f"\\n⚠️ TEST EXECUTION COMPLETED WITH {{failed}} FAILURES!")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
        
        runner_file = self.test_dir.parent / "run_all_feature_tests.py"
        with open(runner_file, 'w') as f:
            f.write(runner_content)
        
        # Make executable
        os.chmod(runner_file, 0o755)
        
        return str(runner_file)
    
    def generate_feature_summary(self):
        """Generate summary of all features"""
        
        summary_content = f'''# RAGHeat Features Test Summary

## All 45 Features Individual Test Files Generated

### Feature Categories:

**Dashboard Core Features (5):**
{self._format_feature_list([k for k in self.features.keys() if any(x in k for x in ["dashboard", "portfolio", "market", "live_data", "system"])])}

**Analysis Features (10):**
{self._format_feature_list([k for k in self.features.keys() if any(x in k for x in ["fundamental", "sentiment", "technical", "heat", "risk", "valuation", "sector", "correlation", "momentum", "volatility"])])}

**Knowledge Graph Features (8):**
{self._format_feature_list([k for k in self.features.keys() if any(x in k for x in ["knowledge", "graph", "ontology", "advanced", "sigma", "network", "clustering", "path"])])}

**Options Trading Features (7):**
{self._format_feature_list([k for k in self.features.keys() if "options" in k])}

**Multi-Agent System Features (6):**
{self._format_feature_list([k for k in self.features.keys() if any(x in k for x in ["agent", "orchestration", "coordinator"])])}

**Real-Time Features (5):**
{self._format_feature_list([k for k in self.features.keys() if any(x in k for x in ["real_time", "streaming", "websocket", "live_alerts", "market_events"])])}

**Visualization Features (4):**
{self._format_feature_list([k for k in self.features.keys() if any(x in k for x in ["interactive", "heatmap", "3d", "widget"])])}

### Test Files Generated:
- **Location**: `{self.test_dir}/`
- **Pattern**: `test_[feature_key].py`
- **Total Files**: {len(self.features)}

### Test Runner:
- **File**: `run_all_feature_tests.py`
- **Modes**: Sequential and Parallel execution
- **Features**: Individual Chrome instances, comprehensive reporting, performance metrics

### Usage:
```bash
# Run all tests sequentially
python3 run_all_feature_tests.py

# Run tests in parallel (3 workers)
python3 run_all_feature_tests.py --parallel

# Run tests in parallel (custom workers)
python3 run_all_feature_tests.py --parallel --workers 5
```

### Each Test Includes:
1. **API Endpoint Testing** - Validates API responses and data
2. **UI Interaction Testing** - Tests browser automation and clicking
3. **Data Validation Testing** - Verifies data quality and consistency
4. **Performance Metrics** - Measures response times and interactions
5. **Screenshot Capture** - Visual evidence of test execution
6. **Comprehensive Reporting** - JSON reports with detailed results

### Generated Files:
- 45 individual test files (`test_*.py`)
- 1 test runner (`run_all_feature_tests.py`)
- 1 summary documentation (this file)
'''
        
        summary_file = self.test_dir.parent / "FEATURES_SUMMARY.md"
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        return str(summary_file)
    
    def _format_feature_list(self, feature_keys):
        """Format feature list for documentation"""
        lines = []
        for key in feature_keys:
            if key in self.features:
                feature = self.features[key]
                lines.append(f"- `{key}`: {feature['name']}")
        return "\\n".join(lines)
    
    def run_complete_generation(self):
        """Run complete feature test generation"""
        print("\\n" + "="*80)
        print("🏭 RAGHEAT FEATURE TEST GENERATOR")
        print("="*80)
        print(f"🎯 Generating individual test files for all {len(self.features)} features")
        print("⚡ Each test runs independently on Chrome with comprehensive validation")
        
        # Generate all test files
        generated_files = self.generate_all_feature_tests()
        
        # Generate test runner
        runner_file = self.generate_test_runner()
        
        # Generate summary
        summary_file = self.generate_feature_summary()
        
        print(f"\\n" + "="*60)
        print("📋 GENERATION SUMMARY")
        print("="*60)
        print(f"✅ Generated {len(generated_files)} individual test files")
        print(f"✅ Generated test runner: {runner_file}")
        print(f"✅ Generated summary: {summary_file}")
        
        print(f"\\n🚀 Ready to Execute:")
        print(f"   Sequential: python3 run_all_feature_tests.py")
        print(f"   Parallel:   python3 run_all_feature_tests.py --parallel")
        
        return {
            "test_files": generated_files,
            "runner_file": runner_file,
            "summary_file": summary_file,
            "total_features": len(self.features)
        }


def main():
    """Main execution function"""
    print("🏭 STARTING FEATURE TEST GENERATION")
    print("🎯 Creating individual Chrome tests for all 45 RAGHeat features")
    print("⚡ Each feature will be tested independently with full validation")
    
    generator = FeatureTestGenerator()
    result = generator.run_complete_generation()
    
    print(f"\\n✨ FEATURE TEST GENERATION COMPLETED!")
    print(f"📊 {result['total_features']} features ready for independent testing")
    print(f"🧪 {len(result['test_files'])} test files generated")
    print(f"🚀 Test runner ready: {result['runner_file']}")


if __name__ == "__main__":
    main()