#!/usr/bin/env python3
"""
RAGHeat API testing client
"""

import json
import time
import requests
from typing import Dict, List, Any, Optional

class RAGHeatAPIClient:
    """API client for RAGHeat testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RAGHeat-TestFramework/1.0'
        })
        self.response_times = []
    
    def _make_request(self, method: str, endpoint: str, **kwargs):
        """Make HTTP request and track performance"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            response = self.session.request(method, url, **kwargs)
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.response_times.append(response_time)
            
            # Log request details
            print(f"🌐 {method} {endpoint} - {response.status_code} - {response_time:.2f}ms")
            
            return response
            
        except requests.RequestException as e:
            print(f"❌ Request failed: {method} {endpoint} - {str(e)}")
            raise
    
    def health_check(self) -> requests.Response:
        """Test health endpoint"""
        return self._make_request('GET', '/api/health')
    
    def system_status(self) -> requests.Response:
        """Get system status"""
        return self._make_request('GET', '/api/system/status')
    
    def construct_portfolio(self, stocks: List[str], market_data: Optional[Dict] = None) -> requests.Response:
        """Construct portfolio via API"""
        payload = {
            "stocks": stocks,
            "market_data": market_data or {}
        }
        return self._make_request('POST', '/api/portfolio/construct', json=payload)
    
    def fundamental_analysis(self, stocks: List[str], parameters: Optional[Dict] = None) -> requests.Response:
        """Request fundamental analysis"""
        payload = {
            "stocks": stocks,
            "analysis_parameters": parameters or {}
        }
        return self._make_request('POST', '/api/analysis/fundamental', json=payload)
    
    def sentiment_analysis(self, stocks: List[str], parameters: Optional[Dict] = None) -> requests.Response:
        """Request sentiment analysis"""
        payload = {
            "stocks": stocks,
            "analysis_parameters": parameters or {}
        }
        return self._make_request('POST', '/api/analysis/sentiment', json=payload)
    
    def technical_analysis(self, stocks: List[str], parameters: Optional[Dict] = None) -> requests.Response:
        """Request technical analysis"""
        payload = {
            "stocks": stocks,
            "analysis_parameters": parameters or {}
        }
        return self._make_request('POST', '/api/analysis/technical', json=payload)
    
    def heat_diffusion_analysis(self, stocks: List[str], parameters: Optional[Dict] = None) -> requests.Response:
        """Request heat diffusion analysis"""
        payload = {
            "stocks": stocks,
            "analysis_parameters": parameters or {}
        }
        return self._make_request('POST', '/api/analysis/heat-diffusion', json=payload)
    
    def get_analysis_tools(self) -> requests.Response:
        """Get available analysis tools"""
        return self._make_request('GET', '/api/analysis/tools')
    
    def get_root_page(self) -> requests.Response:
        """Get root page (serves React frontend)"""
        return self._make_request('GET', '/')
    
    def get_average_response_time(self) -> float:
        """Get average response time"""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0
    
    def get_max_response_time(self) -> float:
        """Get maximum response time"""
        return max(self.response_times) if self.response_times else 0
    
    def validate_json_response(self, response: requests.Response, expected_keys: List[str] = None) -> bool:
        """Validate JSON response structure"""
        try:
            data = response.json()
            if expected_keys:
                return all(key in data for key in expected_keys)
            return True
        except json.JSONDecodeError:
            return False
    
    def validate_portfolio_response(self, response: requests.Response) -> Dict[str, bool]:
        """Validate portfolio construction response"""
        validation_results = {
            'status_code_ok': response.status_code == 200,
            'is_json': False,
            'has_portfolio_weights': False,
            'has_risk_metrics': False,
            'has_agent_insights': False,
            'weights_sum_valid': False
        }
        
        try:
            data = response.json()
            validation_results['is_json'] = True
            
            # Check required keys
            validation_results['has_portfolio_weights'] = 'portfolio_weights' in data
            validation_results['has_risk_metrics'] = 'risk_metrics' in data
            validation_results['has_agent_insights'] = 'agent_insights' in data
            
            # Validate portfolio weights
            if 'portfolio_weights' in data:
                weights = data['portfolio_weights']
                if isinstance(weights, dict) and weights:
                    total_weight = sum(weights.values())
                    validation_results['weights_sum_valid'] = 0.95 <= total_weight <= 1.05
                    
        except json.JSONDecodeError:
            pass
        
        return validation_results
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary"""
        if not self.response_times:
            return {}
        
        return {
            'total_requests': len(self.response_times),
            'average_response_time_ms': self.get_average_response_time(),
            'max_response_time_ms': self.get_max_response_time(),
            'min_response_time_ms': min(self.response_times)
        }