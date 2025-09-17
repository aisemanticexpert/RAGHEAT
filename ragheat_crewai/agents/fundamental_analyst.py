"""
Fundamental Analyst Agent for RAGHeat CrewAI System
===================================================

This agent specializes in deep fundamental analysis of companies using SEC filings,
financial statements, and sector analysis to assess long-term value and financial health.
"""

from typing import Dict, Any, List
from .base_agent import RAGHeatBaseAgent
import logging

logger = logging.getLogger(__name__)

class FundamentalAnalystAgent(RAGHeatBaseAgent):
    """
    Fundamental Analyst Agent for comprehensive company analysis.
    
    Specializes in:
    - SEC filing analysis (10-K, 10-Q, 8-K)
    - Financial statement analysis
    - Financial ratio calculations
    - Sector and competitive analysis
    - Long-term value assessment
    """
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis on given stocks.
        
        Args:
            input_data: Dictionary containing:
                - stocks: List of stock tickers to analyze
                - time_horizon: Analysis time horizon (default: 1 year)
                - benchmark_sector: Sector for comparison
                - financial_data: Optional pre-loaded financial data
        
        Returns:
            Fundamental analysis results with recommendations
        """
        try:
            stocks = input_data.get("stocks", [])
            time_horizon = input_data.get("time_horizon", "1y")
            
            if not stocks:
                return {"error": "No stocks provided for analysis"}
            
            logger.info(f"Fundamental analysis starting for {len(stocks)} stocks")
            
            # Prepare analysis context
            analysis_context = {
                "stocks": stocks,
                "time_horizon": time_horizon,
                "analysis_type": "fundamental",
                "focus_areas": [
                    "financial_health",
                    "growth_prospects", 
                    "valuation_metrics",
                    "competitive_position",
                    "management_quality"
                ]
            }
            
            # Execute fundamental analysis task
            task_description = f"""
            Conduct comprehensive fundamental analysis for the following stocks: {', '.join(stocks)}
            
            For each stock, analyze:
            
            1. FINANCIAL HEALTH ASSESSMENT:
               - Balance sheet strength (debt-to-equity, current ratio, working capital)
               - Income statement analysis (revenue growth, margin trends, profitability)
               - Cash flow analysis (operating CF, free CF, CF coverage ratios)
               - Key financial ratios and their trends over time
            
            2. GROWTH PROSPECTS EVALUATION:
               - Revenue and earnings growth rates (historical and projected)
               - Market opportunity and addressable market size
               - Product pipeline and innovation capability
               - Management guidance and forward-looking statements
            
            3. VALUATION ANALYSIS:
               - P/E ratio analysis (current vs historical vs peers)
               - Price-to-Book, EV/EBITDA, and other valuation multiples
               - Discounted Cash Flow (DCF) model estimates
               - Relative valuation vs sector and market
            
            4. COMPETITIVE POSITIONING:
               - Market share and competitive advantages
               - Moat analysis (economic moats and competitive barriers)
               - Industry dynamics and competitive threats
               - Regulatory environment and compliance
            
            5. MANAGEMENT QUALITY:
               - Track record of management team
               - Capital allocation decisions
               - Strategic vision and execution capability
               - Corporate governance and transparency
            
            For each stock, provide:
            - Financial Health Score (1-10)
            - Growth Potential Score (1-10) 
            - Value Score (1-10)
            - Overall Fundamental Score (1-10)
            - BUY/HOLD/SELL recommendation with confidence level
            - Key catalysts and risks
            - Price target and investment thesis
            
            Time horizon: {time_horizon}
            Focus on long-term value creation and sustainable competitive advantages.
            """
            
            result = self.execute_task(task_description, analysis_context)
            
            # Post-process results to ensure structured output
            processed_result = self._structure_fundamental_analysis(result, stocks)
            
            logger.info(f"Fundamental analysis completed for {len(stocks)} stocks")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return {
                "error": str(e),
                "agent": "fundamental_analyst",
                "analysis_type": "fundamental"
            }
    
    def _structure_fundamental_analysis(self, raw_result: Dict[str, Any], stocks: List[str]) -> Dict[str, Any]:
        """Structure the fundamental analysis results."""
        
        structured_result = {
            "analysis_type": "fundamental",
            "agent": "fundamental_analyst",
            "timestamp": self._get_current_timestamp(),
            "stocks_analyzed": stocks,
            "overall_analysis": raw_result.get("result", ""),
            "stock_recommendations": {},
            "key_insights": [],
            "risk_factors": [],
            "sector_outlook": "",
            "investment_themes": []
        }
        
        # Extract structured data from result text if possible
        result_text = str(raw_result.get("result", ""))
        
        # Extract key insights
        insights = self._extract_insights_from_text(result_text)
        structured_result["key_insights"] = insights
        
        # Extract risk factors
        risks = self._extract_risks_from_text(result_text)
        structured_result["risk_factors"] = risks
        
        # For each stock, try to extract specific recommendations
        for stock in stocks:
            stock_info = self._extract_stock_specific_info(result_text, stock)
            if stock_info:
                structured_result["stock_recommendations"][stock] = stock_info
        
        return structured_result
    
    def _extract_insights_from_text(self, text: str) -> List[str]:
        """Extract key insights from analysis text."""
        insights = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['insight:', 'key finding:', 'important:', 'notable:']):
                clean_insight = line.split(':', 1)[-1].strip()
                if clean_insight and len(clean_insight) > 20:
                    insights.append(clean_insight)
        
        return insights[:5]  # Limit to top 5 insights
    
    def _extract_risks_from_text(self, text: str) -> List[str]:
        """Extract risk factors from analysis text."""
        risks = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['risk:', 'concern:', 'challenge:', 'threat:']):
                clean_risk = line.split(':', 1)[-1].strip()
                if clean_risk and len(clean_risk) > 15:
                    risks.append(clean_risk)
        
        return risks[:3]  # Limit to top 3 risks
    
    def _extract_stock_specific_info(self, text: str, stock: str) -> Dict[str, Any]:
        """Extract stock-specific information from analysis text."""
        # This is a simplified extraction - in practice, you'd use more sophisticated NLP
        stock_info = {
            "ticker": stock,
            "recommendation": "HOLD",  # Default
            "confidence": 0.6,
            "financial_health_score": 7.0,
            "growth_score": 7.0,
            "value_score": 7.0,
            "overall_score": 7.0,
            "key_strengths": [],
            "key_concerns": [],
            "price_target": None
        }
        
        # Look for stock-specific sections in the text
        text_lower = text.lower()
        stock_lower = stock.lower()
        
        # Extract recommendation
        if f"{stock_lower} buy" in text_lower or f"buy {stock_lower}" in text_lower:
            stock_info["recommendation"] = "BUY"
            stock_info["confidence"] = 0.8
        elif f"{stock_lower} sell" in text_lower or f"sell {stock_lower}" in text_lower:
            stock_info["recommendation"] = "SELL"
            stock_info["confidence"] = 0.8
        
        return stock_info
    
    def analyze_financial_ratios(self, stocks: List[str]) -> Dict[str, Any]:
        """
        Specialized method for detailed financial ratio analysis.
        
        Args:
            stocks: List of stock tickers
            
        Returns:
            Detailed financial ratio analysis
        """
        task_description = f"""
        Perform detailed financial ratio analysis for: {', '.join(stocks)}
        
        Calculate and analyze the following ratios with 3-year trends:
        
        LIQUIDITY RATIOS:
        - Current Ratio
        - Quick Ratio
        - Cash Ratio
        
        LEVERAGE RATIOS:
        - Debt-to-Equity
        - Debt-to-Assets
        - Interest Coverage Ratio
        
        EFFICIENCY RATIOS:
        - Asset Turnover
        - Inventory Turnover
        - Receivables Turnover
        
        PROFITABILITY RATIOS:
        - Gross Margin
        - Operating Margin
        - Net Margin
        - ROE, ROA, ROIC
        
        VALUATION RATIOS:
        - P/E Ratio
        - P/B Ratio
        - EV/EBITDA
        - PEG Ratio
        
        Compare each company's ratios to:
        1. Industry averages
        2. Historical performance
        3. Best-in-class competitors
        
        Identify ratio trends and their implications for financial health.
        """
        
        return self.execute_task(task_description, {"stocks": stocks, "analysis_type": "financial_ratios"})
    
    def analyze_competitive_position(self, stocks: List[str], sector: str) -> Dict[str, Any]:
        """
        Analyze competitive positioning within sector.
        
        Args:
            stocks: List of stock tickers
            sector: Sector for comparison
            
        Returns:
            Competitive position analysis
        """
        task_description = f"""
        Analyze competitive positioning for {', '.join(stocks)} within the {sector} sector.
        
        Evaluate:
        1. MARKET POSITION:
           - Market share and ranking
           - Geographic presence
           - Customer base diversity
        
        2. COMPETITIVE ADVANTAGES:
           - Economic moats (network effects, switching costs, etc.)
           - Unique assets or capabilities
           - Barriers to entry
        
        3. COMPETITIVE THREATS:
           - Direct competitors and their strategies
           - Potential disruptors
           - Substitute products/services
        
        4. STRATEGIC POSITIONING:
           - Value proposition differentiation
           - Cost structure advantages
           - Innovation capability
        
        Rank each company's competitive strength on a scale of 1-10.
        """
        
        context = {
            "stocks": stocks,
            "sector": sector,
            "analysis_type": "competitive_position"
        }
        
        return self.execute_task(task_description, context)