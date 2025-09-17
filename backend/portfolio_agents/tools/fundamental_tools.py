"""
Fundamental Analysis Tools for Portfolio Construction
==================================================

Tools for analyzing company fundamentals, SEC filings, and financial health.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from crewai_tools import BaseTool
from pydantic import BaseModel, Field
from loguru import logger
import time
from datetime import datetime, timedelta

class FundamentalReportPullTool(BaseTool):
    """Tool for pulling fundamental reports and SEC filings."""
    
    name: str = "fundamental_report_pull"
    description: str = "Pull fundamental reports, SEC filings (10-K, 10-Q), and company information for analysis"
    
    def _run(self, symbol: str) -> Dict[str, Any]:
        """Pull fundamental data for a given stock symbol."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic company info
            info = ticker.info
            
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # Get key statistics
            key_stats = self._extract_key_statistics(info)
            
            # Get analyst recommendations
            recommendations = ticker.recommendations
            
            return {
                'symbol': symbol,
                'company_info': {
                    'name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'full_time_employees': info.get('fullTimeEmployees', 0),
                    'description': info.get('longBusinessSummary', '')
                },
                'key_statistics': key_stats,
                'financial_statements': {
                    'income_statement': financials.to_dict() if not financials.empty else {},
                    'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                    'cash_flow': cash_flow.to_dict() if not cash_flow.empty else {}
                },
                'analyst_recommendations': recommendations.to_dict() if recommendations is not None else {},
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error pulling fundamental data for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _extract_key_statistics(self, info: Dict) -> Dict[str, Any]:
        """Extract key financial statistics."""
        return {
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'enterprise_to_revenue': info.get('enterpriseToRevenue'),
            'enterprise_to_ebitda': info.get('enterpriseToEbitda'),
            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),
            'beta': info.get('beta')
        }

class FinancialReportRAGTool(BaseTool):
    """Tool for RAG-based analysis of financial reports using LLM."""
    
    name: str = "financial_report_rag"
    description: str = "Perform RAG-based analysis of financial reports and SEC filings using LLM"
    
    def _run(self, symbol: str, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial reports using RAG approach."""
        try:
            from anthropic import Anthropic
            from ..config.settings import settings
            
            client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            
            # Prepare financial data for analysis
            analysis_prompt = self._create_analysis_prompt(symbol, report_data)
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": analysis_prompt
                }]
            )
            
            analysis = response.content[0].text
            
            # Extract structured insights
            insights = self._extract_structured_insights(analysis)
            
            return {
                'symbol': symbol,
                'rag_analysis': analysis,
                'structured_insights': insights,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in RAG analysis for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _create_analysis_prompt(self, symbol: str, report_data: Dict) -> str:
        """Create analysis prompt for LLM."""
        return f"""
        Analyze the financial health and investment prospects of {symbol} based on the following data:
        
        Company Information:
        {report_data.get('company_info', {})}
        
        Key Financial Metrics:
        {report_data.get('key_statistics', {})}
        
        Please provide:
        1. Financial Health Assessment (1-10 scale)
        2. Growth Prospects Analysis
        3. Key Strengths and Weaknesses
        4. Risk Factors
        5. Investment Recommendation (BUY/HOLD/SELL)
        6. Fair Value Estimate
        
        Focus on long-term fundamental value creation potential.
        """
    
    def _extract_structured_insights(self, analysis: str) -> Dict[str, Any]:
        """Extract structured insights from LLM analysis."""
        # Simple extraction logic - can be enhanced with regex or NLP
        insights = {
            'financial_health_score': None,
            'growth_prospects': '',
            'key_strengths': [],
            'key_weaknesses': [],
            'risk_factors': [],
            'recommendation': '',
            'fair_value': None
        }
        
        # This would be enhanced with proper NLP extraction
        lines = analysis.split('\n')
        for line in lines:
            if 'financial health' in line.lower() and any(char.isdigit() for char in line):
                try:
                    score = float([char for char in line if char.isdigit()][0])
                    insights['financial_health_score'] = score
                except:
                    pass
        
        return insights

class SECFilingAnalyzerTool(BaseTool):
    """Tool for analyzing SEC filings and extracting key information."""
    
    name: str = "sec_filing_analyzer"
    description: str = "Analyze SEC filings (10-K, 10-Q) for key business insights and risks"
    
    def _run(self, symbol: str, filing_type: str = "10-K") -> Dict[str, Any]:
        """Analyze SEC filings for a company."""
        try:
            # This would integrate with SEC EDGAR API
            # For now, using mock analysis with available data
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Mock SEC filing analysis
            filing_analysis = {
                'filing_type': filing_type,
                'business_overview': info.get('longBusinessSummary', ''),
                'risk_factors': self._extract_risk_factors(info),
                'competitive_advantages': self._identify_competitive_advantages(info),
                'growth_drivers': self._identify_growth_drivers(info),
                'financial_highlights': self._extract_financial_highlights(ticker)
            }
            
            return {
                'symbol': symbol,
                'filing_analysis': filing_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing SEC filings for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _extract_risk_factors(self, info: Dict) -> list:
        """Extract risk factors from company information."""
        # Mock risk factor extraction
        risk_factors = []
        
        if info.get('beta', 1.0) > 1.5:
            risk_factors.append("High market volatility")
        
        if info.get('debtToEquity', 0) > 1.0:
            risk_factors.append("High debt levels")
            
        if info.get('currentRatio', 1.0) < 1.0:
            risk_factors.append("Liquidity concerns")
            
        return risk_factors
    
    def _identify_competitive_advantages(self, info: Dict) -> list:
        """Identify competitive advantages."""
        advantages = []
        
        if info.get('profitMargins', 0) > 0.15:
            advantages.append("Strong profit margins")
            
        if info.get('returnOnEquity', 0) > 0.15:
            advantages.append("High return on equity")
            
        if info.get('marketCap', 0) > 100_000_000_000:
            advantages.append("Large market capitalization")
            
        return advantages
    
    def _identify_growth_drivers(self, info: Dict) -> list:
        """Identify growth drivers."""
        drivers = []
        
        if info.get('revenueGrowth', 0) > 0.1:
            drivers.append("Strong revenue growth")
            
        if info.get('earningsGrowth', 0) > 0.1:
            drivers.append("Growing earnings")
            
        return drivers
    
    def _extract_financial_highlights(self, ticker) -> Dict:
        """Extract financial highlights."""
        try:
            financials = ticker.financials
            if not financials.empty:
                latest_year = financials.columns[0]
                return {
                    'revenue': financials.loc['Total Revenue', latest_year] if 'Total Revenue' in financials.index else None,
                    'net_income': financials.loc['Net Income', latest_year] if 'Net Income' in financials.index else None,
                    'operating_income': financials.loc['Operating Income', latest_year] if 'Operating Income' in financials.index else None
                }
        except:
            pass
        return {}

class FinancialRatioCalculatorTool(BaseTool):
    """Tool for calculating comprehensive financial ratios."""
    
    name: str = "financial_ratio_calculator"
    description: str = "Calculate comprehensive financial ratios for fundamental analysis"
    
    def _run(self, symbol: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate financial ratios from financial statements."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Calculate various financial ratios
            ratios = {
                'liquidity_ratios': self._calculate_liquidity_ratios(info),
                'profitability_ratios': self._calculate_profitability_ratios(info),
                'leverage_ratios': self._calculate_leverage_ratios(info),
                'efficiency_ratios': self._calculate_efficiency_ratios(info),
                'valuation_ratios': self._calculate_valuation_ratios(info),
                'growth_ratios': self._calculate_growth_ratios(info)
            }
            
            # Calculate composite score
            fundamental_score = self._calculate_fundamental_score(ratios)
            
            return {
                'symbol': symbol,
                'financial_ratios': ratios,
                'fundamental_score': fundamental_score,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating ratios for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _calculate_liquidity_ratios(self, info: Dict) -> Dict[str, float]:
        """Calculate liquidity ratios."""
        return {
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'cash_ratio': info.get('totalCash', 0) / max(info.get('totalCurrentLiabilities', 1), 1)
        }
    
    def _calculate_profitability_ratios(self, info: Dict) -> Dict[str, float]:
        """Calculate profitability ratios."""
        return {
            'gross_margin': info.get('grossMargins'),
            'operating_margin': info.get('operatingMargins'),
            'net_margin': info.get('profitMargins'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'roic': info.get('returnOnCapital')
        }
    
    def _calculate_leverage_ratios(self, info: Dict) -> Dict[str, float]:
        """Calculate leverage ratios."""
        return {
            'debt_to_equity': info.get('debtToEquity'),
            'debt_to_assets': info.get('totalDebt', 0) / max(info.get('totalAssets', 1), 1),
            'interest_coverage': info.get('interestCoverage')
        }
    
    def _calculate_efficiency_ratios(self, info: Dict) -> Dict[str, float]:
        """Calculate efficiency ratios."""
        return {
            'asset_turnover': info.get('totalRevenue', 0) / max(info.get('totalAssets', 1), 1),
            'inventory_turnover': info.get('totalRevenue', 0) / max(info.get('inventory', 1), 1),
            'receivables_turnover': info.get('totalRevenue', 0) / max(info.get('accountsReceivable', 1), 1)
        }
    
    def _calculate_valuation_ratios(self, info: Dict) -> Dict[str, float]:
        """Calculate valuation ratios."""
        return {
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'ev_to_revenue': info.get('enterpriseToRevenue'),
            'ev_to_ebitda': info.get('enterpriseToEbitda'),
            'peg_ratio': info.get('pegRatio')
        }
    
    def _calculate_growth_ratios(self, info: Dict) -> Dict[str, float]:
        """Calculate growth ratios."""
        return {
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'book_value_growth': info.get('bookValueGrowth'),
            'dividend_growth': info.get('dividendGrowth')
        }
    
    def _calculate_fundamental_score(self, ratios: Dict) -> float:
        """Calculate a composite fundamental score (1-10)."""
        try:
            score = 5.0  # Base score
            
            # Profitability scoring
            roe = ratios['profitability_ratios'].get('roe', 0)
            if roe and roe > 0.15:
                score += 1
            elif roe and roe > 0.10:
                score += 0.5
            
            # Liquidity scoring
            current_ratio = ratios['liquidity_ratios'].get('current_ratio', 0)
            if current_ratio and current_ratio > 2.0:
                score += 0.5
            elif current_ratio and current_ratio > 1.5:
                score += 0.25
            
            # Growth scoring
            revenue_growth = ratios['growth_ratios'].get('revenue_growth', 0)
            if revenue_growth and revenue_growth > 0.1:
                score += 1
            elif revenue_growth and revenue_growth > 0.05:
                score += 0.5
            
            # Valuation scoring (lower P/E is better)
            pe_ratio = ratios['valuation_ratios'].get('pe_ratio', 0)
            if pe_ratio and 10 < pe_ratio < 20:
                score += 0.5
            
            return min(10.0, max(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return 5.0

# Initialize tools
fundamental_report_pull = FundamentalReportPullTool()
financial_report_rag = FinancialReportRAGTool()
sec_filing_analyzer = SECFilingAnalyzerTool()
financial_ratio_calculator = FinancialRatioCalculatorTool()