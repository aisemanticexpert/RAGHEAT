"""
Fundamental Analysis Tools for RAGHeat CrewAI System
===================================================

Tools for fundamental analysis including SEC filings, financial ratios, and company research.
"""

from typing import Dict, Any, List, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import logging
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FundamentalReportPullInput(BaseModel):
    """Input schema for fundamental report pull tool."""
    ticker: str = Field(..., description="Stock ticker symbol")
    report_types: List[str] = Field(default=["10-K", "10-Q"], description="Types of reports to pull")
    lookback_days: int = Field(default=365, description="Days to look back for reports")

class FundamentalReportPull(BaseTool):
    """Tool for pulling fundamental reports and SEC filings."""
    
    name: str = Field(default="fundamental_report_pull")
    description: str = Field(default="Pull fundamental reports and SEC filings for stock analysis")
    args_schema: type[BaseModel] = FundamentalReportPullInput
    
    def _run(self, ticker: str, report_types: List[str] = None, lookback_days: int = 365) -> Dict[str, Any]:
        """
        Pull fundamental reports for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            report_types: Types of reports to pull
            lookback_days: Days to look back for reports
            
        Returns:
            Dictionary containing report data and metadata
        """
        try:
            report_types = report_types or ["10-K", "10-Q"]
            
            # Get basic company info using yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get financial statements
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            # Extract key financial metrics
            key_metrics = self._extract_key_metrics(info, financials, balance_sheet, cashflow)
            
            result = {
                "ticker": ticker,
                "company_name": info.get("longName", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "key_metrics": key_metrics,
                "financial_statements": {
                    "income_statement": financials.to_dict() if not financials.empty else {},
                    "balance_sheet": balance_sheet.to_dict() if not balance_sheet.empty else {},
                    "cash_flow": cashflow.to_dict() if not cashflow.empty else {}
                },
                "report_summary": self._generate_report_summary(key_metrics, info),
                "data_timestamp": datetime.now().isoformat(),
                "data_quality": "high" if key_metrics else "limited"
            }
            
            logger.info(f"Successfully pulled fundamental data for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error pulling fundamental data for {ticker}: {e}")
            return {
                "ticker": ticker,
                "error": str(e),
                "data_timestamp": datetime.now().isoformat(),
                "data_quality": "error"
            }
    
    def _extract_key_metrics(self, info: Dict, financials: pd.DataFrame, 
                           balance_sheet: pd.DataFrame, cashflow: pd.DataFrame) -> Dict[str, Any]:
        """Extract key financial metrics from financial statements."""
        metrics = {}
        
        try:
            # Valuation metrics
            metrics["pe_ratio"] = info.get("trailingPE")
            metrics["forward_pe"] = info.get("forwardPE")
            metrics["price_to_book"] = info.get("priceToBook")
            metrics["price_to_sales"] = info.get("priceToSalesTrailing12Months")
            metrics["ev_to_ebitda"] = info.get("enterpriseToEbitda")
            
            # Profitability metrics
            metrics["profit_margin"] = info.get("profitMargins")
            metrics["operating_margin"] = info.get("operatingMargins")
            metrics["return_on_equity"] = info.get("returnOnEquity")
            metrics["return_on_assets"] = info.get("returnOnAssets")
            
            # Growth metrics
            metrics["revenue_growth"] = info.get("revenueGrowth")
            metrics["earnings_growth"] = info.get("earningsGrowth")
            
            # Financial health metrics
            metrics["debt_to_equity"] = info.get("debtToEquity")
            metrics["current_ratio"] = info.get("currentRatio")
            metrics["quick_ratio"] = info.get("quickRatio")
            
            # Cash flow metrics
            metrics["free_cash_flow"] = info.get("freeCashflow")
            metrics["operating_cash_flow"] = info.get("operatingCashflow")
            
            # Dividend metrics
            metrics["dividend_yield"] = info.get("dividendYield")
            metrics["payout_ratio"] = info.get("payoutRatio")
            
        except Exception as e:
            logger.warning(f"Error extracting metrics: {e}")
        
        return {k: v for k, v in metrics.items() if v is not None}
    
    def _generate_report_summary(self, metrics: Dict[str, Any], info: Dict[str, Any]) -> str:
        """Generate a summary of the fundamental analysis."""
        summary_parts = []
        
        company_name = info.get("longName", "Company")
        sector = info.get("sector", "Unknown sector")
        
        summary_parts.append(f"{company_name} operates in the {sector} sector.")
        
        # Valuation assessment
        pe_ratio = metrics.get("pe_ratio")
        if pe_ratio:
            if pe_ratio < 15:
                summary_parts.append("The stock appears undervalued based on P/E ratio.")
            elif pe_ratio > 25:
                summary_parts.append("The stock appears expensive based on P/E ratio.")
            else:
                summary_parts.append("The stock is fairly valued based on P/E ratio.")
        
        # Profitability assessment
        profit_margin = metrics.get("profit_margin")
        if profit_margin:
            if profit_margin > 0.15:
                summary_parts.append("The company shows strong profitability.")
            elif profit_margin > 0.05:
                summary_parts.append("The company shows moderate profitability.")
            else:
                summary_parts.append("The company has low profitability.")
        
        # Growth assessment
        revenue_growth = metrics.get("revenue_growth")
        if revenue_growth:
            if revenue_growth > 0.1:
                summary_parts.append("The company demonstrates strong revenue growth.")
            elif revenue_growth > 0:
                summary_parts.append("The company shows positive revenue growth.")
            else:
                summary_parts.append("The company is experiencing revenue decline.")
        
        return " ".join(summary_parts)

class FinancialRatioCalculatorInput(BaseModel):
    """Input schema for financial ratio calculator."""
    ticker: str = Field(..., description="Stock ticker symbol")
    ratios: List[str] = Field(default=["all"], description="Specific ratios to calculate")
    periods: int = Field(default=4, description="Number of periods to analyze")

class FinancialRatioCalculator(BaseTool):
    """Tool for calculating comprehensive financial ratios."""
    
    name: str = Field(default="financial_ratio_calculator")
    description: str = Field(default="Calculate comprehensive financial ratios for fundamental analysis")
    args_schema: type[BaseModel] = FinancialRatioCalculatorInput
    
    def _run(self, ticker: str, ratios: List[str] = None, periods: int = 4) -> Dict[str, Any]:
        """
        Calculate financial ratios for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            ratios: Specific ratios to calculate
            periods: Number of periods to analyze
            
        Returns:
            Dictionary containing calculated ratios
        """
        try:
            stock = yf.Ticker(ticker)
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            
            if financials.empty or balance_sheet.empty:
                return {
                    "ticker": ticker,
                    "error": "Insufficient financial data available",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Calculate ratios
            calculated_ratios = {}
            
            # Liquidity ratios
            calculated_ratios["liquidity"] = self._calculate_liquidity_ratios(balance_sheet)
            
            # Leverage ratios
            calculated_ratios["leverage"] = self._calculate_leverage_ratios(balance_sheet, financials)
            
            # Efficiency ratios
            calculated_ratios["efficiency"] = self._calculate_efficiency_ratios(financials, balance_sheet)
            
            # Profitability ratios
            calculated_ratios["profitability"] = self._calculate_profitability_ratios(financials, balance_sheet)
            
            # Market ratios (from current market data)
            calculated_ratios["market"] = self._calculate_market_ratios(stock)
            
            # Trend analysis
            calculated_ratios["trends"] = self._analyze_ratio_trends(calculated_ratios, periods)
            
            result = {
                "ticker": ticker,
                "ratios": calculated_ratios,
                "analysis_summary": self._generate_ratio_analysis(calculated_ratios),
                "periods_analyzed": periods,
                "timestamp": datetime.now().isoformat(),
                "data_quality": "high"
            }
            
            logger.info(f"Successfully calculated ratios for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating ratios for {ticker}: {e}")
            return {
                "ticker": ticker,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_liquidity_ratios(self, balance_sheet: pd.DataFrame) -> Dict[str, float]:
        """Calculate liquidity ratios."""
        ratios = {}
        
        try:
            latest_data = balance_sheet.iloc[:, 0]  # Most recent period
            
            current_assets = latest_data.get("Total Current Assets", 0)
            current_liabilities = latest_data.get("Total Current Liabilities", 0)
            cash = latest_data.get("Cash And Cash Equivalents", 0)
            inventory = latest_data.get("Inventory", 0)
            
            if current_liabilities > 0:
                ratios["current_ratio"] = current_assets / current_liabilities
                ratios["quick_ratio"] = (current_assets - inventory) / current_liabilities
                ratios["cash_ratio"] = cash / current_liabilities
            
        except Exception as e:
            logger.warning(f"Error calculating liquidity ratios: {e}")
        
        return ratios
    
    def _calculate_leverage_ratios(self, balance_sheet: pd.DataFrame, financials: pd.DataFrame) -> Dict[str, float]:
        """Calculate leverage ratios."""
        ratios = {}
        
        try:
            bs_latest = balance_sheet.iloc[:, 0]
            fin_latest = financials.iloc[:, 0]
            
            total_debt = bs_latest.get("Total Debt", 0)
            total_equity = bs_latest.get("Total Stockholder Equity", 0)
            total_assets = bs_latest.get("Total Assets", 0)
            interest_expense = abs(fin_latest.get("Interest Expense", 0))
            ebit = fin_latest.get("EBIT", 0)
            
            if total_equity > 0:
                ratios["debt_to_equity"] = total_debt / total_equity
            
            if total_assets > 0:
                ratios["debt_to_assets"] = total_debt / total_assets
            
            if interest_expense > 0 and ebit > 0:
                ratios["interest_coverage"] = ebit / interest_expense
            
        except Exception as e:
            logger.warning(f"Error calculating leverage ratios: {e}")
        
        return ratios
    
    def _calculate_efficiency_ratios(self, financials: pd.DataFrame, balance_sheet: pd.DataFrame) -> Dict[str, float]:
        """Calculate efficiency ratios."""
        ratios = {}
        
        try:
            fin_latest = financials.iloc[:, 0]
            bs_latest = balance_sheet.iloc[:, 0]
            
            revenue = fin_latest.get("Total Revenue", 0)
            total_assets = bs_latest.get("Total Assets", 0)
            inventory = bs_latest.get("Inventory", 0)
            accounts_receivable = bs_latest.get("Accounts Receivable", 0)
            cogs = fin_latest.get("Cost Of Revenue", 0)
            
            if total_assets > 0:
                ratios["asset_turnover"] = revenue / total_assets
            
            if inventory > 0:
                ratios["inventory_turnover"] = cogs / inventory
            
            if accounts_receivable > 0:
                ratios["receivables_turnover"] = revenue / accounts_receivable
            
        except Exception as e:
            logger.warning(f"Error calculating efficiency ratios: {e}")
        
        return ratios
    
    def _calculate_profitability_ratios(self, financials: pd.DataFrame, balance_sheet: pd.DataFrame) -> Dict[str, float]:
        """Calculate profitability ratios."""
        ratios = {}
        
        try:
            fin_latest = financials.iloc[:, 0]
            bs_latest = balance_sheet.iloc[:, 0]
            
            revenue = fin_latest.get("Total Revenue", 0)
            gross_profit = fin_latest.get("Gross Profit", 0)
            operating_income = fin_latest.get("Operating Income", 0)
            net_income = fin_latest.get("Net Income", 0)
            total_assets = bs_latest.get("Total Assets", 0)
            total_equity = bs_latest.get("Total Stockholder Equity", 0)
            
            if revenue > 0:
                ratios["gross_margin"] = gross_profit / revenue
                ratios["operating_margin"] = operating_income / revenue
                ratios["net_margin"] = net_income / revenue
            
            if total_assets > 0:
                ratios["return_on_assets"] = net_income / total_assets
            
            if total_equity > 0:
                ratios["return_on_equity"] = net_income / total_equity
            
        except Exception as e:
            logger.warning(f"Error calculating profitability ratios: {e}")
        
        return ratios
    
    def _calculate_market_ratios(self, stock) -> Dict[str, float]:
        """Calculate market-based ratios."""
        ratios = {}
        
        try:
            info = stock.info
            
            ratios["pe_ratio"] = info.get("trailingPE")
            ratios["forward_pe"] = info.get("forwardPE")
            ratios["price_to_book"] = info.get("priceToBook")
            ratios["price_to_sales"] = info.get("priceToSalesTrailing12Months")
            ratios["ev_to_ebitda"] = info.get("enterpriseToEbitda")
            ratios["peg_ratio"] = info.get("pegRatio")
            
        except Exception as e:
            logger.warning(f"Error calculating market ratios: {e}")
        
        return {k: v for k, v in ratios.items() if v is not None}
    
    def _analyze_ratio_trends(self, ratios: Dict[str, Any], periods: int) -> Dict[str, str]:
        """Analyze ratio trends over time."""
        trends = {}
        
        # This is a simplified trend analysis
        # In a real implementation, you would compare ratios across multiple periods
        
        try:
            profitability = ratios.get("profitability", {})
            leverage = ratios.get("leverage", {})
            liquidity = ratios.get("liquidity", {})
            
            # Analyze profitability trend
            net_margin = profitability.get("net_margin", 0)
            if net_margin > 0.1:
                trends["profitability"] = "Strong"
            elif net_margin > 0.05:
                trends["profitability"] = "Moderate"
            else:
                trends["profitability"] = "Weak"
            
            # Analyze leverage trend
            debt_to_equity = leverage.get("debt_to_equity", 0)
            if debt_to_equity < 0.3:
                trends["leverage"] = "Conservative"
            elif debt_to_equity < 0.6:
                trends["leverage"] = "Moderate"
            else:
                trends["leverage"] = "High"
            
            # Analyze liquidity trend
            current_ratio = liquidity.get("current_ratio", 0)
            if current_ratio > 2:
                trends["liquidity"] = "Strong"
            elif current_ratio > 1:
                trends["liquidity"] = "Adequate"
            else:
                trends["liquidity"] = "Weak"
                
        except Exception as e:
            logger.warning(f"Error analyzing trends: {e}")
        
        return trends
    
    def _generate_ratio_analysis(self, ratios: Dict[str, Any]) -> str:
        """Generate analysis summary from calculated ratios."""
        analysis_parts = []
        
        try:
            profitability = ratios.get("profitability", {})
            leverage = ratios.get("leverage", {})
            liquidity = ratios.get("liquidity", {})
            market = ratios.get("market", {})
            
            # Profitability analysis
            net_margin = profitability.get("net_margin", 0)
            if net_margin > 0.1:
                analysis_parts.append("Strong profitability with healthy margins.")
            elif net_margin > 0:
                analysis_parts.append("Moderate profitability.")
            else:
                analysis_parts.append("Profitability concerns.")
            
            # Leverage analysis
            debt_to_equity = leverage.get("debt_to_equity", 0)
            if debt_to_equity < 0.5:
                analysis_parts.append("Conservative debt levels.")
            elif debt_to_equity < 1.0:
                analysis_parts.append("Moderate debt levels.")
            else:
                analysis_parts.append("High debt levels may pose risk.")
            
            # Liquidity analysis
            current_ratio = liquidity.get("current_ratio", 0)
            if current_ratio > 1.5:
                analysis_parts.append("Strong liquidity position.")
            elif current_ratio > 1:
                analysis_parts.append("Adequate liquidity.")
            else:
                analysis_parts.append("Liquidity concerns.")
            
            # Valuation analysis
            pe_ratio = market.get("pe_ratio", 0)
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 15:
                    analysis_parts.append("Potentially undervalued.")
                elif pe_ratio > 25:
                    analysis_parts.append("May be overvalued.")
                else:
                    analysis_parts.append("Reasonable valuation.")
                    
        except Exception as e:
            logger.warning(f"Error generating analysis: {e}")
            analysis_parts.append("Analysis could not be completed.")
        
        return " ".join(analysis_parts)

# Register tools
from .tool_registry import register_tool

register_tool("fundamental_report_pull", FundamentalReportPull)
register_tool("financial_ratio_calculator", FinancialRatioCalculator)