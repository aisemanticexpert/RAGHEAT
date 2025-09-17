"""
Valuation Analyst Agent for RAGHeat CrewAI System
================================================

This agent specializes in quantitative valuation analysis, technical indicators,
and risk-adjusted return metrics to determine optimal entry and exit points.
"""

from typing import Dict, Any, List
from .base_agent import RAGHeatBaseAgent
import logging

logger = logging.getLogger(__name__)

class ValuationAnalystAgent(RAGHeatBaseAgent):
    """
    Valuation Analyst Agent for quantitative analysis and technical valuation.
    
    Specializes in:
    - Technical analysis and chart patterns
    - Quantitative valuation metrics
    - Risk-adjusted return calculations
    - Price and volume analysis
    - Options flow and market microstructure
    """
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive valuation and technical analysis.
        
        Args:
            input_data: Dictionary containing:
                - stocks: List of stock tickers to analyze
                - timeframes: Analysis timeframes (1d, 1w, 1m, 3m, 1y)
                - indicators: Technical indicators to calculate
                - benchmark: Benchmark for relative valuation
        
        Returns:
            Valuation analysis results with entry/exit recommendations
        """
        try:
            stocks = input_data.get("stocks", [])
            timeframes = input_data.get("timeframes", ["1d", "1w", "1m", "3m", "1y"])
            indicators = input_data.get("indicators", ["RSI", "MACD", "BB", "MA"])
            benchmark = input_data.get("benchmark", "SPY")
            
            if not stocks:
                return {"error": "No stocks provided for valuation analysis"}
            
            logger.info(f"Valuation analysis starting for {len(stocks)} stocks")
            
            # Prepare analysis context
            analysis_context = {
                "stocks": stocks,
                "timeframes": timeframes,
                "indicators": indicators,
                "benchmark": benchmark,
                "analysis_type": "valuation",
                "focus_areas": [
                    "technical_indicators",
                    "price_action",
                    "volume_analysis",
                    "volatility_metrics",
                    "relative_valuation"
                ]
            }
            
            # Execute valuation analysis task
            task_description = f"""
            Conduct comprehensive valuation and technical analysis for: {', '.join(stocks)}
            
            Timeframes: {', '.join(timeframes)}
            Technical Indicators: {', '.join(indicators)}
            Benchmark: {benchmark}
            
            For each stock, analyze:
            
            1. TECHNICAL INDICATOR ANALYSIS:
               - RSI (14-period): Overbought/oversold levels and divergences
               - MACD: Signal line crossovers and histogram analysis
               - Bollinger Bands: Price position relative to bands and squeeze patterns
               - Moving Averages: 20, 50, 200-day MA support/resistance and crossovers
               - Volume indicators: OBV, Volume Price Trend, Accumulation/Distribution
               - Momentum indicators: Stochastic, Williams %R, Rate of Change
            
            2. PRICE ACTION AND CHART PATTERNS:
               - Support and resistance levels identification
               - Trend analysis (uptrend, downtrend, sideways)
               - Chart patterns: Head & Shoulders, Double Top/Bottom, Triangles, Flags
               - Candlestick patterns and reversal signals
               - Price gaps and their implications
               - Fibonacci retracement and extension levels
            
            3. VOLUME ANALYSIS:
               - Volume trends and price-volume relationships
               - Volume breakouts and confirmation signals
               - Accumulation vs distribution patterns
               - Volume at price (VAP) analysis
               - Institutional vs retail volume patterns
            
            4. VOLATILITY AND RISK METRICS:
               - Historical volatility analysis (10, 20, 30-day)
               - Implied volatility from options (if available)
               - VaR (Value at Risk) calculations
               - Maximum drawdown analysis
               - Beta relative to benchmark
               - Correlation analysis with market and sector
            
            5. QUANTITATIVE VALUATION:
               - Statistical measures: Mean reversion potential
               - Z-score analysis relative to historical ranges
               - Sharpe ratio and risk-adjusted returns
               - Alpha generation vs benchmark
               - Information ratio and tracking error
               - Sortino ratio (downside deviation focus)
            
            6. RELATIVE VALUATION:
               - Performance vs benchmark and sector
               - Relative strength index vs market
               - Sector rotation implications
               - Cross-asset correlations (bonds, commodities, currencies)
            
            7. OPTIONS FLOW ANALYSIS (if applicable):
               - Put/Call ratios and sentiment
               - Unusual options activity
               - Gamma exposure and dealer positioning
               - Implied volatility skew analysis
            
            For each stock, provide:
            - Technical Rating (1-10): Overall technical strength
            - Trend Status: Strong Uptrend, Uptrend, Neutral, Downtrend, Strong Downtrend
            - Momentum Score (1-10): Price and volume momentum
            - Volatility Assessment: Low, Medium, High with percentile ranking
            - Entry/Exit Signals: Buy, Sell, Hold with specific price levels
            - Risk Level: Conservative, Moderate, Aggressive
            - Time Horizon Recommendation: Short-term (1-30d), Medium-term (1-6m), Long-term (6m+)
            - Price Targets: Conservative, Base Case, Optimistic scenarios
            - Stop Loss Levels: Technical and volatility-based stops
            - Position Sizing Recommendations: Based on volatility and risk
            
            Identify:
            - Optimal entry and exit points with specific price levels
            - Risk management recommendations and stop-loss placement
            - Breakout/breakdown levels and confirmation signals
            - Mean reversion opportunities
            - Correlation trades and hedging opportunities
            
            Focus on risk-adjusted returns and optimal timing for position entry/exit.
            """
            
            result = self.execute_task(task_description, analysis_context)
            
            # Post-process results to ensure structured output
            processed_result = self._structure_valuation_analysis(result, stocks)
            
            logger.info(f"Valuation analysis completed for {len(stocks)} stocks")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error in valuation analysis: {e}")
            return {
                "error": str(e),
                "agent": "valuation_analyst",
                "analysis_type": "valuation"
            }
    
    def _structure_valuation_analysis(self, raw_result: Dict[str, Any], stocks: List[str]) -> Dict[str, Any]:
        """Structure the valuation analysis results."""
        
        structured_result = {
            "analysis_type": "valuation",
            "agent": "valuation_analyst",
            "timestamp": self._get_current_timestamp(),
            "stocks_analyzed": stocks,
            "overall_analysis": raw_result.get("result", ""),
            "technical_ratings": {},
            "entry_exit_signals": {},
            "risk_assessments": {},
            "market_overview": "",
            "trading_opportunities": [],
            "risk_alerts": []
        }
        
        # Extract structured data from result text
        result_text = str(raw_result.get("result", ""))
        
        # Extract trading opportunities
        opportunities = self._extract_trading_opportunities(result_text)
        structured_result["trading_opportunities"] = opportunities
        
        # Extract risk alerts
        alerts = self._extract_risk_alerts(result_text)
        structured_result["risk_alerts"] = alerts
        
        # For each stock, extract specific valuation data
        for stock in stocks:
            valuation_data = self._extract_stock_valuation_data(result_text, stock)
            if valuation_data:
                structured_result["technical_ratings"][stock] = valuation_data
        
        return structured_result
    
    def _extract_trading_opportunities(self, text: str) -> List[str]:
        """Extract trading opportunities from analysis text."""
        opportunities = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['opportunity:', 'breakout:', 'entry point:', 'trade:']):
                clean_opportunity = line.split(':', 1)[-1].strip() if ':' in line else line
                if clean_opportunity and len(clean_opportunity) > 20:
                    opportunities.append(clean_opportunity)
        
        return opportunities[:5]  # Limit to top 5 opportunities
    
    def _extract_risk_alerts(self, text: str) -> List[str]:
        """Extract risk alerts from analysis text."""
        alerts = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['risk:', 'warning:', 'caution:', 'alert:']):
                clean_alert = line.split(':', 1)[-1].strip() if ':' in line else line
                if clean_alert and len(clean_alert) > 15:
                    alerts.append(clean_alert)
        
        return alerts[:3]  # Limit to top 3 alerts
    
    def _extract_stock_valuation_data(self, text: str, stock: str) -> Dict[str, Any]:
        """Extract stock-specific valuation data from analysis text."""
        valuation_data = {
            "ticker": stock,
            "technical_rating": 7.0,  # Default
            "trend_status": "Neutral",
            "momentum_score": 7.0,
            "volatility_level": "Medium",
            "entry_signal": "HOLD",
            "risk_level": "Moderate",
            "time_horizon": "Medium-term",
            "price_targets": {
                "conservative": None,
                "base_case": None,
                "optimistic": None
            },
            "stop_loss": None,
            "position_size": "Standard",
            "key_levels": []
        }
        
        # Look for stock-specific technical indicators
        text_lower = text.lower()
        stock_lower = stock.lower()
        
        # Extract signals
        if f"{stock_lower} buy" in text_lower or f"bullish {stock_lower}" in text_lower:
            valuation_data["entry_signal"] = "BUY"
            valuation_data["technical_rating"] = 8.0
        elif f"{stock_lower} sell" in text_lower or f"bearish {stock_lower}" in text_lower:
            valuation_data["entry_signal"] = "SELL"
            valuation_data["technical_rating"] = 4.0
        
        return valuation_data
    
    def analyze_technical_indicators(self, stocks: List[str], indicators: List[str] = None) -> Dict[str, Any]:
        """
        Specialized method for technical indicator analysis.
        
        Args:
            stocks: List of stock tickers
            indicators: Specific indicators to analyze
            
        Returns:
            Detailed technical indicator analysis
        """
        indicators = indicators or ["RSI", "MACD", "BB", "MA", "Volume"]
        
        task_description = f"""
        Perform detailed technical indicator analysis for: {', '.join(stocks)}
        
        Indicators: {', '.join(indicators)}
        
        For each indicator, analyze:
        
        1. RSI ANALYSIS:
           - Current RSI level and trend
           - Overbought (>70) and oversold (<30) conditions
           - Bullish and bearish divergences
           - RSI support and resistance levels
        
        2. MACD ANALYSIS:
           - MACD line vs signal line position
           - Histogram expansion/contraction
           - Bullish/bearish crossovers
           - Divergence patterns with price
        
        3. BOLLINGER BANDS:
           - Price position relative to bands
           - Band width and volatility implications
           - Squeeze patterns and potential breakouts
           - Band walk patterns (strong trends)
        
        4. MOVING AVERAGES:
           - Price vs 20, 50, 200-day MAs
           - MA crossover signals (golden/death cross)
           - MA slope and trend confirmation
           - Dynamic support/resistance levels
        
        5. VOLUME INDICATORS:
           - Volume trend vs price trend
           - Volume breakouts and confirmations
           - On-Balance Volume (OBV) divergences
           - Volume-weighted average price (VWAP)
        
        Provide specific entry/exit signals and confluence areas.
        """
        
        context = {
            "stocks": stocks,
            "indicators": indicators,
            "analysis_type": "technical_indicators"
        }
        
        return self.execute_task(task_description, context)
    
    def analyze_risk_metrics(self, stocks: List[str], benchmark: str = "SPY") -> Dict[str, Any]:
        """
        Specialized method for risk and volatility analysis.
        
        Args:
            stocks: List of stock tickers
            benchmark: Benchmark for relative analysis
            
        Returns:
            Risk and volatility analysis
        """
        task_description = f"""
        Analyze risk metrics and volatility for: {', '.join(stocks)}
        
        Benchmark: {benchmark}
        
        Calculate and analyze:
        
        1. VOLATILITY METRICS:
           - Historical volatility (10, 20, 30, 60-day)
           - Realized vs implied volatility
           - Volatility percentiles and rankings
           - Volatility clustering patterns
        
        2. RISK-ADJUSTED RETURNS:
           - Sharpe ratio (excess return/volatility)
           - Sortino ratio (downside deviation focus)
           - Calmar ratio (return/max drawdown)
           - Information ratio vs benchmark
        
        3. DRAWDOWN ANALYSIS:
           - Maximum drawdown periods
           - Current drawdown from highs
           - Drawdown recovery time analysis
           - Drawdown frequency and magnitude
        
        4. CORRELATION ANALYSIS:
           - Correlation with benchmark
           - Sector correlation patterns
           - Cross-asset correlations
           - Rolling correlation trends
        
        5. VALUE AT RISK (VAR):
           - 1-day, 1-week VaR estimates
           - Conditional VaR (Expected Shortfall)
           - Historical vs parametric VaR
           - Stress testing scenarios
        
        6. BETA ANALYSIS:
           - Beta vs benchmark
           - Up/down market beta
           - Rolling beta trends
           - Beta stability assessment
        
        Provide risk-based position sizing recommendations.
        """
        
        context = {
            "stocks": stocks,
            "benchmark": benchmark,
            "analysis_type": "risk_metrics"
        }
        
        return self.execute_task(task_description, context)
    
    def analyze_options_flow(self, stocks: List[str]) -> Dict[str, Any]:
        """
        Specialized method for options flow and sentiment analysis.
        
        Args:
            stocks: List of stock tickers
            
        Returns:
            Options flow analysis
        """
        task_description = f"""
        Analyze options flow and sentiment for: {', '.join(stocks)}
        
        Analyze:
        
        1. OPTIONS VOLUME ANALYSIS:
           - Call vs put volume ratios
           - Unusual options activity (UOA)
           - Volume vs open interest patterns
           - Expiration-based flow analysis
        
        2. IMPLIED VOLATILITY:
           - IV rank and percentiles
           - IV skew analysis (put vs call IV)
           - Term structure patterns
           - IV vs realized volatility gaps
        
        3. GAMMA EXPOSURE:
           - Dealer gamma positioning
           - Gamma-driven price levels
           - Gamma squeeze potential
           - Zero gamma levels identification
        
        4. OPTIONS SENTIMENT:
           - Put/call ratio trends
           - Skew sentiment indicators
           - Fear/greed from options positioning
           - Institutional vs retail flow
        
        5. FLOW-BASED SIGNALS:
           - Bullish/bearish flow patterns
           - Options-driven price targets
           - Hedge-driven vs directional flow
           - Cross-asset options correlations
        
        Identify options-driven price catalysts and technical levels.
        """
        
        context = {
            "stocks": stocks,
            "analysis_type": "options_flow"
        }
        
        return self.execute_task(task_description, context)