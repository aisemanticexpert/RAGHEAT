# RAGHeat CrewAI Examples

## Basic Portfolio Construction

### Simple Portfolio Analysis

```python
from crewai import PortfolioConstructionCrew

# Initialize crew with default settings
crew = PortfolioConstructionCrew()

# Analyze a tech-focused portfolio
result = crew.construct_portfolio(
    target_stocks=["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"],
    investment_objective="growth",
    risk_tolerance="moderate",
    time_horizon="1y"
)

# Access results
print("Portfolio Recommendations:")
for stock, data in result["portfolio_recommendations"].items():
    print(f"{stock}: {data['recommendation']} (Weight: {data['weight']:.2%})")

print("\nRisk Assessment:")
print(f"Expected Return: {result['risk_assessment']['expected_return']:.2%}")
print(f"Expected Volatility: {result['risk_assessment']['volatility']:.2%}")
print(f"Sharpe Ratio: {result['risk_assessment']['sharpe_ratio']:.2f}")
```

### Custom Risk Parameters

```python
# Conservative portfolio with specific constraints
custom_constraints = {
    "max_position_weight": 0.08,  # 8% max per position
    "min_position_weight": 0.02,  # 2% min per position
    "sector_limits": {
        "technology": 0.35,
        "healthcare": 0.25,
        "financials": 0.20
    },
    "esg_score_minimum": 7.0,
    "exclude_stocks": ["TSLA"]  # Exclude specific stocks
}

result = crew.construct_portfolio(
    target_stocks=["AAPL", "GOOGL", "MSFT", "JNJ", "PFE", "JPM", "BAC"],
    investment_objective="balanced_growth",
    risk_tolerance="conservative", 
    time_horizon="2y",
    custom_constraints=custom_constraints
)
```

## Advanced Usage Examples

### Multi-Timeframe Analysis

```python
# Analyze same stocks across different timeframes
timeframes = ["3m", "6m", "1y", "2y"]
results = {}

for horizon in timeframes:
    results[horizon] = crew.construct_portfolio(
        target_stocks=["AAPL", "MSFT", "GOOGL"],
        investment_objective="growth",
        risk_tolerance="moderate",
        time_horizon=horizon
    )

# Compare recommendations across timeframes
for stock in ["AAPL", "MSFT", "GOOGL"]:
    print(f"\n{stock} Recommendations:")
    for horizon in timeframes:
        rec = results[horizon]["portfolio_recommendations"].get(stock, {})
        print(f"  {horizon}: {rec.get('recommendation', 'N/A')} "
              f"(Confidence: {rec.get('confidence', 0):.1f})")
```

### Sector Rotation Strategy

```python
# Analyze different sectors for rotation opportunities
sectors = {
    "technology": ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "healthcare": ["JNJ", "PFE", "UNH", "ABBV"],
    "financials": ["JPM", "BAC", "WFC", "GS"],
    "energy": ["XOM", "CVX", "COP", "EOG"]
}

sector_analysis = {}

for sector_name, stocks in sectors.items():
    print(f"\nAnalyzing {sector_name.title()} Sector...")
    
    sector_analysis[sector_name] = crew.construct_portfolio(
        target_stocks=stocks,
        investment_objective="growth",
        risk_tolerance="moderate",
        time_horizon="6m"
    )
    
    # Extract sector insights
    heat_analysis = sector_analysis[sector_name]["agent_analyses"].get("heat_diffusion_analyst", {})
    sentiment = sector_analysis[sector_name]["agent_analyses"].get("sentiment_analyst", {})
    
    print(f"  Heat Diffusion Risk: {heat_analysis.get('systemic_risk_score', 'N/A')}")
    print(f"  Sector Sentiment: {sentiment.get('aggregate_sentiment', 'N/A')}")

# Compare sectors
best_sector = max(sector_analysis.keys(), 
                 key=lambda s: sector_analysis[s]["risk_assessment"]["expected_return"])
print(f"\nBest performing sector: {best_sector}")
```

### Event-Driven Analysis

```python
# Analyze portfolio impact of specific events
shock_events = [
    {
        "event": "Federal Reserve Rate Hike", 
        "impact_magnitude": 0.8,
        "affected_sectors": ["financials", "real_estate"]
    },
    {
        "event": "Technology Earnings Season",
        "impact_magnitude": 0.6, 
        "affected_sectors": ["technology"]
    }
]

for event in shock_events:
    print(f"\nAnalyzing impact of: {event['event']}")
    
    # Get heat diffusion agent for event simulation
    heat_agent = crew.agents["heat_diffusion_analyst"]
    
    event_result = heat_agent.simulate_shock_event(
        event_description=event["event"],
        initial_impact=event["impact_magnitude"],
        affected_entities=event["affected_sectors"]
    )
    
    print(f"Expected cascade effects:")
    for entity, impact in event_result.get("entity_impacts", {}).items():
        print(f"  {entity}: {impact:.2f}")
```

### Custom Agent Configuration

```python
# Create crew with custom agent settings
custom_agent_config = {
    "fundamental_analyst": {
        "max_iter": 7,  # More thorough analysis
        "confidence_threshold": 0.8
    },
    "sentiment_analyst": {
        "sentiment_sources": ["news", "social_media", "analyst_ratings", "options_flow"],
        "lookback_days": 60
    },
    "heat_diffusion_analyst": {
        "diffusion_coefficient": 0.15,  # Higher propagation speed
        "iterations": 75
    }
}

custom_crew = PortfolioConstructionCrew(custom_agent_config)

result = custom_crew.construct_portfolio(
    target_stocks=["AAPL", "GOOGL", "TSLA"],
    investment_objective="growth",
    risk_tolerance="aggressive"
)
```

## Integration Examples

### FastAPI Web Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI(title="RAGHeat Portfolio API")

# Global crew instance
crew = PortfolioConstructionCrew()

class PortfolioRequest(BaseModel):
    stocks: List[str]
    objective: str = "balanced_growth"
    risk_tolerance: str = "moderate"
    time_horizon: str = "1y"
    constraints: Optional[dict] = None

class PortfolioResponse(BaseModel):
    portfolio_recommendations: dict
    risk_assessment: dict
    explanation_summary: str
    execution_time: float

@app.post("/portfolio/construct", response_model=PortfolioResponse)
async def construct_portfolio(request: PortfolioRequest):
    try:
        import time
        start_time = time.time()
        
        result = crew.construct_portfolio(
            target_stocks=request.stocks,
            investment_objective=request.objective,
            risk_tolerance=request.risk_tolerance,
            time_horizon=request.time_horizon,
            custom_constraints=request.constraints
        )
        
        execution_time = time.time() - start_time
        
        return PortfolioResponse(
            portfolio_recommendations=result["portfolio_recommendations"],
            risk_assessment=result["risk_assessment"],
            explanation_summary=result["explanation_package"]["executive_summary"],
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/agents/status")
async def get_agent_status():
    return crew.get_agent_performance()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Scheduled Portfolio Monitoring

```python
import schedule
import time
import json
from datetime import datetime
import smtplib
from email.mime.text import MimeText

class PortfolioMonitor:
    def __init__(self, watchlist: List[str]):
        self.crew = PortfolioConstructionCrew()
        self.watchlist = watchlist
        self.previous_results = {}
    
    def daily_analysis(self):
        """Perform daily portfolio analysis."""
        print(f"Running daily analysis at {datetime.now()}")
        
        result = self.crew.construct_portfolio(
            target_stocks=self.watchlist,
            investment_objective="balanced_growth",
            risk_tolerance="moderate"
        )
        
        # Check for significant changes
        changes = self.detect_changes(result)
        
        if changes["significant_changes"]:
            self.send_alert(changes)
        
        # Store results
        self.previous_results[datetime.now().date()] = result
        
        # Save to file
        with open(f"daily_analysis_{datetime.now().date()}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
    
    def detect_changes(self, current_result):
        """Detect significant changes from previous analysis."""
        changes = {
            "significant_changes": False,
            "new_buy_signals": [],
            "new_sell_signals": [],
            "risk_changes": []
        }
        
        if not self.previous_results:
            return changes
        
        latest_date = max(self.previous_results.keys())
        previous = self.previous_results[latest_date]
        
        # Compare recommendations
        current_recs = current_result["portfolio_recommendations"]
        previous_recs = previous["portfolio_recommendations"]
        
        for stock in self.watchlist:
            curr_rec = current_recs.get(stock, {}).get("recommendation")
            prev_rec = previous_recs.get(stock, {}).get("recommendation")
            
            if prev_rec != "BUY" and curr_rec == "BUY":
                changes["new_buy_signals"].append(stock)
                changes["significant_changes"] = True
            elif prev_rec != "SELL" and curr_rec == "SELL":
                changes["new_sell_signals"].append(stock)
                changes["significant_changes"] = True
        
        return changes
    
    def send_alert(self, changes):
        """Send email alert for significant changes."""
        subject = "RAGHeat Portfolio Alert"
        
        body = "Significant portfolio changes detected:\n\n"
        
        if changes["new_buy_signals"]:
            body += f"New BUY signals: {', '.join(changes['new_buy_signals'])}\n"
        
        if changes["new_sell_signals"]:
            body += f"New SELL signals: {', '.join(changes['new_sell_signals'])}\n"
        
        # Send email (configure SMTP settings)
        # self.send_email(subject, body)
        print(f"Alert: {body}")
    
    def weekly_deep_analysis(self):
        """Perform comprehensive weekly analysis."""
        print(f"Running weekly deep analysis at {datetime.now()}")
        
        # Analyze with longer timeframe
        result = self.crew.construct_portfolio(
            target_stocks=self.watchlist,
            investment_objective="balanced_growth",
            risk_tolerance="moderate",
            time_horizon="1y"
        )
        
        # Generate comprehensive report
        self.generate_weekly_report(result)
    
    def generate_weekly_report(self, result):
        """Generate detailed weekly report."""
        report = f"""
        RAGHeat Weekly Portfolio Report
        Generated: {datetime.now()}
        
        Portfolio Overview:
        - Stocks Analyzed: {len(self.watchlist)}
        - Buy Recommendations: {sum(1 for r in result['portfolio_recommendations'].values() if r.get('recommendation') == 'BUY')}
        - Hold Recommendations: {sum(1 for r in result['portfolio_recommendations'].values() if r.get('recommendation') == 'HOLD')}
        - Sell Recommendations: {sum(1 for r in result['portfolio_recommendations'].values() if r.get('recommendation') == 'SELL')}
        
        Risk Assessment:
        - Expected Return: {result['risk_assessment'].get('expected_return', 'N/A')}
        - Portfolio Volatility: {result['risk_assessment'].get('volatility', 'N/A')}
        - Sharpe Ratio: {result['risk_assessment'].get('sharpe_ratio', 'N/A')}
        
        Key Insights:
        {result['explanation_package'].get('executive_summary', 'No summary available')}
        """
        
        # Save report
        with open(f"weekly_report_{datetime.now().date()}.txt", "w") as f:
            f.write(report)
        
        print("Weekly report generated")

# Usage
monitor = PortfolioMonitor(["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])

# Schedule tasks
schedule.every().day.at("09:00").do(monitor.daily_analysis)
schedule.every().monday.at("08:00").do(monitor.weekly_deep_analysis)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(60)
```

### Backtesting Framework

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RAGHeatBacktester:
    def __init__(self, start_date: str, end_date: str):
        self.crew = PortfolioConstructionCrew()
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.results = []
    
    def run_backtest(self, stocks: List[str], rebalance_frequency: str = "monthly"):
        """Run historical backtest."""
        print(f"Running backtest from {self.start_date} to {self.end_date}")
        
        current_date = self.start_date
        
        while current_date <= self.end_date:
            print(f"Analyzing portfolio for {current_date.date()}")
            
            # Simulate historical analysis (in practice, would use historical data)
            result = self.crew.construct_portfolio(
                target_stocks=stocks,
                investment_objective="balanced_growth",
                risk_tolerance="moderate"
            )
            
            # Store result with date
            result["analysis_date"] = current_date
            self.results.append(result)
            
            # Move to next rebalance date
            if rebalance_frequency == "monthly":
                current_date = self.add_months(current_date, 1)
            elif rebalance_frequency == "weekly":
                current_date += timedelta(weeks=1)
            elif rebalance_frequency == "quarterly":
                current_date = self.add_months(current_date, 3)
    
    def add_months(self, date, months):
        """Add months to a date."""
        month = date.month - 1 + months
        year = date.year + month // 12
        month = month % 12 + 1
        day = min(date.day, [31,29,31,30,31,30,31,31,30,31,30,31][month-1])
        return date.replace(year=year, month=month, day=day)
    
    def calculate_performance(self):
        """Calculate backtest performance metrics."""
        if not self.results:
            return None
        
        # Extract returns (simulated)
        returns = []
        for result in self.results:
            expected_return = result["risk_assessment"].get("expected_return", 0.08)
            returns.append(expected_return)
        
        # Calculate metrics
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        max_drawdown = self.calculate_max_drawdown(returns)
        
        return {
            "total_periods": len(returns),
            "average_return": avg_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": sum(1 for r in returns if r > 0) / len(returns)
        }
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown."""
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return abs(min(drawdowns))
    
    def generate_report(self):
        """Generate backtest report."""
        performance = self.calculate_performance()
        
        report = f"""
        RAGHeat Backtest Report
        ======================
        
        Period: {self.start_date.date()} to {self.end_date.date()}
        Total Periods: {performance['total_periods']}
        
        Performance Metrics:
        - Average Return: {performance['average_return']:.2%}
        - Volatility: {performance['volatility']:.2%}
        - Sharpe Ratio: {performance['sharpe_ratio']:.2f}
        - Maximum Drawdown: {performance['max_drawdown']:.2%}
        - Win Rate: {performance['win_rate']:.2%}
        
        Agent Performance Summary:
        {self.crew.get_agent_performance()}
        """
        
        return report

# Usage
backtester = RAGHeatBacktester("2023-01-01", "2024-01-01")
backtester.run_backtest(["AAPL", "GOOGL", "MSFT"], "monthly")
print(backtester.generate_report())
```

## Error Handling Examples

### Robust Portfolio Construction

```python
import logging
from typing import Dict, Any, Optional

def robust_portfolio_construction(
    stocks: List[str],
    max_retries: int = 3,
    fallback_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Construct portfolio with error handling and fallbacks."""
    
    crew = PortfolioConstructionCrew()
    
    for attempt in range(max_retries):
        try:
            print(f"Portfolio construction attempt {attempt + 1}")
            
            result = crew.construct_portfolio(
                target_stocks=stocks,
                investment_objective="balanced_growth",
                risk_tolerance="moderate"
            )
            
            # Validate result
            if validate_portfolio_result(result):
                print("Portfolio construction successful")
                return result
            else:
                raise ValueError("Invalid portfolio result")
                
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries - 1:
                # Last attempt - use fallback
                print("Using fallback configuration")
                return fallback_portfolio_construction(stocks, fallback_config)
            
            # Wait before retry
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return {"error": "All attempts failed"}

def validate_portfolio_result(result: Dict[str, Any]) -> bool:
    """Validate portfolio construction result."""
    required_keys = ["portfolio_recommendations", "risk_assessment"]
    
    if not all(key in result for key in required_keys):
        return False
    
    if not result["portfolio_recommendations"]:
        return False
    
    # Check for reasonable risk metrics
    risk_assessment = result["risk_assessment"]
    if "expected_return" in risk_assessment:
        if not (-1 <= risk_assessment["expected_return"] <= 1):
            return False
    
    return True

def fallback_portfolio_construction(
    stocks: List[str], 
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Fallback portfolio construction with simplified logic."""
    
    # Simple equal-weight portfolio as fallback
    weight_per_stock = 1.0 / len(stocks)
    
    recommendations = {}
    for stock in stocks:
        recommendations[stock] = {
            "recommendation": "HOLD",
            "weight": weight_per_stock,
            "confidence": 0.5,
            "rationale": "Equal-weight fallback allocation"
        }
    
    return {
        "portfolio_recommendations": recommendations,
        "risk_assessment": {
            "expected_return": 0.08,
            "volatility": 0.16,
            "sharpe_ratio": 0.5
        },
        "execution_metadata": {
            "method": "fallback",
            "timestamp": datetime.now().isoformat()
        }
    }

# Usage
result = robust_portfolio_construction(["AAPL", "GOOGL", "MSFT"])
```

These examples demonstrate the flexibility and power of the RAGHeat CrewAI system for various portfolio construction scenarios, from simple analysis to complex backtesting and production deployments.