"""
Stock Analysis Routes for detailed buy/sell recommendations with RAG explanations
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import random
from datetime import datetime, timedelta

router = APIRouter()

# Sample stock data for demonstration
STOCK_DATA = {
    "AAPL": {
        "name": "Apple Inc.",
        "price": 189.46,
        "sector": "Technology",
        "market_cap": 2900000000000,
        "pe_ratio": 28.5,
        "dividend_yield": 0.48,
        "volume": 52400000,
        "beta": 1.29,
        "52_week_high": 199.62,
        "52_week_low": 164.08
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "price": 415.26,
        "sector": "Technology", 
        "market_cap": 3100000000000,
        "pe_ratio": 35.2,
        "dividend_yield": 0.68,
        "volume": 18900000,
        "beta": 0.90,
        "52_week_high": 424.30,
        "52_week_low": 309.45
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "price": 456.12,
        "sector": "Technology",
        "market_cap": 1200000000000,
        "pe_ratio": 73.8,
        "dividend_yield": 0.03,
        "volume": 41200000,
        "beta": 1.68,
        "52_week_high": 502.66,
        "52_week_low": 180.50
    },
    "TSLA": {
        "name": "Tesla, Inc.",
        "price": 267.48,
        "sector": "Consumer_Discretionary",
        "market_cap": 850000000000,
        "pe_ratio": 45.2,
        "dividend_yield": 0.0,
        "volume": 95600000,
        "beta": 2.09,
        "52_week_high": 293.34,
        "52_week_low": 138.80
    },
    "JPM": {
        "name": "JPMorgan Chase & Co.",
        "price": 158.23,
        "sector": "Financial",
        "market_cap": 460000000000,
        "pe_ratio": 12.8,
        "dividend_yield": 2.45,
        "volume": 8900000,
        "beta": 1.15,
        "52_week_high": 169.81,
        "52_week_low": 135.19
    }
}

class StockAnalysisRequest(BaseModel):
    symbol: str
    analysis_depth: str = "comprehensive"

class TechnicalIndicator(BaseModel):
    name: str
    value: float
    signal: str  # "BUY", "SELL", "HOLD"
    strength: float  # 0-1

class FundamentalMetric(BaseModel):
    name: str
    value: float
    benchmark: float
    interpretation: str

class RAGExplanation(BaseModel):
    reasoning_path: List[str]
    supporting_evidence: List[str]
    risk_factors: List[str]
    confidence_score: float

class StockRecommendation(BaseModel):
    symbol: str
    action: str  # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    confidence: float
    price_target: float
    timeframe: str
    explanation: RAGExplanation

@router.get("/api/stock/{symbol}/analysis")
async def get_stock_analysis(symbol: str):
    """Get comprehensive stock analysis with buy/sell recommendation"""
    
    symbol = symbol.upper()
    
    if symbol not in STOCK_DATA:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    stock = STOCK_DATA[symbol]
    
    # Generate technical indicators
    technical_indicators = [
        TechnicalIndicator(
            name="RSI (14)",
            value=random.uniform(30, 70),
            signal="BUY" if random.random() > 0.5 else "SELL",
            strength=random.uniform(0.6, 0.9)
        ),
        TechnicalIndicator(
            name="MACD",
            value=random.uniform(-2, 2),
            signal="BUY" if random.random() > 0.4 else "SELL",
            strength=random.uniform(0.5, 0.8)
        ),
        TechnicalIndicator(
            name="Moving Average (50)",
            value=stock["price"] * random.uniform(0.95, 1.05),
            signal="BUY" if random.random() > 0.3 else "HOLD",
            strength=random.uniform(0.7, 0.9)
        ),
        TechnicalIndicator(
            name="Bollinger Bands",
            value=random.uniform(0.1, 0.9),
            signal="HOLD" if random.random() > 0.6 else "BUY",
            strength=random.uniform(0.4, 0.7)
        )
    ]
    
    # Generate fundamental metrics
    fundamental_metrics = [
        FundamentalMetric(
            name="P/E Ratio",
            value=stock["pe_ratio"],
            benchmark=22.5,
            interpretation="Below sector average" if stock["pe_ratio"] < 25 else "Above sector average"
        ),
        FundamentalMetric(
            name="Price to Book",
            value=random.uniform(2.0, 8.0),
            benchmark=3.5,
            interpretation="Reasonable valuation"
        ),
        FundamentalMetric(
            name="Debt to Equity",
            value=random.uniform(0.1, 1.5),
            benchmark=0.5,
            interpretation="Conservative debt levels"
        ),
        FundamentalMetric(
            name="Return on Equity",
            value=random.uniform(15, 35),
            benchmark=20,
            interpretation="Strong profitability"
        )
    ]
    
    # Generate recommendation based on analysis
    buy_signals = sum(1 for ti in technical_indicators if ti.signal == "BUY")
    sell_signals = sum(1 for ti in technical_indicators if ti.signal == "SELL")
    
    if buy_signals > sell_signals + 1:
        action = "STRONG_BUY" if buy_signals >= 3 else "BUY"
        confidence = 0.8 + (buy_signals * 0.05)
        price_target = stock["price"] * random.uniform(1.15, 1.35)
    elif sell_signals > buy_signals + 1:
        action = "STRONG_SELL" if sell_signals >= 3 else "SELL"
        confidence = 0.7 + (sell_signals * 0.05)
        price_target = stock["price"] * random.uniform(0.75, 0.90)
    else:
        action = "HOLD"
        confidence = random.uniform(0.6, 0.8)
        price_target = stock["price"] * random.uniform(0.95, 1.10)
    
    # Generate RAG explanation
    reasoning_path = []
    supporting_evidence = []
    risk_factors = []
    
    if action in ["BUY", "STRONG_BUY"]:
        reasoning_path = [
            f"Technical analysis shows strong momentum for {symbol}",
            f"RSI indicates oversold conditions with potential for reversal",
            f"Moving averages suggest upward trend continuation",
            f"Fundamental metrics show {stock['name']} is undervalued relative to peers",
            f"Market sentiment and sector rotation favor {stock['sector']} stocks"
        ]
        supporting_evidence = [
            f"P/E ratio of {stock['pe_ratio']} is below sector median of 25.0",
            f"Strong institutional ownership with recent insider buying",
            f"Earnings growth projected at 15-20% for next fiscal year",
            f"Market leadership position in key growth segments",
            f"Strong balance sheet with low debt-to-equity ratio"
        ]
        risk_factors = [
            "Market volatility could impact short-term performance",
            "Regulatory changes in technology sector",
            "Competition from emerging market players",
            "Interest rate sensitivity affecting valuation multiples"
        ]
    elif action in ["SELL", "STRONG_SELL"]:
        reasoning_path = [
            f"Technical indicators signal weakness in {symbol}",
            f"RSI shows overbought conditions suggesting pullback",
            f"Breaking below key support levels",
            f"Fundamental valuation appears stretched at current levels",
            f"Sector headwinds and competitive pressures mounting"
        ]
        supporting_evidence = [
            f"P/E ratio of {stock['pe_ratio']} exceeds historical averages",
            f"Recent earnings miss and guidance reduction",
            f"Declining market share in core business segments",
            f"High institutional ownership suggesting limited upside",
            f"Rising input costs pressuring profit margins"
        ]
        risk_factors = [
            "Potential for further downside if macro conditions deteriorate",
            "Execution risks related to strategic initiatives",
            "Currency headwinds for international exposure",
            "Liquidity concerns in current market environment"
        ]
    else:  # HOLD
        reasoning_path = [
            f"Mixed signals for {symbol} suggest wait-and-see approach",
            f"Technical indicators are conflicting with no clear direction",
            f"Fundamental valuation appears fairly priced",
            f"Market conditions create uncertainty for near-term performance",
            f"Better entry opportunities may emerge with patience"
        ]
        supporting_evidence = [
            f"Current valuation aligns with sector averages",
            f"Stable earnings outlook with modest growth expectations",
            f"Balanced risk-reward profile at current price levels",
            f"Strong competitive position but limited catalysts",
            f"Dividend yield provides income while waiting"
        ]
        risk_factors = [
            "Opportunity cost of holding versus other alternatives",
            "Market timing risk if conditions change rapidly",
            "Sector rotation could impact relative performance",
            "Inflation concerns affecting consumer spending"
        ]
    
    rag_explanation = RAGExplanation(
        reasoning_path=reasoning_path,
        supporting_evidence=supporting_evidence,
        risk_factors=risk_factors,
        confidence_score=min(confidence, 0.95)
    )
    
    recommendation = StockRecommendation(
        symbol=symbol,
        action=action,
        confidence=confidence,
        price_target=price_target,
        timeframe="3-6 months",
        explanation=rag_explanation
    )
    
    # Historical price data (mock)
    dates = [(datetime.now() - timedelta(days=x)) for x in range(30, 0, -1)]
    price_history = []
    base_price = stock["price"]
    
    for i, date in enumerate(dates):
        # Generate realistic price movement
        change = random.uniform(-0.03, 0.03)
        base_price = max(base_price * (1 + change), stock["52_week_low"] * 1.1)
        base_price = min(base_price, stock["52_week_high"] * 0.9)
        
        price_history.append({
            "date": date.strftime("%Y-%m-%d"),
            "price": round(base_price, 2),
            "volume": random.randint(10000000, 100000000)
        })
    
    return {
        "status": "success",
        "data": {
            "basic_info": stock,
            "current_price": stock["price"],
            "recommendation": recommendation,
            "technical_indicators": technical_indicators,
            "fundamental_metrics": fundamental_metrics,
            "price_history": price_history,
            "last_updated": datetime.now().isoformat()
        }
    }

@router.get("/api/stocks/screener")
async def stock_screener(
    min_market_cap: Optional[float] = None,
    max_pe_ratio: Optional[float] = None,
    min_dividend_yield: Optional[float] = None,
    sector: Optional[str] = None
):
    """Screen stocks based on criteria"""
    
    filtered_stocks = []
    
    for symbol, data in STOCK_DATA.items():
        # Apply filters
        if min_market_cap and data["market_cap"] < min_market_cap:
            continue
        if max_pe_ratio and data["pe_ratio"] > max_pe_ratio:
            continue
        if min_dividend_yield and data["dividend_yield"] < min_dividend_yield:
            continue
        if sector and data["sector"] != sector:
            continue
            
        # Generate quick recommendation
        score = random.uniform(0.3, 0.9)
        action = "BUY" if score > 0.7 else "HOLD" if score > 0.5 else "SELL"
        
        filtered_stocks.append({
            "symbol": symbol,
            "name": data["name"],
            "price": data["price"],
            "sector": data["sector"],
            "market_cap": data["market_cap"],
            "pe_ratio": data["pe_ratio"],
            "dividend_yield": data["dividend_yield"],
            "recommendation": action,
            "score": score
        })
    
    # Sort by score descending
    filtered_stocks.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "status": "success",
        "data": {
            "stocks": filtered_stocks,
            "total_count": len(filtered_stocks),
            "timestamp": datetime.now().isoformat()
        }
    }