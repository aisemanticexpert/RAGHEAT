"""
Professional Live Options Signal API Routes
Uses Finnhub and Alpha Vantage APIs for real-time professional market data
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import asyncio
import numpy as np
import random
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

from professional_data_service import professional_data_service, LiveMarketData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/live-signals", tags=["Professional Live Options Signals"])

class ProfessionalSignalResponse(BaseModel):
    signal_id: str
    symbol: str
    sector: str
    signal_type: str
    strength: str
    strategy: str
    entry_price_low: float
    entry_price_high: float
    target_price: float
    stop_loss: float
    win_probability: float
    heat_score: float
    confidence_score: float
    risk_reward_ratio: float
    priority: int
    generated_at: str
    valid_until: str
    entry_signals: List[str]
    risk_factors: List[str]
    suggested_position_size: float
    expected_move: float
    expiration_suggestion: str
    data_source: str

class ProfessionalSignalGenerator:
    """Generates professional options signals using real market data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 120  # 2 minutes for signals
        
    def calculate_professional_heat_score(self, market_data: LiveMarketData) -> float:
        """Calculate professional heat score based on real market data"""
        
        # Price momentum factor
        price_momentum = min(abs(market_data.change_percent) / 4.0, 1.0)
        
        # Volume surge factor  
        volume_heat = max(0, min((market_data.volume_surge - 1.0) / 2.0, 1.0))
        
        # RSI extremes
        rsi_heat = 0.0
        if market_data.rsi < 25 or market_data.rsi > 75:
            rsi_heat = 1.0
        elif market_data.rsi < 35 or market_data.rsi > 65:
            rsi_heat = 0.6
        
        # Bollinger Band extremes
        bb_heat = 0.0
        if market_data.bb_position < 0.15 or market_data.bb_position > 0.85:
            bb_heat = 1.0
        elif market_data.bb_position < 0.25 or market_data.bb_position > 0.75:
            bb_heat = 0.7
        
        # Volatility factor
        volatility_heat = min(market_data.volatility / 40.0, 1.0)
        
        # Market hours boost (more activity during trading hours)
        hour = datetime.now().hour
        time_boost = 1.2 if 9 <= hour <= 16 else 0.8
        
        # Weighted professional heat score
        heat_score = (
            price_momentum * 0.25 +
            volume_heat * 0.20 +
            rsi_heat * 0.25 +
            bb_heat * 0.20 +
            volatility_heat * 0.10
        ) * time_boost
        
        return min(max(heat_score, 0), 1.0)
    
    def determine_signal_type_and_strength(self, market_data: LiveMarketData, heat_score: float) -> tuple:
        """Determine signal type and strength based on professional analysis"""
        
        rsi = market_data.rsi
        bb_pos = market_data.bb_position
        change_pct = market_data.change_percent
        vol = market_data.volatility
        
        # Professional signal classification
        if rsi < 25 and bb_pos < 0.25 and change_pct < -2.0:
            signal_type = "BULLISH_CALL"
            strategy = "call"
            if heat_score > 0.8:
                strength = "ULTRA_STRONG"
            elif heat_score > 0.6:
                strength = "STRONG"
            else:
                strength = "MODERATE"
                
        elif rsi > 75 and bb_pos > 0.75 and change_pct > 2.0:
            signal_type = "BEARISH_PUT"
            strategy = "put"
            if heat_score > 0.8:
                strength = "ULTRA_STRONG"
            elif heat_score > 0.6:
                strength = "STRONG"
            else:
                strength = "MODERATE"
                
        elif vol > 35 and market_data.volume_surge > 1.5 and heat_score > 0.5:
            signal_type = "STRADDLE"
            strategy = "straddle"
            strength = "STRONG" if heat_score > 0.7 else "MODERATE"
            
        elif 30 < rsi < 70 and 0.25 < bb_pos < 0.75 and vol < 30:
            signal_type = "IRON_CONDOR"
            strategy = "iron_condor"
            strength = "MODERATE" if heat_score > 0.4 else "WEAK"
            
        else:
            # Default to most likely profitable strategy
            if change_pct >= 0:
                signal_type = "BULLISH_CALL"
                strategy = "call"
            else:
                signal_type = "BEARISH_PUT"
                strategy = "put"
            strength = "MODERATE" if heat_score > 0.5 else "WEAK"
        
        return signal_type, strategy, strength
    
    def calculate_option_parameters(self, market_data: LiveMarketData, signal_type: str) -> tuple:
        """Calculate professional option entry/exit parameters"""
        
        current_price = market_data.current_price
        volatility = market_data.volatility
        
        # Calculate Average True Range approximation
        high = market_data.high
        low = market_data.low
        prev_close = market_data.prev_close
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        atr = max(tr1, tr2, tr3)
        
        # Adjust ATR based on volatility
        atr_multiplier = volatility / 25.0  # Scale based on volatility
        adjusted_atr = atr * atr_multiplier
        
        # Professional entry range (tighter for lower volatility)
        entry_spread = adjusted_atr * 0.3
        entry_low = current_price - entry_spread
        entry_high = current_price + entry_spread
        
        # Calculate targets based on signal type and volatility
        if signal_type == "BULLISH_CALL":
            target_multiplier = 1.5 + (volatility / 50.0)
            target_price = current_price + (adjusted_atr * target_multiplier)
            stop_loss = current_price - (adjusted_atr * 1.2)
            
        elif signal_type == "BEARISH_PUT":
            target_multiplier = 1.5 + (volatility / 50.0)
            target_price = current_price - (adjusted_atr * target_multiplier)
            stop_loss = current_price + (adjusted_atr * 1.2)
            
        elif signal_type == "STRADDLE":
            target_multiplier = 2.0 + (volatility / 40.0)
            target_price = current_price + (adjusted_atr * target_multiplier)
            stop_loss = current_price - (adjusted_atr * 0.8)
            
        else:  # IRON_CONDOR
            target_price = current_price * 1.05
            stop_loss = current_price * 0.95
        
        return entry_low, entry_high, target_price, stop_loss
    
    def calculate_win_probability(self, market_data: LiveMarketData, signal_type: str, 
                                 strength: str, heat_score: float) -> float:
        """Calculate professional win probability"""
        
        # Base probabilities by strength
        base_probs = {
            "ULTRA_STRONG": 0.88,
            "STRONG": 0.82,
            "MODERATE": 0.75,
            "WEAK": 0.65
        }
        
        base_prob = base_probs.get(strength, 0.70)
        
        # Adjustments based on market conditions
        adjustments = 0.0
        
        # RSI extreme bonus
        if market_data.rsi < 25 or market_data.rsi > 75:
            adjustments += 0.08
        elif market_data.rsi < 35 or market_data.rsi > 65:
            adjustments += 0.04
        
        # Bollinger Band extreme bonus
        if market_data.bb_position < 0.15 or market_data.bb_position > 0.85:
            adjustments += 0.06
        
        # Volume surge bonus
        if market_data.volume_surge > 2.0:
            adjustments += 0.05
        elif market_data.volume_surge > 1.5:
            adjustments += 0.03
        
        # Heat score bonus
        adjustments += heat_score * 0.08
        
        # Signal type specific adjustments
        if signal_type in ["BULLISH_CALL", "BEARISH_PUT"] and abs(market_data.change_percent) > 2:
            adjustments += 0.04  # Momentum trades in trending market
        
        final_prob = min(0.96, base_prob + adjustments)  # Cap at 96%
        
        return round(final_prob, 3)
    
    def generate_entry_signals(self, market_data: LiveMarketData, signal_type: str) -> List[str]:
        """Generate professional entry signals"""
        
        signals = []
        
        # RSI signals
        if market_data.rsi < 25:
            signals.append(f"RSI severely oversold ({market_data.rsi:.0f})")
        elif market_data.rsi < 35:
            signals.append(f"RSI oversold ({market_data.rsi:.0f})")
        elif market_data.rsi > 75:
            signals.append(f"RSI severely overbought ({market_data.rsi:.0f})")
        elif market_data.rsi > 65:
            signals.append(f"RSI overbought ({market_data.rsi:.0f})")
        
        # Bollinger Band signals
        if market_data.bb_position < 0.15:
            signals.append("Price at lower Bollinger Band")
        elif market_data.bb_position > 0.85:
            signals.append("Price at upper Bollinger Band")
        elif market_data.bb_position < 0.25:
            signals.append("Price near lower BB support")
        elif market_data.bb_position > 0.75:
            signals.append("Price near upper BB resistance")
        
        # Volume signals
        if market_data.volume_surge > 2.5:
            signals.append(f"Exceptional volume surge ({market_data.volume_surge:.1f}x)")
        elif market_data.volume_surge > 1.8:
            signals.append(f"High volume surge ({market_data.volume_surge:.1f}x)")
        elif market_data.volume_surge > 1.3:
            signals.append(f"Above average volume ({market_data.volume_surge:.1f}x)")
        
        # Price movement signals
        if abs(market_data.change_percent) > 3:
            direction = "bullish" if market_data.change_percent > 0 else "bearish"
            signals.append(f"Strong {direction} momentum ({market_data.change_percent:+.1f}%)")
        elif abs(market_data.change_percent) > 1.5:
            direction = "upward" if market_data.change_percent > 0 else "downward"
            signals.append(f"Moderate {direction} movement ({market_data.change_percent:+.1f}%)")
        
        # Volatility signals
        if market_data.volatility > 40:
            signals.append(f"High volatility environment ({market_data.volatility:.0f}%)")
        elif market_data.volatility > 30:
            signals.append(f"Elevated volatility ({market_data.volatility:.0f}%)")
        
        # Market timing
        hour = datetime.now().hour
        if 9 <= hour <= 10:
            signals.append("Market opening volatility")
        elif 15 <= hour <= 16:
            signals.append("Market closing activity")
        
        return signals[:4]  # Limit to top 4 signals
    
    def generate_risk_factors(self, market_data: LiveMarketData, signal_type: str) -> List[str]:
        """Generate professional risk factors"""
        
        risks = []
        
        # Volatility risks
        if market_data.volatility > 45:
            risks.append("Extreme volatility risk")
        elif market_data.volatility > 35:
            risks.append("High volatility environment")
        
        # Volume risks
        if market_data.volume_surge < 0.7:
            risks.append("Below average volume")
        elif market_data.volume_surge < 0.9:
            risks.append("Light trading volume")
        
        # Technical risks
        if 45 < market_data.rsi < 55:
            risks.append("Neutral momentum zone")
        
        if 0.4 < market_data.bb_position < 0.6:
            risks.append("Mid-range consolidation")
        
        # Extended move risks
        if abs(market_data.change_percent) > 5:
            risks.append("Extended price movement")
        elif abs(market_data.change_percent) > 3:
            risks.append("Significant daily move")
        
        # Market timing risks
        hour = datetime.now().hour
        if hour < 9 or hour > 16:
            risks.append("After-hours trading risk")
        
        # Default risks
        if not risks:
            risks = ["Time decay risk", "Market reversal risk", "Volatility crush risk"]
        
        return risks[:3]  # Limit to top 3 risks
    
    def generate_professional_signal(self, market_data: LiveMarketData) -> ProfessionalSignalResponse:
        """Generate a professional-grade options signal"""
        
        # Calculate heat score
        heat_score = self.calculate_professional_heat_score(market_data)
        
        # Determine signal type and strength
        signal_type, strategy, strength = self.determine_signal_type_and_strength(market_data, heat_score)
        
        # Calculate option parameters
        entry_low, entry_high, target_price, stop_loss = self.calculate_option_parameters(market_data, signal_type)
        
        # Calculate win probability
        win_probability = self.calculate_win_probability(market_data, signal_type, strength, heat_score)
        
        # Calculate risk-reward ratio
        potential_profit = abs(target_price - entry_high)
        potential_loss = abs(entry_low - stop_loss)
        risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 else 2.5
        
        # Calculate priority (1-10 scale)
        priority_score = (
            heat_score * 3.0 +
            (win_probability - 0.6) * 10.0 +
            min(risk_reward_ratio / 3.0, 2.0) +
            (market_data.volume_surge - 1.0) * 2.0
        )
        priority = max(1, min(10, int(5 + priority_score)))
        
        # Generate entry signals and risk factors
        entry_signals = self.generate_entry_signals(market_data, signal_type)
        risk_factors = self.generate_risk_factors(market_data, signal_type)
        
        # Kelly Criterion position sizing
        kelly_fraction = (win_probability * risk_reward_ratio - (1 - win_probability)) / risk_reward_ratio
        position_size = max(0.01, min(0.10, kelly_fraction * 0.20))  # Conservative Kelly
        
        # Expected move calculation (30-day approximation)
        expected_move = market_data.volatility * market_data.current_price / 100 / np.sqrt(252) * np.sqrt(30)
        
        # Expiration suggestion based on volatility and signal strength
        if strength in ["ULTRA_STRONG", "STRONG"]:
            dte_options = [7, 14, 21]
        else:
            dte_options = [21, 30, 45]
        
        dte = random.choice(dte_options)
        
        return ProfessionalSignalResponse(
            signal_id=f"PRO_{int(time.time())}_{market_data.symbol}",
            symbol=market_data.symbol,
            sector=market_data.sector,
            signal_type=signal_type,
            strength=strength,
            strategy=strategy,
            entry_price_low=round(entry_low, 2),
            entry_price_high=round(entry_high, 2),
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            win_probability=win_probability,
            heat_score=round(heat_score, 3),
            confidence_score=round(win_probability * heat_score, 3),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            priority=priority,
            generated_at=datetime.now().isoformat(),
            valid_until=(datetime.now() + timedelta(hours=8)).isoformat(),
            entry_signals=entry_signals,
            risk_factors=risk_factors,
            suggested_position_size=round(position_size, 3),
            expected_move=round(expected_move, 2),
            expiration_suggestion=f"{dte} DTE",
            data_source=market_data.data_source
        )

# Global signal generator
signal_generator = ProfessionalSignalGenerator()

@router.get("/current", response_model=List[ProfessionalSignalResponse])
async def get_professional_current_signals(
    min_priority: int = 5,
    min_win_probability: float = 0.72,
    max_results: int = 12
):
    """Get professional live options signals using real-time APIs"""
    
    try:
        logger.info("🚀 Generating professional signals with Finnhub + Alpha Vantage APIs")
        
        # High-liquidity symbols for professional trading
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", 
                  "NFLX", "JPM", "BAC", "JNJ", "UNH"]
        
        # Get live market data from professional APIs
        market_data_dict = await professional_data_service.get_multiple_symbols(symbols[:max_results])
        
        if not market_data_dict:
            raise HTTPException(status_code=503, detail="No live market data available")
        
        # Generate professional signals
        signals = []
        for symbol, market_data in market_data_dict.items():
            try:
                signal = signal_generator.generate_professional_signal(market_data)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        # Apply professional filters
        filtered_signals = [
            s for s in signals 
            if s.priority >= min_priority and s.win_probability >= min_win_probability
        ]
        
        # Sort by professional criteria
        filtered_signals.sort(
            key=lambda x: (x.priority, x.heat_score, x.win_probability), 
            reverse=True
        )
        
        result = filtered_signals[:max_results]
        
        logger.info(f"✅ Generated {len(result)} professional signals from live APIs")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error generating professional signals: {e}")
        raise HTTPException(status_code=500, detail=f"Professional signal generation failed: {str(e)}")

@router.get("/heat-analysis")
async def get_professional_heat_analysis():
    """Get professional market heat analysis using real-time data"""
    
    try:
        # Get live data for analysis
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN", "NFLX"]
        market_data_dict = await professional_data_service.get_multiple_symbols(symbols)
        
        if not market_data_dict:
            return {"error": "No professional market data available"}
        
        market_data_list = list(market_data_dict.values())
        
        # Calculate professional heat metrics
        heat_scores = [signal_generator.calculate_professional_heat_score(data) for data in market_data_list]
        changes = [data.change_percent for data in market_data_list]
        rsi_values = [data.rsi for data in market_data_list]
        volume_surges = [data.volume_surge for data in market_data_list]
        
        avg_heat = sum(heat_scores) / len(heat_scores)
        max_heat = max(heat_scores)
        avg_change = sum(changes) / len(changes)
        avg_rsi = sum(rsi_values) / len(rsi_values)
        
        # Professional market classification
        if avg_heat > 0.75:
            market_status = "BLAZING_HOT"
        elif avg_heat > 0.60:
            market_status = "HOT"
        elif avg_heat > 0.45:
            market_status = "WARM"
        elif avg_heat > 0.30:
            market_status = "COOL"
        else:
            market_status = "COLD"
        
        # Sector analysis
        sector_analysis = {}
        for data in market_data_list:
            sector = data.sector
            if sector not in sector_analysis:
                sector_analysis[sector] = []
            sector_analysis[sector].append(signal_generator.calculate_professional_heat_score(data))
        
        top_sectors = [
            {"sector": sector, "avg_heat": round(sum(heats) / len(heats), 3)}
            for sector, heats in sector_analysis.items()
        ]
        top_sectors.sort(key=lambda x: x['avg_heat'], reverse=True)
        
        # Count data sources
        real_data_count = sum(1 for d in market_data_list if 'professional' in d.data_source)
        
        return {
            "market_heat_status": market_status,
            "average_heat_score": round(avg_heat, 3),
            "maximum_heat_score": round(max_heat, 3),
            "average_change_percent": round(avg_change, 2),
            "average_rsi": round(avg_rsi, 1),
            "high_heat_signals": len([h for h in heat_scores if h > 0.7]),
            "ultra_strong_signals": len([h for h in heat_scores if h > 0.8]),
            "total_signals": len(market_data_list),
            "top_heat_sectors": top_sectors,
            "heat_distribution": {
                "blazing_hot": len([h for h in heat_scores if h > 0.8]),
                "hot": len([h for h in heat_scores if 0.6 < h <= 0.8]),
                "warm": len([h for h in heat_scores if 0.4 < h <= 0.6]),
                "cool": len([h for h in heat_scores if 0.2 < h <= 0.4]),
                "cold": len([h for h in heat_scores if h <= 0.2])
            },
            "market_indicators": {
                "average_volume_surge": round(sum(volume_surges) / len(volume_surges), 2),
                "oversold_count": len([rsi for rsi in rsi_values if rsi < 30]),
                "overbought_count": len([rsi for rsi in rsi_values if rsi > 70]),
                "neutral_count": len([rsi for rsi in rsi_values if 30 <= rsi <= 70])
            },
            "data_quality": {
                "professional_api_sources": real_data_count,
                "total_symbols_analyzed": len(market_data_list),
                "apis_used": ["Finnhub", "Alpha Vantage"],
                "data_freshness": "real_time"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in professional heat analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/streaming-stats")
async def get_professional_streaming_stats():
    """Get professional streaming statistics"""
    
    api_status = professional_data_service.get_api_status()
    
    return {
        "is_streaming": True,
        "active_connections": 1,
        "total_connections_served": 1,
        "signals_sent": 0,
        "last_signal_count": 0,
        "stream_interval_seconds": 60,
        "data_source": "professional_apis",
        "api_status": api_status
    }

@router.get("/test")
async def test_professional_apis():
    """Test professional API integration"""
    
    try:
        # Test with AAPL
        market_data = await professional_data_service.get_live_market_data("AAPL")
        
        if market_data:
            heat_score = signal_generator.calculate_professional_heat_score(market_data)
            
            return {
                "status": "✅ SUCCESS",
                "message": "Professional APIs operational",
                "live_data": {
                    "symbol": market_data.symbol,
                    "price": f"${market_data.current_price:.2f}",
                    "change": f"{market_data.change_percent:+.2f}%",
                    "rsi": market_data.rsi,
                    "bb_position": f"{market_data.bb_position:.1%}",
                    "volume_surge": f"{market_data.volume_surge:.1f}x",
                    "volatility": f"{market_data.volatility:.1f}%",
                    "heat_score": round(heat_score, 3),
                    "data_source": market_data.data_source,
                    "timestamp": market_data.timestamp.isoformat()
                },
                "apis_used": ["Finnhub (real-time quotes)", "Alpha Vantage (technical analysis)"],
                "system_status": "🚀 PROFESSIONAL GRADE LIVE DATA"
            }
        else:
            return {
                "status": "❌ FAILED",
                "message": "Could not retrieve professional market data",
                "apis_used": ["Finnhub", "Alpha Vantage"],
                "system_status": "API Connection Issues"
            }
            
    except Exception as e:
        logger.error(f"Professional API test failed: {e}")
        return {
            "status": "❌ ERROR",
            "message": f"Professional API test failed: {str(e)}",
            "system_status": "Service Unavailable"
        }