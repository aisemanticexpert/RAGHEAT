from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.options_trading_engine import OptionsStrategy, OptionsSignal, OptionAction, SignalStrength

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/options", tags=["Options Trading"])

# Initialize the options strategy engine
options_engine = OptionsStrategy()

# In-memory storage for signals (in production, use Redis or database)
active_signals: Dict[str, OptionsSignal] = {}
signal_history: List[OptionsSignal] = []

class OptionsRequest(BaseModel):
    symbol: str
    heat_score: Optional[float] = None

class BulkOptionsRequest(BaseModel):
    symbols: List[str]
    heat_scores: Optional[Dict[str, float]] = {}

class OptionsResponse(BaseModel):
    symbol: str
    timestamp: str
    action: str
    strength: str
    strike_price: float
    expiration: str
    probability: float
    heat_score: float
    pathrag_reasoning: str
    technical_indicators: Dict
    risk_reward: Dict
    market_context: Dict

class NotificationSettings(BaseModel):
    min_probability: float = 0.7
    min_heat_score: float = 0.6
    actions: List[str] = ["BUY_CALL", "BUY_PUT"]
    symbols: List[str] = ["SPY", "QQQ", "SOXS"]

# Global notification settings
notification_settings = NotificationSettings()

@router.post("/signal", response_model=OptionsResponse)
async def generate_options_signal(request: OptionsRequest):
    """Generate real-time options trading signal for a symbol"""
    try:
        signal = options_engine.generate_options_signal(
            symbol=request.symbol,
            heat_score=request.heat_score
        )
        
        # Store active signal
        active_signals[request.symbol] = signal
        
        # Add to history
        signal_history.append(signal)
        
        # Keep only last 100 signals in history
        if len(signal_history) > 100:
            signal_history.pop(0)
        
        return OptionsResponse(
            symbol=signal.symbol,
            timestamp=signal.timestamp.isoformat(),
            action=signal.action.value,
            strength=signal.strength.value,
            strike_price=signal.strike_price,
            expiration=signal.expiration,
            probability=signal.probability,
            heat_score=signal.heat_score,
            pathrag_reasoning=signal.pathrag_reasoning,
            technical_indicators=signal.technical_indicators,
            risk_reward=signal.risk_reward,
            market_context=signal.market_context
        )
        
    except Exception as e:
        logger.error(f"Error generating options signal for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk-signals", response_model=List[OptionsResponse])
async def generate_bulk_options_signals(request: BulkOptionsRequest):
    """Generate options signals for multiple symbols"""
    try:
        signals = []
        
        for symbol in request.symbols:
            heat_score = request.heat_scores.get(symbol) if request.heat_scores else None
            
            signal = options_engine.generate_options_signal(
                symbol=symbol,
                heat_score=heat_score
            )
            
            # Store active signal
            active_signals[symbol] = signal
            signal_history.append(signal)
            
            signals.append(OptionsResponse(
                symbol=signal.symbol,
                timestamp=signal.timestamp.isoformat(),
                action=signal.action.value,
                strength=signal.strength.value,
                strike_price=signal.strike_price,
                expiration=signal.expiration,
                probability=signal.probability,
                heat_score=signal.heat_score,
                pathrag_reasoning=signal.pathrag_reasoning,
                technical_indicators=signal.technical_indicators,
                risk_reward=signal.risk_reward,
                market_context=signal.market_context
            ))
        
        # Keep history manageable
        if len(signal_history) > 100:
            signal_history[:] = signal_history[-100:]
        
        return signals
        
    except Exception as e:
        logger.error(f"Error generating bulk options signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active-signals", response_model=Dict[str, OptionsResponse])
async def get_active_signals():
    """Get all currently active options signals"""
    try:
        result = {}
        for symbol, signal in active_signals.items():
            result[symbol] = OptionsResponse(
                symbol=signal.symbol,
                timestamp=signal.timestamp.isoformat(),
                action=signal.action.value,
                strength=signal.strength.value,
                strike_price=signal.strike_price,
                expiration=signal.expiration,
                probability=signal.probability,
                heat_score=signal.heat_score,
                pathrag_reasoning=signal.pathrag_reasoning,
                technical_indicators=signal.technical_indicators,
                risk_reward=signal.risk_reward,
                market_context=signal.market_context
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting active signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signal-history", response_model=List[OptionsResponse])
async def get_signal_history(limit: int = 50):
    """Get recent options signals history"""
    try:
        recent_signals = signal_history[-limit:] if signal_history else []
        
        result = []
        for signal in recent_signals:
            result.append(OptionsResponse(
                symbol=signal.symbol,
                timestamp=signal.timestamp.isoformat(),
                action=signal.action.value,
                strength=signal.strength.value,
                strike_price=signal.strike_price,
                expiration=signal.expiration,
                probability=signal.probability,
                heat_score=signal.heat_score,
                pathrag_reasoning=signal.pathrag_reasoning,
                technical_indicators=signal.technical_indicators,
                risk_reward=signal.risk_reward,
                market_context=signal.market_context
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting signal history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hot-signals")
async def get_hot_signals():
    """Get high-probability options signals for immediate attention"""
    try:
        hot_signals = []
        
        for signal in active_signals.values():
            if (signal.probability >= notification_settings.min_probability and 
                signal.heat_score >= notification_settings.min_heat_score and
                signal.action.value in notification_settings.actions):
                
                hot_signals.append({
                    'symbol': signal.symbol,
                    'action': signal.action.value,
                    'strength': signal.strength.value,
                    'probability': signal.probability,
                    'heat_score': signal.heat_score,
                    'strike_price': signal.strike_price,
                    'expiration': signal.expiration,
                    'pathrag_reasoning': signal.pathrag_reasoning,
                    'risk_reward_ratio': signal.risk_reward.get('ratio', 0),
                    'timestamp': signal.timestamp.isoformat()
                })
        
        # Sort by probability and heat score
        hot_signals.sort(key=lambda x: (x['probability'], x['heat_score']), reverse=True)
        
        return {
            'count': len(hot_signals),
            'signals': hot_signals[:10],  # Top 10 hot signals
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting hot signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/notification-settings")
async def update_notification_settings(settings: NotificationSettings):
    """Update notification settings for options alerts"""
    try:
        global notification_settings
        notification_settings = settings
        
        return {
            'status': 'success',
            'message': 'Notification settings updated',
            'settings': {
                'min_probability': settings.min_probability,
                'min_heat_score': settings.min_heat_score,
                'actions': settings.actions,
                'symbols': settings.symbols
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating notification settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-hours")
async def get_market_hours():
    """Get current market status and hours"""
    try:
        now = datetime.now()
        is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
        current_hour = now.hour
        
        # US market hours: 9:30 AM - 4:00 PM ET
        is_market_open = is_weekday and 9 <= current_hour < 16
        
        # Pre-market: 4:00 AM - 9:30 AM ET
        is_premarket = is_weekday and 4 <= current_hour < 9
        
        # After-hours: 4:00 PM - 8:00 PM ET
        is_afterhours = is_weekday and 16 <= current_hour < 20
        
        status = 'CLOSED'
        if is_market_open:
            status = 'OPEN'
        elif is_premarket:
            status = 'PREMARKET'
        elif is_afterhours:
            status = 'AFTERHOURS'
        
        return {
            'status': status,
            'is_market_open': is_market_open,
            'is_trading_day': is_weekday,
            'current_time': now.isoformat(),
            'next_open': 'Next trading day 9:30 AM ET' if not is_weekday else '9:30 AM ET',
            'market_close': '4:00 PM ET' if is_market_open else None
        }
        
    except Exception as e:
        logger.error(f"Error getting market hours: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance-metrics")
async def get_performance_metrics():
    """Get performance metrics for options signals"""
    try:
        if not signal_history:
            return {
                'total_signals': 0,
                'accuracy': 0,
                'avg_probability': 0,
                'avg_heat_score': 0,
                'action_distribution': {},
                'symbol_distribution': {}
            }
        
        # Calculate metrics from signal history
        total_signals = len(signal_history)
        
        probabilities = [s.probability for s in signal_history]
        heat_scores = [s.heat_score for s in signal_history]
        
        avg_probability = sum(probabilities) / len(probabilities)
        avg_heat_score = sum(heat_scores) / len(heat_scores)
        
        # Action distribution
        action_counts = {}
        for signal in signal_history:
            action = signal.action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Symbol distribution
        symbol_counts = {}
        for signal in signal_history:
            symbol = signal.symbol
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # High probability signals
        high_prob_signals = len([s for s in signal_history if s.probability >= 0.7])
        
        return {
            'total_signals': total_signals,
            'high_probability_signals': high_prob_signals,
            'high_prob_rate': high_prob_signals / total_signals if total_signals > 0 else 0,
            'avg_probability': avg_probability,
            'avg_heat_score': avg_heat_score,
            'action_distribution': action_counts,
            'symbol_distribution': symbol_counts,
            'last_24h_signals': len([s for s in signal_history if (datetime.now() - s.timestamp).days == 0])
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task to continuously monitor and update signals
async def monitor_options_signals():
    """Background task to continuously monitor options for all tracked symbols"""
    while True:
        try:
            # Monitor key symbols
            symbols = notification_settings.symbols
            
            for symbol in symbols:
                try:
                    signal = options_engine.generate_options_signal(symbol)
                    active_signals[symbol] = signal
                    
                    # Add to history if it's a significant signal
                    if (signal.probability >= 0.6 or 
                        signal.heat_score >= 0.6 or 
                        signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]):
                        signal_history.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error monitoring {symbol}: {e}")
                    continue
            
            # Cleanup old history
            if len(signal_history) > 200:
                signal_history[:] = signal_history[-150:]
            
            # Wait 30 seconds before next update
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in options monitoring: {e}")
            await asyncio.sleep(60)  # Wait longer on error