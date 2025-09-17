"""
Live Options Signal API Routes
Real-time options trading signals with WebSocket streaming
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json
from dataclasses import asdict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

try:
    from services.live_options_signal_generator import (
        LiveOptionsSignalGenerator,
        LiveOptionsSignal,
        SignalAlert,
        SignalStrength,
        SignalType,
        live_signal_generator
    )
    from services.signal_streaming_service import SignalStreamManager, signal_stream_manager
except ImportError:
    from live_options_signal_generator import (
        LiveOptionsSignalGenerator,
        LiveOptionsSignal,
        SignalAlert,
        SignalStrength,
        SignalType,
        live_signal_generator
    )
    from signal_streaming_service import SignalStreamManager, signal_stream_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/live-signals", tags=["Live Options Signals"])

# Pydantic models for API responses
class LiveSignalResponse(BaseModel):
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
    expiration_suggestion: str
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

class SignalStreamStats(BaseModel):
    is_streaming: bool
    active_connections: int
    total_connections_served: int
    signals_sent: int
    last_signal_count: int
    stream_interval_seconds: int

class ClientFilter(BaseModel):
    min_priority: Optional[int] = 5
    min_win_probability: Optional[float] = 0.85
    symbols: Optional[List[str]] = []
    signal_types: Optional[List[str]] = []
    max_signals: Optional[int] = 20

# WebSocket endpoint for real-time signal streaming
@router.websocket("/stream")
async def signal_websocket(websocket: WebSocket, client_id: Optional[str] = None):
    """WebSocket endpoint for real-time options signal streaming"""
    
    actual_client_id = None
    
    try:
        # Connect client to streaming service
        actual_client_id = await signal_stream_manager.connect_client(websocket, client_id)
        logger.info(f"WebSocket client {actual_client_id} connected for signal streaming")
        
        # Start streaming if not already active
        if not signal_stream_manager.is_streaming:
            await signal_stream_manager.start_streaming()
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages (filters, commands, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle filter updates
                if message.get("type") == "update_filters":
                    filters = message.get("filters", {})
                    await signal_stream_manager.update_client_filters(actual_client_id, filters)
                
                # Handle ping/pong for connection health
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message from {actual_client_id}: {e}")
                break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {actual_client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {actual_client_id}: {e}")
    finally:
        if actual_client_id:
            await signal_stream_manager.disconnect_client(actual_client_id)

@router.get("/current", response_model=List[LiveSignalResponse])
async def get_current_signals(
    min_priority: int = 5,
    min_win_probability: float = 0.85,
    max_results: int = 20,
    symbol: Optional[str] = None
):
    """Get current active live options signals"""
    
    try:
        # Generate fresh signals
        signals = await live_signal_generator.generate_live_signals()
        
        # Apply filters
        filtered_signals = signals.copy()
        
        # Priority filter
        filtered_signals = [s for s in filtered_signals if s.priority >= min_priority]
        
        # Win probability filter
        filtered_signals = [s for s in filtered_signals if s.win_probability >= min_win_probability]
        
        # Symbol filter
        if symbol:
            filtered_signals = [s for s in filtered_signals if s.symbol.upper() == symbol.upper()]
        
        # Limit results
        filtered_signals = filtered_signals[:max_results]
        
        # Convert to response format
        response_signals = []
        for signal in filtered_signals:
            response_signals.append(LiveSignalResponse(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                sector=signal.sector,
                signal_type=signal.signal_type.value,
                strength=signal.strength.value,
                strategy=signal.strategy,
                entry_price_low=signal.entry_price_range[0],
                entry_price_high=signal.entry_price_range[1],
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                expiration_suggestion=signal.expiration_suggestion,
                win_probability=signal.win_probability,
                heat_score=signal.heat_score,
                confidence_score=signal.confidence_score,
                risk_reward_ratio=signal.risk_reward_ratio,
                priority=signal.priority,
                generated_at=signal.generated_at.isoformat(),
                valid_until=signal.valid_until.isoformat(),
                entry_signals=signal.entry_signals,
                risk_factors=signal.risk_factors,
                suggested_position_size=signal.suggested_position_size,
                expected_move=signal.expected_move
            ))
        
        return response_signals
        
    except Exception as e:
        logger.error(f"Error getting current signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ultra-strong", response_model=List[LiveSignalResponse])
async def get_ultra_strong_signals():
    """Get only ultra-strong signals (95%+ probability)"""
    
    try:
        ultra_strong = await live_signal_generator.get_ultra_strong_signals()
        
        response_signals = []
        for signal in ultra_strong:
            response_signals.append(LiveSignalResponse(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                sector=signal.sector,
                signal_type=signal.signal_type.value,
                strength=signal.strength.value,
                strategy=signal.strategy,
                entry_price_low=signal.entry_price_range[0],
                entry_price_high=signal.entry_price_range[1],
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                expiration_suggestion=signal.expiration_suggestion,
                win_probability=signal.win_probability,
                heat_score=signal.heat_score,
                confidence_score=signal.confidence_score,
                risk_reward_ratio=signal.risk_reward_ratio,
                priority=signal.priority,
                generated_at=signal.generated_at.isoformat(),
                valid_until=signal.valid_until.isoformat(),
                entry_signals=signal.entry_signals,
                risk_factors=signal.risk_factors,
                suggested_position_size=signal.suggested_position_size,
                expected_move=signal.expected_move
            ))
        
        return response_signals
        
    except Exception as e:
        logger.error(f"Error getting ultra-strong signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/by-symbol/{symbol}", response_model=List[LiveSignalResponse])
async def get_signals_for_symbol(symbol: str):
    """Get all active signals for a specific symbol"""
    
    try:
        all_signals = await live_signal_generator.get_active_signals(min_priority=1)
        symbol_signals = [s for s in all_signals if s.symbol.upper() == symbol.upper()]
        
        response_signals = []
        for signal in symbol_signals:
            response_signals.append(LiveSignalResponse(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                sector=signal.sector,
                signal_type=signal.signal_type.value,
                strength=signal.strength.value,
                strategy=signal.strategy,
                entry_price_low=signal.entry_price_range[0],
                entry_price_high=signal.entry_price_range[1],
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                expiration_suggestion=signal.expiration_suggestion,
                win_probability=signal.win_probability,
                heat_score=signal.heat_score,
                confidence_score=signal.confidence_score,
                risk_reward_ratio=signal.risk_reward_ratio,
                priority=signal.priority,
                generated_at=signal.generated_at.isoformat(),
                valid_until=signal.valid_until.isoformat(),
                entry_signals=signal.entry_signals,
                risk_factors=signal.risk_factors,
                suggested_position_size=signal.suggested_position_size,
                expected_move=signal.expected_move
            ))
        
        return response_signals
        
    except Exception as e:
        logger.error(f"Error getting signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/heat-analysis")
async def get_heat_analysis():
    """Get detailed heat analysis for current market conditions"""
    
    try:
        # Generate signals to get heat analysis
        signals = await live_signal_generator.generate_live_signals()
        
        if not signals:
            return {
                "message": "No signals generated",
                "market_heat_status": "COLD",
                "total_signals": 0
            }
        
        # Analyze heat distribution
        heat_scores = [s.heat_score for s in signals]
        win_probabilities = [s.win_probability for s in signals]
        
        # Heat analysis
        avg_heat = sum(heat_scores) / len(heat_scores)
        max_heat = max(heat_scores)
        high_heat_count = len([h for h in heat_scores if h > 0.8])
        
        # Market heat status
        if avg_heat > 0.8:
            market_status = "BLAZING_HOT"
        elif avg_heat > 0.7:
            market_status = "HOT"
        elif avg_heat > 0.6:
            market_status = "WARM"
        elif avg_heat > 0.4:
            market_status = "COOL"
        else:
            market_status = "COLD"
        
        # Top heat sectors
        sector_heat = {}
        for signal in signals:
            if signal.sector not in sector_heat:
                sector_heat[signal.sector] = []
            sector_heat[signal.sector].append(signal.heat_score)
        
        sector_avg_heat = {
            sector: sum(scores) / len(scores)
            for sector, scores in sector_heat.items()
        }
        
        top_heat_sectors = sorted(sector_avg_heat.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "market_heat_status": market_status,
            "average_heat_score": round(avg_heat, 3),
            "maximum_heat_score": round(max_heat, 3),
            "high_heat_signals": high_heat_count,
            "total_signals": len(signals),
            "average_win_probability": round(sum(win_probabilities) / len(win_probabilities), 3),
            "ultra_strong_signals": len([s for s in signals if s.strength == SignalStrength.ULTRA_STRONG]),
            "top_heat_sectors": [{"sector": sector, "avg_heat": round(heat, 3)} for sector, heat in top_heat_sectors],
            "heat_distribution": {
                "blazing_hot": len([h for h in heat_scores if h > 0.9]),
                "hot": len([h for h in heat_scores if 0.8 < h <= 0.9]),
                "warm": len([h for h in heat_scores if 0.6 < h <= 0.8]),
                "cool": len([h for h in heat_scores if 0.4 < h <= 0.6]),
                "cold": len([h for h in heat_scores if h <= 0.4])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting heat analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-streaming")
async def start_signal_streaming(background_tasks: BackgroundTasks):
    """Start the live signal streaming service"""
    
    try:
        if signal_stream_manager.is_streaming:
            return {
                "status": "already_running",
                "message": "Signal streaming is already active",
                "active_connections": len(signal_stream_manager.active_connections)
            }
        
        background_tasks.add_task(signal_stream_manager.start_streaming)
        
        return {
            "status": "started",
            "message": "Live signal streaming started",
            "stream_interval_seconds": signal_stream_manager.stream_interval
        }
        
    except Exception as e:
        logger.error(f"Error starting signal streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-streaming")
async def stop_signal_streaming():
    """Stop the live signal streaming service"""
    
    try:
        await signal_stream_manager.stop_streaming()
        
        return {
            "status": "stopped",
            "message": "Live signal streaming stopped"
        }
        
    except Exception as e:
        logger.error(f"Error stopping signal streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/streaming-stats", response_model=SignalStreamStats)
async def get_streaming_stats():
    """Get streaming service statistics"""
    
    try:
        stats = signal_stream_manager.get_streaming_stats()
        
        return SignalStreamStats(
            is_streaming=stats["is_streaming"],
            active_connections=stats["active_connections"],
            total_connections_served=stats["total_connections_served"],
            signals_sent=stats["signals_sent"],
            last_signal_count=stats["last_signal_count"],
            stream_interval_seconds=stats["stream_interval_seconds"]
        )
        
    except Exception as e:
        logger.error(f"Error getting streaming stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/manual-alert")
async def send_manual_alert(
    symbol: str,
    message: str,
    urgency: str = "MEDIUM",
    client_ids: Optional[List[str]] = None
):
    """Send a manual alert to connected clients"""
    
    try:
        # Create a mock signal for the alert (in production, would use real signal)
        from services.live_options_signal_generator import LiveOptionsSignal, SignalType, SignalStrength
        from datetime import datetime, timedelta
        import uuid
        
        mock_signal = LiveOptionsSignal(
            signal_id=f"MANUAL_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            sector="Manual",
            signal_type=SignalType.BULLISH_CALL,
            strength=SignalStrength.MODERATE,
            strategy="manual",
            entry_price_range=(100.0, 102.0),
            target_price=105.0,
            stop_loss=98.0,
            expiration_suggestion="Manual alert",
            win_probability=0.0,
            heat_score=0.0,
            confidence_score=0.0,
            risk_reward_ratio=0.0,
            rsi=50.0,
            bollinger_position=0.5,
            volume_surge=1.0,
            volatility_rank=0.5,
            sector_momentum=0.0,
            news_sentiment=0.0,
            catalyst_potential=0.0,
            earnings_proximity=999,
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=1),
            priority=5,
            entry_signals=[],
            risk_factors=[],
            suggested_position_size=0.02,
            max_loss_per_contract=50.0,
            expected_move=2.0,
            implied_volatility_rank=0.5
        )
        
        alert = SignalAlert(
            alert_id=f"MANUAL_{int(datetime.now().timestamp())}",
            signal=mock_signal,
            alert_type="MANUAL",
            message=message,
            urgency=urgency,
            timestamp=datetime.now()
        )
        
        await signal_stream_manager.send_manual_alert(alert, client_ids)
        
        return {
            "status": "sent",
            "message": f"Manual alert sent for {symbol}",
            "recipients": len(client_ids) if client_ids else len(signal_stream_manager.active_connections)
        }
        
    except Exception as e:
        logger.error(f"Error sending manual alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signal-types")
async def get_available_signal_types():
    """Get all available signal types and strengths for filtering"""
    
    return {
        "signal_types": [signal_type.value for signal_type in SignalType],
        "signal_strengths": [strength.value for strength in SignalStrength],
        "strategies": ["call", "put", "straddle", "iron_condor"]
    }