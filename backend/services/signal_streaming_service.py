"""
Real-time Signal Streaming Service
WebSocket-based live options signal streaming with real-time updates
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
from dataclasses import asdict
import websockets
from websockets.exceptions import ConnectionClosed
from fastapi import WebSocket, WebSocketDisconnect
import uuid

try:
    from services.live_options_signal_generator import (
        LiveOptionsSignalGenerator, 
        LiveOptionsSignal, 
        SignalAlert,
        SignalStrength,
        SignalType,
        live_signal_generator
    )
except ImportError:
    from live_options_signal_generator import (
        LiveOptionsSignalGenerator, 
        LiveOptionsSignal, 
        SignalAlert,
        SignalStrength,
        SignalType,
        live_signal_generator
    )

logger = logging.getLogger(__name__)

class SignalStreamManager:
    """Manages WebSocket connections for real-time signal streaming"""
    
    def __init__(self):
        # Active WebSocket connections
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_filters: Dict[str, Dict] = {}  # Client-specific filters
        
        # Signal tracking
        self.last_signals: List[LiveOptionsSignal] = []
        self.signal_alerts: List[SignalAlert] = []
        
        # Streaming configuration
        self.stream_interval = 5  # seconds
        self.is_streaming = False
        self.stream_task = None
        
        # Performance tracking
        self.signals_sent = 0
        self.connections_served = 0
    
    async def connect_client(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """Connect a new client to the signal stream"""
        
        if not client_id:
            client_id = f"client_{uuid.uuid4().hex[:8]}"
        
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_filters[client_id] = {
            'min_priority': 5,
            'min_win_probability': 0.85,
            'symbols': [],  # Empty means all symbols
            'signal_types': [],  # Empty means all types
            'max_signals': 20
        }
        
        self.connections_served += 1
        logger.info(f"Client {client_id} connected. Active connections: {len(self.active_connections)}")
        
        # Send initial signals
        await self._send_initial_signals(client_id)
        
        return client_id
    
    async def disconnect_client(self, client_id: str):
        """Disconnect a client from the signal stream"""
        
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].close()
            except:
                pass  # Connection might already be closed
            
            del self.active_connections[client_id]
            del self.connection_filters[client_id]
            
            logger.info(f"Client {client_id} disconnected. Active connections: {len(self.active_connections)}")
    
    async def update_client_filters(self, client_id: str, filters: Dict):
        """Update filtering preferences for a client"""
        
        if client_id in self.connection_filters:
            self.connection_filters[client_id].update(filters)
            logger.info(f"Updated filters for client {client_id}: {filters}")
            
            # Send filtered signals immediately
            await self._send_filtered_signals(client_id)
    
    async def start_streaming(self):
        """Start the real-time signal streaming"""
        
        if self.is_streaming:
            logger.warning("Signal streaming already active")
            return
        
        self.is_streaming = True
        self.stream_task = asyncio.create_task(self._streaming_loop())
        logger.info(f"Signal streaming started with {self.stream_interval}s interval")
    
    async def stop_streaming(self):
        """Stop the real-time signal streaming"""
        
        self.is_streaming = False
        
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Signal streaming stopped")
    
    async def _streaming_loop(self):
        """Main streaming loop that generates and broadcasts signals"""
        
        logger.info("Starting signal streaming loop...")
        
        while self.is_streaming:
            try:
                # Generate new signals
                start_time = datetime.now()
                new_signals = await live_signal_generator.generate_live_signals()
                generation_time = (datetime.now() - start_time).total_seconds()
                
                if new_signals:
                    logger.info(f"Generated {len(new_signals)} signals in {generation_time:.2f}s")
                    
                    # Detect signal changes and create alerts
                    alerts = self._detect_signal_changes(new_signals)
                    
                    # Update stored signals
                    self.last_signals = new_signals
                    self.signal_alerts.extend(alerts)
                    
                    # Broadcast to all connected clients
                    await self._broadcast_signals(new_signals, alerts)
                
                # Wait for next interval
                await asyncio.sleep(self.stream_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                await asyncio.sleep(self.stream_interval)
        
        logger.info("Signal streaming loop ended")
    
    def _detect_signal_changes(self, new_signals: List[LiveOptionsSignal]) -> List[SignalAlert]:
        """Detect changes in signals and generate alerts"""
        
        alerts = []
        current_time = datetime.now()
        
        # Create lookup for previous signals
        prev_signals = {s.symbol: s for s in self.last_signals}
        new_signals_dict = {s.symbol: s for s in new_signals}
        
        # Detect new signals
        for symbol, signal in new_signals_dict.items():
            if symbol not in prev_signals:
                alerts.append(SignalAlert(
                    alert_id=f"NEW_{signal.signal_id}",
                    signal=signal,
                    alert_type="NEW",
                    message=f"🚨 NEW {signal.strength.value} {signal.signal_type.value} signal for {symbol}",
                    urgency="HIGH" if signal.strength in [SignalStrength.STRONG, SignalStrength.ULTRA_STRONG] else "MEDIUM",
                    timestamp=current_time
                ))
        
        # Detect strength changes
        for symbol in set(prev_signals.keys()) & set(new_signals_dict.keys()):
            prev_signal = prev_signals[symbol]
            new_signal = new_signals_dict[symbol]
            
            if prev_signal.strength != new_signal.strength:
                direction = "UPGRADED" if new_signal.priority > prev_signal.priority else "DOWNGRADED"
                alerts.append(SignalAlert(
                    alert_id=f"UPDATE_{new_signal.signal_id}",
                    signal=new_signal,
                    alert_type="UPDATE",
                    message=f"📈 Signal {direction}: {symbol} now {new_signal.strength.value}",
                    urgency="MEDIUM",
                    timestamp=current_time
                ))
        
        # Detect disappeared signals (closed positions)
        for symbol in set(prev_signals.keys()) - set(new_signals_dict.keys()):
            prev_signal = prev_signals[symbol]
            alerts.append(SignalAlert(
                alert_id=f"CLOSE_{prev_signal.signal_id}",
                signal=prev_signal,
                alert_type="CLOSE",
                message=f"🔒 Signal closed: {symbol} - Monitor for exit",
                urgency="LOW",
                timestamp=current_time
            ))
        
        return alerts
    
    async def _broadcast_signals(self, signals: List[LiveOptionsSignal], alerts: List[SignalAlert]):
        """Broadcast signals to all connected clients with their filters"""
        
        if not self.active_connections:
            return
        
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                # Apply client-specific filters
                filtered_signals = self._apply_client_filters(signals, client_id)
                filtered_alerts = self._apply_alert_filters(alerts, client_id)
                
                if filtered_signals or filtered_alerts:
                    message = {
                        "type": "signal_update",
                        "timestamp": datetime.now().isoformat(),
                        "signals": [self._serialize_signal(s) for s in filtered_signals],
                        "alerts": [self._serialize_alert(a) for a in filtered_alerts],
                        "metadata": {
                            "total_signals": len(signals),
                            "filtered_count": len(filtered_signals),
                            "client_id": client_id
                        }
                    }
                    
                    await websocket.send_text(json.dumps(message))
                    self.signals_sent += 1
                
            except ConnectionClosed:
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect_client(client_id)
    
    def _apply_client_filters(self, signals: List[LiveOptionsSignal], client_id: str) -> List[LiveOptionsSignal]:
        """Apply client-specific filters to signals"""
        
        filters = self.connection_filters.get(client_id, {})
        filtered = signals.copy()
        
        # Priority filter
        min_priority = filters.get('min_priority', 0)
        filtered = [s for s in filtered if s.priority >= min_priority]
        
        # Win probability filter
        min_win_prob = filters.get('min_win_probability', 0)
        filtered = [s for s in filtered if s.win_probability >= min_win_prob]
        
        # Symbol filter
        symbols = filters.get('symbols', [])
        if symbols:
            filtered = [s for s in filtered if s.symbol in symbols]
        
        # Signal type filter
        signal_types = filters.get('signal_types', [])
        if signal_types:
            filtered = [s for s in filtered if s.signal_type.value in signal_types]
        
        # Limit results
        max_signals = filters.get('max_signals', 50)
        filtered = filtered[:max_signals]
        
        return filtered
    
    def _apply_alert_filters(self, alerts: List[SignalAlert], client_id: str) -> List[SignalAlert]:
        """Apply client-specific filters to alerts"""
        
        filters = self.connection_filters.get(client_id, {})
        filtered = alerts.copy()
        
        # Symbol filter
        symbols = filters.get('symbols', [])
        if symbols:
            filtered = [a for a in filtered if a.signal.symbol in symbols]
        
        return filtered
    
    def _serialize_signal(self, signal: LiveOptionsSignal) -> Dict:
        """Serialize signal for JSON transmission"""
        
        signal_dict = asdict(signal)
        
        # Convert enums to strings
        signal_dict['signal_type'] = signal.signal_type.value
        signal_dict['strength'] = signal.strength.value
        
        # Convert datetime objects
        signal_dict['generated_at'] = signal.generated_at.isoformat()
        signal_dict['valid_until'] = signal.valid_until.isoformat()
        
        # Format entry price range
        signal_dict['entry_price_range'] = list(signal.entry_price_range)
        
        return signal_dict
    
    def _serialize_alert(self, alert: SignalAlert) -> Dict:
        """Serialize alert for JSON transmission"""
        
        return {
            "alert_id": alert.alert_id,
            "signal_id": alert.signal.signal_id,
            "symbol": alert.signal.symbol,
            "alert_type": alert.alert_type,
            "message": alert.message,
            "urgency": alert.urgency,
            "timestamp": alert.timestamp.isoformat()
        }
    
    async def _send_initial_signals(self, client_id: str):
        """Send current active signals to a newly connected client"""
        
        if not self.last_signals:
            # Generate initial signals if none exist
            self.last_signals = await live_signal_generator.generate_live_signals()
        
        filtered_signals = self._apply_client_filters(self.last_signals, client_id)
        
        if filtered_signals:
            message = {
                "type": "initial_signals",
                "timestamp": datetime.now().isoformat(),
                "signals": [self._serialize_signal(s) for s in filtered_signals],
                "metadata": {
                    "total_signals": len(self.last_signals),
                    "filtered_count": len(filtered_signals),
                    "client_id": client_id
                }
            }
            
            websocket = self.active_connections[client_id]
            await websocket.send_text(json.dumps(message))
    
    async def _send_filtered_signals(self, client_id: str):
        """Send filtered signals to a specific client after filter update"""
        
        if client_id not in self.active_connections:
            return
        
        filtered_signals = self._apply_client_filters(self.last_signals, client_id)
        
        message = {
            "type": "filtered_update",
            "timestamp": datetime.now().isoformat(),
            "signals": [self._serialize_signal(s) for s in filtered_signals],
            "metadata": {
                "total_signals": len(self.last_signals),
                "filtered_count": len(filtered_signals),
                "filters_applied": self.connection_filters[client_id]
            }
        }
        
        websocket = self.active_connections[client_id]
        await websocket.send_text(json.dumps(message))
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming service statistics"""
        
        return {
            "is_streaming": self.is_streaming,
            "active_connections": len(self.active_connections),
            "total_connections_served": self.connections_served,
            "signals_sent": self.signals_sent,
            "last_signal_count": len(self.last_signals),
            "stream_interval_seconds": self.stream_interval,
            "uptime_minutes": 0  # Would track actual uptime
        }
    
    async def send_manual_alert(self, alert: SignalAlert, client_ids: Optional[List[str]] = None):
        """Send a manual alert to specific clients or all clients"""
        
        target_clients = client_ids or list(self.active_connections.keys())
        
        for client_id in target_clients:
            if client_id in self.active_connections:
                try:
                    message = {
                        "type": "manual_alert",
                        "timestamp": datetime.now().isoformat(),
                        "alert": self._serialize_alert(alert)
                    }
                    
                    await self.active_connections[client_id].send_text(json.dumps(message))
                    
                except Exception as e:
                    logger.error(f"Error sending manual alert to {client_id}: {e}")

# Global streaming manager
signal_stream_manager = SignalStreamManager()