"""
Market Hours Detection and API Switching Service
Detects when markets are open/closed and switches between real-time and synthetic data
"""

import asyncio
from datetime import datetime, time, timezone, timedelta
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import pytz
import holidays
import json
import logging
from dataclasses import dataclass, asdict

# Import our synthetic generator
from data_pipeline.synthetic_market_generator import get_synthetic_market_data, get_synthetic_neo4j_data

logger = logging.getLogger(__name__)


class MarketStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    HOLIDAY = "holiday"
    WEEKEND = "weekend"


@dataclass
class MarketSession:
    """Market trading session information"""
    name: str
    start_time: time
    end_time: time
    timezone: str
    active: bool = True


@dataclass 
class MarketHoursInfo:
    """Complete market hours information"""
    status: MarketStatus
    current_time: datetime
    market_timezone: str
    next_open: Optional[datetime]
    next_close: Optional[datetime]
    session_info: Optional[Dict[str, Any]]
    is_holiday: bool
    holiday_name: Optional[str]
    time_until_open: Optional[timedelta]
    time_until_close: Optional[timedelta]


class MarketHoursDetector:
    """
    Comprehensive market hours detection with holiday support
    """
    
    def __init__(self):
        self.market_timezone = pytz.timezone('America/New_York')  # NYSE/NASDAQ timezone
        self.us_holidays = holidays.UnitedStates()
        
        # Define trading sessions
        self.sessions = {
            'pre_market': MarketSession(
                name='Pre-Market',
                start_time=time(4, 0),   # 4:00 AM ET
                end_time=time(9, 30),    # 9:30 AM ET
                timezone='America/New_York'
            ),
            'regular': MarketSession(
                name='Regular Trading',
                start_time=time(9, 30),  # 9:30 AM ET
                end_time=time(16, 0),    # 4:00 PM ET
                timezone='America/New_York'
            ),
            'after_hours': MarketSession(
                name='After Hours',
                start_time=time(16, 0),  # 4:00 PM ET
                end_time=time(20, 0),    # 8:00 PM ET
                timezone='America/New_York'
            )
        }
        
        # Cache for market status to avoid excessive calculations
        self._last_status_check = None
        self._cached_status = None
        self._cache_duration = timedelta(minutes=1)
        
    def get_current_market_time(self) -> datetime:
        """Get current time in market timezone"""
        return datetime.now(self.market_timezone)
    
    def is_trading_day(self, date: datetime) -> Tuple[bool, Optional[str]]:
        """Check if given date is a trading day"""
        # Check if weekend
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False, "Weekend"
        
        # Check if holiday
        date_only = date.date()
        if date_only in self.us_holidays:
            holiday_name = self.us_holidays.get(date_only)
            return False, holiday_name
        
        return True, None
    
    def get_session_for_time(self, market_time: datetime) -> Tuple[Optional[str], MarketSession]:
        """Get the trading session for a given market time"""
        current_time = market_time.time()
        
        for session_name, session in self.sessions.items():
            if session.start_time <= current_time < session.end_time:
                return session_name, session
        
        return None, None
    
    def calculate_next_market_open(self, current_time: datetime) -> datetime:
        """Calculate next market opening time"""
        # Start with today
        check_date = current_time.date()
        
        # Look up to 10 days ahead to find next trading day
        for _ in range(10):
            check_datetime = self.market_timezone.localize(
                datetime.combine(check_date, self.sessions['regular'].start_time)
            )
            
            is_trading, _ = self.is_trading_day(check_datetime)
            
            # If it's a trading day and we haven't passed market open
            if is_trading and check_datetime > current_time:
                return check_datetime
            
            # Move to next day
            check_date += timedelta(days=1)
        
        # Fallback - shouldn't happen in normal circumstances
        return current_time + timedelta(days=1)
    
    def calculate_next_market_close(self, current_time: datetime) -> datetime:
        """Calculate next market closing time"""
        # Check if market is currently open today
        today_close = self.market_timezone.localize(
            datetime.combine(current_time.date(), self.sessions['regular'].end_time)
        )
        
        is_trading, _ = self.is_trading_day(current_time)
        
        # If today is trading day and we haven't passed close
        if is_trading and today_close > current_time:
            return today_close
        
        # Otherwise find next trading day's close
        next_open = self.calculate_next_market_open(current_time)
        next_close = self.market_timezone.localize(
            datetime.combine(next_open.date(), self.sessions['regular'].end_time)
        )
        
        return next_close
    
    def get_market_status(self, force_refresh: bool = False) -> MarketHoursInfo:
        """Get comprehensive market status information"""
        current_time = self.get_current_market_time()
        
        # Use cache if available and not forced to refresh
        if not force_refresh and self._cached_status and self._last_status_check:
            if current_time - self._last_status_check < self._cache_duration:
                return self._cached_status
        
        # Check if it's a trading day
        is_trading, holiday_name = self.is_trading_day(current_time)
        
        if not is_trading:
            if current_time.weekday() >= 5:
                status = MarketStatus.WEEKEND
            else:
                status = MarketStatus.HOLIDAY
                
            next_open = self.calculate_next_market_open(current_time)
            time_until_open = next_open - current_time
            
            market_info = MarketHoursInfo(
                status=status,
                current_time=current_time,
                market_timezone=str(self.market_timezone),
                next_open=next_open,
                next_close=None,
                session_info=None,
                is_holiday=status == MarketStatus.HOLIDAY,
                holiday_name=holiday_name,
                time_until_open=time_until_open,
                time_until_close=None
            )
        else:
            # It's a trading day, determine session
            session_name, session = self.get_session_for_time(current_time)
            
            if session_name == 'regular':
                status = MarketStatus.OPEN
                next_close = self.calculate_next_market_close(current_time)
                time_until_close = next_close - current_time
                time_until_open = None
            elif session_name == 'pre_market':
                status = MarketStatus.PRE_MARKET
                next_open = self.market_timezone.localize(
                    datetime.combine(current_time.date(), self.sessions['regular'].start_time)
                )
                next_close = self.calculate_next_market_close(current_time)
                time_until_open = next_open - current_time
                time_until_close = next_close - current_time
            elif session_name == 'after_hours':
                status = MarketStatus.AFTER_HOURS
                next_open = self.calculate_next_market_open(current_time)
                time_until_open = next_open - current_time
                time_until_close = None
            else:
                # Market is closed during trading day (between sessions)
                status = MarketStatus.CLOSED
                next_open = self.calculate_next_market_open(current_time)
                time_until_open = next_open - current_time
                time_until_close = None
            
            session_info = asdict(session) if session else None
            
            market_info = MarketHoursInfo(
                status=status,
                current_time=current_time,
                market_timezone=str(self.market_timezone),
                next_open=next_open if status != MarketStatus.OPEN else None,
                next_close=next_close if status == MarketStatus.OPEN else None,
                session_info=session_info,
                is_holiday=False,
                holiday_name=None,
                time_until_open=time_until_open,
                time_until_close=time_until_close
            )
        
        # Cache the result
        self._cached_status = market_info
        self._last_status_check = current_time
        
        return market_info
    
    def should_use_synthetic_data(self) -> bool:
        """Determine if synthetic data should be used"""
        status = self.get_market_status()
        
        # Use synthetic data when markets are closed
        return status.status in [
            MarketStatus.CLOSED,
            MarketStatus.WEEKEND,
            MarketStatus.HOLIDAY
        ]
    
    def get_trading_session_summary(self) -> Dict[str, Any]:
        """Get summary of all trading sessions for today"""
        current_time = self.get_current_market_time()
        is_trading, holiday_name = self.is_trading_day(current_time)
        
        if not is_trading:
            return {
                "date": current_time.date().isoformat(),
                "is_trading_day": False,
                "reason": holiday_name if holiday_name else "Weekend",
                "sessions": []
            }
        
        sessions_info = []
        for name, session in self.sessions.items():
            start_dt = self.market_timezone.localize(
                datetime.combine(current_time.date(), session.start_time)
            )
            end_dt = self.market_timezone.localize(
                datetime.combine(current_time.date(), session.end_time)
            )
            
            is_active = start_dt <= current_time < end_dt
            
            sessions_info.append({
                "name": session.name,
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "is_active": is_active,
                "status": "active" if is_active else "inactive"
            })
        
        return {
            "date": current_time.date().isoformat(),
            "is_trading_day": True,
            "sessions": sessions_info,
            "current_time": current_time.isoformat()
        }


class DataSourceManager:
    """
    Manages switching between real-time and synthetic data sources
    """
    
    def __init__(self):
        self.market_detector = MarketHoursDetector()
        self.real_time_apis = {
            'market_overview': 'http://localhost:8001/api/live-data/market-overview',
            'heat_distribution': 'http://localhost:8001/api/heat/distribution',
            'neo4j_stocks': 'http://localhost:8001/api/streaming/neo4j/query/top-performers',
            'neo4j_sectors': 'http://localhost:8001/api/streaming/neo4j/query/sector-performance'
        }
        
    async def get_market_data(self, data_type: str = 'market_overview') -> Dict[str, Any]:
        """Get market data - real-time if markets open, synthetic if closed"""
        market_status = self.market_detector.get_market_status()
        use_synthetic = self.market_detector.should_use_synthetic_data()
        
        if use_synthetic:
            logger.info(f"🧪 Using synthetic data - Market Status: {market_status.status.value}")
            return await self._get_synthetic_data(data_type, market_status)
        else:
            logger.info(f"📡 Using real-time data - Market Status: {market_status.status.value}")
            return await self._get_real_time_data(data_type, market_status)
    
    async def _get_synthetic_data(self, data_type: str, market_status: MarketHoursInfo) -> Dict[str, Any]:
        """Get synthetic data with market status context"""
        if data_type in ['market_overview', 'heat_distribution']:
            data = get_synthetic_market_data()
        elif data_type in ['neo4j_stocks', 'neo4j_sectors']:
            data = get_synthetic_neo4j_data()
        else:
            # Fallback to market overview
            data = get_synthetic_market_data()
        
        # Add market context
        data['market_status'] = {
            'status': market_status.status.value,
            'data_source': 'synthetic',
            'market_closed_reason': market_status.holiday_name or market_status.status.value,
            'next_open': market_status.next_open.isoformat() if market_status.next_open else None,
            'time_until_open': str(market_status.time_until_open) if market_status.time_until_open else None
        }
        
        return data
    
    async def _get_real_time_data(self, data_type: str, market_status: MarketHoursInfo) -> Dict[str, Any]:
        """Get real-time data with fallback to synthetic"""
        try:
            import aiohttp
            
            api_url = self.real_time_apis.get(data_type)
            if not api_url:
                raise ValueError(f"Unknown data type: {data_type}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Add market context
                        data['market_status'] = {
                            'status': market_status.status.value,
                            'data_source': 'real_time',
                            'session_info': market_status.session_info,
                            'next_close': market_status.next_close.isoformat() if market_status.next_close else None,
                            'time_until_close': str(market_status.time_until_close) if market_status.time_until_close else None
                        }
                        
                        return data
                    else:
                        raise Exception(f"API returned status {response.status}")
        
        except Exception as e:
            logger.warning(f"⚠️ Real-time API failed: {e}. Falling back to synthetic data.")
            return await self._get_synthetic_data(data_type, market_status)
    
    def get_data_source_status(self) -> Dict[str, Any]:
        """Get current data source configuration and status"""
        market_status = self.market_detector.get_market_status()
        use_synthetic = self.market_detector.should_use_synthetic_data()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "market_status": {
                "status": market_status.status.value,
                "current_time": market_status.current_time.isoformat(),
                "is_trading_day": not market_status.is_holiday and market_status.status != MarketStatus.WEEKEND,
                "holiday_name": market_status.holiday_name,
                "next_open": market_status.next_open.isoformat() if market_status.next_open else None,
                "next_close": market_status.next_close.isoformat() if market_status.next_close else None,
                "time_until_open": str(market_status.time_until_open) if market_status.time_until_open else None,
                "time_until_close": str(market_status.time_until_close) if market_status.time_until_close else None
            },
            "data_source": {
                "active_source": "synthetic" if use_synthetic else "real_time",
                "reason": "Market is closed" if use_synthetic else "Market is open",
                "real_time_apis_available": len(self.real_time_apis),
                "synthetic_generator_active": True
            },
            "trading_sessions": self.market_detector.get_trading_session_summary()
        }


# Global instances
market_detector = MarketHoursDetector()
data_source_manager = DataSourceManager()


# Convenience functions
def get_market_hours_info() -> Dict[str, Any]:
    """Get current market hours information"""
    status = market_detector.get_market_status()
    return {
        "status": status.status.value,
        "current_time": status.current_time.isoformat(),
        "is_open": status.status == MarketStatus.OPEN,
        "next_open": status.next_open.isoformat() if status.next_open else None,
        "next_close": status.next_close.isoformat() if status.next_close else None,
        "session_info": status.session_info,
        "is_holiday": status.is_holiday,
        "holiday_name": status.holiday_name,
        "data_source_recommendation": "synthetic" if market_detector.should_use_synthetic_data() else "real_time"
    }


async def get_smart_market_data(data_type: str = 'market_overview') -> Dict[str, Any]:
    """Get market data with intelligent source selection"""
    return await data_source_manager.get_market_data(data_type)


if __name__ == "__main__":
    # Test the market hours detector
    detector = MarketHoursDetector()
    
    print("🕐 Market Hours Detection Test")
    print("=" * 50)
    
    status = detector.get_market_status()
    print(f"Current Status: {status.status.value}")
    print(f"Current Time (ET): {status.current_time}")
    print(f"Is Trading Day: {not status.is_holiday}")
    print(f"Use Synthetic Data: {detector.should_use_synthetic_data()}")
    
    if status.next_open:
        print(f"Next Market Open: {status.next_open}")
        print(f"Time Until Open: {status.time_until_open}")
    
    if status.next_close:
        print(f"Next Market Close: {status.next_close}")
        print(f"Time Until Close: {status.time_until_close}")
    
    # Test data source manager
    print("\n📊 Testing Data Source Manager")
    manager = DataSourceManager()
    source_status = manager.get_data_source_status()
    print(f"Active Data Source: {source_status['data_source']['active_source']}")
    print(f"Reason: {source_status['data_source']['reason']}")