"""
Live Options Signal Generator
Real-time options trading signals based on heat analysis and comprehensive algorithms
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import json
import numpy as np
from enum import Enum
import uuid

try:
    from services.market_orchestrator import MarketOrchestrator, market_orchestrator
    from services.sector_performance_analyzer import SectorPerformance
    from services.options_analyzer import OptionsOpportunity
    from services.news_sentiment_analyzer import SentimentAnalysis
    from services.dynamic_graph_builder import DynamicGraphBuilder
except ImportError:
    from market_orchestrator import MarketOrchestrator, market_orchestrator
    from sector_performance_analyzer import SectorPerformance
    from options_analyzer import OptionsOpportunity
    from news_sentiment_analyzer import SentimentAnalysis
    from dynamic_graph_builder import DynamicGraphBuilder

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    ULTRA_STRONG = "ULTRA_STRONG"

class SignalType(Enum):
    BULLISH_CALL = "BULLISH_CALL"
    BEARISH_PUT = "BEARISH_PUT"
    VOLATILITY_STRADDLE = "VOLATILITY_STRADDLE"
    EARNINGS_PLAY = "EARNINGS_PLAY"
    BREAKOUT_MOMENTUM = "BREAKOUT_MOMENTUM"
    REVERSAL_CONTRARIAN = "REVERSAL_CONTRARIAN"

@dataclass
class LiveOptionsSignal:
    signal_id: str
    symbol: str
    sector: str
    signal_type: SignalType
    strength: SignalStrength
    strategy: str  # call, put, straddle
    entry_price_range: Tuple[float, float]
    target_price: float
    stop_loss: float
    expiration_suggestion: str
    win_probability: float
    heat_score: float
    confidence_score: float
    risk_reward_ratio: float
    
    # Technical indicators
    rsi: float
    bollinger_position: float
    volume_surge: float
    volatility_rank: float
    
    # Market context
    sector_momentum: float
    news_sentiment: float
    catalyst_potential: float
    earnings_proximity: int
    
    # Signal metadata
    generated_at: datetime
    valid_until: datetime
    priority: int  # 1-10, 10 being highest
    entry_signals: List[str]
    risk_factors: List[str]
    
    # Execution details
    suggested_position_size: float  # % of portfolio
    max_loss_per_contract: float
    expected_move: float
    implied_volatility_rank: float

@dataclass
class SignalAlert:
    alert_id: str
    signal: LiveOptionsSignal
    alert_type: str  # "NEW", "UPDATE", "CLOSE", "WARNING"
    message: str
    urgency: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    timestamp: datetime

class HeatBasedSignalScorer:
    """Advanced scoring system that combines heat analysis with all algorithms"""
    
    def __init__(self):
        self.heat_weights = {
            'price_momentum': 0.25,
            'volume_heat': 0.20,
            'volatility_heat': 0.15,
            'sentiment_heat': 0.15,
            'sector_heat': 0.10,
            'technical_heat': 0.10,
            'catalyst_heat': 0.05
        }
    
    def calculate_comprehensive_heat_score(self, 
                                         stock_data: Dict,
                                         sector_data: SectorPerformance,
                                         sentiment_data: Optional[SentimentAnalysis],
                                         options_data: OptionsOpportunity) -> float:
        """Calculate multi-dimensional heat score"""
        
        heat_components = {}
        
        # 1. Price Momentum Heat
        price_change = stock_data.get('change_percent', 0)
        momentum_score = min(abs(price_change) / 5.0, 1.0)  # Normalize to 5% max
        heat_components['price_momentum'] = momentum_score
        
        # 2. Volume Heat (unusual volume activity)
        volume_ratio = stock_data.get('volume_ratio', 1.0)
        volume_heat = min((volume_ratio - 1.0) / 2.0, 1.0) if volume_ratio > 1 else 0
        heat_components['volume_heat'] = max(0, volume_heat)
        
        # 3. Volatility Heat (expanding volatility)
        vol_rank = options_data.volatility_rank
        volatility_heat = vol_rank if vol_rank > 0.7 else vol_rank * 0.5  # Premium for high vol
        heat_components['volatility_heat'] = volatility_heat
        
        # 4. Sentiment Heat
        if sentiment_data:
            sentiment_intensity = abs(sentiment_data.overall_sentiment) * sentiment_data.sentiment_strength
            sentiment_heat = min(sentiment_intensity, 1.0)
        else:
            sentiment_heat = 0.1  # Neutral baseline
        heat_components['sentiment_heat'] = sentiment_heat
        
        # 5. Sector Heat
        sector_performance = max(0, sector_data.performance_score / 10.0)  # Normalize
        sector_heat = min(sector_performance, 1.0)
        heat_components['sector_heat'] = sector_heat
        
        # 6. Technical Heat (RSI extremes, BB position)
        rsi = stock_data.get('rsi', 50)
        bb_position = stock_data.get('bb_position', 0.5)
        
        # RSI heat (extremes are hot)
        rsi_heat = max((rsi - 70) / 30, (30 - rsi) / 30, 0) if rsi > 70 or rsi < 30 else 0
        
        # BB heat (extremes are hot)
        bb_heat = max((bb_position - 0.8) / 0.2, (0.2 - bb_position) / 0.2, 0)
        
        technical_heat = (rsi_heat + bb_heat) / 2
        heat_components['technical_heat'] = technical_heat
        
        # 7. Catalyst Heat (earnings, news events)
        catalyst_heat = 0
        if sentiment_data:
            catalyst_heat = sentiment_data.catalyst_potential
        
        # Earnings proximity boost
        earnings_days = options_data.earnings_date
        if earnings_days and isinstance(earnings_days, str) and "days" in earnings_days:
            try:
                days = int(earnings_days.split()[0])
                if days <= 7:
                    catalyst_heat += 0.3  # Strong catalyst
                elif days <= 14:
                    catalyst_heat += 0.2  # Moderate catalyst
            except:
                pass
        
        heat_components['catalyst_heat'] = min(catalyst_heat, 1.0)
        
        # Calculate weighted heat score
        total_heat = sum(
            score * self.heat_weights[component]
            for component, score in heat_components.items()
        )
        
        logger.debug(f"Heat components for {options_data.symbol}: {heat_components}")
        logger.debug(f"Total heat score: {total_heat:.3f}")
        
        return min(total_heat, 1.0)

class LiveOptionsSignalGenerator:
    """Main signal generation engine"""
    
    def __init__(self):
        self.orchestrator = market_orchestrator
        self.heat_scorer = HeatBasedSignalScorer()
        self.active_signals = {}  # signal_id -> LiveOptionsSignal
        self.signal_history = []
        self.last_analysis_time = None
        self.graph_builder = DynamicGraphBuilder()
        
        # Signal thresholds
        self.min_heat_score = 0.6  # Minimum heat for signal generation
        self.min_win_probability = 0.85  # 85% minimum win probability
        self.ultra_strong_threshold = 0.95  # 95%+ for ultra strong signals
        
    async def generate_live_signals(self) -> List[LiveOptionsSignal]:
        """Generate comprehensive live options signals"""
        
        try:
            logger.info("Starting live signal generation...")
            start_time = datetime.now()
            
            # Get comprehensive market analysis
            analysis = await self.orchestrator.run_comprehensive_analysis()
            
            signals = []
            
            # Process each high-probability opportunity
            for opp in analysis.high_probability_opportunities:
                
                # Get additional data for signal generation
                sector_data = self._find_sector_data(opp.sector, analysis.top_sectors + analysis.worst_sectors)
                sentiment_data = analysis.sentiment_analysis.get(opp.symbol)
                
                # Create enhanced stock data (mock - would come from real data service)
                stock_data = await self._get_enhanced_stock_data(opp.symbol)
                
                # Calculate comprehensive heat score
                heat_score = self.heat_scorer.calculate_comprehensive_heat_score(
                    stock_data, sector_data, sentiment_data, opp
                )
                
                # Generate signal if criteria met
                if heat_score >= self.min_heat_score and opp.win_probability >= self.min_win_probability:
                    signal = await self._create_live_signal(
                        opp, sector_data, sentiment_data, stock_data, heat_score
                    )
                    if signal:
                        signals.append(signal)
            
            # Sort signals by priority (heat × probability × confidence)
            signals.sort(key=lambda s: s.heat_score * s.win_probability * s.confidence_score, reverse=True)
            
            # Update active signals
            self._update_active_signals(signals)
            
            # Store in graph database
            await self._store_signals_in_graph(signals)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Generated {len(signals)} live signals in {execution_time:.2f}s")
            
            return signals[:20]  # Return top 20 signals
            
        except Exception as e:
            logger.error(f"Error generating live signals: {e}")
            return []
    
    async def _get_enhanced_stock_data(self, symbol: str) -> Dict:
        """Get enhanced stock data with technical indicators"""
        # Mock implementation - would integrate with real data service
        import random
        
        return {
            'symbol': symbol,
            'current_price': 150 + random.uniform(-10, 10),
            'change_percent': random.uniform(-5, 5),
            'volume_ratio': random.uniform(0.5, 3.0),
            'rsi': random.uniform(20, 80),
            'bb_position': random.uniform(0, 1),
            'support_level': 145,
            'resistance_level': 160,
            'atr': random.uniform(2, 8),
            'beta': random.uniform(0.5, 2.0)
        }
    
    def _find_sector_data(self, sector: str, all_sectors: List[SectorPerformance]) -> Optional[SectorPerformance]:
        """Find sector data from analysis results"""
        for s in all_sectors:
            if s.sector == sector:
                return s
        return None
    
    async def _create_live_signal(self, 
                                 opp: OptionsOpportunity,
                                 sector_data: SectorPerformance,
                                 sentiment_data: Optional[SentimentAnalysis],
                                 stock_data: Dict,
                                 heat_score: float) -> Optional[LiveOptionsSignal]:
        """Create a comprehensive live options signal"""
        
        try:
            # Determine signal type based on strategy and conditions
            signal_type = self._determine_signal_type(opp, sector_data, sentiment_data)
            
            # Calculate signal strength
            strength = self._calculate_signal_strength(opp.win_probability, heat_score, opp.confidence_score)
            
            # Calculate entry prices and targets
            entry_range, target_price, stop_loss = self._calculate_price_targets(opp, stock_data)
            
            # Generate expiration suggestion
            expiration = self._suggest_expiration(opp, signal_type)
            
            # Calculate position sizing
            position_size = self._calculate_position_size(opp, heat_score, strength)
            
            # Risk assessment
            risk_factors = self._assess_risk_factors(opp, stock_data, sector_data)
            
            # Priority calculation (1-10)
            priority = self._calculate_priority(heat_score, opp.win_probability, strength)
            
            signal = LiveOptionsSignal(
                signal_id=f"LIVE_{opp.symbol}_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                symbol=opp.symbol,
                sector=opp.sector,
                signal_type=signal_type,
                strength=strength,
                strategy=opp.strategy,
                entry_price_range=entry_range,
                target_price=target_price,
                stop_loss=stop_loss,
                expiration_suggestion=expiration,
                win_probability=opp.win_probability,
                heat_score=heat_score,
                confidence_score=opp.confidence_score,
                risk_reward_ratio=opp.risk_reward_ratio,
                
                # Technical indicators
                rsi=stock_data.get('rsi', 50),
                bollinger_position=stock_data.get('bb_position', 0.5),
                volume_surge=opp.volume_surge,
                volatility_rank=opp.volatility_rank,
                
                # Market context
                sector_momentum=sector_data.momentum if sector_data else 0,
                news_sentiment=sentiment_data.overall_sentiment if sentiment_data else 0,
                catalyst_potential=sentiment_data.catalyst_potential if sentiment_data else 0,
                earnings_proximity=self._extract_earnings_days(opp.earnings_date),
                
                # Signal metadata
                generated_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=4),  # Valid for 4 hours
                priority=priority,
                entry_signals=opp.entry_signals.copy(),
                risk_factors=risk_factors,
                
                # Execution details
                suggested_position_size=position_size,
                max_loss_per_contract=entry_range[1] * 0.5,  # 50% max loss
                expected_move=opp.expected_move,
                implied_volatility_rank=opp.volatility_rank
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating signal for {opp.symbol}: {e}")
            return None
    
    def _determine_signal_type(self, opp: OptionsOpportunity, 
                              sector_data: SectorPerformance,
                              sentiment_data: Optional[SentimentAnalysis]) -> SignalType:
        """Determine the type of options signal"""
        
        # Earnings proximity check
        if self._extract_earnings_days(opp.earnings_date) <= 7:
            return SignalType.EARNINGS_PLAY
        
        # Volatility expansion for straddles
        if opp.strategy == "straddle":
            return SignalType.VOLATILITY_STRADDLE
        
        # Momentum vs reversal
        if sector_data and abs(sector_data.momentum) > 2.0:
            if opp.strategy == "call" and sector_data.momentum > 0:
                return SignalType.BREAKOUT_MOMENTUM
            elif opp.strategy == "put" and sector_data.momentum < 0:
                return SignalType.BREAKOUT_MOMENTUM
        
        # Strong sentiment divergence (contrarian)
        if sentiment_data and abs(sentiment_data.overall_sentiment) > 0.7:
            if (opp.strategy == "put" and sentiment_data.overall_sentiment > 0.5) or \
               (opp.strategy == "call" and sentiment_data.overall_sentiment < -0.5):
                return SignalType.REVERSAL_CONTRARIAN
        
        # Default directional signals
        if opp.strategy == "call":
            return SignalType.BULLISH_CALL
        else:
            return SignalType.BEARISH_PUT
    
    def _calculate_signal_strength(self, win_prob: float, heat_score: float, confidence: float) -> SignalStrength:
        """Calculate overall signal strength"""
        combined_score = (win_prob * 0.5) + (heat_score * 0.3) + (confidence * 0.2)
        
        if combined_score >= 0.95:
            return SignalStrength.ULTRA_STRONG
        elif combined_score >= 0.90:
            return SignalStrength.STRONG
        elif combined_score >= 0.80:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _calculate_price_targets(self, opp: OptionsOpportunity, stock_data: Dict) -> Tuple[Tuple[float, float], float, float]:
        """Calculate entry range, target, and stop loss prices"""
        
        current_price = stock_data.get('current_price', opp.target_price)
        atr = stock_data.get('atr', current_price * 0.02)  # 2% ATR default
        
        if opp.strategy == "call":
            # Call option targets
            entry_low = current_price + (atr * 0.5)
            entry_high = current_price + (atr * 1.0)
            target = current_price * 1.05  # 5% target
            stop_loss = current_price * 0.98  # 2% stop
            
        elif opp.strategy == "put":
            # Put option targets
            entry_high = current_price - (atr * 0.5)
            entry_low = current_price - (atr * 1.0)
            target = current_price * 0.95  # 5% target down
            stop_loss = current_price * 1.02  # 2% stop up
            
        else:  # straddle
            # Straddle targets
            entry_low = current_price - (atr * 0.5)
            entry_high = current_price + (atr * 0.5)
            target = current_price  # ATM target
            stop_loss = current_price  # Managed differently for straddles
        
        return (entry_low, entry_high), target, stop_loss
    
    def _suggest_expiration(self, opp: OptionsOpportunity, signal_type: SignalType) -> str:
        """Suggest optimal expiration based on signal type"""
        
        if signal_type == SignalType.EARNINGS_PLAY:
            return "1-2 weeks (post-earnings)"
        elif signal_type == SignalType.VOLATILITY_STRADDLE:
            return "2-4 weeks (vol expansion)"
        elif signal_type == SignalType.BREAKOUT_MOMENTUM:
            return "2-6 weeks (momentum follow-through)"
        elif signal_type == SignalType.REVERSAL_CONTRARIAN:
            return "1-3 weeks (quick reversal)"
        else:
            return "2-4 weeks (standard)"
    
    def _calculate_position_size(self, opp: OptionsOpportunity, heat_score: float, strength: SignalStrength) -> float:
        """Calculate suggested position size as % of portfolio"""
        
        base_size = 0.02  # 2% base position
        
        # Adjust based on signal strength
        strength_multiplier = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MODERATE: 0.75,
            SignalStrength.STRONG: 1.0,
            SignalStrength.ULTRA_STRONG: 1.5
        }
        
        # Adjust based on win probability
        prob_multiplier = min(opp.win_probability / 0.85, 2.0)  # Cap at 2x
        
        # Adjust based on heat score
        heat_multiplier = min(heat_score / 0.6, 1.5)  # Cap at 1.5x
        
        suggested_size = base_size * strength_multiplier[strength] * prob_multiplier * heat_multiplier
        
        return min(suggested_size, 0.08)  # Cap at 8% of portfolio
    
    def _assess_risk_factors(self, opp: OptionsOpportunity, stock_data: Dict, sector_data: SectorPerformance) -> List[str]:
        """Assess and list risk factors"""
        
        risks = []
        
        # Market risks
        if sector_data and sector_data.volatility > 30:
            risks.append("High sector volatility")
        
        # Technical risks
        rsi = stock_data.get('rsi', 50)
        if rsi > 80:
            risks.append("Overbought conditions (RSI > 80)")
        elif rsi < 20:
            risks.append("Oversold conditions (RSI < 20)")
        
        # Volume risks
        if opp.volume_surge < 1.2:
            risks.append("Below average volume confirmation")
        
        # Time decay risks
        if opp.time_decay_factor > 0.08:
            risks.append("High time decay risk")
        
        # Earnings risks
        earnings_days = self._extract_earnings_days(opp.earnings_date)
        if 1 <= earnings_days <= 3:
            risks.append("Immediate earnings risk")
        
        # Volatility risks
        if opp.volatility_rank > 0.9:
            risks.append("Extremely high implied volatility")
        elif opp.volatility_rank < 0.2:
            risks.append("Very low implied volatility")
        
        return risks
    
    def _calculate_priority(self, heat_score: float, win_prob: float, strength: SignalStrength) -> int:
        """Calculate signal priority (1-10)"""
        
        base_score = (heat_score * 0.4) + (win_prob * 0.6)
        
        strength_bonus = {
            SignalStrength.WEAK: 0,
            SignalStrength.MODERATE: 1,
            SignalStrength.STRONG: 2,
            SignalStrength.ULTRA_STRONG: 3
        }
        
        priority = int((base_score * 7) + strength_bonus[strength])
        return min(max(priority, 1), 10)
    
    def _extract_earnings_days(self, earnings_date: Optional[str]) -> int:
        """Extract days to earnings from string"""
        if not earnings_date or "days" not in earnings_date:
            return 999  # No earnings info
        
        try:
            return int(earnings_date.split()[0])
        except:
            return 999
    
    def _update_active_signals(self, new_signals: List[LiveOptionsSignal]):
        """Update active signals tracking"""
        current_time = datetime.now()
        
        # Remove expired signals
        expired_ids = [
            signal_id for signal_id, signal in self.active_signals.items()
            if signal.valid_until < current_time
        ]
        
        for signal_id in expired_ids:
            del self.active_signals[signal_id]
        
        # Add new signals
        for signal in new_signals:
            self.active_signals[signal.signal_id] = signal
        
        # Keep history
        self.signal_history.extend(new_signals)
        
        # Limit history size (keep last 1000)
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    async def _store_signals_in_graph(self, signals: List[LiveOptionsSignal]):
        """Store signals in Neo4j for analysis"""
        try:
            with self.graph_builder.driver.session() as session:
                for signal in signals:
                    session.run("""
                        CREATE (ls:LiveSignal {
                            signal_id: $signal_id,
                            symbol: $symbol,
                            signal_type: $signal_type,
                            strategy: $strategy,
                            strength: $strength,
                            win_probability: $win_probability,
                            heat_score: $heat_score,
                            confidence_score: $confidence_score,
                            priority: $priority,
                            generated_at: $generated_at,
                            valid_until: $valid_until
                        })
                    """,
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    signal_type=signal.signal_type.value,
                    strategy=signal.strategy,
                    strength=signal.strength.value,
                    win_probability=signal.win_probability,
                    heat_score=signal.heat_score,
                    confidence_score=signal.confidence_score,
                    priority=signal.priority,
                    generated_at=signal.generated_at.isoformat(),
                    valid_until=signal.valid_until.isoformat()
                    )
        except Exception as e:
            logger.error(f"Error storing signals in graph: {e}")
    
    async def get_active_signals(self, min_priority: int = 5) -> List[LiveOptionsSignal]:
        """Get currently active signals above priority threshold"""
        current_time = datetime.now()
        
        active = [
            signal for signal in self.active_signals.values()
            if signal.valid_until > current_time and signal.priority >= min_priority
        ]
        
        return sorted(active, key=lambda s: s.priority, reverse=True)
    
    async def get_ultra_strong_signals(self) -> List[LiveOptionsSignal]:
        """Get only ultra-strong signals"""
        return [
            signal for signal in self.active_signals.values()
            if signal.strength == SignalStrength.ULTRA_STRONG and 
               signal.valid_until > datetime.now()
        ]

# Global signal generator instance
live_signal_generator = LiveOptionsSignalGenerator()