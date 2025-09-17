"""
Intelligent LLM Stock Analysis Service
Integrates GraphRAG, PathRAG, and heat diffusion models for comprehensive stock analysis
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from enum import Enum

# Import our advanced reasoning systems
from models.knowledge_graph.graph_rag_engine import graph_rag_engine, GraphRAGResult
from models.knowledge_graph.path_rag_engine import initialize_path_rag, PathRAGResult, PathRAGQuery, PathType
from data_pipeline.synthetic_market_generator import synthetic_generator
from services.market_hours_detector import get_market_hours_info

logger = logging.getLogger(__name__)


class AnalysisDepth(Enum):
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP_RESEARCH = "deep_research"


class TradeSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    WATCH = "watch"


@dataclass
class MarketContext:
    """Current market context for analysis"""
    market_status: str
    data_source: str
    market_sentiment: float
    volatility_regime: str
    heated_sectors: List[str]
    market_heat_index: float
    regime: str
    timestamp: datetime


@dataclass
class ReasoningEvidence:
    """Evidence supporting reasoning"""
    source: str  # "GraphRAG", "PathRAG", "HeatModel", "LLM"
    evidence_type: str
    confidence: float
    description: str
    supporting_data: Dict[str, Any]


@dataclass
class TradingRecommendation:
    """Comprehensive trading recommendation"""
    symbol: str
    signal: TradeSignal
    confidence: float
    price_target: Optional[float]
    stop_loss: Optional[float]
    time_horizon: str  # "immediate", "short_term", "medium_term", "long_term"
    position_size: str  # "small", "medium", "large"
    reasoning: str
    evidence: List[ReasoningEvidence]
    risk_factors: List[str]
    catalysts: List[str]


@dataclass
class ComprehensiveAnalysis:
    """Complete analysis result"""
    symbol: str
    query: str
    analysis_depth: AnalysisDepth
    market_context: MarketContext
    recommendation: TradingRecommendation
    graph_rag_result: Optional[GraphRAGResult]
    path_rag_result: Optional[PathRAGResult]
    heat_analysis: Dict[str, Any]
    llm_narrative: str
    key_insights: List[str]
    execution_summary: Dict[str, Any]
    timestamp: datetime


class IntelligentLLMAnalyzer:
    """
    Advanced LLM-powered stock analyzer with multi-modal reasoning
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        
        # Initialize PathRAG with GraphRAG
        self.path_rag = initialize_path_rag(graph_rag_engine)
        
        # Analysis templates for different depths
        self.analysis_templates = {
            AnalysisDepth.QUICK: {
                "max_reasoning_paths": 3,
                "max_graph_hops": 2,
                "include_heat_analysis": True,
                "include_path_analysis": False,
                "llm_tokens": 500
            },
            AnalysisDepth.STANDARD: {
                "max_reasoning_paths": 5,
                "max_graph_hops": 3,
                "include_heat_analysis": True,
                "include_path_analysis": True,
                "llm_tokens": 1000
            },
            AnalysisDepth.COMPREHENSIVE: {
                "max_reasoning_paths": 8,
                "max_graph_hops": 4,
                "include_heat_analysis": True,
                "include_path_analysis": True,
                "llm_tokens": 1500
            },
            AnalysisDepth.DEEP_RESEARCH: {
                "max_reasoning_paths": 12,
                "max_graph_hops": 5,
                "include_heat_analysis": True,
                "include_path_analysis": True,
                "llm_tokens": 2000
            }
        }
    
    async def analyze_stock(self, symbol: str, query: Optional[str] = None, 
                           depth: AnalysisDepth = AnalysisDepth.STANDARD) -> ComprehensiveAnalysis:
        """Perform comprehensive stock analysis"""
        start_time = datetime.now()
        
        try:
            # Default query if none provided
            if not query:
                query = f"Should I buy {symbol}? Provide comprehensive analysis with clear reasoning."
            
            # Get current market context
            market_context = await self.get_market_context()
            
            # Update knowledge graph with latest data
            await self.update_knowledge_graph()
            
            # Get analysis template
            template = self.analysis_templates[depth]
            
            # Perform GraphRAG analysis
            logger.info(f"🧠 Starting GraphRAG analysis for {symbol}")
            graph_rag_result = await self.perform_graph_rag_analysis(symbol, query, template)
            
            # Perform PathRAG analysis (if enabled)
            path_rag_result = None
            if template["include_path_analysis"]:
                logger.info(f"🛤️ Starting PathRAG analysis for {symbol}")
                path_rag_result = await self.perform_path_rag_analysis(symbol, template)
            
            # Perform heat diffusion analysis
            logger.info(f"🔥 Analyzing heat diffusion for {symbol}")
            heat_analysis = await self.perform_heat_analysis(symbol, market_context)
            
            # Synthesize all analysis into recommendation
            logger.info(f"⚡ Synthesizing analysis for {symbol}")
            recommendation = await self.synthesize_recommendation(
                symbol, graph_rag_result, path_rag_result, heat_analysis, market_context
            )
            
            # Generate LLM narrative
            logger.info(f"📝 Generating LLM narrative for {symbol}")
            llm_narrative = await self.generate_llm_narrative(
                symbol, query, graph_rag_result, path_rag_result, heat_analysis, 
                recommendation, template["llm_tokens"]
            )
            
            # Extract key insights
            key_insights = self.extract_key_insights(
                graph_rag_result, path_rag_result, heat_analysis, recommendation
            )
            
            # Execution summary
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_summary = {
                "execution_time_seconds": execution_time,
                "analysis_depth": depth.value,
                "components_analyzed": {
                    "graph_rag": graph_rag_result is not None,
                    "path_rag": path_rag_result is not None,
                    "heat_analysis": len(heat_analysis) > 0,
                    "llm_narrative": len(llm_narrative) > 0
                },
                "data_sources_used": [
                    market_context.data_source,
                    "knowledge_graph",
                    "heat_diffusion_model"
                ]
            }
            
            return ComprehensiveAnalysis(
                symbol=symbol,
                query=query,
                analysis_depth=depth,
                market_context=market_context,
                recommendation=recommendation,
                graph_rag_result=graph_rag_result,
                path_rag_result=path_rag_result,
                heat_analysis=heat_analysis,
                llm_narrative=llm_narrative,
                key_insights=key_insights,
                execution_summary=execution_summary,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive analysis for {symbol}: {e}")
            
            # Return error analysis
            return ComprehensiveAnalysis(
                symbol=symbol,
                query=query or f"Analysis of {symbol}",
                analysis_depth=depth,
                market_context=await self.get_market_context(),
                recommendation=TradingRecommendation(
                    symbol=symbol,
                    signal=TradeSignal.WATCH,
                    confidence=0.0,
                    price_target=None,
                    stop_loss=None,
                    time_horizon="unknown",
                    position_size="small",
                    reasoning=f"Analysis failed: {e}",
                    evidence=[],
                    risk_factors=[f"Analysis error: {e}"],
                    catalysts=[]
                ),
                graph_rag_result=None,
                path_rag_result=None,
                heat_analysis={"error": str(e)},
                llm_narrative=f"Analysis failed due to error: {e}",
                key_insights=[f"Error in analysis: {e}"],
                execution_summary={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def get_market_context(self) -> MarketContext:
        """Get current market context"""
        try:
            # Get market hours info
            market_hours = get_market_hours_info()
            
            # Get synthetic market data (works regardless of market status)
            market_data = synthetic_generator.generate_market_overview()
            
            market_overview = market_data.get('market_overview', {})
            heated_sectors = [hs['sector'] for hs in market_data.get('heated_sectors', [])]
            
            return MarketContext(
                market_status=market_hours.get('status', 'unknown'),
                data_source=market_hours.get('data_source_recommendation', 'synthetic'),
                market_sentiment=market_overview.get('sentiment', 0.0),
                volatility_regime="high" if market_overview.get('market_volatility', 0) > 2 else "normal",
                heated_sectors=heated_sectors,
                market_heat_index=market_overview.get('market_heat_index', 0),
                regime=market_overview.get('regime', 'unknown'),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting market context: {e}")
            return MarketContext(
                market_status="unknown",
                data_source="error",
                market_sentiment=0.0,
                volatility_regime="unknown",
                heated_sectors=[],
                market_heat_index=0.0,
                regime="unknown",
                timestamp=datetime.now()
            )
    
    async def update_knowledge_graph(self):
        """Update knowledge graph with latest market data"""
        try:
            # Get latest market data
            market_data = synthetic_generator.generate_market_overview()
            
            # Update GraphRAG engine
            graph_rag_engine.update_from_market_data(market_data)
            
            logger.info("✅ Knowledge graph updated with latest market data")
            
        except Exception as e:
            logger.error(f"❌ Error updating knowledge graph: {e}")
    
    async def perform_graph_rag_analysis(self, symbol: str, query: str, template: Dict) -> Optional[GraphRAGResult]:
        """Perform GraphRAG analysis"""
        try:
            entity_id = f"STOCK_{symbol}"
            result = await graph_rag_engine.query(query, entity_id)
            return result
            
        except Exception as e:
            logger.error(f"❌ GraphRAG analysis error for {symbol}: {e}")
            return None
    
    async def perform_path_rag_analysis(self, symbol: str, template: Dict) -> Optional[PathRAGResult]:
        """Perform PathRAG analysis"""
        try:
            if not self.path_rag:
                return None
                
            query = PathRAGQuery(
                query_text=f"Analyze multi-hop reasoning paths for {symbol}",
                start_entity_ids=[f"STOCK_{symbol}"],
                max_hops=template["max_graph_hops"],
                path_types=[PathType.CAUSAL, PathType.CORRELATIONAL, PathType.HIERARCHICAL],
                min_confidence=0.3,
                max_paths=template["max_reasoning_paths"]
            )
            
            result = await self.path_rag.execute_path_rag_query(query)
            return result
            
        except Exception as e:
            logger.error(f"❌ PathRAG analysis error for {symbol}: {e}")
            return None
    
    async def perform_heat_analysis(self, symbol: str, market_context: MarketContext) -> Dict[str, Any]:
        """Perform heat diffusion analysis"""
        try:
            # Get current market data
            market_data = synthetic_generator.generate_market_overview()
            
            # Find stock data
            stock_data = market_data.get('stocks', {}).get(symbol, {})
            
            if not stock_data:
                return {"error": f"No data found for {symbol}"}
            
            # Heat analysis
            stock_heat = stock_data.get('heat_level', 0)
            sector = stock_data.get('sector', 'unknown')
            sector_data = market_data.get('sectors', {}).get(sector, {})
            sector_heat = sector_data.get('heat_level', 0)
            
            # Heat diffusion context
            heated_sectors = market_data.get('heated_sectors', [])
            is_in_heated_sector = any(hs['sector'] == sector for hs in heated_sectors)
            
            # Heat trend (simple calculation)
            heat_momentum = "increasing" if stock_heat > sector_heat else "decreasing"
            
            # Heat interpretation
            heat_interpretation = self.interpret_heat_levels(stock_heat, sector_heat, market_context.market_heat_index)
            
            return {
                "stock_heat_level": stock_heat,
                "sector_heat_level": sector_heat,
                "market_heat_index": market_context.market_heat_index,
                "sector": sector,
                "is_in_heated_sector": is_in_heated_sector,
                "heat_momentum": heat_momentum,
                "heat_interpretation": heat_interpretation,
                "diffusion_active": True,
                "heat_sources": [hs for hs in heated_sectors if hs['sector'] == sector],
                "relative_heat_strength": abs(stock_heat) / max(abs(market_context.market_heat_index/100), 0.1)
            }
            
        except Exception as e:
            logger.error(f"❌ Heat analysis error for {symbol}: {e}")
            return {"error": str(e)}
    
    def interpret_heat_levels(self, stock_heat: float, sector_heat: float, market_heat: float) -> Dict[str, str]:
        """Interpret heat levels for trading signals"""
        interpretations = {}
        
        # Stock heat interpretation
        if stock_heat > 0.7:
            interpretations["stock"] = "Very hot - strong bullish momentum"
        elif stock_heat > 0.3:
            interpretations["stock"] = "Hot - moderate bullish momentum"
        elif stock_heat < -0.7:
            interpretations["stock"] = "Very cold - strong bearish momentum"
        elif stock_heat < -0.3:
            interpretations["stock"] = "Cold - moderate bearish momentum"
        else:
            interpretations["stock"] = "Neutral - no significant heat"
        
        # Sector heat interpretation
        if sector_heat > 0.5:
            interpretations["sector"] = "Sector is heating up - rotation likely"
        elif sector_heat < -0.5:
            interpretations["sector"] = "Sector is cooling down - rotation away"
        else:
            interpretations["sector"] = "Sector heat is neutral"
        
        # Market heat interpretation  
        if market_heat > 70:
            interpretations["market"] = "Market overheated - caution advised"
        elif market_heat > 40:
            interpretations["market"] = "Market warming - bullish conditions"
        elif market_heat < -40:
            interpretations["market"] = "Market cooling - bearish conditions"
        else:
            interpretations["market"] = "Market heat neutral"
        
        return interpretations
    
    async def synthesize_recommendation(self, symbol: str, graph_rag: Optional[GraphRAGResult], 
                                      path_rag: Optional[PathRAGResult], heat_analysis: Dict,
                                      market_context: MarketContext) -> TradingRecommendation:
        """Synthesize all analysis into a trading recommendation"""
        try:
            evidence = []
            confidence_scores = []
            risk_factors = []
            catalysts = []
            
            # Process GraphRAG evidence
            if graph_rag:
                evidence.append(ReasoningEvidence(
                    source="GraphRAG",
                    evidence_type="knowledge_graph_reasoning",
                    confidence=graph_rag.confidence,
                    description="Knowledge graph analysis with LLM reasoning",
                    supporting_data={"reasoning_paths": len(graph_rag.reasoning_paths)}
                ))
                confidence_scores.append(graph_rag.confidence)
                risk_factors.extend(graph_rag.risk_factors)
            
            # Process PathRAG evidence
            if path_rag and path_rag.discovered_paths:
                avg_path_confidence = np.mean([p.total_confidence for p in path_rag.discovered_paths])
                evidence.append(ReasoningEvidence(
                    source="PathRAG",
                    evidence_type="multi_hop_reasoning",
                    confidence=avg_path_confidence,
                    description=f"Multi-hop reasoning across {len(path_rag.discovered_paths)} paths",
                    supporting_data={"paths": len(path_rag.discovered_paths)}
                ))
                confidence_scores.append(avg_path_confidence)
                
                # Add consensus insights as catalysts
                catalysts.extend(path_rag.consensus_insights)
            
            # Process heat analysis evidence
            if "error" not in heat_analysis:
                stock_heat = heat_analysis.get('stock_heat_level', 0)
                heat_confidence = min(abs(stock_heat) * 2, 1.0)  # Convert heat to confidence
                
                evidence.append(ReasoningEvidence(
                    source="HeatModel",
                    evidence_type="heat_diffusion_analysis",
                    confidence=heat_confidence,
                    description=f"Heat diffusion model analysis",
                    supporting_data=heat_analysis
                ))
                confidence_scores.append(heat_confidence)
                
                # Heat-based catalysts
                if heat_analysis.get('is_in_heated_sector'):
                    catalysts.append(f"Stock is in heated sector: {heat_analysis.get('sector')}")
            
            # Calculate overall confidence
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.3
            
            # Determine signal based on evidence
            signal = self.determine_trade_signal(evidence, heat_analysis, market_context)
            
            # Generate reasoning
            reasoning = self.generate_reasoning_summary(evidence, heat_analysis, market_context, signal)
            
            # Determine price targets and risk management
            price_target, stop_loss = self.calculate_price_targets(
                symbol, signal, overall_confidence, heat_analysis
            )
            
            # Determine position size and time horizon
            position_size = self.determine_position_size(overall_confidence, market_context, signal)
            time_horizon = self.determine_time_horizon(evidence, heat_analysis, path_rag)
            
            return TradingRecommendation(
                symbol=symbol,
                signal=signal,
                confidence=overall_confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                time_horizon=time_horizon,
                position_size=position_size,
                reasoning=reasoning,
                evidence=evidence,
                risk_factors=risk_factors[:5],  # Top 5 risk factors
                catalysts=catalysts[:5]  # Top 5 catalysts
            )
            
        except Exception as e:
            logger.error(f"❌ Error synthesizing recommendation for {symbol}: {e}")
            
            return TradingRecommendation(
                symbol=symbol,
                signal=TradeSignal.WATCH,
                confidence=0.0,
                price_target=None,
                stop_loss=None,
                time_horizon="unknown",
                position_size="small",
                reasoning=f"Error in analysis synthesis: {e}",
                evidence=[],
                risk_factors=[f"Analysis error: {e}"],
                catalysts=[]
            )
    
    def determine_trade_signal(self, evidence: List[ReasoningEvidence], heat_analysis: Dict, 
                             market_context: MarketContext) -> TradeSignal:
        """Determine trade signal from evidence"""
        bullish_signals = 0
        bearish_signals = 0
        signal_strength = 0
        
        # Analyze evidence
        for ev in evidence:
            if ev.source == "HeatModel":
                stock_heat = heat_analysis.get('stock_heat_level', 0)
                if stock_heat > 0.5:
                    bullish_signals += 2
                    signal_strength += stock_heat
                elif stock_heat < -0.5:
                    bearish_signals += 2
                    signal_strength += abs(stock_heat)
                elif stock_heat > 0.2:
                    bullish_signals += 1
                elif stock_heat < -0.2:
                    bearish_signals += 1
            
            # GraphRAG and PathRAG contribute based on confidence
            elif ev.confidence > 0.7:
                bullish_signals += 2  # Assume high confidence is bullish (could be enhanced)
                signal_strength += ev.confidence
            elif ev.confidence > 0.5:
                bullish_signals += 1
        
        # Market context adjustments
        if market_context.volatility_regime == "high":
            signal_strength *= 0.8  # Reduce signal strength in high volatility
        
        # Determine final signal
        net_signals = bullish_signals - bearish_signals
        
        if net_signals >= 4 and signal_strength > 1.5:
            return TradeSignal.STRONG_BUY
        elif net_signals >= 2 and signal_strength > 1.0:
            return TradeSignal.BUY
        elif net_signals <= -4 and signal_strength > 1.5:
            return TradeSignal.STRONG_SELL
        elif net_signals <= -2 and signal_strength > 1.0:
            return TradeSignal.SELL
        elif abs(net_signals) <= 1:
            return TradeSignal.HOLD
        else:
            return TradeSignal.WATCH
    
    def generate_reasoning_summary(self, evidence: List[ReasoningEvidence], heat_analysis: Dict,
                                 market_context: MarketContext, signal: TradeSignal) -> str:
        """Generate human-readable reasoning summary"""
        reasoning_parts = []
        
        # Market context
        reasoning_parts.append(f"Market is currently {market_context.market_status} with {market_context.data_source} data.")
        
        # Evidence summary
        evidence_summary = f"Analysis based on {len(evidence)} evidence sources: "
        evidence_sources = [ev.source for ev in evidence]
        evidence_summary += ", ".join(set(evidence_sources))
        reasoning_parts.append(evidence_summary)
        
        # Heat analysis
        if "error" not in heat_analysis:
            stock_heat = heat_analysis.get('stock_heat_level', 0)
            sector = heat_analysis.get('sector', 'unknown')
            heat_interpretation = heat_analysis.get('heat_interpretation', {})
            
            heat_summary = f"Heat analysis: Stock heat level {stock_heat:.2f} in {sector} sector. "
            heat_summary += heat_interpretation.get('stock', 'No heat interpretation available.')
            reasoning_parts.append(heat_summary)
        
        # Signal explanation
        signal_explanation = f"Recommendation: {signal.value.upper().replace('_', ' ')} based on convergence of evidence sources."
        reasoning_parts.append(signal_explanation)
        
        return " ".join(reasoning_parts)
    
    def calculate_price_targets(self, symbol: str, signal: TradeSignal, confidence: float, 
                              heat_analysis: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Calculate price targets and stop losses"""
        # This is a simplified implementation - would be enhanced with actual price data
        try:
            if "error" in heat_analysis:
                return None, None
            
            # Use synthetic data for price estimation
            market_data = synthetic_generator.generate_market_overview()
            stock_data = market_data.get('stocks', {}).get(symbol, {})
            
            if not stock_data:
                return None, None
            
            current_price = stock_data.get('price', 0)
            if current_price <= 0:
                return None, None
            
            stock_heat = heat_analysis.get('stock_heat_level', 0)
            
            # Simple price target calculation based on heat and confidence
            if signal in [TradeSignal.BUY, TradeSignal.STRONG_BUY]:
                # Upside target based on heat and confidence
                upside_factor = (confidence * abs(stock_heat) * 0.1) + 0.05  # 5-15% typical
                price_target = current_price * (1 + upside_factor)
                
                # Stop loss 3-8% below current price
                stop_loss_factor = 0.03 + (0.05 * (1 - confidence))
                stop_loss = current_price * (1 - stop_loss_factor)
                
                return round(price_target, 2), round(stop_loss, 2)
            
            elif signal in [TradeSignal.SELL, TradeSignal.STRONG_SELL]:
                # Downside target
                downside_factor = (confidence * abs(stock_heat) * 0.1) + 0.05
                price_target = current_price * (1 - downside_factor)
                
                # Stop loss above current price (for short positions)
                stop_loss = current_price * (1 + stop_loss_factor)
                
                return round(price_target, 2), round(stop_loss, 2)
            
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Error calculating price targets for {symbol}: {e}")
            return None, None
    
    def determine_position_size(self, confidence: float, market_context: MarketContext, 
                              signal: TradeSignal) -> str:
        """Determine position size recommendation"""
        if confidence > 0.8 and market_context.volatility_regime != "high":
            return "large"
        elif confidence > 0.6 and signal in [TradeSignal.STRONG_BUY, TradeSignal.STRONG_SELL]:
            return "medium"
        elif confidence > 0.4:
            return "medium"
        else:
            return "small"
    
    def determine_time_horizon(self, evidence: List[ReasoningEvidence], heat_analysis: Dict,
                             path_rag: Optional[PathRAGResult]) -> str:
        """Determine time horizon for the trade"""
        # Heat-based signals are often shorter-term
        if heat_analysis.get('heat_momentum') == "increasing" and abs(heat_analysis.get('stock_heat_level', 0)) > 0.5:
            return "immediate"
        
        # PathRAG time sensitivity
        if path_rag and path_rag.discovered_paths:
            immediate_paths = sum(1 for p in path_rag.discovered_paths if p.time_sensitivity == "immediate")
            if immediate_paths > len(path_rag.discovered_paths) / 2:
                return "immediate"
        
        # Default based on evidence strength
        avg_confidence = np.mean([ev.confidence for ev in evidence]) if evidence else 0.3
        
        if avg_confidence > 0.8:
            return "short_term"
        elif avg_confidence > 0.6:
            return "medium_term"
        else:
            return "long_term"
    
    async def generate_llm_narrative(self, symbol: str, query: str, graph_rag: Optional[GraphRAGResult],
                                   path_rag: Optional[PathRAGResult], heat_analysis: Dict,
                                   recommendation: TradingRecommendation, max_tokens: int) -> str:
        """Generate comprehensive LLM narrative"""
        try:
            # Prepare comprehensive context
            context_parts = []
            
            # Market context
            context_parts.append("=== MARKET CONTEXT ===")
            context_parts.append(f"Symbol: {symbol}")
            context_parts.append(f"Query: {query}")
            
            # GraphRAG insights
            if graph_rag:
                context_parts.append("\n=== KNOWLEDGE GRAPH ANALYSIS ===")
                context_parts.append(f"Confidence: {graph_rag.confidence:.2f}")
                context_parts.append(f"Reasoning paths: {len(graph_rag.reasoning_paths)}")
                if graph_rag.llm_analysis:
                    context_parts.append(f"GraphRAG Analysis: {graph_rag.llm_analysis[:300]}...")
            
            # PathRAG insights
            if path_rag and path_rag.discovered_paths:
                context_parts.append("\n=== MULTI-HOP REASONING ===")
                context_parts.append(f"Paths discovered: {len(path_rag.discovered_paths)}")
                context_parts.append(f"Consensus insights: {', '.join(path_rag.consensus_insights[:3])}")
            
            # Heat analysis
            context_parts.append("\n=== HEAT DIFFUSION ANALYSIS ===")
            if "error" not in heat_analysis:
                context_parts.append(f"Stock heat: {heat_analysis.get('stock_heat_level', 0):.2f}")
                context_parts.append(f"Sector heat: {heat_analysis.get('sector_heat_level', 0):.2f}")
                heat_interp = heat_analysis.get('heat_interpretation', {})
                if heat_interp:
                    context_parts.append(f"Interpretation: {heat_interp.get('stock', '')}")
            
            # Final recommendation
            context_parts.append("\n=== RECOMMENDATION ===")
            context_parts.append(f"Signal: {recommendation.signal.value}")
            context_parts.append(f"Confidence: {recommendation.confidence:.2f}")
            context_parts.append(f"Reasoning: {recommendation.reasoning}")
            
            context_text = "\n".join(context_parts)
            
            # For now, return a structured narrative (LLM integration can be added later)
            narrative_parts = []
            
            narrative_parts.append(f"## Comprehensive Analysis for {symbol}")
            narrative_parts.append(f"**Recommendation:** {recommendation.signal.value.upper()} with {recommendation.confidence:.0%} confidence")
            
            if recommendation.price_target:
                narrative_parts.append(f"**Price Target:** ${recommendation.price_target}")
            if recommendation.stop_loss:
                narrative_parts.append(f"**Stop Loss:** ${recommendation.stop_loss}")
            
            narrative_parts.append(f"**Time Horizon:** {recommendation.time_horizon}")
            narrative_parts.append(f"**Position Size:** {recommendation.position_size}")
            
            narrative_parts.append("\n### Key Evidence:")
            for i, evidence in enumerate(recommendation.evidence[:3], 1):
                narrative_parts.append(f"{i}. **{evidence.source}:** {evidence.description} (Confidence: {evidence.confidence:.0%})")
            
            if recommendation.catalysts:
                narrative_parts.append("\n### Catalysts:")
                for catalyst in recommendation.catalysts[:3]:
                    narrative_parts.append(f"• {catalyst}")
            
            if recommendation.risk_factors:
                narrative_parts.append("\n### Risk Factors:")
                for risk in recommendation.risk_factors[:3]:
                    narrative_parts.append(f"• {risk}")
            
            narrative_parts.append(f"\n### Analysis Summary:")
            narrative_parts.append(recommendation.reasoning)
            
            return "\n".join(narrative_parts)
            
        except Exception as e:
            logger.error(f"❌ Error generating LLM narrative for {symbol}: {e}")
            return f"Error generating narrative: {e}"
    
    def extract_key_insights(self, graph_rag: Optional[GraphRAGResult], path_rag: Optional[PathRAGResult],
                           heat_analysis: Dict, recommendation: TradingRecommendation) -> List[str]:
        """Extract key insights from all analysis components"""
        insights = []
        
        try:
            # GraphRAG insights
            if graph_rag and graph_rag.reasoning_paths:
                insights.append(f"Knowledge graph analysis found {len(graph_rag.reasoning_paths)} reasoning paths")
                if graph_rag.confidence > 0.7:
                    insights.append("High confidence in knowledge graph analysis")
            
            # PathRAG insights
            if path_rag and path_rag.discovered_paths:
                insights.append(f"Multi-hop reasoning discovered {len(path_rag.discovered_paths)} paths")
                if path_rag.consensus_insights:
                    insights.extend(path_rag.consensus_insights[:2])
            
            # Heat insights
            if "error" not in heat_analysis:
                stock_heat = heat_analysis.get('stock_heat_level', 0)
                if abs(stock_heat) > 0.5:
                    heat_direction = "bullish" if stock_heat > 0 else "bearish"
                    insights.append(f"Strong {heat_direction} heat signal detected ({stock_heat:.2f})")
                
                if heat_analysis.get('is_in_heated_sector'):
                    insights.append(f"Stock is in a heated sector: {heat_analysis.get('sector')}")
            
            # Recommendation insights
            if recommendation.confidence > 0.8:
                insights.append(f"High confidence {recommendation.signal.value} signal")
            
            if len(recommendation.evidence) > 2:
                insights.append(f"Multiple evidence sources support the analysis ({len(recommendation.evidence)} sources)")
            
            return insights[:8]  # Top 8 insights
            
        except Exception as e:
            logger.error(f"❌ Error extracting insights: {e}")
            return [f"Error extracting insights: {e}"]


# Global analyzer instance
intelligent_analyzer = IntelligentLLMAnalyzer()


# Convenience functions
async def analyze_stock_comprehensive(symbol: str, query: Optional[str] = None,
                                    depth: AnalysisDepth = AnalysisDepth.STANDARD) -> ComprehensiveAnalysis:
    """Perform comprehensive stock analysis"""
    return await intelligent_analyzer.analyze_stock(symbol, query, depth)


async def get_quick_analysis(symbol: str) -> ComprehensiveAnalysis:
    """Get quick analysis for a stock"""
    return await intelligent_analyzer.analyze_stock(symbol, depth=AnalysisDepth.QUICK)


async def get_deep_research(symbol: str, query: str) -> ComprehensiveAnalysis:
    """Get deep research analysis for a stock"""
    return await intelligent_analyzer.analyze_stock(symbol, query, AnalysisDepth.DEEP_RESEARCH)


if __name__ == "__main__":
    # Test the intelligent analyzer
    async def test_analyzer():
        print("🤖 Testing Intelligent LLM Analyzer")
        print("=" * 50)
        
        # Test quick analysis
        result = await get_quick_analysis("AAPL")
        
        print(f"Symbol: {result.symbol}")
        print(f"Signal: {result.recommendation.signal.value}")
        print(f"Confidence: {result.recommendation.confidence:.0%}")
        print(f"Analysis depth: {result.analysis_depth.value}")
        print(f"Execution time: {result.execution_summary['execution_time_seconds']:.2f}s")
        
        print("\nKey Insights:")
        for insight in result.key_insights[:3]:
            print(f"• {insight}")
    
    # Run test
    asyncio.run(test_analyzer())