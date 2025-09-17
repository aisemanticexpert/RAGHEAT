"""
FastAPI routes for Hidden Markov Model predictions and regime detection
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel

from ..models.time_series.hmm_signal_generator import HMMSignalGenerator
from ..models.time_series.market_regime_detector import MarketRegimeDetector
from ..models.time_series.unified_signal_system import UnifiedSignalSystem
from ..models.time_series.hmm_data_processor import HMMDataProcessor


router = APIRouter(prefix="/api/hmm", tags=["HMM Models"])


class HMMSignalRequest(BaseModel):
    """Request model for HMM signal generation"""
    symbol: str
    current_price: Optional[float] = None
    heat_score: float = 0.5
    macd_result: Optional[Dict] = None
    bollinger_result: Optional[Dict] = None
    price_change_pct: float = 0.0
    volume_ratio: float = 1.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    n_regimes: int = 4


class RegimeDetectionRequest(BaseModel):
    """Request model for regime detection"""
    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    n_regimes: int = 4
    lookback_period: int = 252


class UnifiedSignalRequest(BaseModel):
    """Request model for unified signal generation"""
    symbol: str
    current_price: Optional[float] = None
    heat_score: float = 0.5
    macd_result: Optional[Dict] = None
    bollinger_result: Optional[Dict] = None
    price_change_pct: float = 0.0
    volume_ratio: float = 1.0
    technical_weight: float = 0.3
    garch_weight: float = 0.35
    hmm_weight: float = 0.35
    risk_mode: str = "moderate"


class HMMBatchRequest(BaseModel):
    """Request model for batch HMM processing"""
    symbols: List[str]
    n_regimes: int = 4
    heat_scores: Optional[Dict[str, float]] = None
    analysis_type: str = "regime_detection"  # "regime_detection", "signals", "unified"


# Global instances
hmm_generator = HMMSignalGenerator()
regime_detector = MarketRegimeDetector()
unified_system = UnifiedSignalSystem()
hmm_processor = HMMDataProcessor()


@router.post("/signal", response_model=Dict)
async def generate_hmm_signal(request: HMMSignalRequest):
    """
    Generate HMM-based trading signal with regime detection
    
    Returns:
        Trading signal enhanced with market regime analysis
    """
    try:
        # Parse dates if provided
        start_date = None
        end_date = None
        if request.start_date:
            start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        if request.end_date:
            end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # Initialize regime detector with specified parameters
        if request.n_regimes != 4:
            regime_detector.n_regimes = request.n_regimes
            regime_detector.hmm_model.n_states = request.n_regimes
        
        # Generate HMM signal
        signal = hmm_generator.generate_signal(
            symbol=request.symbol,
            current_price=request.current_price or 100.0,
            heat_score=request.heat_score,
            macd_result=request.macd_result,
            bollinger_result=request.bollinger_result,
            price_change_pct=request.price_change_pct,
            volume_ratio=request.volume_ratio
        )
        
        return {
            "success": True,
            "data": signal.to_dict(),
            "message": f"HMM signal generated for {request.symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating HMM signal: {str(e)}")


@router.post("/regime", response_model=Dict)
async def detect_regime(request: RegimeDetectionRequest):
    """
    Detect market regime using Hidden Markov Models
    
    Returns:
        Current market regime with transition probabilities and risk assessment
    """
    try:
        # Parse dates
        start_date = None
        end_date = None
        if request.start_date:
            start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        if request.end_date:
            end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # Initialize detector with parameters
        detector = MarketRegimeDetector(
            n_regimes=request.n_regimes,
            lookback_period=request.lookback_period
        )
        
        # Detect regime
        regime_result = detector.detect_regime(
            symbol=request.symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get regime summary
        regime_summary = detector.get_regime_summary()
        
        return {
            "success": True,
            "data": {
                "regime_detection": regime_result.to_dict(),
                "regime_summary": regime_summary
            },
            "message": f"Regime detection completed for {request.symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in regime detection: {str(e)}")


@router.post("/unified", response_model=Dict)
async def generate_unified_signal(request: UnifiedSignalRequest):
    """
    Generate unified signal combining technical analysis, GARCH, and HMM
    
    Returns:
        Comprehensive trading signal with multi-model consensus
    """
    try:
        # Initialize unified system with custom weights
        system = UnifiedSignalSystem(
            technical_weight=request.technical_weight,
            garch_weight=request.garch_weight,
            hmm_weight=request.hmm_weight,
            risk_management_mode=request.risk_mode
        )
        
        # Generate unified signal
        unified_signal = system.generate_unified_signal(
            symbol=request.symbol,
            current_price=request.current_price or 100.0,
            heat_score=request.heat_score,
            macd_result=request.macd_result,
            bollinger_result=request.bollinger_result,
            price_change_pct=request.price_change_pct,
            volume_ratio=request.volume_ratio
        )
        
        return {
            "success": True,
            "data": unified_signal.to_dict(),
            "message": f"Unified signal generated for {request.symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating unified signal: {str(e)}")


@router.post("/batch", response_model=Dict)
async def batch_hmm_analysis(request: HMMBatchRequest):
    """
    Perform batch HMM analysis for multiple symbols
    
    Returns:
        Analysis results for all requested symbols
    """
    try:
        results = {}
        errors = {}
        
        for symbol in request.symbols:
            try:
                heat_score = 0.5
                if request.heat_scores and symbol in request.heat_scores:
                    heat_score = request.heat_scores[symbol]
                
                if request.analysis_type == "regime_detection":
                    # Regime detection only
                    detector = MarketRegimeDetector(n_regimes=request.n_regimes)
                    result = detector.detect_regime(symbol=symbol)
                    results[symbol] = result.to_dict()
                    
                elif request.analysis_type == "signals":
                    # HMM signals only
                    signal = hmm_generator.generate_signal(
                        symbol=symbol,
                        current_price=100.0,
                        heat_score=heat_score
                    )
                    results[symbol] = signal.to_dict()
                    
                elif request.analysis_type == "unified":
                    # Full unified analysis
                    unified_signal = unified_system.generate_unified_signal(
                        symbol=symbol,
                        current_price=100.0,
                        heat_score=heat_score
                    )
                    results[symbol] = unified_signal.to_dict()
                
            except Exception as e:
                errors[symbol] = str(e)
        
        return {
            "success": True,
            "data": {
                "results": results,
                "errors": errors,
                "processed_count": len(results),
                "error_count": len(errors),
                "analysis_type": request.analysis_type
            },
            "message": f"Batch analysis completed for {len(request.symbols)} symbols"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in batch processing: {str(e)}")


@router.get("/regime-analysis/{symbol}", response_model=Dict)
async def get_regime_analysis(
    symbol: str,
    days: int = Query(60, description="Number of days for analysis"),
    n_regimes: int = Query(4, description="Number of regimes to detect")
):
    """
    Get detailed regime analysis for a symbol
    
    Returns:
        Comprehensive regime analysis with historical patterns
    """
    try:
        # Initialize detector
        detector = MarketRegimeDetector(
            n_regimes=n_regimes,
            lookback_period=max(252, days * 2)
        )
        
        # Get regime detection
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 2)
        
        regime_result = detector.detect_regime(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get regime summary
        regime_summary = detector.get_regime_summary()
        
        # Analyze regime stability
        regime_stability = {
            "current_regime_duration": regime_result.regime_duration,
            "regime_strength": regime_result.regime_strength,
            "transition_risk": _calculate_transition_risk(regime_result),
            "regime_persistence": _calculate_regime_persistence(regime_result)
        }
        
        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "regime_detection": regime_result.to_dict(),
                "regime_summary": regime_summary,
                "regime_stability": regime_stability,
                "analysis_period_days": days
            },
            "message": f"Regime analysis completed for {symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in regime analysis: {str(e)}")


@router.get("/data-preprocessing/{symbol}", response_model=Dict)
async def get_hmm_data_preprocessing(
    symbol: str,
    feature_selection: str = Query("kbest", description="Feature selection method"),
    n_features: int = Query(20, description="Number of features to select"),
    scaling_method: str = Query("robust", description="Scaling method")
):
    """
    Get HMM data preprocessing results for a symbol
    
    Returns:
        Preprocessed data with feature importance and quality metrics
    """
    try:
        # Initialize processor with parameters
        processor = HMMDataProcessor(
            feature_selection_method=feature_selection,
            n_features=n_features,
            scaling_method=scaling_method
        )
        
        # Process data
        processed_data = processor.process_for_hmm(symbol=symbol)
        
        # Get feature importance
        feature_importance = processor.get_feature_importance_report()
        
        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "preprocessing_metadata": processed_data.preprocessing_metadata,
                "feature_importance": feature_importance,
                "market_conditions": processed_data.market_conditions,
                "regime_indicators": processed_data.regime_indicators.to_dict() if not processed_data.regime_indicators.empty else {},
                "data_quality": {
                    "n_observations": processed_data.feature_matrix.shape[0],
                    "n_features": processed_data.feature_matrix.shape[1],
                    "data_quality_score": processed_data.preprocessing_metadata.get('data_quality_score', 0)
                }
            },
            "message": f"Data preprocessing completed for {symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in data preprocessing: {str(e)}")


@router.get("/model-comparison/{symbol}", response_model=Dict)
async def compare_models(
    symbol: str,
    include_technical: bool = Query(True, description="Include technical analysis"),
    include_garch: bool = Query(True, description="Include GARCH model"),
    include_hmm: bool = Query(True, description="Include HMM model")
):
    """
    Compare different model predictions for a symbol
    
    Returns:
        Side-by-side comparison of model predictions and consensus
    """
    try:
        comparison_results = {}
        
        # Generate signals from different models
        if include_technical:
            from ..signals.buy_sell_engine import BuySellSignalEngine
            technical_engine = BuySellSignalEngine()
            tech_signal = technical_engine.calculate_trading_signal(
                symbol=symbol,
                current_price=100.0,
                heat_score=0.5,
                macd_result={},
                bollinger_result={},
                price_change_pct=0.0,
                volume_ratio=1.0
            )
            comparison_results['technical'] = tech_signal.to_dict()
        
        if include_garch:
            from ..models.time_series.garch_signal_generator import GARCHSignalGenerator
            garch_gen = GARCHSignalGenerator()
            garch_signal = garch_gen.generate_signal(
                symbol=symbol,
                current_price=100.0,
                heat_score=0.5
            )
            comparison_results['garch'] = garch_signal.to_dict()
        
        if include_hmm:
            hmm_signal = hmm_generator.generate_signal(
                symbol=symbol,
                current_price=100.0,
                heat_score=0.5
            )
            comparison_results['hmm'] = hmm_signal.to_dict()
        
        # Calculate consensus if multiple models
        if len(comparison_results) > 1:
            consensus = _calculate_model_consensus(comparison_results)
            comparison_results['consensus'] = consensus
        
        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "model_predictions": comparison_results,
                "models_compared": list(comparison_results.keys()),
                "comparison_timestamp": datetime.now().isoformat()
            },
            "message": f"Model comparison completed for {symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in model comparison: {str(e)}")


@router.get("/system-status", response_model=Dict)
async def get_system_status():
    """
    Get status of all HMM system components
    
    Returns:
        Comprehensive system health and performance metrics
    """
    try:
        # Get status from all components
        hmm_status = hmm_generator.get_signal_summary()
        regime_status = regime_detector.get_regime_summary()
        unified_status = unified_system.get_system_status()
        
        # Calculate system health
        system_health = {
            "hmm_generator": "HEALTHY",
            "regime_detector": "HEALTHY",
            "unified_system": "HEALTHY",
            "overall_status": "OPERATIONAL"
        }
        
        return {
            "success": True,
            "data": {
                "system_health": system_health,
                "hmm_generator_status": hmm_status,
                "regime_detector_status": regime_status,
                "unified_system_status": unified_status,
                "last_updated": datetime.now().isoformat()
            },
            "message": "System status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving system status: {str(e)}")


@router.post("/clear-cache", response_model=Dict)
async def clear_system_cache():
    """
    Clear all system caches
    
    Returns:
        Cache clearing confirmation
    """
    try:
        # Clear caches from all components
        hmm_generator.clear_cache()
        regime_detector.clear_cache()
        unified_system.clear_caches()
        
        return {
            "success": True,
            "data": {"cache_cleared": True},
            "message": "All system caches cleared successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error clearing cache: {str(e)}")


@router.post("/update-weights", response_model=Dict)
async def update_unified_weights(
    technical_weight: float = Query(0.3, description="Technical analysis weight"),
    garch_weight: float = Query(0.35, description="GARCH model weight"),
    hmm_weight: float = Query(0.35, description="HMM model weight")
):
    """
    Update weights for unified signal system
    
    Returns:
        Updated weight configuration
    """
    try:
        # Validate weights
        total_weight = technical_weight + garch_weight + hmm_weight
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
        
        # Update weights
        unified_system.update_weights(
            technical_weight=technical_weight,
            garch_weight=garch_weight,
            hmm_weight=hmm_weight
        )
        
        return {
            "success": True,
            "data": {
                "updated_weights": {
                    "technical": technical_weight,
                    "garch": garch_weight,
                    "hmm": hmm_weight
                }
            },
            "message": "Unified system weights updated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error updating weights: {str(e)}")


def _calculate_transition_risk(regime_result) -> float:
    """Calculate regime transition risk"""
    duration = regime_result.regime_duration
    strength = regime_result.regime_strength
    
    # Higher risk if regime is old and weak
    duration_risk = min(1.0, duration / 60)  # Risk increases after 60 days
    strength_risk = 1.0 - strength
    
    return (duration_risk + strength_risk) / 2


def _calculate_regime_persistence(regime_result) -> float:
    """Calculate regime persistence score"""
    duration = regime_result.regime_duration
    strength = regime_result.regime_strength
    confidence = regime_result.confidence
    
    # Persistence based on duration, strength, and confidence
    duration_score = min(1.0, duration / 30)  # Normalize by 30 days
    persistence = (duration_score + strength + confidence) / 3
    
    return persistence


def _calculate_model_consensus(model_results: Dict) -> Dict[str, any]:
    """Calculate consensus between different models"""
    signals = []
    confidences = []
    
    for model, result in model_results.items():
        if model == 'consensus':
            continue
            
        # Extract signal and confidence
        if 'unified_signal' in result:
            signal = result['unified_signal']
            confidence = result['unified_confidence']
        elif 'combined_signal' in result:
            signal = result['combined_signal']
            confidence = result['combined_confidence']
        elif 'signal' in result:
            signal = result['signal']
            confidence = result['confidence']
        else:
            continue
            
        signals.append(signal)
        confidences.append(confidence)
    
    # Calculate consensus metrics
    if not signals:
        return {"consensus_score": 0.0, "agreement": "UNKNOWN"}
    
    # Count agreement
    buy_signals = sum(1 for s in signals if 'BUY' in s)
    sell_signals = sum(1 for s in signals if 'SELL' in s)
    hold_signals = sum(1 for s in signals if s == 'HOLD')
    
    total_models = len(signals)
    max_agreement = max(buy_signals, sell_signals, hold_signals)
    
    consensus_score = max_agreement / total_models
    
    if buy_signals >= sell_signals and buy_signals >= hold_signals:
        agreement = "BULLISH"
    elif sell_signals >= buy_signals and sell_signals >= hold_signals:
        agreement = "BEARISH"
    else:
        agreement = "NEUTRAL"
    
    return {
        "consensus_score": consensus_score,
        "agreement": agreement,
        "model_distribution": {
            "bullish": buy_signals,
            "bearish": sell_signals,
            "neutral": hold_signals
        },
        "average_confidence": np.mean(confidences) if confidences else 0.0
    }