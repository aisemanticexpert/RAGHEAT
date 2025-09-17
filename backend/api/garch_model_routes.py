"""
FastAPI routes for GARCH model predictions and signals
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel

from ..models.time_series.garch_signal_generator import GARCHSignalGenerator, GARCHSignal
from ..models.time_series.garch_model import GARCHModel, GARCHPrediction
from ..models.time_series.data_preprocessor import TimeSeriesPreprocessor


router = APIRouter(prefix="/api/garch", tags=["GARCH Models"])


class GARCHSignalRequest(BaseModel):
    """Request model for GARCH signal generation"""
    symbol: str
    current_price: Optional[float] = None
    heat_score: float = 0.5
    macd_result: Optional[Dict] = None
    bollinger_result: Optional[Dict] = None
    price_change_pct: float = 0.0
    volume_ratio: float = 1.0
    forecast_horizon: int = 1
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class GARCHPredictionRequest(BaseModel):
    """Request model for GARCH volatility prediction only"""
    symbol: str
    forecast_horizon: int = 1
    model_type: str = "GARCH"
    p: int = 1
    q: int = 1
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class GARCHBatchRequest(BaseModel):
    """Request model for batch GARCH processing"""
    symbols: List[str]
    forecast_horizon: int = 1
    heat_scores: Optional[Dict[str, float]] = None


# Global signal generator instance
signal_generator = GARCHSignalGenerator()


@router.post("/signal", response_model=Dict)
async def generate_garch_signal(request: GARCHSignalRequest):
    """
    Generate enhanced trading signal with GARCH volatility prediction
    
    Returns:
        Comprehensive trading signal with volatility forecasts and risk metrics
    """
    try:
        # Parse dates if provided
        start_date = None
        end_date = None
        if request.start_date:
            start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        if request.end_date:
            end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # Generate signal
        signal = signal_generator.generate_signal(
            symbol=request.symbol,
            current_price=request.current_price or 100.0,  # Default if not provided
            heat_score=request.heat_score,
            macd_result=request.macd_result,
            bollinger_result=request.bollinger_result,
            price_change_pct=request.price_change_pct,
            volume_ratio=request.volume_ratio,
            forecast_horizon=request.forecast_horizon
        )
        
        return {
            "success": True,
            "data": signal.to_dict(),
            "message": f"GARCH signal generated for {request.symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating GARCH signal: {str(e)}")


@router.post("/prediction", response_model=Dict)
async def generate_garch_prediction(request: GARCHPredictionRequest):
    """
    Generate GARCH volatility prediction only (without trading signals)
    
    Returns:
        GARCH volatility forecast and model diagnostics
    """
    try:
        # Initialize preprocessor and model
        preprocessor = TimeSeriesPreprocessor()
        garch_model = GARCHModel(
            model_type=request.model_type,
            p=request.p,
            q=request.q
        )
        
        # Parse dates
        start_date = None
        end_date = None
        if request.start_date:
            start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        if request.end_date:
            end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # Preprocess data
        data = preprocessor.preprocess(
            request.symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Fit model and predict
        garch_model.fit(data.log_returns)
        prediction = garch_model.predict(
            data.log_returns,
            horizon=request.forecast_horizon
        )
        
        # Get model diagnostics
        diagnostics = garch_model.get_model_diagnostics()
        
        return {
            "success": True,
            "data": {
                "prediction": prediction.to_dict(),
                "model_diagnostics": diagnostics,
                "data_quality": preprocessor.validate_data_quality(data)
            },
            "message": f"GARCH prediction generated for {request.symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating GARCH prediction: {str(e)}")


@router.post("/batch", response_model=Dict)
async def generate_batch_signals(request: GARCHBatchRequest):
    """
    Generate GARCH signals for multiple symbols
    
    Returns:
        Dictionary of signals for each symbol
    """
    try:
        results = {}
        errors = {}
        
        for symbol in request.symbols:
            try:
                heat_score = 0.5
                if request.heat_scores and symbol in request.heat_scores:
                    heat_score = request.heat_scores[symbol]
                
                signal = signal_generator.generate_signal(
                    symbol=symbol,
                    current_price=100.0,  # Default, would need real price feed
                    heat_score=heat_score,
                    forecast_horizon=request.forecast_horizon
                )
                
                results[symbol] = signal.to_dict()
                
            except Exception as e:
                errors[symbol] = str(e)
        
        return {
            "success": True,
            "data": {
                "signals": results,
                "errors": errors,
                "processed_count": len(results),
                "error_count": len(errors)
            },
            "message": f"Batch processing completed for {len(request.symbols)} symbols"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in batch processing: {str(e)}")


@router.get("/volatility/{symbol}", response_model=Dict)
async def get_volatility_analysis(
    symbol: str,
    days: int = Query(30, description="Number of days for analysis"),
    model_type: str = Query("GARCH", description="Model type: GARCH, EGARCH, GJR-GARCH")
):
    """
    Get detailed volatility analysis for a symbol
    
    Returns:
        Volatility metrics, regime analysis, and forecasts
    """
    try:
        # Initialize components
        preprocessor = TimeSeriesPreprocessor()
        
        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=max(252, days * 2))  # At least 1 year of data
        
        data = preprocessor.preprocess(symbol, start_date=start_date, end_date=end_date)
        
        # Fit GARCH model
        garch_model = GARCHModel(model_type=model_type)
        garch_model.fit(data.log_returns)
        
        # Generate predictions for next few days
        forecasts = []
        for horizon in [1, 5, 10, 20]:
            pred = garch_model.predict(data.log_returns, horizon=horizon)
            forecasts.append({
                "horizon": horizon,
                "volatility": pred.predicted_volatility,
                "returns": pred.predicted_returns,
                "regime": pred.volatility_regime,
                "confidence_interval": pred.confidence_interval
            })
        
        # Calculate additional metrics
        recent_returns = data.returns.tail(days)
        
        volatility_metrics = {
            "current_volatility": recent_returns.std(),
            "annualized_volatility": recent_returns.std() * np.sqrt(252),
            "volatility_percentile": (data.volatility < recent_returns.std()).mean(),
            "garch_forecasts": forecasts,
            "model_diagnostics": garch_model.get_model_diagnostics(),
            "data_quality": preprocessor.validate_data_quality(data),
            "regime_analysis": {
                "current_regime": forecasts[0]["regime"] if forecasts else "unknown",
                "regime_persistence": _calculate_regime_persistence(data.volatility),
                "regime_transitions": _count_regime_transitions(data.volatility)
            }
        }
        
        return {
            "success": True,
            "data": volatility_metrics,
            "message": f"Volatility analysis completed for {symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in volatility analysis: {str(e)}")


@router.get("/model/status", response_model=Dict)
async def get_model_status():
    """
    Get status and performance metrics of GARCH models
    
    Returns:
        Model cache status and performance metrics
    """
    try:
        cache_info = {
            "cached_models": len(signal_generator._model_cache),
            "cache_details": []
        }
        
        for key, (model, timestamp) in signal_generator._model_cache.items():
            cache_info["cache_details"].append({
                "key": key,
                "last_updated": timestamp.isoformat(),
                "age_hours": (datetime.now() - timestamp).total_seconds() / 3600,
                "model_accuracy": model.model_accuracy if hasattr(model, 'model_accuracy') else 0.0
            })
        
        return {
            "success": True,
            "data": {
                "signal_generator_config": {
                    "garch_weight": signal_generator.garch_weight,
                    "technical_weight": signal_generator.technical_weight,
                    "volatility_thresholds": {
                        "penalty": signal_generator.volatility_penalty_threshold,
                        "boost": signal_generator.volatility_boost_threshold
                    }
                },
                "cache_info": cache_info
            },
            "message": "Model status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving model status: {str(e)}")


@router.post("/model/clear-cache", response_model=Dict)
async def clear_model_cache():
    """
    Clear the GARCH model cache to force retraining
    
    Returns:
        Success confirmation
    """
    try:
        cleared_count = len(signal_generator._model_cache)
        signal_generator._model_cache.clear()
        
        return {
            "success": True,
            "data": {"cleared_models": cleared_count},
            "message": f"Cleared {cleared_count} cached models"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error clearing cache: {str(e)}")


def _calculate_regime_persistence(volatility: pd.Series) -> Dict[str, float]:
    """Calculate how long volatility stays in each regime"""
    if volatility.empty:
        return {"low": 0, "medium": 0, "high": 0}
    
    # Define regime thresholds
    low_threshold = volatility.quantile(0.33)
    high_threshold = volatility.quantile(0.67)
    
    # Classify regimes
    regimes = pd.cut(volatility, 
                    bins=[-np.inf, low_threshold, high_threshold, np.inf],
                    labels=['low', 'medium', 'high'])
    
    # Calculate persistence (average run length)
    persistence = {}
    for regime in ['low', 'medium', 'high']:
        regime_periods = (regimes == regime)
        run_lengths = []
        current_run = 0
        
        for is_regime in regime_periods:
            if is_regime:
                current_run += 1
            else:
                if current_run > 0:
                    run_lengths.append(current_run)
                current_run = 0
        
        if current_run > 0:
            run_lengths.append(current_run)
        
        persistence[regime] = np.mean(run_lengths) if run_lengths else 0
    
    return persistence


def _count_regime_transitions(volatility: pd.Series) -> Dict[str, int]:
    """Count transitions between volatility regimes"""
    if volatility.empty:
        return {}
    
    low_threshold = volatility.quantile(0.33)
    high_threshold = volatility.quantile(0.67)
    
    regimes = pd.cut(volatility,
                    bins=[-np.inf, low_threshold, high_threshold, np.inf],
                    labels=['low', 'medium', 'high'])
    
    transitions = {}
    prev_regime = None
    
    for regime in regimes:
        if prev_regime is not None and regime != prev_regime:
            transition_key = f"{prev_regime}_to_{regime}"
            transitions[transition_key] = transitions.get(transition_key, 0) + 1
        prev_regime = regime
    
    return transitions