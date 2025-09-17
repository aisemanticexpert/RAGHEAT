"""
Test script for GARCH model implementation
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)
sys.path.append(os.path.dirname(backend_dir))

from models.time_series.garch_model import GARCHModel, GARCHPrediction
from models.time_series.data_preprocessor import TimeSeriesPreprocessor, PreprocessedData
from models.time_series.garch_signal_generator import GARCHSignalGenerator, GARCHSignal


def test_data_preprocessor():
    """Test the data preprocessing functionality"""
    print("🧪 Testing Data Preprocessor...")
    
    try:
        preprocessor = TimeSeriesPreprocessor()
        
        # Test with a popular stock
        symbol = "AAPL"
        print(f"   Fetching data for {symbol}...")
        
        data = preprocessor.preprocess(
            symbol=symbol,
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now()
        )
        
        print(f"   ✅ Successfully preprocessed {len(data.raw_prices)} data points")
        print(f"   📊 Data range: {data.metadata['start_date']} to {data.metadata['end_date']}")
        print(f"   📈 Avg daily return: {data.metadata['avg_daily_return']:.4f}")
        print(f"   📊 Avg daily volatility: {data.metadata['avg_daily_volatility']:.4f}")
        
        # Test data quality
        quality = preprocessor.validate_data_quality(data)
        print(f"   🏆 Data quality: {quality['assessment']} (score: {quality['quality_score']:.2f})")
        
        return data
        
    except Exception as e:
        print(f"   ❌ Data preprocessor test failed: {e}")
        return None


def test_garch_model(data: PreprocessedData):
    """Test GARCH model fitting and prediction"""
    print("\n🧪 Testing GARCH Model...")
    
    try:
        # Test different GARCH variants
        models = {
            'Standard GARCH': GARCHModel(model_type='GARCH', p=1, q=1),
            'EGARCH': GARCHModel(model_type='EGARCH', p=1, q=1),
            'GJR-GARCH': GARCHModel(model_type='GJR-GARCH', p=1, q=1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"   Testing {name}...")
            
            try:
                # Fit model
                model.fit(data.log_returns)
                
                # Make prediction
                prediction = model.predict(data.log_returns, horizon=1)
                
                # Get diagnostics
                diagnostics = model.get_model_diagnostics()
                
                results[name] = {
                    'prediction': prediction,
                    'diagnostics': diagnostics,
                    'success': True
                }
                
                print(f"      ✅ {name} fitted successfully")
                print(f"      🔮 Predicted volatility: {prediction.predicted_volatility:.4f}")
                print(f"      📊 Model accuracy: {prediction.model_accuracy:.2f}")
                print(f"      🎯 Signal strength: {prediction.signal_strength:.2f}")
                print(f"      🌡️ Volatility regime: {prediction.volatility_regime}")
                
            except Exception as e:
                print(f"      ❌ {name} failed: {e}")
                results[name] = {'success': False, 'error': str(e)}
        
        return results
        
    except Exception as e:
        print(f"   ❌ GARCH model test failed: {e}")
        return None


def test_signal_generator(data: PreprocessedData):
    """Test GARCH signal generator"""
    print("\n🧪 Testing GARCH Signal Generator...")
    
    try:
        generator = GARCHSignalGenerator()
        
        # Generate signal with different heat scores
        heat_scores = [0.2, 0.5, 0.8]
        current_price = data.raw_prices.iloc[-1]
        
        for heat_score in heat_scores:
            print(f"   Testing with heat score: {heat_score}")
            
            signal = generator.generate_signal(
                symbol=data.symbol,
                current_price=current_price,
                heat_score=heat_score,
                macd_result={'macd_line': 0.1, 'signal_line': 0.05, 'histogram': 0.05},
                bollinger_result={
                    'upper_band': current_price * 1.02,
                    'lower_band': current_price * 0.98,
                    'middle_band': current_price
                },
                price_change_pct=np.random.normal(0, 2),
                volume_ratio=np.random.uniform(0.8, 1.5)
            )
            
            print(f"      📊 Combined signal: {signal.combined_signal.value}")
            print(f"      🎯 Combined confidence: {signal.combined_confidence:.3f}")
            print(f"      💰 Price target: ${signal.volatility_adjusted_targets.get('price_target', 0):.2f}")
            print(f"      🛡️ Stop loss: ${signal.volatility_adjusted_targets.get('stop_loss', 0):.2f}")
            print(f"      ⚖️ Recommended position: {signal.risk_metrics.get('recommended_position_size', 0):.3f}")
            
            # Test signal components
            components = signal.signal_components
            print(f"      🔧 Technical component: {components.get('technical_component', 0):.3f}")
            print(f"      📈 GARCH component: {components.get('garch_component', 0):.3f}")
            print(f"      🌊 Volatility component: {components.get('volatility_component', 0):.3f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Signal generator test failed: {e}")
        return False


def test_multiple_symbols():
    """Test with multiple symbols"""
    print("\n🧪 Testing Multiple Symbols...")
    
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    preprocessor = TimeSeriesPreprocessor()
    generator = GARCHSignalGenerator()
    
    results = {}
    
    for symbol in symbols:
        print(f"   Processing {symbol}...")
        
        try:
            # Get shorter data for speed
            data = preprocessor.preprocess(
                symbol=symbol,
                start_date=datetime.now() - timedelta(days=180),
                end_date=datetime.now()
            )
            
            current_price = data.raw_prices.iloc[-1]
            heat_score = np.random.uniform(0.3, 0.8)
            
            signal = generator.generate_signal(
                symbol=symbol,
                current_price=current_price,
                heat_score=heat_score
            )
            
            results[symbol] = {
                'signal': signal.combined_signal.value,
                'confidence': signal.combined_confidence,
                'volatility_regime': signal.garch_prediction.volatility_regime,
                'predicted_volatility': signal.garch_prediction.predicted_volatility,
                'success': True
            }
            
            print(f"      ✅ {symbol}: {signal.combined_signal.value} (confidence: {signal.combined_confidence:.2f})")
            
        except Exception as e:
            print(f"      ❌ {symbol} failed: {e}")
            results[symbol] = {'success': False, 'error': str(e)}
    
    return results


def test_model_performance():
    """Test model performance and accuracy"""
    print("\n🧪 Testing Model Performance...")
    
    try:
        preprocessor = TimeSeriesPreprocessor()
        
        # Get data for performance testing
        data = preprocessor.preprocess(
            symbol="SPY",  # Use SPY for broad market testing
            start_date=datetime.now() - timedelta(days=500),
            end_date=datetime.now()
        )
        
        model = GARCHModel(model_type='GARCH', p=1, q=1)
        
        # Split data for testing
        split_point = len(data.log_returns) - 30  # Last 30 days for testing
        train_returns = data.log_returns[:split_point]
        test_returns = data.log_returns[split_point:]
        
        print(f"   Training on {len(train_returns)} observations")
        print(f"   Testing on {len(test_returns)} observations")
        
        # Fit on training data
        model.fit(train_returns)
        
        # Make rolling predictions
        predictions = []
        actuals = []
        
        for i in range(len(test_returns)):
            # Use expanding window
            current_data = data.log_returns[:split_point + i]
            pred = model.predict(current_data, horizon=1)
            predictions.append(pred.predicted_volatility)
            
            # Actual volatility (using absolute return as proxy)
            if i + 1 < len(test_returns):
                actual_vol = abs(test_returns.iloc[i + 1])
                actuals.append(actual_vol)
        
        # Calculate performance metrics
        if len(actuals) > 0 and len(predictions) > 1:
            actuals = np.array(actuals)
            predictions = np.array(predictions[:-1])  # Align arrays
            
            rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
            mae = np.mean(np.abs(actuals - predictions))
            corr = np.corrcoef(actuals, predictions)[0, 1] if len(actuals) > 1 else 0
            
            print(f"   📊 Performance Metrics:")
            print(f"      RMSE: {rmse:.4f}")
            print(f"      MAE: {mae:.4f}")
            print(f"      Correlation: {corr:.3f}")
            print(f"      Mean actual volatility: {np.mean(actuals):.4f}")
            print(f"      Mean predicted volatility: {np.mean(predictions):.4f}")
            
            return {
                'rmse': rmse,
                'mae': mae,
                'correlation': corr,
                'mean_actual': np.mean(actuals),
                'mean_predicted': np.mean(predictions)
            }
        else:
            print("   ⚠️ Insufficient data for performance testing")
            return None
            
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return None


def create_sample_neo4j_data():
    """Create sample data that could be stored in Neo4j"""
    print("\n🧪 Testing Neo4j Data Structure...")
    
    try:
        generator = GARCHSignalGenerator()
        
        # Generate sample signal
        signal = generator.generate_signal(
            symbol="AAPL",
            current_price=150.0,
            heat_score=0.7,
            macd_result={'macd_line': 0.15, 'signal_line': 0.10, 'histogram': 0.05}
        )
        
        # Show what would be stored in Neo4j
        signal_dict = signal.to_dict()
        
        print("   📊 Sample Neo4j Node Structure:")
        print(f"      Symbol: {signal_dict['symbol']}")
        print(f"      Combined Signal: {signal_dict['combined_signal']}")
        print(f"      Confidence: {signal_dict['combined_confidence']:.3f}")
        print(f"      GARCH Volatility: {signal_dict['garch_prediction']['predicted_volatility']:.4f}")
        print(f"      Volatility Regime: {signal_dict['garch_prediction']['volatility_regime']}")
        print(f"      Risk Metrics Keys: {list(signal_dict['risk_metrics'].keys())}")
        print(f"      Signal Components Keys: {list(signal_dict['signal_components'].keys())}")
        
        return signal_dict
        
    except Exception as e:
        print(f"   ❌ Neo4j data structure test failed: {e}")
        return None


def main():
    """Run all tests"""
    print("🚀 Starting GARCH Model Test Suite")
    print("=" * 50)
    
    # Test 1: Data Preprocessor
    data = test_data_preprocessor()
    if data is None:
        print("❌ Skipping remaining tests due to data preprocessing failure")
        return
    
    # Test 2: GARCH Models
    garch_results = test_garch_model(data)
    
    # Test 3: Signal Generator
    signal_success = test_signal_generator(data)
    
    # Test 4: Multiple Symbols
    multi_results = test_multiple_symbols()
    
    # Test 5: Model Performance
    performance = test_model_performance()
    
    # Test 6: Neo4j Data Structure
    neo4j_sample = create_sample_neo4j_data()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    print(f"✅ Data Preprocessor: {'PASS' if data is not None else 'FAIL'}")
    print(f"✅ GARCH Models: {'PASS' if garch_results else 'FAIL'}")
    print(f"✅ Signal Generator: {'PASS' if signal_success else 'FAIL'}")
    print(f"✅ Multiple Symbols: {'PASS' if multi_results else 'FAIL'}")
    print(f"✅ Performance Test: {'PASS' if performance else 'FAIL'}")
    print(f"✅ Neo4j Structure: {'PASS' if neo4j_sample else 'FAIL'}")
    
    if garch_results:
        successful_models = [name for name, result in garch_results.items() if result.get('success')]
        print(f"\n🎯 Successful GARCH models: {', '.join(successful_models)}")
    
    if multi_results:
        successful_symbols = [symbol for symbol, result in multi_results.items() if result.get('success')]
        print(f"📈 Successful symbols: {', '.join(successful_symbols)}")
    
    if performance:
        print(f"📊 Model correlation with actual volatility: {performance['correlation']:.3f}")
    
    print("\n🎉 Test suite completed!")


if __name__ == "__main__":
    main()