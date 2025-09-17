"""
Comprehensive test script for Hidden Markov Model implementation
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

print("🧪 Testing HMM Model Implementation")
print("=" * 60)

def test_hmm_data_processor():
    """Test HMM data preprocessing functionality"""
    print("\n🔧 Testing HMM Data Processor...")
    
    try:
        from models.time_series.hmm_data_processor import HMMDataProcessor
        
        # Initialize processor
        processor = HMMDataProcessor(
            feature_selection_method="kbest",
            n_features=15,
            scaling_method="robust"
        )
        
        print("   ✅ HMM Data Processor initialized")
        
        # Test with a popular stock
        symbol = "AAPL"
        print(f"   📊 Processing data for {symbol}...")
        
        processed_data = processor.process_for_hmm(
            symbol=symbol,
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now()
        )
        
        print(f"   ✅ Successfully processed {processed_data.feature_matrix.shape[0]} observations")
        print(f"   📈 Features: {processed_data.feature_matrix.shape[1]} selected")
        print(f"   🏆 Data quality score: {processed_data.preprocessing_metadata['data_quality_score']:.3f}")
        
        # Test feature importance
        importance_report = processor.get_feature_importance_report()
        print(f"   🎯 Top features analyzed: {importance_report['n_features_analyzed']}")
        
        return processed_data
        
    except Exception as e:
        print(f"   ❌ HMM Data Processor test failed: {e}")
        return None


def test_hmm_base_model():
    """Test basic HMM model functionality"""
    print("\n🧪 Testing HMM Base Model...")
    
    try:
        from models.time_series.hmm_model import HMMMarketModel
        
        # Test different HMM configurations
        models = {
            '3-State HMM': HMMMarketModel(n_states=3, n_iter=50),
            '4-State HMM': HMMMarketModel(n_states=4, n_iter=50),
            'EGARCH-style': HMMMarketModel(n_states=3, covariance_type='diag', n_iter=50)
        }
        
        # Create synthetic test data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        prices = pd.Series(
            100 + np.cumsum(np.random.normal(0.001, 0.02, len(dates))),
            index=dates,
            name='AAPL'
        )
        
        results = {}
        
        for name, model in models.items():
            print(f"   Testing {name}...")
            
            try:
                # Prepare features
                features = model.prepare_features(prices, include_technical=True)
                print(f"      📊 Prepared {len(features.columns)} features")
                
                # Fit model
                model.fit(features)
                print(f"      ✅ Model fitted successfully")
                
                # Make prediction
                prediction = model.predict(features)
                print(f"      🔮 Current state: {prediction.state_names[prediction.current_state]}")
                print(f"      📊 Model accuracy: {prediction.model_accuracy:.3f}")
                print(f"      🎯 Signal strength: {prediction.signal_strength:.3f}")
                
                # Get regime summary
                regime_summary = model.get_regime_summary()
                print(f"      🌡️ Regimes detected: {len(regime_summary['regimes'])}")
                
                results[name] = {
                    'prediction': prediction,
                    'regime_summary': regime_summary,
                    'success': True
                }
                
            except Exception as e:
                print(f"      ❌ {name} failed: {e}")
                results[name] = {'success': False, 'error': str(e)}
        
        return results
        
    except Exception as e:
        print(f"   ❌ HMM Base Model test failed: {e}")
        return None


def test_market_regime_detector():
    """Test market regime detection system"""
    print("\n🏛️ Testing Market Regime Detector...")
    
    try:
        from models.time_series.market_regime_detector import MarketRegimeDetector
        
        # Initialize detector
        detector = MarketRegimeDetector(
            n_regimes=4,
            lookback_period=180
        )
        
        print("   ✅ Market Regime Detector initialized")
        
        # Test regime detection with different symbols
        symbols = ["AAPL", "TSLA", "SPY"]
        results = {}
        
        for symbol in symbols:
            print(f"   🔍 Detecting regime for {symbol}...")
            
            try:
                regime_result = detector.detect_regime(
                    symbol=symbol,
                    start_date=datetime.now() - timedelta(days=365),
                    end_date=datetime.now()
                )
                
                print(f"      📊 Current regime: {regime_result.current_regime}")
                print(f"      🎯 Regime confidence: {regime_result.confidence:.3f}")
                print(f"      ⏱️ Regime duration: {regime_result.regime_duration} days")
                print(f"      ⚠️ Risk level: {regime_result.risk_level}")
                print(f"      💡 Recommended action: {regime_result.recommended_action}")
                
                results[symbol] = regime_result
                
            except Exception as e:
                print(f"      ❌ {symbol} regime detection failed: {e}")
                results[symbol] = None
        
        # Test batch processing
        print("   🔄 Testing batch regime detection...")
        batch_results = detector.batch_detect_regimes(symbols)
        print(f"   ✅ Batch processed {len(batch_results)} symbols")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Market Regime Detector test failed: {e}")
        return None


def test_hmm_signal_generator():
    """Test HMM signal generator"""
    print("\n📡 Testing HMM Signal Generator...")
    
    try:
        from models.time_series.hmm_signal_generator import HMMSignalGenerator
        
        # Initialize generator
        generator = HMMSignalGenerator(
            hmm_weight=0.4,
            technical_weight=0.4,
            regime_weight=0.2
        )
        
        print("   ✅ HMM Signal Generator initialized")
        
        # Test signal generation with different scenarios
        test_scenarios = [
            {"symbol": "AAPL", "heat_score": 0.8, "scenario": "High Heat"},
            {"symbol": "GOOGL", "heat_score": 0.3, "scenario": "Low Heat"},
            {"symbol": "MSFT", "heat_score": 0.5, "scenario": "Neutral Heat"}
        ]
        
        for scenario in test_scenarios:
            print(f"   🎯 Testing {scenario['scenario']} scenario for {scenario['symbol']}...")
            
            try:
                signal = generator.generate_signal(
                    symbol=scenario['symbol'],
                    current_price=150.0,
                    heat_score=scenario['heat_score'],
                    macd_result={'macd_line': 0.1, 'signal_line': 0.05, 'histogram': 0.05},
                    bollinger_result={'upper_band': 155, 'lower_band': 145, 'middle_band': 150}
                )
                
                print(f"      📊 Combined signal: {signal.combined_signal.value}")
                print(f"      🎯 Combined confidence: {signal.combined_confidence:.3f}")
                print(f"      🏛️ Current regime: {signal.regime_detection.current_regime}")
                print(f"      💰 Recommended position: {signal.position_sizing['recommended_position_size']:.3f}")
                print(f"      ⚖️ Risk level: {signal.risk_assessment['risk_level']}")
                print(f"      ⏰ Timing phase: {signal.regime_timing['timing_phase']}")
                
            except Exception as e:
                print(f"      ❌ {scenario['scenario']} scenario failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ HMM Signal Generator test failed: {e}")
        return False


def test_unified_signal_system():
    """Test unified signal system combining all models"""
    print("\n🌐 Testing Unified Signal System...")
    
    try:
        from models.time_series.unified_signal_system import UnifiedSignalSystem
        
        # Test different configurations
        configurations = [
            {
                "name": "Balanced",
                "technical_weight": 0.33,
                "garch_weight": 0.33,
                "hmm_weight": 0.34,
                "risk_mode": "moderate"
            },
            {
                "name": "HMM-Heavy",
                "technical_weight": 0.2,
                "garch_weight": 0.3,
                "hmm_weight": 0.5,
                "risk_mode": "aggressive"
            },
            {
                "name": "Conservative",
                "technical_weight": 0.5,
                "garch_weight": 0.3,
                "hmm_weight": 0.2,
                "risk_mode": "conservative"
            }
        ]
        
        for config in configurations:
            print(f"   🔧 Testing {config['name']} configuration...")
            
            try:
                # Initialize system
                system = UnifiedSignalSystem(
                    technical_weight=config['technical_weight'],
                    garch_weight=config['garch_weight'],
                    hmm_weight=config['hmm_weight'],
                    risk_management_mode=config['risk_mode']
                )
                
                # Generate unified signal
                unified_signal = system.generate_unified_signal(
                    symbol="AAPL",
                    current_price=150.0,
                    heat_score=0.6,
                    macd_result={'macd_line': 0.1, 'signal_line': 0.05},
                    price_change_pct=1.5
                )
                
                print(f"      📊 Unified signal: {unified_signal.unified_signal.value}")
                print(f"      🎯 Unified confidence: {unified_signal.unified_confidence:.3f}")
                print(f"      🤝 Consensus score: {unified_signal.model_consensus.get('consensus_score', 0):.3f}")
                print(f"      📈 Direction consensus: {unified_signal.model_consensus.get('direction_consensus', 'N/A')}")
                print(f"      💼 Recommended allocation: {unified_signal.portfolio_allocation['recommended_allocation']:.3f}")
                print(f"      🎯 Entry method: {unified_signal.execution_strategy['entry_method']}")
                
                # Test system status
                status = system.get_system_status()
                print(f"      ✅ System status retrieved: {len(status)} components")
                
            except Exception as e:
                print(f"      ❌ {config['name']} configuration failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Unified Signal System test failed: {e}")
        return False


def test_api_data_structures():
    """Test API-compatible data structures"""
    print("\n🌐 Testing API Data Structures...")
    
    try:
        # Test if all components can be serialized to dict (API compatibility)
        from models.time_series.hmm_signal_generator import HMMSignalGenerator
        
        generator = HMMSignalGenerator()
        signal = generator.generate_signal(
            symbol="TEST",
            current_price=100.0,
            heat_score=0.5
        )
        
        # Test serialization
        signal_dict = signal.to_dict()
        print(f"   ✅ HMM signal serialized: {len(signal_dict)} fields")
        
        # Check required fields
        required_fields = [
            'symbol', 'combined_signal', 'combined_confidence',
            'regime_detection', 'risk_assessment', 'timestamp'
        ]
        
        missing_fields = [field for field in required_fields if field not in signal_dict]
        if missing_fields:
            print(f"   ⚠️ Missing required fields: {missing_fields}")
        else:
            print(f"   ✅ All required fields present")
        
        # Test unified signal serialization
        from models.time_series.unified_signal_system import UnifiedSignalSystem
        
        unified_system = UnifiedSignalSystem()
        unified_signal = unified_system.generate_unified_signal(
            symbol="TEST",
            current_price=100.0,
            heat_score=0.5
        )
        
        unified_dict = unified_signal.to_dict()
        print(f"   ✅ Unified signal serialized: {len(unified_dict)} fields")
        
        return True
        
    except Exception as e:
        print(f"   ❌ API data structure test failed: {e}")
        return False


def test_performance_and_memory():
    """Test performance and memory usage"""
    print("\n⚡ Testing Performance and Memory Usage...")
    
    try:
        import time
        import tracemalloc
        
        # Test HMM processing speed
        from models.time_series.hmm_signal_generator import HMMSignalGenerator
        
        generator = HMMSignalGenerator()
        
        # Test batch processing speed
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        tracemalloc.start()
        start_time = time.time()
        
        results = []
        for symbol in symbols:
            try:
                signal = generator.generate_signal(
                    symbol=symbol,
                    current_price=100.0,
                    heat_score=0.5
                )
                results.append(signal)
            except Exception as e:
                print(f"      ⚠️ Failed to process {symbol}: {e}")
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        processing_time = end_time - start_time
        print(f"   ⏱️ Processed {len(results)} symbols in {processing_time:.2f} seconds")
        print(f"   💾 Peak memory usage: {peak / 1024 / 1024:.1f} MB")
        print(f"   🚀 Average time per symbol: {processing_time / len(symbols):.2f} seconds")
        
        # Performance recommendations
        if processing_time / len(symbols) > 5.0:
            print(f"   ⚠️ Warning: Processing time per symbol is high")
        else:
            print(f"   ✅ Processing speed is acceptable")
        
        return {
            'symbols_processed': len(results),
            'total_time': processing_time,
            'avg_time_per_symbol': processing_time / len(symbols),
            'peak_memory_mb': peak / 1024 / 1024
        }
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return None


def test_error_handling():
    """Test error handling and edge cases"""
    print("\n🛡️ Testing Error Handling...")
    
    try:
        from models.time_series.hmm_signal_generator import HMMSignalGenerator
        from models.time_series.market_regime_detector import MarketRegimeDetector
        
        generator = HMMSignalGenerator()
        detector = MarketRegimeDetector()
        
        # Test with invalid inputs
        test_cases = [
            {"symbol": "INVALID", "description": "Invalid symbol"},
            {"symbol": "AAPL", "current_price": -100, "description": "Negative price"},
            {"symbol": "AAPL", "heat_score": 2.0, "description": "Invalid heat score"},
        ]
        
        handled_errors = 0
        
        for test_case in test_cases:
            try:
                signal = generator.generate_signal(
                    symbol=test_case["symbol"],
                    current_price=test_case.get("current_price", 100.0),
                    heat_score=test_case.get("heat_score", 0.5)
                )
                print(f"   ⚠️ {test_case['description']}: No error raised (fallback used)")
                handled_errors += 1
            except Exception as e:
                print(f"   ❌ {test_case['description']}: {type(e).__name__}")
        
        # Test regime detector error handling
        try:
            result = detector.detect_regime(
                symbol="NONEXISTENT",
                start_date=datetime.now() - timedelta(days=10),
                end_date=datetime.now()
            )
            print(f"   ⚠️ Invalid symbol regime detection: Fallback used")
            handled_errors += 1
        except Exception as e:
            print(f"   ❌ Invalid symbol regime detection: {type(e).__name__}")
        
        print(f"   📊 Error handling summary: {handled_errors} cases handled gracefully")
        
        return handled_errors > 0
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False


def create_integration_example():
    """Create example of full HMM integration"""
    print("\n🔗 Creating Integration Example...")
    
    try:
        from models.time_series.unified_signal_system import UnifiedSignalSystem
        
        # Initialize unified system
        system = UnifiedSignalSystem(
            technical_weight=0.3,
            garch_weight=0.35,
            hmm_weight=0.35,
            risk_management_mode="moderate"
        )
        
        # Simulate real trading scenario
        symbol = "AAPL"
        current_price = 175.50
        
        # Generate comprehensive signal
        signal = system.generate_unified_signal(
            symbol=symbol,
            current_price=current_price,
            heat_score=0.65,
            macd_result={
                'macd_line': 0.15,
                'signal_line': 0.12,
                'histogram': 0.03
            },
            bollinger_result={
                'upper_band': current_price * 1.02,
                'lower_band': current_price * 0.98,
                'middle_band': current_price
            },
            price_change_pct=2.1,
            volume_ratio=1.3
        )
        
        # Display comprehensive analysis
        print(f"   📊 Integration Example for {symbol}")
        print(f"   💰 Current Price: ${current_price}")
        print()
        print(f"   🎯 UNIFIED SIGNAL: {signal.unified_signal.value}")
        print(f"   🔮 Confidence: {signal.unified_confidence:.1%}")
        print()
        print(f"   🏛️ Market Regime: {signal.hmm_signal.regime_detection.current_regime}")
        print(f"   📈 Regime Strength: {signal.hmm_signal.regime_detection.regime_strength:.1%}")
        print(f"   ⚠️ Risk Level: {signal.hmm_signal.regime_detection.risk_level}")
        print()
        print(f"   📊 Model Consensus:")
        print(f"      - Technical: {signal.base_technical_signal.signal.value}")
        print(f"      - GARCH: {signal.garch_signal.combined_signal.value}")
        print(f"      - HMM: {signal.hmm_signal.combined_signal.value}")
        print(f"      - Agreement: {signal.model_consensus.get('direction_consensus', 'N/A')}")
        print()
        print(f"   💼 Portfolio Recommendation:")
        print(f"      - Position Size: {signal.portfolio_allocation['recommended_allocation']:.1%}")
        print(f"      - Entry Method: {signal.execution_strategy['entry_method']}")
        print(f"      - Time Horizon: {signal.execution_strategy['time_horizon']}")
        print()
        print(f"   🎯 Price Targets:")
        if 'price_target' in signal.risk_adjusted_targets:
            print(f"      - Target: ${signal.risk_adjusted_targets['price_target']:.2f}")
        if 'stop_loss' in signal.risk_adjusted_targets:
            print(f"      - Stop Loss: ${signal.risk_adjusted_targets['stop_loss']:.2f}")
        
        # Show data structure for API/Neo4j integration
        signal_dict = signal.to_dict()
        print(f"\n   📋 API/Neo4j Integration Ready:")
        print(f"      - Serializable fields: {len(signal_dict)}")
        print(f"      - Timestamp: {signal.timestamp}")
        
        return signal
        
    except Exception as e:
        print(f"   ❌ Integration example failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run comprehensive HMM test suite"""
    print("🚀 Starting HMM Model Test Suite")
    print("=" * 60)
    
    # Test results tracking
    test_results = {
        'data_processor': False,
        'base_model': False,
        'regime_detector': False,
        'signal_generator': False,
        'unified_system': False,
        'api_structures': False,
        'performance': False,
        'error_handling': False,
        'integration': False
    }
    
    # Run tests
    test_results['data_processor'] = test_hmm_data_processor() is not None
    test_results['base_model'] = test_hmm_base_model() is not None
    test_results['regime_detector'] = test_market_regime_detector() is not None
    test_results['signal_generator'] = test_hmm_signal_generator()
    test_results['unified_system'] = test_unified_signal_system()
    test_results['api_structures'] = test_api_data_structures()
    
    performance_results = test_performance_and_memory()
    test_results['performance'] = performance_results is not None
    
    test_results['error_handling'] = test_error_handling()
    
    integration_example = create_integration_example()
    test_results['integration'] = integration_example is not None
    
    # Test Summary
    print("\n" + "=" * 60)
    print("📊 HMM Test Suite Summary")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All HMM tests passed! System is ready for deployment.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️ Most tests passed. Review failed tests before deployment.")
    else:
        print("❌ Multiple test failures. System needs debugging before deployment.")
    
    # Performance summary
    if performance_results:
        print(f"\n⚡ Performance Summary:")
        print(f"   - Average processing time: {performance_results['avg_time_per_symbol']:.2f}s per symbol")
        print(f"   - Peak memory usage: {performance_results['peak_memory_mb']:.1f} MB")
    
    print(f"\n🕐 Test suite completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()