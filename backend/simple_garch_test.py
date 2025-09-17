"""
Simple GARCH test to identify issues
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)
sys.path.append(os.path.dirname(backend_dir))

print("Testing basic imports...")

try:
    from models.time_series.garch_model import GARCHModel, GARCHPrediction
    print("✅ GARCH model imported successfully")
except Exception as e:
    print(f"❌ GARCH model import failed: {e}")

try:
    from models.time_series.data_preprocessor import TimeSeriesPreprocessor
    print("✅ Data preprocessor imported successfully")
except Exception as e:
    print(f"❌ Data preprocessor import failed: {e}")

print("\nTesting basic data fetching...")

try:
    import yfinance as yf
    data = yf.download("AAPL", period="1y", progress=False)
    print(f"✅ Yahoo Finance data fetched: {len(data)} rows")
    print(f"   Columns: {list(data.columns)}")
    print(f"   Index type: {type(data.index)}")
    print(f"   Data types: {data.dtypes}")
except Exception as e:
    print(f"❌ Yahoo Finance failed: {e}")

print("\nTesting basic preprocessing...")

try:
    preprocessor = TimeSeriesPreprocessor()
    
    # Create simple test data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    prices = pd.Series(
        100 + np.cumsum(np.random.normal(0, 1, len(dates))),
        index=dates,
        name='Close'
    )
    
    simple_df = pd.DataFrame({'Close': prices})
    print(f"✅ Created test data: {len(simple_df)} rows")
    
    # Test returns calculation
    returns = preprocessor.calculate_returns(prices)
    print(f"✅ Returns calculated: {len(returns)} values")
    
    # Test outlier detection
    outliers = preprocessor.detect_outliers(returns)
    print(f"✅ Outliers detected: {outliers.sum()} outliers")
    
    # Test preprocessing
    processed = preprocessor.preprocess("TEST", data=simple_df, clean_outliers=False)
    print(f"✅ Basic preprocessing successful")
    print(f"   Symbol: {processed.symbol}")
    print(f"   Raw prices: {len(processed.raw_prices)}")
    print(f"   Returns: {len(processed.returns)}")
    
except Exception as e:
    print(f"❌ Basic preprocessing failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting GARCH model...")

try:
    # Create synthetic returns data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 100), name='returns')
    
    model = GARCHModel(model_type='GARCH', p=1, q=1)
    print("✅ GARCH model created")
    
    model.fit(returns)
    print("✅ GARCH model fitted")
    
    prediction = model.predict(returns)
    print("✅ GARCH prediction generated")
    print(f"   Predicted volatility: {prediction.predicted_volatility:.4f}")
    print(f"   Signal strength: {prediction.signal_strength:.3f}")
    print(f"   Volatility regime: {prediction.volatility_regime}")
    
except Exception as e:
    print(f"❌ GARCH model failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Simple test completed!")