#!/usr/bin/env python3
"""Test real market data sources"""

import yfinance as yf
from datetime import datetime

def test_yfinance():
    """Test yfinance for real AAPL data"""
    print("=== TESTING YFINANCE FOR REAL MARKET DATA ===")
    
    try:
        # Get AAPL ticker
        ticker = yf.Ticker('AAPL')
        
        # Get current info
        info = ticker.info
        current_price = info.get('currentPrice', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        volume = info.get('regularMarketVolume', 'N/A')
        
        print(f"AAPL Current Price: ${current_price}")
        print(f"Market Cap: ${market_cap:,}" if isinstance(market_cap, (int, float)) else f"Market Cap: {market_cap}")
        print(f"Volume: {volume:,}" if isinstance(volume, (int, float)) else f"Volume: {volume}")
        
        # Get recent price data
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            latest = hist.iloc[-1]
            print(f"Latest Close: ${latest['Close']:.2f}")
            print(f"Latest Volume: {latest['Volume']:,.0f}")
            print(f"Timestamp: {latest.name}")
        else:
            print("No historical data available")
            
        return current_price != 'N/A' and isinstance(current_price, (int, float))
        
    except Exception as e:
        print(f"Error testing yfinance: {e}")
        return False

if __name__ == "__main__":
    success = test_yfinance()
    print(f"\nYfinance test {'PASSED' if success else 'FAILED'}")