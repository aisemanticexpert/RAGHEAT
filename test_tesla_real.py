#!/usr/bin/env python3
"""Test real Tesla data"""

import yfinance as yf
from datetime import datetime

def test_tesla():
    """Get real Tesla price"""
    print("=== TESTING REAL TESLA DATA ===")
    
    try:
        ticker = yf.Ticker('TSLA')
        info = ticker.info
        current_price = info.get('currentPrice', 'N/A')
        
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            latest = hist.iloc[-1]
            latest_close = latest['Close']
            print(f"TSLA Current Price: ${current_price}")
            print(f"TSLA Latest Close: ${latest_close:.2f}")
            print(f"TSLA Volume: {info.get('regularMarketVolume', 'N/A'):,}" if isinstance(info.get('regularMarketVolume'), (int, float)) else f"TSLA Volume: {info.get('regularMarketVolume', 'N/A')}")
            print(f"Market Cap: ${info.get('marketCap', 'N/A'):,}" if isinstance(info.get('marketCap'), (int, float)) else f"Market Cap: {info.get('marketCap', 'N/A')}")
            return latest_close
        else:
            print("No Tesla data available")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    real_price = test_tesla()
    print(f"\nReal Tesla Price: ${real_price:.2f}" if real_price else "Failed to get real Tesla price")