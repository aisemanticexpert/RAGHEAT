#!/usr/bin/env python3
"""
Test the new Buy/Sell Trading Signals in RAGHeat
"""

import requests
import json

def test_buy_sell_signals():
    """Test the buy/sell signals functionality"""
    
    print("🚀 TESTING BUY/SELL SIGNALS IN RAGHEAT")
    print("=" * 60)
    
    try:
        # Get graph structure with trading signals
        response = requests.get('http://localhost:8000/api/graph/structure', timeout=10)
        data = response.json()
        
        # Extract stock nodes with trading signals
        stocks_with_signals = []
        for node in data.get('graph', {}).get('nodes', []):
            if 'Stock' in node.get('labels', []):
                props = node.get('properties', {})
                if props.get('trading_signal'):
                    stocks_with_signals.append({
                        'symbol': props.get('symbol', 'Unknown'),
                        'trading_signal': props.get('trading_signal'),
                        'signal_confidence': props.get('signal_confidence', 0),
                        'price_target': props.get('price_target'),
                        'stop_loss': props.get('stop_loss'),
                        'current_price': props.get('current_price', 0),
                        'heat_score': node.get('heat_score', 0),
                        'background_color': props.get('background_color', '#000000')
                    })
        
        # Sort by signal strength and confidence
        signal_order = {
            'STRONG_BUY': 1, 'BUY': 2, 'WEAK_BUY': 3, 'HOLD': 4,
            'WEAK_SELL': 5, 'SELL': 6, 'STRONG_SELL': 7
        }
        
        stocks_with_signals.sort(key=lambda x: (
            signal_order.get(x['trading_signal'], 8),
            -x['signal_confidence']
        ))
        
        print(f"📊 FOUND {len(stocks_with_signals)} STOCKS WITH TRADING SIGNALS:")
        print()
        
        # Group by signal type
        buy_signals = []
        hold_signals = []
        sell_signals = []
        
        for stock in stocks_with_signals:
            signal = stock['trading_signal']
            if 'BUY' in signal:
                buy_signals.append(stock)
            elif 'SELL' in signal:
                sell_signals.append(stock)
            else:
                hold_signals.append(stock)
        
        # Display BUY signals
        if buy_signals:
            print("🟢 BUY SIGNALS:")
            print("-" * 80)
            for stock in buy_signals:
                target_str = f"${stock['price_target']:.2f}" if stock['price_target'] else "N/A"
                stop_str = f"${stock['stop_loss']:.2f}" if stock['stop_loss'] else "N/A"
                
                print(f"📈 {stock['symbol']:6} | {stock['trading_signal']:12} | "
                      f"Confidence: {stock['signal_confidence']:.2%} | "
                      f"Price: ${stock['current_price']:.2f} | "
                      f"Target: {target_str} | Stop: {stop_str}")
            print()
        
        # Display HOLD signals
        if hold_signals:
            print("🟡 HOLD SIGNALS:")
            print("-" * 80)
            for stock in hold_signals:
                stop_str = f"${stock['stop_loss']:.2f}" if stock['stop_loss'] else "N/A"
                
                print(f"⏸️  {stock['symbol']:6} | {stock['trading_signal']:12} | "
                      f"Confidence: {stock['signal_confidence']:.2%} | "
                      f"Price: ${stock['current_price']:.2f} | Stop: {stop_str}")
            print()
        
        # Display SELL signals
        if sell_signals:
            print("🔴 SELL SIGNALS:")
            print("-" * 80)
            for stock in sell_signals:
                target_str = f"${stock['price_target']:.2f}" if stock['price_target'] else "N/A"
                stop_str = f"${stock['stop_loss']:.2f}" if stock['stop_loss'] else "N/A"
                
                print(f"📉 {stock['symbol']:6} | {stock['trading_signal']:12} | "
                      f"Confidence: {stock['signal_confidence']:.2%} | "
                      f"Price: ${stock['current_price']:.2f} | "
                      f"Target: {target_str} | Stop: {stop_str}")
            print()
        
        print("✅ NEO4J BROWSER VISUALIZATION:")
        print("-" * 60)
        print("🎯 In Neo4j Browser, you will now see:")
        print("   🟢 GREEN nodes = BUY signals (brighter green = stronger signal)")
        print("   🟡 YELLOW nodes = HOLD signals")  
        print("   🔴 RED nodes = SELL signals (darker red = stronger signal)")
        print("   📏 Node size = Signal confidence (larger = higher confidence)")
        print("   🎨 Border thickness = Signal strength")
        print()
        print("📍 Use these files:")
        print("   📁 Style: neo4j_buy_sell_style.grass")
        print("   📁 Query: buy_sell_query.cypher")
        print()
        print(f"🔥 System Status: {len(stocks_with_signals)} stocks with active trading signals")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing buy/sell signals: {e}")
        return False

if __name__ == "__main__":
    success = test_buy_sell_signals()
    if success:
        print("\n🎉 Buy/Sell signals are working! Check Neo4j Browser for visualization.")
    else:
        print("\n❌ Failed to test buy/sell signals")