"""
Revolutionary Synthetic Stock Data Generator
Creates realistic market data with volatility patterns, correlations, and market regimes
"""
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import random
import math
from pathlib import Path

@dataclass
class StockData:
    symbol: str
    timestamp: str
    price: float
    volume: int
    open_price: float
    high: float
    low: float
    close: float
    change_percent: float
    volatility: float
    heat_score: float
    sector: str
    market_cap: float
    correlation_signals: Dict[str, float]

class SyntheticDataGenerator:
    def __init__(self):
        self.data_dir = Path("data/synthetic_stocks")
        self.real_data_dir = Path("data/real_market_data")
        self.processed_dir = Path("data/processed")
        
        # Ensure directories exist
        for dir_path in [self.data_dir, self.real_data_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Define major stocks and sectors
        self.stocks = {
            'AAPL': {'sector': 'Technology', 'volatility_base': 0.25, 'correlation_group': 'tech'},
            'GOOGL': {'sector': 'Technology', 'volatility_base': 0.28, 'correlation_group': 'tech'},
            'MSFT': {'sector': 'Technology', 'volatility_base': 0.24, 'correlation_group': 'tech'},
            'AMZN': {'sector': 'Technology', 'volatility_base': 0.32, 'correlation_group': 'tech'},
            'TSLA': {'sector': 'Automotive', 'volatility_base': 0.45, 'correlation_group': 'growth'},
            'NVDA': {'sector': 'Technology', 'volatility_base': 0.38, 'correlation_group': 'ai'},
            'META': {'sector': 'Technology', 'volatility_base': 0.35, 'correlation_group': 'social'},
            'NFLX': {'sector': 'Entertainment', 'volatility_base': 0.33, 'correlation_group': 'streaming'},
            'JPM': {'sector': 'Finance', 'volatility_base': 0.22, 'correlation_group': 'banking'},
            'BAC': {'sector': 'Finance', 'volatility_base': 0.25, 'correlation_group': 'banking'},
            'JNJ': {'sector': 'Healthcare', 'volatility_base': 0.18, 'correlation_group': 'healthcare'},
            'PFE': {'sector': 'Healthcare', 'volatility_base': 0.24, 'correlation_group': 'pharma'},
            'XOM': {'sector': 'Energy', 'volatility_base': 0.28, 'correlation_group': 'oil'},
            'CVX': {'sector': 'Energy', 'volatility_base': 0.26, 'correlation_group': 'oil'},
            'WMT': {'sector': 'Retail', 'volatility_base': 0.20, 'correlation_group': 'retail'},
            'HD': {'sector': 'Retail', 'volatility_base': 0.22, 'correlation_group': 'retail'},
            'PG': {'sector': 'Consumer', 'volatility_base': 0.16, 'correlation_group': 'staples'},
            'KO': {'sector': 'Consumer', 'volatility_base': 0.15, 'correlation_group': 'staples'},
            'DIS': {'sector': 'Entertainment', 'volatility_base': 0.29, 'correlation_group': 'media'},
            'V': {'sector': 'Finance', 'volatility_base': 0.21, 'correlation_group': 'payments'}
        }
        
        # Market regime states
        self.market_regimes = {
            'bull': {'trend_factor': 0.0008, 'volatility_multiplier': 0.8},
            'bear': {'trend_factor': -0.0006, 'volatility_multiplier': 1.4},
            'sideways': {'trend_factor': 0.0001, 'volatility_multiplier': 1.0},
            'crash': {'trend_factor': -0.003, 'volatility_multiplier': 2.5}
        }
        
        # Current market state
        self.current_regime = 'sideways'
        self.regime_duration = 0
        self.max_regime_duration = random.randint(100, 500)
        
        # Price history for each stock
        self.price_history = {}
        self.initialize_prices()
        
        # Correlation matrix
        self.correlation_matrix = self.build_correlation_matrix()
        
        # Heat propagation network
        self.heat_network = self.build_heat_network()
        
    def initialize_prices(self):
        """Initialize realistic starting prices for all stocks"""
        base_prices = {
            'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 330.0, 'AMZN': 3200.0,
            'TSLA': 800.0, 'NVDA': 450.0, 'META': 320.0, 'NFLX': 420.0,
            'JPM': 140.0, 'BAC': 35.0, 'JNJ': 165.0, 'PFE': 45.0,
            'XOM': 85.0, 'CVX': 155.0, 'WMT': 145.0, 'HD': 310.0,
            'PG': 150.0, 'KO': 58.0, 'DIS': 110.0, 'V': 220.0
        }
        
        for symbol, base_price in base_prices.items():
            # Add some random variation to base prices
            current_price = base_price * (1 + np.random.normal(0, 0.1))
            self.price_history[symbol] = {
                'current': current_price,
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'returns': []
            }
    
    def build_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build realistic correlation matrix based on sectors and groups"""
        correlation_groups = {}
        for symbol, info in self.stocks.items():
            group = info['correlation_group']
            if group not in correlation_groups:
                correlation_groups[group] = []
            correlation_groups[group].append(symbol)
        
        correlations = {}
        for symbol1 in self.stocks:
            correlations[symbol1] = {}
            for symbol2 in self.stocks:
                if symbol1 == symbol2:
                    correlations[symbol1][symbol2] = 1.0
                else:
                    # Higher correlation within same group
                    group1 = self.stocks[symbol1]['correlation_group']
                    group2 = self.stocks[symbol2]['correlation_group']
                    
                    if group1 == group2:
                        base_correlation = 0.6 + np.random.normal(0, 0.1)
                    else:
                        sector1 = self.stocks[symbol1]['sector']
                        sector2 = self.stocks[symbol2]['sector']
                        if sector1 == sector2:
                            base_correlation = 0.3 + np.random.normal(0, 0.1)
                        else:
                            base_correlation = 0.1 + np.random.normal(0, 0.05)
                    
                    correlations[symbol1][symbol2] = np.clip(base_correlation, -0.5, 0.8)
        
        return correlations
    
    def build_heat_network(self) -> Dict[str, Dict[str, float]]:
        """Build heat propagation network based on correlations"""
        heat_network = {}
        for symbol1 in self.stocks:
            heat_network[symbol1] = {}
            for symbol2 in self.stocks:
                if symbol1 != symbol2:
                    correlation = self.correlation_matrix[symbol1][symbol2]
                    heat_strength = abs(correlation) * (1 + np.random.normal(0, 0.1))
                    heat_network[symbol1][symbol2] = np.clip(heat_strength, 0.1, 1.0)
        return heat_network
    
    def update_market_regime(self):
        """Update market regime periodically"""
        self.regime_duration += 1
        
        if self.regime_duration >= self.max_regime_duration:
            # Transition to new regime
            regimes = list(self.market_regimes.keys())
            # Higher probability of normal regimes
            weights = [0.4, 0.25, 0.3, 0.05]  # bull, bear, sideways, crash
            self.current_regime = np.random.choice(regimes, p=weights)
            self.regime_duration = 0
            self.max_regime_duration = random.randint(50, 300)
            
            logging.info(f"Market regime changed to: {self.current_regime}")
    
    def calculate_heat_propagation(self, symbol: str, base_volatility: float) -> float:
        """Calculate heat score based on network propagation"""
        heat_score = base_volatility * 50  # Base heat from volatility
        
        # Add heat from connected stocks
        for connected_symbol, strength in self.heat_network[symbol].items():
            connected_volatility = self.stocks[connected_symbol]['volatility_base']
            if connected_symbol in self.price_history:
                recent_returns = self.price_history[connected_symbol]['returns'][-5:]
                if recent_returns:
                    recent_volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.01
                    propagated_heat = recent_volatility * strength * 100
                    heat_score += propagated_heat
        
        # Market regime effect on heat
        regime_multiplier = self.market_regimes[self.current_regime]['volatility_multiplier']
        heat_score *= regime_multiplier
        
        return np.clip(heat_score, 0, 100)
    
    def generate_correlated_returns(self) -> Dict[str, float]:
        """Generate correlated returns for all stocks"""
        # Base random returns
        random_returns = np.random.multivariate_normal(
            mean=[0] * len(self.stocks),
            cov=np.eye(len(self.stocks)) * 0.01
        )
        
        # Apply correlation structure
        symbols = list(self.stocks.keys())
        correlated_returns = {}
        
        for i, symbol in enumerate(symbols):
            base_return = random_returns[i]
            
            # Add correlated components
            correlated_component = 0
            for j, other_symbol in enumerate(symbols):
                if i != j:
                    correlation = self.correlation_matrix[symbol][other_symbol]
                    correlated_component += correlation * random_returns[j] * 0.3
            
            # Market regime effect
            regime = self.market_regimes[self.current_regime]
            trend_component = regime['trend_factor']
            volatility_multiplier = regime['volatility_multiplier']
            
            # Stock-specific volatility
            stock_volatility = self.stocks[symbol]['volatility_base']
            
            final_return = (base_return + correlated_component + trend_component) * volatility_multiplier * stock_volatility
            correlated_returns[symbol] = final_return
        
        return correlated_returns
    
    def generate_realistic_volume(self, symbol: str, price_change: float) -> int:
        """Generate realistic volume based on price movement and stock characteristics"""
        # Base volume varies by stock
        base_volumes = {
            'AAPL': 80000000, 'GOOGL': 25000000, 'MSFT': 35000000, 'AMZN': 30000000,
            'TSLA': 45000000, 'NVDA': 40000000, 'META': 25000000, 'NFLX': 8000000,
            'JPM': 15000000, 'BAC': 45000000, 'JNJ': 12000000, 'PFE': 35000000,
            'XOM': 20000000, 'CVX': 12000000, 'WMT': 15000000, 'HD': 8000000,
            'PG': 10000000, 'KO': 18000000, 'DIS': 12000000, 'V': 8000000
        }
        
        base_volume = base_volumes.get(symbol, 20000000)
        
        # Volume increases with volatility
        volatility_factor = 1 + abs(price_change) * 10
        
        # Random market activity
        activity_factor = np.random.lognormal(0, 0.3)
        
        # Market regime affects volume
        regime_volume_multiplier = {
            'bull': 1.2, 'bear': 1.5, 'sideways': 0.8, 'crash': 3.0
        }
        
        final_volume = int(base_volume * volatility_factor * activity_factor * 
                          regime_volume_multiplier[self.current_regime])
        
        return max(final_volume, 100000)  # Minimum volume
    
    def generate_single_tick(self, symbol: str) -> StockData:
        """Generate a single realistic tick for a stock"""
        returns = self.generate_correlated_returns()
        return_pct = returns[symbol]
        
        # Update price history
        history = self.price_history[symbol]
        current_price = history['current']
        new_price = current_price * (1 + return_pct)
        
        # Update OHLC
        history['high'] = max(history['high'], new_price)
        history['low'] = min(history['low'], new_price)
        history['current'] = new_price
        history['returns'].append(return_pct)
        
        # Keep only last 100 returns for memory efficiency
        if len(history['returns']) > 100:
            history['returns'] = history['returns'][-100:]
        
        # Calculate metrics
        volume = self.generate_realistic_volume(symbol, abs(return_pct))
        volatility = np.std(history['returns']) if len(history['returns']) > 1 else 0.01
        heat_score = self.calculate_heat_propagation(symbol, volatility)
        
        # Calculate correlation signals
        correlation_signals = {}
        for other_symbol in self.stocks:
            if other_symbol != symbol:
                correlation_signals[other_symbol] = self.correlation_matrix[symbol][other_symbol]
        
        # Market cap estimation (simplified)
        shares_outstanding = {
            'AAPL': 16000000000, 'GOOGL': 13000000000, 'MSFT': 7500000000,
            'AMZN': 10500000000, 'TSLA': 3200000000, 'NVDA': 2500000000,
            'META': 2700000000, 'NFLX': 440000000, 'JPM': 3000000000,
            'BAC': 8500000000, 'JNJ': 2600000000, 'PFE': 5600000000,
            'XOM': 4200000000, 'CVX': 1900000000, 'WMT': 2700000000,
            'HD': 1100000000, 'PG': 2400000000, 'KO': 4300000000,
            'DIS': 1800000000, 'V': 2000000000
        }
        
        market_cap = new_price * shares_outstanding.get(symbol, 1000000000)
        
        return StockData(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            price=round(new_price, 2),
            volume=volume,
            open_price=round(history['open'], 2),
            high=round(history['high'], 2),
            low=round(history['low'], 2),
            close=round(new_price, 2),
            change_percent=round(return_pct * 100, 3),
            volatility=round(volatility * 100, 3),
            heat_score=round(heat_score, 2),
            sector=self.stocks[symbol]['sector'],
            market_cap=market_cap,
            correlation_signals=correlation_signals
        )
    
    def reset_daily_ohlc(self):
        """Reset daily OHLC values (simulate market open)"""
        for symbol in self.stocks:
            history = self.price_history[symbol]
            current_price = history['current']
            history['open'] = current_price
            history['high'] = current_price
            history['low'] = current_price
    
    def generate_batch_data(self, num_ticks: int = 100) -> List[Dict]:
        """Generate a batch of synthetic data for all stocks"""
        batch_data = []
        
        for _ in range(num_ticks):
            self.update_market_regime()
            
            tick_data = {}
            for symbol in self.stocks:
                stock_data = self.generate_single_tick(symbol)
                tick_data[symbol] = asdict(stock_data)
            
            batch_data.append({
                'timestamp': datetime.utcnow().isoformat(),
                'market_regime': self.current_regime,
                'stocks': tick_data
            })
            
            # Small random delay to simulate realistic timing
            if np.random.random() < 0.1:  # 10% chance of micro-pause
                continue
        
        return batch_data
    
    def save_to_json_files(self, data: List[Dict]):
        """Save data to individual JSON files by timestamp"""
        for i, batch in enumerate(data):
            timestamp = batch['timestamp'].replace(':', '-').replace('.', '-')
            filename = f"market_data_{timestamp}_{i:04d}.json"
            filepath = self.data_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(batch, f, indent=2)
    
    async def download_real_market_data(self):
        """Download real historical data to enhance synthetic generation"""
        try:
            logging.info("Downloading real market data...")
            
            # Download recent data for major stocks
            symbols = list(self.stocks.keys())
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist_data = ticker.history(start=start_date, end=end_date, interval='1m')
                    
                    if not hist_data.empty:
                        # Convert to JSON-serializable format
                        data_dict = {
                            'symbol': symbol,
                            'data': hist_data.reset_index().to_dict('records')
                        }
                        
                        # Save real data
                        filepath = self.real_data_dir / f"{symbol}_real_data.json"
                        with open(filepath, 'w') as f:
                            json.dump(data_dict, f, indent=2, default=str)
                        
                        logging.info(f"Downloaded {len(hist_data)} records for {symbol}")
                        
                    await asyncio.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logging.error(f"Failed to download data for {symbol}: {e}")
                    
        except Exception as e:
            logging.error(f"Real data download failed: {e}")
            logging.info("Continuing with synthetic data generation...")

# Usage example and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generator = SyntheticDataGenerator()
    
    # Generate synthetic data
    print("Generating synthetic market data...")
    synthetic_data = generator.generate_batch_data(500)  # 500 ticks
    generator.save_to_json_files(synthetic_data)
    
    print(f"Generated {len(synthetic_data)} market data files")
    print("Sample data structure:")
    print(json.dumps(synthetic_data[0], indent=2)[:500] + "...")