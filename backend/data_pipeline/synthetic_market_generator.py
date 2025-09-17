"""
Synthetic Market Data Generator for RAGHeat System
Generates realistic market patterns when markets are closed for testing all models,
heat equations, and knowledge graph reasoning.
"""

import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import random
import math
from enum import Enum


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways" 
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"


@dataclass
class SectorProfile:
    """Profile for generating realistic sector behavior"""
    name: str
    base_volatility: float
    momentum_factor: float
    correlation_to_market: float
    seasonal_patterns: Dict[str, float]
    stocks: List[str]
    current_heat_level: float = 0.0
    heat_propagation_rate: float = 0.1


@dataclass
class StockProfile:
    """Profile for individual stock behavior"""
    symbol: str
    sector: str
    base_price: float
    volatility: float
    beta: float  # Market correlation
    fundamental_score: float  # -1 to 1
    momentum_persistence: float  # 0 to 1
    news_sensitivity: float  # 0 to 1
    current_heat: float = 0.0


class SyntheticMarketGenerator:
    """
    Advanced synthetic market data generator with realistic patterns
    """
    
    def __init__(self):
        self.current_time = datetime.now()
        self.market_regime = MarketRegime.SIDEWAYS
        self.market_sentiment = 0.0  # -1 to 1
        self.volatility_regime = 0.5  # 0 to 1
        self.setup_sector_profiles()
        self.setup_stock_profiles()
        self.heat_diffusion_matrix = self.create_heat_diffusion_matrix()
        self.news_events = []
        
    def setup_sector_profiles(self):
        """Initialize sector profiles with realistic characteristics"""
        self.sectors = {
            'technology': SectorProfile(
                name='Technology',
                base_volatility=0.25,
                momentum_factor=1.2,
                correlation_to_market=0.8,
                seasonal_patterns={'Q1': 1.1, 'Q2': 0.9, 'Q3': 1.0, 'Q4': 1.15},
                stocks=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMZN', 'NFLX'],
                current_heat_level=0.0,
                heat_propagation_rate=0.15
            ),
            'healthcare': SectorProfile(
                name='Healthcare',
                base_volatility=0.18,
                momentum_factor=0.8,
                correlation_to_market=0.6,
                seasonal_patterns={'Q1': 0.95, 'Q2': 1.0, 'Q3': 1.05, 'Q4': 1.0},
                stocks=['JNJ', 'PFE', 'UNH', 'ABT', 'TMO', 'DHR', 'BMY'],
                current_heat_level=0.0,
                heat_propagation_rate=0.08
            ),
            'financial': SectorProfile(
                name='Financial',
                base_volatility=0.22,
                momentum_factor=0.9,
                correlation_to_market=0.85,
                seasonal_patterns={'Q1': 1.05, 'Q2': 0.95, 'Q3': 1.0, 'Q4': 1.1},
                stocks=['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP'],
                current_heat_level=0.0,
                heat_propagation_rate=0.12
            ),
            'energy': SectorProfile(
                name='Energy',
                base_volatility=0.35,
                momentum_factor=1.5,
                correlation_to_market=0.7,
                seasonal_patterns={'Q1': 0.9, 'Q2': 1.1, 'Q3': 1.2, 'Q4': 0.8},
                stocks=['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD'],
                current_heat_level=0.0,
                heat_propagation_rate=0.20
            ),
            'consumer': SectorProfile(
                name='Consumer',
                base_volatility=0.16,
                momentum_factor=0.7,
                correlation_to_market=0.65,
                seasonal_patterns={'Q1': 0.85, 'Q2': 1.0, 'Q3': 1.1, 'Q4': 1.25},
                stocks=['WMT', 'PG', 'KO', 'PEP', 'MCD', 'HD', 'NKE'],
                current_heat_level=0.0,
                heat_propagation_rate=0.10
            )
        }
    
    def setup_stock_profiles(self):
        """Initialize individual stock profiles"""
        self.stocks = {}
        
        # Tech stocks with high volatility and beta
        tech_stocks = {
            'AAPL': (180.0, 0.22, 1.1, 0.8, 0.7, 0.6),
            'MSFT': (350.0, 0.20, 1.0, 0.7, 0.8, 0.5),
            'GOOGL': (140.0, 0.25, 1.2, 0.6, 0.6, 0.7),
            'NVDA': (450.0, 0.40, 1.8, 0.9, 0.9, 0.9),
            'META': (320.0, 0.35, 1.4, 0.5, 0.8, 0.8),
            'TSLA': (220.0, 0.50, 2.0, 0.7, 0.9, 0.9),
            'AMZN': (150.0, 0.28, 1.3, 0.8, 0.7, 0.6),
            'NFLX': (450.0, 0.38, 1.5, 0.4, 0.8, 0.8)
        }
        
        for symbol, (price, vol, beta, fund, mom, news) in tech_stocks.items():
            self.stocks[symbol] = StockProfile(
                symbol=symbol, sector='technology', base_price=price,
                volatility=vol, beta=beta, fundamental_score=fund,
                momentum_persistence=mom, news_sensitivity=news
            )
        
        # Add other sector stocks
        other_stocks = {
            'JNJ': (160.0, 0.15, 0.7, 0.9, 0.5, 0.4),
            'JPM': (145.0, 0.25, 1.1, 0.8, 0.6, 0.6),
            'XOM': (105.0, 0.35, 1.0, 0.6, 0.8, 0.8),
            'WMT': (160.0, 0.18, 0.5, 0.8, 0.4, 0.3)
        }
        
        sector_mapping = {'JNJ': 'healthcare', 'JPM': 'financial', 'XOM': 'energy', 'WMT': 'consumer'}
        
        for symbol, (price, vol, beta, fund, mom, news) in other_stocks.items():
            self.stocks[symbol] = StockProfile(
                symbol=symbol, sector=sector_mapping[symbol], base_price=price,
                volatility=vol, beta=beta, fundamental_score=fund,
                momentum_persistence=mom, news_sensitivity=news
            )
    
    def create_heat_diffusion_matrix(self) -> np.ndarray:
        """Create heat diffusion matrix for sector correlations"""
        sectors = list(self.sectors.keys())
        n = len(sectors)
        matrix = np.eye(n) * 0.7  # Self-correlation
        
        # Add cross-sector correlations
        correlations = {
            ('technology', 'consumer'): 0.3,
            ('financial', 'energy'): 0.4,
            ('healthcare', 'consumer'): 0.2,
            ('technology', 'financial'): 0.25,
            ('energy', 'consumer'): 0.15
        }
        
        for i, sector1 in enumerate(sectors):
            for j, sector2 in enumerate(sectors):
                if i != j:
                    corr = correlations.get((sector1, sector2), 0.1)
                    matrix[i][j] = corr
        
        return matrix
    
    def generate_market_regime_shift(self):
        """Randomly shift market regime with realistic probabilities"""
        regime_transitions = {
            MarketRegime.BULL: {MarketRegime.BULL: 0.85, MarketRegime.SIDEWAYS: 0.12, MarketRegime.BEAR: 0.03},
            MarketRegime.BEAR: {MarketRegime.BEAR: 0.80, MarketRegime.SIDEWAYS: 0.15, MarketRegime.BULL: 0.05},
            MarketRegime.SIDEWAYS: {MarketRegime.SIDEWAYS: 0.70, MarketRegime.BULL: 0.15, MarketRegime.BEAR: 0.15}
        }
        
        current_probs = regime_transitions.get(self.market_regime, {self.market_regime: 1.0})
        regimes = list(current_probs.keys())
        probabilities = list(current_probs.values())
        
        self.market_regime = np.random.choice(regimes, p=probabilities)
        
        # Update market sentiment based on regime
        if self.market_regime == MarketRegime.BULL:
            self.market_sentiment = np.clip(self.market_sentiment + np.random.normal(0.1, 0.2), -1, 1)
        elif self.market_regime == MarketRegime.BEAR:
            self.market_sentiment = np.clip(self.market_sentiment - np.random.normal(0.1, 0.2), -1, 1)
        else:
            self.market_sentiment += np.random.normal(0, 0.05)
    
    def generate_news_events(self) -> List[Dict[str, Any]]:
        """Generate synthetic news events that affect market heat"""
        news_templates = [
            {"type": "earnings", "impact": 0.3, "sectors": ["technology"], "probability": 0.05},
            {"type": "fed_announcement", "impact": 0.5, "sectors": ["financial"], "probability": 0.02},
            {"type": "oil_price_shock", "impact": 0.4, "sectors": ["energy"], "probability": 0.03},
            {"type": "tech_breakthrough", "impact": 0.6, "sectors": ["technology"], "probability": 0.01},
            {"type": "healthcare_approval", "impact": 0.4, "sectors": ["healthcare"], "probability": 0.02}
        ]
        
        events = []
        for template in news_templates:
            if np.random.random() < template["probability"]:
                event = {
                    "timestamp": self.current_time,
                    "type": template["type"],
                    "impact": template["impact"] * np.random.uniform(0.5, 1.5),
                    "affected_sectors": template["sectors"],
                    "sentiment": np.random.choice([-1, 1]) * np.random.uniform(0.3, 1.0)
                }
                events.append(event)
        
        return events
    
    def apply_heat_diffusion(self):
        """Apply heat diffusion across sectors using physics-based model"""
        sector_names = list(self.sectors.keys())
        current_heat = np.array([self.sectors[s].current_heat_level for s in sector_names])
        
        # Heat diffusion equation: dH/dt = α∇²H + source_terms
        alpha = 0.1  # Thermal diffusivity
        dt = 1.0     # Time step
        
        # Laplacian operator approximation using correlation matrix
        laplacian = self.heat_diffusion_matrix - np.eye(len(sector_names))
        
        # Add random heat sources (news, events, momentum)
        source_terms = np.random.normal(0, 0.05, len(sector_names))
        
        # Apply news event heating
        for event in self.news_events:
            for sector in event["affected_sectors"]:
                if sector in sector_names:
                    idx = sector_names.index(sector)
                    source_terms[idx] += event["impact"] * event["sentiment"]
        
        # Update heat levels
        heat_change = alpha * np.dot(laplacian, current_heat) + source_terms
        new_heat = current_heat + heat_change * dt
        
        # Apply cooling and bounds
        cooling_rate = 0.02
        new_heat = (1 - cooling_rate) * new_heat
        new_heat = np.clip(new_heat, -1.0, 1.0)
        
        # Update sector heat levels
        for i, sector in enumerate(sector_names):
            self.sectors[sector].current_heat_level = new_heat[i]
    
    def generate_stock_price(self, symbol: str, previous_price: Optional[float] = None) -> Dict[str, Any]:
        """Generate realistic stock price with all market factors"""
        stock = self.stocks[symbol]
        sector = self.sectors[stock.sector]
        
        if previous_price is None:
            previous_price = stock.base_price
        
        # Base return components
        market_return = self.market_sentiment * 0.001  # Market direction
        sector_heat_effect = sector.current_heat_level * 0.002  # Heat impact
        stock_heat_effect = stock.current_heat * 0.003  # Stock-specific heat
        
        # Regime-based adjustments
        regime_multiplier = {
            MarketRegime.BULL: 1.2,
            MarketRegime.BEAR: -0.8,
            MarketRegime.SIDEWAYS: 0.1,
            MarketRegime.HIGH_VOLATILITY: 0.0,
            MarketRegime.LOW_VOLATILITY: 0.0
        }[self.market_regime]
        
        base_return = (market_return + sector_heat_effect + stock_heat_effect) * regime_multiplier
        
        # Volatility clustering (GARCH-like)
        vol_shock = np.random.normal(0, 1)
        realized_vol = stock.volatility * (1 + self.volatility_regime * 0.5)
        
        # Mean reversion to fundamental value
        fundamental_price = stock.base_price * (1 + stock.fundamental_score * 0.1)
        mean_reversion = -0.001 * (previous_price - fundamental_price) / fundamental_price
        
        # Momentum effect
        momentum = stock.momentum_persistence * base_return * 0.5
        
        # Final return calculation
        total_return = base_return + vol_shock * realized_vol / np.sqrt(252) + mean_reversion + momentum
        new_price = previous_price * (1 + total_return)
        
        # Generate volume with correlation to volatility and price movement
        base_volume = 1000000
        volume_multiplier = 1 + abs(total_return) * 50 + abs(stock.current_heat) * 20
        volume = int(base_volume * volume_multiplier * np.random.lognormal(0, 0.3))
        
        # Update stock heat based on price movement and volume
        price_momentum = total_return * 100  # Convert to heat scale
        volume_heat = (volume / base_volume - 1) * 0.1
        stock.current_heat = 0.9 * stock.current_heat + 0.1 * (price_momentum + volume_heat)
        stock.current_heat = np.clip(stock.current_heat, -1, 1)
        
        return {
            "symbol": symbol,
            "price": round(new_price, 2),
            "change": round(new_price - previous_price, 2),
            "change_percent": round((new_price - previous_price) / previous_price * 100, 2),
            "volume": volume,
            "heat_level": round(stock.current_heat, 3),
            "sector": stock.sector,
            "sector_heat": round(sector.current_heat_level, 3),
            "timestamp": self.current_time.isoformat(),
            "regime": self.market_regime.value,
            "market_sentiment": round(self.market_sentiment, 3),
            "volatility": round(realized_vol, 3),
            "fundamental_score": stock.fundamental_score,
            "beta": stock.beta
        }
    
    def generate_market_overview(self) -> Dict[str, Any]:
        """Generate complete market overview with all stocks and sectors"""
        # Update market dynamics
        self.generate_market_regime_shift()
        self.news_events = self.generate_news_events()
        self.apply_heat_diffusion()
        self.current_time += timedelta(seconds=5)
        
        # Generate all stock data
        stocks_data = {}
        for symbol in self.stocks.keys():
            stocks_data[symbol] = self.generate_stock_price(symbol)
        
        # Calculate sector summaries
        sectors_data = {}
        for sector_name, sector in self.sectors.items():
            sector_stocks = [s for s in stocks_data.values() if s["sector"] == sector_name]
            if sector_stocks:
                avg_change = np.mean([s["change_percent"] for s in sector_stocks])
                total_volume = sum([s["volume"] for s in sector_stocks])
                max_heat = max([s["heat_level"] for s in sector_stocks])
                
                sectors_data[sector_name] = {
                    "name": sector.name,
                    "average_change": round(avg_change, 2),
                    "total_volume": total_volume,
                    "heat_level": round(sector.current_heat_level, 3),
                    "max_stock_heat": round(max_heat, 3),
                    "stock_count": len(sector_stocks),
                    "momentum_factor": sector.momentum_factor
                }
        
        # Find heated sectors (for red nodes)
        heated_sectors = []
        for name, data in sectors_data.items():
            if data["heat_level"] > 0.3 or data["max_stock_heat"] > 0.5:
                heated_sectors.append({
                    "sector": name,
                    "heat_level": data["heat_level"],
                    "reason": self.generate_heat_reason(name, data)
                })
        
        # Market-wide statistics
        all_changes = [s["change_percent"] for s in stocks_data.values()]
        market_return = np.mean(all_changes)
        market_volatility = np.std(all_changes)
        total_volume = sum([s["volume"] for s in stocks_data.values()])
        
        return {
            "timestamp": self.current_time.isoformat(),
            "market_overview": {
                "market_return": round(market_return, 2),
                "market_volatility": round(market_volatility, 2),
                "total_volume": total_volume,
                "market_heat_index": round(np.mean([s.current_heat_level for s in self.sectors.values()]) * 100, 1),
                "regime": self.market_regime.value,
                "sentiment": round(self.market_sentiment, 3),
                "active_stocks": len(stocks_data),
                "heated_sectors_count": len(heated_sectors)
            },
            "stocks": stocks_data,
            "sectors": sectors_data,
            "heated_sectors": heated_sectors,
            "news_events": self.news_events,
            "heat_diffusion_active": True,
            "data_source": "synthetic_generator",
            "generation_metadata": {
                "model_version": "v1.0",
                "heat_equation_applied": True,
                "market_regime_active": True,
                "news_simulation_active": True
            }
        }
    
    def generate_heat_reason(self, sector: str, sector_data: Dict) -> str:
        """Generate explanation for why a sector is heated"""
        reasons = []
        
        if sector_data["heat_level"] > 0.5:
            reasons.append("Strong sector-wide momentum")
        if sector_data["max_stock_heat"] > 0.7:
            reasons.append("Individual stock breakouts")
        if sector_data["total_volume"] > 10000000:
            reasons.append("High trading volume")
        if sector_data["average_change"] > 2.0:
            reasons.append("Positive price momentum")
        
        # Add news-based reasons
        for event in self.news_events:
            if sector in event["affected_sectors"] and event["sentiment"] > 0.3:
                reasons.append(f"Positive {event['type']} news impact")
        
        if not reasons:
            reasons.append("Heat diffusion from correlated sectors")
        
        return "; ".join(reasons[:2])  # Limit to top 2 reasons
    
    def get_neo4j_graph_data(self) -> Dict[str, Any]:
        """Generate Neo4j compatible graph structure with heat mapping"""
        nodes = []
        relationships = []
        
        # Market node (root)
        market_heat = np.mean([s.current_heat_level for s in self.sectors.values()])
        nodes.append({
            "id": "market",
            "type": "Market",
            "name": "US Market",
            "heat_level": round(market_heat, 3),
            "properties": {
                "regime": self.market_regime.value,
                "sentiment": round(self.market_sentiment, 3),
                "timestamp": self.current_time.isoformat()
            }
        })
        
        # Sector nodes
        for sector_name, sector in self.sectors.items():
            heat_color = "red" if sector.current_heat_level > 0.3 else "orange" if sector.current_heat_level > 0.1 else "green"
            nodes.append({
                "id": sector_name,
                "type": "Sector", 
                "name": sector.name,
                "heat_level": round(sector.current_heat_level, 3),
                "color": heat_color,
                "properties": {
                    "base_volatility": sector.base_volatility,
                    "momentum_factor": sector.momentum_factor,
                    "correlation_to_market": sector.correlation_to_market,
                    "propagation_rate": sector.heat_propagation_rate
                }
            })
            
            # Market -> Sector relationship
            relationships.append({
                "source": "market",
                "target": sector_name,
                "type": "CONTAINS",
                "weight": sector.correlation_to_market,
                "properties": {"relationship_strength": sector.correlation_to_market}
            })
        
        # Stock nodes
        for symbol, stock in self.stocks.items():
            heat_intensity = abs(stock.current_heat)
            heat_color = "red" if heat_intensity > 0.5 else "orange" if heat_intensity > 0.2 else "blue"
            
            nodes.append({
                "id": symbol,
                "type": "Stock",
                "name": symbol,
                "heat_level": round(stock.current_heat, 3),
                "color": heat_color,
                "properties": {
                    "base_price": stock.base_price,
                    "volatility": stock.volatility,
                    "beta": stock.beta,
                    "fundamental_score": stock.fundamental_score,
                    "sector": stock.sector
                }
            })
            
            # Sector -> Stock relationship
            relationships.append({
                "source": stock.sector,
                "target": symbol,
                "type": "BELONGS_TO",
                "weight": 1.0,
                "properties": {"beta": stock.beta}
            })
        
        # Add cross-sector correlations as relationships
        sector_names = list(self.sectors.keys())
        for i, sector1 in enumerate(sector_names):
            for j, sector2 in enumerate(sector_names[i+1:], i+1):
                correlation = self.heat_diffusion_matrix[i][j]
                if correlation > 0.2:  # Only show significant correlations
                    relationships.append({
                        "source": sector1,
                        "target": sector2,
                        "type": "CORRELATED_WITH",
                        "weight": correlation,
                        "properties": {"correlation_strength": round(correlation, 3)}
                    })
        
        return {
            "nodes": nodes,
            "relationships": relationships,
            "metadata": {
                "timestamp": self.current_time.isoformat(),
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "heated_sectors": len([s for s in self.sectors.values() if s.current_heat_level > 0.3]),
                "heated_stocks": len([s for s in self.stocks.values() if abs(s.current_heat) > 0.5]),
                "market_regime": self.market_regime.value,
                "heat_diffusion_active": True
            }
        }


# Global generator instance
synthetic_generator = SyntheticMarketGenerator()


def get_synthetic_market_data() -> Dict[str, Any]:
    """Get synthetic market data for when markets are closed"""
    return synthetic_generator.generate_market_overview()


def get_synthetic_neo4j_data() -> Dict[str, Any]:
    """Get synthetic Neo4j graph data"""
    return synthetic_generator.get_neo4j_graph_data()


if __name__ == "__main__":
    # Test the generator
    generator = SyntheticMarketGenerator()
    
    print("🧪 Testing Synthetic Market Data Generator")
    print("=" * 60)
    
    # Generate 5 cycles to show evolution
    for i in range(5):
        print(f"\n📊 Cycle {i+1}:")
        data = generator.generate_market_overview()
        
        market = data["market_overview"]
        print(f"  Market Return: {market['market_return']:.2f}%")
        print(f"  Market Heat: {market['market_heat_index']:.1f}%")
        print(f"  Regime: {market['regime']}")
        print(f"  Heated Sectors: {market['heated_sectors_count']}")
        
        if data["heated_sectors"]:
            print("  🔥 Hot Sectors:")
            for hot in data["heated_sectors"]:
                print(f"    - {hot['sector']}: {hot['heat_level']:.2f} ({hot['reason']})")
        
        # Show top movers
        stocks = data["stocks"]
        top_movers = sorted(stocks.values(), key=lambda x: abs(x["change_percent"]), reverse=True)[:3]
        print("  📈 Top Movers:")
        for stock in top_movers:
            print(f"    - {stock['symbol']}: {stock['change_percent']:.2f}% (Heat: {stock['heat_level']:.2f})")
        
        print(f"  News Events: {len(data['news_events'])}")