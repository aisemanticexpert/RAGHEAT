import asyncio
import json
from typing import Dict, List, Optional
from datetime import datetime
import yfinance as yf
import logging
import time

logger = logging.getLogger(__name__)

class MarketDataStreamer:
    '''Streams market data and updates the knowledge graph in real-time'''

    def __init__(self, knowledge_graph, heat_engine, orchestrator):
        self.knowledge_graph = knowledge_graph
        self.heat_engine = heat_engine
        self.orchestrator = orchestrator
        self.running = False

        # Stock universe - for demo, using major tech stocks
        self.stock_universe = {
            'Technology': ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA'],
            'Healthcare': ['JNJ', 'PFE', 'UNH'],
            'Finance': ['JPM', 'BAC', 'GS'],
            'Energy': ['XOM', 'CVX'],
            'Consumer_Goods': ['AMZN', 'WMT']
        }

    async def start_streaming(self):
        '''Start the data streaming process'''
        self.running = True

        # Initialize stocks in knowledge graph
        await self._initialize_stocks()

        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._stream_prices()),
            asyncio.create_task(self._calculate_heat_periodically())
        ]

        await asyncio.gather(*tasks)

    async def _initialize_stocks(self):
        '''Initialize all stocks in the knowledge graph'''
        for sector, stocks in self.stock_universe.items():
            for symbol in stocks:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info

                    attributes = {
                        'symbol': symbol,
                        'name': info.get('longName', symbol),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'volume': info.get('volume', 0),
                        'price': info.get('currentPrice', info.get('regularMarketPrice', 0))
                    }

                    self.knowledge_graph.add_stock(symbol, sector, attributes)
                    logger.info(f"Added {symbol} to {sector}")

                except Exception as e:
                    logger.error(f"Error initializing {symbol}: {str(e)}")

        logger.info("Stock initialization complete")

    async def _stream_prices(self):
        '''Stream price updates'''
        while self.running:
            try:
                for sector, stocks in self.stock_universe.items():
                    for symbol in stocks:
                        try:
                            # Get latest price
                            stock = yf.Ticker(symbol)
                            info = stock.info

                            # Update node attributes
                            node_id = f"STOCK_{symbol}"
                            self.knowledge_graph.update_node_attributes(
                                node_id,
                                {
                                    'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                                    'volume': info.get('volume', 0),
                                    'last_update': datetime.now().isoformat()
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error updating {symbol}: {str(e)}")

                # Wait before next update
                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error streaming prices: {str(e)}")
                await asyncio.sleep(10)

    async def _calculate_heat_periodically(self):
        '''Periodically recalculate heat distribution'''
        while self.running:
            try:
                # Get current heat sources (simplified for demo)
                heat_sources = {}

                # For demo, assign random heat to some stocks
                import random
                all_stocks = [
                    f"STOCK_{stock}"
                    for stocks in self.stock_universe.values() 
                    for stock in stocks
                ]

                # Randomly select stocks as heat sources
                num_sources = min(5, len(all_stocks))
                selected_stocks = random.sample(all_stocks, num_sources)

                for stock_id in selected_stocks:
                    heat_sources[stock_id] = random.uniform(0.5, 1.0)

                # Calculate and propagate heat
                if heat_sources:
                    heat_distribution = self.heat_engine.calculate_heat_distribution(
                        heat_sources,
                        propagation_steps=5
                    )

                    # Update all nodes with new heat scores
                    for node_id, heat in heat_distribution.items():
                        self.knowledge_graph.heat_scores[node_id] = heat

                    # Log top heated sectors
                    top_sectors = self.heat_engine.get_heated_sectors(top_k=3)
                    logger.info(f"Top heated sectors: {top_sectors}")

                await asyncio.sleep(30)  # Recalculate every 30 seconds

            except Exception as e:
                logger.error(f"Error calculating heat: {str(e)}")
                await asyncio.sleep(60)

    def stop_streaming(self):
        '''Stop the streaming process'''
        self.running = False