import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class Node:
    '''Represents a node in the knowledge graph'''
    id: str
    type: str  # 'sector', 'stock', 'feature', 'event'
    level: int
    attributes: Dict
    heat_score: float = 0.0
    timestamp: datetime = None

class FinancialKnowledgeGraph:
    '''
    Manages the financial knowledge graph with hierarchical structure:
    Level 0: Root (SECTOR)
    Level 1: Sectors (Technology, Healthcare, etc.)
    Level 2: Individual Stocks
    '''

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_mapping = {}
        self.heat_scores = {}
        self.initialize_graph()

    def initialize_graph(self):
        '''Initialize the graph with root and sector nodes'''
        # Add root node
        root = Node(
            id="ROOT_SECTOR",
            type="root",
            level=0,
            attributes={"name": "Market"},
            heat_score=0.0,
            timestamp=datetime.now()
        )
        self.add_node(root)

        # Add sector nodes
        sectors = [
            "Technology", "Healthcare", "Finance", "Energy",
            "Consumer_Goods", "Industrial", "Utilities",
            "Real_Estate", "Materials", "Communication"
        ]

        for sector in sectors:
            sector_node = Node(
                id=f"SECTOR_{sector.upper()}",
                type="sector",
                level=1,
                attributes={
                    "name": sector,
                    "market_cap": 0,
                    "avg_pe": 0,
                    "volatility": 0
                },
                heat_score=0.0,
                timestamp=datetime.now()
            )
            self.add_node(sector_node)
            self.add_edge("ROOT_SECTOR", sector_node.id, weight=1.0)

    def add_node(self, node: Node):
        '''Add a node to the graph'''
        self.graph.add_node(
            node.id,
            type=node.type,
            level=node.level,
            attributes=node.attributes,
            heat_score=node.heat_score,
            timestamp=node.timestamp
        )
        self.node_mapping[node.id] = node
        self.heat_scores[node.id] = node.heat_score

    def add_edge(self, source: str, target: str, weight: float = 1.0):
        '''Add an edge between nodes'''
        self.graph.add_edge(source, target, weight=weight)

    def add_stock(self, stock_symbol: str, sector: str, attributes: Dict):
        '''Add a stock node to the appropriate sector'''
        stock_node = Node(
            id=f"STOCK_{stock_symbol}",
            type="stock",
            level=2,
            attributes={
                "symbol": stock_symbol,
                "sector": sector,
                **attributes
            },
            heat_score=0.0,
            timestamp=datetime.now()
        )

        self.add_node(stock_node)
        sector_id = f"SECTOR_{sector.upper()}"
        if sector_id in self.node_mapping:
            self.add_edge(sector_id, stock_node.id)

        return stock_node

    def update_node_attributes(self, node_id: str, attributes: Dict):
        '''Update attributes of an existing node'''
        if node_id in self.graph.nodes:
            self.graph.nodes[node_id]['attributes'].update(attributes)
            self.graph.nodes[node_id]['timestamp'] = datetime.now()

    def get_neighbors(self, node_id: str) -> List[str]:
        '''Get all neighbors of a node'''
        return list(self.graph.neighbors(node_id))

    def get_node_by_level(self, level: int) -> List[Node]:
        '''Get all nodes at a specific level'''
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('level') == level:
                nodes.append(self.node_mapping[node_id])
        return nodes

    def to_json(self) -> str:
        '''Export graph to JSON format'''
        nodes = []
        edges = []

        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "type": data.get("type"),
                "level": data.get("level"),
                "attributes": data.get("attributes"),
                "heat_score": self.heat_scores.get(node_id, 0)
            })

        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "weight": data.get("weight", 1.0)
            })

        return json.dumps({"nodes": nodes, "edges": edges}, indent=2)