"""
Comprehensive NASDAQ sector configuration with top stocks for multi-sector analysis
Including Tesla and other high-profile stocks across all major sectors
"""

SECTOR_STOCKS = {
    "Technology": {
        "top_stocks": ["AAPL", "MSFT", "NVDA", "GOOGL"],
        "all_stocks": [
            "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "NFLX", "CRM", 
            "ORCL", "ADBE", "INTC", "AMD", "QCOM", "AVGO", "TXN"
        ],
        "sector_name": "Technology",
        "color": "#00d4ff"
    },
    
    "Healthcare": {
        "top_stocks": ["JNJ", "PFE", "UNH", "ABBV"],
        "all_stocks": [
            "JNJ", "PFE", "UNH", "ABBV", "TMO", "DHR", "ABT", "BMY", 
            "LLY", "MRK", "AMGN", "GILD", "BIIB", "REGN", "VRTX"
        ],
        "sector_name": "Healthcare & Biotech",
        "color": "#00ff88"
    },
    
    "Financial": {
        "top_stocks": ["JPM", "BAC", "WFC", "GS"],
        "all_stocks": [
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", 
            "CB", "SPGI", "CME", "ICE", "MCO", "AON", "MMC"
        ],
        "sector_name": "Financial Services",
        "color": "#ff6b35"
    },
    
    "Consumer_Discretionary": {
        "top_stocks": ["AMZN", "TSLA", "HD", "MCD"],
        "all_stocks": [
            "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", 
            "BKNG", "CMG", "MAR", "RCL", "NCLH", "CCL", "ABNB"
        ],
        "sector_name": "Consumer Discretionary",
        "color": "#ff3366"
    },
    
    "Communication": {
        "top_stocks": ["GOOGL", "META", "NFLX", "DIS"],
        "all_stocks": [
            "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR", 
            "TMUS", "TWTR", "SNAP", "PINS", "ROKU", "SPOT", "ZM"
        ],
        "sector_name": "Communication Services",
        "color": "#8b5cf6"
    },
    
    "Consumer_Staples": {
        "top_stocks": ["PG", "KO", "PEP", "WMT"],
        "all_stocks": [
            "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "KHC", 
            "GIS", "K", "CPB", "CAG", "SJM", "HRL", "TSN"
        ],
        "sector_name": "Consumer Staples",
        "color": "#10b981"
    },
    
    "Industrial": {
        "top_stocks": ["BA", "CAT", "GE", "MMM"],
        "all_stocks": [
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX", "LMT", 
            "NOC", "FDX", "CSX", "NSC", "UNP", "PCAR", "IR"
        ],
        "sector_name": "Industrial",
        "color": "#f59e0b"
    },
    
    "Energy": {
        "top_stocks": ["XOM", "CVX", "COP", "EOG"],
        "all_stocks": [
            "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "MPC", "VLO", 
            "PSX", "KMI", "OKE", "EPD", "ET", "WMB", "ENPH"
        ],
        "sector_name": "Energy",
        "color": "#dc2626"
    },
    
    "Utilities": {
        "top_stocks": ["NEE", "DUK", "SO", "D"],
        "all_stocks": [
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", 
            "PEG", "PCG", "ED", "ES", "FE", "ETR", "AWK"
        ],
        "sector_name": "Utilities",
        "color": "#059669"
    },
    
    "Real_Estate": {
        "top_stocks": ["AMT", "PLD", "CCI", "EQIX"],
        "all_stocks": [
            "AMT", "PLD", "CCI", "EQIX", "WELL", "DLR", "PSA", "EXR", 
            "AVB", "EQR", "SPG", "O", "VICI", "WY", "ARE"
        ],
        "sector_name": "Real Estate",
        "color": "#7c3aed"
    }
}

# Configuration for efficient data fetching
FETCH_CONFIG = {
    "batch_size": 5,   # Process 5 stocks at a time (reduced to avoid rate limits)
    "cache_duration": 300,  # Cache for 5 minutes (increased to reduce API calls)
    "timeout": 10,  # 10 second timeout per request
    "max_concurrent": 3,  # Reduced concurrent requests
    "retry_attempts": 1  # Single retry to avoid excessive requests
}

# Priority stocks for immediate analysis (including Tesla)
PRIORITY_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]

def get_all_stocks():
    """Get all unique stocks across all sectors"""
    all_stocks = set()
    for sector_data in SECTOR_STOCKS.values():
        all_stocks.update(sector_data["all_stocks"])
    return sorted(list(all_stocks))

def get_top_stocks_by_sector():
    """Get top 4 stocks from each sector"""
    return {
        sector: data["top_stocks"] 
        for sector, data in SECTOR_STOCKS.items()
    }

def get_sector_for_stock(symbol):
    """Find which sector a stock belongs to"""
    for sector, data in SECTOR_STOCKS.items():
        if symbol in data["all_stocks"]:
            return sector
    return "Unknown"

def get_sector_color(sector):
    """Get color for a sector"""
    return SECTOR_STOCKS.get(sector, {}).get("color", "#6b7280")

# Total stocks: 150 across 10 sectors
TOTAL_STOCKS = len(get_all_stocks())