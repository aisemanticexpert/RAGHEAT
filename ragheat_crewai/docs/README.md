# RAGHeat CrewAI Multi-Agent Portfolio Construction System

## Overview

The RAGHeat CrewAI system is a sophisticated multi-agent portfolio construction framework that leverages the latest CrewAI technology to implement the RAGHeat and AlphaAgents research methodologies. The system coordinates multiple specialized AI agents through structured debate and consensus-building to construct optimal risk-adjusted portfolios with full explainability.

## Architecture

### System Components

```
crewai/
├── config/                 # Configuration files
│   ├── agents.yaml        # Agent definitions and roles
│   ├── tasks.yaml         # Task workflows and dependencies
│   └── settings.py        # System configuration
├── agents/                # Specialized agent implementations
│   ├── fundamental_analyst.py
│   ├── sentiment_analyst.py
│   ├── valuation_analyst.py
│   ├── knowledge_graph_engineer.py
│   ├── heat_diffusion_analyst.py
│   ├── portfolio_coordinator.py
│   └── explanation_generator.py
├── tools/                 # Agent tools and utilities
│   ├── fundamental_tools.py
│   ├── sentiment_tools.py
│   ├── valuation_tools.py
│   ├── graph_tools.py
│   ├── heat_diffusion_tools.py
│   ├── portfolio_tools.py
│   └── explanation_tools.py
├── crews/                 # Crew orchestration
│   └── portfolio_crew.py
└── docs/                  # Documentation
    ├── README.md
    ├── agents.md
    ├── tasks.md
    └── examples.md
```

### Core Agents

1. **Fundamental Analyst** - Deep fundamental analysis using SEC filings and financial statements
2. **Sentiment Analyst** - Market sentiment analysis from news, social media, and analyst ratings
3. **Valuation Analyst** - Technical analysis and quantitative valuation metrics
4. **Knowledge Graph Engineer** - Construction and maintenance of financial knowledge graphs
5. **Heat Diffusion Analyst** - Influence propagation modeling using heat diffusion equations
6. **Portfolio Coordinator** - Multi-agent coordination and consensus building
7. **Explanation Generator** - Investment rationale and transparency documentation

## Key Features

### Multi-Agent Coordination
- **Structured Debate**: Agents engage in evidence-based debates to reach consensus
- **Conflict Resolution**: Weighted voting and evidence evaluation for disagreements
- **Consensus Building**: Collaborative decision-making with confidence scoring

### Heat Diffusion Analysis
- **Influence Propagation**: Model how market shocks cascade through interconnected assets
- **Cascade Risk Assessment**: Identify systemic risks and indirect exposures
- **Network Analysis**: Leverage knowledge graphs for relationship-based insights

### Explainable AI
- **Chain-of-Thought Reasoning**: Traceable decision logic for all recommendations
- **Visual Explanations**: Interactive charts and heat maps for decision support
- **Regulatory Compliance**: Audit trails and documentation for institutional requirements

### Risk Management
- **Multi-Dimensional Risk Assessment**: Traditional, behavioral, and systemic risk analysis
- **Stress Testing**: Portfolio resilience under various market scenarios
- **Dynamic Rebalancing**: Adaptive position management based on changing conditions

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Basic Usage

```python
from crewai import PortfolioConstructionCrew

# Initialize the crew
crew = PortfolioConstructionCrew()

# Construct a portfolio
result = crew.construct_portfolio(
    target_stocks=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
    investment_objective="balanced_growth",
    risk_tolerance="moderate",
    time_horizon="1y"
)

# View results
print(result["portfolio_recommendations"])
print(result["explanation_package"])
```

### Advanced Configuration

```python
# Custom configuration
custom_config = {
    "risk_tolerance": "aggressive",
    "max_positions": 15,
    "sector_limits": {"tech": 0.4, "finance": 0.3},
    "esg_constraints": True
}

crew = PortfolioConstructionCrew(custom_config)
```

## Agent Documentation

### Fundamental Analyst Agent

**Role**: Deep fundamental analysis of companies and sectors

**Capabilities**:
- SEC filing analysis (10-K, 10-Q, 8-K)
- Financial statement analysis and ratio calculations
- Sector and competitive analysis
- Long-term value assessment

**Key Methods**:
- `analyze()` - Comprehensive fundamental analysis
- `analyze_financial_ratios()` - Detailed ratio analysis
- `analyze_competitive_position()` - Competitive landscape assessment

**Example Usage**:
```python
fundamental_agent = FundamentalAnalystAgent(config)
result = fundamental_agent.analyze({
    "stocks": ["AAPL", "MSFT"],
    "time_horizon": "1y",
    "benchmark_sector": "technology"
})
```

### Sentiment Analyst Agent

**Role**: Market sentiment and news analysis

**Capabilities**:
- Financial news sentiment analysis
- Social media monitoring (Twitter, Reddit, StockTwits)
- Analyst rating and consensus tracking
- Behavioral finance insights

**Key Methods**:
- `analyze()` - Comprehensive sentiment analysis
- `analyze_news_sentiment()` - News-specific analysis
- `analyze_social_media_sentiment()` - Social platform analysis
- `analyze_analyst_sentiment()` - Professional consensus analysis

### Valuation Analyst Agent

**Role**: Quantitative valuation and technical analysis

**Capabilities**:
- Technical indicator calculations (RSI, MACD, Bollinger Bands)
- Risk-adjusted return metrics
- Volatility and correlation analysis
- Options flow analysis

**Key Methods**:
- `analyze()` - Complete valuation analysis
- `analyze_technical_indicators()` - Technical signal analysis
- `analyze_risk_metrics()` - Risk and volatility assessment
- `analyze_options_flow()` - Options market sentiment

### Knowledge Graph Engineer Agent

**Role**: Financial knowledge graph construction and maintenance

**Capabilities**:
- Entity extraction and relationship modeling
- Semantic network construction
- Graph database management
- Ontology development

**Key Methods**:
- `analyze()` - Graph construction and updates
- `extract_entities()` - Entity identification and extraction
- `model_relationships()` - Relationship mapping
- `query_graph()` - Graph traversal and analysis

### Heat Diffusion Analyst Agent

**Role**: Influence propagation modeling using physics-based approaches

**Capabilities**:
- Heat equation solving on financial networks
- Cascade risk assessment
- Event impact simulation
- Systemic risk quantification

**Key Methods**:
- `analyze()` - Complete diffusion analysis
- `simulate_shock_event()` - Event impact modeling
- `analyze_cascade_risk()` - Systemic risk assessment
- `optimize_diffusion_parameters()` - Model calibration

### Portfolio Coordinator Agent

**Role**: Multi-agent coordination and portfolio optimization

**Capabilities**:
- Structured debate facilitation
- Consensus building and conflict resolution
- Portfolio optimization
- Risk management integration

**Key Methods**:
- `analyze()` - Complete coordination process
- `facilitate_agent_debate()` - Debate moderation
- `optimize_portfolio_allocation()` - Allocation optimization
- `assess_portfolio_risk()` - Risk evaluation

### Explanation Generator Agent

**Role**: Investment rationale and transparency

**Capabilities**:
- Chain-of-thought explanation generation
- Visual representation creation
- Multi-audience communication
- Regulatory compliance documentation

**Key Methods**:
- `analyze()` - Complete explanation generation
- `generate_executive_summary()` - High-level summaries
- `create_position_explanations()` - Detailed rationales
- `design_visualization_package()` - Visual design

## Task Workflows

### 1. Knowledge Graph Construction
**Agent**: Knowledge Graph Engineer
**Purpose**: Build foundational knowledge infrastructure
**Dependencies**: None
**Output**: Comprehensive financial knowledge graph

### 2. Fundamental Analysis
**Agent**: Fundamental Analyst
**Purpose**: Assess long-term company value and health
**Dependencies**: Knowledge Graph
**Output**: Financial health scores and investment recommendations

### 3. Sentiment Analysis
**Agent**: Sentiment Analyst
**Purpose**: Gauge market momentum and perception
**Dependencies**: Knowledge Graph
**Output**: Sentiment scores and short-term outlook

### 4. Valuation Analysis
**Agent**: Valuation Analyst
**Purpose**: Determine optimal entry/exit points
**Dependencies**: None
**Output**: Technical signals and risk-adjusted metrics

### 5. Heat Diffusion Simulation
**Agent**: Heat Diffusion Analyst
**Purpose**: Model cascade risks and indirect exposures
**Dependencies**: Knowledge Graph, Fundamental Analysis, Sentiment Analysis
**Output**: Influence maps and systemic risk assessment

### 6. Agent Debate Facilitation
**Agent**: Portfolio Coordinator
**Purpose**: Synthesize diverse perspectives through structured debate
**Dependencies**: All analysis outputs
**Output**: Consensus recommendations and conflict resolutions

### 7. Portfolio Construction
**Agent**: Portfolio Coordinator
**Purpose**: Build optimal portfolio allocation
**Dependencies**: Agent Debate Results
**Output**: Final portfolio composition and weights

### 8. Investment Rationale Generation
**Agent**: Explanation Generator
**Purpose**: Create comprehensive explanations and documentation
**Dependencies**: Portfolio Construction, Heat Diffusion, Agent Debate
**Output**: Complete explanation package with visualizations

## Configuration

### Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key

# Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### Risk Tolerance Levels

- **Conservative**: Low volatility, high quality, maximum 15% volatility
- **Moderate**: Balanced approach, maximum 25% volatility  
- **Aggressive**: Growth focus, maximum 40% volatility

### Investment Objectives

- **Growth**: Focus on capital appreciation
- **Income**: Emphasis on dividend yield
- **Balanced Growth**: Mix of growth and income
- **Value**: Undervalued opportunities

## Performance Monitoring

### Agent Performance Metrics
- Success rate per agent
- Average execution time
- Confidence levels
- Historical accuracy

### Portfolio Performance
- Risk-adjusted returns
- Sharpe ratio
- Maximum drawdown
- Benchmark comparison

### System Health
- Agent coordination effectiveness
- Consensus achievement rate
- Explanation quality scores
- User satisfaction metrics

## Integration Examples

### REST API Integration

```python
from fastapi import FastAPI
from crewai import PortfolioConstructionCrew

app = FastAPI()
crew = PortfolioConstructionCrew()

@app.post("/construct-portfolio")
async def construct_portfolio(request: PortfolioRequest):
    result = crew.construct_portfolio(
        target_stocks=request.stocks,
        investment_objective=request.objective,
        risk_tolerance=request.risk_tolerance
    )
    return result
```

### Scheduled Analysis

```python
import schedule
import time

def daily_portfolio_review():
    crew = PortfolioConstructionCrew()
    result = crew.construct_portfolio(
        target_stocks=get_watchlist(),
        investment_objective="balanced_growth"
    )
    save_analysis_results(result)

schedule.every().day.at("09:00").do(daily_portfolio_review)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Troubleshooting

### Common Issues

1. **Agent Initialization Failures**
   - Check API key configuration
   - Verify tool dependencies
   - Review agent configuration syntax

2. **Task Execution Errors**
   - Validate input data format
   - Check task dependencies
   - Review timeout settings

3. **Consensus Building Issues**
   - Adjust consensus thresholds
   - Review agent confidence levels
   - Check debate parameters

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

crew = PortfolioConstructionCrew()
crew.construct_portfolio(target_stocks=["AAPL"], debug=True)
```

## Contributing

### Development Setup

```bash
git clone <repository>
cd ragheat-poc/crewai
pip install -e .
pre-commit install
```

### Adding New Agents

1. Create agent class inheriting from `RAGHeatBaseAgent`
2. Implement required methods
3. Add configuration to `agents.yaml`
4. Register tools in tool registry
5. Update crew orchestration

### Adding New Tools

1. Create tool class inheriting from `BaseTool`
2. Implement `_run()` method
3. Register in tool registry
4. Add to agent tool lists

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- GitHub Issues: [Repository Issues](https://github.com/your-org/ragheat-poc/issues)
- Documentation: [Full Documentation](https://docs.ragheat.com)
- Email: support@ragheat.com