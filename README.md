# RAGHeat POC - Real-time Stock Recommendation System

    ## 🚀 Quick Start

    ### Prerequisites
    - Python 3.9+
    - Node.js 16+
    - Docker and Docker Compose

    ### Installation & Setup

    1. **Clone and navigate to project:**
```bash
    cd ragheat-poc
    2. **Set up environment variables:**
    # Edit .env file and add your API keys
    ANTHROPIC_API_KEY=your_anthropic_key_here
    ALPHA_VANTAGE_KEY=your_alpha_vantage_key
    FINNHUB_KEY=your_finnhub_key

    3. **Start infrastructure services:**
    docker-compose up -d
    4. **Install Python dependencies:**
    pip install -r requirements.txt
    5. **Start the backend:**
    cd backend
    python -m api.main
    6. **In a new terminal, set up frontend:**
    cd frontend
    npm install
    npm start
    7. **Access the application:**
    - Frontend: http://localhost:3000
    - API Documentation: http://localhost:8000/docs
    - Neo4j Browser: http://localhost:7474

    ## 📊 Features

    - **Real-time Knowledge Graph**: Hierarchical structure (Root → Sectors → Stocks)
    - **Heat Diffusion Algorithm**: Mathematical model for influence propagation
    - **Multi-Agent System**: Fundamental, Sentiment, and Valuation agents using CrewAI
    - **Real-time Updates**: WebSocket support for live heat distribution
    - **RESTful API**: Comprehensive endpoints with Swagger documentation
    - **Interactive Dashboard**: React-based visualization

    ## 🏗️ Architecture

    ### Heat Diffusion Model
    The system implements the discrete heat equation on graphs:
    - dh(t)/dt = -βL·h(t)
    - Heat propagates from high-momentum stocks to related entities
    - Multi-hop influence paths are calculated

    ### Multi-Agent Collaboration
    - **Fundamental Agent**: Analyzes financial statements and metrics
    - **Sentiment Agent**: Processes news and social media sentiment
    - **Valuation Agent**: Performs technical analysis
    - **Orchestrator**: Synthesizes analyses and generates recommendations

    ## 📡 API Endpoints

    - `GET /api/graph/structure` - Get knowledge graph structure
    - `GET /api/heat/distribution` - Get current heat distribution
    - `POST /api/analyze/stock` - Analyze specific stock
    - `GET /api/recommendations/top` - Get top recommendations
    - `WS /ws/heat-updates` - Real-time heat updates

    ## 🧪 Testing
    # Run backend tests
        pytest backend/tests/

        # Run frontend tests
        cd frontend && npm test
        ## 🔧 Configuration

    Edit `backend/config/settings.py` to customize:
    - Heat diffusion parameters
    - Update intervals
    - Stock universe
    - API configurations

    ## 📈 Production Deployment

    1. Use environment variables for sensitive data
    2. Deploy with Docker/Kubernetes
    3. Set up monitoring (Prometheus/Grafana)
    4. Configure rate limiting and authentication
    5. Use production database (PostgreSQL/MongoDB)

    ## 🤝 Contributing

    1. Fork the repository
    2. Create feature branch
    3. Commit changes
    4. Push to branch
    5. Create Pull Request

    ## 📄 License

    MIT License

    ## 🆘 Support

    For issues or questions, please open a GitHub issue.