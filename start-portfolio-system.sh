#!/bin/bash

# RAGHeat Portfolio System Startup Script
# This script starts the complete multi-agent portfolio construction system

set -e

echo "🚀 Starting RAGHeat Portfolio Construction System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

# Create necessary directories
setup_directories() {
    print_status "Setting up directories..."
    
    mkdir -p data/neo4j
    mkdir -p data/redis
    mkdir -p logs/portfolio-api
    mkdir -p logs/nginx
    mkdir -p monitoring/prometheus
    mkdir -p monitoring/grafana
    
    print_success "Directories created"
}

# Set environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from .env.example..."
        if [ -f .env.example ]; then
            cp .env.example .env
        else
            cat > .env << 'EOF'
# RAGHeat Portfolio System Environment Variables
ANTHROPIC_API_KEY=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

# Database Configuration
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# API Configuration
MAX_ITERATIONS=10
VERBOSE_LOGGING=true
ENABLE_MEMORY=true
RISK_FREE_RATE=0.05

# Security (change in production)
SECRET_KEY=your-secret-key-change-in-production
EOF
        fi
    fi
    
    # Source environment variables
    set -a
    source .env
    set +a
    
    print_success "Environment variables configured"
}

# Build and start services
start_services() {
    print_status "Building and starting services..."
    
    # Pull latest images
    print_status "Pulling latest images..."
    docker-compose -f docker-compose-portfolio-agents.yml pull --ignore-pull-failures
    
    # Build custom images
    print_status "Building custom images..."
    docker-compose -f docker-compose-portfolio-agents.yml build --no-cache
    
    # Start services
    print_status "Starting services..."
    docker-compose -f docker-compose-portfolio-agents.yml up -d
    
    print_success "Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for Neo4j
    print_status "Waiting for Neo4j to be ready..."
    timeout=120
    while ! docker-compose -f docker-compose-portfolio-agents.yml exec -T neo4j cypher-shell -u neo4j -p password "RETURN 1" &>/dev/null; do
        sleep 5
        timeout=$((timeout - 5))
        if [ $timeout -le 0 ]; then
            print_error "Neo4j failed to start within timeout"
            exit 1
        fi
    done
    print_success "Neo4j is ready"
    
    # Wait for Redis
    print_status "Waiting for Redis to be ready..."
    timeout=60
    while ! docker-compose -f docker-compose-portfolio-agents.yml exec -T redis redis-cli ping &>/dev/null; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            print_error "Redis failed to start within timeout"
            exit 1
        fi
    done
    print_success "Redis is ready"
    
    # Wait for Portfolio API
    print_status "Waiting for Portfolio API to be ready..."
    timeout=180
    while ! curl -f http://localhost:8001/health &>/dev/null; do
        sleep 5
        timeout=$((timeout - 5))
        if [ $timeout -le 0 ]; then
            print_error "Portfolio API failed to start within timeout"
            exit 1
        fi
    done
    print_success "Portfolio API is ready"
    
    # Wait for Frontend
    print_status "Waiting for Frontend to be ready..."
    timeout=120
    while ! curl -f http://localhost:3001 &>/dev/null; do
        sleep 5
        timeout=$((timeout - 5))
        if [ $timeout -le 0 ]; then
            print_error "Frontend failed to start within timeout"
            exit 1
        fi
    done
    print_success "Frontend is ready"
}

# Initialize data
initialize_data() {
    print_status "Initializing system data..."
    
    # Initialize Neo4j with sample data (optional)
    print_status "Setting up Neo4j constraints and indexes..."
    docker-compose -f docker-compose-portfolio-agents.yml exec -T neo4j cypher-shell -u neo4j -p password << 'EOF'
CREATE CONSTRAINT stock_symbol IF NOT EXISTS FOR (s:Stock) REQUIRE s.symbol IS UNIQUE;
CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE;
CREATE INDEX stock_sector IF NOT EXISTS FOR (s:Stock) ON (s.sector);
CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.date);
EOF
    
    print_success "Neo4j initialized"
    
    # Test API endpoints
    print_status "Testing API endpoints..."
    
    # Test portfolio API health
    response=$(curl -s http://localhost:8001/health)
    if echo "$response" | grep -q "healthy"; then
        print_success "Portfolio API health check passed"
    else
        print_warning "Portfolio API health check returned: $response"
    fi
    
    # Test system status
    response=$(curl -s http://localhost:8001/system/status)
    if echo "$response" | grep -q "agents"; then
        print_success "System status check passed"
    else
        print_warning "System status check returned unexpected response"
    fi
}

# Display service URLs
show_service_urls() {
    print_success "🎉 RAGHeat Portfolio System is now running!"
    echo ""
    echo "📊 Service URLs:"
    echo "  • Portfolio Dashboard:    http://localhost:3001"
    echo "  • Portfolio API:          http://localhost:8001"
    echo "  • API Documentation:      http://localhost:8001/docs"
    echo "  • Neo4j Browser:          http://localhost:7474 (neo4j/password)"
    echo "  • Original RAGHeat API:   http://localhost:8000"
    echo "  • Monitoring (Grafana):   http://localhost:3000 (admin/admin)"
    echo ""
    echo "🔧 System Status:"
    echo "  • Check system health:    curl http://localhost:8001/health"
    echo "  • View system status:     curl http://localhost:8001/system/status"
    echo "  • View logs:              docker-compose -f docker-compose-portfolio-agents.yml logs -f"
    echo ""
    echo "📚 Quick Start:"
    echo "  1. Open the Portfolio Dashboard at http://localhost:3001"
    echo "  2. Navigate to the API docs at http://localhost:8001/docs"
    echo "  3. Try the /portfolio/construct endpoint with sample stocks"
    echo ""
    echo "🛑 To stop the system: ./stop-portfolio-system.sh"
}

# Main execution
main() {
    echo ""
    echo "██████╗  █████╗  ██████╗ ██╗  ██╗███████╗ █████╗ ████████╗"
    echo "██╔══██╗██╔══██╗██╔════╝ ██║  ██║██╔════╝██╔══██╗╚══██╔══╝"
    echo "██████╔╝███████║██║  ███╗███████║█████╗  ███████║   ██║"
    echo "██╔══██╗██╔══██║██║   ██║██╔══██║██╔══╝  ██╔══██║   ██║"
    echo "██║  ██║██║  ██║╚██████╔╝██║  ██║███████╗██║  ██║   ██║"
    echo "╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝"
    echo ""
    echo "Multi-Agent Portfolio Construction System"
    echo ""
    
    check_dependencies
    setup_directories
    setup_environment
    start_services
    wait_for_services
    initialize_data
    show_service_urls
}

# Handle script interruption
cleanup() {
    print_warning "Script interrupted. Cleaning up..."
    docker-compose -f docker-compose-portfolio-agents.yml down --remove-orphans
    exit 1
}

trap cleanup INT TERM

# Execute main function
main "$@"