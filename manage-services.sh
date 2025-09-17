#!/bin/bash

# RAGHeat Portfolio System Management Script
# Comprehensive Docker-based service management with cleanup and monitoring

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose-portfolio-agents.yml"
SERVICE_PORTS=(3000 3001 6379 7474 7687 8000 8001 9090 3000)
API_PORT=8001
FRONTEND_PORT=3001
HEALTH_CHECK_TIMEOUT=300

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

print_header() {
    echo -e "${PURPLE}[HEADER]${NC} $1"
}

# Display banner
show_banner() {
    clear
    echo ""
    echo -e "${CYAN}██████╗  █████╗  ██████╗ ██╗  ██╗███████╗ █████╗ ████████╗${NC}"
    echo -e "${CYAN}██╔══██╗██╔══██╗██╔════╝ ██║  ██║██╔════╝██╔══██╗╚══██╔══╝${NC}"
    echo -e "${CYAN}██████╔╝███████║██║  ███╗███████║█████╗  ███████║   ██║${NC}"
    echo -e "${CYAN}██╔══██╗██╔══██║██║   ██║██╔══██║██╔══╝  ██╔══██║   ██║${NC}"
    echo -e "${CYAN}██║  ██║██║  ██║╚██████╔╝██║  ██║███████╗██║  ██║   ██║${NC}"
    echo -e "${CYAN}╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝${NC}"
    echo ""
    echo -e "${PURPLE}Multi-Agent Portfolio Construction System${NC}"
    echo -e "${PURPLE}Service Management Console${NC}"
    echo ""
}

# Check dependencies
check_dependencies() {
    print_status "Checking system dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    print_success "All dependencies are available"
}

# Kill processes on specific ports
kill_port_processes() {
    print_status "Cleaning up processes on service ports..."
    
    for port in "${SERVICE_PORTS[@]}"; do
        if lsof -ti:$port &> /dev/null; then
            print_warning "Killing processes on port $port"
            lsof -ti:$port | xargs kill -9 2>/dev/null || true
            sleep 1
        fi
    done
    
    print_success "Port cleanup completed"
}

# Clean up Docker resources
cleanup_docker() {
    print_status "Cleaning up Docker resources..."
    
    # Stop and remove containers
    if docker-compose -f $COMPOSE_FILE ps -q &> /dev/null; then
        print_status "Stopping existing containers..."
        docker-compose -f $COMPOSE_FILE down --remove-orphans 2>/dev/null || true
    fi
    
    # Remove dangling images and containers
    print_status "Removing dangling Docker resources..."
    docker container prune -f &> /dev/null || true
    docker image prune -f &> /dev/null || true
    docker network prune -f &> /dev/null || true
    
    # Free up disk space if needed
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 5000000 ]; then
        print_warning "Low disk space detected. Running system cleanup..."
        docker system prune -f --volumes &> /dev/null || true
    fi
    
    print_success "Docker cleanup completed"
}

# Setup environment and directories
setup_environment() {
    print_status "Setting up environment..."
    
    # Create necessary directories
    mkdir -p data/neo4j data/redis logs/portfolio-api logs/nginx
    mkdir -p monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources
    
    # Setup environment variables
    if [ ! -f .env ]; then
        print_status "Creating .env file..."
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
    
    # Load environment variables
    set -a
    source .env 2>/dev/null || true
    set +a
    
    print_success "Environment setup completed"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Check available disk space before building
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10000000 ]; then
        print_error "Insufficient disk space for building images (need at least 10GB free)"
        exit 1
    fi
    
    # Build images with no cache to ensure fresh builds
    print_status "Building portfolio API image..."
    docker-compose -f $COMPOSE_FILE build --no-cache portfolio-api
    
    print_status "Building frontend image..."
    docker-compose -f $COMPOSE_FILE build --no-cache portfolio-frontend
    
    print_status "Building original RAGHeat API image..."
    docker-compose -f $COMPOSE_FILE build --no-cache ragheat-api
    
    print_success "All images built successfully"
}

# Start services
start_services() {
    print_status "Starting all services..."
    
    # Pull external images first
    print_status "Pulling external images..."
    docker-compose -f $COMPOSE_FILE pull neo4j redis nginx prometheus grafana 2>/dev/null || true
    
    # Start services in dependency order
    print_status "Starting database services..."
    docker-compose -f $COMPOSE_FILE up -d neo4j redis
    
    print_status "Waiting for databases to be ready..."
    wait_for_service "neo4j" "RETURN 1" "cypher-shell -u neo4j -p password"
    wait_for_service "redis" "ping" "redis-cli"
    
    print_status "Starting API services..."
    docker-compose -f $COMPOSE_FILE up -d portfolio-api ragheat-api
    
    print_status "Starting frontend and proxy..."
    docker-compose -f $COMPOSE_FILE up -d portfolio-frontend nginx
    
    print_status "Starting monitoring services..."
    docker-compose -f $COMPOSE_FILE up -d prometheus grafana
    
    print_success "All services started successfully"
}

# Wait for service to be ready
wait_for_service() {
    local service_name=$1
    local test_command=$2
    local client_command=$3
    local timeout=120
    local count=0
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $count -lt $timeout ]; do
        if docker-compose -f $COMPOSE_FILE exec -T $service_name $client_command "$test_command" &>/dev/null; then
            print_success "$service_name is ready"
            return 0
        fi
        sleep 2
        count=$((count + 2))
        if [ $((count % 20)) -eq 0 ]; then
            print_status "Still waiting for $service_name... ($count/${timeout}s)"
        fi
    done
    
    print_error "$service_name failed to start within timeout"
    return 1
}

# Health check all services
health_check() {
    print_status "Performing health checks..."
    
    local failed_services=()
    
    # Check Portfolio API
    if curl -f http://localhost:$API_PORT/health &>/dev/null; then
        print_success "Portfolio API is healthy"
    else
        print_error "Portfolio API health check failed"
        failed_services+=("portfolio-api")
    fi
    
    # Check Frontend
    if curl -f http://localhost:$FRONTEND_PORT &>/dev/null; then
        print_success "Frontend is accessible"
    else
        print_error "Frontend health check failed"
        failed_services+=("frontend")
    fi
    
    # Check Neo4j
    if curl -f http://localhost:7474 &>/dev/null; then
        print_success "Neo4j browser is accessible"
    else
        print_warning "Neo4j browser may not be ready yet"
    fi
    
    # Check system status
    if curl -s http://localhost:$API_PORT/system/status | grep -q "active"; then
        print_success "Multi-agent system is active"
    else
        print_warning "Multi-agent system status could not be verified"
    fi
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        print_success "All critical services are healthy"
        return 0
    else
        print_error "Failed services: ${failed_services[*]}"
        return 1
    fi
}

# Show service URLs and status
show_service_info() {
    print_success "🎉 RAGHeat Portfolio System is running!"
    echo ""
    print_header "📊 Service URLs:"
    echo "  • Portfolio Dashboard:    http://localhost:$FRONTEND_PORT"
    echo "  • Portfolio API:          http://localhost:$API_PORT"
    echo "  • API Documentation:      http://localhost:$API_PORT/docs"
    echo "  • Neo4j Browser:          http://localhost:7474 (neo4j/password)"
    echo "  • Original RAGHeat API:   http://localhost:8000"
    echo "  • Monitoring (Grafana):   http://localhost:3000 (admin/admin)"
    echo ""
    print_header "🔧 Management Commands:"
    echo "  • View logs:              docker-compose -f $COMPOSE_FILE logs -f"
    echo "  • Check status:           docker-compose -f $COMPOSE_FILE ps"
    echo "  • Stop system:            $0 stop"
    echo "  • Restart system:         $0 restart"
    echo ""
    print_header "📚 Quick Start:"
    echo "  1. Open http://localhost:$FRONTEND_PORT"
    echo "  2. Click '🤖 Portfolio AI' button"
    echo "  3. Add stocks and construct portfolio"
    echo ""
}

# Stop all services
stop_services() {
    print_status "Stopping all services..."
    
    # Stop Docker services gracefully
    if docker-compose -f $COMPOSE_FILE ps -q &> /dev/null; then
        docker-compose -f $COMPOSE_FILE stop
        docker-compose -f $COMPOSE_FILE down --remove-orphans
    fi
    
    # Kill any remaining processes on ports
    kill_port_processes
    
    print_success "All services stopped"
}

# Show logs
show_logs() {
    if [ -n "$2" ]; then
        docker-compose -f $COMPOSE_FILE logs -f "$2"
    else
        docker-compose -f $COMPOSE_FILE logs -f
    fi
}

# Show service status
show_status() {
    print_header "Docker Services Status:"
    docker-compose -f $COMPOSE_FILE ps
    echo ""
    
    print_header "Port Status:"
    for port in "${SERVICE_PORTS[@]}"; do
        if lsof -ti:$port &> /dev/null; then
            echo "  Port $port: ✅ In Use"
        else
            echo "  Port $port: ❌ Free"
        fi
    done
    echo ""
    
    print_header "Health Status:"
    health_check 2>/dev/null && echo "  System Health: ✅ Healthy" || echo "  System Health: ❌ Issues Detected"
}

# Test endpoints
test_endpoints() {
    print_header "Testing API Endpoints:"
    
    # Test health endpoint
    echo -n "  Health Check: "
    if response=$(curl -s http://localhost:$API_PORT/health 2>/dev/null); then
        echo "✅ $(echo $response | jq -r .status 2>/dev/null || echo "OK")"
    else
        echo "❌ Failed"
    fi
    
    # Test system status
    echo -n "  System Status: "
    if response=$(curl -s http://localhost:$API_PORT/system/status 2>/dev/null); then
        agent_count=$(echo $response | jq -r '.agents | length' 2>/dev/null || echo "0")
        echo "✅ $agent_count agents active"
    else
        echo "❌ Failed"
    fi
    
    # Test portfolio construction
    echo -n "  Portfolio Construction: "
    if response=$(curl -s -X POST http://localhost:$API_PORT/portfolio/construct \
        -H "Content-Type: application/json" \
        -d '{"stocks": ["AAPL", "GOOGL"], "market_data": {"risk_free_rate": 0.05}}' 2>/dev/null); then
        status=$(echo $response | jq -r .status 2>/dev/null || echo "unknown")
        echo "✅ $status"
    else
        echo "❌ Failed"
    fi
    
    echo ""
}

# Main function
main() {
    show_banner
    
    case "${1:-help}" in
        start)
            print_header "Starting RAGHeat Portfolio System"
            check_dependencies
            kill_port_processes
            cleanup_docker
            setup_environment
            build_images
            start_services
            sleep 10
            health_check
            show_service_info
            ;;
        stop)
            print_header "Stopping RAGHeat Portfolio System"
            stop_services
            ;;
        restart)
            print_header "Restarting RAGHeat Portfolio System"
            stop_services
            sleep 5
            main start
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "$@"
            ;;
        test)
            test_endpoints
            ;;
        cleanup)
            print_header "Performing Deep Cleanup"
            stop_services
            cleanup_docker
            docker system prune -af --volumes
            print_success "Deep cleanup completed"
            ;;
        help|*)
            echo "Usage: $0 {start|stop|restart|status|logs [service]|test|cleanup|help}"
            echo ""
            echo "Commands:"
            echo "  start    - Start all services with full setup"
            echo "  stop     - Stop all services and cleanup"
            echo "  restart  - Stop and start all services"
            echo "  status   - Show service and system status"
            echo "  logs     - Show service logs (optionally specify service)"
            echo "  test     - Test all API endpoints"
            echo "  cleanup  - Deep cleanup of Docker resources"
            echo "  help     - Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 start                    # Start the entire system"
            echo "  $0 logs portfolio-api       # View portfolio API logs"
            echo "  $0 test                     # Test all endpoints"
            echo ""
            ;;
    esac
}

# Handle script interruption
cleanup_on_exit() {
    print_warning "Script interrupted. Cleaning up..."
    stop_services 2>/dev/null || true
    exit 1
}

trap cleanup_on_exit INT TERM

# Execute main function with all arguments
main "$@"