#!/bin/bash
# RAGHeat Trading System Deployment Script

set -e

echo "🚀 RAGHeat Trading System Deployment"
echo "====================================="

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

# Check if Docker is running
check_docker() {
    print_status "Checking Docker installation..."
    if ! docker --version > /dev/null 2>&1; then
        print_error "Docker is not installed or not running"
        exit 1
    fi
    
    if ! docker-compose --version > /dev/null 2>&1; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    print_success "Docker and Docker Compose are available"
}

# Create environment file if it doesn't exist
setup_environment() {
    print_status "Setting up environment..."
    
    if [ ! -f .env ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please edit .env file with your actual configuration values"
    else
        print_success "Environment file exists"
    fi
}

# Build and start services
deploy_services() {
    print_status "Building and starting RAGHeat services..."
    
    # Pull latest images
    print_status "Pulling latest Docker images..."
    docker-compose -f docker-compose-full.yml pull
    
    # Build custom images
    print_status "Building custom images..."
    docker-compose -f docker-compose-full.yml build --no-cache
    
    # Start services
    print_status "Starting services..."
    docker-compose -f docker-compose-full.yml up -d
    
    print_success "All services started successfully"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be healthy..."
    
    services=("postgres" "redis" "neo4j" "kafka" "ragheat-backend")
    
    for service in "${services[@]}"; do
        print_status "Waiting for $service..."
        
        timeout=120
        while [ $timeout -gt 0 ]; do
            if docker-compose -f docker-compose-full.yml ps $service | grep -q "Up (healthy)"; then
                print_success "$service is healthy"
                break
            fi
            
            sleep 5
            timeout=$((timeout - 5))
        done
        
        if [ $timeout -le 0 ]; then
            print_error "$service failed to become healthy"
            docker-compose -f docker-compose-full.yml logs $service
            exit 1
        fi
    done
}

# Initialize databases and services
initialize_services() {
    print_status "Initializing services..."
    
    # Initialize Neo4j with sample data (optional)
    print_status "Initializing Neo4j database..."
    # You can add Neo4j initialization commands here
    
    # Initialize Kafka topics
    print_status "Creating Kafka topics..."
    docker-compose -f docker-compose-full.yml exec kafka kafka-topics --create \
        --topic ragheat.market.overview \
        --bootstrap-server localhost:9092 \
        --partitions 3 \
        --replication-factor 1 \
        --if-not-exists || true
    
    docker-compose -f docker-compose-full.yml exec kafka kafka-topics --create \
        --topic ragheat.market.heat \
        --bootstrap-server localhost:9092 \
        --partitions 3 \
        --replication-factor 1 \
        --if-not-exists || true
        
    docker-compose -f docker-compose-full.yml exec kafka kafka-topics --create \
        --topic ragheat.graph.neo4j \
        --bootstrap-server localhost:9092 \
        --partitions 3 \
        --replication-factor 1 \
        --if-not-exists || true
    
    print_success "Services initialized successfully"
}

# Display service URLs
display_urls() {
    print_success "RAGHeat Trading System deployed successfully!"
    echo ""
    echo "🌐 Service URLs:"
    echo "================================"
    echo "Frontend (React):     http://localhost:3000"
    echo "Backend API:          http://localhost:8001"
    echo "Neo4j Browser:        http://localhost:7474"
    echo "Prometheus:           http://localhost:9090"
    echo "Grafana:              http://localhost:3001"
    echo ""
    echo "📊 Default Credentials:"
    echo "================================"
    echo "Neo4j:               neo4j / ragheat_neo4j"
    echo "Grafana:             admin / ragheat_admin"
    echo ""
    echo "📋 Useful Commands:"
    echo "================================"
    echo "View logs:           docker-compose -f docker-compose-full.yml logs -f [service]"
    echo "Stop services:       docker-compose -f docker-compose-full.yml down"
    echo "Restart service:     docker-compose -f docker-compose-full.yml restart [service]"
    echo "Scale service:       docker-compose -f docker-compose-full.yml up -d --scale [service]=3"
    echo ""
    echo "🧪 Run tests:         cd backend && python -m pytest tests/"
    echo "📈 Monitor Kafka:     docker-compose -f docker-compose-full.yml exec kafka kafka-console-consumer --topic ragheat.market.overview --bootstrap-server localhost:9092"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up old containers and images..."
    docker system prune -f
    print_success "Cleanup completed"
}

# Health check
health_check() {
    print_status "Running health checks..."
    
    # Check API health
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        print_success "Backend API is healthy"
    else
        print_error "Backend API health check failed"
        return 1
    fi
    
    # Check frontend
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        print_success "Frontend is healthy"
    else
        print_error "Frontend health check failed"
        return 1
    fi
    
    print_success "All health checks passed"
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        "deploy")
            check_docker
            setup_environment
            deploy_services
            wait_for_services
            initialize_services
            display_urls
            ;;
        "stop")
            print_status "Stopping all services..."
            docker-compose -f docker-compose-full.yml down
            print_success "All services stopped"
            ;;
        "restart")
            print_status "Restarting all services..."
            docker-compose -f docker-compose-full.yml restart
            print_success "All services restarted"
            ;;
        "health")
            health_check
            ;;
        "cleanup")
            cleanup
            ;;
        "logs")
            service=${2:-}
            if [ -z "$service" ]; then
                docker-compose -f docker-compose-full.yml logs -f
            else
                docker-compose -f docker-compose-full.yml logs -f $service
            fi
            ;;
        *)
            echo "Usage: $0 {deploy|stop|restart|health|cleanup|logs [service]}"
            echo ""
            echo "Commands:"
            echo "  deploy    - Full deployment (default)"
            echo "  stop      - Stop all services"
            echo "  restart   - Restart all services"
            echo "  health    - Run health checks"
            echo "  cleanup   - Clean up Docker resources"
            echo "  logs      - View logs (optionally for specific service)"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"