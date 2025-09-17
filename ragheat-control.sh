#!/bin/bash

# RAGHeat Portfolio System Control Script
# Universal service management for Docker and local execution

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
SERVICE_PORTS=(3000 3001 6379 7474 7687 8000 8001 9090)
API_PORT=8001
FRONTEND_PORT=3001
DOCKER_MODE="simple"  # simple or full

# PIDs for local services
API_PID=""
FRONTEND_PID=""

# Function to print colored output
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${PURPLE}[HEADER]${NC} $1"; }

# Display banner
show_banner() {
    clear
    cat << 'EOF'

██████╗  █████╗  ██████╗ ██╗  ██╗███████╗ █████╗ ████████╗
██╔══██╗██╔══██╗██╔════╝ ██║  ██║██╔════╝██╔══██╗╚══██╔══╝
██████╔╝███████║██║  ███╗███████║█████╗  ███████║   ██║   
██╔══██╗██╔══██║██║   ██║██╔══██║██╔══╝  ██╔══██║   ██║   
██║  ██║██║  ██║╚██████╔╝██║  ██║███████╗██║  ██║   ██║   
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   

         Multi-Agent Portfolio Construction System
                     Service Control Center

EOF
}

# Kill all processes on service ports
kill_all_ports() {
    print_status "Cleaning up all service ports..."
    
    for port in "${SERVICE_PORTS[@]}"; do
        if lsof -ti:$port &> /dev/null; then
            print_warning "Killing processes on port $port"
            lsof -ti:$port | xargs kill -9 2>/dev/null || true
            sleep 1
        fi
    done
    
    print_success "Port cleanup completed"
}

# Check if Docker is available and working
check_docker() {
    if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
        if command -v docker-compose &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

# Docker-based service management
docker_start() {
    local compose_file="docker-compose-simple.yml"
    
    print_status "Starting services with Docker..."
    
    # Cleanup first
    docker_stop
    kill_all_ports
    
    # Free up Docker space if needed
    print_status "Cleaning up Docker resources..."
    docker system prune -f &> /dev/null || true
    
    # Start services
    print_status "Building and starting Docker services..."
    if docker-compose -f $compose_file up --build -d; then
        print_success "Docker services started"
        
        # Wait for services
        sleep 10
        print_status "Waiting for services to be ready..."
        
        # Check API
        local count=0
        while [ $count -lt 30 ]; do
            if curl -s http://localhost:$API_PORT/health &> /dev/null; then
                print_success "Portfolio API is ready"
                break
            fi
            sleep 2
            count=$((count + 1))
        done
        
        # Check Frontend
        count=0
        while [ $count -lt 30 ]; do
            if curl -s http://localhost:$FRONTEND_PORT &> /dev/null; then
                print_success "Frontend is ready"
                break
            fi
            sleep 2
            count=$((count + 1))
        done
        
        return 0
    else
        print_error "Docker services failed to start"
        return 1
    fi
}

# Stop Docker services
docker_stop() {
    print_status "Stopping Docker services..."
    
    for compose_file in docker-compose-simple.yml docker-compose-portfolio-agents.yml; do
        if [ -f "$compose_file" ]; then
            docker-compose -f $compose_file down --remove-orphans 2>/dev/null || true
        fi
    done
    
    print_success "Docker services stopped"
}

# Local service management
local_start() {
    print_status "Starting services locally..."
    
    # Stop any existing services
    local_stop
    kill_all_ports
    
    # Check dependencies
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required for local execution"
        return 1
    fi
    
    if ! command -v npm &> /dev/null; then
        print_error "Node.js/npm is required for local execution"
        return 1
    fi
    
    # Install API dependencies
    print_status "Installing API dependencies..."
    if ! pip install fastapi uvicorn pydantic &> /dev/null; then
        print_error "Failed to install API dependencies"
        return 1
    fi
    
    # Start API server
    print_status "Starting Portfolio API server..."
    cd backend/portfolio_agents
    python simple_main.py &
    API_PID=$!
    cd ../..
    
    # Wait for API to be ready
    sleep 5
    if ! curl -s http://localhost:$API_PORT/health &> /dev/null; then
        print_error "Portfolio API failed to start"
        return 1
    fi
    print_success "Portfolio API started (PID: $API_PID)"
    
    # Start Frontend
    print_status "Starting Frontend server..."
    cd frontend
    if [ ! -d node_modules ]; then
        print_status "Installing frontend dependencies..."
        npm install &> /dev/null || {
            print_error "Failed to install frontend dependencies"
            return 1
        }
    fi
    
    PORT=$FRONTEND_PORT npm start &
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend
    sleep 10
    print_status "Waiting for frontend to be ready..."
    local count=0
    while [ $count -lt 60 ]; do
        if curl -s http://localhost:$FRONTEND_PORT &> /dev/null; then
            print_success "Frontend started (PID: $FRONTEND_PID)"
            return 0
        fi
        sleep 2
        count=$((count + 1))
    done
    
    print_error "Frontend failed to start within timeout"
    return 1
}

# Stop local services
local_stop() {
    print_status "Stopping local services..."
    
    # Stop processes by PID if available
    if [ -n "$API_PID" ] && kill -0 $API_PID 2>/dev/null; then
        kill $API_PID 2>/dev/null || true
        print_success "Portfolio API stopped"
    fi
    
    if [ -n "$FRONTEND_PID" ] && kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID 2>/dev/null || true
        print_success "Frontend stopped"
    fi
    
    # Fallback: kill by port
    kill_all_ports
}

# Health check
health_check() {
    print_header "System Health Check:"
    
    local all_healthy=true
    
    # Check API
    echo -n "  Portfolio API ($API_PORT): "
    if curl -s http://localhost:$API_PORT/health | grep -q "healthy"; then
        echo -e "${GREEN}✅ Healthy${NC}"
    else
        echo -e "${RED}❌ Unhealthy${NC}"
        all_healthy=false
    fi
    
    # Check Frontend
    echo -n "  Frontend ($FRONTEND_PORT): "
    if curl -s http://localhost:$FRONTEND_PORT &> /dev/null; then
        echo -e "${GREEN}✅ Accessible${NC}"
    else
        echo -e "${RED}❌ Inaccessible${NC}"
        all_healthy=false
    fi
    
    # Check system endpoints
    echo -n "  System Status: "
    if curl -s http://localhost:$API_PORT/system/status | grep -q "active"; then
        echo -e "${GREEN}✅ Active${NC}"
    else
        echo -e "${YELLOW}⚠️  Not Ready${NC}"
    fi
    
    echo -n "  Portfolio Construction: "
    if curl -s -X POST http://localhost:$API_PORT/portfolio/construct \
        -H "Content-Type: application/json" \
        -d '{"stocks": ["AAPL"], "market_data": {}}' | grep -q "completed"; then
        echo -e "${GREEN}✅ Working${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
        all_healthy=false
    fi
    
    echo ""
    if $all_healthy; then
        print_success "All services are healthy and functional"
        return 0
    else
        print_error "Some services have issues"
        return 1
    fi
}

# Show service information
show_info() {
    print_success "🎉 RAGHeat Portfolio System Information"
    echo ""
    print_header "📊 Service URLs:"
    echo "  • Portfolio Dashboard:    http://localhost:$FRONTEND_PORT"
    echo "  • Portfolio API:          http://localhost:$API_PORT"  
    echo "  • API Documentation:      http://localhost:$API_PORT/docs"
    echo ""
    print_header "🔧 Quick Actions:"
    echo "  • Health Check:           $0 health"
    echo "  • View Status:            $0 status"
    echo "  • Test Endpoints:         $0 test"
    echo "  • Stop System:            $0 stop"
    echo ""
    print_header "📚 How to Use:"
    echo "  1. Open http://localhost:$FRONTEND_PORT in browser"
    echo "  2. Click '🤖 Portfolio AI' navigation button"
    echo "  3. Add stock symbols and construct portfolio"
    echo ""
}

# Test all endpoints
test_endpoints() {
    print_header "Testing Portfolio System Endpoints:"
    echo ""
    
    # Test health
    echo -n "Testing health endpoint: "
    if response=$(curl -s http://localhost:$API_PORT/health); then
        echo -e "${GREEN}✅ $(echo $response | jq -r .status 2>/dev/null || echo 'OK')${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
    fi
    
    # Test system status
    echo -n "Testing system status: "
    if response=$(curl -s http://localhost:$API_PORT/system/status); then
        agents=$(echo $response | jq -r '.agents | length' 2>/dev/null || echo "N/A")
        echo -e "${GREEN}✅ $agents agents${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
    fi
    
    # Test portfolio construction
    echo -n "Testing portfolio construction: "
    if response=$(curl -s -X POST http://localhost:$API_PORT/portfolio/construct \
        -H "Content-Type: application/json" \
        -d '{"stocks": ["AAPL", "GOOGL"], "market_data": {"risk_free_rate": 0.05}}'); then
        status=$(echo $response | jq -r .status 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✅ $status${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
    fi
    
    # Test individual analyses
    for analysis in fundamental sentiment technical heat-diffusion; do
        echo -n "Testing $analysis analysis: "
        if curl -s -X POST http://localhost:$API_PORT/analysis/$analysis \
            -H "Content-Type: application/json" \
            -d '{"stocks": ["AAPL"]}' | grep -q "analysis_type"; then
            echo -e "${GREEN}✅ Working${NC}"
        else
            echo -e "${RED}❌ Failed${NC}"
        fi
    done
    
    echo ""
    print_success "Endpoint testing completed"
}

# Show current status
show_status() {
    print_header "RAGHeat Portfolio System Status:"
    echo ""
    
    # Check if services are running
    echo "Service Status:"
    for port in $API_PORT $FRONTEND_PORT; do
        echo -n "  Port $port: "
        if lsof -ti:$port &> /dev/null; then
            echo -e "${GREEN}✅ Active${NC}"
        else
            echo -e "${RED}❌ Inactive${NC}"
        fi
    done
    
    echo ""
    
    # Check Docker status if available
    if check_docker; then
        echo "Docker Status:"
        if docker-compose -f docker-compose-simple.yml ps &> /dev/null; then
            docker-compose -f docker-compose-simple.yml ps
        else
            echo "  No Docker services running"
        fi
        echo ""
    fi
    
    # Health check
    if lsof -ti:$API_PORT &> /dev/null; then
        health_check
    else
        print_warning "Services not running - cannot perform health check"
    fi
}

# Main execution
main() {
    show_banner
    
    case "${1:-help}" in
        start)
            print_header "Starting RAGHeat Portfolio System"
            if check_docker; then
                print_status "Docker available - using Docker mode"
                if docker_start; then
                    show_info
                else
                    print_warning "Docker start failed - falling back to local mode"
                    if local_start; then
                        show_info
                    else
                        print_error "Both Docker and local startup failed"
                        exit 1
                    fi
                fi
            else
                print_status "Docker not available - using local mode"
                if local_start; then
                    show_info
                else
                    print_error "Local startup failed"
                    exit 1
                fi
            fi
            ;;
        stop)
            print_header "Stopping RAGHeat Portfolio System"
            docker_stop
            local_stop
            print_success "All services stopped"
            ;;
        restart)
            print_header "Restarting RAGHeat Portfolio System"
            main stop
            sleep 3
            main start
            ;;
        status)
            show_status
            ;;
        health)
            health_check
            ;;
        test)
            test_endpoints
            ;;
        info)
            show_info
            ;;
        docker)
            print_header "Starting with Docker (forced)"
            if check_docker; then
                if docker_start; then
                    show_info
                else
                    exit 1
                fi
            else
                print_error "Docker is not available"
                exit 1
            fi
            ;;
        local)
            print_header "Starting with Local Mode (forced)"
            if local_start; then
                show_info
            else
                exit 1
            fi
            ;;
        help|*)
            cat << 'EOF'
Usage: ./ragheat-control.sh {start|stop|restart|status|health|test|info|docker|local|help}

Commands:
  start    - Auto-start system (Docker preferred, local fallback)
  stop     - Stop all services (Docker + local)
  restart  - Stop and start all services
  status   - Show detailed system status
  health   - Perform health check on running services
  test     - Test all API endpoints
  info     - Show service information and URLs
  docker   - Force start with Docker only
  local    - Force start with local mode only
  help     - Show this help message

Examples:
  ./ragheat-control.sh start     # Start the system
  ./ragheat-control.sh health    # Check if everything is working
  ./ragheat-control.sh test      # Test all API endpoints
  ./ragheat-control.sh stop      # Stop everything

The system will automatically:
  • Clean up all ports before starting
  • Try Docker first, fallback to local execution
  • Wait for services to be ready
  • Perform health checks
  • Show you the URLs to access the system

EOF
            ;;
    esac
}

# Handle interruption
trap 'print_warning "Interrupted!"; main stop; exit 1' INT TERM

# Execute
main "$@"