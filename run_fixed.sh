#!/bin/bash

echo "Starting RAGHeat POC..."

# Function to kill processes on a specific port
kill_port() {
    local port=$1
    echo "Checking port $port..."
    local pids=$(lsof -ti:$port)
    if [ ! -z "$pids" ]; then
        echo "Killing processes on port $port: $pids"
        kill -9 $pids 2>/dev/null
        sleep 2
    fi
}

# Kill any existing processes on ports 3000 and 8000
kill_port 3000
kill_port 8000

# Check if Docker is running (optional for demo)
if docker info > /dev/null 2>&1; then
    echo "Docker is available. Starting infrastructure services..."
    docker-compose up -d
    echo "Waiting for services to start..."
    sleep 5
else
    echo "Docker not running. Skipping infrastructure services (demo will work without them)."
fi

# Install minimal Python dependencies for demo
echo "Installing minimal Python dependencies..."
pip install fastapi uvicorn pydantic python-dotenv > /dev/null 2>&1

# Start backend with simplified version
echo "Starting backend API..."
cd backend
python api/main_simple.py &
BACKEND_PID=$!

cd ..

# Wait for backend to start
sleep 3

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install --silent > /dev/null 2>&1

# Start frontend
echo "Starting frontend..."
npm start &
FRONTEND_PID=$!

echo ""
echo "🚀 RAGHeat is starting up..."
echo "⏳ Please wait for the services to fully initialize..."
echo ""
echo "📍 Services will be available at:"
echo "   Frontend: http://localhost:3000"
echo "   API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "✋ Press Ctrl+C to stop all services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    if docker info > /dev/null 2>&1; then
        docker-compose down 2>/dev/null
    fi
    echo "✅ All services stopped"
    exit
}

# Set trap for cleanup
trap cleanup INT

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Test if services are running
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ Backend API is running"
else
    echo "❌ Backend API failed to start"
fi

# Wait for interrupt
wait