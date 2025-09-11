#!/bin/bash

echo "Starting RAGHeat POC..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Start infrastructure
echo "Starting infrastructure services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start backend
echo "Starting backend API..."
cd backend
python -m api.main &
BACKEND_PID=$!

cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install

# Start frontend
echo "Starting frontend..."
npm start &
FRONTEND_PID=$!

echo "RAGHeat is running!"
echo "Frontend: http://localhost:3000"
echo "API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; docker-compose down; exit" INT
wait