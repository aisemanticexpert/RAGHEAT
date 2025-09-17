#!/usr/bin/env python3
"""
Simple RAGHeat Trading System Startup
Minimal version that starts core services
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def start_api_server():
    """Start the API server with minimal dependencies"""
    print("🚀 Starting RAGHeat API Server...")
    
    backend_path = Path(__file__).parent / "backend"
    
    # Start the existing live API server which has fewer dependencies
    api_script = backend_path / "api" / "main_live.py"
    
    if api_script.exists():
        env = os.environ.copy()
        env['PYTHONPATH'] = str(backend_path)
        
        process = subprocess.Popen([
            sys.executable, str(api_script)
        ], cwd=str(backend_path), env=env)
        
        print("✅ API Server started on port 8000")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("🔥 Live Heat Updates: ws://localhost:8000/ws/heat-updates")
        return process
    else:
        print("❌ API script not found")
        return None

def start_frontend():
    """Start the React frontend"""
    print("🎨 Starting React Frontend...")
    
    frontend_path = Path(__file__).parent / "frontend"
    
    if frontend_path.exists():
        try:
            env = os.environ.copy()
            env['PORT'] = '3000'
            env['BROWSER'] = 'none'
            
            process = subprocess.Popen(['npm', 'start'], 
                                     cwd=str(frontend_path), 
                                     env=env)
            
            print("✅ Frontend starting on port 3000")
            print("🌐 Dashboard: http://localhost:3000")
            return process
        except Exception as e:
            print(f"❌ Error starting frontend: {e}")
            return None
    else:
        print("❌ Frontend directory not found")
        return None

def main():
    """Main startup function"""
    print("🔥 RAGHeat Trading System - Simple Startup")
    print("=" * 50)
    
    processes = []
    
    # Start API server
    api_process = start_api_server()
    if api_process:
        processes.append(api_process)
    
    # Wait a bit for API to start
    time.sleep(3)
    
    # Start frontend
    frontend_process = start_frontend()
    if frontend_process:
        processes.append(frontend_process)
    
    print("\n🎉 Services started!")
    print("\n📊 Available Services:")
    print("• API Server: http://localhost:8000")
    print("• API Docs: http://localhost:8000/docs")
    print("• Dashboard: http://localhost:3000")
    print("• Heat Updates: ws://localhost:8000/ws/heat-updates")
    
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            for process in processes:
                if process.poll() is not None:
                    print("⚠️ A service has stopped")
                    
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        for process in processes:
            try:
                process.terminate()
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
            except:
                pass
        print("✅ All services stopped")

if __name__ == "__main__":
    main()