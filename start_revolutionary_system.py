#!/usr/bin/env python3
"""
RAGHeat Revolutionary Trading System - Master Startup Script
Launches all services with proper initialization and monitoring

🚀 Revolutionary Features:
- Viral Heat Propagation Engine
- Hierarchical Sector Analysis  
- Advanced ML Predictions
- Knowledge Graph Reasoning
- Option Pricing & Strategies
- Real-time WebSocket Streams
"""

import os
import sys
import subprocess
import time
import signal
import requests
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import webbrowser
from datetime import datetime

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    """Print startup banner"""
    banner = f"""
{Colors.HEADER}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 RAGHeat Revolutionary Trading System 🚀                ║
║                                                                              ║
║  🔥 Viral Heat Propagation    📊 Hierarchical Analysis                      ║
║  🧠 Advanced ML Models        🕸️  Knowledge Graph Reasoning                  ║
║  📈 Option Pricing           ⚡ Real-time WebSocket Streams                 ║
║                                                                              ║
║                        🎯 TARGET: 1000% RETURNS 🎯                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.ENDC}
"""
    print(banner)

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"{Colors.FAIL}❌ Python 3.8+ required. Current version: {version.major}.{version.minor}{Colors.ENDC}")
        return False
    
    print(f"{Colors.OKGREEN}✅ Python {version.major}.{version.minor}.{version.micro} - Compatible{Colors.ENDC}")
    return True

def check_requirements():
    """Check if required packages are installed"""
    print(f"{Colors.OKBLUE}📦 Checking dependencies...{Colors.ENDC}")
    
    required_packages = [
        'fastapi', 'uvicorn', 'websockets', 'pandas', 'numpy', 
        'scikit-learn', 'yfinance', 'networkx', 'asyncio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"{Colors.OKGREEN}  ✅ {package}{Colors.ENDC}")
        except ImportError:
            missing_packages.append(package)
            print(f"{Colors.FAIL}  ❌ {package} - Missing{Colors.ENDC}")
    
    if missing_packages:
        print(f"{Colors.WARNING}⚠️  Installing missing packages...{Colors.ENDC}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"{Colors.OKGREEN}  ✅ Installed {package}{Colors.ENDC}")
            except subprocess.CalledProcessError:
                print(f"{Colors.FAIL}  ❌ Failed to install {package}{Colors.ENDC}")
                return False
    
    return True

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    return None

def check_port_availability():
    """Check if required ports are available"""
    print(f"{Colors.OKBLUE}🔌 Checking port availability...{Colors.ENDC}")
    
    required_ports = {
        8001: "Revolutionary API Server",
        3000: "Frontend React App", 
        8000: "Legacy API Server (optional)"
    }
    
    available_ports = {}
    
    for port, service in required_ports.items():
        alternative_port = find_available_port(port)
        if alternative_port == port:
            print(f"{Colors.OKGREEN}  ✅ Port {port} available for {service}{Colors.ENDC}")
            available_ports[service] = port
        elif alternative_port:
            print(f"{Colors.WARNING}  ⚠️  Port {port} occupied, using {alternative_port} for {service}{Colors.ENDC}")
            available_ports[service] = alternative_port
        else:
            print(f"{Colors.FAIL}  ❌ No available ports found for {service}{Colors.ENDC}")
            return None
    
    return available_ports

def kill_existing_processes():
    """Kill any existing processes on our ports"""
    ports_to_check = [8001, 3000, 8000]
    
    for port in ports_to_check:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections'] or []:
                    if conn.laddr.port == port:
                        print(f"{Colors.WARNING}🔄 Killing existing process on port {port} (PID: {proc.info['pid']}){Colors.ENDC}")
                        proc.kill()
                        time.sleep(1)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue

class ServiceManager:
    """Manages the lifecycle of all services"""
    
    def __init__(self):
        self.processes = {}
        self.project_root = Path(__file__).parent
        self.backend_path = self.project_root / "backend"
        self.frontend_path = self.project_root / "frontend" 
        
    def start_revolutionary_api(self, port=8001):
        """Start the revolutionary API server"""
        print(f"{Colors.OKBLUE}🚀 Starting Revolutionary API Server on port {port}...{Colors.ENDC}")
        
        api_script = self.backend_path / "api" / "revolutionary_main.py"
        
        if not api_script.exists():
            print(f"{Colors.FAIL}❌ Revolutionary API script not found: {api_script}{Colors.ENDC}")
            return None
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.backend_path)
        
        try:
            # Start the revolutionary API server
            process = subprocess.Popen([
                sys.executable, str(api_script)
            ], 
            cwd=str(self.backend_path),
            env=env,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
            )
            
            self.processes['revolutionary_api'] = process
            
            # Wait for startup
            max_wait = 30
            for i in range(max_wait):
                try:
                    response = requests.get(f'http://localhost:{port}/health', timeout=2)
                    if response.status_code == 200:
                        print(f"{Colors.OKGREEN}✅ Revolutionary API Server started successfully!{Colors.ENDC}")
                        return process
                except requests.RequestException:
                    pass
                
                time.sleep(1)
                if i % 5 == 0:
                    print(f"{Colors.WARNING}⏳ Waiting for API server... ({i}/{max_wait}){Colors.ENDC}")
            
            print(f"{Colors.FAIL}❌ Revolutionary API Server failed to start within {max_wait} seconds{Colors.ENDC}")
            return None
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error starting Revolutionary API: {str(e)}{Colors.ENDC}")
            return None
    
    def start_legacy_api(self, port=8000):
        """Start the legacy API server (optional)"""
        print(f"{Colors.OKBLUE}📡 Starting Legacy API Server on port {port}...{Colors.ENDC}")
        
        api_script = self.backend_path / "api" / "main_live.py"
        
        if not api_script.exists():
            print(f"{Colors.WARNING}⚠️  Legacy API script not found, skipping...{Colors.ENDC}")
            return None
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.backend_path)
        
        try:
            process = subprocess.Popen([
                sys.executable, str(api_script)
            ],
            cwd=str(self.backend_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
            )
            
            self.processes['legacy_api'] = process
            
            # Quick check
            time.sleep(3)
            try:
                response = requests.get(f'http://localhost:{port}/', timeout=2)
                if response.status_code == 200:
                    print(f"{Colors.OKGREEN}✅ Legacy API Server started{Colors.ENDC}")
                    return process
            except requests.RequestException:
                pass
            
            print(f"{Colors.WARNING}⚠️  Legacy API may not be fully ready{Colors.ENDC}")
            return process
            
        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Could not start Legacy API: {str(e)}{Colors.ENDC}")
            return None
    
    def start_frontend(self, port=3000):
        """Start the React frontend"""
        print(f"{Colors.OKBLUE}🎨 Starting React Frontend on port {port}...{Colors.ENDC}")
        
        if not self.frontend_path.exists():
            print(f"{Colors.FAIL}❌ Frontend directory not found: {self.frontend_path}{Colors.ENDC}")
            return None
        
        package_json = self.frontend_path / "package.json"
        if not package_json.exists():
            print(f"{Colors.FAIL}❌ package.json not found in frontend directory{Colors.ENDC}")
            return None
        
        # Check if node_modules exists
        node_modules = self.frontend_path / "node_modules"
        if not node_modules.exists():
            print(f"{Colors.WARNING}📦 Installing frontend dependencies...{Colors.ENDC}")
            try:
                subprocess.check_call(['npm', 'install'], cwd=str(self.frontend_path))
                print(f"{Colors.OKGREEN}✅ Frontend dependencies installed{Colors.ENDC}")
            except subprocess.CalledProcessError as e:
                print(f"{Colors.FAIL}❌ Failed to install frontend dependencies: {str(e)}{Colors.ENDC}")
                return None
        
        try:
            env = os.environ.copy()
            env['PORT'] = str(port)
            env['BROWSER'] = 'none'  # Don't auto-open browser
            
            process = subprocess.Popen(['npm', 'start'],
                                     cwd=str(self.frontend_path),
                                     env=env,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
            
            self.processes['frontend'] = process
            
            # Wait for frontend to start
            max_wait = 60  # Frontend takes longer
            print(f"{Colors.WARNING}⏳ Frontend starting... (this may take up to 60 seconds){Colors.ENDC}")
            
            for i in range(max_wait):
                try:
                    response = requests.get(f'http://localhost:{port}', timeout=2)
                    if response.status_code == 200:
                        print(f"{Colors.OKGREEN}✅ React Frontend started successfully!{Colors.ENDC}")
                        return process
                except requests.RequestException:
                    pass
                
                time.sleep(1)
                if i % 10 == 0 and i > 0:
                    print(f"{Colors.WARNING}⏳ Still starting frontend... ({i}/{max_wait}){Colors.ENDC}")
            
            print(f"{Colors.WARNING}⚠️  Frontend may still be starting...{Colors.ENDC}")
            return process
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error starting frontend: {str(e)}{Colors.ENDC}")
            return None
    
    def stop_all_services(self):
        """Stop all running services"""
        print(f"{Colors.WARNING}🛑 Stopping all services...{Colors.ENDC}")
        
        for service_name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"{Colors.WARNING}  Stopping {service_name}...{Colors.ENDC}")
                try:
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                    print(f"{Colors.OKGREEN}  ✅ {service_name} stopped{Colors.ENDC}")
                except Exception as e:
                    print(f"{Colors.FAIL}  ❌ Error stopping {service_name}: {str(e)}{Colors.ENDC}")
        
        self.processes.clear()
    
    def show_service_status(self):
        """Show status of all services"""
        print(f"\n{Colors.OKBLUE}📊 Service Status:{Colors.ENDC}")
        
        services = [
            ("Revolutionary API", "http://localhost:8001/health"),
            ("React Frontend", "http://localhost:3000"),
            ("Legacy API", "http://localhost:8000")
        ]
        
        for service_name, url in services:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"{Colors.OKGREEN}  ✅ {service_name} - Running{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}  ⚠️  {service_name} - Status: {response.status_code}{Colors.ENDC}")
            except requests.RequestException:
                print(f"{Colors.FAIL}  ❌ {service_name} - Not responding{Colors.ENDC}")

def show_urls():
    """Show all service URLs"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}🌐 Service URLs:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}📊 Revolutionary Dashboard: http://localhost:3000{Colors.ENDC}")
    print(f"{Colors.OKCYAN}📖 API Documentation:     http://localhost:8001/docs{Colors.ENDC}")
    print(f"{Colors.OKCYAN}🔥 API Health Check:      http://localhost:8001/health{Colors.ENDC}")
    print(f"{Colors.OKCYAN}📈 System Status:         http://localhost:8001/api/system/status{Colors.ENDC}")
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}🔌 WebSocket Streams:{Colors.ENDC}")
    print(f"{Colors.OKBLUE}⚡ Live Signals:         ws://localhost:8001/ws/live-signals{Colors.ENDC}")
    print(f"{Colors.OKBLUE}🔥 Heat Propagation:     ws://localhost:8001/ws/heat-propagation{Colors.ENDC}")
    print(f"{Colors.OKBLUE}📊 Sector Analysis:      ws://localhost:8001/ws/sector-analysis{Colors.ENDC}")
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}🚀 Revolutionary Features:{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  • Viral Heat Propagation Engine{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  • Hierarchical Sector-Stock Analysis{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  • Advanced ML Ensemble (7+ models){Colors.ENDC}")
    print(f"{Colors.OKGREEN}  • Knowledge Graph Reasoning{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  • Advanced Option Pricing & Greeks{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  • Real-time Risk Management{Colors.ENDC}")

def main():
    """Main startup function"""
    print_banner()
    
    # Pre-flight checks
    if not check_python_version():
        sys.exit(1)
    
    if not check_requirements():
        print(f"{Colors.FAIL}❌ Dependency check failed{Colors.ENDC}")
        sys.exit(1)
    
    # Check ports
    available_ports = check_port_availability()
    if not available_ports:
        print(f"{Colors.FAIL}❌ Port availability check failed{Colors.ENDC}")
        sys.exit(1)
    
    # Kill existing processes
    kill_existing_processes()
    
    # Initialize service manager
    service_manager = ServiceManager()
    
    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        print(f"\n{Colors.WARNING}🛑 Received shutdown signal...{Colors.ENDC}")
        service_manager.stop_all_services()
        print(f"{Colors.OKGREEN}✅ Revolutionary Trading System shutdown complete{Colors.ENDC}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start services
        print(f"\n{Colors.HEADER}{Colors.BOLD}🚀 Starting Revolutionary Trading System...{Colors.ENDC}")
        
        # 1. Start Revolutionary API (most important)
        api_process = service_manager.start_revolutionary_api(available_ports.get("Revolutionary API Server", 8001))
        if not api_process:
            print(f"{Colors.FAIL}❌ Critical: Revolutionary API failed to start{Colors.ENDC}")
            sys.exit(1)
        
        # 2. Start Frontend  
        frontend_process = service_manager.start_frontend(available_ports.get("Frontend React App", 3000))
        
        # 3. Start Legacy API (optional)
        legacy_process = service_manager.start_legacy_api(available_ports.get("Legacy API Server (optional)", 8000))
        
        # Show status
        time.sleep(2)
        service_manager.show_service_status()
        show_urls()
        
        # Open browser to dashboard
        print(f"\n{Colors.OKGREEN}🎉 Revolutionary Trading System is now running!{Colors.ENDC}")
        
        try:
            webbrowser.open('http://localhost:3000')
            print(f"{Colors.OKCYAN}🌐 Opening dashboard in browser...{Colors.ENDC}")
        except:
            pass
        
        print(f"\n{Colors.WARNING}Press Ctrl+C to stop all services{Colors.ENDC}")
        
        # Keep running
        while True:
            time.sleep(10)
            
            # Check if critical processes are still running
            if api_process and api_process.poll() is not None:
                print(f"{Colors.FAIL}❌ Revolutionary API has stopped unexpectedly{Colors.ENDC}")
                break
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"{Colors.FAIL}❌ Unexpected error: {str(e)}{Colors.ENDC}")
    finally:
        service_manager.stop_all_services()

if __name__ == "__main__":
    main()