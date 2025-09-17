"""
WSGI configuration for RAGHeat Portfolio System
Optimized for Hostinger shared hosting deployment
"""

import os
import sys
from pathlib import Path

# Add the project directory to Python path
project_path = Path(__file__).parent
sys.path.insert(0, str(project_path))
sys.path.insert(0, str(project_path / 'api'))

# Set environment variables for production
os.environ.setdefault('PYTHONPATH', str(project_path))
os.environ.setdefault('ENVIRONMENT', 'production')
os.environ.setdefault('HOST', 'www.semanticdataservices.com')

try:
    from api.main import app
    
    # WSGI application for production deployment
    application = app
    
    # For debugging
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8000)),
            log_level="info"
        )
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install fastapi uvicorn pydantic")
    
    # Fallback WSGI application
    def application(environ, start_response):
        status = '500 Internal Server Error'
        headers = [('Content-type', 'text/plain')]
        start_response(status, headers)
        return [b'Application failed to load. Check dependencies.']