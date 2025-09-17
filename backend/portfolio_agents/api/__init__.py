"""
API Module for RAGHeat Portfolio Construction System
==================================================

FastAPI-based API endpoints for accessing the multi-agent portfolio construction system.
"""

from .main import app
from .routes import *

__all__ = ['app']