FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-simple.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-simple.txt

# Copy the entire application
COPY ragheat-poc .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "backend.api.simple_main"]