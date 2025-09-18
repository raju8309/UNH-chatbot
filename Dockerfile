# Set public url environment variable
ENV PUBLIC_URL=https://whitemount.sr.unh.edu

# Use official Python image
FROM python:3.10-slim

# Install Node.js, npm, and build tools for Python packages
RUN apt-get update && \
    apt-get install -y nodejs npm build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy backend and scrape folders
COPY backend/ ./backend/
COPY scrape/ ./scrape/

# Copy frontend folder
COPY frontend/ ./frontend/

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build frontend
WORKDIR /app/frontend
RUN npm install && npm run build

# Set workdir to backend for running the server
WORKDIR /app/backend

# Expose port
EXPOSE 8003

# Start main script
CMD ["python", "main.py"]