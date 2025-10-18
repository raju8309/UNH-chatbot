# Use official Python image
FROM python:3.10-slim

# Set public url environment variable
ENV PUBLIC_URL=https://whitemount-t3.sr.unh.edu/

# Install Node.js, npm, and build tools for Python packages
RUN apt-get update && \
    apt-get install -y nodejs npm build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy backend, scraper, frontend, and automation_testing folders
COPY backend/ ./backend/
COPY scraper/ ./scraper/
COPY frontend/ ./frontend/
COPY automation_testing/ ./automation_testing/

# Copy chat_logs.csv for frontend build
COPY chat_logs.csv ./chat_logs.csv

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build frontend
WORKDIR /app/frontend

# Update gen_questions.py to use the correct path if needed
RUN sed -i 's|..\/chat_logs.csv|\/app\/chat_logs.csv|g' gen_questions.py || true

RUN npm install && npm run build

# Set workdir to backend for running the server
WORKDIR /app/backend

# Expose port
EXPOSE 8003
