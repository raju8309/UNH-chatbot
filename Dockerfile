# Use official Python image
FROM python:3.10-slim

# Set public url environment variable
ENV PUBLIC_URL=https://whitemount-t3.sr.unh.edu/
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# Install Node.js, npm, and build tools for Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends nodejs npm build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set work directory
WORKDIR /app

# Install Python dependencies FIRST (better caching and avoid space issues)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip /tmp/*

# NOW copy application code
COPY backend/ ./backend/
COPY scraper/ ./scraper/
COPY frontend/ ./frontend/
COPY automation_testing/ ./automation_testing/
COPY chat_logs.csv ./chat_logs.csv

# Build frontend
WORKDIR /app/frontend
RUN sed -i 's|..\/chat_logs.csv|\/app\/chat_logs.csv|g' gen_questions.py || true
RUN npm install && \
    npm run build && \
    rm -rf node_modules /root/.npm /tmp/*

# Set workdir to backend for running the server
WORKDIR /app/backend

# Expose port
EXPOSE 8003

# Start backend
CMD ["python3", "main.py"]
