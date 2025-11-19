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

# Install Python dependencies in smaller batches to reduce peak disk usage
COPY requirements.txt .

# Install numpy first
RUN pip install --no-cache-dir numpy>=1.26.0 && rm -rf /tmp/* /root/.cache

# Install PyTorch CPU-only version (MUCH smaller than GPU version)
RUN pip install --no-cache-dir torch>=2.8.0 --index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /tmp/* /root/.cache

# Install sentence transformers and related packages
RUN pip install --no-cache-dir sentence-transformers>=3.0.0 transformers && rm -rf /tmp/* /root/.cache

# Install FastAPI and uvicorn
RUN pip install --no-cache-dir fastapi>=0.115.0 uvicorn[standard]>=0.30.0 && rm -rf /tmp/* /root/.cache

# Install remaining packages
RUN pip install --no-cache-dir \
    bert-score>=0.3.13 \
    openpyxl>=3.1.0 \
    protobuf>=4.21.0 \
    accelerate>=1.10.0 \
    sentencepiece>=0.1.99 \
    datasets>=4.1.1 \
    langchain==0.0.335 \
    langchain-core==0.0.13 && \
    rm -rf /tmp/* /root/.cache

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
