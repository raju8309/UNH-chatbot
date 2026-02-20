# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PUBLIC_URL=http://localhost:8080
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# Install Node.js, npm, and build tools for Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends nodejs npm build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set work directory
WORKDIR /app

COPY requirements.txt .

# Install numpy first
RUN pip install --no-cache-dir numpy>=1.26.0 && rm -rf /tmp/* /root/.cache

# Install PyTorch CPU-only version
RUN pip install --no-cache-dir torch>=2.8.0 --index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /tmp/* /root/.cache

# ✅ FIX: Pin transformers to version that supports text2text-generation
RUN pip install --no-cache-dir sentence-transformers>=3.0.0 transformers==4.44.0 && rm -rf /tmp/* /root/.cache

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

# Copy application code
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

# Set workdir to backend
WORKDIR /app/backend

# Expose port 8080 for AWS
EXPOSE 8080

# Start backend with uvicorn on port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]