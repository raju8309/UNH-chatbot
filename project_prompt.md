# UNH Graduate Catalog RAG Chatbot - Project Prompt

## Project Description
Create a sophisticated Retrieval-Augmented Generation (RAG) chatbot system specifically designed to answer questions about the UNH Graduate Catalog using semantic search, contextual retrieval, and fine-tuned language models.

## Technical Requirements

### Backend Architecture
- **Framework**: FastAPI with Uvicorn server (Python 3.9+)
- **Port**: 8003 (development), 8080 (production/AWS)
- **ML Stack**: PyTorch 2.8.0+, Transformers 4.44.0, Sentence-Transformers 3.0.0+
- **Core Models**:
  - `sentence-transformers/all-MiniLM-L6-v2` for document embeddings
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` for result reranking
  - `google/flan-t5-small` for answer generation (with fine-tuning capability)

### Frontend Architecture
- **Framework**: Next.js 15.5.3 with React 19.1.0
- **Styling**: TailwindCSS 4.1.13
- **Language**: TypeScript
- **Build Process**: Static export served by FastAPI backend

### Key Features to Implement

#### 1. Tiered Retrieval System
- **Tier 0**: Gold standard Q&A pairs (highest priority, 3.0x boost)
- **Tier 1**: Academic regulations and policies (1.5x boost)
- **Tier 2**: General graduate information (1.2x boost)
- **Tier 3**: Course descriptions (1.0x boost)
- **Tier 4**: Program-specific content (1.0x boost)

#### 2. Advanced Query Processing Pipeline
1. **Query Enhancement**: Expand acronyms, boost key terms
2. **Semantic Search**: Vector similarity search using embeddings
3. **Tier Boosting**: Apply priority-based scoring multipliers
4. **Re-ranking**: Cross-encoder + TF-IDF refinement
5. **Context Compression**: Extract relevant sentences only
6. **Answer Generation**: Fine-tuned FLAN-T5 model responses

#### 3. Model Training & Fine-tuning
- Custom FLAN-T5-small fine-tuning on catalog-specific Q&A pairs
- Synthetic Q&A generation from catalog content
- GPU-accelerated training with CPU fallback
- Training data format: `[{"query": "...", "answer": "...", "url": "..."}]`
- Model saving to `backend/train/models/flan-t5-small-finetuned/`

#### 4. Automated Testing Framework
- Gold standard evaluation with multiple metrics:
  - **Nugget-based**: Precision, Recall, F1 (key information coverage)
  - **Semantic**: SBERT cosine similarity (meaning preservation)
  - **Token-level**: BERTScore F1 (lexical quality)
  - **Retrieval**: Recall@k, NDCG@k (ranking quality)
- Test data format in `automation_testing/gold.jsonl`

#### 5. Real-time Dashboard
- Test results visualization
- Per-question analysis with metrics
- Category performance breakdown
- Retrieval path analysis
- Model comparison capabilities

## Directory Structure Requirements

```
UNH-chatbot/
├── backend/
│   ├── config/
│   │   ├── retrieval.yaml      # Main configuration
│   │   ├── query_rewrite.json  # Query transformation rules
│   │   └── settings.py         # Configuration loader
│   ├── services/
│   │   ├── chunk_service.py           # Document chunking/indexing
│   │   ├── retrieval_service.py       # Semantic search
│   │   ├── qa_service.py              # Answer generation
│   │   ├── query_pipeline.py          # End-to-end processing
│   │   ├── reranking_service.py       # Result refinement
│   │   ├── compression_service.py     # Context compression
│   │   ├── synthetic_qa_service.py     # Q&A generation
│   │   ├── gold_set_service.py        # Evaluation dataset
│   │   └── calendar_fallback.py       # Date/deadline handling
│   ├── models/
│   │   ├── api_models.py      # Pydantic request/response models
│   │   └── ml_models.py       # Model initialization
│   ├── routers/
│   │   ├── chat.py           # Chat endpoints
│   │   └── dashboard.py      # Dashboard endpoints
│   ├── train/
│   │   ├── train.py         # Training script
│   │   └── data/            # Training data
│   └── main.py              # FastAPI application entry
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Main chat interface
│   │   └── dashboard/       # Dashboard pages
│   ├── public/              # Static assets
│   ├── package.json         # Node.js dependencies
│   └── next.config.ts       # Next.js configuration
├── scraper/                  # Web scraping utilities
├── automation_testing/       # Evaluation framework
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Configuration Requirements

### retrieval.yaml Structure
```yaml
# Retrieval settings
retrieval_sizes:
  topn_default: 120  # Candidates to retrieve
  k: 5               # Final results to return

# Tier boost multipliers
tier_boosts:
  0: 3.0   # Gold set
  1: 1.5   # Academic regulations
  2: 1.2   # General info
  3: 1.0   # Course descriptions
  4: 1.0   # Program-specific

# Feature toggles
enhancements:
  enabled: false
  query_enhancement:
    enabled: false
  reranking:
    enabled: false
  compression:
    enabled: false

# Performance settings
performance:
  max_tokens: 200
  use_finetuned_model: false

# Synthetic Q&A generation
synthetic_qa:
  enabled: true
  boost_synthetic_qa: 1.3
  generate_for_tiers: [1, 2]
```

## API Endpoints Specification

### Chat API
```
POST /chat
Headers: X-Session-Id: <session-id>
Body: {
  "message": "What is the minimum GPA?",
  "history": []
}
Response: {
  "answer": "Graduate students must maintain a 3.0 GPA.",
  "sources": ["https://catalog.unh.edu/..."],
  "retrieval_path": [...],
  "transformed_query": null
}
```

### Dashboard APIs
```
GET  /reports              # Get all test results
POST /run-tests            # Trigger new test run
GET  /dashboard            # Serve dashboard HTML
POST /reset-session        # Clear session
POST /reset                # Clear all sessions
```

## Deployment Requirements

### Local Development
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
npm install
npm run build
```

### Docker Container
```bash
docker build -t goopy-app .
docker run -p 8003:8003 --name goopy-app goopy-app
```

### AWS Elastic Beanstalk
```bash
docker build -t goopy-app .
docker run -d --name goopy-app -p 8003:8003 \
  -v $(pwd)/backend/train/models:/app/backend/train/models \
  goopy-app
```

## Key Dependencies

### Python (requirements.txt)
- numpy>=1.26.0
- sentence-transformers>=3.0.0
- fastapi>=0.115.0
- uvicorn[standard]>=0.30.0
- torch>=2.8.0
- transformers==4.44.0
- bert-score>=0.3.13
- langchain==0.0.335
- datasets>=4.1.1

### Node.js (package.json)
- next: 15.5.3
- react: 19.1.0
- react-dom: 19.1.0
- tailwindcss: 4.1.13
- typescript: 5

## Monitoring & Maintenance

### Container Monitoring
- `monitor.sh`: Real-time container uptime monitoring
- `setup-monitor.sh`: Automated monitoring setup
- Email/Teams integration for alerts

### Performance Tracking
- Chat logging to CSV format
- Real-time dashboard metrics
- Automated test result reporting
- Memory usage optimization

## Success Criteria

1. **Accuracy**: High-quality answers using RAG pipeline
2. **Performance**: Sub-3 second response times
3. **Scalability**: Handle multiple concurrent users
4. **Maintainability**: Clean, documented codebase
5. **Monitoring**: Comprehensive testing and alerting
6. **Deployment**: Containerized production deployment

## Development Guidelines

1. Follow existing code style (black, isort formatting)
2. Implement comprehensive error handling
3. Add logging for debugging and monitoring
4. Write unit tests for core services
5. Document API endpoints and configuration
6. Optimize for memory usage and response time
7. Ensure backward compatibility for configuration changes

This project represents a production-ready RAG implementation with advanced features specifically tailored for academic catalog queries and university information systems.
