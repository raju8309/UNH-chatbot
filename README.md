# UNH Graduate Catalog Chatbot

A chatbot integrated with UNH's graduate catalog to help current and prospective graduate students navigate programs, courses, and policies efficiently.

## Features

- Query the graduate catalog for program details, course descriptions, and requirements.  
- Uses **semantic search with embeddings** for context-aware retrieval.  
- Answers questions using a local **Flan-T5 Small** model (optionally fine-tuned).  
- Provides **citations** for retrieved information, linking to specific paragraphs from sources.  
- Automated testing using **BERTScore** for detailed reports on accuracy.  
- Test dashboard for viewing and comparing automated test results.  
- **Instant loading** thanks to caching of all text chunks from the entire sitemap.
- **Contextual awareness** - maintains conversation context for follow-up questions.

## Requirements

- Python 3.8+  
- Pip dependencies listed in `requirements.txt`  
- Node.js & NPM  
- Tailwind CSS

## Design

The chatbot follows a **retrieval-augmented generation (RAG)** architecture:

### 1. Data Loading & Caching

- JSON files containing scraped catalog data are loaded into memory.  
- Each section is broken into text chunks with source metadata for citations.  
- All chunks are **preprocessed and cached** in `chunks_cache.pkl`, allowing semantic search to run instantly.
- Chunks are automatically assigned tier levels (1-4) based on their source URL for intelligent retrieval prioritization.

### 2. Embeddings & Semantic Search

- Text chunks are converted into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.  
- User questions are embedded and matched with top relevant text chunks via cosine similarity.
- **Intelligent retrieval** with configurable boosts for policy pages, program-specific content, and course descriptions.
- **Context-aware filtering** based on question intent (course queries, policy queries, program-specific queries).

### 3. Answer Generation

- Context chunks are passed to a local **Flan-T5 Small** model (`google/flan-t5-small`).  
- Generated answers are limited to the context and include citations for transparency.  
- **Answer hierarchy:**  
  - General pages (like the academic standards page) are cited before specific program pages.  
  - Specific program pages are cited only if the question explicitly asks about them.  
- **Context-aware sessions:**  
  - Users can ask follow-up questions.  
  - The system maintains session state including intent, program context, and conversation history.
  - Example: If the first question is "What happens if I get a C grade?" and the answer is about certificate programs, the user can follow up with "I'm not in a certificate program" to get information relevant to Ph.D. programs.

### 4. User Interface

- Displays both answers and source citations with **text fragments** for precise highlighting.
- Built with **Next.js** and Tailwind CSS for an interactive web chat.
- Follows [UNH branding guidelines](https://www.unh.edu/marketing/resources).
- **Popular questions** are dynamically generated from chat logs and test sets.
- Session-based conversation with reset capability.

## Model Training

Before deployment, the **Flan-T5 Small** model can be fine-tuned to improve accuracy. The fine-tuned model is too large to store in GitHub, so this step is **optional but recommended** for local deployment.

### Steps to Train

1. Ensure Python dependencies are installed (via `requirements.txt`).  
2. Place your training data JSON files in the `backend/train/data/` directory.  
3. Run the training script from the backend directory:

```bash
cd backend
python3 train/train.py
```

### The script will:

- Load training data and generate training examples using the retrieval pipeline.  
- Split the data into training and validation sets.  
- Fine-tune Flan-T5 Small using GPU if available (CPU fallback included).  
- Save the fine-tuned model to `backend/train/models/flan-t5-small-finetuned`.  
- Evaluate the model and display sample predictions.
- Create `.out` verification files showing the training data context and answers.

### Notes:

- GPU is strongly recommended for faster training.  
- After training, the model moves to CPU for deployment compatibility.  
- Once trained, the backend will automatically use the fine-tuned model.

## Testing

Automated tests are located in `automation_testing/run_tests.py`.

- Creates a timestamped folder in `automation_testing/reports/` for each run.  
- Copies the gold set (`gold.jsonl`) for safe continuous updates.  
- Predictions are generated using the full retrieval pipeline and compared against `gold.jsonl` with `evaluator.py`.  
- Generates **BERTScore reports** with detailed per-question and aggregate metrics including:
  - Nugget precision, recall, and F1
  - SBERT cosine similarity (answer vs reference and answer vs retrieved chunks)
  - BERTScore F1
  - Retrieval metrics: Recall@1, Recall@3, Recall@5, NDCG@1, NDCG@3, NDCG@5
- Response times can be analyzed with `automation_testing/test_time.py`.

### Contextual Awareness Testing

A separate test suite in `automation_testing/contextual_awareness/` validates the chatbot's ability to maintain context across multi-turn conversations:

```bash
cd automation_testing/contextual_awareness
python3 run_context_tests.py
```

### Test Dashboard

A comprehensive dashboard UI allows you to quickly run, interpret, and compare test results:

- After running the program, visit [http://localhost:8003/dashboard](http://localhost:8003/dashboard).
- **Features:**
  - View all historical test runs with summary metrics
  - Drill down into individual questions with detailed metrics
  - Compare two test runs side-by-side
  - Filter by question category and search
  - View retrieved documents and their relevance
  - Run new tests directly from the dashboard

## Setup & Usage

Clone the repository and install dependencies. You will only need to run the backend, as the frontend is bundled with it. Connect to the local IP address output, http://localhost:8003/, to view the chatbot in your browser.

### Run Locally

**Install Repo**
```bash
git clone https://github.com/UNHM-TEAM-PROJECT/Fall2025-Team-Goopy.git
cd Fall2025-Team-Goopy
pip install -r requirements.txt
```

**Export Frontend**
```bash
cd frontend
npm install
npm run build
```

**Run Backend**
```bash
cd ../backend
python3 main.py
```

The backend will:
- Load retrieval configuration from `config/retrieval.yaml`
- Initialize models (embeddings and QA pipeline)
- Load or build the chunks cache
- Serve the chatbot at http://localhost:8003/
- Serve the test dashboard at http://localhost:8003/dashboard

**Run Containerized (Optional)**
```bash
docker system prune -a --volumes # Free space before building (optional)
docker build -t goopy-app .
docker run -p 8003:8003 --name goopy-app -e PUBLIC_URL=http://localhost:8003/ goopy-app
```

### Deploy

```bash
git clone https://github.com/UNHM-TEAM-PROJECT/Fall2025-Team-Goopy.git
cd Fall2025-Team-Goopy
docker system prune -a --volumes
docker build -t goopy-app .
docker run -d -p 8003:8003 \
  -v $(pwd)/backend/train/models:/app/backend/train/models \
  goopy-app
```

### Debug UI
```bash
cd frontend
npm run dev
# Connect to localhost:3000 in your browser
```

## Configuration

The chatbot's retrieval behavior can be tuned via `backend/config/retrieval.yaml`:

- **policy_terms**: Keywords that trigger policy-focused retrieval
- **tier_boosts**: Multipliers for each tier level (1-4) in retrieval ranking
- **intent keywords**: Terms that help classify user questions (courses, degrees, admissions, etc.)
- **nudges**: Fine-grained score adjustments for specific content types
- **guarantees**: Ensure certain tier results appear for specific question types
- **tier4_gate**: Embedding-based filtering for program pages
- **retrieval_sizes**: Control how many candidates to retrieve (topn) and return (k)
- **course_filters**: Strict subject filtering for course code queries

## Architecture

### Backend Structure

- **config/**: Configuration files and settings
  - `retrieval.yaml`: Retrieval behavior configuration
  - `settings.py`: Configuration loader
- **models/**: Model definitions and initialization
  - `ml_models.py`: Embedding and QA model management
  - `api_models.py`: Request/response schemas
- **services/**: Core business logic
  - `chunk_service.py`: Data loading and chunk management
  - `retrieval_service.py`: Semantic search and ranking
  - `qa_service.py`: Answer generation with citations
  - `query_pipeline.py`: End-to-end question processing
  - `intent_service.py`: Intent detection and context management
  - `session_service.py`: Session state management
- **routers/**: API endpoints
  - `chat.py`: Chat and session endpoints
  - `dashboard.py`: Test dashboard endpoints
- **utils/**: Helper functions
  - `course_utils.py`: Course code detection and parsing
  - `program_utils.py`: Program matching and fuzzy search
  - `logging_utils.py`: Chat interaction logging
- **train/**: Model training infrastructure

### Frontend Structure

- **app/**: Next.js app directory
  - `page.tsx`: Main chat interface
  - `dashboard/page.tsx`: Test results dashboard
  - `globals.css`: Global styles with UNH branding
- **public/**: Static assets
  - `popular_questions.json`: Dynamically generated question suggestions
- **gen_questions.py**: Script to update popular questions from logs and tests

## Data Pipeline

1. **Scraping** (`scraper/scrape_catalog.py`):
   - Reads URLs from `graduate_catalog.xlsx`
   - Fetches and parses HTML content
   - Extracts structured data (sections, text, lists, links)
   - Assigns tier levels and metadata
   - Outputs to `unh_catalog.json`

2. **Loading** (`backend/services/chunk_service.py`):
   - Parses JSON catalog data
   - Creates text chunks with metadata
   - Generates embeddings
   - Caches to `chunks_cache.pkl` for fast startup

3. **Retrieval** (`backend/services/retrieval_service.py`):
   - Embeds user question
   - Computes cosine similarity scores
   - Applies tier boosts and intent-based filters
   - Returns ranked, relevant chunks

4. **Generation** (`backend/services/qa_service.py`):
   - Constructs context from top chunks
   - Generates answer with Flan-T5
   - Adds citations with text fragment URLs
   - Applies domain-specific fallbacks (credits, GRE, course details)

## Logging and Analytics

- Chat interactions are logged to `chat_logs.csv` with timestamps
- Logs feed into popular questions generation
- Test results stored in timestamped directories under `automation_testing/reports/`
- Each test run includes: predictions, gold standard copy, and detailed metrics report
