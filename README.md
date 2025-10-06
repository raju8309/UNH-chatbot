# UNH Graduate Catalog Chatbot

A chatbot integrated with UNH’s graduate catalog to help current and prospective graduate students navigate programs, courses, and policies efficiently.

## Features

- Query the graduate catalog for program details, course descriptions, and requirements.  
- Uses **semantic search with embeddings** for context-aware retrieval.  
- Answers questions using a local **Flan-T5 Small** model (optionally fine-tuned).  
- Provides **citations** for retrieved information, linking to specific paragraphs from sources.  
- Automated testing using **BERTScore** for detailed reports on accuracy.  
- Test dashboard for viewing and comparing automated test results.  
- **Instant loading** thanks to caching of all text chunks from the entire sitemap.

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
- All chunks are **preprocessed and cached**, allowing semantic search to run instantly.

### 2. Embeddings & Semantic Search

- Text chunks are converted into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.  
- User questions are embedded and matched with top relevant text chunks via cosine similarity.

### 3. Answer Generation

- Context chunks are passed to a local **Flan-T5 Small** model (`google/flan-t5-small`).  
- Generated answers are limited to the context and include citations for transparency.  
- **Answer hierarchy:**  
  - General pages (like the academic standards page) are cited before specific program pages.  
  - Specific program pages are cited only if the question explicitly asks about them.  
- **Context-aware sessions:**  
  - Users can ask follow-up questions.  
  - Example: If the first question is “What happens if I get a C grade?” and the answer is about certificate programs, the user can follow up with “I’m not in a certificate program” to get information relevant to Ph.D. programs.

### 4. User Interface

- Displays both answers and source citations.  
- Built with **Next.js** and Tailwind CSS for an interactive web chat.  
- Follows [UNH branding guidelines](https://www.unh.edu/marketing/resources).

## Model Training

Before deployment, the **Flan-T5 Small** model can be fine-tuned to improve accuracy. The fine-tuned model is too large to store in GitHub, so this step is **required for local deployment**.

### Steps to Train

1. Ensure Python dependencies are installed (via `requirements.txt`).  
2. Place your `gold.jsonl` file in the `automation_testing` directory.  
3. Run the training script from the repository root:

```bash
cd backend
python3 train.py
```

### The script will:

- Load the gold dataset and generate training examples using the retrieval pipeline.  
- Split the data into training and validation sets.  
- Fine-tune Flan-T5 Small using GPU if available (CPU fallback included).  
- Save the fine-tuned model to `backend/models/flan-t5-small-finetuned`.  
- Evaluate the model and display sample predictions.

### Notes:

- GPU is strongly recommended for faster training.  
- After training, the model moves to CPU for deployment compatibility.  
- Once trained, the backend will automatically use the fine-tuned model.

## Testing

Automated tests are located in `automation_testing/run_tests.py`.

- Creates a timestamped folder in `reports` for each run.  
- Copies the gold set for safe continuous updates.  
- Predictions are generated using `predict.py` and compared against `gold.jsonl` with `evaluator.py`.  
- Generates **BERTScore reports** for accuracy.  
- Response times are analyzed with `automation_testing/test_times.py`.

### Test Dashboard

A dashboard UI allows you to quickly run, interpret, and compare test results:

- After running the program, visit [http://localhost:8003/dashboard](http://localhost:8003/dashboard).

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
docker run -d -p 8003:8003 --name goopy-app goopy-app
```

### Debug UI
```bash
cd frontend
npm run dev
# Connect to localhost:3000 in your browser
```