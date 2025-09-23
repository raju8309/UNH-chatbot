## UNH Graduate Catalog Chatbot

A chatbot integrated into UNH’s graduate student academic catalog. It helps current and prospective graduate students navigate programs, courses, and policies in the catalog more easily.
 
## Features
 
- Query the graduate catalog for program details, course descriptions, and requirements
- Uses semantic search with embeddings for relevant context retrieval
- Answers questions using a local **Flan-T5 Small** model
- Provides citations for the retrieved information, linking to specific paragraphs from the source(s)
- Automated testing using BERTScore for insightful reports on accurary
 
## Requirements
 
- Python 3.8+
- `pip` dependencies listed in `requirements.txt`
- NPM, Node.js, Tailwind CSS

## Design
 
The chatbot follows a simple retrieval-augmented generation (RAG) architecture:
 
1. **Data Loading**
   - JSON files containing scraped website data are loaded into memory.
   - Each section of the JSON is broken into text chunks, along with source metadata for citations.
 
2. **Embeddings & Semantic Search**
   - Each text chunk is converted into a vector embedding using `sentence-transformers/all-MiniLM-L6-v2`.
   - When a user asks a question, the chatbot computes the embedding of the question and finds the top relevant text chunks using cosine similarity.
 
3. **Answer Generation**
   - The selected text chunks are passed as context to a local **Flan-T5 Small** model (`google/flan-t5-small`).
   - The model generates an answer restricted to the context and includes citations for transparency.
 
4. **User Interface**
   - Displays both the answer and source citations for each query.
   - Built using **Next.js** with TailwindCSS for an interactive web-based chat interface.
   - Follows the [UNH branding guidelines](https://www.unh.edu/marketing/resources).

## Testing
 
- Automated tests are conducted using `automation_testing/run_tests.py`. The script creates answers with `predict.py`, compares those against a created gold set `gold.jsonl` with `evaluator.py`, and finally outputs a report using BERTScore.
   - Sample question: *"What are the requirements for a Master's in Computer Science?"* returns a correct answer with citations and links.
- The speed of responses can be analyzed with `automation_testing/test_times.py`, which outputs average repsonse times.
 
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
docker system prune -a --volumes
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
