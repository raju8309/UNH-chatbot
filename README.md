## UNH Graduate Catalog Chatbot

A chatbot integrated into UNH’s graduate student academic catalog. It helps current and prospective graduate students navigate programs, courses, and policies in the catalog more easily.
 
## Features
 
- Query the graduate catalog for program details, course descriptions, and requirements
- Uses semantic search with embeddings for relevant context retrieval
- Answers questions using a local **Flan-T5 Small** model
- Provides citations for the retrieved information
 
## Requirements
 
- Python 3.8+
- `pip` dependencies listed in `requirements.txt`
- NPM, Node.js (WSL if on Windows), Tailwind CSS

## Design
 
The chatbot follows a simple retrieval-augmented generation (RAG) architecture:
 
1. **Data Loading**
   - JSON files containing course descriptions and degree requirements are loaded into memory.
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
 
- Automated tests conducted using `test/run_tests.py` with a CSV file of sample questions and expected answers.
- Sample question: *"What are the requirements for a Master's in Computer Science?"* returns correct answer with citations and links.
- The testing script checks each answer against expected phrases and reports pass/fail for each case.
- Manual test evidence can be added if needed, but automated testing ensures consistent validation as the code recieves updates.
- The speed of responses can be analyzed with test_times.py, which outputs average repsonse times.
 
## Setup & Usage
 
Clone the repository, install dependencies, and run the project. You will then have to run the backend and frontend. Connect to the local IP address output by the frontend, http://192.168.0.20:3000/, to view the chatbot in your browser.

**Install Repo**
```bash
git clone https://github.com/UNHM-TEAM-PROJECT/Fall2025-Team-Goopy.git
cd Fall2025-Team-Goopy
pip install -r requirements.txt
```

**Run Backend**
```bash
python3 backend/main.py
```

**Run Frontend**
```bash
cd frontend
npm install
npm run dev