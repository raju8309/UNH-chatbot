## UNH Graduate Catalog Chatbot

A chatbot integrated into UNH’s graduate student academic catalog. It helps current and prospective graduate students navigate programs, courses, and policies in the catalog more easily.
 
## Features
 
- Query the graduate catalog for program details, course descriptions, and requirements  
- Uses semantic search with embeddings for relevant context retrieval  
- Answers questions using a local **Flan-T5 Base** model  
- Provides citations for the retrieved information  
 
## Requirements
 
- Python 3.8+  
- `pip` dependencies listed in `requirements.txt`   
 
## Design
 
The chatbot follows a simple retrieval-augmented generation (RAG) architecture:
 
1. **Data Loading**  
   - JSON files containing course descriptions and degree requirements are loaded into memory.  
   - Each section of the JSON is broken into text chunks, along with source metadata for citations.  
 
2. **Embeddings & Semantic Search**  
   - Each text chunk is converted into a vector embedding using `sentence-transformers/all-MiniLM-L6-v2`.  
   - When a user asks a question, the chatbot computes the embedding of the question and finds the top relevant text chunks using cosine similarity.  
 
3. **Answer Generation**  
   - The selected text chunks are passed as context to a local **Flan-T5 Base** model (`google/flan-t5-base`).  
   - The model generates an answer restricted to the context and includes citations for transparency.  
 
4. **User Interface**  
   - Built using **Gradio** for an interactive web-based chat interface.  
   - Displays both the answer and source citations for each query.
 
## Testing
 
- Automated tests conducted using `automation_testing/run_tests.py` with a CSV file of sample questions and expected answers.  
- Sample question: *"What are the requirements for a Master's in Computer Science?"* returns correct answer with citations and links.  
- The testing script checks each answer against expected phrases and reports pass/fail for each case.  
- Manual test evidence can be added if needed, but automated testing ensures consistent validation as the code recieves updates.
 
## Setup & Usage
 
Clone the repository, install dependencies, and run the project. Then, open the local IP address in your web browser to interact with the chatbot.
 
```bash
git clone https://github.com/UNHM-TEAM-PROJECT/Fall2025-Team-Goopy.git
cd Fall2025-Team-Goopy
pip install -r requirements.txt
python3 main.py