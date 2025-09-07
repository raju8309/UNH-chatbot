import gradio as gr
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# declare global vars for chunking pdf
pdf_chunks_embeddings = None
pdf_chunk_texts = []

# embeddings model (local + small)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# load local Flan-T5 Base for queries
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base", # requires ~1gb 
    device=-1  # CPU currently better performance on GPU (0)
)

# pdf utilities
def load_and_chunk_pdf(file_paths, chunk_size=500):
    global pdf_chunks_embeddings, pdf_chunk_texts
    pdf_chunks_embeddings = []
    pdf_chunk_texts = []

    if not file_paths:
        return "No files uploaded."

    for file_path in file_paths:
        with pdfplumber.open(file_path.name) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                words = text.split()
                for j in range(0, len(words), chunk_size):
                    chunk_text = " ".join(words[j:j+chunk_size])
                    pdf_chunk_texts.append(f"[Page {i}] {chunk_text}")

    if pdf_chunk_texts:
        pdf_chunks_embeddings = embed_model.encode(pdf_chunk_texts, convert_to_numpy=True)
        return f"loaded {len(pdf_chunk_texts)} chunks from {len(file_paths)} PDF(s)"
    else:
        pdf_chunks_embeddings = None
        return "ERROR: no text found in the uploaded PDF(s)"

# retrieval utilities
def get_top_chunks(question, top_k=3):
    if pdf_chunks_embeddings is None or len(pdf_chunks_embeddings) == 0:
        return ["no PDF loaded"]

    question_vec = embed_model.encode([question], convert_to_numpy=True)[0]

    scores = np.dot(pdf_chunks_embeddings, question_vec) / (
        np.linalg.norm(pdf_chunks_embeddings, axis=1) * np.linalg.norm(question_vec) + 1e-10
    )
    top_indices = scores.argsort()[-top_k:][::-1]
    return [pdf_chunk_texts[i] for i in top_indices]

# answer generation with Flan-T5 Base
def answer_question(message, history=None):
    top_chunks = get_top_chunks(message)
    context = " ".join(top_chunks)

    system_msg = (
        "Answer the question ONLY using the provided PDF context. "
        "If the answer cannot be found in the context, say you don't know."
    )

    prompt = f"{system_msg}\n\nContext:\n{context}\n\nQuestion: {message}\nAnswer:"

    try:
        result = qa_pipeline(prompt, max_new_tokens=256)
        return result[0]["generated_text"].strip()
    except Exception as e:
        return f"ERROR running local model: {e}"

# UI elements

unh_blue = "#003366"
unh_white = "#FFFFFF"
unh_accent = "#00AEEF"

custom_css = f"""
#header-box {{
    background-color: {unh_blue};
    color: {unh_white};
    padding: 20px;
    border-radius: 8px;
    text-align: center;
}}
#header-box h1, #header-box p {{
    color: {unh_white};
}}

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # header
    with gr.Row(elem_id="header"):
        gr.HTML(
            """
            <div id="header-box">
                <h1>UNH Graduate Catalog Chatbot</h1>
                <p>Ask questions about programs, courses, and policies from the UNH Graduate Catalog</p>
            </div>
            """
        )

    # chatbot
    chatbot = gr.ChatInterface(
        fn=answer_question,
        type="messages",
        title="",
        description="",
    )

    # footer
    with gr.Row(elem_id="footer"):
        gr.Markdown(
            f"""
            <div style="text-align:center; padding:10px; font-size:14px; color:#555;">
                <hr style="margin:10px 0;">
                <p>Built for the <strong>University of New Hampshire</strong> Graduate Catalog project</p>
            </div>
            """
        )

if __name__ == "__main__":
    demo.launch()
