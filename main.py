import gradio as gr
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# declare global vars for chunking pdf
pdf_chunks_embeddings = None
pdf_chunk_texts = []

# embeddings model (local + small)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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

def answer_question(message, history, hf_token: gr.OAuthToken | None = None):
    """
    message: str (latest user input)
    history: list of dicts [{"role":"user","content":...}, {"role":"assistant","content":...}]
    hf_token: optional Hugging Face login token
    """

    # Build context from PDFs
    top_chunks = get_top_chunks(message)
    context = " ".join(top_chunks)

    # System instructions
    system_msg = (
        "You are a helpful assistant that answers ONLY using the provided PDF context. "
        "If the answer cannot be found in the context, say you don't know."
    )

    # Rebuild chat history into messages for the model
    messages = [{"role": "system", "content": system_msg}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}\nAnswer:"})

    try:
        client = InferenceClient(
            model="openai/gpt-oss-20b",
            token=(hf_token.token if (hf_token and getattr(hf_token, 'token', None)) else None),
        )

        resp = client.chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.3,
            top_p=0.9,
            stop=["\nQuestion:", "\nContext:"],
            stream=False,
        )

        if resp.choices and resp.choices[0].message:
            return resp.choices[0].message.content.strip()

        return "No answer returned."

    except Exception as e:
        return f"ERROR calling Inference API: {e}"

# UI elements
with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.Markdown("## PDF Chat")
        gr.LoginButton()
        pdf_files = gr.Files(label="Upload PDF(s)", file_types=[".pdf"])
        pdf_status = gr.Textbox(label="PDF status", interactive=False)

    chatbot = gr.ChatInterface(
        fn=answer_question,
        type="messages",
        title="Ask questions about your PDFs",
        description="Upload PDFs in the sidebar first, then chat with them.",
    )

    pdf_files.upload(
        fn=load_and_chunk_pdf,
        inputs=[pdf_files],
        outputs=[pdf_status],
    )

if __name__ == "__main__":
    demo.launch()
