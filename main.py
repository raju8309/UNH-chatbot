import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
from functools import lru_cache
from text_fragments import build_text_fragment_url, choose_snippet, is_synthetic_label
from pathlib import Path

# for chunking text
chunks_embeddings = None
chunk_texts = []
chunk_sources = []

# paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "scraper"

# embeddings model (local + small)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# load local Flan-T5 Small for queries
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1,  # CPU currently better performance on GPU (0)
)

def load_json_file(path):
    """Load a JSON file into global embeddings/texts/sources."""
    global chunks_embeddings, chunk_texts, chunk_sources

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_texts = []
    new_sources = []

    def recurse_sections(sections, parent_title=""):
        for sec in sections:
            title = sec.get("title", "")
            full_title = f"{parent_title} > {title}" if parent_title else title

            # add plain text
            for t in sec.get("text", []):
                new_texts.append(t)
                new_sources.append({
                    "title": full_title,
                    "url": data.get("url", "")
                })

            # add links (label + URL)
            for link in sec.get("links", []):
                label = link.get("label")
                url = link.get("url")
                if label and url:
                    new_texts.append(f"Courses: {label}")
                    new_sources.append({
                        "title": label,
                        "url": url
                    })

            if "subsections" in sec:
                recurse_sections(sec["subsections"], parent_title=full_title)

    recurse_sections(data.get("sections", []))

    # append new chunks to existing ones
    if new_texts:
        if chunks_embeddings is None:
            chunks_embeddings = embed_model.encode(new_texts, convert_to_numpy=True)
            chunk_texts.extend(new_texts)
            chunk_sources.extend(new_sources)
        else:
            new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)
            chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])
            chunk_texts.extend(new_texts)
            chunk_sources.extend(new_sources)

        print(f"Loaded {len(new_texts)} chunks from {path}")
    else:
        print(f"WARNING: no text found in {path}")


# retrieval utilities
def get_top_chunks(question, top_k=3):
    if chunks_embeddings is None or len(chunks_embeddings) == 0:
        return []
    question_vec = embed_model.encode([question], convert_to_numpy=True)[0]
    scores = np.dot(chunks_embeddings, question_vec) / (
        np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(question_vec) + 1e-10
    )
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(chunk_texts[i], chunk_sources[i]) for i in top_indices]


def _wrap_sources_with_text_fragments(sources_with_passages, question: str):
    """
    Input: list of tuples (passage_text, source_dict)
    Output: list of dicts like source_dict but with url replaced by a text-fragment URL
    """
    wrapped = []
    for passage, src in sources_with_passages:
        url = src.get("url", "")
        # Some chunks are synthetic labels (e.g., "Courses: …"); don't build fragments for those
        if not url or is_synthetic_label(passage):
            wrapped.append({**src, "url": url})
            continue

        # Choose a compact snippet (tries to align to the question)
        snippet = choose_snippet(passage, hint=question, max_chars=160)

        if snippet:
            frag_url = build_text_fragment_url(url, text=snippet)
            wrapped.append({**src, "url": frag_url})
        else:
            wrapped.append({**src, "url": url})
    return wrapped


# core answer function (not cached directly)
def _answer_question(question):
    top_chunks = get_top_chunks(question)
    context = " ".join([text for text, _ in top_chunks])

    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know. "
        "If the context does not mention the degree, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    try:
        result = qa_pipeline(prompt, max_new_tokens=128)  # limit tokens
        answer = result[0]["generated_text"].strip()

        # Build citation links WITH text fragments, one per unique (title,url)
        enriched_sources = _wrap_sources_with_text_fragments(top_chunks, question)

        seen = set()
        citation_lines = []
        for src in enriched_sources:
            key = (src.get("title"), src.get("url"))
            if key in seen:
                continue
            seen.add(key)
            line = f"- {src.get('title','Source')}"
            if src.get("url"):
                line += f" ({src['url']})"
            citation_lines.append(line)

        citations = "\n".join(citation_lines)
        return f"{answer}\n\nSources:\n{citations}"
    except Exception as e:
        return f"ERROR running local model: {e}"


# cached wrapper to handle Gradio inputs safely
@lru_cache(maxsize=128)
def cached_answer_str(question_str):
    return _answer_question(question_str)


def answer_question(message, history=None):
    # message might be a list from Gradio, convert to string
    if isinstance(message, list):
        message = " ".join(message)
    return cached_answer_str(message)


# UI elements
unh_blue = "#003366"
unh_white = "#FFFFFF"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # header
    with gr.Row(elem_id="header"):
        gr.HTML(
            """
            <div style="background-color: #003366; color: white; padding:20px; border-radius:8px; text-align:center;">
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
            """
            <div style="text-align:center; padding:10px; font-size:14px; color:#555;">
                <hr style="margin:10px 0;">
                <p>Built for the <strong>University of New Hampshire</strong> Graduate Catalog project</p>
            </div>
            """
        )

if __name__ == "__main__":
    # Load scraped JSONs from the scraper/ folder
    filenames = [
        "course_descriptions.json",
        "degree_requirements.json",
        "academic_standards.json",
        "graduation.json",
        "graduation_grading.json",
    ]
    for name in filenames:
        path = DATA_DIR / name
        if path.exists():
            load_json_file(str(path))
        else:
            print(f"WARNING: {path} not found, skipping.")

    demo.launch()
