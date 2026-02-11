from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import ollama

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load index & chunks
index = faiss.read_index("audio.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

SYSTEM_PROMPT = """
You are an audio assistant.
Below are transcript segments with timestamps.
Answer the question and ALWAYS specify the time range where the answer was found.
"""

def ask(question):
    q_emb = embedder.encode([question]).astype("float32")

    _, ids = index.search(q_emb, 2)
    context = "\n\n".join(chunks[i] for i in ids[0])

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}
"""

    response = ollama.generate(model="qwen2.5:3b", prompt=prompt)
    return response["response"]
