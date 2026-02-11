from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from datetime import timedelta

# Models
whisper = WhisperModel("small", compute_type="int8")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def fmt(seconds):
    return str(timedelta(seconds=int(seconds)))

# --- Transcribe ---
segments, _ = whisper.transcribe("data/lecture.mp3", word_timestamps=True)
segments = list(segments)

# --- Sliding Window Chunking ---
WINDOW = 5
STRIDE = 2
chunks = []

for i in range(0, len(segments), STRIDE):
    window = segments[i:i+WINDOW]
    if not window:
        continue

    start = fmt(window[0].start)
    end = fmt(window[-1].end)
    text = " ".join(seg.text for seg in window)
    chunks.append(f"[{start} - {end}] {text}")

# --- Embed ---
embeddings = embedder.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

# --- Store FAISS index ---
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "audio.index")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ Indexed {len(chunks)} chunks using FAISS")


