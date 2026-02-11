# 🎧 Audio RAG System (Whisper + FAISS + Timestamp Retrieval)

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system for audio files.

The system:
- Transcribes audio using Whisper
- Splits transcript into timestamped chunks
- Stores embeddings in FAISS
- Retrieves relevant segments
- Answers questions with exact time ranges

---

## Architecture

1. Audio File → Whisper Transcription
2. Transcript → Chunking with Timestamps
3. Chunk Embeddings → SentenceTransformer
4. Stored in FAISS Index
5. Query → Embedding → Top-K Retrieval
6. LLM generates answer using retrieved context
7. Output includes timestamp range

---

##  Tech Stack

- Python
- Whisper (Speech-to-Text)
- SentenceTransformers
- FAISS (Vector Search)
- Local LLM (Optional)
- Pickle for chunk storage

---

