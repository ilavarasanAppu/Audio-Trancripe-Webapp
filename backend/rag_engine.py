"""
=============================================================================
VaniScript AI - Audio & Video Semantic RAG Engine
=============================================================================
This module provides fast semantic retrieval and natural language Q&A across
all transcribed audio and video files.

It features:
  - Multi-script Unicode tokenization (Tamil, Telugu, Malayalam, Kannada, Hindi, English).
  - Sublinear TF-IDF and n-gram vector matching.
  - Natural language answer synthesis with exact audio timestamp citations.

Functions:
  - build_rag_index(): Builds the search index from SQLite transcripts.
  - query_audio_rag(question, top_k): Searches transcripts and returns citations & answers.
=============================================================================
"""

import os
import sys
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import get_db

# Global vector index variables
_GLOBAL_VECTORIZER: Optional[TfidfVectorizer] = None
_GLOBAL_TFIDF_MATRIX = None
_GLOBAL_CHUNK_IDS: List[int] = []

def build_rag_index() -> bool:
    """
    Reads all transcribed chunks from the SQLite database and builds a TF-IDF semantic search matrix.
    Supports English as well as South Indian scripts (Tamil, Telugu, Malayalam, Kannada, Hindi).
    """
    global _GLOBAL_VECTORIZER, _GLOBAL_TFIDF_MATRIX, _GLOBAL_CHUNK_IDS

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.id, c.text, c.start_time, c.end_time, f.filename, f.file_path, s.name as speaker_name
        FROM transcript_chunks c
        JOIN files f ON c.file_id = f.id
        LEFT JOIN speakers s ON c.speaker_id = s.id
        WHERE c.text IS NOT NULL AND LENGTH(TRIM(c.text)) > 0
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        _GLOBAL_VECTORIZER = None
        _GLOBAL_TFIDF_MATRIX = None
        _GLOBAL_CHUNK_IDS = []
        return False

    corpus = [r["text"] for r in rows]
    _GLOBAL_CHUNK_IDS = [r["id"] for r in rows]
    
    # Token pattern matches English letters + Indic Unicode blocks:
    # \u0B80-\u0BFF (Tamil), \u0C00-\u0C7F (Telugu), \u0D00-\u0D7F (Malayalam),
    # \u0C80-\u0CFF (Kannada), \u0900-\u097F (Devanagari/Hindi)
    _GLOBAL_VECTORIZER = TfidfVectorizer(
        ngram_range=(1, 3),
        token_pattern=r'(?u)\b[\w\u0B80-\u0BFF\u0C00-\u0C7F\u0D00-\u0D7F\u0C80-\u0CFF\u0900-\u097F]+\b',
        sublinear_tf=True
    )
    _GLOBAL_TFIDF_MATRIX = _GLOBAL_VECTORIZER.fit_transform(corpus)
    return True

def query_audio_rag(question: str, top_k: int = 6) -> Dict[str, Any]:
    """
    Queries the semantic audio search index and generates a natural language answer with timestamps.
    
    Parameters:
      - question: User question or search keywords
      - top_k: Number of relevant audio clips to return
      
    Returns:
      - Dict containing 'question', 'answer', and 'citations' list.
    """
    build_rag_index()

    if _GLOBAL_VECTORIZER is None or _GLOBAL_TFIDF_MATRIX is None or len(_GLOBAL_CHUNK_IDS) == 0:
        return {
            "question": question,
            "answer": "No audio files have been indexed yet. Enter a folder path in the Scanner above and click 'Scan Folder' -> 'Index All'.",
            "citations": []
        }

    # Transform query to vector
    try:
        q_vec = _GLOBAL_VECTORIZER.transform([question])
        scores = cosine_similarity(q_vec, _GLOBAL_TFIDF_MATRIX).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
    except Exception as e:
        print(f"[RAG] Search error: {e}")
        top_indices = []

    matched_chunk_ids = [_GLOBAL_CHUNK_IDS[i] for i in top_indices if scores[i] > 0.04]
    
    if not matched_chunk_ids:
        return {
            "question": question,
            "answer": f"I couldn't find any direct matches in your media library for '{question}'. Try rephrasing or searching for related keywords.",
            "citations": []
        }

    conn = get_db()
    cursor = conn.cursor()
    placeholders = ",".join(["?"] * len(matched_chunk_ids))
    
    cursor.execute(f"""
        SELECT c.id as chunk_id, c.file_id, c.start_time, c.end_time, c.text, c.language,
               f.filename, f.file_path,
               COALESCE(s.name, s.display_label, 'Unknown Speaker') as speaker_name
        FROM transcript_chunks c
        JOIN files f ON c.file_id = f.id
        LEFT JOIN speakers s ON c.speaker_id = s.id
        WHERE c.id IN ({placeholders})
    """, matched_chunk_ids)
    
    fetched_rows = cursor.fetchall()
    conn.close()

    # Map retrieved records with scores
    id_to_score = { _GLOBAL_CHUNK_IDS[idx]: float(scores[idx]) for idx in top_indices }
    citations = []
    
    for r in fetched_rows:
        item = dict(r)
        item["similarity_score"] = round(id_to_score.get(item["chunk_id"], 0.0), 3)
        citations.append(item)

    # Sort citations by relevance score
    citations.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Generate synthesized summary answer
    context_snippets = [f'[{c["filename"]} @ {int(c["start_time"])}s-{int(c["end_time"])}s]: "{c["text"]}"' for c in citations[:3]]
    synthesized_answer = (
        f"Based on {len(citations)} relevant audio segment(s) in your library: "
        + " ... ".join(context_snippets)
    )

    return {
        "question": question,
        "answer": synthesized_answer,
        "citations": citations
    }

# Backward compatibility class wrapper
class AudioRAGEngine:
    def build_index(self):
        return build_rag_index()
    def query(self, question: str, top_k: int = 6):
        return query_audio_rag(question, top_k)

rag_engine = AudioRAGEngine()
