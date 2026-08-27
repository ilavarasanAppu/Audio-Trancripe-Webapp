"""
=============================================================================
VaniScript AI - Speaker Diarization & Voice Classification Module
=============================================================================
This module extracts acoustic voice embeddings from speech segments and uses
unsupervised clustering (Agglomerative Hierarchical Clustering) to group
speech into distinct human speakers without requiring cloud APIs.

Functions:
  - extract_voice_features(y, sr): Extracts a 46-dimensional voice fingerprint
    (MFCCs, spectral centroid, spectral rolloff, zero-crossing rate).
  - extract_segment_voice_features(audio_path, start_time, end_time): Slices
    and extracts embeddings from a specific timestamp in an audio file.
  - cluster_and_assign_speakers(chunks): Clusters embeddings and assigns
    speakers into the database.
=============================================================================
"""

import os
import sys
import json
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import get_db

def extract_voice_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts a rich 46-dimensional voice fingerprint from an audio array.
    
    Acoustic features computed:
      1. MFCCs (20 coefficients: mean & std = 40 dims)
      2. Spectral Centroid (mean & std = 2 dims) - captures voice brightness/timbre
      3. Spectral Rolloff (mean & std = 2 dims) - captures high frequency cutoff
      4. Zero Crossing Rate (mean & std = 2 dims) - captures voice noisiness/fricatives
    """
    # Pad short audio chunks if under 0.3s
    if len(y) < sr * 0.3:
        y = np.pad(y, (0, int(sr * 0.3) - len(y)))
    
    # 1. Mel-Frequency Cepstral Coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 2. Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = np.mean(cent)
    cent_std = np.std(cent)

    # 3. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Concatenate into 46-dim feature vector
    feature_vector = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [cent_mean, cent_std, rolloff_mean, rolloff_std, zcr_mean, zcr_std]
    ])

    # L2-normalize to unit length for cosine distance
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm

    return feature_vector

def extract_segment_voice_features(
    audio_path: str, 
    start_time: float, 
    end_time: float, 
    sr: int = 16000
) -> Optional[np.ndarray]:
    """
    Loads a specific timestamp slice of an audio file and extracts its voice embedding.
    """
    try:
        duration = max(0.5, end_time - start_time)
        y, _ = librosa.load(audio_path, sr=sr, offset=max(0, start_time), duration=duration, mono=True)
        if len(y) == 0:
            return None
        return extract_voice_features(y, sr)
    except Exception as e:
        print(f"[Diarization] Error extracting voice features for {audio_path} [{start_time}-{end_time}]: {e}")
        return None

def cluster_and_assign_speakers(
    chunks: List[Dict[str, Any]], 
    similarity_threshold: float = 0.78
) -> List[Dict[str, Any]]:
    """
    Clusters speech segments by speaker timbre and maps them to known speaker profiles in SQLite.
    
    Parameters:
      - chunks: List of transcript chunk dicts with 'voice_embedding' arrays.
      - similarity_threshold: Minimum cosine similarity to merge with existing speaker.
      
    Returns:
      - chunks with assigned 'speaker_id' and 'speaker_name'.
    """
    valid_chunks = [ch for ch in chunks if ch.get("voice_embedding") is not None]
    if not valid_chunks:
        return chunks

    conn = get_db()
    cursor = conn.cursor()

    # Load existing speaker profiles from DB
    cursor.execute("SELECT id, name, display_label, voice_embedding FROM speakers")
    existing_speakers = cursor.fetchall()
    
    known_speakers: List[Dict[str, Any]] = []
    for sp in existing_speakers:
        if sp["voice_embedding"]:
            emb = np.array(json.loads(sp["voice_embedding"]))
            known_speakers.append({
                "id": sp["id"],
                "name": sp["name"],
                "display_label": sp["display_label"],
                "embedding": emb
            })

    embeddings = np.array([ch["voice_embedding"] for ch in valid_chunks])

    # If only 1 chunk or all embeddings identical
    if len(valid_chunks) == 1:
        cluster_labels = [0]
    else:
        # Agglomerative clustering with cosine distance
        n_clusters = max(1, min(len(valid_chunks), 5))
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.35,  # 1 - 0.65 similarity
                metric="cosine",
                linkage="average"
            )
            cluster_labels = clustering.fit_predict(embeddings)
        except Exception:
            cluster_labels = [0] * len(valid_chunks)

    # Assign each cluster to an existing or new speaker profile
    unique_clusters = list(set(cluster_labels))
    cluster_to_speaker_id: Dict[int, int] = {}

    for c_id in unique_clusters:
        cluster_mask = (cluster_labels == c_id)
        cluster_mean_emb = np.mean(embeddings[cluster_mask], axis=0)
        cluster_mean_emb = cluster_mean_emb / (np.linalg.norm(cluster_mean_emb) + 1e-8)

        matched_speaker_id = None
        best_sim = -1.0

        for sp in known_speakers:
            sim = float(cosine_similarity([cluster_mean_emb], [sp["embedding"]])[0][0])
            if sim > best_sim:
                best_sim = sim
                if sim >= similarity_threshold:
                    matched_speaker_id = sp["id"]

        if matched_speaker_id is None:
            # Create a new speaker profile in DB
            new_label = f"Speaker {len(known_speakers) + len(cluster_to_speaker_id) + 1}"
            cursor.execute("""
                INSERT INTO speakers (name, display_label, voice_embedding)
                VALUES (?, ?, ?)
            """, (new_label, new_label, json.dumps(cluster_mean_emb.tolist())))
            new_id = cursor.lastrowid
            cluster_to_speaker_id[c_id] = new_id
            
            known_speakers.append({
                "id": new_id,
                "name": new_label,
                "display_label": new_label,
                "embedding": cluster_mean_emb
            })
        else:
            cluster_to_speaker_id[c_id] = matched_speaker_id

    conn.commit()
    conn.close()

    # Tag each chunk with assigned speaker ID
    for idx, ch in enumerate(valid_chunks):
        c_label = cluster_labels[idx]
        ch["speaker_id"] = cluster_to_speaker_id[c_label]

    return chunks
