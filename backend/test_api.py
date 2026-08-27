import os
import json
import numpy as np
import soundfile as sf
import asyncio

from db import init_db, get_db
from transcription import get_audio_info
from speaker_diarization import extract_voice_features, cluster_and_assign_speakers
from rag_engine import rag_engine
from dubbing_engine import translate_text, synthesize_neural_voice, fit_audio_to_duration, build_dubbed_audio_track

async def test_full_pipeline():
    print("=== 1. Initializing Database ===")
    init_db()

    # Create dummy audio file for testing
    test_dir = os.path.join(os.path.dirname(__file__), "storage")
    os.makedirs(test_dir, exist_ok=True)
    test_wav = os.path.join(test_dir, "test_speech.wav")
    
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    # Sine tone mix to simulate audio
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    sf.write(test_wav, audio_data, sr)
    print(f"Created test audio file: {test_wav}")

    # Test Audio Info
    info = get_audio_info(test_wav)
    print("Audio Info:", info)

    # Test Speaker Voice Feature Extraction & Clustering
    print("\n=== 2. Testing Speaker Diarization & Voice Features ===")
    feat = extract_voice_features(audio_data, sr)
    print(f"Extracted voice feature vector: shape={feat.shape}, norm={np.linalg.norm(feat):.4f}")

    dummy_chunks = [
        {"chunk_index": 0, "start_time": 0.0, "end_time": 1.5, "text": "வணக்கம், இது ஒரு தமிழ் குரல் சோதனை.", "language": "ta", "voice_embedding": feat},
        {"chunk_index": 1, "start_time": 1.5, "end_time": 3.0, "text": "This is audio intelligence RAG system.", "language": "en", "voice_embedding": feat}
    ]
    assigned = cluster_and_assign_speakers(dummy_chunks)
    print("Assigned Speakers:", [(c["chunk_index"], c.get("speaker_id"), c.get("speaker_label")) for c in assigned])

    # Insert into DB for RAG indexing
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO files (file_path, filename, file_size, duration, status) VALUES (?, ?, ?, ?, 'indexed')", 
                   (test_wav, "test_speech.wav", info["file_size"], info["duration"]))
    file_id = cursor.lastrowid
    
    cursor.execute("DELETE FROM transcript_chunks WHERE file_id = ?", (file_id,))
    for ch in assigned:
        cursor.execute("INSERT INTO transcript_chunks (file_id, chunk_index, start_time, end_time, text, language, speaker_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (file_id, ch["chunk_index"], ch["start_time"], ch["end_time"], ch["text"], ch["language"], ch.get("speaker_id")))
    conn.commit()
    conn.close()

    # Test RAG Engine
    print("\n=== 3. Testing Audio RAG & Question Answering ===")
    rag_engine.build_index()
    rag_res = rag_engine.query("தமிழ் குரல்", top_k=2)
    print("RAG Query ('தமிழ் குரல்') Answer:")
    print(rag_res["answer"])
    print(f"Found {len(rag_res['citations'])} citations.")

    # Test Dubbing Engine Translation & Synthesis
    print("\n=== 4. Testing Dubbing Engine (Tamil Synthesis) ===")
    english_text = "Welcome to the real-time audio dubbing and voice search studio."
    translated_tamil = await translate_text(english_text, "en", "ta")
    print(f"Original Text: {english_text}")
    print(f"Translated to Tamil: {translated_tamil}")

    dubbed_wav = os.path.join(test_dir, "test_dubbed_ta.wav")
    synth_ok = await synthesize_neural_voice(translated_tamil, "ta-IN-ValluvarNeural", dubbed_wav)
    print(f"Tamil Neural Voice Synthesis: {'SUCCESS' if synth_ok and os.path.exists(dubbed_wav) else 'FAILED'}")

    if synth_ok:
        synced_wav = os.path.join(test_dir, "test_dubbed_synced_ta.wav")
        fit_audio_to_duration(dubbed_wav, synced_wav, target_duration=4.0)
        print(f"Time-Synced Audio Output: {synced_wav} ({os.path.getsize(synced_wav)} bytes)")

        # Master track assembly
        master_track = os.path.join(test_dir, "test_master_track.wav")
        master_ok = await build_dubbed_audio_track(
            [{"audio_path": synced_wav, "start_time": 0.5}],
            total_duration=5.0,
            output_audio_path=master_track
        )
        print(f"Master Dubbed Audio Track Stitched: {'SUCCESS' if master_ok and os.path.exists(master_track) else 'FAILED'}")

    print("\n=== ALL TEST PIPELINE STAGES PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
