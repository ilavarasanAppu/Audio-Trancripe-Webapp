"""
=============================================================================
VaniScript AI - SQLite Database Layer
=============================================================================
This module provides lightweight, function-based SQLite database management.
Stores files, transcript chunks, speaker profiles, dubbing projects, and settings.

Functions:
  - get_db(): Returns a connection with sqlite3.Row factory.
  - init_db(): Creates all database tables if they don't exist.
  - get_setting(key, default): Fetches a key-value setting.
  - set_setting(key, value): Sets or updates a key-value setting.
=============================================================================
"""

import sqlite3
import os
import json
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_rag.db")

def get_db() -> sqlite3.Connection:
    """
    Returns an active SQLite connection with row dictionary mapping enabled.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes SQLite tables for settings, files, transcript chunks, speakers, and dubbing.
    """
    conn = get_db()
    cursor = conn.cursor()

    # 1. Settings Table (API keys, preferences)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    # 2. Media Files Table (Audio & Video indexed files)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE,
        filename TEXT,
        file_size INTEGER,
        duration REAL,
        sample_rate INTEGER,
        channels INTEGER,
        file_format TEXT,
        status TEXT DEFAULT 'pending',
        language TEXT DEFAULT 'unknown',
        full_transcript TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 3. Transcript Chunks Table (Timestamped sentences with speaker IDs)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcript_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER,
        chunk_index INTEGER,
        start_time REAL,
        end_time REAL,
        text TEXT,
        language TEXT,
        speaker_id INTEGER,
        voice_embedding TEXT,
        text_embedding TEXT,
        FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE,
        FOREIGN KEY (speaker_id) REFERENCES speakers (id) ON DELETE SET NULL
    )
    """)

    # 4. Speaker Profiles Table (Voice embeddings and human assigned names)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS speakers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        display_label TEXT,
        voice_embedding TEXT,
        gender TEXT DEFAULT 'unknown',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 5. Dubbing Projects Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dubbing_projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        source_file_path TEXT,
        source_lang TEXT,
        target_lang TEXT,
        engine TEXT DEFAULT 'neural',
        status TEXT DEFAULT 'created',
        result_audio_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 6. Dubbing Segments Table (Timeline slices with translated text & audio paths)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dubbing_segments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        segment_index INTEGER,
        start_time REAL,
        end_time REAL,
        original_text TEXT,
        translated_text TEXT,
        voice_name TEXT,
        audio_path TEXT,
        pitch_shift REAL DEFAULT 0.0,
        speed_rate REAL DEFAULT 1.0,
        FOREIGN KEY (project_id) REFERENCES dubbing_projects (id) ON DELETE CASCADE
    )
    """)

    # Populate default settings if empty
    default_settings = {
        "whisper_model_size": "base",
        "default_target_lang": "ta",
        "sarvam_api_key": "",
        "default_asr_engine": "faster_whisper",
        "default_tts_engine": "edge_neural",
        "default_translation_engine": "google_deep"
    }
    for k, v in default_settings.items():
        cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))

    conn.commit()
    conn.close()

def get_setting(key: str, default: str = "") -> str:
    """Fetches a setting value from SQLite."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row["value"] if row and row["value"] is not None else default

def set_setting(key: str, value: str):
    """Inserts or updates a setting in SQLite."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO settings (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
    """, (key, value))
    conn.commit()
    conn.close()
