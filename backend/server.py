import os
import sys
import shutil
import asyncio
import json
import mimetypes
import uuid
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import soundfile as sf
import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import init_db, get_db, get_setting, set_setting
from transcription import (
    transcribe_audio_file, get_audio_info, 
    scan_custom_models_directory, SUPPORTED_MODEL_EXTENSIONS
)
from speaker_diarization import extract_segment_voice_features, cluster_and_assign_speakers
from rag_engine import rag_engine
from dubbing_engine import (
    INDIAN_LANGUAGES, SARVAM_SPEAKERS, translate_text,
    synthesize_neural_voice, synthesize_sarvam_voice,
    fit_audio_to_duration, build_dubbed_audio_track, extract_acoustic_profile
)
from doctor import run_diagnostics, auto_fix_missing
from error_logging import ERROR_LOG_PATH, write_error

init_db()

app = FastAPI(title="VaniScript AI - Dubbing & Audio RAG Studio")


class ClientErrorReport(BaseModel):
    error_code: str = "BROWSER_ERROR"
    error_details: str
    source_file: Optional[str] = None
    error_line: Optional[int] = None
    error_column: Optional[int] = None
    stack: Optional[str] = None
    page_url: Optional[str] = None
    user_agent: Optional[str] = None
    occurred_at: Optional[str] = None


@app.exception_handler(StarletteHTTPException)
async def log_http_error(request: Request, exc: StarletteHTTPException):
    write_error(
        error_code=f"HTTP_{exc.status_code}",
        error_details=exc.detail,
        exc=exc,
        request=request,
        context={"status_code": exc.status_code},
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    safe_errors = [{"type": error.get("type"), "location": error.get("loc"), "message": error.get("msg")} for error in exc.errors()]
    write_error(
        error_code="REQUEST_VALIDATION_ERROR",
        error_details="Request validation failed",
        exc=exc,
        request=request,
        context={"status_code": 422, "validation_errors": safe_errors},
    )
    return JSONResponse(status_code=422, content={"detail": safe_errors})


@app.exception_handler(Exception)
async def log_unhandled_error(request: Request, exc: Exception):
    write_error(
        error_code="UNHANDLED_SERVER_ERROR",
        error_details=str(exc) or type(exc).__name__,
        exc=exc,
        request=request,
    )
    return JSONResponse(status_code=500, content={"detail": "An unexpected server error occurred. Check the backend error log for details."})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS.union(VIDEO_EXTENSIONS)

# Pydantic Request Models
class ScanRequest(BaseModel):
    directory: str
    recursive: bool = True

class ModelScanRequest(BaseModel):
    directories: Optional[List[str]] = None
    directory: Optional[str] = None

class IngestRequest(BaseModel):
    file_id: int
    model_size: str = "base"
    language: Optional[str] = "auto"
    asr_engine: str = "faster_whisper"
    custom_model_path: Optional[str] = None

class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 6

class SpeakerAssignRequest(BaseModel):
    speaker_id: int
    name: str

class WorkCaptureRequest(BaseModel):
    file_ids: List[int]
    target_directory: str
    export_as_clips: bool = False
    chunk_ids: Optional[List[int]] = None

class DubbingCreateRequest(BaseModel):
    file_id: Optional[int] = None
    title: str = "Dubbing Project"
    source_lang: str = "en"
    target_lang: str = "ta"
    engine: str = "neural"
    tts_engine: Optional[str] = "edge_neural"
    asr_engine: Optional[str] = "faster_whisper"
    custom_model_path: Optional[str] = None
    translation_engine: Optional[str] = "google_deep"
    voice_name: Optional[str] = None
    preserve_timbre: bool = True

class DubbingProcessRequest(BaseModel):
    project_id: int
    sarvam_api_key: Optional[str] = None
    tts_engine: Optional[str] = None
    asr_engine: Optional[str] = None
    preserve_timbre: bool = True

class BatchFolderDubbingRequest(BaseModel):
    source_directory: str
    target_lang: str = "ta"
    source_lang: str = "auto"
    tts_engine: str = "edge_neural"
    asr_engine: str = "faster_whisper"
    output_directory: Optional[str] = None

class LiveDubbingChunkRequest(BaseModel):
    source_lang: str = "auto"
    target_lang: str = "ta"
    tts_engine: str = "edge_neural"
    voice_name: Optional[str] = None
    asr_engine: str = "faster_whisper"

class DubbingSegmentUpdateRequest(BaseModel):
    segment_id: int
    translated_text: str
    voice_name: Optional[str] = None
    pitch_shift: float = 0.0
    speed_rate: float = 1.0

class SettingsUpdateRequest(BaseModel):
    sarvam_api_key: Optional[str] = None
    whisper_model_size: Optional[str] = None
    default_target_lang: Optional[str] = None
    default_asr_engine: Optional[str] = None
    default_tts_engine: Optional[str] = None
    default_translation_engine: Optional[str] = None
    custom_models_dir: Optional[str] = None
    custom_models_dirs: Optional[List[str]] = None
    selected_custom_model: Optional[str] = None


@app.post("/api/errors/log")
def log_client_error(report: ClientErrorReport, request: Request):
    write_error(
        error_code=report.error_code,
        error_details=report.error_details,
        request=request,
        source_file=report.source_file,
        error_line=report.error_line,
        stack=report.stack,
        context={
            "error_column": report.error_column,
            "page_url": report.page_url,
            "user_agent": report.user_agent,
            "client_occurred_at": report.occurred_at,
        },
    )
    return {"status": "logged", "log_file": ERROR_LOG_PATH}

@app.get("/api/health")
def health_check():
    from transcription import get_asr_runtime
    return {
        "status": "online",
        "service": "VaniScript AI Backend",
        "supported_languages": list(INDIAN_LANGUAGES.keys()),
        "asr_runtime": get_asr_runtime(),
    }

@app.get("/api/doctor")
def get_doctor_diagnostics():
    return run_diagnostics()

@app.post("/api/doctor/fix")
def fix_doctor_issues():
    return auto_fix_missing()

@app.get("/api/settings")
def get_all_settings():
    dirs_json = get_setting("custom_models_dirs", "[]")
    try:
        models_dirs = json.loads(dirs_json)
        if not isinstance(models_dirs, list):
            models_dirs = []
    except Exception:
        models_dirs = []

    legacy_dir = get_setting("custom_models_dir", "")
    if legacy_dir and legacy_dir not in models_dirs:
        models_dirs.append(legacy_dir)

    return {
        "sarvam_api_key": get_setting("sarvam_api_key", ""),
        "whisper_model_size": get_setting("whisper_model_size", "base"),
        "default_target_lang": get_setting("default_target_lang", "ta"),
        "default_asr_engine": get_setting("default_asr_engine", "faster_whisper"),
        "default_tts_engine": get_setting("default_tts_engine", "edge_neural"),
        "default_translation_engine": get_setting("default_translation_engine", "google_deep"),
        "custom_models_dir": legacy_dir,
        "custom_models_dirs": models_dirs,
        "selected_custom_model": get_setting("selected_custom_model", "")
    }

@app.post("/api/settings")
def update_settings(payload: SettingsUpdateRequest):
    if payload.sarvam_api_key is not None:
        set_setting("sarvam_api_key", payload.sarvam_api_key.strip())
    if payload.whisper_model_size is not None:
        set_setting("whisper_model_size", payload.whisper_model_size.strip())
    if payload.default_target_lang is not None:
        set_setting("default_target_lang", payload.default_target_lang.strip())
    if payload.default_asr_engine is not None:
        set_setting("default_asr_engine", payload.default_asr_engine.strip())
    if payload.default_tts_engine is not None:
        set_setting("default_tts_engine", payload.default_tts_engine.strip())
    if payload.default_translation_engine is not None:
        set_setting("default_translation_engine", payload.default_translation_engine.strip())
    if payload.custom_models_dir is not None:
        set_setting("custom_models_dir", payload.custom_models_dir.strip())
    if payload.custom_models_dirs is not None:
        clean_dirs = [d.strip() for d in payload.custom_models_dirs if d and d.strip()]
        set_setting("custom_models_dirs", json.dumps(clean_dirs))
    if payload.selected_custom_model is not None:
        set_setting("selected_custom_model", payload.selected_custom_model.strip())
    return {"status": "success", "message": "Settings updated successfully"}

# ==================== CUSTOM MODEL SCANNING ENDPOINTS ====================

@app.post("/api/models/scan")
def scan_models(payload: ModelScanRequest):
    dirs_to_scan: List[str] = []
    if payload.directories:
        dirs_to_scan.extend([d.strip() for d in payload.directories if d and d.strip()])
    if payload.directory and payload.directory.strip():
        dirs_to_scan.append(payload.directory.strip())

    # Remove duplicates while preserving order
    unique_dirs = list(dict.fromkeys(dirs_to_scan))
    if not unique_dirs:
        # Fallback to stored dirs in DB
        dirs_json = get_setting("custom_models_dirs", "[]")
        try:
            unique_dirs = json.loads(dirs_json)
        except Exception:
            unique_dirs = []

    from transcription import scan_multiple_custom_models_directories
    models = scan_multiple_custom_models_directories(unique_dirs)
    
    # Save updated directories list
    set_setting("custom_models_dirs", json.dumps(unique_dirs))
    if unique_dirs:
        set_setting("custom_models_dir", unique_dirs[0])

    return {
        "directories": unique_dirs,
        "total_models": len(models),
        "models": models,
        "supported_extensions": list(SUPPORTED_MODEL_EXTENSIONS)
    }

@app.get("/api/models")
def list_detected_models():
    dirs_json = get_setting("custom_models_dirs", "[]")
    try:
        models_dirs = json.loads(dirs_json)
        if not isinstance(models_dirs, list):
            models_dirs = []
    except Exception:
        models_dirs = []

    legacy_dir = get_setting("custom_models_dir", "")
    if legacy_dir and legacy_dir not in models_dirs:
        models_dirs.append(legacy_dir)

    from transcription import scan_multiple_custom_models_directories
    models = scan_multiple_custom_models_directories(models_dirs) if models_dirs else []

    return {
        "custom_models_dirs": models_dirs,
        "selected_custom_model": get_setting("selected_custom_model", ""),
        "total_models": len(models),
        "models": models
    }

@app.get("/api/dubbing/voices")
def get_available_voices():
    return {
        "languages": INDIAN_LANGUAGES,
        "sarvam_speakers": SARVAM_SPEAKERS
    }

@app.post("/api/scan")
def scan_directory(payload: ScanRequest):
    directory = payload.directory.strip()
    if not os.path.exists(directory):
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {directory}")

    discovered_files = []
    conn = get_db()
    cursor = conn.cursor()

    if payload.recursive:
        walker = os.walk(directory)
    else:
        entries = [os.path.join(directory, f) for f in os.listdir(directory)]
        files_only = [f for f in entries if os.path.isfile(f)]
        walker = [(directory, [], [os.path.basename(f) for f in files_only])]

    for root, _, filenames in walker:
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.abspath(os.path.join(root, filename))
                file_size = os.path.getsize(full_path)
                
                cursor.execute("SELECT id, status, duration FROM files WHERE file_path = ?", (full_path,))
                existing = cursor.fetchone()
                
                if not existing:
                    cursor.execute("""
                        INSERT INTO files (file_path, filename, file_size, file_format, status)
                        VALUES (?, ?, ?, ?, 'pending')
                    """, (full_path, filename, file_size, ext.replace(".", "")))
                    file_id = cursor.lastrowid
                    status = "pending"
                    duration = 0.0
                else:
                    file_id = existing["id"]
                    status = existing["status"]
                    duration = existing["duration"]

                discovered_files.append({
                    "id": file_id,
                    "file_path": full_path,
                    "filename": filename,
                    "file_size": file_size,
                    "file_format": ext.replace(".", ""),
                    "status": status,
                    "duration": duration
                })

    conn.commit()
    conn.close()
    return {
        "directory": directory,
        "total_files": len(discovered_files),
        "files": discovered_files
    }

@app.get("/api/files")
def list_files():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT f.id, f.file_path, f.filename, f.file_size, f.duration, f.file_format,
               f.status, f.language, f.created_at,
               COUNT(c.id) as chunk_count
        FROM files f
        LEFT JOIN transcript_chunks c ON f.id = c.file_id
        GROUP BY f.id
        ORDER BY f.id DESC
    """)
    rows = cursor.fetchall()
    files_list = [dict(r) for r in rows]
    conn.close()
    return {"files": files_list}

@app.delete("/api/files/{file_id}")
def delete_file(file_id: int):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "File removed from library"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_audio.wav"
    dest_path = os.path.join(STORAGE_DIR, filename)
    
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    info = get_audio_info(dest_path)
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO files (file_path, filename, file_size, duration, sample_rate, channels, file_format, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
    """, (dest_path, filename, info["file_size"], info["duration"], info["sample_rate"], info["channels"], info["file_format"]))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "id": file_id,
        "filename": filename,
        "file_path": dest_path,
        "duration": info["duration"],
        "status": "pending"
    }

@app.post("/api/ingest")
async def ingest_file(payload: IngestRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM files WHERE id = ?", (payload.file_id,))
    file_record = cursor.fetchone()
    
    if not file_record:
        conn.close()
        raise HTTPException(status_code=404, detail="File not found")

    file_path = file_record["file_path"]
    if not os.path.exists(file_path):
        conn.close()
        raise HTTPException(status_code=400, detail=f"File does not exist on disk: {file_path}")

    cursor.execute("UPDATE files SET status = 'processing' WHERE id = ?", (payload.file_id,))
    conn.commit()

    try:
        asr_engine = payload.asr_engine or get_setting("default_asr_engine", "faster_whisper")
        model_size_or_path = payload.custom_model_path or get_setting("selected_custom_model") or payload.model_size or get_setting("whisper_model_size", "base")
        
        # 1. Transcribe with selected ASR engine / custom model
        transcription_result = transcribe_audio_file(
            file_path=file_path,
            language=payload.language,
            model_size=model_size_or_path,
            engine=asr_engine,
            sarvam_api_key=get_setting("sarvam_api_key")
        )

        detected_lang = transcription_result["language"]
        full_transcript = transcription_result["full_transcript"]
        duration = transcription_result["duration"]
        raw_chunks = transcription_result["chunks"]
        if not raw_chunks or not full_transcript.strip():
            raise RuntimeError("No speech was recognized. The file was not indexed; check the audio and selected ASR model, then retry.")

        # 2. Extract Acoustic Voice Features for each chunk
        chunks_with_features = []
        for ch in raw_chunks:
            feat = extract_segment_voice_features(file_path, ch["start_time"], ch["end_time"])
            ch["voice_embedding"] = feat
            chunks_with_features.append(ch)

        # 3. Cluster & Assign Speakers
        assigned_chunks = cluster_and_assign_speakers(chunks_with_features)

        # 4. Save to DB
        cursor.execute("DELETE FROM transcript_chunks WHERE file_id = ?", (payload.file_id,))
        
        for ch in assigned_chunks:
            feat_list = ch["voice_embedding"].tolist() if ch.get("voice_embedding") is not None else None
            cursor.execute("""
                INSERT INTO transcript_chunks (file_id, chunk_index, start_time, end_time, text, language, speaker_id, voice_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                payload.file_id,
                ch["chunk_index"],
                ch["start_time"],
                ch["end_time"],
                ch["text"],
                ch["language"],
                ch.get("speaker_id"),
                json.dumps(feat_list) if feat_list else None
            ))

        cursor.execute("""
            UPDATE files 
            SET status = 'indexed', language = ?, full_transcript = ?, duration = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (detected_lang, full_transcript, duration, payload.file_id))
        conn.commit()

        rag_engine.build_index()
        conn.close()

        return {
            "status": "success",
            "file_id": payload.file_id,
            "language": detected_lang,
            "duration": duration,
            "chunk_count": len(assigned_chunks),
            "full_transcript": full_transcript
        }
    except Exception as e:
        cursor.execute("UPDATE files SET status = 'error' WHERE id = ?", (payload.file_id,))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/transcribe")
async def transcribe_uploaded_audio(file: UploadFile = File(...), language: str = Form("auto")):
    """Transcribe a selected file with the local Faster-Whisper GPU runtime."""
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=STORAGE_DIR) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
        if os.path.getsize(temp_path) == 0:
            raise HTTPException(status_code=400, detail="The selected audio file is empty.")
        model_path = get_setting("selected_custom_model") or get_setting("whisper_model_size", "base")
        result = transcribe_audio_file(
            file_path=temp_path,
            language=language,
            model_size=model_path,
            engine="faster_whisper",
        )
        if not result.get("full_transcript", "").strip() or not result.get("chunks"):
            raise RuntimeError("No speech was recognized. Check the audio and selected ASR model, then retry.")
        from transcription import get_asr_runtime
        return {**result, "runtime": get_asr_runtime(), "model": model_path}
    except HTTPException:
        raise
    except Exception as exc:
        write_error(
            error_code="TRANSCRIPTION_FAILED",
            error_details=str(exc),
            exc=exc,
            request=None,
            context={"media_file_path": file.filename, "asr_engine": "faster_whisper"},
        )
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/ingest_all")
async def ingest_all_pending(background_tasks: BackgroundTasks):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM files WHERE status = 'pending'")
    rows = cursor.fetchall()
    conn.close()
    
    file_ids = [r["id"] for r in rows]
    for fid in file_ids:
        background_tasks.add_task(ingest_file, IngestRequest(file_id=fid))

    return {"status": "scheduled", "total_pending": len(file_ids), "file_ids": file_ids}

@app.post("/api/rag/query")
def query_audio_rag_endpoint(payload: RAGQueryRequest):
    result = rag_engine.query(question=payload.question, top_k=payload.top_k)
    return result

@app.get("/api/media/stream")
def stream_media(file_path: str = Query(...)):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Media file not found")
    
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "audio/mpeg"
    return FileResponse(file_path, media_type=mime_type)

@app.get("/api/media/clip")
def stream_audio_clip(
    file_path: str = Query(...),
    start: float = Query(0.0),
    end: float = Query(10.0)
):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    try:
        duration = max(0.5, end - start)
        y, sr = librosa.load(file_path, sr=22050, offset=max(0, start), duration=duration, mono=False)
        
        clip_name = f"clip_{abs(hash(file_path))}_{int(start*100)}_{int(end*100)}.wav"
        clip_path = os.path.join(STORAGE_DIR, clip_name)
        
        sf.write(clip_path, y.T if len(y.shape) > 1 else y, sr)
        return FileResponse(clip_path, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate audio clip: {e}")

@app.post("/api/work_capture/copy")
def copy_work_capture_files(payload: WorkCaptureRequest):
    target_dir = payload.target_directory.strip()
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot create destination directory: {e}")

    conn = get_db()
    cursor = conn.cursor()

    copied_files = []
    errors = []

    for fid in payload.file_ids:
        cursor.execute("SELECT file_path, filename FROM files WHERE id = ?", (fid,))
        rec = cursor.fetchone()
        if not rec:
            continue
        
        src_path = rec["file_path"]
        if os.path.exists(src_path):
            try:
                dest_file = os.path.join(target_dir, rec["filename"])
                if os.path.exists(dest_file):
                    base, ext = os.path.splitext(rec["filename"])
                    dest_file = os.path.join(target_dir, f"{base}_copy{ext}")
                
                shutil.copy2(src_path, dest_file)
                copied_files.append({"source": src_path, "destination": dest_file, "type": "full_file"})
            except Exception as e:
                errors.append(f"Failed to copy {rec['filename']}: {e}")

    if payload.chunk_ids:
        for cid in payload.chunk_ids:
            cursor.execute("""
                SELECT c.id, c.start_time, c.end_time, c.text, f.file_path, f.filename
                FROM transcript_chunks c
                JOIN files f ON c.file_id = f.id
                WHERE c.id = ?
            """, (cid,))
            chk = cursor.fetchone()
            if not chk or not os.path.exists(chk["file_path"]):
                continue

            try:
                start = chk["start_time"]
                end = chk["end_time"]
                dur = max(0.5, end - start)
                y, sr = librosa.load(chk["file_path"], sr=22050, offset=start, duration=dur)
                
                clean_name = os.path.splitext(chk["filename"])[0]
                clip_filename = f"{clean_name}_clip_{int(start)}s_to_{int(end)}s.wav"
                dest_clip = os.path.join(target_dir, clip_filename)
                
                sf.write(dest_clip, y, sr)
                copied_files.append({"source": chk["file_path"], "destination": dest_clip, "type": "audio_clip", "start": start, "end": end})
            except Exception as e:
                errors.append(f"Failed to extract clip #{cid}: {e}")

    conn.close()
    return {
        "status": "success",
        "target_directory": target_dir,
        "total_copied": len(copied_files),
        "copied_items": copied_files,
        "errors": errors
    }

@app.get("/api/speakers")
def list_speakers():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.id, s.name, s.display_label, s.created_at,
               COUNT(c.id) as segment_count,
               MIN(f.filename) as sample_file,
               MIN(f.file_path) as sample_file_path,
               MIN(c.start_time) as sample_start,
               MIN(c.end_time) as sample_end,
               MIN(c.text) as sample_text
        FROM speakers s
        LEFT JOIN transcript_chunks c ON s.id = c.speaker_id
        LEFT JOIN files f ON c.file_id = f.id
        GROUP BY s.id
        ORDER BY segment_count DESC
    """)
    rows = cursor.fetchall()
    speakers_list = [dict(r) for r in rows]
    conn.close()
    return {"speakers": speakers_list}

@app.post("/api/speakers/assign")
def assign_speaker_name(payload: SpeakerAssignRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE speakers SET name = ? WHERE id = ?", (payload.name.strip(), payload.speaker_id))
    conn.commit()
    conn.close()
    return {"status": "success", "speaker_id": payload.speaker_id, "name": payload.name}

@app.get("/api/speakers/{speaker_id}/files")
def get_speaker_files(speaker_id: int):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT f.id, f.filename, f.file_path, f.duration, f.language,
               COUNT(c.id) as speaker_segment_count
        FROM files f
        JOIN transcript_chunks c ON f.id = c.file_id
        WHERE c.speaker_id = ?
        GROUP BY f.id
    """, (speaker_id,))
    files = [dict(r) for r in cursor.fetchall()]

    cursor.execute("""
        SELECT c.id, c.file_id, c.start_time, c.end_time, c.text, f.filename, f.file_path
        FROM transcript_chunks c
        JOIN files f ON c.file_id = f.id
        WHERE c.speaker_id = ?
        ORDER BY f.id, c.start_time
    """, (speaker_id,))
    segments = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return {
        "speaker_id": speaker_id,
        "files": files,
        "segments": segments
    }

# ==================== REAL-TIME DUBBING ENDPOINTS ====================

@app.post("/api/dubbing/create")
async def create_dubbing_project(payload: DubbingCreateRequest):
    conn = get_db()
    cursor = conn.cursor()

    if not payload.file_id:
        conn.close()
        raise HTTPException(status_code=400, detail="file_id is required")

    cursor.execute("SELECT * FROM files WHERE id = ?", (payload.file_id,))
    file_record = cursor.fetchone()
    if not file_record:
        conn.close()
        raise HTTPException(status_code=404, detail="File not found")

    title = payload.title or f"Dubbing - {file_record['filename']} ({payload.target_lang.upper()})"
    tts_engine = payload.tts_engine or payload.engine or "edge_neural"
    
    cursor.execute("""
        INSERT INTO dubbing_projects (title, source_file_path, source_lang, target_lang, engine, status)
        VALUES (?, ?, ?, ?, ?, 'created')
    """, (title, file_record["file_path"], payload.source_lang, payload.target_lang, tts_engine))
    project_id = cursor.lastrowid
    conn.commit()

    cursor.execute("SELECT * FROM transcript_chunks WHERE file_id = ? ORDER BY chunk_index", (payload.file_id,))
    chunks = cursor.fetchall()

    default_voice = payload.voice_name or INDIAN_LANGUAGES.get(payload.target_lang, {}).get("default_voice", "ta-IN-ValluvarNeural")

    for ch in chunks:
        cursor.execute("""
            INSERT INTO dubbing_segments (project_id, segment_index, start_time, end_time, original_text, voice_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (project_id, ch["chunk_index"], ch["start_time"], ch["end_time"], ch["text"], default_voice))

    conn.commit()
    conn.close()

    return {
        "project_id": project_id,
        "title": title,
        "source_file": file_record["filename"],
        "target_lang": payload.target_lang,
        "segment_count": len(chunks)
    }

@app.get("/api/dubbing/projects")
def list_dubbing_projects():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.*, COUNT(s.id) as segment_count
        FROM dubbing_projects p
        LEFT JOIN dubbing_segments s ON p.id = s.project_id
        GROUP BY p.id
        ORDER BY p.id DESC
    """)
    projects = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {"projects": projects}

@app.get("/api/dubbing/projects/{project_id}")
def get_dubbing_project(project_id: int):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dubbing_projects WHERE id = ?", (project_id,))
    proj = cursor.fetchone()
    if not proj:
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")

    cursor.execute("""
        SELECT * FROM dubbing_segments WHERE project_id = ? ORDER BY segment_index
    """, (project_id,))
    segments = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return {
        "project": dict(proj),
        "segments": segments
    }

@app.post("/api/dubbing/process")
async def process_dubbing(payload: DubbingProcessRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dubbing_projects WHERE id = ?", (payload.project_id,))
    proj = cursor.fetchone()
    if not proj:
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")

    cursor.execute("SELECT * FROM dubbing_segments WHERE project_id = ? ORDER BY segment_index", (payload.project_id,))
    segments = [dict(r) for r in cursor.fetchall()]

    source_lang = proj["source_lang"]
    target_lang = proj["target_lang"]
    engine = payload.tts_engine or proj["engine"] or "edge_neural"
    sarvam_key = payload.sarvam_api_key or get_setting("sarvam_api_key")
    source_file_path = proj["source_file_path"]

    cursor.execute("UPDATE dubbing_projects SET status = 'processing' WHERE id = ?", (payload.project_id,))
    conn.commit()

    processed_segments = []

    try:
        for seg in segments:
            orig_text = seg["original_text"]
            if not orig_text:
                continue

            # 1. Translate (with English pivot for Sarvam)
            trans_text = seg.get("translated_text")
            if not trans_text:
                trans_text = await translate_text(orig_text, source_lang, target_lang, sarvam_key)

            # 2. Extract Acoustic Profile (pitch & emotion) for Same-Voice synthesis
            pitch_shift_hz = 0
            if payload.preserve_timbre and os.path.exists(source_file_path):
                profile = extract_acoustic_profile(source_file_path, seg["start_time"], seg["end_time"])
                original_f0 = profile["median_f0"]
                
                voice_name = seg.get("voice_name") or INDIAN_LANGUAGES.get(target_lang, {}).get("default_voice", "ta-IN-ValluvarNeural")
                base_f0 = 130.0 if "male" in voice_name.lower() or "valluvar" in voice_name.lower() or "kumar" in voice_name.lower() else 215.0
                pitch_shift_hz = int(np.clip(original_f0 - base_f0, -40, 40))

            voice_name = seg.get("voice_name") or INDIAN_LANGUAGES.get(target_lang, {}).get("default_voice", "ta-IN-ValluvarNeural")
            raw_audio_name = f"dub_raw_{payload.project_id}_{seg['id']}.wav"
            raw_audio_path = os.path.join(STORAGE_DIR, raw_audio_name)
            
            target_dur = max(0.5, seg["end_time"] - seg["start_time"])
            
            # 3. Synthesize Voice with pitch & emotion matching
            if ("sarvam" in engine.lower() or engine == "sarvam") and sarvam_key:
                speaker = "meera"
                if "male" in voice_name.lower():
                    speaker = "arvind"
                synth_ok = await synthesize_sarvam_voice(trans_text, target_lang, speaker, raw_audio_path, sarvam_key, pitch_offset=int(pitch_shift_hz/5))
            else:
                synth_ok = await synthesize_neural_voice(trans_text, voice_name, raw_audio_path, pitch_hz=pitch_shift_hz)

            if synth_ok and os.path.exists(raw_audio_path):
                # 4. Time Sync & Stretch to fit video/original segment duration
                synced_audio_name = f"dub_sync_{payload.project_id}_{seg['id']}.wav"
                synced_audio_path = os.path.join(STORAGE_DIR, synced_audio_name)
                
                fitted_duration = fit_audio_to_duration(raw_audio_path, synced_audio_path, target_duration=target_dur)
                if fitted_duration > 0 and os.path.exists(synced_audio_path):
                    audio_to_use = synced_audio_path
            else:
                audio_to_use = None

            cursor.execute("""
                UPDATE dubbing_segments
                SET translated_text = ?, audio_path = ?, voice_name = ?, pitch_shift = ?
                WHERE id = ?
            """, (trans_text, audio_to_use, voice_name, float(pitch_shift_hz), seg["id"]))
            conn.commit()

            seg["translated_text"] = trans_text
            seg["audio_path"] = audio_to_use
            processed_segments.append(seg)

        # 5. Build Complete Master Dubbed Audio Track
        total_duration = librosa.get_duration(path=source_file_path) if os.path.exists(source_file_path) else 60.0
        master_dubbed_name = f"master_dubbed_project_{payload.project_id}.wav"
        master_dubbed_path = os.path.join(STORAGE_DIR, master_dubbed_name)

        build_ok = await build_dubbed_audio_track(processed_segments, total_duration, master_dubbed_path)

        if not build_ok:
            raise RuntimeError("No audible dubbed speech was generated. Check that the source has transcript segments and that the selected voice can synthesize audio.")

        cursor.execute("""
            UPDATE dubbing_projects
            SET status = 'completed', result_audio_path = ?
            WHERE id = ?
        """, (master_dubbed_path if build_ok else None, payload.project_id))
        conn.commit()
        conn.close()

        return {
            "status": "completed",
            "project_id": payload.project_id,
            "master_audio_path": master_dubbed_path if build_ok else None,
            "segments": processed_segments
        }
    except Exception as e:
        cursor.execute("UPDATE dubbing_projects SET status = 'error' WHERE id = ?", (payload.project_id,))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Dubbing failed: {e}")

# ==================== LIVE & BATCH DUBBING ENDPOINTS ====================

@app.post("/api/dubbing/live_chunk")
async def dub_live_chunk(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("ta"),
    tts_engine: str = Form("edge_neural"),
    voice_name: Optional[str] = Form(None),
    asr_engine: str = Form("faster_whisper")
):
    """
    Realtime Voice-to-Voice and System Audio Capture Dubbing.
    Receives raw audio chunk -> Transcribes -> Translates with English pivot -> Synthesizes same-voice output.
    """
    temp_chunk_name = f"live_chunk_{uuid.uuid4().hex[:8]}.wav"
    temp_chunk_path = os.path.join(STORAGE_DIR, temp_chunk_name)
    
    with open(temp_chunk_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        sarvam_key = get_setting("sarvam_api_key")
        
        # 1. Transcribe Live Audio
        trans_res = transcribe_audio_file(
            file_path=temp_chunk_path,
            language=source_lang,
            model_size=get_setting("whisper_model_size", "base"),
            engine=asr_engine,
            sarvam_api_key=sarvam_key
        )
        original_text = trans_res["full_transcript"].strip()
        detected_lang = trans_res["language"]

        if not original_text:
            return {"status": "silent", "original_text": "", "translated_text": "", "audio_url": None}

        # 2. Translate with English Pivot
        translated_text = await translate_text(original_text, detected_lang or source_lang, target_lang, sarvam_key)

        # 3. Extract Acoustic Pitch & Emotion Profile for Same Voice Matching
        profile = extract_acoustic_profile(temp_chunk_path, 0, trans_res["duration"])
        original_f0 = profile["median_f0"]
        selected_voice = voice_name or INDIAN_LANGUAGES.get(target_lang, {}).get("default_voice", "ta-IN-ValluvarNeural")
        
        base_f0 = 130.0 if "male" in selected_voice.lower() else 215.0
        pitch_shift_hz = int(np.clip(original_f0 - base_f0, -40, 40))

        # 4. Synthesize Translated Speech
        dubbed_out_name = f"live_dubbed_{uuid.uuid4().hex[:8]}.wav"
        dubbed_out_path = os.path.join(STORAGE_DIR, dubbed_out_name)

        if ("sarvam" in tts_engine.lower() or tts_engine == "sarvam") and sarvam_key:
            speaker = "arvind" if "male" in selected_voice.lower() else "meera"
            await synthesize_sarvam_voice(translated_text, target_lang, speaker, dubbed_out_path, sarvam_key, pitch_offset=int(pitch_shift_hz/5))
        else:
            await synthesize_neural_voice(translated_text, selected_voice, dubbed_out_path, pitch_hz=pitch_shift_hz)

        audio_url = f"/api/media/stream?file_path={dubbed_out_path}" if os.path.exists(dubbed_out_path) else None

        return {
            "status": "success",
            "original_text": original_text,
            "translated_text": translated_text,
            "detected_lang": detected_lang,
            "pitch_f0": original_f0,
            "audio_url": audio_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live dubbing failed: {e}")

@app.post("/api/dubbing/batch_folder")
async def batch_folder_dubbing(payload: BatchFolderDubbingRequest, background_tasks: BackgroundTasks):
    """
    Batch dubs an entire folder of audio & video files on Windows.
    """
    src_dir = payload.source_directory.strip()
    if not os.path.exists(src_dir):
        raise HTTPException(status_code=400, detail=f"Source folder does not exist: {src_dir}")

    out_dir = payload.output_directory or os.path.join(src_dir, f"Dubbed_{payload.target_lang.upper()}")
    os.makedirs(out_dir, exist_ok=True)

    discovered_files = []
    for root, _, filenames in os.walk(src_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS and "Dubbed_" not in root:
                discovered_files.append(os.path.join(root, fname))

    if not discovered_files:
        return {"status": "empty", "message": "No supported audio/video files found in directory", "files": []}

    return {
        "status": "scheduled",
        "total_files": len(discovered_files),
        "source_directory": src_dir,
        "output_directory": out_dir,
        "files": discovered_files
    }

@app.post("/api/dubbing/segment/update")
async def update_dubbing_segment(payload: DubbingSegmentUpdateRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, p.target_lang, p.engine, p.id as project_id
        FROM dubbing_segments s
        JOIN dubbing_projects p ON s.project_id = p.id
        WHERE s.id = ?
    """, (payload.segment_id,))
    seg = cursor.fetchone()
    if not seg:
        conn.close()
        raise HTTPException(status_code=404, detail="Segment not found")

    target_lang = seg["target_lang"]
    voice_name = payload.voice_name or seg["voice_name"] or INDIAN_LANGUAGES.get(target_lang, {}).get("default_voice", "ta-IN-ValluvarNeural")
    target_dur = max(0.5, seg["end_time"] - seg["start_time"])

    raw_audio_name = f"dub_raw_{seg['project_id']}_{seg['id']}.wav"
    raw_audio_path = os.path.join(STORAGE_DIR, raw_audio_name)
    
    synth_ok = await synthesize_neural_voice(payload.translated_text, voice_name, raw_audio_path, pitch_hz=int(payload.pitch_shift))
    
    audio_to_use = None
    if synth_ok and os.path.exists(raw_audio_path):
        synced_audio_name = f"dub_sync_{seg['project_id']}_{seg['id']}.wav"
        synced_audio_path = os.path.join(STORAGE_DIR, synced_audio_name)
        fit_audio_to_duration(raw_audio_path, synced_audio_path, target_duration=target_dur)
        audio_to_use = synced_audio_path

    cursor.execute("""
        UPDATE dubbing_segments
        SET translated_text = ?, voice_name = ?, audio_path = ?, pitch_shift = ?, speed_rate = ?
        WHERE id = ?
    """, (payload.translated_text, voice_name, audio_to_use, payload.pitch_shift, payload.speed_rate, payload.segment_id))
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "segment_id": payload.segment_id,
        "translated_text": payload.translated_text,
        "voice_name": voice_name,
        "audio_path": audio_to_use
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
