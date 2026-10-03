"""
=============================================================================
VaniScript AI - Audio Transcription & Custom Model Loader Module (ASR)
=============================================================================
This module provides beginner-friendly, function-based audio transcription with
comprehensive support for:
  1. Standard Faster-Whisper models ('tiny', 'base', 'small', 'medium', 'large-v3')
  2. Custom Local Model Files & Folders:
     - .bin (CTranslate2, GGML, PyTorch weights)
     - .pth / .pt (PyTorch models)
     - .safetensors (HuggingFace SafeTensors)
     - .nemo (NVIDIA NeMo Indic ASR models)
     - .onnx (ONNX Runtime speech models)
     - CTranslate2 & HuggingFace custom model folders
  3. Sarvam AI Saaras v2 (Cloud Indic specialized speech recognition)

Functions:
  - scan_custom_models_directory(directory_path): Discovers all local AI models.
  - get_whisper_model(model_size_or_path): Loads local Whisper model or custom path.
  - transcribe_with_sarvam_saaras(...): Cloud Indic speech-to-text.
  - transcribe_with_faster_whisper(...): Local timestamped speech-to-text.
  - transcribe_audio_file(...): Unified entry point for all ASR engines.
=============================================================================
"""

import os
import sys
import ctypes
import httpx
import soundfile as sf
import librosa
from typing import Dict, Any, List, Optional
from faster_whisper import WhisperModel

# Ensure backend imports resolve properly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import get_setting

# Supported Custom Model Extensions
SUPPORTED_MODEL_EXTENSIONS = {".bin", ".pth", ".pt", ".safetensors", ".nemo", ".onnx"}

# Global cache to reuse loaded Whisper models in memory
_WHISPER_MODELS: Dict[str, WhisperModel] = {}


def get_asr_runtime() -> Dict[str, Any]:
    """Prefer the NVIDIA GPU when CTranslate2 has CUDA support available."""
    try:
        import ctranslate2
        device_count = int(ctranslate2.get_cuda_device_count())
        if device_count > 0:
            required_libraries = (
                ("cublas64_12.dll", "cudnn64_9.dll") if os.name == "nt"
                else ("libcublas.so.12", "libcudnn.so.9")
            )
            missing = []
            for library in required_libraries:
                try:
                    (ctypes.WinDLL if os.name == "nt" else ctypes.CDLL)(library)
                except OSError:
                    missing.append(library)
            if missing:
                return {
                    "device": "cpu", "compute_type": "int8", "gpu_available": True,
                    "gpu_ready": False, "gpu_count": device_count,
                    "gpu_error": f"CUDA device found, but required runtime libraries are missing: {', '.join(missing)}",
                }
            return {"device": "cuda", "compute_type": "float16", "gpu_available": True, "gpu_ready": True, "gpu_count": device_count}
        return {"device": "cpu", "compute_type": "int8", "gpu_available": False, "gpu_count": 0}
    except Exception as exc:
        return {"device": "cpu", "compute_type": "int8", "gpu_available": False, "gpu_count": 0, "gpu_error": str(exc)}

def scan_custom_models_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Scans a single folder for custom AI model files and folders.
    """
    if not directory_path or not os.path.exists(directory_path):
        return []
    return scan_multiple_custom_models_directories([directory_path])

def scan_multiple_custom_models_directories(directories: List[str]) -> List[Dict[str, Any]]:
    """
    Gathers and aggregates all AI models across MULTIPLE different folders on Windows.
    Supports: .bin, .pth, .pt, .safetensors, .nemo, .onnx, and CTranslate2/HuggingFace folders.
    
    Parameters:
      - directories: List of folder paths to search.
      
    Returns:
      - Consolidated list of discovered models across all folders with 'folder_origin'.
    """
    discovered_models: List[Dict[str, Any]] = []
    seen_paths = set()

    for directory_path in directories:
        directory_path = directory_path.strip() if isinstance(directory_path, str) else ""
        if not directory_path or not os.path.exists(directory_path):
            continue

        try:
            for root, dirs, files in os.walk(directory_path):
                # 1. Check if the directory itself is a CTranslate2 or HuggingFace model folder
                model_marker_files = {"model.bin", "model.safetensors", "config.json", "vocabulary.json", "vocabulary.txt"}
                found_markers = model_marker_files.intersection(set(f.lower() for f in files))
                
                if len(found_markers) >= 2 or "model.bin" in [f.lower() for f in files] or "model.safetensors" in [f.lower() for f in files]:
                    folder_path = os.path.abspath(root)
                    # The configured directory itself may already be the model
                    # folder (for example, a user points directly at a folder
                    # containing model.bin). Include it in the discovered list.
                    if folder_path not in seen_paths:
                        total_size = sum(os.path.getsize(os.path.join(root, f)) for f in files if os.path.isfile(os.path.join(root, f)))
                        folder_name = os.path.basename(folder_path)
                        
                        format_type = "CTranslate2 / HuggingFace Model Folder"
                        if "nemo" in folder_name.lower():
                            format_type = "NVIDIA NeMo ASR Folder"
                        elif any(f.endswith(".safetensors") for f in files):
                            format_type = "SafeTensors Model Folder"

                        discovered_models.append({
                            "name": folder_name,
                            "path": folder_path,
                            "folder_origin": os.path.abspath(directory_path),
                            "is_directory": True,
                            "format": format_type,
                            "extension": "folder",
                            "size_mb": round(total_size / (1024 * 1024), 2),
                            "model_type": "ASR / Speech Model",
                            "asr_compatible": "model.bin" in [f.lower() for f in files] and "config.json" in [f.lower() for f in files],
                        })
                        seen_paths.add(folder_path)

                # 2. Check individual model files
                for file_name in files:
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext in SUPPORTED_MODEL_EXTENSIONS:
                        file_path = os.path.abspath(os.path.join(root, file_name))
                        if file_path not in seen_paths:
                            file_size = os.path.getsize(file_path)
                            
                            format_name = "Binary Model (.bin)"
                            if ext in [".pth", ".pt"]:
                                format_name = "PyTorch Weights (.pth / .pt)"
                            elif ext == ".safetensors":
                                format_name = "SafeTensors (.safetensors)"
                            elif ext == ".nemo":
                                format_name = "NVIDIA NeMo (.nemo)"
                            elif ext == ".onnx":
                                format_name = "ONNX Runtime (.onnx)"

                            discovered_models.append({
                                "name": file_name,
                                "path": file_path,
                                "folder_origin": os.path.abspath(directory_path),
                                "is_directory": False,
                                "format": format_name,
                                "extension": ext.replace(".", ""),
                                "size_mb": round(file_size / (1024 * 1024), 2),
                                "model_type": "ASR / Neural Model",
                                "asr_compatible": False,
                            })
                            seen_paths.add(file_path)
        except Exception as e:
            print(f"[Model Scanner] Error scanning directory {directory_path}: {e}")

    # Sort models alphabetically by name
    discovered_models.sort(key=lambda m: m["name"].lower())
    return discovered_models

def get_whisper_model(
    model_size_or_path: str = "base", 
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    media_file_path: Optional[str] = None,
) -> WhisperModel:
    """
    Loads and caches a Whisper model in memory.
    Supports standard sizes ('tiny', 'base', 'small', 'medium') OR a direct custom model path / folder!
    """
    # The caller resolves the selected model path. Do not replace an explicit
    # request with the saved global setting, which made model selection ineffective.
    target_model = model_size_or_path

    runtime = get_asr_runtime()
    device = device or runtime["device"]
    compute_type = compute_type or runtime["compute_type"]

    cache_key = f"{target_model}_{device}_{compute_type}"
    if cache_key not in _WHISPER_MODELS:
        print(f"[ASR] Loading model '{target_model}' on {device} ({compute_type})...")
        try:
            _WHISPER_MODELS[cache_key] = WhisperModel(target_model, device=device, compute_type=compute_type)
        except Exception as e:
            from error_logging import write_error
            write_error(
                error_code="ASR_MODEL_LOAD_FAILED",
                error_details=str(e),
                exc=e,
                context={
                    "selected_model_path": target_model,
                    "device": device,
                    "compute_type": compute_type,
                    "media_file_path": media_file_path,
                },
            )
            raise
            
    return _WHISPER_MODELS[cache_key]

def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    Extracts duration, file format, and size of any media file.
    """
    try:
        duration = librosa.get_duration(path=file_path)
        file_size = os.path.getsize(file_path)
        ext = os.path.splitext(file_path)[1].lower().replace(".", "")
        return {
            "duration": float(duration),
            "file_size": file_size,
            "sample_rate": 16000,
            "channels": 1,
            "file_format": ext
        }
    except Exception as e:
        print(f"[ASR] Error extracting audio metadata for {file_path}: {e}")
        return {
            "duration": 0.0,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "sample_rate": 16000,
            "channels": 1,
            "file_format": os.path.splitext(file_path)[1].lower().replace(".", "")
        }

def transcribe_with_sarvam_saaras(
    file_path: str, 
    language_code: str = "ta-IN", 
    api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Transcribes audio using Sarvam AI's Saaras v2 cloud model for Indic languages.
    """
    sarvam_key = api_key or get_setting("sarvam_api_key")
    if not sarvam_key:
        return None
    try:
        url = "https://api.sarvam.ai/speech-to-text"
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/wav")}
            data = {"language_code": language_code, "model": "saaras:v2"}
            headers = {"api-subscription-key": sarvam_key}
            
            with httpx.Client(timeout=60.0) as client:
                res = client.post(url, headers=headers, files=files, data=data)
                if res.status_code == 200:
                    resp_json = res.json()
                    transcript = resp_json.get("transcript", "")
                    duration = librosa.get_duration(path=file_path)
                    
                    sentences = [s.strip() for s in transcript.split(".") if s.strip()]
                    chunks = []
                    if sentences:
                        sec_per_sent = max(1.5, duration / len(sentences))
                        for i, sent in enumerate(sentences):
                            start = round(i * sec_per_sent, 2)
                            end = round(min(duration, (i + 1) * sec_per_sent), 2)
                            chunks.append({
                                "chunk_index": i,
                                "start_time": start,
                                "end_time": end,
                                "text": sent,
                                "language": language_code.split("-")[0]
                            })
                    else:
                        chunks.append({
                            "chunk_index": 0,
                            "start_time": 0.0,
                            "end_time": duration,
                            "text": transcript,
                            "language": language_code.split("-")[0]
                        })

                    return {
                        "language": language_code.split("-")[0],
                        "full_transcript": transcript,
                        "duration": duration,
                        "chunks": chunks,
                        "engine": "sarvam_saaras"
                    }
    except Exception as e:
        print(f"[ASR] Sarvam Saaras transcription error: {e}")
    return None

def transcribe_with_faster_whisper(
    file_path: str,
    language: Optional[str] = "auto",
    model_size_or_path: str = "base"
) -> Dict[str, Any]:
    """
    Transcribes audio offline using Faster-Whisper or custom model with timestamps.
    """
    lang_param = None if (not language or language == "auto") else language
    runtime = get_asr_runtime()

    def run(device: str, compute_type: str):
        model = get_whisper_model(
            model_size_or_path=model_size_or_path,
            device=device,
            compute_type=compute_type,
            media_file_path=file_path,
        )
        segments, info = model.transcribe(
            file_path,
            language=lang_param,
            beam_size=5,
            word_timestamps=False,
            # Avoid filtering singing, quiet speech, or noisy recordings.
            vad_filter=False,
        )
        output_chunks: List[Dict[str, Any]] = []
        full_text: List[str] = []
        for idx, seg in enumerate(segments):
            cleaned_text = seg.text.strip()
            if not cleaned_text:
                continue
            full_text.append(cleaned_text)
            output_chunks.append({
                "chunk_index": idx,
                "start_time": round(seg.start, 2),
                "end_time": round(seg.end, 2),
                "text": cleaned_text,
                "language": info.language,
            })
        return info.language, info.duration, output_chunks, " ".join(full_text)

    try:
        detected_lang, duration, chunks, transcript = run(runtime["device"], runtime["compute_type"])
    except Exception as exc:
        if runtime["device"] != "cuda":
            raise
        from error_logging import write_error
        write_error(
            error_code="ASR_GPU_RUNTIME_FAILED",
            error_details=str(exc),
            exc=exc,
            context={"selected_model_path": model_size_or_path, "media_file_path": file_path, "fallback_device": "cpu"},
        )
        _WHISPER_MODELS.pop(f"{model_size_or_path}_cuda_float16", None)
        detected_lang, duration, chunks, transcript = run("cpu", "int8")

    result = {
        "language": detected_lang,
        "full_transcript": transcript,
        "duration": duration,
        "chunks": chunks,
        "engine": "faster_whisper"
    }
    if not result["full_transcript"].strip():
        raise RuntimeError(
            "The selected Whisper model returned no speech. Check the audio, language, and model selection, then retry."
        )
    return result

def transcribe_audio_file(
    file_path: str,
    language: Optional[str] = "auto",
    model_size: str = "base",
    engine: str = "faster_whisper",
    sarvam_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Unified entry point for transcribing any audio file with selectable engine.
    """
    if "sarvam" in engine.lower():
        lang_code = f"{language}-IN" if language and language != "auto" else "ta-IN"
        sarvam_res = transcribe_with_sarvam_saaras(file_path, language_code=lang_code, api_key=sarvam_api_key)
        if sarvam_res:
            return sarvam_res

    return transcribe_with_faster_whisper(file_path, language=language, model_size_or_path=model_size)
