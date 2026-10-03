"""Structured, append-only error logging shared by API handlers."""

import json
import os
import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


ERROR_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "errors.jsonl")
_WRITE_LOCK = threading.Lock()
os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)
open(ERROR_LOG_PATH, "a", encoding="utf-8").close()


def _source_details(exc: Optional[BaseException]) -> tuple[Optional[str], Optional[int], Optional[str]]:
    if exc is None:
        return None, None, None
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return None, None, None
    frame = frames[-1]
    return frame.filename, frame.lineno, "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def write_error(
    *,
    error_code: str,
    error_details: Any,
    exc: Optional[BaseException] = None,
    request: Any = None,
    context: Optional[dict[str, Any]] = None,
    source_file: Optional[str] = None,
    error_line: Optional[int] = None,
    stack: Optional[str] = None,
) -> dict[str, Any]:
    """Write one JSONL record and return it; secrets and request bodies are not captured."""
    traceback_file, traceback_line, traceback_text = _source_details(exc)
    url_path = str(request.url.path) if request is not None else None
    media_path = (context or {}).get("media_file_path")
    if request is not None:
        media_path = request.query_params.get("file_path") or request.query_params.get("path") or media_path
    resolved_source = source_file or traceback_file
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": uuid.uuid4().hex,
        "error_code": error_code,
        "error_details": str(error_details),
        "source_file": resolved_source,
        "source_filename": os.path.basename(resolved_source) if resolved_source else None,
        "error_line": error_line if error_line is not None else traceback_line,
        "error_column": context.get("error_column") if context else None,
        "traceback": stack or traceback_text,
        "log_file_path": ERROR_LOG_PATH,
        "request_method": request.method if request is not None else None,
        "request_path": url_path,
        "media_file_path": media_path,
        "media_filename": os.path.basename(media_path) if media_path else None,
        "client_host": request.client.host if request is not None and request.client else None,
        "context": context or {},
    }
    os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)
    with _WRITE_LOCK:
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record
