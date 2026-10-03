import sys
import os
import subprocess
import shutil
import importlib
from typing import Dict, Any, List

REQUIRED_PIP_PACKAGES = [
    "fastapi",
    "uvicorn",
    "faster_whisper",
    "edge_tts",
    "deep_translator",
    "librosa",
    "soundfile",
    "scikit-learn",
    "numpy",
    "pydub",
    "httpx",
    "requests",
    "huggingface_hub"
]

def check_python_environment() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "executable": sys.executable,
        "platform": sys.platform
    }

def check_ffmpeg() -> Dict[str, Any]:
    ffmpeg_path = shutil.which("ffmpeg")
    return {
        "installed": ffmpeg_path is not None,
        "path": ffmpeg_path or "Not found in PATH (librosa and soundfile will use fallback/pydub)"
    }

PACKAGE_ALIASES = {
    "scikit-learn": "sklearn"
}

def check_package(package_name: str) -> Dict[str, Any]:
    import_name = PACKAGE_ALIASES.get(package_name, package_name.replace("-", "_"))
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "Installed")
        return {"package": package_name, "status": "ok", "version": str(version), "installed": True}
    except Exception as e:
        return {"package": package_name, "status": "missing", "error": str(e), "installed": False}

def run_diagnostics() -> Dict[str, Any]:
    packages_status = [check_package(pkg) for pkg in REQUIRED_PIP_PACKAGES]
    missing_packages = [p["package"] for p in packages_status if not p["installed"]]
    
    # Check ASR engines
    asr_engines = {
        "faster_whisper": check_package("faster_whisper")["installed"],
        "sarvam_saaras": True, # Cloud API
        "browser_wasm": True
    }

    try:
        from transcription import get_asr_runtime
        asr_runtime = get_asr_runtime()
    except Exception as exc:
        asr_runtime = {"device": "cpu", "compute_type": "int8", "gpu_available": False, "gpu_count": 0, "gpu_error": str(exc)}

    # Check TTS engines
    tts_engines = {
        "edge_neural": check_package("edge_tts")["installed"],
        "sarvam_bulbul": True, # Cloud API
        "deep_translator": check_package("deep_translator")["installed"]
    }

    all_healthy = len(missing_packages) == 0

    return {
        "healthy": all_healthy,
        "missing_count": len(missing_packages),
        "missing_packages": missing_packages,
        "packages": packages_status,
        "ffmpeg": check_ffmpeg(),
        "asr_engines": asr_engines,
        "asr_runtime": asr_runtime,
        "tts_engines": tts_engines,
        "python": check_python_environment()
    }

def auto_fix_missing() -> Dict[str, Any]:
    diag = run_diagnostics()
    missing = diag["missing_packages"]
    if not missing:
        return {"status": "success", "message": "All dependencies are already healthy!", "installed": []}

    print(f"Auto-fixing missing packages: {missing}...")
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {
            "status": "success",
            "message": f"Successfully installed {len(missing)} package(s).",
            "installed": missing,
            "stdout": res.stdout
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to auto-install packages: {e}",
            "installed": []
        }

if __name__ == "__main__":
    print("=== VaniScript AI System Doctor ===")
    diag = run_diagnostics()
    print(f"Health Status: {'ALL HEALTHY' if diag['healthy'] else 'ISSUES FOUND'}")
    print(f"Missing packages: {diag['missing_packages']}")
    if diag["missing_packages"]:
        print("Running auto-fix...")
        fix_res = auto_fix_missing()
        print(fix_res["message"])
