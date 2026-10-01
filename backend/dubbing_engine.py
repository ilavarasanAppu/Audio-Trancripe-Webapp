import os
import sys
import asyncio
import json
import httpx
import librosa
import soundfile as sf
import numpy as np
import edge_tts
from typing import List, Dict, Any, Optional, Tuple
from deep_translator import GoogleTranslator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import get_db, get_setting

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Language Mapping & Built-in Neural Voices
INDIAN_LANGUAGES = {
    "ta": {
        "name": "Tamil (தமிழ்)",
        "code": "ta",
        "sarvam_code": "ta-IN",
        "voices": [
            {"id": "ta-IN-ValluvarNeural", "name": "Valluvar (Male - Deep/Authoritative)", "gender": "male", "base_f0": 125},
            {"id": "ta-IN-PallaviNeural", "name": "Pallavi (Female - Clear/Empathic)", "gender": "female", "base_f0": 215},
            {"id": "ta-LK-KumarNeural", "name": "Kumar (Male - Sri Lanka)", "gender": "male", "base_f0": 130},
            {"id": "ta-LK-SaranyaNeural", "name": "Saranya (Female - Sri Lanka)", "gender": "female", "base_f0": 220}
        ],
        "default_voice": "ta-IN-ValluvarNeural"
    },
    "te": {
        "name": "Telugu (తెలుగు)",
        "code": "te",
        "sarvam_code": "te-IN",
        "voices": [
            {"id": "te-IN-MohanNeural", "name": "Mohan (Male)", "gender": "male", "base_f0": 125},
            {"id": "te-IN-ShrutiNeural", "name": "Shruti (Female)", "gender": "female", "base_f0": 215}
        ],
        "default_voice": "te-IN-MohanNeural"
    },
    "ml": {
        "name": "Malayalam (മലയാളം)",
        "code": "ml",
        "sarvam_code": "ml-IN",
        "voices": [
            {"id": "ml-IN-MidhunNeural", "name": "Midhun (Male)", "gender": "male", "base_f0": 125},
            {"id": "ml-IN-SobhanaNeural", "name": "Sobhana (Female)", "gender": "female", "base_f0": 215}
        ],
        "default_voice": "ml-IN-MidhunNeural"
    },
    "kn": {
        "name": "Kannada (ಕನ್ನಡ)",
        "code": "kn",
        "sarvam_code": "kn-IN",
        "voices": [
            {"id": "kn-IN-GaganNeural", "name": "Gagan (Male)", "gender": "male", "base_f0": 125},
            {"id": "kn-IN-SapnaNeural", "name": "Sapna (Female)", "gender": "female", "base_f0": 215}
        ],
        "default_voice": "kn-IN-GaganNeural"
    },
    "hi": {
        "name": "Hindi (हिन्दी)",
        "code": "hi",
        "sarvam_code": "hi-IN",
        "voices": [
            {"id": "hi-IN-MadhurNeural", "name": "Madhur (Male)", "gender": "male", "base_f0": 125},
            {"id": "hi-IN-SwaraNeural", "name": "Swara (Female)", "gender": "female", "base_f0": 215}
        ],
        "default_voice": "hi-IN-MadhurNeural"
    },
    "en": {
        "name": "English (India/US)",
        "code": "en",
        "sarvam_code": "en-IN",
        "voices": [
            {"id": "en-IN-PrabhatNeural", "name": "Prabhat (Male - India)", "gender": "male", "base_f0": 130},
            {"id": "en-IN-NeerjaNeural", "name": "Neerja (Female - India)", "gender": "female", "base_f0": 215},
            {"id": "en-US-GuyNeural", "name": "Guy (Male - US)", "gender": "male", "base_f0": 120},
            {"id": "en-US-JennyNeural", "name": "Jenny (Female - US)", "gender": "female", "base_f0": 210}
        ],
        "default_voice": "en-IN-PrabhatNeural"
    },
    "mr": {
        "name": "Marathi (मराठी)",
        "code": "mr",
        "sarvam_code": "mr-IN",
        "voices": [
            {"id": "mr-IN-ManoharNeural", "name": "Manohar (Male)", "gender": "male", "base_f0": 125},
            {"id": "mr-IN-AarohiNeural", "name": "Aarohi (Female)", "gender": "female", "base_f0": 215}
        ],
        "default_voice": "mr-IN-ManoharNeural"
    },
    "bn": {
        "name": "Bengali (বাংলা)",
        "code": "bn",
        "sarvam_code": "bn-IN",
        "voices": [
            {"id": "bn-IN-BashkarNeural", "name": "Bashkar (Male)", "gender": "male", "base_f0": 125},
            {"id": "bn-IN-TanishaaNeural", "name": "Tanishaa (Female)", "gender": "female", "base_f0": 215}
        ],
        "default_voice": "bn-IN-BashkarNeural"
    },
    "gu": {
        "name": "Gujarati (ગુજરાતી)",
        "code": "gu",
        "sarvam_code": "gu-IN",
        "voices": [
            {"id": "gu-IN-NiranjanNeural", "name": "Niranjan (Male)", "gender": "male", "base_f0": 125},
            {"id": "gu-IN-DhwaniNeural", "name": "Dhwani (Female)", "gender": "female", "base_f0": 215}
        ],
        "default_voice": "gu-IN-NiranjanNeural"
    }
}

SARVAM_SPEAKERS = [
    {"id": "meera", "name": "Meera (Female - Natural)", "gender": "female"},
    {"id": "pavithra", "name": "Pavithra (Female - Clear)", "gender": "female"},
    {"id": "maitreyi", "name": "Maitreyi (Female - Expressive)", "gender": "female"},
    {"id": "arvind", "name": "Arvind (Male - Deep)", "gender": "male"},
    {"id": "amartya", "name": "Amartya (Male - Energetic)", "gender": "male"}
]

def extract_acoustic_profile(audio_path: str, start_time: float, end_time: float) -> Dict[str, float]:
    """
    Extracts fundamental frequency (F0 pitch in Hz), speech energy (loudness),
    and tempo metrics to ensure the synthesized voice reproduces the same person's pitch and emotion.
    """
    try:
        dur = max(0.4, end_time - start_time)
        y, sr = librosa.load(audio_path, sr=16000, offset=max(0, start_time), duration=dur, mono=True)
        if len(y) == 0:
            return {"median_f0": 140.0, "rms_energy": 0.1}

        # 1. Pitch estimation using YIN
        f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr)
        voiced_f0 = f0[f0 > 65]
        median_f0 = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else 140.0

        # 2. RMS Energy (dynamic emotion/emphasis)
        rms = float(np.mean(librosa.feature.rms(y=y)))

        return {
            "median_f0": round(median_f0, 1),
            "rms_energy": round(rms, 4)
        }
    except Exception as e:
        print(f"Error extracting acoustic profile: {e}")
        return {"median_f0": 140.0, "rms_energy": 0.1}

async def translate_sarvam_direct(text: str, src_code: str, tgt_code: str, sarvam_key: str) -> Optional[str]:
    """Single-hop Sarvam Mayura translation call."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.sarvam.ai/translate",
                headers={"api-subscription-key": sarvam_key, "Content-Type": "application/json"},
                json={
                    "input": text,
                    "source_language_code": src_code,
                    "target_language_code": tgt_code,
                    "speaker_gender": "Male",
                    "mode": "formal"
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("translated_text", "")
    except Exception as e:
        print(f"Sarvam translation step failed ({src_code} -> {tgt_code}): {e}")
    return None

async def translate_text(text: str, source_lang: str, target_lang: str, sarvam_api_key: Optional[str] = None) -> str:
    """
    Translates text to target language.
    Requirement 1: For Sarvam AI, since Sarvam Mayura only translates directly from/to English (en-IN),
    every non-English language is first converted to English, and then from English to the target language!
    """
    text = text.strip()
    if not text:
        return ""
    if source_lang == target_lang:
        return text

    sarvam_key = sarvam_api_key or get_setting("sarvam_api_key")

    # If Sarvam AI is selected and API key is present:
    if sarvam_key:
        src_code = INDIAN_LANGUAGES.get(source_lang, {}).get("sarvam_code", f"{source_lang}-IN" if source_lang != "auto" else "en-IN")
        tgt_code = INDIAN_LANGUAGES.get(target_lang, {}).get("sarvam_code", f"{target_lang}-IN")

        # Case A: Source is already English -> Direct English to Target (en-IN -> tgt_code)
        if source_lang == "en" or src_code == "en-IN":
            res = await translate_sarvam_direct(text, "en-IN", tgt_code, sarvam_key)
            if res:
                return res

        # Case B: Target is English -> Direct Source to English (src_code -> en-IN)
        elif target_lang == "en" or tgt_code == "en-IN":
            res = await translate_sarvam_direct(text, src_code, "en-IN", sarvam_key)
            if res:
                return res

        # Case C: Both Source and Target are non-English (e.g. Tamil -> Telugu, Hindi -> Tamil):
        # First pivot: Source -> English, Second pivot: English -> Target
        else:
            print(f"Sarvam English Pivot: Step 1 ({src_code} -> en-IN)...")
            intermediate_english = await translate_sarvam_direct(text, src_code, "en-IN", sarvam_key)
            if not intermediate_english:
                # Fallback intermediate to GoogleTranslator
                try:
                    intermediate_english = GoogleTranslator(source=source_lang, target="en").translate(text)
                except Exception:
                    intermediate_english = text

            if intermediate_english:
                print(f"Sarvam English Pivot: Step 2 (en-IN -> {tgt_code})...")
                final_res = await translate_sarvam_direct(intermediate_english, "en-IN", tgt_code, sarvam_key)
                if final_res:
                    return final_res

    # Fallback to DeepTranslator / Google
    try:
        translated = GoogleTranslator(source=source_lang if source_lang != "auto" else "auto", target=target_lang).translate(text)
        return translated or text
    except Exception as e:
        print(f"DeepTranslator error: {e}")
        return text

async def synthesize_neural_voice(
    text: str,
    voice_id: str,
    output_path: str,
    rate_pct: int = 0,
    pitch_hz: int = 0,
    speaker_reference_audio: Optional[str] = None
) -> bool:
    """
    Synthesizes speech using Microsoft Edge Neural TTS with exact pitch & speed modulation.
    If speaker_reference_audio is provided, extracts pitch/energy from it for matching.
    """
    try:
        # If we have a reference audio, extract acoustic profile to match
        if speaker_reference_audio and os.path.exists(speaker_reference_audio):
            profile = extract_acoustic_profile(speaker_reference_audio, 0, 30)
            if pitch_hz == 0:
                pitch_hz = int(profile["median_f0"] - 140)  # Adjust relative to neutral
        
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
        pitch_str = f"+{pitch_hz}Hz" if pitch_hz >= 0 else f"{pitch_hz}Hz"
        
        communicate = edge_tts.Communicate(text, voice_id, rate=rate_str, pitch=pitch_str)
        await communicate.save(output_path)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        print(f"Error synthesizing voice with {voice_id}: {e}")
        return False

async def synthesize_sarvam_voice(
    text: str,
    target_lang: str,
    speaker: str,
    output_path: str,
    sarvam_api_key: Optional[str] = None,
    pitch_offset: int = 0
) -> bool:
    """
    Synthesizes speech using Sarvam AI Bulbul TTS.
    """
    sarvam_key = sarvam_api_key or get_setting("sarvam_api_key")
    if not sarvam_key:
        return False
    try:
        lang_code = INDIAN_LANGUAGES.get(target_lang, {}).get("sarvam_code", "ta-IN")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={"api-subscription-key": sarvam_key, "Content-Type": "application/json"},
                json={
                    "inputs": [text],
                    "target_language_code": lang_code,
                    "speaker": speaker or "meera",
                    "pitch": pitch_offset,
                    "pace": 1.0,
                    "loudness": 1.5,
                    "speech_sample_rate": 22050,
                    "enable_preprocessing": True,
                    "model": "bulbul:v1"
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                audios = data.get("audios", [])
                if audios:
                    import base64
                    audio_bytes = base64.b64decode(audios[0])
                    with open(output_path, "wb") as f:
                        f.write(audio_bytes)
                    return True
    except Exception as e:
        print(f"Error in Sarvam TTS synthesis: {e}")
    return False

def fit_audio_to_duration(input_wav: str, output_wav: str, target_duration: float, max_stretch: float = 1.45) -> float:
    """
    Adjusts audio playback rate/stretches audio so it fits the target video segment duration.
    """
    try:
        y, sr = librosa.load(input_wav, sr=22050, mono=True)
        current_duration = len(y) / sr
        
        if current_duration <= 0 or target_duration <= 0:
            sf.write(output_wav, y, sr)
            return current_duration

        rate = current_duration / target_duration
        clamped_rate = max(1.0 / max_stretch, min(max_stretch, rate))
        
        if abs(clamped_rate - 1.0) > 0.04:
            y_stretched = librosa.effects.time_stretch(y, rate=clamped_rate)
        else:
            y_stretched = y

        target_len = int(target_duration * sr)
        if len(y_stretched) < target_len:
            y_stretched = np.pad(y_stretched, (0, target_len - len(y_stretched)))
        elif len(y_stretched) > target_len:
            y_stretched = y_stretched[:target_len]
            fade_len = min(int(0.04 * sr), target_len)
            y_stretched[-fade_len:] *= np.linspace(1, 0, fade_len)

        sf.write(output_wav, y_stretched, sr)
        return len(y_stretched) / sr
    except Exception as e:
        print(f"Error time-syncing audio: {e}")
        return 0.0

async def build_dubbed_audio_track(
    segments: List[Dict[str, Any]],
    total_duration: float,
    output_audio_path: str,
    sr: int = 22050
) -> bool:
    """
    Stitches all dubbed segments onto a silent timeline master track.
    """
    try:
        master_samples = int(total_duration * sr) + sr
        master_audio = np.zeros(master_samples, dtype=np.float32)

        for seg in segments:
            audio_path = seg.get("audio_path")
            if not audio_path or not os.path.exists(audio_path):
                continue
            
            y, _ = librosa.load(audio_path, sr=sr, mono=True)
            start_sample = int(seg["start_time"] * sr)
            end_sample = start_sample + len(y)
            
            if start_sample < len(master_audio):
                actual_end = min(end_sample, len(master_audio))
                y_fit = y[:actual_end - start_sample]
                master_audio[start_sample:actual_end] += y_fit

        max_val = np.max(np.abs(master_audio))
        if max_val > 0.95:
            master_audio = master_audio * (0.95 / max_val)

        sf.write(output_audio_path, master_audio, sr)
        return True
    except Exception as e:
        print(f"Error assembling master dubbed track: {e}")
        return False
