"""
FreeClone.net GPU Backend Server
FastAPI server handling transcription, voice cloning, dubbing, translation, and video captioning.

Requires: CUDA-capable GPU, Python 3.11+, faster-whisper, CosyVoice2, FFmpeg
Run: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import time
import json
import asyncio
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ========================================================================
# Configuration
# ========================================================================

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/freeclone/uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/freeclone/outputs"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")
COSYVOICE_MODEL = os.getenv("COSYVOICE_MODEL", "pretrained_models/CosyVoice2-0.5B")
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 500 * 1024 * 1024))  # 500MB

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("freeclone-backend")

# ========================================================================
# App Setup
# ========================================================================

app = FastAPI(title="FreeClone GPU Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU/GPU-bound work
executor = ThreadPoolExecutor(max_workers=4)

# ========================================================================
# Global Model References (lazy-loaded)
# ========================================================================

_whisper_model = None
_whisper_pipeline = None
_cosyvoice_model = None


def get_whisper():
    """Lazy-load faster-whisper model with batched inference pipeline."""
    global _whisper_model, _whisper_pipeline
    if _whisper_model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE} ({WHISPER_COMPUTE})")
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        _whisper_pipeline = BatchedInferencePipeline(model=_whisper_model)
        logger.info("Whisper model loaded successfully")
    return _whisper_pipeline


def get_cosyvoice():
    """Lazy-load CosyVoice2 model."""
    global _cosyvoice_model
    if _cosyvoice_model is None:
        logger.info(f"Loading CosyVoice2 model: {COSYVOICE_MODEL}")
        from cosyvoice.cli.cosyvoice import CosyVoice2
        _cosyvoice_model = CosyVoice2(COSYVOICE_MODEL, load_jit=False, load_trt=False)
        logger.info("CosyVoice2 model loaded successfully")
    return _cosyvoice_model


# ========================================================================
# Job Store (in-memory, swap to Redis for production multi-worker)
# ========================================================================

jobs: dict = {}


class JobStatus:
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


def update_job(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        jobs[job_id]["updatedAt"] = time.time()


# ========================================================================
# Request Models
# ========================================================================

class ProcessRequest(BaseModel):
    jobId: str
    service: str  # clone, transcribe, translate, dub, caption
    sourceLanguage: Optional[str] = "en"
    targetLanguage: Optional[str] = None
    hasTranscript: Optional[bool] = False
    scriptText: Optional[str] = None
    captionOptions: Optional[dict] = None


class URLExtractRequest(BaseModel):
    url: str


# ========================================================================
# Transcription (faster-whisper)
# ========================================================================

def run_transcription(job_id: str, audio_path: str, language: str = None):
    """Run Whisper transcription with word-level timestamps."""
    try:
        update_job(job_id, status=JobStatus.PROCESSING, progress=10, message="Loading audio...")
        pipeline = get_whisper()

        update_job(job_id, progress=20, message="Transcribing audio...")

        # Use batched inference for speed (4-8x faster)
        segments, info = pipeline.transcribe(
            audio_path,
            language=language if language and language != "auto" else None,
            batch_size=16,
            word_timestamps=True,
        )

        update_job(job_id, progress=60, message="Processing segments...")

        result_segments = []
        full_text = []

        for segment in segments:
            seg_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": [],
            }
            if segment.words:
                for word in segment.words:
                    seg_data["words"].append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": round(word.probability, 3),
                    })
            result_segments.append(seg_data)
            full_text.append(segment.text.strip())

        transcript = " ".join(full_text)

        # Save transcript JSON
        output_path = OUTPUT_DIR / f"{job_id}_transcript.json"
        with open(output_path, "w") as f:
            json.dump({
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": info.duration,
                "segments": result_segments,
                "transcript": transcript,
            }, f, indent=2)

        update_job(
            job_id,
            status=JobStatus.DONE,
            progress=100,
            message="Transcription complete",
            transcript=transcript,
            segments=result_segments,
            language=info.language,
            duration=info.duration,
            outputFile=str(output_path),
        )
        logger.info(f"Job {job_id}: Transcription complete ({len(result_segments)} segments)")

    except Exception as e:
        logger.error(f"Job {job_id}: Transcription failed: {e}")
        update_job(job_id, status=JobStatus.ERROR, message=str(e))


# ========================================================================
# Caption Generation (SRT / VTT / Burned-in via FFmpeg)
# ========================================================================

def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def generate_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def generate_vtt(segments: list) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{format_vtt_time(seg['start'])} --> {format_vtt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def burn_captions_ffmpeg(video_path: str, srt_path: str, output_path: str, font_style: str = "default"):
    """Burn captions into video using FFmpeg subtitles filter."""
    style = ""
    if font_style == "bold":
        style = "force_style='FontSize=24,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'"
    elif font_style == "minimal":
        style = "force_style='FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H40000000,Outline=1'"
    else:
        style = "force_style='FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1'"

    filter_str = f"subtitles={srt_path}:{style}"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", filter_str,
        "-c:a", "copy",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        output_path
    ]
    logger.info(f"FFmpeg burn-in: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")
    return output_path


def run_captioning(job_id: str, audio_path: str, video_path: str = None,
                   language: str = None, caption_options: dict = None):
    """Transcribe audio and generate caption files. Optionally burn into video."""
    try:
        update_job(job_id, status=JobStatus.PROCESSING, progress=5, message="Starting captioning...")

        # Step 1: Transcribe
        pipeline = get_whisper()
        update_job(job_id, progress=15, message="Transcribing for captions...")

        segments, info = pipeline.transcribe(
            audio_path,
            language=language if language and language != "auto" else None,
            batch_size=16,
            word_timestamps=True,
        )

        seg_list = []
        for seg in segments:
            seg_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })

        update_job(job_id, progress=50, message="Generating caption files...")

        outputs = {}
        opts = caption_options or {}

        # SRT
        if opts.get("srt", True):
            srt_content = generate_srt(seg_list)
            srt_path = OUTPUT_DIR / f"{job_id}.srt"
            srt_path.write_text(srt_content, encoding="utf-8")
            outputs["srt"] = str(srt_path)

        # VTT
        if opts.get("vtt", True):
            vtt_content = generate_vtt(seg_list)
            vtt_path = OUTPUT_DIR / f"{job_id}.vtt"
            vtt_path.write_text(vtt_content, encoding="utf-8")
            outputs["vtt"] = str(vtt_path)

        # Burned-in (requires video file)
        if opts.get("burnedIn") and video_path and os.path.exists(video_path):
            update_job(job_id, progress=70, message="Burning captions into video...")
            srt_path = OUTPUT_DIR / f"{job_id}.srt"
            if not srt_path.exists():
                srt_path.write_text(generate_srt(seg_list), encoding="utf-8")

            burned_path = str(OUTPUT_DIR / f"{job_id}_captioned.mp4")
            burn_captions_ffmpeg(video_path, str(srt_path), burned_path, opts.get("fontStyle", "default"))
            outputs["burnedIn"] = burned_path

        update_job(
            job_id,
            status=JobStatus.DONE,
            progress=100,
            message="Captioning complete",
            captions=seg_list,
            captionFiles=outputs,
            language=info.language,
        )
        logger.info(f"Job {job_id}: Captioning complete ({len(seg_list)} segments)")

    except Exception as e:
        logger.error(f"Job {job_id}: Captioning failed: {e}")
        update_job(job_id, status=JobStatus.ERROR, message=str(e))


# ========================================================================
# Voice Cloning (CosyVoice2)
# ========================================================================

def run_voice_clone(job_id: str, audio_path: str, script_text: str,
                    prompt_text: str = None):
    """Clone voice using CosyVoice2 zero-shot inference."""
    try:
        update_job(job_id, status=JobStatus.PROCESSING, progress=10, message="Loading voice model...")
        model = get_cosyvoice()

        update_job(job_id, progress=30, message="Analyzing voice sample...")

        # If no prompt_text provided, transcribe the audio to get it
        if not prompt_text:
            update_job(job_id, progress=35, message="Transcribing voice sample for prompt...")
            pipeline = get_whisper()
            segments, _ = pipeline.transcribe(audio_path, batch_size=16)
            prompt_text = " ".join([s.text.strip() for s in segments])
            if not prompt_text:
                raise ValueError("Could not transcribe voice sample. Please provide clearer audio.")

        update_job(job_id, progress=50, message="Cloning voice...")

        # CosyVoice2 zero-shot: takes target text, prompt text, and prompt audio
        import torchaudio
        output_path = OUTPUT_DIR / f"{job_id}_cloned.wav"

        # Process in chunks if script is long (CosyVoice2 streams chunks)
        all_audio = []
        for chunk in model.inference_zero_shot(script_text, prompt_text, audio_path, stream=False):
            all_audio.append(chunk["tts_speech"])

        if all_audio:
            import torch
            combined = torch.cat(all_audio, dim=-1)
            torchaudio.save(str(output_path), combined, model.sample_rate)
        else:
            raise RuntimeError("CosyVoice2 produced no output")

        update_job(
            job_id,
            status=JobStatus.DONE,
            progress=100,
            message="Voice cloning complete",
            audioUrl=f"/api/download/{job_id}_cloned.wav",
            outputFile=str(output_path),
        )
        logger.info(f"Job {job_id}: Voice cloning complete")

    except Exception as e:
        logger.error(f"Job {job_id}: Voice cloning failed: {e}")
        update_job(job_id, status=JobStatus.ERROR, message=str(e))


# ========================================================================
# Dubbing (Transcribe + Translate + Clone)
# ========================================================================

def run_dubbing(job_id: str, audio_path: str, source_lang: str, target_lang: str,
                script_text: str = None):
    """Full dubbing pipeline: transcribe -> translate -> voice clone."""
    try:
        update_job(job_id, status=JobStatus.PROCESSING, progress=5, message="Starting dubbing pipeline...")

        # Step 1: Transcribe source audio
        if not script_text:
            update_job(job_id, progress=10, message="Transcribing source audio...")
            pipeline = get_whisper()
            segments, info = pipeline.transcribe(
                audio_path,
                language=source_lang if source_lang != "auto" else None,
                batch_size=16,
            )
            script_text = " ".join([s.text.strip() for s in segments])
            source_lang = info.language

        update_job(job_id, progress=30, message=f"Translating {source_lang} -> {target_lang}...")

        # Step 2: Translate
        # Use Whisper's built-in translation to English, or external translation API
        if target_lang == "en" and source_lang != "en":
            # Whisper can translate to English directly
            pipeline = get_whisper()
            segments, _ = pipeline.transcribe(audio_path, task="translate", batch_size=16)
            translated_text = " ".join([s.text.strip() for s in segments])
        else:
            # For other language pairs, we'd use an external translation service
            # Placeholder: in production, integrate DeepL, Google Translate, or NLLB
            translated_text = f"[Translation to {target_lang}]: {script_text}"
            logger.warning(f"Job {job_id}: Non-English translation not yet implemented, using placeholder")

        update_job(job_id, progress=50, message="Cloning voice with translated text...")

        # Step 3: Voice clone with translated text
        model = get_cosyvoice()
        import torchaudio
        import torch

        # Use source audio as voice reference
        pipeline_whisper = get_whisper()
        ref_segments, _ = pipeline_whisper.transcribe(audio_path, batch_size=16)
        prompt_text = " ".join([s.text.strip() for s in ref_segments])[:200]  # First ~200 chars as prompt

        all_audio = []
        for chunk in model.inference_zero_shot(translated_text, prompt_text, audio_path, stream=False):
            all_audio.append(chunk["tts_speech"])

        output_path = OUTPUT_DIR / f"{job_id}_dubbed.wav"
        if all_audio:
            combined = torch.cat(all_audio, dim=-1)
            torchaudio.save(str(output_path), combined, model.sample_rate)
        else:
            raise RuntimeError("Voice synthesis produced no output")

        update_job(
            job_id,
            status=JobStatus.DONE,
            progress=100,
            message="Dubbing complete",
            audioUrl=f"/api/download/{job_id}_dubbed.wav",
            outputFile=str(output_path),
            originalText=script_text,
            translatedText=translated_text,
        )
        logger.info(f"Job {job_id}: Dubbing complete")

    except Exception as e:
        logger.error(f"Job {job_id}: Dubbing failed: {e}")
        update_job(job_id, status=JobStatus.ERROR, message=str(e))


# ========================================================================
# URL Audio Extraction (yt-dlp)
# ========================================================================

def run_url_extract(job_id: str, url: str):
    """Extract audio from a URL using yt-dlp."""
    try:
        update_job(job_id, status=JobStatus.PROCESSING, progress=10, message="Extracting audio from URL...")

        output_template = str(UPLOAD_DIR / f"{job_id}_extracted.%(ext)s")
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", output_template,
            "--no-playlist",
            url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr[-500:]}")

        # Find the output file
        extracted = list(UPLOAD_DIR.glob(f"{job_id}_extracted.*"))
        if not extracted:
            raise RuntimeError("No audio file extracted")

        audio_path = str(extracted[0])

        # Get duration via ffprobe
        probe_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "csv=p=0", audio_path]
        duration_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration = float(duration_result.stdout.strip()) if duration_result.returncode == 0 else 0

        update_job(
            job_id,
            status=JobStatus.DONE,
            progress=100,
            message="Audio extracted",
            audioPath=audio_path,
            duration=duration,
        )

    except Exception as e:
        logger.error(f"Job {job_id}: URL extraction failed: {e}")
        update_job(job_id, status=JobStatus.ERROR, message=str(e))


# ========================================================================
# API Endpoints
# ========================================================================

@app.get("/")
async def root():
    """Server discovery and status."""
    return {
        "service": "FreeClone GPU Backend",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "POST /api/upload",
            "POST /api/process",
            "GET /api/job/{jobId}",
            "POST /api/url-extract",
            "GET /api/download/{filename}",
            "GET /health",
        ],
        "models": {
            "whisper": WHISPER_MODEL,
            "cosyvoice": COSYVOICE_MODEL,
        }
    }


@app.get("/health")
async def health():
    """Health check with model status."""
    return {
        "status": "ok",
        "gpu": WHISPER_DEVICE,
        "whisperLoaded": _whisper_model is not None,
        "cosyvoiceLoaded": _cosyvoice_model is not None,
        "activeJobs": sum(1 for j in jobs.values() if j.get("status") == JobStatus.PROCESSING),
        "totalJobs": len(jobs),
    }


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(None),
    url: str = Form(None),
    background_tasks: BackgroundTasks = None,
):
    """Upload audio/video file or provide a URL for extraction."""
    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    if file:
        # Save uploaded file
        ext = Path(file.filename).suffix or ".wav"
        save_path = UPLOAD_DIR / f"{job_id}{ext}"

        with open(save_path, "wb") as f:
            content = await file.read()
            if len(content) > MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=413, detail="File too large (max 500MB)")
            f.write(content)

        jobs[job_id] = {
            "jobId": job_id,
            "status": JobStatus.UPLOADED,
            "filePath": str(save_path),
            "fileName": file.filename,
            "fileSize": len(content),
            "createdAt": time.time(),
            "updatedAt": time.time(),
        }

        return {
            "jobId": job_id,
            "status": "uploaded",
            "message": "File received. Ready to process.",
            "size": len(content),
            "fileName": file.filename,
        }

    elif url:
        # URL extraction runs in background
        jobs[job_id] = {
            "jobId": job_id,
            "status": JobStatus.PROCESSING,
            "url": url,
            "createdAt": time.time(),
            "updatedAt": time.time(),
        }
        background_tasks.add_task(run_url_extract, job_id, url)

        return {
            "jobId": job_id,
            "status": "processing",
            "message": "Extracting audio from URL...",
        }

    else:
        raise HTTPException(status_code=400, detail="No file or URL provided")


@app.post("/api/process")
async def process_job(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Start processing a previously uploaded job."""
    job = jobs.get(req.jobId)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {req.jobId} not found")

    audio_path = job.get("filePath") or job.get("audioPath")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail="No audio file found for this job")

    # Determine video path (for captioning with burn-in)
    video_path = None
    if audio_path and any(audio_path.endswith(ext) for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]):
        video_path = audio_path

    service = req.service

    if service == "transcribe":
        background_tasks.add_task(
            run_transcription, req.jobId, audio_path, req.sourceLanguage
        )
    elif service == "clone":
        if not req.scriptText:
            raise HTTPException(status_code=400, detail="scriptText required for voice cloning")
        background_tasks.add_task(
            run_voice_clone, req.jobId, audio_path, req.scriptText
        )
    elif service == "caption":
        background_tasks.add_task(
            run_captioning, req.jobId, audio_path, video_path,
            req.sourceLanguage, req.captionOptions
        )
    elif service == "dub":
        if not req.targetLanguage:
            raise HTTPException(status_code=400, detail="targetLanguage required for dubbing")
        background_tasks.add_task(
            run_dubbing, req.jobId, audio_path,
            req.sourceLanguage or "auto", req.targetLanguage, req.scriptText
        )
    elif service == "translate":
        # Translation is transcribe + translate task via Whisper
        background_tasks.add_task(
            run_transcription, req.jobId, audio_path, req.sourceLanguage
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown service: {service}")

    update_job(req.jobId, status=JobStatus.PROCESSING, service=service, progress=0)

    return {
        "jobId": req.jobId,
        "status": "processing",
        "service": service,
        "message": f"Started {service} processing",
    }


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and results."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Return a safe subset (exclude internal paths)
    safe_keys = [
        "jobId", "status", "progress", "message", "service",
        "transcript", "segments", "captions", "captionFiles",
        "audioUrl", "language", "duration",
        "originalText", "translatedText",
        "createdAt", "updatedAt",
    ]
    return {k: job.get(k) for k in safe_keys if job.get(k) is not None}


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download a processed output file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security: ensure path is within OUTPUT_DIR
    if not file_path.resolve().is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")

    media_type = "audio/wav"
    if filename.endswith(".mp4"):
        media_type = "video/mp4"
    elif filename.endswith(".srt"):
        media_type = "text/plain"
    elif filename.endswith(".vtt"):
        media_type = "text/vtt"
    elif filename.endswith(".json"):
        media_type = "application/json"

    return FileResponse(file_path, media_type=media_type, filename=filename)


@app.post("/api/url-extract")
async def url_extract(req: URLExtractRequest, background_tasks: BackgroundTasks):
    """Extract audio from a URL (YouTube, etc.)."""
    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    jobs[job_id] = {
        "jobId": job_id,
        "status": JobStatus.PROCESSING,
        "url": req.url,
        "createdAt": time.time(),
        "updatedAt": time.time(),
    }

    background_tasks.add_task(run_url_extract, job_id, req.url)

    return {
        "jobId": job_id,
        "status": "processing",
        "message": "Extracting audio from URL...",
    }


# ========================================================================
# Preload Models (optional, run at startup for faster first request)
# ========================================================================

@app.on_event("startup")
async def startup_preload():
    """Optionally preload models at startup."""
    preload = os.getenv("PRELOAD_MODELS", "false").lower() == "true"
    if preload:
        logger.info("Preloading models at startup...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, get_whisper)
        await loop.run_in_executor(executor, get_cosyvoice)
        logger.info("All models preloaded")
    else:
        logger.info("Models will be loaded on first request (set PRELOAD_MODELS=true to preload)")


# ========================================================================
# Run directly
# ========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
