# FreeClone GPU Backend

FastAPI server for audio transcription, voice cloning, dubbing, and video captioning.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (RTX 3090/4090 recommended)
- FFmpeg installed system-wide
- yt-dlp installed system-wide

## Quick Start (Local)

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Install CosyVoice2 from source
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice && pip install -e . && cd ..

# Download CosyVoice2 model
mkdir -p pretrained_models
# Download CosyVoice2-0.5B from https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B

# Run server
python server.py
# Or: uvicorn server:app --host 0.0.0.0 --port 8000
```

## Quick Start (Vast.ai / RunPod)

```bash
# Use PyTorch CUDA base image
# Template: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

apt-get update && apt-get install -y ffmpeg
pip install yt-dlp

cd /workspace
git clone https://github.com/banksythequantLab/freeclone-site.git
cd freeclone-site/backend
pip install -r requirements.txt

# Install CosyVoice2
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice && pip install -e . && cd ..

# Download model weights
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

# Start with model preloading
PRELOAD_MODELS=true python server.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UPLOAD_DIR` | `/tmp/freeclone/uploads` | Upload storage path |
| `OUTPUT_DIR` | `/tmp/freeclone/outputs` | Output files path |
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | Device (cuda/cpu) |
| `WHISPER_COMPUTE` | `float16` | Compute type |
| `COSYVOICE_MODEL` | `pretrained_models/CosyVoice2-0.5B` | CosyVoice model path |
| `MAX_UPLOAD_SIZE` | `524288000` | Max upload bytes (500MB) |
| `PRELOAD_MODELS` | `false` | Load models at startup |

## API Endpoints

- `POST /api/upload` - Upload audio/video file
- `POST /api/process` - Start processing (transcribe/clone/dub/caption)
- `GET /api/job/{id}` - Check job status
- `POST /api/url-extract` - Extract audio from URL
- `GET /api/download/{file}` - Download output file
- `GET /health` - Server health check

## Architecture

```
Cloudflare Worker (frontend) --> GPU Backend (this server)
     :8787                          :8000

Worker forwards /api/* requests to this backend.
Set BACKEND_URL in worker.js to point to this server.
```

## Services

| Service | Model | Description |
|---------|-------|-------------|
| `transcribe` | faster-whisper large-v3 | Speech-to-text with word timestamps |
| `clone` | CosyVoice2-0.5B | Zero-shot voice cloning |
| `dub` | Whisper + CosyVoice2 | Transcribe + translate + clone voice |
| `caption` | Whisper + FFmpeg | Generate SRT/VTT or burn into video |
| `translate` | Whisper | Translate audio to English |
