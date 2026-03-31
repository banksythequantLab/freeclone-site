FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir yt-dlp

# Install CosyVoice2 from source
RUN git clone https://github.com/FunAudioLLM/CosyVoice.git /tmp/CosyVoice \
    && cd /tmp/CosyVoice && pip install --no-cache-dir -e . \
    && cd /app

# Download model weights (commented out - large download, do at runtime or bake into image)
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

# Copy server
COPY server.py .

# Create dirs
RUN mkdir -p /tmp/freeclone/uploads /tmp/freeclone/outputs

EXPOSE 8000

ENV PRELOAD_MODELS=false
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE=float16

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
