FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch FIRST (avoids ~3 GB CUDA wheels)
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY app/ ./app/
COPY data/ ./data/

# Pre-download Silero VAD model into the torch hub cache so no network
# access is needed at runtime (avoids NO_SOCKET errors on Railway / HF Spaces)
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, verbose=False, trust_repo=True)"

# Hugging Face Spaces routes traffic to port 7860 and injects $PORT.
# Binding to $PORT keeps this portable across HF Spaces / Railway / local.
EXPOSE 7860

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
