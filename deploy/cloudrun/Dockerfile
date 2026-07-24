# BlueSpotter inference API — one image for both CPU and L4 GPU on Cloud Run.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    MPLCONFIGDIR=/tmp/mpl \
    HF_HOME=/tmp/hf

WORKDIR /app

# System libs cellpose/scikit-image/matplotlib need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py .

# Cloud Run sends traffic to $PORT (default 8080). One worker: the model is loaded
# once per instance and inference is CPU/GPU-bound; scale out with instances, not
# workers, so each instance keeps a single warm model in memory.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
