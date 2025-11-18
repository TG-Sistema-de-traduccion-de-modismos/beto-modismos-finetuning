FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY dataset.json dataset.json
COPY grid_search.py grid_search.py

CMD ["/opt/venv/bin/python", "grid_search.py"]