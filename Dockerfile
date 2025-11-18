# Reproducible container for Khmer Sentiment Analysis Pipeline
# Base: slim Python with build tools and git (for optional commit metadata)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt || true

# Copy project sources
COPY . /app

# Default command opens a shell. Override with `docker run ... python modeling/train_baseline.py ...`
CMD ["bash"]
