# syntax=docker/dockerfile:1.4
# ==============================================================================
# hailo-yolo11-compiler Containerfile
# Provides reproducible Portable ML Mode & testing environment
# ==============================================================================
FROM python:3.12-slim-bookworm AS base

# Install core runtime dependencies
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root runner user
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /workspace

# Install Python requirements
COPY pyproject.toml README.md /workspace/
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[training,onnx,dev]"

# Copy source tree
COPY . /workspace/
RUN chown -R appuser:appuser /workspace

USER appuser

VOLUME ["/workspace/artifacts", "/workspace/data"]

ENTRYPOINT ["python", "train_and_compile.py"]
CMD ["doctor"]
