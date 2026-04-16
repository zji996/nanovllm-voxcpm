# syntax=docker/dockerfile:1.7
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu22.04
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

FROM ${CUDA_IMAGE}

# Re-declare build args for this stage so they can be used in ENV/RUN.
ARG TORCH_CUDA_ARCH_LIST

ENV DEBIAN_FRONTEND=noninteractive \
    NANOVLLM_ATTENTION_BACKEND=sdpa \
    MAX_JOBS=1 \
    NVCC_THREADS=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_CACHE_DIR=/home/appuser/.cache/uv \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Keep the image SDPA-first. Native build deps remain available for transitive
# wheels and optional extensions, but the runtime defaults to PyTorch SDPA.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      build-essential \
      ninja-build \
      git \
      python3-dev \
      python3-pip \
      pkg-config \
      ffmpeg \
      libsndfile1-dev \
  && rm -rf /var/lib/apt/lists/*

# Install uv
RUN python3 -m pip install --no-cache-dir uv

ENV NANOVLLM_CACHE_DIR=/var/cache/nanovllm
RUN useradd -m -u 10001 appuser \
 && mkdir -p "$NANOVLLM_CACHE_DIR" \
 && mkdir -p "$UV_CACHE_DIR" \
 && chown -R appuser:appuser "$NANOVLLM_CACHE_DIR" \
 && chown -R appuser:appuser "$UV_CACHE_DIR" \
 && chown -R appuser:appuser /app

USER appuser

COPY --chown=appuser:appuser pyproject.toml uv.lock README.md LICENSE ./
COPY --chown=appuser:appuser deployment/pyproject.toml deployment/README.md ./deployment/

# Install third-party dependencies first so repeated source-only rebuilds can
# reuse the uv cache and avoid re-downloading large wheels.
RUN --mount=type=cache,target=/home/appuser/.cache/uv \
    uv sync --all-packages --frozen --no-dev -v --compile-bytecode --no-editable --no-install-workspace

COPY --chown=appuser:appuser nanovllm_voxcpm ./nanovllm_voxcpm
COPY --chown=appuser:appuser deployment/app ./deployment/app

# Install the local workspace packages after copying source. This still reuses
# the previously populated uv cache, so code-only rebuilds avoid downloading
# dependencies again.
RUN --mount=type=cache,target=/home/appuser/.cache/uv \
    uv sync --all-packages --frozen --no-dev -v --compile-bytecode --no-editable

EXPOSE 8020

# k8s probes can use /health and /ready.
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]
