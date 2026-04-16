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
 && chown -R appuser:appuser "$NANOVLLM_CACHE_DIR" \
 && chown -R appuser:appuser /app

USER appuser

COPY --chown=appuser:appuser . .

# Install workspace packages from the frozen lockfile.
RUN uv sync --all-packages --frozen --no-dev -v --compile-bytecode --no-cache --no-editable

EXPOSE 8020

# k8s probes can use /health and /ready.
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]
