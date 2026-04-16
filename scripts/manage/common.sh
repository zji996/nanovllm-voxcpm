#!/usr/bin/env bash

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

default_model_repo() {
  local model_source
  model_source="${1:-$(default_model_source)}"

  if [[ -n "${MODEL_REPO:-}" ]]; then
    printf '%s\n' "${MODEL_REPO}"
    return
  fi

  case "${model_source}" in
    huggingface|hf)
      printf '%s\n' "openbmb/VoxCPM2"
      ;;
    modelscope|ms)
      printf '%s\n' "OpenBMB/VoxCPM2"
      ;;
    *)
      printf '%s\n' "openbmb/VoxCPM2"
      ;;
  esac
}

default_model_source() {
  printf '%s\n' "${MODEL_SOURCE:-modelscope}"
}

default_model_revision() {
  printf '%s\n' "${MODEL_REVISION:-}"
}

default_model_dir() {
  printf '%s\n' "${MODEL_DIR:-$(repo_root)/models/VoxCPM2}"
}

default_service_host() {
  printf '%s\n' "${NANOVLLM_SERVICE_HOST:-0.0.0.0}"
}

default_service_port() {
  printf '%s\n' "${NANOVLLM_SERVICE_PORT:-8010}"
}

default_devices() {
  printf '%s\n' "${NANOVLLM_SERVERPOOL_DEVICES:-0}"
}

default_cuda_arch_list() {
  printf '%s\n' "${TORCH_CUDA_ARCH_LIST:-8.6}"
}

require_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found in PATH." >&2
    exit 1
  fi
}
