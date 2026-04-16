#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

dev_api() {
  require_uv
  local root model_dir host port devices cuda_arch_list
  root="$(repo_root)"
  model_dir="$(default_model_dir)"
  host="$(default_service_host)"
  port="$(default_service_port)"
  devices="$(default_devices)"
  cuda_arch_list="$(default_cuda_arch_list)"

  (
    cd "${root}"
    export NANOVLLM_MODEL_PATH="${NANOVLLM_MODEL_PATH:-${model_dir}}"
    export NANOVLLM_SERVERPOOL_DEVICES="${devices}"
    export TORCH_CUDA_ARCH_LIST="${cuda_arch_list}"
    export NANOVLLM_ATTENTION_BACKEND="${NANOVLLM_ATTENTION_BACKEND:-sdpa}"
    uv run --package nano-vllm-voxcpm-deployment \
      fastapi run deployment/app/main.py --host "${host}" --port "${port}"
  )
}
