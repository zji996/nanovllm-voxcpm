#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

normalize_model_source() {
  local model_source
  model_source="${1:-$(default_model_source)}"
  case "${model_source}" in
    huggingface|hf)
      printf '%s\n' "huggingface"
      ;;
    modelscope|ms)
      printf '%s\n' "modelscope"
      ;;
    *)
      echo "Unsupported model source: ${model_source}" >&2
      echo "Expected one of: huggingface, hf, modelscope, ms" >&2
      return 1
      ;;
  esac
}

setup_env() {
  require_uv
  local root cuda_arch_list
  root="$(repo_root)"
  cuda_arch_list="$(default_cuda_arch_list)"
  (
    cd "${root}"
    TORCH_CUDA_ARCH_LIST="${cuda_arch_list}" uv sync --all-packages --frozen
  )
}

setup_model() {
  require_uv
  local requested_source root model_source model_repo model_dir model_revision
  requested_source="${1:-}"
  root="$(repo_root)"
  model_source="$(normalize_model_source "${requested_source}")"
  model_repo="$(default_model_repo "${model_source}")"
  model_dir="$(default_model_dir)"
  model_revision="$(default_model_revision)"

  mkdir -p "${model_dir}"

  (
    cd "${root}"
    case "${model_source}" in
      huggingface)
        MODEL_REPO="${model_repo}" MODEL_DIR="${model_dir}" MODEL_REVISION="${model_revision}" \
          uv run --with huggingface_hub python - <<'PY'
import os

from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_REPO"]
local_dir = os.environ["MODEL_DIR"]
revision = os.environ.get("MODEL_REVISION") or None

print(f"Downloading {repo_id} into {local_dir}")
snapshot_download(repo_id=repo_id, local_dir=local_dir, revision=revision)
print(f"Model ready at {local_dir}")
PY
        ;;
      modelscope)
        MODEL_REPO="${model_repo}" MODEL_DIR="${model_dir}" MODEL_REVISION="${model_revision}" \
          uv run --with modelscope python - <<'PY'
import os

from modelscope import snapshot_download

repo_id = os.environ["MODEL_REPO"]
local_dir = os.environ["MODEL_DIR"]
revision = os.environ.get("MODEL_REVISION") or None

print(f"Downloading {repo_id} from ModelScope into {local_dir}")
snapshot_download(model_id=repo_id, local_dir=local_dir, revision=revision)
print(f"Model ready at {local_dir}")
PY
        ;;
      *)
        echo "Unsupported MODEL_SOURCE: ${model_source}" >&2
        exit 1
        ;;
    esac
  )
}

setup_all() {
  setup_env
  setup_model
}
