#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

check_docs() {
  local root
  root="$(repo_root)"
  (
    cd "${root}"
    ./scripts/check-docs.sh
  )
}

check_quick() {
  require_uv
  local root
  root="$(repo_root)"
  check_docs
  (
    cd "${root}"
    uv run python -m compileall nanovllm_voxcpm deployment tests
  )
}
