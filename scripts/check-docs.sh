#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

require_file() {
  local path="$1"
  if [[ ! -f "${ROOT_DIR}/${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_heading() {
  local path="$1"
  local pattern="$2"
  if ! rg -q "${pattern}" "${ROOT_DIR}/${path}"; then
    echo "Missing required heading or marker in ${path}: ${pattern}" >&2
    exit 1
  fi
}

require_file "docs/README.md"
require_file "docs/tasks/README.md"
require_file "docs/tasks/runtime-performance-roadmap.md"
require_file "docs/tasks/voxcpm2-local-run.md"
require_file "docs/reference/runtime-baseline.md"
require_file "docs/roadmap/repo-alignment.md"

require_heading "docs/tasks/README.md" "🎯 当前焦点"
require_heading "docs/tasks/runtime-performance-roadmap.md" "^## 实施批次"
require_heading "docs/tasks/voxcpm2-local-run.md" "^## 验证方式"
require_heading "docs/reference/runtime-baseline.md" "^## 当前运行面"
require_heading "docs/roadmap/repo-alignment.md" "^## 已落地"

echo "Docs checks passed."
