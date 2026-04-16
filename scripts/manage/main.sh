#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${SCRIPT_DIR}/help.sh"
source "${SCRIPT_DIR}/setup.sh"
source "${SCRIPT_DIR}/dev.sh"
source "${SCRIPT_DIR}/check.sh"

cmd="${1:-help}"
subcmd="${2:-}"
target="${3:-}"

case "${cmd}" in
  help|-h|--help)
    print_help
    ;;
  setup)
    case "${subcmd}" in
      env) setup_env ;;
      model) setup_model "${target}" ;;
      huggingface|hf) setup_model "huggingface" ;;
      modelscope|ms) setup_model "modelscope" ;;
      all) setup_all ;;
      *)
        echo "Unknown setup target: ${subcmd:-<empty>}" >&2
        print_help
        exit 1
        ;;
    esac
    ;;
  dev)
    case "${subcmd}" in
      api) dev_api ;;
      *)
        echo "Unknown dev target: ${subcmd:-<empty>}" >&2
        print_help
        exit 1
        ;;
    esac
    ;;
  check)
    case "${subcmd}" in
      docs) check_docs ;;
      quick) check_quick ;;
      *)
        echo "Unknown check target: ${subcmd:-<empty>}" >&2
        print_help
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    print_help
    exit 1
    ;;
esac
