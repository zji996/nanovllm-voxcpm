#!/usr/bin/env bash

print_help() {
  cat <<'EOF'
Usage:
  ./manage.sh help
  ./manage.sh setup env
  ./manage.sh setup model
  ./manage.sh setup model huggingface
  ./manage.sh setup model modelscope
  ./manage.sh setup huggingface
  ./manage.sh setup modelscope
  ./manage.sh setup all
  ./manage.sh dev api
  ./manage.sh check docs
  ./manage.sh check quick

Commands:
  help
    Show this message.

  setup env
    Sync the workspace with deployment dependencies.
    Defaults TORCH_CUDA_ARCH_LIST to 8.6 for this RTX 3080 host.

  setup model
    Download VoxCPM2 into the gitignored ./models/VoxCPM2 directory by default.
    Override with MODEL_SOURCE, MODEL_REPO, MODEL_DIR, and MODEL_REVISION.
    Optional third arg: huggingface or modelscope.
    Defaults to ModelScope repo `OpenBMB/VoxCPM2`.

  setup huggingface
    Download the default Hugging Face repo `openbmb/VoxCPM2` into ./models/VoxCPM2.

  setup modelscope
    Download the default ModelScope repo `OpenBMB/VoxCPM2` into ./models/VoxCPM2.

  setup all
    Run setup env, then setup model.

  dev api
    Start the FastAPI service with GPU 0 and port 8010 by default.
    Override with NANOVLLM_SERVERPOOL_DEVICES and NANOVLLM_SERVICE_PORT.

  check docs
    Validate the lightweight docs routing contract.

  check quick
    Run the docs check plus Python syntax compilation.
EOF
}
