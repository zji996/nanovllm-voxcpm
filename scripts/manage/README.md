# Manage Scripts

This directory keeps the repository's thin command entrypoints.

- `main.sh`: command router used by `./manage.sh`
- `setup.sh`: environment and model download helpers, including `./manage.sh setup model`, `./manage.sh setup huggingface`, and `./manage.sh setup modelscope`
- `dev.sh`: local service entrypoints
- `check.sh`: verification entrypoints
- `common.sh`: shared defaults for model path, port, and GPU selection
