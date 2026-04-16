from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_server
from app.schemas.http import ErrorResponse, InfoResponse, LoRAInfo, ModelInfo, Mp3Info

router = APIRouter(tags=["info"])


@router.get(
    "/info",
    response_model=InfoResponse,
    summary="Get model and service metadata",
    responses={
        503: {
            "description": "Model server not ready",
            "model": ErrorResponse,
        }
    },
)
async def info(request: Request, server: Any = Depends(get_server)) -> InfoResponse:
    """Return model metadata and instance-level configuration."""

    cfg = getattr(request.app.state, "cfg", None)
    model_info = await server.get_model_info()

    lora_state = getattr(request.app.state, "lora", {})
    return InfoResponse(
        model=ModelInfo(
            sample_rate=int(model_info["sample_rate"]),
            channels=int(model_info["channels"]),
            feat_dim=int(model_info["feat_dim"]),
            patch_size=int(model_info["patch_size"]),
            model_path=str(model_info["model_path"]),
            configured_max_model_len=(
                int(model_info["configured_max_model_len"]) if "configured_max_model_len" in model_info else None
            ),
            model_max_length=(int(model_info["model_max_length"]) if "model_max_length" in model_info else None),
            max_position_embeddings=(
                int(model_info["max_position_embeddings"]) if "max_position_embeddings" in model_info else None
            ),
            default_max_generate_length=(
                int(model_info["default_max_generate_length"]) if "default_max_generate_length" in model_info else None
            ),
            approx_step_audio_seconds=(
                float(model_info["approx_step_audio_seconds"]) if "approx_step_audio_seconds" in model_info else None
            ),
            approx_max_audio_seconds_no_prompt=(
                float(model_info["approx_max_audio_seconds_no_prompt"])
                if "approx_max_audio_seconds_no_prompt" in model_info
                else None
            ),
        ),
        lora=LoRAInfo(
            lora_uri=lora_state.get("lora_uri"),
            lora_id=lora_state.get("lora_id"),
            cache_dir=lora_state.get("cache_dir"),
            loaded=bool(lora_state.get("loaded", False)),
        ),
        mp3=Mp3Info(
            bitrate_kbps=getattr(getattr(cfg, "mp3", None), "bitrate_kbps", None),
            quality=getattr(getattr(cfg, "mp3", None), "quality", None),
        ),
    )
