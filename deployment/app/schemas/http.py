from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Standard health response for liveness/readiness endpoints."""

    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    """Default error response shape produced by FastAPI for HTTPException.

    Note: FastAPI's validation errors (422) use a different schema.
    """

    detail: str = Field(..., description="Human-readable error message.")


class ModelInfo(BaseModel):
    """Read-only model metadata returned by the engine."""

    sample_rate: int = Field(..., description="Audio sample rate in Hz.", examples=[16000])
    channels: int = Field(..., description="Number of audio channels.", examples=[1])
    feat_dim: int = Field(..., description="Latent feature dimension.", examples=[64])
    patch_size: int = Field(..., description="Model patch size.", examples=[2])
    model_path: str = Field(
        ...,
        description="Resolved model path used by this instance.",
        examples=["/models/VoxCPM1.5"],
    )
    configured_max_model_len: int | None = Field(
        None,
        description="Runtime max context length configured for this service instance.",
        examples=[8192],
    )
    model_max_length: int | None = Field(
        None,
        description="Model-level max_length from config.json when available.",
        examples=[8192],
    )
    max_position_embeddings: int | None = Field(
        None,
        description="Underlying LM max_position_embeddings when available.",
        examples=[32768],
    )
    default_max_generate_length: int | None = Field(
        None,
        description="Default max_generate_length used by the API when the caller omits it.",
        examples=[2000],
    )
    approx_step_audio_seconds: float | None = Field(
        None,
        description="Approximate audio seconds produced by one generation step.",
        examples=[0.16],
    )
    approx_max_audio_seconds_no_prompt: float | None = Field(
        None,
        description="Approximate upper bound for zero-shot audio duration at the configured max_model_len.",
        examples=[1310.72],
    )


class Mp3Info(BaseModel):
    """MP3 encoder configuration used by /generate."""

    bitrate_kbps: int | None = Field(None, description="Constant bitrate used for MP3 encoding.", examples=[192])
    quality: int | None = Field(None, description="LAME quality preset (0 is best, 2 is fast).", examples=[2])


class LoRAInfo(BaseModel):
    """LoRA startup configuration and load status."""

    lora_uri: str | None = Field(
        None,
        description="LoRA artifact URI (if configured).",
        examples=["hf://org/repo@main?path=step_0002000"],
    )
    lora_id: str | None = Field(
        None,
        description="Logical LoRA ID associated with this instance.",
        examples=["my-lora"],
    )
    cache_dir: str | None = Field(None, description="Cache directory used to resolve LoRA artifacts.")
    loaded: bool = Field(
        ...,
        description="Whether LoRA weights have been loaded and enabled.",
        examples=[False],
    )


class InfoResponse(BaseModel):
    """Response for GET /info."""

    model: ModelInfo
    lora: LoRAInfo
    mp3: Mp3Info


class EncodeLatentsRequest(BaseModel):
    """Request body for POST /encode_latents."""

    wav_base64: str = Field(
        ...,
        description="Base64-encoded audio file bytes (entire file contents). Do not include a data URI prefix.",
        examples=["UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."],
    )
    wav_format: str = Field(
        ...,
        description="Audio container format for decoding (e.g. 'wav', 'flac', 'mp3'); passed to torchaudio.",
        examples=["wav"],
    )


class EncodeLatentsResponse(BaseModel):
    """Response body for POST /encode_latents."""

    prompt_latents_base64: str
    feat_dim: int
    latents_dtype: Literal["float32"] = "float32"
    sample_rate: int
    channels: int


class GenerateRequest(BaseModel):
    """Request body for POST /generate.

    Prompt forms (mutually exclusive):

    - Zero-shot: omit all prompt_* fields.
    - WAV prompt: set prompt_wav_base64 + prompt_wav_format + prompt_text.
    - Latents prompt: set prompt_latents_base64 + prompt_text.

    Reference audio (optional, mutually exclusive within the ref_audio_* group):

    - WAV reference: set ref_audio_wav_base64 + ref_audio_wav_format.
    - Latents reference: set ref_audio_latents_base64.
    """

    target_text: str = Field(..., description="Text to synthesize.")

    # Prompt forms (mutually exclusive):
    prompt_wav_base64: str | None = Field(
        None,
        description="(wav prompt) Base64-encoded audio file bytes (entire file contents).",
    )
    prompt_wav_format: str | None = Field(
        None,
        description="(wav prompt) Audio container format for decoding (e.g. 'wav', 'flac', 'mp3').",
    )
    prompt_latents_base64: str | None = Field(
        None,
        description="(latents prompt) Base64-encoded float32 bytes returned by /encode_latents.",
    )
    prompt_text: str | None = Field(
        None,
        description="Prompt transcript text. Required for wav/latents prompt; omitted for zero-shot.",
    )

    ref_audio_wav_base64: str | None = Field(
        None,
        description="(reference audio) Base64-encoded audio file bytes (entire file contents).",
    )
    ref_audio_wav_format: str | None = Field(
        None,
        description="(reference audio) Audio container format for decoding (e.g. 'wav', 'flac', 'mp3').",
    )
    ref_audio_latents_base64: str | None = Field(
        None,
        description="(reference audio) Base64-encoded float32 bytes returned by /encode_latents.",
    )

    max_generate_length: int = Field(
        2000,
        ge=1,
        description=(
            "Maximum number of model generation steps. "
            "The effective upper bound is still limited by the runtime max_model_len "
            "and current prompt length; see GET /info for the configured limit."
        ),
    )
    temperature: float = Field(1.0, ge=0.0, description="Sampling temperature.")
    cfg_value: float = Field(1.5, ge=0.0, description="Classifier-free guidance scale.")
