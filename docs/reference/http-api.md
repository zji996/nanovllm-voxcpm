# HTTP API Reference

这份文档面向直接接入 deployment 服务的调用方，目标是把接口约束、流式行为、上限字段和常见接入方式说清楚。

基础假设:

- 服务默认监听 `http://127.0.0.1:8020`
- `GET /ready` 返回 `200` 后，模型才算真正可用
- `POST /generate` 返回的是流式 MP3 字节流，不是 JSON

## 快速流程

最常见的接入顺序:

1. 调 `GET /ready` 确认服务已经加载完成
2. 调 `GET /info` 读取模型信息、上下文上限、默认生成步数
3. 如果需要 prompt/reference latents，先调 `POST /encode_latents`
4. 调 `POST /generate` 并按流式字节消费返回体

## 端点概览

### `GET /health`

用途:

- liveness probe

成功响应:

```json
{"status":"ok"}
```

### `GET /ready`

用途:

- readiness probe
- 只有模型真正 ready 后才会返回 `200`

失败时:

- `503 {"detail":"not ready"}`

### `GET /info`

用途:

- 查询当前实例的模型和运行时元信息

典型响应字段:

- `model.sample_rate`: 主要音频采样率
- `model.channels`: 当前固定为 `1`
- `model.feat_dim`
- `model.patch_size`
- `model.model_path`
- `model.configured_max_model_len`: 当前服务实例实际采用的上下文上限
- `model.model_max_length`: 模型 `config.json` 里的 `max_length`
- `model.max_position_embeddings`: 底层 LM 的位置编码上限
- `model.default_max_generate_length`: API 默认生成步数
- `model.approx_step_audio_seconds`: 单个 generation step 大约产生多少秒音频
- `model.approx_max_audio_seconds_no_prompt`: 零样本且不带额外 prompt 时的大致最长音频时长
- `mp3.bitrate_kbps`
- `mp3.quality`

示例:

```bash
curl http://127.0.0.1:8020/info
```

示例响应:

```json
{
  "model": {
    "sample_rate": 48000,
    "channels": 1,
    "feat_dim": 64,
    "patch_size": 4,
    "model_path": "/models/VoxCPM2",
    "configured_max_model_len": 8192,
    "model_max_length": 8192,
    "max_position_embeddings": 32768,
    "default_max_generate_length": 2000,
    "approx_step_audio_seconds": 0.16,
    "approx_max_audio_seconds_no_prompt": 1310.72
  },
  "lora": {
    "lora_uri": null,
    "lora_id": null,
    "cache_dir": "/var/cache/nanovllm",
    "loaded": false
  },
  "mp3": {
    "bitrate_kbps": 192,
    "quality": 2
  }
}
```

## `POST /encode_latents`

用途:

- 把一段完整音频文件预编码成 prompt/reference latents

请求体:

```json
{
  "wav_base64": "<base64 of full audio file bytes>",
  "wav_format": "wav"
}
```

说明:

- `wav_base64` 是整个音频文件的字节内容，不是 data URI
- `wav_format` 常见取值是 `wav`、`flac`、`mp3`

成功响应:

```json
{
  "prompt_latents_base64": "<base64 float32 bytes>",
  "feat_dim": 64,
  "latents_dtype": "float32",
  "sample_rate": 16000,
  "channels": 1
}
```

解码方式:

```python
import base64
import numpy as np

payload = ...  # /encode_latents response json
buf = base64.b64decode(payload["prompt_latents_base64"])
latents = np.frombuffer(buf, dtype=np.float32).reshape(-1, payload["feat_dim"])
```

## `POST /generate`

用途:

- 生成语音
- 返回体是流式 MP3

返回特征:

- `Content-Type: audio/mpeg`
- 通常会带 `Transfer-Encoding: chunked`
- 响应头里还会有:
  - `X-Audio-Sample-Rate`
  - `X-Audio-Channels`

### 三种 prompt 形式

零样本:

- 只传 `target_text`

WAV prompt:

- `prompt_wav_base64`
- `prompt_wav_format`
- `prompt_text`

Latents prompt:

- `prompt_latents_base64`
- `prompt_text`

### Reference audio

Reference audio 是独立于 prompt 的第二组条件输入，可选:

- `ref_audio_wav_base64 + ref_audio_wav_format`
- 或 `ref_audio_latents_base64`

它和 prompt 可以组合使用，但各自内部是互斥的。

### 请求字段

- `target_text`: 必填
- `prompt_wav_base64`: 可选
- `prompt_wav_format`: 可选
- `prompt_latents_base64`: 可选
- `prompt_text`: wav/latents prompt 时必填；zero-shot 时不要传
- `ref_audio_wav_base64`: 可选
- `ref_audio_wav_format`: 可选
- `ref_audio_latents_base64`: 可选
- `max_generate_length`: 可选，默认 `2000`
- `temperature`: 可选，默认 `1.0`
- `cfg_value`: 可选，默认 `1.5`

### 上限如何理解

`max_generate_length` 不是唯一上限。

真实可生成长度还受下面这个约束影响:

- `prompt_len + max_generate_length <= configured_max_model_len`

因此:

- 先看 `GET /info` 里的 `model.configured_max_model_len`
- 再结合 prompt 长度判断还能生成多少 step

### 最小 zero-shot 示例

```bash
curl -X POST http://127.0.0.1:8020/generate \
  -H 'Content-Type: application/json' \
  -o out.mp3 \
  -d '{
    "target_text": "Please speak clearly.",
    "max_generate_length": 200
  }'
```

### 带 prompt latents 的示例

```bash
curl -X POST http://127.0.0.1:8020/generate \
  -H 'Content-Type: application/json' \
  -o out.mp3 \
  -d '{
    "target_text": "Continue in the same voice.",
    "prompt_latents_base64": "<prompt latents>",
    "prompt_text": "Hello from the prompt audio.",
    "max_generate_length": 200
  }'
```

### Python 流式消费示例

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8020/generate",
    json={
        "target_text": "Please speak clearly.",
        "max_generate_length": 200,
    },
    stream=True,
)
resp.raise_for_status()

with open("out.mp3", "wb") as f:
    for chunk in resp.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
```

### Python 低延迟播放/转发思路

如果你不是想等完整文件落盘，而是想边收边播:

- 把 `iter_content(...)` 读到的 chunk 直接写给播放器 stdin
- 或者直接转发给下游 websocket / HTTP client

重点是:

- 不要假设响应一次性返回
- 不要按 JSON 解码
- 按字节流持续消费即可

## 错误约束

常见 `400`:

- `prompt_wav_* and prompt_latents_base64 are mutually exclusive`
- `ref_audio_wav_* and ref_audio_latents_base64 are mutually exclusive`
- `wav prompt requires prompt_wav_base64 + prompt_wav_format`
- `wav prompt requires non-empty prompt_text`
- `latents prompt requires non-empty prompt_text`
- `prompt_text is not allowed for zero-shot`
- `Invalid base64 in ...`
- `Invalid latent payload in ...`

常见 `503`:

- `Model server not ready`

常见 `500`:

- `server misconfigured: missing app.state.cfg`
- `Only mono is supported (...)`

## 推荐接入建议

- 上线前先调用一次 `GET /info`，把 `configured_max_model_len` 和 `approx_step_audio_seconds` 打进客户端日志
- 如果你需要稳定的时长控制，自己在客户端把 `max_generate_length` 和 prompt 长度一起算掉
- 如果你要做长音频托管，优先流式消费 `POST /generate`，不要等完整结果落盘后再处理
