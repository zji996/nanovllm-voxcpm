# HTTP API Reference

这份文档面向直接接入 deployment 服务的调用方，目标是把接口约束、流式行为、上限字段和常见接入方式说清楚。

文中的 `/info` 示例基于当前仓库默认 `compose.yaml` 配置；默认实例加载的是 `/models/VoxCPM2`。

基础假设:

- 服务默认监听 `http://127.0.0.1:8020`
- `GET /ready` 返回 `200` 后，模型才算真正可用
- `POST /generate` 返回的是流式 MP3 字节流，不是 JSON

## 快速流程

最常见的接入顺序:

1. 调 `GET /ready` 确认服务已经加载完成
2. 调 `GET /info` 读取模型信息、上下文上限和默认生成步数
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

当前默认实例的典型响应:

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

当前默认实例补充说明:

- 当前默认实例的关键值是: `sample_rate=48000`、`channels=1`、`feat_dim=64`、`patch_size=4`
- LoRA 当前未加载，`mp3` 编码参数是 `192 kbps` / `quality=2`
- 按当前仓库里的 [`models/VoxCPM2/config.json`](/home/zji/docker/nanovllm-voxcpm/models/VoxCPM2/config.json)，模型静态配置是 `max_length=8192`、`lm_config.max_position_embeddings=32768`
- 按当前 [`compose.yaml`](/home/zji/docker/nanovllm-voxcpm/compose.yaml) 和 [`deployment/app/core/config.py`](/home/zji/docker/nanovllm-voxcpm/deployment/app/core/config.py)，服务没有显式设置 `NANOVLLM_SERVERPOOL_MAX_MODEL_LEN` 时，会从模型 `config.json` 自动探测默认值；对当前模型默认就是 `8192`

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
- 响应头里还会有 `X-Audio-Sample-Rate` 和 `X-Audio-Channels`
- `2026-04-17` 的实测 zero-shot 请求返回了 `X-Audio-Sample-Rate: 48000`、`X-Audio-Channels: 1`
- 实测落盘文件可被识别为 `MPEG ADTS, layer III, 192 kbps, 48 kHz, Monaural`

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

运行时真实可生成长度仍然受下面这个约束影响:

- `prompt_len + max_generate_length <= configured_max_model_len`

对当前默认 VoxCPM2 实例:

- `POST /generate` 的请求体默认值依然是 `max_generate_length=2000`
- 默认 `configured_max_model_len` 是 `8192`
- 因此剩余生成空间可以近似按 `8192 - prompt_len` 估算

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

## 生产应用推荐设计

如果你要基于这套 HTTP API 设计一个生产音频应用，推荐把能力分成两层:

- API 层只负责通用原子能力: `encode_latents`、`generate`、streaming MP3、基础元信息查询
- 应用层负责会话编排: 用户音色资产、续轮策略、失败重试、缓存、权限和审计

原因:

- `prompt` / `reference audio` 的使用策略高度依赖具体产品目标，不同业务对稳定性、时长、延迟、成本的权衡不一样
- 如果把“续音色会话”直接做成服务端隐式状态，后续会引入会话生命周期、跨实例共享、缓存淘汰和隐私边界问题
- 当前 API 已经提供了足够的基础拼装能力，应用层可以在不改推理服务的前提下快速迭代策略

### 推荐职责划分

更推荐的边界是:

- deployment API 保持无状态或尽量少状态
- 应用服务保存“voice session”元数据
- 应用服务决定每次请求到底传 zero-shot、prompt 还是 reference
- 应用服务缓存 latents，而不是每轮都重复上传大段 WAV

不太推荐一开始就在 deployment API 里直接做:

- `session_id -> 自动续音色`
- `voice_profile_id -> 服务端隐式拼 prompt/reference`
- “自动从上一轮生成结果里截一段再回灌”的黑盒逻辑

这些做法并不是不能做，而是更适合作为第二阶段封装，而不是第一版基础 API。

### 推荐调用模式

首轮建声:

- 如果你只有文本，没有参考音频，用 zero-shot: 只传 `target_text`
- 如果你有一段稳定的目标音色样本，优先把它作为 `prompt` 使用
- 对于频繁复用的音色样本，先调用 `POST /encode_latents`，把返回的 `prompt_latents_base64` 缓存在应用侧

续轮生成:

- 不要只依赖首轮 prompt 反复开新会话；跨轮次时音色可能会漂
- 更稳的做法是带一点“历史上已经生成成功的音频”作为 reference
- 推荐应用侧在每轮生成完成后，保留一小段你认为音色最稳定的音频，编码成 latents，作为下一轮的 `ref_audio_latents_base64`
- 如果同时需要说话人音色和强文本上下文，可以继续传 `prompt_*`，再叠加 `ref_audio_*`

推荐优先级可以简单理解为:

1. 零样本: 最简单，但音色连续性最弱
2. 固定 prompt: 能定一个基准音色，但多轮重开会话时稳定性一般
3. prompt + reference: 更适合做生产里的多轮续写、分段生成和长音频拼接

### 推荐状态模型

应用侧至少维护这些字段:

- `voice_profile_id`: 业务上的说话人或音色模板 ID
- `prompt_text`: 和 prompt 音频对应的文本
- `prompt_latents_base64`: 长期稳定复用的 prompt latents
- `last_good_ref_latents_base64`: 最近一次你认为质量合格的 reference latents
- `model_path` 或模型版本号: 避免不同模型之间混用缓存 latents
- 可选的质量标签: 比如人工确认、自动评分、最近更新时间

推荐缓存策略:

- prompt latents 长期缓存
- reference latents 短期缓存，并允许被更新
- 模型切换后重新编码，不要假设跨模型兼容

### 推荐请求策略

一个更适合生产的简单策略是:

1. 首次生成时:
   传 `prompt_latents_base64 + prompt_text`
2. 首次生成成功后:
   从结果里挑一小段稳定音频，调用 `POST /encode_latents`
3. 后续生成时:
   传 `prompt_latents_base64 + prompt_text + ref_audio_latents_base64`
4. 如果某轮 reference 效果明显变差:
   回退到上一个可用 reference，或者只保留 prompt

应用层最好自己做这些保护:

- 对 reference 音频做长度和质量筛选，不要无脑拿整段结果回灌
- 不要把明显异常、噪声重、断句差的输出直接升级成新的 reference
- 为每轮生成记录输入参数和产物，便于回溯哪种组合最稳

### 推荐工程实践

如果你是做真正的生产应用，建议再加上这些外围能力:

- 在应用网关层做鉴权，不要把当前 deployment 服务直接裸露到公网
- 对 `prompt_latents_base64` / `ref_audio_latents_base64` 做对象存储或数据库缓存，避免重复 base64 传输大音频
- 对长文本先做分段，再逐段生成，并在段间复用 reference
- 为生成任务加超时、重试和幂等键
- 把 `GET /info` 的关键字段和每次生成参数打进日志，方便定位模型切换或运行时差异

### 一个实用的落地结论

如果你的目标是“尽量稳定地延续前一轮的音色”，推荐方案是:

- API 层继续只提供 `prompt` / `reference` / `latents` 这些基础能力
- 应用层决定什么时候把上一轮生成结果的一小段转成新的 reference latents
- 默认优先传 latents，而不是每轮都传原始 WAV

也就是说，“带一点生成出来的音频让下一轮更稳”这件事，更推荐在应用侧实现，而不是先把策略硬编码进 deployment API。

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

- 上线前先调用一次 `GET /info`，把 `model_path`、`sample_rate`、`feat_dim`、`patch_size`、`configured_max_model_len` 打进客户端日志
- 如果你需要稳定的时长控制，自己在客户端把 `max_generate_length` 和 prompt 长度一起算掉；对当前默认实例可按 `8192` 做估算
- 如果你要做长音频托管，优先流式消费 `POST /generate`，不要等完整结果落盘后再处理
- 如果你要做多轮稳定续音色，把“reference 选取和缓存策略”放在应用侧，而不是一开始就压进 deployment API
