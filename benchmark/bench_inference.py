"""Benchmark VoxCPM inference throughput/latency.

Run (recommended via uv):
  uv run python benchmark/bench_inference.py --model ~/VoxCPM1.5 --concurrency 4 --iters 5 --warmup 1

Notes:
- This repo is GPU-centric; CPU-only execution is not supported.
- Metrics are end-to-end (parent process wall time) and include IPC overhead.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch

DEFAULT_TEXT = "Hello world."


def _parse_devices(devices: str) -> list[int]:
    items = [x.strip() for x in devices.split(",") if x.strip()]
    if not items:
        return [0]
    return [int(x) for x in items]


def _maybe_read_sample_rate(model: str) -> int | None:
    model_path = os.path.expanduser(model)
    if not os.path.isdir(model_path):
        return None
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None

    audio_cfg = cfg.get("audio_vae_config")
    if isinstance(audio_cfg, dict):
        sr = audio_cfg.get("sample_rate")
        if isinstance(sr, int):
            return sr
    return None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if c == f:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _fmt_float(x: float | None) -> str:
    if x is None:
        return "n/a"
    if x != x:  # NaN
        return "nan"
    return f"{x:.4f}"


def _mean(xs: Iterable[float]) -> float:
    values = list(xs)
    return statistics.mean(values) if values else float("nan")


def _stdev(xs: Iterable[float]) -> float:
    values = list(xs)
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def _read_binary_file(path: str) -> bytes:
    with open(os.path.expanduser(path), "rb") as f:
        return f.read()


def _dedupe_failures(messages: Iterable[str], limit: int = 5) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        unique.append(message)
        if len(unique) >= limit:
            break
    return unique


@dataclass(frozen=True)
class ScenarioInputs:
    name: str
    prompt_latents: bytes | None
    prompt_text: str
    ref_audio_latents: bytes | None


@dataclass(frozen=True)
class OneRequestResult:
    total_samples: int
    num_chunks: int
    ttfb_s: float
    wall_s: float
    error: str | None = None


@dataclass(frozen=True)
class IterationResult:
    concurrency: int
    wall_s: float
    total_samples: int
    total_chunks: int
    failed_requests: int
    failure_messages: tuple[str, ...]
    ttfb_p50_s: float
    ttfb_p90_s: float
    ttfb_p95_s: float
    audio_s_p50: float | None
    audio_s_p90: float | None
    audio_s_p95: float | None
    audio_s_p99: float | None
    audio_s_mean: float | None
    audio_s_stdev: float | None
    audio_s_per_req_mean: float | None
    audio_seconds_per_second: float | None
    rtf_per_req_mean: float | None


@dataclass(frozen=True)
class GpuUtilSample:
    timestamp_s: float
    device_index: int
    utilization_gpu: float


async def _prepare_scenario_inputs(server_pool: Any, args: argparse.Namespace) -> ScenarioInputs:
    has_prompt_wav = args.prompt_wav_file is not None or args.prompt_wav_format is not None
    has_prompt_latents = args.prompt_latents_file is not None
    has_ref_wav = args.ref_audio_wav_file is not None or args.ref_audio_wav_format is not None
    has_ref_latents = args.ref_audio_latents_file is not None

    if has_prompt_wav and has_prompt_latents:
        raise ValueError("prompt wav and prompt latents are mutually exclusive")
    if has_ref_wav and has_ref_latents:
        raise ValueError("reference wav and reference latents are mutually exclusive")

    if has_prompt_wav and (args.prompt_wav_file is None or args.prompt_wav_format is None):
        raise ValueError("prompt wav requires --prompt-wav-file and --prompt-wav-format")
    if has_ref_wav and (args.ref_audio_wav_file is None or args.ref_audio_wav_format is None):
        raise ValueError("reference wav requires --ref-audio-wav-file and --ref-audio-wav-format")

    if (has_prompt_wav or has_prompt_latents) and not args.prompt_text:
        raise ValueError("prompt reuse requires non-empty --prompt-text")
    if args.prompt_text and not (has_prompt_wav or has_prompt_latents):
        raise ValueError("--prompt-text is only valid with --prompt-wav-file or --prompt-latents-file")

    prompt_latents: bytes | None = None
    ref_audio_latents: bytes | None = None

    if args.prompt_latents_file is not None:
        prompt_latents = _read_binary_file(args.prompt_latents_file)
    elif args.prompt_wav_file is not None:
        prompt_wav = _read_binary_file(args.prompt_wav_file)
        prompt_latents = await server_pool.encode_latents(prompt_wav, args.prompt_wav_format)

    if args.ref_audio_latents_file is not None:
        ref_audio_latents = _read_binary_file(args.ref_audio_latents_file)
    elif args.ref_audio_wav_file is not None:
        ref_wav = _read_binary_file(args.ref_audio_wav_file)
        ref_audio_latents = await server_pool.encode_latents(ref_wav, args.ref_audio_wav_format)

    scenario_parts: list[str] = []
    if prompt_latents is not None:
        scenario_parts.append("prompt-latents")
    if ref_audio_latents is not None:
        scenario_parts.append("reference-latents")
    if not scenario_parts:
        scenario_parts.append("zero-shot")

    return ScenarioInputs(
        name="+".join(scenario_parts),
        prompt_latents=prompt_latents,
        prompt_text=args.prompt_text,
        ref_audio_latents=ref_audio_latents,
    )


async def _consume_one(
    server: Any,
    *,
    target_text: str,
    scenario_inputs: ScenarioInputs,
    max_generate_length: int,
    temperature: float,
    cfg_value: float,
) -> OneRequestResult:
    start = time.perf_counter()
    first_chunk_t: float | None = None
    total_samples = 0
    num_chunks = 0

    try:
        generate_kwargs = {
            "target_text": target_text,
            "prompt_latents": scenario_inputs.prompt_latents,
            "prompt_text": scenario_inputs.prompt_text,
            "max_generate_length": max_generate_length,
            "temperature": temperature,
            "cfg_value": cfg_value,
        }
        if scenario_inputs.ref_audio_latents is not None:
            generate_kwargs["ref_audio_latents"] = scenario_inputs.ref_audio_latents

        async for chunk in server.generate(**generate_kwargs):
            if first_chunk_t is None:
                first_chunk_t = time.perf_counter()

            total_samples += int(chunk.shape[0])
            num_chunks += 1
    except Exception as exc:
        end = time.perf_counter()
        if first_chunk_t is None:
            first_chunk_t = end
        return OneRequestResult(
            total_samples=total_samples,
            num_chunks=num_chunks,
            ttfb_s=first_chunk_t - start,
            wall_s=end - start,
            error=f"{type(exc).__name__}: {exc}",
        )

    end = time.perf_counter()
    if first_chunk_t is None:
        first_chunk_t = end
    return OneRequestResult(
        total_samples=total_samples,
        num_chunks=num_chunks,
        ttfb_s=first_chunk_t - start,
        wall_s=end - start,
    )


async def _run_iteration(
    server_pool: Any,
    *,
    concurrency: int,
    target_text: str,
    scenario_inputs: ScenarioInputs,
    max_generate_length: int,
    temperature: float,
    cfg_value: float,
    sample_rate: int | None,
) -> IterationResult:
    start = time.perf_counter()
    tasks = [
        asyncio.create_task(
            _consume_one(
                server_pool,
                target_text=target_text,
                scenario_inputs=scenario_inputs,
                max_generate_length=max_generate_length,
                temperature=temperature,
                cfg_value=cfg_value,
            )
        )
        for _ in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)
    end = time.perf_counter()

    success_results = [result for result in results if result.error is None]
    failure_messages = tuple(_dedupe_failures(result.error for result in results if result.error is not None))

    ttfbs = [result.ttfb_s for result in success_results]

    audio_s_p50: float | None
    audio_s_p90: float | None
    audio_s_p95: float | None
    audio_s_p99: float | None
    audio_s_mean: float | None
    audio_s_stdev: float | None
    audio_s_per_req_mean: float | None
    audio_seconds_per_second: float | None
    rtf_per_req_mean: float | None
    if sample_rate is not None and sample_rate > 0 and success_results:
        audio_s_per_req = [result.total_samples / float(sample_rate) for result in success_results]
        rtfs = [result.wall_s / audio_s if audio_s > 0 else float("inf") for result, audio_s in zip(success_results, audio_s_per_req)]
        audio_s_total = sum(result.total_samples for result in results) / float(sample_rate)
        audio_s_per_req_mean = _mean(audio_s_per_req)
        audio_seconds_per_second = audio_s_total / (end - start) if end > start else float("nan")
        rtf_per_req_mean = _mean(rtfs)

        audio_s_p50 = _percentile(audio_s_per_req, 50)
        audio_s_p90 = _percentile(audio_s_per_req, 90)
        audio_s_p95 = _percentile(audio_s_per_req, 95)
        audio_s_p99 = _percentile(audio_s_per_req, 99)
        audio_s_mean = _mean(audio_s_per_req)
        audio_s_stdev = _stdev(audio_s_per_req)
    else:
        audio_s_p50 = None
        audio_s_p90 = None
        audio_s_p95 = None
        audio_s_p99 = None
        audio_s_mean = None
        audio_s_stdev = None
        audio_s_per_req_mean = None
        audio_seconds_per_second = None
        rtf_per_req_mean = None

    return IterationResult(
        concurrency=concurrency,
        wall_s=end - start,
        total_samples=sum(result.total_samples for result in results),
        total_chunks=sum(result.num_chunks for result in results),
        failed_requests=len(results) - len(success_results),
        failure_messages=failure_messages,
        ttfb_p50_s=_percentile(ttfbs, 50),
        ttfb_p90_s=_percentile(ttfbs, 90),
        ttfb_p95_s=_percentile(ttfbs, 95),
        audio_s_p50=audio_s_p50,
        audio_s_p90=audio_s_p90,
        audio_s_p95=audio_s_p95,
        audio_s_p99=audio_s_p99,
        audio_s_mean=audio_s_mean,
        audio_s_stdev=audio_s_stdev,
        audio_s_per_req_mean=audio_s_per_req_mean,
        audio_seconds_per_second=audio_seconds_per_second,
        rtf_per_req_mean=rtf_per_req_mean,
    )


def _query_gpu_util_samples(devices: list[int]) -> list[GpuUtilSample]:
    if shutil.which("nvidia-smi") is None:
        raise FileNotFoundError("nvidia-smi not found")

    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    selected_devices = set(devices)
    timestamp = time.perf_counter()
    samples: list[GpuUtilSample] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        device_index = int(parts[0])
        if device_index not in selected_devices:
            continue
        samples.append(
            GpuUtilSample(
                timestamp_s=timestamp,
                device_index=device_index,
                utilization_gpu=float(parts[1]),
            )
        )
    return samples


async def _sample_gpu_utilization(
    devices: list[int],
    interval_s: float,
    samples: list[GpuUtilSample],
    errors: list[str],
    stop_event: asyncio.Event,
) -> None:
    while True:
        try:
            samples.extend(await asyncio.to_thread(_query_gpu_util_samples, devices))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            return

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            return
        except asyncio.TimeoutError:
            continue


def _summarize_gpu_utilization(samples: list[GpuUtilSample]) -> dict[str, Any] | None:
    if not samples:
        return None

    overall = [sample.utilization_gpu for sample in samples]
    per_device: dict[str, dict[str, float]] = {}
    device_values: dict[int, list[float]] = {}
    for sample in samples:
        device_values.setdefault(sample.device_index, []).append(sample.utilization_gpu)

    for device_index, values in sorted(device_values.items()):
        per_device[str(device_index)] = {
            "mean": _mean(values),
            "p95": _percentile(values, 95),
            "max": max(values),
        }

    return {
        "overall": {
            "mean": _mean(overall),
            "p95": _percentile(overall, 95),
            "max": max(overall),
        },
        "per_device": per_device,
        "num_samples": len(samples),
    }


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark VoxCPM inference speed")
    parser.add_argument("--model", required=True, help="Local model directory (or HF repo id)")
    parser.add_argument(
        "--devices",
        default="0",
        help="Comma-separated CUDA device indices, e.g. '0' or '0,1'",
    )
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")

    parser.add_argument("--target-text", default=DEFAULT_TEXT)
    parser.add_argument("--target-text-file", default=None, help="Read target text from file (UTF-8)")
    parser.add_argument("--max-generate-length", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-value", type=float, default=2.0)

    parser.add_argument("--prompt-text", default="", help="Prompt text for prompt latents or prompt wav reuse")
    parser.add_argument("--prompt-latents-file", default=None, help="Raw float32 prompt latent bytes to reuse")
    parser.add_argument("--prompt-wav-file", default=None, help="Prompt audio file to encode once before the benchmark")
    parser.add_argument("--prompt-wav-format", default=None, help="Prompt audio format, e.g. wav/flac/mp3")
    parser.add_argument(
        "--ref-audio-latents-file",
        default=None,
        help="Raw float32 reference latent bytes to reuse",
    )
    parser.add_argument(
        "--ref-audio-wav-file",
        default=None,
        help="Reference audio file to encode once before the benchmark",
    )
    parser.add_argument(
        "--ref-audio-wav-format",
        default=None,
        help="Reference audio format, e.g. wav/flac/mp3",
    )

    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (not included in stats)")
    parser.add_argument("--iters", type=int, default=5, help="Measured iterations")
    parser.add_argument(
        "--gpu-sample-interval-ms",
        type=float,
        default=500.0,
        help="nvidia-smi sampling interval during measured iterations; <=0 disables GPU utilization sampling",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Override sample rate for RTF calc; otherwise best-effort from config.json (fallback: omit RTF)",
    )
    parser.add_argument("--json-out", default=None, help="Write results JSON to this path")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this project does not support CPU-only benchmarking")

    if args.target_text_file is not None:
        with open(args.target_text_file, encoding="utf-8") as f:
            args.target_text = f.read().strip()
        if not args.target_text:
            raise ValueError("target text is empty")

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iters <= 0:
        raise ValueError("--iters must be >= 1")

    sample_rate = args.sample_rate
    if sample_rate is None:
        sample_rate = _maybe_read_sample_rate(args.model)

    devices = _parse_devices(args.devices)

    from nanovllm_voxcpm import VoxCPM

    server_pool = VoxCPM.from_pretrained(
        model=args.model,
        inference_timesteps=args.inference_timesteps,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        devices=devices,
    )

    iterations: list[IterationResult] = []
    scenario_inputs: ScenarioInputs | None = None
    gpu_samples: list[GpuUtilSample] = []
    gpu_errors: list[str] = []
    gpu_stop_event = asyncio.Event()
    gpu_task: asyncio.Task[None] | None = None
    try:
        await server_pool.wait_for_ready()
        scenario_inputs = await _prepare_scenario_inputs(server_pool, args)

        if args.sample_rate is None:
            try:
                model_info = await server_pool.get_model_info()
                sample_rate = int(model_info["sample_rate"])
            except Exception:
                pass

        for _ in range(args.warmup):
            await _run_iteration(
                server_pool,
                concurrency=args.concurrency,
                target_text=args.target_text,
                scenario_inputs=scenario_inputs,
                max_generate_length=args.max_generate_length,
                temperature=args.temperature,
                cfg_value=args.cfg_value,
                sample_rate=sample_rate,
            )

        if args.gpu_sample_interval_ms > 0:
            gpu_task = asyncio.create_task(
                _sample_gpu_utilization(
                    devices=devices,
                    interval_s=args.gpu_sample_interval_ms / 1000.0,
                    samples=gpu_samples,
                    errors=gpu_errors,
                    stop_event=gpu_stop_event,
                )
            )

        for _ in range(args.iters):
            iterations.append(
                await _run_iteration(
                    server_pool,
                    concurrency=args.concurrency,
                    target_text=args.target_text,
                    scenario_inputs=scenario_inputs,
                    max_generate_length=args.max_generate_length,
                    temperature=args.temperature,
                    cfg_value=args.cfg_value,
                    sample_rate=sample_rate,
                )
            )
    finally:
        gpu_stop_event.set()
        if gpu_task is not None:
            with contextlib.suppress(Exception):
                await gpu_task
        await server_pool.stop()

    assert scenario_inputs is not None

    wall_s = [iteration.wall_s for iteration in iterations]
    total_samples = [iteration.total_samples for iteration in iterations]
    total_chunks = [iteration.total_chunks for iteration in iterations]
    ttfb_p50_s = [iteration.ttfb_p50_s for iteration in iterations]
    ttfb_p90_s = [iteration.ttfb_p90_s for iteration in iterations]
    ttfb_p95_s = [iteration.ttfb_p95_s for iteration in iterations]

    audio_s_p50 = [iteration.audio_s_p50 for iteration in iterations if iteration.audio_s_p50 is not None]
    audio_s_p90 = [iteration.audio_s_p90 for iteration in iterations if iteration.audio_s_p90 is not None]
    audio_s_p95 = [iteration.audio_s_p95 for iteration in iterations if iteration.audio_s_p95 is not None]
    audio_s_p99 = [iteration.audio_s_p99 for iteration in iterations if iteration.audio_s_p99 is not None]
    audio_s_mean = [iteration.audio_s_mean for iteration in iterations if iteration.audio_s_mean is not None]
    audio_s_stdev = [iteration.audio_s_stdev for iteration in iterations if iteration.audio_s_stdev is not None]
    audio_s_per_req_mean = [
        iteration.audio_s_per_req_mean for iteration in iterations if iteration.audio_s_per_req_mean is not None
    ]
    audio_seconds_per_second = [
        iteration.audio_seconds_per_second for iteration in iterations if iteration.audio_seconds_per_second is not None
    ]
    rtf_per_req_mean = [iteration.rtf_per_req_mean for iteration in iterations if iteration.rtf_per_req_mean is not None]

    samples_per_s = [samples / wall for samples, wall in zip(total_samples, wall_s)]
    chunks_per_s = [chunks / wall for chunks, wall in zip(total_chunks, wall_s)]
    failed_requests_total = sum(iteration.failed_requests for iteration in iterations)
    failure_notes = _dedupe_failures(
        message
        for iteration in iterations
        for message in iteration.failure_messages
    )

    audio_s_total: list[float] | None
    if sample_rate is not None and sample_rate > 0:
        audio_s_total = [samples / float(sample_rate) for samples in total_samples]
    else:
        audio_s_total = None

    gpu_summary = _summarize_gpu_utilization(gpu_samples)

    print("Benchmark finished")
    print(f"  model: {args.model}")
    print(f"  devices: {devices}")
    print(f"  scenario: {scenario_inputs.name}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  iters: {args.iters} (warmup {args.warmup})")
    print(f"  queue_coalesce_ms: {os.environ.get('NANOVLLM_QUEUE_COALESCE_MS', '2')}")
    print(f"  recv_queue_mode: {os.environ.get('NANOVLLM_RECV_QUEUE_MODE', 'bridge')}")
    if sample_rate is not None:
        print(f"  sample_rate: {sample_rate}")
    else:
        print("  sample_rate: unknown (RTF omitted)")
    print("Metrics (mean +/- stdev over measured iterations)")
    print(f"  wall_s: {_fmt_float(_mean(wall_s))} +/- {_fmt_float(_stdev(wall_s))}")
    if audio_s_total is not None:
        print(f"  audio_s_total: {_fmt_float(_mean(audio_s_total))} +/- {_fmt_float(_stdev(audio_s_total))}")
    if audio_s_per_req_mean:
        print(
            "  audio_s_per_req_mean: "
            f"{_fmt_float(_mean(audio_s_per_req_mean))} +/- {_fmt_float(_stdev(audio_s_per_req_mean))}"
        )
    if audio_seconds_per_second:
        print(
            "  audio_seconds_per_second: "
            f"{_fmt_float(_mean(audio_seconds_per_second))} +/- {_fmt_float(_stdev(audio_seconds_per_second))}"
        )
    if audio_s_p50:
        print("  audio_s_per_req_dist (seconds):")
        print(f"    p50: {_fmt_float(_mean(audio_s_p50))} +/- {_fmt_float(_stdev(audio_s_p50))}")
        print(f"    p90: {_fmt_float(_mean(audio_s_p90))} +/- {_fmt_float(_stdev(audio_s_p90))}")
        print(f"    p95: {_fmt_float(_mean(audio_s_p95))} +/- {_fmt_float(_stdev(audio_s_p95))}")
        print(f"    p99: {_fmt_float(_mean(audio_s_p99))} +/- {_fmt_float(_stdev(audio_s_p99))}")
        print(f"    mean +/- stdev: {_fmt_float(_mean(audio_s_mean))} +/- {_fmt_float(_mean(audio_s_stdev))}")
    if rtf_per_req_mean:
        print(f"  RTF_per_req_mean: {_fmt_float(_mean(rtf_per_req_mean))} +/- {_fmt_float(_stdev(rtf_per_req_mean))}")
    print(f"  samples/s: {_fmt_float(_mean(samples_per_s))} +/- {_fmt_float(_stdev(samples_per_s))}")
    print(f"  chunks/s: {_fmt_float(_mean(chunks_per_s))} +/- {_fmt_float(_stdev(chunks_per_s))}")
    print(f"  TTFB p50 (s): {_fmt_float(_mean(ttfb_p50_s))} +/- {_fmt_float(_stdev(ttfb_p50_s))}")
    print(f"  TTFB p90 (s): {_fmt_float(_mean(ttfb_p90_s))} +/- {_fmt_float(_stdev(ttfb_p90_s))}")
    print(f"  TTFB p95 (s): {_fmt_float(_mean(ttfb_p95_s))} +/- {_fmt_float(_stdev(ttfb_p95_s))}")
    print(f"  failed_requests_total: {failed_requests_total}")
    if failure_notes:
        print("  failure_notes:")
        for note in failure_notes:
            print(f"    - {note}")
    if gpu_summary is not None:
        overall = gpu_summary["overall"]
        print(
            "  GPU util (%): "
            f"mean={_fmt_float(overall['mean'])}, p95={_fmt_float(overall['p95'])}, max={_fmt_float(overall['max'])}"
        )
    elif gpu_errors:
        print(f"  GPU util: unavailable ({gpu_errors[0]})")

    payload: dict[str, Any] = {
        "args": vars(args),
        "devices": devices,
        "sample_rate": sample_rate,
        "scenario": scenario_inputs.name,
        "runtime_env": {
            "queue_coalesce_ms": os.environ.get("NANOVLLM_QUEUE_COALESCE_MS", "2"),
            "recv_queue_mode": os.environ.get("NANOVLLM_RECV_QUEUE_MODE", "bridge"),
        },
        "iterations": [asdict(iteration) for iteration in iterations],
        "summary": {
            "wall_s_mean": _mean(wall_s),
            "wall_s_stdev": _stdev(wall_s),
            "samples_per_s_mean": _mean(samples_per_s),
            "samples_per_s_stdev": _stdev(samples_per_s),
            "chunks_per_s_mean": _mean(chunks_per_s),
            "chunks_per_s_stdev": _stdev(chunks_per_s),
            "ttfb_p50_s_mean": _mean(ttfb_p50_s),
            "ttfb_p50_s_stdev": _stdev(ttfb_p50_s),
            "ttfb_p90_s_mean": _mean(ttfb_p90_s),
            "ttfb_p90_s_stdev": _stdev(ttfb_p90_s),
            "ttfb_p95_s_mean": _mean(ttfb_p95_s),
            "ttfb_p95_s_stdev": _stdev(ttfb_p95_s),
            "failed_requests_total": failed_requests_total,
            "failure_notes": failure_notes,
        },
    }
    if audio_s_total is not None:
        payload["summary"].update(
            {
                "audio_s_total_mean": _mean(audio_s_total),
                "audio_s_total_stdev": _stdev(audio_s_total),
            }
        )
    if audio_s_p50:
        payload["summary"].update(
            {
                "audio_s_p50_mean": _mean(audio_s_p50),
                "audio_s_p50_stdev": _stdev(audio_s_p50),
                "audio_s_p90_mean": _mean(audio_s_p90),
                "audio_s_p90_stdev": _stdev(audio_s_p90),
                "audio_s_p95_mean": _mean(audio_s_p95),
                "audio_s_p95_stdev": _stdev(audio_s_p95),
                "audio_s_p99_mean": _mean(audio_s_p99),
                "audio_s_p99_stdev": _stdev(audio_s_p99),
                "audio_s_mean_mean": _mean(audio_s_mean),
                "audio_s_mean_stdev": _stdev(audio_s_mean),
                "audio_s_stdev_mean": _mean(audio_s_stdev),
                "audio_s_stdev_stdev": _stdev(audio_s_stdev),
            }
        )
    if audio_s_per_req_mean:
        payload["summary"].update(
            {
                "audio_s_per_req_mean_mean": _mean(audio_s_per_req_mean),
                "audio_s_per_req_mean_stdev": _stdev(audio_s_per_req_mean),
            }
        )
    if audio_seconds_per_second:
        payload["summary"].update(
            {
                "audio_seconds_per_second_mean": _mean(audio_seconds_per_second),
                "audio_seconds_per_second_stdev": _stdev(audio_seconds_per_second),
            }
        )
    if rtf_per_req_mean:
        payload["summary"].update(
            {
                "rtf_per_req_mean_mean": _mean(rtf_per_req_mean),
                "rtf_per_req_mean_stdev": _stdev(rtf_per_req_mean),
            }
        )
    if gpu_summary is not None:
        payload["gpu_utilization"] = gpu_summary
        payload["summary"].update(
            {
                "gpu_util_mean": gpu_summary["overall"]["mean"],
                "gpu_util_p95": gpu_summary["overall"]["p95"],
                "gpu_util_max": gpu_summary["overall"]["max"],
            }
        )
    if gpu_errors:
        payload["gpu_utilization_errors"] = gpu_errors

    if args.json_out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
