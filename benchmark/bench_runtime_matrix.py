"""Run a reproducible runtime benchmark matrix for VoxCPM server settings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _slugify(value: str) -> str:
    chars: list[str] = []
    for ch in value:
        if ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        if value != value:
            return "nan"
        return f"{float(value):.4f}"
    return str(value)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    cli_args: tuple[str, ...]


def _has_prompt_inputs(args: argparse.Namespace) -> bool:
    return args.prompt_latents_file is not None or args.prompt_wav_file is not None


def _has_reference_inputs(args: argparse.Namespace) -> bool:
    return args.ref_audio_latents_file is not None or args.ref_audio_wav_file is not None


def _build_scenario_specs(args: argparse.Namespace) -> list[ScenarioSpec]:
    specs: list[ScenarioSpec] = []
    for scenario_name in _parse_csv_strings(args.scenarios):
        if scenario_name == "zero-shot":
            specs.append(ScenarioSpec(name="zero-shot", cli_args=()))
            continue

        if scenario_name == "prompt-latents":
            if not _has_prompt_inputs(args):
                raise ValueError("scenario prompt-latents requires prompt latents or prompt wav inputs")
            if not args.prompt_text:
                raise ValueError("scenario prompt-latents requires non-empty --prompt-text")
            cli_args: list[str] = ["--prompt-text", args.prompt_text]
            if args.prompt_latents_file is not None:
                cli_args.extend(["--prompt-latents-file", args.prompt_latents_file])
            else:
                assert args.prompt_wav_file is not None
                assert args.prompt_wav_format is not None
                cli_args.extend(["--prompt-wav-file", args.prompt_wav_file, "--prompt-wav-format", args.prompt_wav_format])
            specs.append(ScenarioSpec(name=scenario_name, cli_args=tuple(cli_args)))
            continue

        if scenario_name == "reference-latents":
            if not _has_reference_inputs(args):
                raise ValueError("scenario reference-latents requires reference latents or reference wav inputs")
            cli_args = []
            if args.ref_audio_latents_file is not None:
                cli_args.extend(["--ref-audio-latents-file", args.ref_audio_latents_file])
            else:
                assert args.ref_audio_wav_file is not None
                assert args.ref_audio_wav_format is not None
                cli_args.extend(
                    [
                        "--ref-audio-wav-file",
                        args.ref_audio_wav_file,
                        "--ref-audio-wav-format",
                        args.ref_audio_wav_format,
                    ]
                )
            specs.append(ScenarioSpec(name=scenario_name, cli_args=tuple(cli_args)))
            continue

        if scenario_name == "prompt+reference":
            if not _has_prompt_inputs(args) or not _has_reference_inputs(args):
                raise ValueError("scenario prompt+reference requires both prompt and reference inputs")
            prompt_spec = _build_scenario_specs(
                argparse.Namespace(
                    scenarios="prompt-latents",
                    prompt_text=args.prompt_text,
                    prompt_latents_file=args.prompt_latents_file,
                    prompt_wav_file=args.prompt_wav_file,
                    prompt_wav_format=args.prompt_wav_format,
                    ref_audio_latents_file=args.ref_audio_latents_file,
                    ref_audio_wav_file=args.ref_audio_wav_file,
                    ref_audio_wav_format=args.ref_audio_wav_format,
                )
            )[0]
            reference_spec = _build_scenario_specs(
                argparse.Namespace(
                    scenarios="reference-latents",
                    prompt_text=args.prompt_text,
                    prompt_latents_file=args.prompt_latents_file,
                    prompt_wav_file=args.prompt_wav_file,
                    prompt_wav_format=args.prompt_wav_format,
                    ref_audio_latents_file=args.ref_audio_latents_file,
                    ref_audio_wav_file=args.ref_audio_wav_file,
                    ref_audio_wav_format=args.ref_audio_wav_format,
                )
            )[0]
            specs.append(
                ScenarioSpec(
                    name=scenario_name,
                    cli_args=prompt_spec.cli_args + reference_spec.cli_args,
                )
            )
            continue

        raise ValueError(f"unsupported scenario: {scenario_name}")
    return specs


def _build_bench_command(
    args: argparse.Namespace,
    model: str,
    concurrency: int,
    scenario: ScenarioSpec,
    json_out: str,
) -> list[str]:
    script_path = Path(__file__).with_name("bench_inference.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--model",
        model,
        "--devices",
        args.devices,
        "--inference-timesteps",
        str(args.inference_timesteps),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--concurrency",
        str(concurrency),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--max-generate-length",
        str(args.max_generate_length),
        "--temperature",
        str(args.temperature),
        "--cfg-value",
        str(args.cfg_value),
        "--gpu-sample-interval-ms",
        str(args.gpu_sample_interval_ms),
        "--json-out",
        json_out,
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.sample_rate is not None:
        cmd.extend(["--sample-rate", str(args.sample_rate)])
    if args.target_text_file is not None:
        cmd.extend(["--target-text-file", args.target_text_file])
    else:
        cmd.extend(["--target-text", args.target_text])
    cmd.extend(scenario.cli_args)
    return cmd


def _extract_summary(record: dict[str, Any]) -> dict[str, Any]:
    summary = record.get("summary", {})
    gpu = record.get("gpu_utilization", {}).get("overall", {})
    return {
        "wall_s_mean": summary.get("wall_s_mean"),
        "audio_seconds_per_second_mean": summary.get("audio_seconds_per_second_mean"),
        "ttfb_p95_s_mean": summary.get("ttfb_p95_s_mean"),
        "gpu_util_mean": summary.get("gpu_util_mean", gpu.get("mean")),
        "failed_requests_total": summary.get("failed_requests_total"),
        "failure_notes": summary.get("failure_notes", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a runtime benchmark matrix for VoxCPM")
    parser.add_argument("--model", action="append", required=True, help="Repeat for each local model directory or repo id")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")

    parser.add_argument("--target-text", default="Hello world.")
    parser.add_argument("--target-text-file", default=None)
    parser.add_argument("--max-generate-length", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=None)

    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--prompt-latents-file", default=None)
    parser.add_argument("--prompt-wav-file", default=None)
    parser.add_argument("--prompt-wav-format", default=None)
    parser.add_argument("--ref-audio-latents-file", default=None)
    parser.add_argument("--ref-audio-wav-file", default=None)
    parser.add_argument("--ref-audio-wav-format", default=None)
    parser.add_argument(
        "--scenarios",
        default="zero-shot",
        help="Comma-separated: zero-shot,prompt-latents,reference-latents,prompt+reference",
    )

    parser.add_argument("--concurrency-values", default="1,2,4")
    parser.add_argument("--queue-coalesce-values", default="0,1,2,5")
    parser.add_argument("--recv-queue-modes", default="bridge,to_thread")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--gpu-sample-interval-ms", type=float, default=500.0)
    parser.add_argument("--out-dir", default=None, help="Directory for per-run JSON outputs and matrix summary")
    args = parser.parse_args(argv)

    scenario_specs = _build_scenario_specs(args)
    concurrency_values = _parse_csv_ints(args.concurrency_values)
    queue_coalesce_values = _parse_csv_strings(args.queue_coalesce_values)
    recv_queue_modes = _parse_csv_strings(args.recv_queue_modes)

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_dir = Path(args.out_dir or f"benchmark/results/runtime-matrix-{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    total_runs = len(args.model) * len(concurrency_values) * len(queue_coalesce_values) * len(recv_queue_modes) * len(
        scenario_specs
    )
    run_index = 0

    for model in args.model:
        for concurrency in concurrency_values:
            for scenario in scenario_specs:
                for recv_queue_mode in recv_queue_modes:
                    for queue_coalesce_ms in queue_coalesce_values:
                        run_index += 1
                        run_slug = (
                            f"{_slugify(model)}-c{concurrency}-{_slugify(scenario.name)}-"
                            f"{_slugify(recv_queue_mode)}-q{_slugify(queue_coalesce_ms)}"
                        )
                        json_out = out_dir / f"{run_slug}.json"
                        cmd = _build_bench_command(args, model, concurrency, scenario, str(json_out))
                        env = os.environ.copy()
                        env["NANOVLLM_QUEUE_COALESCE_MS"] = str(queue_coalesce_ms)
                        env["NANOVLLM_RECV_QUEUE_MODE"] = recv_queue_mode

                        print(
                            f"[{run_index}/{total_runs}] model={model} concurrency={concurrency} "
                            f"scenario={scenario.name} recv={recv_queue_mode} coalesce={queue_coalesce_ms}"
                        )
                        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

                        record: dict[str, Any] = {
                            "model": model,
                            "concurrency": concurrency,
                            "scenario": scenario.name,
                            "recv_queue_mode": recv_queue_mode,
                            "queue_coalesce_ms": queue_coalesce_ms,
                            "command": cmd,
                            "returncode": proc.returncode,
                            "stdout_tail": proc.stdout[-2000:],
                            "stderr_tail": proc.stderr[-2000:],
                            "json_out": str(json_out),
                        }
                        if proc.returncode == 0 and json_out.is_file():
                            with open(json_out, encoding="utf-8") as f:
                                payload = json.load(f)
                            record["payload"] = payload
                            record["summary"] = _extract_summary(payload)
                        else:
                            record["summary"] = {
                                "wall_s_mean": None,
                                "audio_seconds_per_second_mean": None,
                                "ttfb_p95_s_mean": None,
                                "gpu_util_mean": None,
                                "failed_requests_total": None,
                                "failure_notes": ["subprocess failed"],
                            }

                        results.append(record)

    matrix_payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "results": results,
    }
    with open(out_dir / "matrix.json", "w", encoding="utf-8") as f:
        json.dump(matrix_payload, f, ensure_ascii=True, indent=2)

    print("Summary")
    print("model | conc | scenario | recv | coalesce | wall_s | audio_s/s | ttfb_p95 | gpu_util | failed")
    for record in results:
        summary = record["summary"]
        print(
            f"{record['model']} | {record['concurrency']} | {record['scenario']} | "
            f"{record['recv_queue_mode']} | {record['queue_coalesce_ms']} | "
            f"{_fmt_float(summary.get('wall_s_mean'))} | "
            f"{_fmt_float(summary.get('audio_seconds_per_second_mean'))} | "
            f"{_fmt_float(summary.get('ttfb_p95_s_mean'))} | "
            f"{_fmt_float(summary.get('gpu_util_mean'))} | "
            f"{summary.get('failed_requests_total')}"
        )

    print(f"Wrote matrix results to {out_dir / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
