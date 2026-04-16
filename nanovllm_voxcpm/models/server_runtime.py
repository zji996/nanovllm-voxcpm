from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time
import traceback
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from queue import Empty
from typing import Any, Generic, TypeVar

import numpy as np
import torch.multiprocessing as mp
from numpy.typing import NDArray

Waveform = NDArray[np.float32]


def gen_uuid() -> str:
    return uuid.uuid4().hex


def normalize_devices(devices: list[int] | None) -> list[int]:
    return [0] if devices is None else list(devices)


def resolve_recv_queue_mode() -> str:
    mode = os.environ.get("NANOVLLM_RECV_QUEUE_MODE", "bridge").strip().lower()
    return mode if mode in {"bridge", "to_thread"} else "bridge"


class _QueueBridgeThread:
    def __init__(
        self,
        source_queue: Any,
        loop: asyncio.AbstractEventLoop,
        target_queue: asyncio.Queue[dict[str, Any]],
        poll_timeout_s: float = 0.25,
    ) -> None:
        self._source_queue = source_queue
        self._loop = loop
        self._target_queue = target_queue
        self._poll_timeout_s = poll_timeout_s
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="nanovllm-queue-bridge")

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._source_queue.get(timeout=self._poll_timeout_s)
            except Empty:
                continue
            except (EOFError, OSError, ValueError):
                break

            try:
                self._loop.call_soon_threadsafe(self._target_queue.put_nowait, item)
            except RuntimeError:
                break

    async def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            await asyncio.to_thread(self._thread.join, self._poll_timeout_s + 1.0)


def run_server_main_loop(
    queue_in: mp.Queue,
    queue_out: mp.Queue,
    server_factory: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        coalesce_ms = float(os.environ.get("NANOVLLM_QUEUE_COALESCE_MS", "2"))
    except ValueError:
        coalesce_ms = 2.0
    if coalesce_ms > 0:
        coalesce_ms = min(coalesce_ms, 50.0)

    try:
        server = server_factory(*args, **kwargs)
    except Exception:
        queue_out.put({"type": "init_error", "error": traceback.format_exc()})
        return
    else:
        queue_out.put({"type": "init_ok"})

    state = {"is_stopped": False}

    def call_method(cmd: dict[str, Any]) -> dict[str, Any]:
        op_id = cmd.get("id", "")
        try:
            method_name = cmd["type"]
            call_args = cmd["args"]
            call_kwargs = cmd["kwargs"]
            if method_name == "stop":
                state["is_stopped"] = True
                return {"type": "response", "id": op_id, "data": None}

            result = getattr(server, method_name)(*call_args, **call_kwargs)
            return {"type": "response", "id": op_id, "data": result}
        except Exception:
            return {"type": "error", "id": op_id, "error": traceback.format_exc()}

    while not state["is_stopped"]:
        cmd = queue_in.get()
        queue_out.put(call_method(cmd))

        if coalesce_ms > 0:
            deadline = time.perf_counter() + (coalesce_ms / 1000.0)
            while not state["is_stopped"]:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    cmd = queue_in.get(timeout=remaining)
                except Empty:
                    break
                queue_out.put(call_method(cmd))

        while not server.is_finished() and not state["is_stopped"]:
            while not state["is_stopped"]:
                try:
                    cmd = queue_in.get_nowait()
                    queue_out.put(call_method(cmd))
                except Empty:
                    break

            if state["is_stopped"]:
                break

            output = server.step()
            for seq in output:
                latest_waveform = seq.custom_payload.generated_waveforms[-1]
                queue_out.put({"type": "stream", "id": seq.seq_id, "data": latest_waveform})
                if seq.is_finished:
                    queue_out.put({"type": "stream", "id": seq.seq_id, "data": None})


class AsyncServerProcess:
    def __init__(
        self,
        server_factory: Callable[..., Any],
        init_args: tuple[Any, ...],
        init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        ctx = mp.get_context("spawn")
        self.queue_in = ctx.Queue()
        self.queue_out = ctx.Queue()
        self.process = ctx.Process(
            target=run_server_main_loop,
            args=(self.queue_in, self.queue_out, server_factory, init_args, init_kwargs or {}),
            daemon=True,
        )
        self.process.start()

        loop = asyncio.get_running_loop()
        self._init_fut: asyncio.Future[None] = loop.create_future()
        self._recv_queue_mode = resolve_recv_queue_mode()
        self._recv_bridge_queue: asyncio.Queue[dict[str, Any]] | None = None
        self._recv_bridge: _QueueBridgeThread | None = None
        if self._recv_queue_mode == "bridge":
            self._recv_bridge_queue = asyncio.Queue()
            self._recv_bridge = _QueueBridgeThread(self.queue_out, loop, self._recv_bridge_queue)
            self._recv_bridge.start()
        self.recv_task: asyncio.Task[None] = asyncio.create_task(self.recv_queue())
        self.op_table: dict[str, asyncio.Future[Any]] = {}
        self.stream_table: dict[str, asyncio.Queue[Waveform | None]] = {}

    async def _get_next_queue_out_message(self) -> dict[str, Any]:
        if self._recv_bridge_queue is not None:
            return await self._recv_bridge_queue.get()

        while True:
            try:
                return await asyncio.to_thread(self.queue_out.get, timeout=1)
            except Empty:
                continue

    async def recv_queue(self) -> None:
        try:
            while True:
                res = await self._get_next_queue_out_message()

                if res.get("type") == "init_ok":
                    if not self._init_fut.done():
                        self._init_fut.set_result(None)
                    continue
                if res.get("type") == "init_error":
                    if not self._init_fut.done():
                        self._init_fut.set_exception(RuntimeError(res.get("error", "unknown init error")))
                    continue

                if res["type"] == "stream":
                    queue = self.stream_table.get(res["id"])
                    if queue is not None:
                        await queue.put(res["data"])
                    continue

                fut = self.op_table.pop(res["id"], None)
                if fut is None:
                    continue
                if res["type"] == "response":
                    fut.set_result(res.get("data"))
                else:
                    fut.set_exception(RuntimeError(res["error"]))
        except asyncio.CancelledError:
            return

    async def submit(self, cmd: str, *args: object, **kwargs: object) -> Any:
        op_id = gen_uuid()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self.op_table[op_id] = fut
        await asyncio.to_thread(self.queue_in.put, {"id": op_id, "type": cmd, "args": args, "kwargs": kwargs})
        return await fut

    async def health(self) -> dict[str, str]:
        return await self.submit("health")

    async def get_model_info(self) -> dict[str, Any]:
        return await self.submit("get_model_info")

    async def wait_for_ready(self) -> None:
        while not self._init_fut.done():
            if self.process.exitcode is not None:
                if not self._init_fut.done():
                    self._init_fut.set_exception(
                        RuntimeError(f"server process exited early: exitcode={self.process.exitcode}")
                    )
                break
            await asyncio.sleep(0.05)
        await self._init_fut

    async def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        return await self.submit("encode_latents", wav, wav_format)

    async def stop(self) -> None:
        graceful_stop = False
        if self.process.exitcode is None and self.process.is_alive():
            try:
                await asyncio.wait_for(self.submit("stop"), timeout=2.0)
                graceful_stop = True
            except Exception:
                pass

        if self._recv_bridge is not None:
            await self._recv_bridge.stop()

        self.recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.recv_task

        if graceful_stop and self.process.is_alive():
            await asyncio.to_thread(self.process.join, 5.0)
        if self.process.is_alive():
            self.process.terminate()
            await asyncio.to_thread(self.process.join, 2.0)
        if self.process.is_alive():
            kill = getattr(self.process, "kill", None)
            if callable(kill):
                kill()
                await asyncio.to_thread(self.process.join, 2.0)

        for queue in (getattr(self, "queue_in", None), getattr(self, "queue_out", None)):
            if queue is None:
                continue
            with contextlib.suppress(Exception):
                queue.close()
            with contextlib.suppress(Exception):
                queue.join_thread()

    async def stream_request(self, *request_args: object) -> AsyncGenerator[Waveform, None]:
        seq_id = gen_uuid()
        self.stream_table[seq_id] = asyncio.Queue()
        is_normal_exit = False
        try:
            await self.submit("add_request", seq_id, *request_args)
            while True:
                data = await self.stream_table[seq_id].get()
                if data is None:
                    is_normal_exit = True
                    break
                yield data
        finally:
            if not is_normal_exit:
                await self.submit("cancel", seq_id)
            self.stream_table.pop(seq_id, None)

    async def set_lora_enabled(self, enabled: bool) -> dict[str, Any]:
        return await self.submit("set_lora_enabled", enabled)

    async def load_lora(self, lora_path: str) -> dict[str, Any]:
        return await self.submit("load_lora", lora_path)

    async def reset_lora(self) -> dict[str, Any]:
        return await self.submit("reset_lora")


@dataclass(frozen=True)
class PromptEntry:
    latents: bytes
    text: str


TAsyncServer = TypeVar("TAsyncServer", bound=AsyncServerProcess)


class AsyncServerPool(Generic[TAsyncServer]):
    def __init__(
        self,
        server_cls: type[TAsyncServer],
        *,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: list[int] | None = None,
        lora_config: Any = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise ValueError(f"Unknown kwargs: {kwargs}")

        devices = normalize_devices(devices)
        self.servers = [
            server_cls(
                model_path=model_path,
                inference_timesteps=inference_timesteps,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                devices=[device_idx],
                lora_config=lora_config,
            )
            for device_idx in devices
        ]
        self.servers_load = np.zeros(len(self.servers), dtype=np.int32)
        self._prompt_pool: dict[str, PromptEntry] = {}
        self._model_info: dict[str, Any] | None = None

    def _least_loaded_index(self) -> int:
        if len(self.servers) == 0:
            raise RuntimeError("server pool is empty")
        return int(np.argmin(self.servers_load))

    @asynccontextmanager
    async def borrow_server(self) -> AsyncIterator[TAsyncServer]:
        server_idx = self._least_loaded_index()
        self.servers_load[server_idx] += 1
        try:
            yield self.servers[server_idx]
        finally:
            self.servers_load[server_idx] -= 1

    async def wait_for_ready(self) -> None:
        await asyncio.gather(*[server.wait_for_ready() for server in self.servers])
        if self._model_info is None:
            self._model_info = await self.servers[0].get_model_info()

    async def stop(self) -> None:
        await asyncio.gather(*[server.stop() for server in self.servers])

    async def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        return await self.servers[self._least_loaded_index()].encode_latents(wav, wav_format)

    async def get_model_info(self) -> dict[str, Any]:
        if self._model_info is None:
            self._model_info = await self.servers[self._least_loaded_index()].get_model_info()
        return dict(self._model_info)

    async def add_prompt(self, wav: bytes, wav_format: str, prompt_text: str) -> str:
        prompt_id = gen_uuid()
        prompt_latents = await self.encode_latents(wav, wav_format)
        self._prompt_pool[prompt_id] = PromptEntry(latents=prompt_latents, text=prompt_text)
        return prompt_id

    async def remove_prompt(self, prompt_id: str) -> None:
        del self._prompt_pool[prompt_id]

    def resolve_prompt_inputs(
        self,
        prompt_latents: bytes | None,
        prompt_text: str,
        prompt_id: str | None,
    ) -> tuple[bytes | None, str]:
        if prompt_id is None:
            return prompt_latents, prompt_text

        if prompt_id not in self._prompt_pool:
            raise ValueError(f"Prompt with id {prompt_id} not found")
        if prompt_latents is not None:
            raise ValueError("Prompt latents and prompt id cannot be provided at the same time")
        if len(prompt_text) > 0:
            raise ValueError("Prompt text and prompt id cannot be provided at the same time")

        prompt_info = self._prompt_pool[prompt_id]
        return prompt_info.latents, prompt_info.text

    async def set_lora_enabled(self, enabled: bool) -> dict[str, Any]:
        results = await asyncio.gather(*[server.set_lora_enabled(enabled) for server in self.servers])
        return results[0]

    async def load_lora(self, lora_path: str) -> dict[str, Any]:
        results = await asyncio.gather(*[server.load_lora(lora_path) for server in self.servers])
        return results[0]

    async def reset_lora(self) -> dict[str, Any]:
        results = await asyncio.gather(*[server.reset_lora() for server in self.servers])
        return results[0]


TAsyncServerPool = TypeVar("TAsyncServerPool")


class SyncServerPool(Generic[TAsyncServerPool]):
    def __init__(self, async_pool_factory: Callable[[], Awaitable[TAsyncServerPool]]) -> None:
        self.loop = asyncio.new_event_loop()
        self.server_pool = self.loop.run_until_complete(async_pool_factory())
        self.loop.run_until_complete(self.server_pool.wait_for_ready())

    def stop(self) -> None:
        assert self.loop is not None
        self.loop.run_until_complete(self.server_pool.stop())
        self.loop.close()
        self.loop = None

    def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.encode_latents(wav, wav_format))

    def get_model_info(self) -> dict[str, Any]:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.get_model_info())

    def add_prompt(self, wav: bytes, wav_format: str, prompt_text: str) -> str:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.add_prompt(wav, wav_format, prompt_text))

    def remove_prompt(self, prompt_id: str) -> None:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.remove_prompt(prompt_id))

    def iterate_async_generator(self, async_gen: AsyncGenerator[Waveform, None]) -> Iterator[Waveform]:
        assert self.loop is not None
        try:
            while True:
                yield self.loop.run_until_complete(async_gen.__anext__())
        except StopAsyncIteration:
            return

    def set_lora_enabled(self, enabled: bool) -> dict[str, Any]:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.set_lora_enabled(enabled))

    def load_lora(self, lora_path: str) -> dict[str, Any]:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.load_lora(lora_path))

    def reset_lora(self) -> dict[str, Any]:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.reset_lora())
