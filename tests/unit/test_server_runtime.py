import asyncio
import queue

from nanovllm_voxcpm.models.server_runtime import _QueueBridgeThread, resolve_recv_queue_mode


def test_async_server_pool_caches_model_info_after_ready():
    from nanovllm_voxcpm.models.server_runtime import AsyncServerPool

    class FakeAsyncServer:
        def __init__(self, **kwargs):
            self.model_info_calls = 0
            self.wait_for_ready_calls = 0

        async def wait_for_ready(self):
            self.wait_for_ready_calls += 1

        async def stop(self):
            return None

        async def get_model_info(self):
            self.model_info_calls += 1
            return {
                "sample_rate": 16000,
                "channels": 1,
                "feat_dim": 64,
                "patch_size": 2,
                "model_path": "/fake/model",
            }

        async def encode_latents(self, wav: bytes, wav_format: str):
            raise AssertionError("encode_latents should not be called in this test")

    async def run():
        pool = AsyncServerPool(FakeAsyncServer, model_path="/fake/model", devices=[0, 1])

        await pool.wait_for_ready()
        first = await pool.get_model_info()
        first["sample_rate"] = 1
        second = await pool.get_model_info()

        assert second["sample_rate"] == 16000
        assert [server.wait_for_ready_calls for server in pool.servers] == [1, 1]
        assert sum(server.model_info_calls for server in pool.servers) == 1

    asyncio.run(run())


def test_resolve_recv_queue_mode_defaults_to_bridge(monkeypatch):
    monkeypatch.delenv("NANOVLLM_RECV_QUEUE_MODE", raising=False)
    assert resolve_recv_queue_mode() == "bridge"

    monkeypatch.setenv("NANOVLLM_RECV_QUEUE_MODE", "to_thread")
    assert resolve_recv_queue_mode() == "to_thread"

    monkeypatch.setenv("NANOVLLM_RECV_QUEUE_MODE", "invalid")
    assert resolve_recv_queue_mode() == "bridge"


def test_queue_bridge_thread_forwards_messages():
    async def run():
        source_queue: queue.Queue[dict[str, object]] = queue.Queue()
        target_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        bridge = _QueueBridgeThread(source_queue, asyncio.get_running_loop(), target_queue, poll_timeout_s=0.01)

        bridge.start()
        source_queue.put({"type": "response", "id": "op-1", "data": 123})

        forwarded = await asyncio.wait_for(target_queue.get(), timeout=0.5)
        await bridge.stop()

        assert forwarded == {"type": "response", "id": "op-1", "data": 123}

    asyncio.run(run())
