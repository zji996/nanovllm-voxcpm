import asyncio


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
