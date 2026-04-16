from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_gather_padded_from_block_table_returns_masked_tokens():
    from nanovllm_voxcpm.layers.attention import _gather_padded_from_block_table

    cache = torch.arange(4 * 2, dtype=torch.float32).view(4, 2, 1, 1)
    block_tables = torch.tensor([[2, -1, 1]], dtype=torch.int32)

    gathered, token_mask = _gather_padded_from_block_table(cache, block_tables)

    assert gathered.shape == (1, 6, 1, 1)
    assert gathered[0, :, 0, 0].tolist() == [4.0, 5.0, 0.0, 0.0, 2.0, 3.0]
    assert token_mask.tolist() == [[True, True, False, False, True, True]]


def test_sdpa_decode_matches_reference_single_sequence_gather():
    from nanovllm_voxcpm.layers.attention import _gather_from_block_table, _sdpa_decode, _sdpa_single_sequence

    torch.manual_seed(0)

    q = torch.randn(2, 4, 3, dtype=torch.float32)
    k_cache = torch.randn(5, 2, 2, 3, dtype=torch.float32)
    v_cache = torch.randn(5, 2, 2, 3, dtype=torch.float32)
    context = SimpleNamespace(
        context_lens=torch.tensor([3, 2], dtype=torch.int32),
        block_tables=torch.tensor(
            [
                [1, 3, 4],
                [0, 2, 4],
            ],
            dtype=torch.int32,
        ),
    )

    outputs = _sdpa_decode(q, k_cache, v_cache, context)

    expected = []
    for seq_idx in range(q.size(0)):
        seq_len = int(context.context_lens[seq_idx].item())
        k_seq = _gather_from_block_table(k_cache, context.block_tables[seq_idx], seq_len)
        v_seq = _gather_from_block_table(v_cache, context.block_tables[seq_idx], seq_len)
        expected.append(
            _sdpa_single_sequence(
                q[seq_idx : seq_idx + 1],
                k_seq,
                v_seq,
                is_causal=True,
                causal_diagonal=seq_len - 1,
            )
        )

    torch.testing.assert_close(outputs, torch.cat(expected, dim=0), atol=1e-5, rtol=1e-5)
