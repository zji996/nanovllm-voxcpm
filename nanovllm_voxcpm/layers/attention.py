import os

import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from torch import nn

from nanovllm_voxcpm.utils.context import get_context

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


def _resolve_attention_backend() -> str:
    backend = os.environ.get("NANOVLLM_ATTENTION_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "flash", "sdpa"}:
        raise RuntimeError(f"Invalid NANOVLLM_ATTENTION_BACKEND={backend!r}; expected auto, flash, or sdpa")
    if backend == "auto":
        return "flash" if flash_attn_func is not None else "sdpa"
    if backend == "flash" and flash_attn_func is None:
        raise RuntimeError("NANOVLLM_ATTENTION_BACKEND=flash requires flash-attn to be installed")
    return backend


def _repeat_kv_heads(x: torch.Tensor, target_num_heads: int) -> torch.Tensor:
    num_kv_heads = x.size(-2)
    if num_kv_heads == target_num_heads:
        return x
    if target_num_heads % num_kv_heads != 0:
        raise RuntimeError(f"Cannot expand {num_kv_heads} KV heads to {target_num_heads} attention heads")
    return x.repeat_interleave(target_num_heads // num_kv_heads, dim=-2)


def _gather_from_block_table(cache: torch.Tensor, block_table: torch.Tensor, seq_len: int) -> torch.Tensor:
    if seq_len == 0:
        return cache.new_empty((0, cache.size(2), cache.size(3)))

    pieces = []
    remaining = seq_len
    for block_id in block_table.detach().cpu().tolist():
        if block_id < 0 or remaining <= 0:
            break
        take = min(cache.size(1), remaining)
        pieces.append(cache[block_id, :take])
        remaining -= take

    if remaining != 0:
        raise RuntimeError("KV cache is missing required blocks for the current sequence")
    return torch.cat(pieces, dim=0)


def _gather_padded_from_block_table(cache: torch.Tensor, block_tables: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if block_tables.ndim == 1:
        block_tables = block_tables.unsqueeze(0)

    batch_size, num_blocks = block_tables.shape
    if num_blocks == 0:
        empty = cache.new_empty((batch_size, 0, cache.size(2), cache.size(3)))
        empty_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=cache.device)
        return empty, empty_mask

    clamped_block_tables = block_tables.clamp_min(0).to(dtype=torch.long)
    gathered = cache.index_select(0, clamped_block_tables.reshape(-1)).view(
        batch_size,
        num_blocks,
        cache.size(1),
        cache.size(2),
        cache.size(3),
    )
    block_mask = block_tables.ge(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    gathered = gathered.masked_fill(~block_mask, 0)
    token_mask = block_tables.ge(0).unsqueeze(-1).expand(batch_size, num_blocks, cache.size(1)).reshape(batch_size, -1)
    return gathered.reshape(batch_size, -1, cache.size(2), cache.size(3)), token_mask


def _sdpa_single_sequence(
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    is_causal: bool,
    causal_diagonal: int = 0,
) -> torch.Tensor:
    q_heads = q_seq.transpose(0, 1).unsqueeze(0)
    k_heads = _repeat_kv_heads(k_seq, q_seq.size(1)).transpose(0, 1).unsqueeze(0)
    v_heads = _repeat_kv_heads(v_seq, q_seq.size(1)).transpose(0, 1).unsqueeze(0)

    attn_mask = None
    if is_causal:
        attn_mask = torch.tril(
            torch.ones(
                q_seq.size(0),
                k_seq.size(0),
                dtype=torch.bool,
                device=q_seq.device,
            ),
            diagonal=causal_diagonal,
        )

    out = F.scaled_dot_product_attention(
        q_heads,
        k_heads,
        v_heads,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return out.squeeze(0).transpose(0, 1).contiguous()


def _sdpa_varlen_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    context,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    q_offsets = context.cu_seqlens_q.detach().cpu().tolist()
    k_offsets = context.cu_seqlens_k.detach().cpu().tolist()
    outputs = []

    for seq_idx in range(len(q_offsets) - 1):
        q_start, q_end = q_offsets[seq_idx], q_offsets[seq_idx + 1]
        k_start, k_end = k_offsets[seq_idx], k_offsets[seq_idx + 1]

        q_seq = q[q_start:q_end]
        if context.block_tables is not None:
            k_seq = _gather_from_block_table(k_cache, context.block_tables[seq_idx], k_end - k_start)
            v_seq = _gather_from_block_table(v_cache, context.block_tables[seq_idx], k_end - k_start)
        else:
            k_seq = k[k_start:k_end]
            v_seq = v[k_start:k_end]

        prefix_len = k_seq.size(0) - q_seq.size(0)
        outputs.append(_sdpa_single_sequence(q_seq, k_seq, v_seq, is_causal=True, causal_diagonal=prefix_len))

    if not outputs:
        return q.new_empty(q.shape)
    return torch.cat(outputs, dim=0)


def _sdpa_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, context) -> torch.Tensor:
    if context.block_tables is None or context.context_lens is None:
        raise RuntimeError("Decode attention requires block_tables and context_lens")

    k_seq, token_mask = _gather_padded_from_block_table(k_cache, context.block_tables)
    v_seq, _ = _gather_padded_from_block_table(v_cache, context.block_tables)

    max_tokens = k_seq.size(1)
    token_positions = torch.arange(max_tokens, device=q.device).unsqueeze(0)
    seq_mask = token_positions < context.context_lens.to(device=q.device, dtype=torch.long).unsqueeze(1)
    attn_mask = (token_mask & seq_mask).unsqueeze(1).unsqueeze(1)

    q_heads = q.unsqueeze(2)
    k_heads = _repeat_kv_heads(k_seq, q.size(1)).permute(0, 2, 1, 3)
    v_heads = _repeat_kv_heads(v_seq, q.size(1)).permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(
        q_heads,
        k_heads,
        v_heads,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return out.squeeze(2).contiguous()


def _sdpa_non_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_heads = q.permute(0, 2, 1, 3)
    k_heads = _repeat_kv_heads(k, q.size(2)).permute(0, 2, 1, 3)
    v_heads = _repeat_kv_heads(v, q.size(2)).permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=0.0, is_causal=False)
    return out.permute(0, 2, 1, 3).contiguous()


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        is_causal: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.is_causal = is_causal
        self.backend = _resolve_attention_backend()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if self.is_causal:
            context = get_context()
            k_cache, v_cache = self.k_cache, self.v_cache
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            if self.backend == "flash":
                if context.is_prefill:
                    if context.block_tables is not None:  # prefix cache
                        k, v = k_cache, v_cache
                    o = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        max_seqlen_q=context.max_seqlen_q,
                        cu_seqlens_q=context.cu_seqlens_q,
                        max_seqlen_k=context.max_seqlen_k,
                        cu_seqlens_k=context.cu_seqlens_k,
                        softmax_scale=self.scale,
                        causal=True,
                        block_table=context.block_tables,
                    )
                else:  # decode
                    o = flash_attn_with_kvcache(
                        q.unsqueeze(1),
                        k_cache,
                        v_cache,
                        cache_seqlens=context.context_lens,
                        block_table=context.block_tables,
                        softmax_scale=self.scale,
                        causal=True,
                    )
            elif context.is_prefill:
                o = _sdpa_varlen_prefill(q, k, v, context, k_cache, v_cache)
            else:
                o = _sdpa_decode(q, k_cache, v_cache, context)
        else:
            if self.backend == "flash":
                o = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=False)
            else:
                o = _sdpa_non_causal(q, k, v)
        return o
