from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def torch_from_numpy_writable(array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a torch tensor without read-only array warnings."""

    if not array.flags.writeable:
        array = np.array(array, copy=True)
    return torch.from_numpy(array)


def concatenate_numpy_arrays(arrays: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
    """Concatenate arrays, but avoid an extra copy for the common single-array case."""

    if len(arrays) == 0:
        raise ValueError("arrays must not be empty")
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=axis)


def torch_from_numpy_sequence(arrays: Sequence[np.ndarray], axis: int = 0) -> torch.Tensor:
    """Convert one or more NumPy arrays to a torch tensor with safe writeable handling."""

    return torch_from_numpy_writable(concatenate_numpy_arrays(arrays, axis=axis))


def float32_array_from_buffer(buffer: bytes, width: int) -> np.ndarray:
    """View float32 bytes as a 2D latent array."""

    return np.frombuffer(buffer, dtype=np.float32).reshape(-1, width)
