import numpy as np
import pytest
import warnings

torch = pytest.importorskip("torch")


def test_concatenate_numpy_arrays_returns_single_array_without_copy():
    from nanovllm_voxcpm.utils.torch_numpy import concatenate_numpy_arrays

    array = np.arange(6, dtype=np.float32).reshape(3, 2)

    assert concatenate_numpy_arrays([array]) is array


def test_torch_from_numpy_sequence_copies_single_read_only_array_without_warning():
    from nanovllm_voxcpm.utils.torch_numpy import torch_from_numpy_sequence

    array = np.arange(6, dtype=np.float32).reshape(3, 2)
    array.setflags(write=False)

    with warnings.catch_warnings(record=True) as caught:
        tensor = torch_from_numpy_sequence([array])

    assert torch.equal(tensor, torch.tensor(array))
    assert not [warning for warning in caught if "not writable" in str(warning.message)]


def test_float32_array_from_buffer_reshapes_latents():
    from nanovllm_voxcpm.utils.torch_numpy import float32_array_from_buffer

    latents = np.arange(8, dtype=np.float32).tobytes()

    array = float32_array_from_buffer(latents, 4)

    assert array.shape == (2, 4)
    assert array.dtype == np.float32
