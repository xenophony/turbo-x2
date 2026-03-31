"""JIT-compiled CUDA extension for TurboQuant inference.

Compiles on first import, cached for subsequent runs.
Falls back gracefully if CUDA toolkit is not available.

Usage:
    from turboquant.cuda_ext import turboquant_forward, is_available

    if is_available():
        output = turboquant_forward(x, indices_packed, codebook, norms, signs, group_size, bit_width)
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Optional

import torch

_ext = None
_available = False


def _load_extension():
    """JIT compile the CUDA extension. Called once on first use."""
    global _ext, _available
    if _ext is not None:
        return

    try:
        from torch.utils.cpp_extension import load

        csrc_dir = Path(__file__).parent / "csrc"
        _ext = load(
            name="turboquant_cuda",
            sources=[
                str(csrc_dir / "turboquant_ext.cpp"),
                str(csrc_dir / "turboquant_cuda_kernel.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        _available = True
    except Exception as e:
        print(f"[TurboQuant] CUDA extension not available: {e}")
        print("[TurboQuant] Falling back to Python/Triton kernels.")
        _available = False


def is_available() -> bool:
    """Check if the CUDA extension compiled successfully."""
    _load_extension()
    return _available


def precompute_signs(rotation_seed: int, group_size: int, in_features: int,
                     device: str = "cuda") -> torch.Tensor:
    """Precompute hadamard sign vectors for all groups.

    Returns: (n_groups, group_size) float32 tensor of ±1 values.
    """
    n_groups = math.ceil(in_features / group_size)
    signs = torch.empty(n_groups, group_size, dtype=torch.float32, device=device)

    for g in range(n_groups):
        g_start = g * group_size
        g_dim = min(group_size, in_features - g_start)
        seed = rotation_seed + g_start
        gen = torch.Generator().manual_seed(seed)
        s = torch.randint(0, 2, (group_size,), generator=gen).float() * 2 - 1
        signs[g] = s

    return signs


def turboquant_forward(
    x: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    weight_norms: torch.Tensor,
    group_signs: torch.Tensor,
    group_size: int,
    bit_width: int,
    use_hadamard: bool = True,
) -> torch.Tensor:
    """Fused TurboQuant forward — one call per layer, all groups in C++/CUDA.

    Args:
        x: (B, in_features) input activations
        indices_packed: (out_features, packed_dim) uint8 packed indices
        codebook: (n_levels,) float32 centroids
        weight_norms: (out_features,) or (out_features, n_groups) float32
        group_signs: (n_groups, group_size) float32 ±1 hadamard signs
        group_size: quantization group size
        bit_width: 2 or 4
        use_hadamard: True for hadamard rotation (must be power of 2 group_size)

    Returns:
        output: (B, out_features) float32
    """
    _load_extension()
    if not _available:
        raise RuntimeError("CUDA extension not available")

    return _ext.forward(
        x, indices_packed, codebook, weight_norms, group_signs,
        group_size, bit_width, use_hadamard,
    )
