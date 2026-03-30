"""Core TurboQuant quantization: single-pass rotation + Lloyd-Max scalar quantization.

Provides both simulation (returns fp32/bf16 approximation) and packed storage
(returns packed indices + norms + codebook) for any supported bit width.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from turboquant.codebook import get_codebook
from turboquant.packing import pack_bits, unpack_bits
from turboquant.rotation import (
    generate_rotation_matrix,
    hadamard_rotate,
    hadamard_rotate_inverse,
)


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _resolve_rotation(rotation: str, dim: int) -> str:
    """Fall back to QR if hadamard is requested but dim isn't a power of 2."""
    if rotation == "hadamard" and not _is_power_of_2(dim):
        return "qr"
    return rotation


# ---------------------------------------------------------------------------
# Single-pass quantization (simulation)
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_quantize(
    W: torch.Tensor,
    bit_width: int = 4,
    group_size: Optional[int] = None,
    seed: int = 42,
    rotation: str = "hadamard",
) -> torch.Tensor:
    """Apply TurboQuant quantization and return the dequantized approximation.

    Steps:
      1. Row-normalize
      2. Rotate by random orthogonal Pi
      3. Scalar quantize with Lloyd-Max codebook
      4. Dequantize (centroids), inverse-rotate, rescale

    Args:
        W: (out_features, in_features) weight matrix
        bit_width: bits per coordinate
        group_size: group size along in_features (None = full row)
        seed: rotation seed
        rotation: "qr" or "hadamard"

    Returns:
        W_approx: same shape/dtype as W
    """
    orig_dtype = W.dtype
    W = W.float()
    out_features, in_features = W.shape

    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    if group_size is None:
        group_size = in_features

    W_approx = torch.zeros_like(W)

    for g_start in range(0, in_features, group_size):
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms

        rot = _resolve_rotation(rotation, g_dim)
        if rot == "hadamard":
            Y = hadamard_rotate(W_norm, seed=seed + g_start)
        else:
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(W.device)
            Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1)
        Y_quant = centroids[indices].reshape(Y_scaled.shape)

        Y_unscaled = Y_quant / scale
        if rot == "hadamard":
            W_g_approx = hadamard_rotate_inverse(Y_unscaled, seed=seed + g_start)
        else:
            W_g_approx = Y_unscaled @ Pi
        W_approx[:, g_start:g_end] = W_g_approx * norms

    return W_approx.to(orig_dtype)


# ---------------------------------------------------------------------------
# Single-pass quantization (packed storage)
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_quantize_packed(
    W: torch.Tensor,
    bit_width: int = 2,
    group_size: Optional[int] = None,
    seed: int = 42,
    rotation: str = "hadamard",
) -> dict:
    """Quantize and return packed representation for storage/inference.

    Args:
        W: (M, N) weight matrix
        bit_width: bits per element (2 or 4)
        group_size: group size (None = full row)
        seed: rotation seed
        rotation: "qr" or "hadamard"

    Returns:
        dict with:
            indices_packed: (M, packed_dim) uint8
            codebook: (2^b,) float32
            norms: (M,) or (M, n_groups) float32
            seed: int
            group_size: int
            shape: (M, N)
            bit_width: int
            rotation: str
    """
    M, N = W.shape
    if group_size is None:
        group_size = N

    W = W.float()
    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    all_norms = []
    all_indices = []

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        rot = _resolve_rotation(rotation, g_dim)
        if rot == "hadamard":
            Y = hadamard_rotate(W_norm, seed=seed + g_start)
        else:
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(W.device)
            Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    full_indices = torch.cat(all_indices, dim=1)
    norms_out = torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]

    # Pad to multiple of pack_factor for packing
    pack_factor = 8 // bit_width  # 4 for 2-bit, 2 for 4-bit
    remainder = N % pack_factor
    if remainder != 0:
        pad_size = pack_factor - remainder
        full_indices = torch.nn.functional.pad(full_indices, (0, pad_size), value=0)

    packed = pack_bits(full_indices, bit_width)

    return {
        "indices_packed": packed,
        "codebook": centroids.cpu(),
        "norms": norms_out.cpu(),
        "seed": seed,
        "group_size": group_size,
        "shape": (M, N),
        "bit_width": bit_width,
        "rotation": rotation,
    }
