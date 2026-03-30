"""Triton fused dequant + matmul kernels for on-the-fly TurboQuant inference.

Fuses unpack → codebook lookup → matmul → norm rescale in one kernel launch,
avoiding materialization of the full dequantized weight matrix.

Supports both 2-bit (4 values/byte) and 4-bit (2 values/byte) packing.

Expects pre-rotated input: x_rot = hadamard_rotate(x, seed) or x @ Pi.T
The rotation is NOT done inside the kernel — call it before.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configurations
# ---------------------------------------------------------------------------

_AUTOTUNE_CONFIGS = [
    # Small batch (inference with B=1..4)
    triton.Config({"BLOCK_B": 1,  "BLOCK_N": 32,  "BLOCK_K": 32},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 1,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 4,  "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 4,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
    # Medium batch
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=8, num_stages=3),
    # Large batch
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=8, num_stages=3),
]


# ---------------------------------------------------------------------------
# 2-bit kernel
# ---------------------------------------------------------------------------


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["B", "N", "K"])
@triton.jit
def _turboquant_2bit_matmul_kernel(
    input_ptr, indices_ptr, codebook_ptr, norms_ptr, output_ptr,
    B, N, K, PACKED_K,
    N_LEVELS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused 2-bit dequant-matmul: output[b,n] = norms_scaled[n] * Σ_k x[b,k] * codebook[idx[n,k]]"""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K

        # Load input tile: (BLOCK_B, BLOCK_K)
        inp_off = rb[:, None] * K + rk[None, :]
        inp_mask = mask_b[:, None] & mask_k[None, :]
        inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0)

        # 2-bit unpack: 4 values per byte
        byte_col = rk // 4
        shift = (rk % 4) * 2
        byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        idx = (packed >> shift[None, :]) & 0x03

        # Codebook lookup (4 entries for 2-bit — fits in registers)
        w_quant = tl.load(codebook_ptr + idx.to(tl.int32), mask=w_mask, other=0.0)

        # TF32 tensor-core matmul
        acc += tl.dot(
            inp_tile.to(tl.float32),
            tl.trans(w_quant.to(tl.float32)),
            allow_tf32=True,
        )

    # Multiply by pre-scaled norms
    norm_vals = tl.load(norms_ptr + rn, mask=mask_n, other=1.0)
    acc = acc * norm_vals[None, :]

    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# 4-bit kernel
# ---------------------------------------------------------------------------


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["B", "N", "K"])
@triton.jit
def _turboquant_4bit_matmul_kernel(
    input_ptr, indices_ptr, codebook_ptr, norms_ptr, output_ptr,
    B, N, K, PACKED_K,
    N_LEVELS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused 4-bit dequant-matmul."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K

        inp_off = rb[:, None] * K + rk[None, :]
        inp_mask = mask_b[:, None] & mask_k[None, :]
        inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0)

        # 4-bit unpack: 2 values per byte
        byte_col = rk // 2
        is_high = (rk % 2) == 1
        byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        idx = tl.where(is_high[None, :], hi, lo)

        w_quant = tl.load(codebook_ptr + idx.to(tl.int32), mask=w_mask, other=0.0)

        acc += tl.dot(
            inp_tile.to(tl.float32),
            tl.trans(w_quant.to(tl.float32)),
            allow_tf32=True,
        )

    norm_vals = tl.load(norms_ptr + rn, mask=mask_n, other=1.0)
    acc = acc * norm_vals[None, :]

    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def triton_fused_matmul(
    x_rot: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    K: int,
    scale: float | None = None,
    bit_width: int = 2,
) -> torch.Tensor:
    """Fused dequant + matmul via Triton.

    Expects pre-rotated input: x_rot = hadamard_rotate(x, seed) or x @ Pi.T

    Args:
        x_rot: (B, K) pre-rotated activations
        indices_packed: (N, packed_K) packed uint8 weight indices
        codebook: (n_levels,) centroids
        norms: (N,) per-row weight norms
        K: dimension of this group
        scale: norm divisor (default: sqrt(K))
        bit_width: 2 or 4

    Returns:
        output: (B, N)
    """
    B = x_rot.shape[0]
    N = indices_packed.shape[0]
    PACKED_K = indices_packed.shape[1]
    if scale is None:
        scale = math.sqrt(K)

    norms_scaled = norms / scale
    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_B"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    if bit_width == 2:
        _turboquant_2bit_matmul_kernel[grid](
            x_rot, indices_packed, codebook, norms_scaled, output,
            B, N, K, PACKED_K,
            N_LEVELS=codebook.shape[0],
        )
    elif bit_width == 4:
        _turboquant_4bit_matmul_kernel[grid](
            x_rot, indices_packed, codebook, norms_scaled, output,
            B, N, K, PACKED_K,
            N_LEVELS=codebook.shape[0],
        )
    else:
        raise ValueError(f"Unsupported bit_width={bit_width}")

    return output
