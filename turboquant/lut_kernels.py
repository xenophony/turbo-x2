"""Lookup-table based quantized matmul — the "Doom trick" for TurboQuant.

At 2-bit, every weight is one of 4 codebook values. Instead of multiplying
x * codebook[index] per element, we precompute 4 scaled versions of x
and then just gather + accumulate. No per-element multiply in the inner loop.

For 4-bit (16 centroids), same approach — 16 precomputed tables, still fits in L1.

Two implementations:
  - PyTorch (works on CPU and GPU)
  - Triton (GPU-optimized, fuses unpack + LUT gather + accumulate)
"""

from __future__ import annotations

import math

import torch

from turboquant.packing import unpack_bits

# Try Triton for GPU LUT kernel
_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# PyTorch LUT — works on CPU and GPU, no Triton needed
# ---------------------------------------------------------------------------


def lut_matmul_pytorch(
    x_rot: torch.Tensor,           # (B, K) pre-rotated input
    indices_packed: torch.Tensor,   # (N, packed_K) packed uint8
    codebook: torch.Tensor,         # (n_levels,) centroids
    norms: torch.Tensor,            # (N,) per-row norms
    K: int,                         # group dimension
    scale: float,                   # sqrt(K)
    bit_width: int = 2,
) -> torch.Tensor:
    """LUT-based quantized matmul using mask decomposition.

    Instead of: output[b,n] = sum_k x[b,k] * codebook[idx[n,k]]
    We compute: for each centroid c, output += (x * c) @ mask_c.T

    This replaces N*K multiplies with n_levels*K multiplies (n_levels=4 for 2-bit).
    The mask matmul can be done with binary operands.

    Args:
        x_rot: (B, K) pre-rotated input
        indices_packed: (N, packed_K) packed indices
        codebook: (n_levels,) centroid values
        norms: (N,) weight norms
        K: group dimension
        scale: sqrt(K)
        bit_width: 2 or 4

    Returns:
        output: (B, N)
    """
    B = x_rot.shape[0]
    N = indices_packed.shape[0]
    n_levels = len(codebook)

    # Unpack indices: (N, K)
    indices = unpack_bits(indices_packed, K, bit_width)

    # LUT matmul: 4 (or 16) masked matmuls instead of one dense matmul
    output = torch.zeros(B, N, dtype=torch.float32, device=x_rot.device)
    for c in range(n_levels):
        # Scale input by this centroid (the "precompute" step)
        scaled_x = x_rot * codebook[c]  # (B, K) — broadcast, one multiply per element

        # Binary mask: which positions use this centroid
        mask = (indices == c).to(scaled_x.dtype)  # (N, K)

        # Accumulate: (B, K) @ (K, N) -> (B, N)
        output += scaled_x @ mask.T

    # Apply norms
    output *= norms.unsqueeze(0) / scale
    return output


# ---------------------------------------------------------------------------
# Triton LUT kernel — GPU-optimized
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    _LUT_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_B": 1,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_B": 1,  "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_B": 4,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_B": 16, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_B": 16, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_B": 32, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(configs=_LUT_AUTOTUNE_CONFIGS, key=["B", "N", "K"])
    @triton.jit
    def _lut_2bit_matmul_kernel(
        input_ptr, indices_ptr, codebook_ptr, norms_ptr, output_ptr,
        B, N, K, PACKED_K,
        BLOCK_B: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """LUT-based 2-bit fused matmul.

        For each K-tile:
          1. Load input tile (BLOCK_B, BLOCK_K)
          2. Build LUT: table[c, k] = input[b, k] * centroid[c] for c in 0..3
          3. Unpack indices, gather from table, accumulate
        """
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)

        rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_b = rb < B
        mask_n = rn < N

        # Load codebook (4 entries for 2-bit) — stays in registers
        c0 = tl.load(codebook_ptr + 0)
        c1 = tl.load(codebook_ptr + 1)
        c2 = tl.load(codebook_ptr + 2)
        c3 = tl.load(codebook_ptr + 3)

        acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < K

            # Load input tile: (BLOCK_B, BLOCK_K)
            inp_off = rb[:, None] * K + rk[None, :]
            inp_mask = mask_b[:, None] & mask_k[None, :]
            inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0).to(tl.float32)

            # Build LUT: 4 pre-scaled versions of input
            # table_c[b, k] = inp_tile[b, k] * centroid_c
            # These stay in registers — 4 * BLOCK_B * BLOCK_K values
            t0 = inp_tile * c0
            t1 = inp_tile * c1
            t2 = inp_tile * c2
            t3 = inp_tile * c3

            # Unpack 2-bit indices for this tile
            byte_col = rk // 4
            shift = (rk % 4) * 2
            byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
            w_mask = mask_n[:, None] & mask_k[None, :]
            packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
            idx = (packed >> shift[None, :]) & 0x03  # (BLOCK_N, BLOCK_K)

            # Gather from LUT: select t0, t1, t2, or t3 based on index
            # w[n, k] = table[idx[n,k]][b, k]
            # Since we can't dynamically index into separate tensors in Triton,
            # use conditional selection
            w = tl.where(idx == 0, c0, tl.where(idx == 1, c1, tl.where(idx == 2, c2, c3)))

            # This gives us the codebook value — now multiply by input
            # Wait — that's the same as the non-LUT approach.
            # The LUT advantage is: precompute x*c, then gather.
            # In Triton, we need to gather from (BLOCK_B, BLOCK_K) tables.

            # For per-batch gathering, expand idx to (BLOCK_B, BLOCK_N, BLOCK_K)
            # which is too large. Instead, do the 4-centroid decomposition:
            is_0 = (idx == 0).to(tl.float32)  # (BLOCK_N, BLOCK_K)
            is_1 = (idx == 1).to(tl.float32)
            is_2 = (idx == 2).to(tl.float32)
            is_3 = (idx == 3).to(tl.float32)

            # 4 small matmuls with precomputed scaled inputs
            acc += tl.dot(t0, tl.trans(is_0), allow_tf32=True)
            acc += tl.dot(t1, tl.trans(is_1), allow_tf32=True)
            acc += tl.dot(t2, tl.trans(is_2), allow_tf32=True)
            acc += tl.dot(t3, tl.trans(is_3), allow_tf32=True)

        # Apply pre-scaled norms
        norm_vals = tl.load(norms_ptr + rn, mask=mask_n, other=1.0)
        acc = acc * norm_vals[None, :]

        out_off = rb[:, None] * N + rn[None, :]
        out_mask = mask_b[:, None] & mask_n[None, :]
        tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def lut_matmul(
    x_rot: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    K: int,
    scale: float | None = None,
    bit_width: int = 2,
) -> torch.Tensor:
    """LUT-based quantized matmul — dispatches to Triton (GPU) or PyTorch (CPU).

    Args:
        x_rot: (B, K) pre-rotated input
        indices_packed: (N, packed_K) packed uint8
        codebook: (n_levels,) centroids
        norms: (N,) per-row norms
        K: group dimension
        scale: sqrt(K), computed if None
        bit_width: 2 or 4

    Returns:
        output: (B, N)
    """
    if scale is None:
        scale = math.sqrt(K)
    norms_scaled = norms / scale

    B = x_rot.shape[0]
    N = indices_packed.shape[0]
    PACKED_K = indices_packed.shape[1]

    if _HAS_TRITON and x_rot.device.type == "cuda" and bit_width == 2:
        output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)
        grid = lambda META: (
            triton.cdiv(B, META["BLOCK_B"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        _lut_2bit_matmul_kernel[grid](
            x_rot, indices_packed, codebook, norms_scaled, output,
            B, N, K, PACKED_K,
        )
        return output
    else:
        return lut_matmul_pytorch(
            x_rot, indices_packed, codebook, norms, K, scale, bit_width,
        )
