"""Bit-packing utilities for TurboQuant quantization indices.

Supports 2-bit (4 values per byte) and 4-bit (2 values per byte) packing,
plus a generic dispatcher.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# 2-bit packing / unpacking (4 values per byte)
# ---------------------------------------------------------------------------


def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit indices (0-3) into uint8, 4 per byte.

    Layout: byte = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)
    where b0 = indices[..., 4i], b1 = indices[..., 4i+1], etc.

    Args:
        indices: int tensor (..., N) with values in [0, 3], N must be divisible by 4

    Returns:
        packed: uint8 tensor (..., N//4)
    """
    assert indices.shape[-1] % 4 == 0, "Last dim must be divisible by 4 for 2-bit packing"
    b0 = indices[..., 0::4].to(torch.uint8)
    b1 = indices[..., 1::4].to(torch.uint8)
    b2 = indices[..., 2::4].to(torch.uint8)
    b3 = indices[..., 3::4].to(torch.uint8)
    return b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)


def unpack_2bit(packed: torch.Tensor, N: int) -> torch.Tensor:
    """Unpack uint8 -> 2-bit indices.

    Args:
        packed: uint8 tensor (..., ceil(N/4))
        N: original last dimension

    Returns:
        indices: int32 tensor (..., N)
    """
    b0 = (packed & 0x03).to(torch.int32)
    b1 = ((packed >> 2) & 0x03).to(torch.int32)
    b2 = ((packed >> 4) & 0x03).to(torch.int32)
    b3 = ((packed >> 6) & 0x03).to(torch.int32)
    result = torch.stack([b0, b1, b2, b3], dim=-1)
    return result.reshape(*packed.shape[:-1], -1)[..., :N]


# ---------------------------------------------------------------------------
# 3-bit packing / unpacking (8 values per 3 bytes)
# ---------------------------------------------------------------------------


def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit indices (0-7) into uint8, 8 values per 3 bytes.

    Layout (24 bits = 8 × 3 bits):
      byte0: [v0:3][v1:3][v2_lo:2]         = bits 0-7
      byte1: [v2_hi:1][v3:3][v4:3][v5_lo:1] = bits 8-15
      byte2: [v5_hi:2][v6:3][v7:3]         = bits 16-23

    Args:
        indices: int tensor (..., N) with values in [0, 7], N must be divisible by 8

    Returns:
        packed: uint8 tensor (..., N*3//8)
    """
    assert indices.shape[-1] % 8 == 0, "Last dim must be divisible by 8 for 3-bit packing"
    idx = indices.to(torch.uint8)

    v0 = idx[..., 0::8]
    v1 = idx[..., 1::8]
    v2 = idx[..., 2::8]
    v3 = idx[..., 3::8]
    v4 = idx[..., 4::8]
    v5 = idx[..., 5::8]
    v6 = idx[..., 6::8]
    v7 = idx[..., 7::8]

    # byte0: v0 (bits 0-2) | v1 (bits 3-5) | v2_lo (bits 6-7)
    byte0 = v0 | (v1 << 3) | ((v2 & 0x03) << 6)

    # byte1: v2_hi (bit 0) | v3 (bits 1-3) | v4 (bits 4-6) | v5_lo (bit 7)
    byte1 = ((v2 >> 2) & 0x01) | (v3 << 1) | (v4 << 4) | ((v5 & 0x01) << 7)

    # byte2: v5_hi (bits 0-1) | v6 (bits 2-4) | v7 (bits 5-7)
    byte2 = ((v5 >> 1) & 0x03) | (v6 << 2) | (v7 << 5)

    # Interleave the 3 bytes
    batch_shape = indices.shape[:-1]
    n_groups = indices.shape[-1] // 8
    result = torch.stack([byte0, byte1, byte2], dim=-1)
    return result.reshape(*batch_shape, n_groups * 3)


def unpack_3bit(packed: torch.Tensor, N: int) -> torch.Tensor:
    """Unpack uint8 -> 3-bit indices.

    Args:
        packed: uint8 tensor (..., N*3//8)
        N: original last dimension (must be divisible by 8)

    Returns:
        indices: int32 tensor (..., N)
    """
    batch_shape = packed.shape[:-1]
    n_groups = packed.shape[-1] // 3
    p = packed.reshape(*batch_shape, n_groups, 3)

    byte0 = p[..., 0].to(torch.int32)
    byte1 = p[..., 1].to(torch.int32)
    byte2 = p[..., 2].to(torch.int32)

    v0 = byte0 & 0x07
    v1 = (byte0 >> 3) & 0x07
    v2 = ((byte0 >> 6) & 0x03) | ((byte1 & 0x01) << 2)
    v3 = (byte1 >> 1) & 0x07
    v4 = (byte1 >> 4) & 0x07
    v5 = ((byte1 >> 7) & 0x01) | ((byte2 & 0x03) << 1)
    v6 = (byte2 >> 2) & 0x07
    v7 = (byte2 >> 5) & 0x07

    result = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)
    return result.reshape(*batch_shape, n_groups * 8)[..., :N]


# ---------------------------------------------------------------------------
# 4-bit packing / unpacking (2 values per byte)
# ---------------------------------------------------------------------------


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices (0-15) into uint8, 2 per byte.

    Layout: byte = lo_nibble | (hi_nibble << 4)
    where lo = indices[..., 2i], hi = indices[..., 2i+1]

    Args:
        indices: int tensor (..., N) with values in [0, 15], N must be even

    Returns:
        packed: uint8 tensor (..., N//2)
    """
    assert indices.shape[-1] % 2 == 0, "Last dim must be even for 4-bit packing"
    lo = indices[..., 0::2].to(torch.uint8)
    hi = indices[..., 1::2].to(torch.uint8)
    return lo | (hi << 4)


def unpack_4bit(packed: torch.Tensor, N: int) -> torch.Tensor:
    """Unpack uint8 -> 4-bit indices.

    Args:
        packed: uint8 tensor (..., ceil(N/2))
        N: original last dimension

    Returns:
        indices: int32 tensor (..., N)
    """
    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)
    result = torch.stack([lo, hi], dim=-1)
    return result.reshape(*packed.shape[:-1], -1)[..., :N]


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------


def pack_bits(indices: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Pack indices at the given bit width.

    Args:
        indices: int tensor with values in [0, 2^bit_width - 1]
        bit_width: 2 or 4

    Returns:
        packed: uint8 tensor
    """
    if bit_width == 2:
        return pack_2bit(indices)
    if bit_width == 3:
        return pack_3bit(indices)
    if bit_width == 4:
        return pack_4bit(indices)
    raise ValueError(f"Unsupported bit_width={bit_width}. Supported: 2, 3, 4")


def unpack_bits(packed: torch.Tensor, N: int, bit_width: int) -> torch.Tensor:
    """Unpack indices at the given bit width.

    Args:
        packed: uint8 tensor
        N: original last dimension
        bit_width: 2 or 4

    Returns:
        indices: int32 tensor (..., N)
    """
    if bit_width == 2:
        return unpack_2bit(packed, N)
    if bit_width == 3:
        return unpack_3bit(packed, N)
    if bit_width == 4:
        return unpack_4bit(packed, N)
    raise ValueError(f"Unsupported bit_width={bit_width}. Supported: 2, 3, 4")
