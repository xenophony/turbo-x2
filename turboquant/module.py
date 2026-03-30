"""TurboQuantLinear — Drop-in nn.Linear replacement with on-the-fly dequantization.

Stores weights as packed n-bit indices + per-row norms + shared codebook.
Supports 2-bit and 4-bit quantization.

On-the-fly forward (pre-rotate input):
  1. x_rot = x @ Pi.T           (rotate input, not weight)
  2. out = x_rot @ codebook[indices].T  (lookup + matmul)
  3. out = out * (norms / scale)  (rescale per output row)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from turboquant.rotation import (
    generate_rotation_matrix,
    hadamard_rotate,
    hadamard_rotate_inverse,
)
from turboquant.packing import pack_bits, unpack_bits
from turboquant.codebook import get_codebook
from turboquant.quantize import _resolve_rotation


class TurboQuantLinear(nn.Module):
    """Linear layer with TurboQuant-compressed weights and on-the-fly dequantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bit_width: int = 2,
        group_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        rotation: str = "hadamard",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width
        self.group_size = group_size or in_features
        self.n_levels = 2**bit_width
        self.rotation = rotation

        pack_factor = 8 // bit_width
        packed_dim = math.ceil(in_features / pack_factor)
        n_groups = math.ceil(in_features / self.group_size)

        # Pass 1 buffers
        self.register_buffer(
            "indices_packed",
            torch.zeros(out_features, packed_dim, dtype=torch.uint8, device=device),
        )
        if n_groups == 1:
            self.register_buffer(
                "weight_norms",
                torch.ones(out_features, dtype=torch.float32, device=device),
            )
        else:
            self.register_buffer(
                "weight_norms",
                torch.ones(out_features, n_groups, dtype=torch.float32, device=device),
            )
        self.register_buffer(
            "codebook",
            torch.zeros(self.n_levels, dtype=torch.float32, device=device),
        )

        # Pass 2 (residual) buffers — None until set
        self.register_buffer("pass2_indices_packed", None)
        self.register_buffer("pass2_weight_norms", None)
        self.register_buffer("pass2_codebook", None)
        self._pass2_seed: Optional[int] = None
        self._pass2_bit_width: Optional[int] = None

        # Bias
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float32, device=device),
            )
        else:
            self.bias = None

        # Rotation cache: dict[seed_offset -> Pi tensor]
        self._rotation_cache: dict[int, torch.Tensor] = {}
        self._rotation_seed: int = 42
        self._scale: float = math.sqrt(self.group_size)

        # Cached unpacked indices (lazy, freed on device change)
        self._cached_indices: Optional[torch.Tensor] = None
        self._cached_pass2_indices: Optional[torch.Tensor] = None
        self._n_groups: int = n_groups

    def set_rotation(self, seed: int):
        self._rotation_seed = seed
        self._rotation_cache.clear()

    def set_pass2(
        self,
        indices_packed: torch.Tensor,
        weight_norms: torch.Tensor,
        codebook: torch.Tensor,
        seed: int,
        bit_width: int,
    ):
        """Set residual (pass 2) quantization data."""
        self.register_buffer("pass2_indices_packed", indices_packed)
        self.register_buffer("pass2_weight_norms", weight_norms)
        self.register_buffer("pass2_codebook", codebook)
        self._pass2_seed = seed
        self._pass2_bit_width = bit_width
        self._cached_pass2_indices = None

    @property
    def has_residual(self) -> bool:
        return self.pass2_indices_packed is not None

    def _get_rotation(self, seed: int, g_start: int = 0, dim: Optional[int] = None) -> torch.Tensor:
        d = dim or self.group_size
        key = (seed + g_start, d)
        if key not in self._rotation_cache:
            # Try GPU first, fall back to CPU for large matrices
            target_device = str(self.indices_packed.device)
            Q = generate_rotation_matrix(d, seed=seed + g_start, device=target_device)
            # Keep large rotation matrices on CPU to save GPU memory
            if Q.device.type != "cpu" and Q.numel() > 10_000_000:  # >10M elements ~40MB
                Q = Q.cpu()
            self._rotation_cache[key] = Q
        return self._rotation_cache[key]

    def _get_indices(self) -> torch.Tensor:
        """Get unpacked indices (on-the-fly, no caching to save GPU memory)."""
        return unpack_bits(self.indices_packed, self.in_features, self.bit_width)

    def _get_pass2_indices(self) -> torch.Tensor:
        if self.pass2_indices_packed is not None:
            bw = self._pass2_bit_width or self.bit_width
            return unpack_bits(self.pass2_indices_packed, self.in_features, bw)
        return None

    def _forward_pass(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None,
        indices_packed: torch.Tensor,
        codebook: torch.Tensor,
        weight_norms: torch.Tensor,
        seed: int,
        bit_width: int,
    ) -> torch.Tensor:
        """Single-pass on-the-fly dequant matmul with group-wise rotation.

        Args:
            x: (B, K) input (float32)
            indices: (N, K) unpacked int32 or None
            indices_packed: (N, packed_dim) packed uint8
            codebook: (n_levels,) float32
            weight_norms: (N,) or (N, n_groups) float32
            seed: base rotation seed
            bit_width: bit width for unpacking

        Returns:
            output: (B, N) float32
        """
        B = x.shape[0]
        N = indices_packed.shape[0]
        K = self.in_features
        device = x.device
        scale = self._scale

        output = torch.zeros(B, N, dtype=torch.float32, device=device)

        for g in range(self._n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, K)

            # Rotate this group's input slice
            g_dim = g_end - g_start
            rot = _resolve_rotation(self.rotation, g_dim)
            if rot == "hadamard":
                x_rot_g = hadamard_rotate(x[:, g_start:g_end], seed + g_start)
            else:
                Pi_g = self._get_rotation(seed, g_start, dim=g_dim).to(device)
                x_rot_g = x[:, g_start:g_end] @ Pi_g.T

            # Per-group norms
            if weight_norms.dim() == 1:
                norms_g = weight_norms
            else:
                norms_g = weight_norms[:, g]

            # PyTorch fallback: explicit unpack + lookup + matmul
            if indices is None:
                indices = unpack_bits(indices_packed, K, bit_width)
            idx_g = indices[:, g_start:g_end]
            W_g = codebook[idx_g.long()]
            out_g = x_rot_g @ W_g.T
            out_g = out_g * (norms_g[None, :] / scale)

            output += out_g

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """On-the-fly dequant forward pass with group-wise rotation."""
        device = x.device
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])

        x_f = x.float()

        # Pass 1
        indices = self._get_indices()
        output = self._forward_pass(
            x_f, indices, self.indices_packed, self.codebook,
            self.weight_norms, self._rotation_seed, self.bit_width,
        )

        # Pass 2 (residual) if present
        if self.has_residual:
            indices2 = self._get_pass2_indices()
            bw2 = self._pass2_bit_width or self.bit_width
            output += self._forward_pass(
                x_f, indices2, self.pass2_indices_packed, self.pass2_codebook,
                self.pass2_weight_norms, self._pass2_seed, bw2,
            )

        # Restore shape
        if len(orig_shape) == 3:
            output = output.reshape(orig_shape[0], orig_shape[1], self.out_features)

        out = output.to(x.dtype)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def dequantize(self) -> torch.Tensor:
        """Full dequantization: returns (M, N) bf16 weight (for debugging)."""
        indices = self._get_indices()
        scale = self._scale

        W = torch.zeros(
            self.out_features, self.in_features,
            dtype=torch.float32, device=self.indices_packed.device,
        )

        for g in range(self._n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, self.in_features)
            g_dim = g_end - g_start
            rot = _resolve_rotation(self.rotation, g_dim)

            if rot == "hadamard":
                Y_g = self.codebook[indices[:, g_start:g_end].long()] / scale
                W_g = hadamard_rotate_inverse(Y_g, self._rotation_seed + g_start)
            else:
                Pi_g = self._get_rotation(self._rotation_seed, g_start, dim=g_dim)
                Y_g = self.codebook[indices[:, g_start:g_end].long()] / scale
                W_g = Y_g @ Pi_g

            if self.weight_norms.dim() == 1:
                W_g = W_g * self.weight_norms.unsqueeze(1)
            else:
                W_g = W_g * self.weight_norms[:, g].unsqueeze(1)

            W[:, g_start:g_end] = W_g

        if self.has_residual:
            indices2 = self._get_pass2_indices()
            bw2 = self._pass2_bit_width or self.bit_width
            for g in range(self._n_groups):
                g_start = g * self.group_size
                g_end = min(g_start + self.group_size, self.in_features)
                g_dim = g_end - g_start
                rot = _resolve_rotation(self.rotation, g_dim)
                if rot == "hadamard":
                    Y_g = self.pass2_codebook[indices2[:, g_start:g_end].long()] / scale
                    W_g = hadamard_rotate_inverse(Y_g, self._pass2_seed + g_start)
                else:
                    Pi2_g = self._get_rotation(self._pass2_seed, g_start, dim=g_dim)
                    Y_g = self.pass2_codebook[indices2[:, g_start:g_end].long()] / scale
                    W_g = Y_g @ Pi2_g
                if self.pass2_weight_norms.dim() == 1:
                    W_g = W_g * self.pass2_weight_norms.unsqueeze(1)
                else:
                    W_g = W_g * self.pass2_weight_norms[:, g].unsqueeze(1)
                W[:, g_start:g_end] += W_g

        return W.to(torch.bfloat16)

    @torch.no_grad()
    def merge_passes(self) -> None:
        """Merge residual pass into the primary pass via rotated-domain addition."""
        if not self.has_residual:
            return

        device = self.indices_packed.device
        K = self.in_features
        N = self.out_features
        scale = self._scale

        centroids, boundaries = get_codebook(self.bit_width)
        centroids = centroids.to(device)
        boundaries = boundaries.to(device)

        same_rotation = self._pass2_seed == self._rotation_seed

        if same_rotation:
            indices1 = self._get_indices()
            indices2 = self._get_pass2_indices()

            all_merged_indices: list[torch.Tensor] = []
            all_merged_norms: list[torch.Tensor] = []

            for g in range(self._n_groups):
                g_start = g * self.group_size
                g_end = min(g_start + self.group_size, K)
                g_dim = g_end - g_start

                Y1 = self.codebook[indices1[:, g_start:g_end].long()] / scale
                n1 = (
                    self.weight_norms
                    if self.weight_norms.dim() == 1
                    else self.weight_norms[:, g]
                )
                Y1 = Y1 * n1.unsqueeze(1)

                Y2 = self.pass2_codebook[indices2[:, g_start:g_end].long()] / scale
                n2 = (
                    self.pass2_weight_norms
                    if self.pass2_weight_norms.dim() == 1
                    else self.pass2_weight_norms[:, g]
                )
                Y2 = Y2 * n2.unsqueeze(1)

                Y_merged = Y1 + Y2

                merged_norms = Y_merged.norm(dim=1, keepdim=True).clamp(min=1e-8)
                Y_norm = Y_merged / merged_norms

                Y_scaled = Y_norm * scale
                idx = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
                idx = idx.clamp(0, len(centroids) - 1).reshape(N, g_dim)

                all_merged_indices.append(idx)
                all_merged_norms.append(merged_norms.squeeze(1))

            full_indices = torch.cat(all_merged_indices, dim=1)
            norms_out = (
                torch.stack(all_merged_norms, dim=1)
                if len(all_merged_norms) > 1
                else all_merged_norms[0]
            )
        else:
            W_merged = self.dequantize().float()
            all_indices: list[torch.Tensor] = []
            all_norms: list[torch.Tensor] = []

            for g_start in range(0, K, self.group_size):
                g_end = min(g_start + self.group_size, K)
                g_dim = g_end - g_start
                W_g = W_merged[:, g_start:g_end]

                norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
                W_norm = W_g / norms
                all_norms.append(norms.squeeze(1))

                Pi = generate_rotation_matrix(
                    g_dim, seed=self._rotation_seed + g_start, device=str(device)
                ).to(device)
                Y = W_norm @ Pi.T
                Y_scaled = Y * scale

                idx = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
                idx = idx.clamp(0, len(centroids) - 1).reshape(N, g_dim)
                all_indices.append(idx)

            full_indices = torch.cat(all_indices, dim=1)
            norms_out = (
                torch.stack(all_norms, dim=1)
                if len(all_norms) > 1
                else all_norms[0]
            )

        # Pad and pack using generic dispatcher
        pack_factor = 8 // self.bit_width
        remainder = K % pack_factor
        if remainder != 0:
            pad_size = pack_factor - remainder
            full_indices = torch.nn.functional.pad(full_indices, (0, pad_size), value=0)

        packed = pack_bits(full_indices, self.bit_width)
        self.indices_packed.copy_(packed)
        self.weight_norms.copy_(norms_out)
        self.codebook.copy_(centroids)

        # Clear residual buffers
        self.register_buffer("pass2_indices_packed", None)
        self.register_buffer("pass2_weight_norms", None)
        self.register_buffer("pass2_codebook", None)
        self._pass2_seed = None
        self._pass2_bit_width = None
        self._cached_indices = None
        self._cached_pass2_indices = None

    def memory_bytes(self) -> int:
        """Compressed storage in bytes."""
        total = self.indices_packed.numel()  # uint8
        total += self.weight_norms.numel() * 4
        total += self.codebook.numel() * 4
        if self.bias is not None:
            total += self.bias.numel() * 4
        if self.pass2_indices_packed is not None:
            total += self.pass2_indices_packed.numel()
            total += self.pass2_weight_norms.numel() * 4
            total += self.pass2_codebook.numel() * 4
        return total

    def extra_repr(self) -> str:
        residual = ", residual=True" if self.has_residual else ""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bit_width={self.bit_width}, group_size={self.group_size}{residual}, "
            f"compressed={self.memory_bytes() / 1024:.1f} KB"
        )


class TurboQuantEmbedding(nn.Module):
    """Drop-in nn.Embedding replacement with TurboQuant-compressed weight table.

    Stores the embedding table as packed quantization indices + per-row norms +
    shared codebook. On forward, selected rows are dequantized on the fly:
      1. Unpack indices for requested token IDs
      2. Look up codebook centroids
      3. Inverse-rotate to recover embedding vectors
      4. Rescale by per-row norms
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bit_width: int = 4,
        group_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        rotation: str = "hadamard",
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.bit_width = bit_width
        self.group_size = group_size or embedding_dim
        self.n_levels = 2**bit_width
        self.rotation = rotation
        self.padding_idx = padding_idx

        pack_factor = 8 // bit_width
        packed_dim = math.ceil(embedding_dim / pack_factor)
        n_groups = math.ceil(embedding_dim / self.group_size)

        self.register_buffer(
            "indices_packed",
            torch.zeros(num_embeddings, packed_dim, dtype=torch.uint8, device=device),
        )
        if n_groups == 1:
            self.register_buffer(
                "weight_norms",
                torch.ones(num_embeddings, dtype=torch.float32, device=device),
            )
        else:
            self.register_buffer(
                "weight_norms",
                torch.ones(num_embeddings, n_groups, dtype=torch.float32, device=device),
            )
        self.register_buffer(
            "codebook",
            torch.zeros(self.n_levels, dtype=torch.float32, device=device),
        )

        self._rotation_seed: int = 42
        self._rotation_cache: dict[tuple, torch.Tensor] = {}
        self._scale: float = math.sqrt(self.group_size)
        self._n_groups: int = n_groups

    def set_rotation(self, seed: int):
        self._rotation_seed = seed
        self._rotation_cache.clear()

    def _get_rotation(self, g_start: int, dim: int) -> torch.Tensor:
        key = (self._rotation_seed + g_start, dim)
        if key not in self._rotation_cache:
            target_device = str(self.indices_packed.device)
            self._rotation_cache[key] = generate_rotation_matrix(
                dim, seed=self._rotation_seed + g_start, device=target_device
            )
        return self._rotation_cache[key]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up and dequantize embeddings for the given token IDs.

        Args:
            input_ids: (...) int tensor of token IDs

        Returns:
            embeddings: (..., embedding_dim) tensor
        """
        orig_shape = input_ids.shape
        flat_ids = input_ids.reshape(-1)

        # Gather packed indices and norms for requested tokens
        packed_rows = self.indices_packed[flat_ids]  # (N, packed_dim)
        indices = unpack_bits(packed_rows, self.embedding_dim, self.bit_width)  # (N, D)

        if self.weight_norms.dim() == 1:
            norms = self.weight_norms[flat_ids]  # (N,)
        else:
            norms = self.weight_norms[flat_ids]  # (N, n_groups)

        scale = self._scale
        D = self.embedding_dim
        result = torch.zeros(len(flat_ids), D, dtype=torch.float32, device=input_ids.device)

        for g in range(self._n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, D)
            g_dim = g_end - g_start
            rot = _resolve_rotation(self.rotation, g_dim)

            # Dequantize: codebook lookup → inverse rotate → rescale
            Y_g = self.codebook[indices[:, g_start:g_end].long()] / scale  # (N, g_dim)

            if rot == "hadamard":
                W_g = hadamard_rotate_inverse(Y_g, self._rotation_seed + g_start)
            else:
                Pi_g = self._get_rotation(g_start, g_dim)
                W_g = Y_g @ Pi_g  # (N, g_dim)

            if norms.dim() == 1:
                W_g = W_g * norms.unsqueeze(1)
            else:
                W_g = W_g * norms[:, g].unsqueeze(1)

            result[:, g_start:g_end] = W_g

        # Reshape to match input
        return result.reshape(*orig_shape, D).to(self.codebook.dtype)

    def memory_bytes(self) -> int:
        total = self.indices_packed.numel()
        total += self.weight_norms.numel() * 4
        total += self.codebook.numel() * 4
        return total

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"bit_width={self.bit_width}, group_size={self.group_size}, "
            f"compressed={self.memory_bytes() / 1024:.1f} KB"
        )
