"""Lloyd-Max optimal codebook computation for Gaussian distribution."""

from __future__ import annotations

import numpy as np
import torch


def _compute_lloyd_max_gaussian(
    n_levels: int, n_iters: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal Lloyd-Max codebook for N(0,1) distribution.

    Returns:
        centroids: (n_levels,) — sorted codebook centroids
        boundaries: (n_levels+1,) — quantization boundaries
    """
    from scipy.stats import norm

    sigma = 1.0
    boundaries = np.linspace(-3.5 * sigma, 3.5 * sigma, n_levels + 1)
    boundaries[0] = -1e10
    boundaries[-1] = 1e10
    centroids = np.zeros(n_levels)

    for _ in range(n_iters):
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            p = norm.cdf(hi) - norm.cdf(lo)
            if p > 1e-15:
                centroids[i] = (norm.pdf(lo) - norm.pdf(hi)) / p
            else:
                centroids[i] = (max(lo, -3.5) + min(hi, 3.5)) / 2

        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    return centroids, boundaries


_CODEBOOK_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def get_codebook(bit_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get precomputed Lloyd-Max codebook for given bit-width.

    Args:
        bit_width: bits per element (1, 2, 3, 4, ...)

    Returns:
        centroids: (2^bit_width,) float32 tensor
        boundaries: (2^bit_width - 1,) float32 tensor (inner boundaries only)
    """
    if bit_width not in _CODEBOOK_CACHE:
        n_levels = 2**bit_width
        centroids, boundaries = _compute_lloyd_max_gaussian(n_levels)
        _CODEBOOK_CACHE[bit_width] = (
            torch.tensor(centroids, dtype=torch.float32),
            torch.tensor(boundaries[1:-1], dtype=torch.float32),
        )
    return _CODEBOOK_CACHE[bit_width]
