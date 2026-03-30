"""Random rotation generation for TurboQuant.

Two methods:
  - "qr":       Haar-distributed random orthogonal matrix via QR (O(d²) storage + compute).
  - "hadamard": Randomised Walsh-Hadamard: sign-flip + fast Walsh-Hadamard transform
                (O(d) storage, O(d log d) compute).  Requires d to be a power of 2.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# QR-based random rotation (Haar-distributed)
# ---------------------------------------------------------------------------


def generate_rotation_matrix(d: int, seed: int = 42, device: str = "cpu") -> torch.Tensor:
    """Generate a Haar-distributed random orthogonal matrix via QR decomposition.

    The resulting matrix maps any unit vector to a nearly uniform point on
    the unit hypersphere, making coordinates approximately independent
    with N(0, 1/d) distribution.

    Args:
        d: dimension
        seed: random seed for reproducibility
        device: device to compute on ("cpu" or "cuda"). Using GPU is much
                faster for large dimensions (>1000).

    Returns:
        Q: orthogonal matrix of shape (d, d), float32
    """
    # Generate random matrix on CPU for deterministic seeding,
    # then move to target device for fast QR
    gen = torch.Generator().manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    if device != "cpu":
        try:
            G = G.to(device)
        except torch.cuda.OutOfMemoryError:
            pass  # fall back to CPU QR
    Q, R = torch.linalg.qr(G)
    # Fix sign ambiguity to get proper Haar distribution
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q


# ---------------------------------------------------------------------------
# Hadamard + random signs rotation
# ---------------------------------------------------------------------------


def _generate_signs(d: int, seed: int) -> torch.Tensor:
    """Generate a vector of d random +/-1 signs."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (d,), generator=gen).float() * 2 - 1


def _fwht(X: torch.Tensor) -> torch.Tensor:
    """Unnormalised Fast Walsh-Hadamard Transform along last dimension.

    X: (..., d) where d must be a power of 2.
    Returns: (..., d)
    """
    d = X.shape[-1]
    h = 1
    while h < d:
        # Reshape so that pairs at distance h are adjacent
        X = X.view(*X.shape[:-1], d // (2 * h), 2, h)
        a = X[..., 0, :]
        b = X[..., 1, :]
        X = torch.stack([a + b, a - b], dim=-2)
        X = X.view(*X.shape[:-3], d)
        h *= 2
    return X


def hadamard_rotate(X: torch.Tensor, seed: int) -> torch.Tensor:
    """Forward Hadamard rotation: equivalent to X @ Pi^T with Pi = H D / sqrt(d).

    Steps: multiply each row by random +/-1 signs, apply normalised FWHT.

    Args:
        X: (..., d) tensor, d must be a power of 2.
        seed: random seed for the sign vector.
    Returns:
        (..., d) rotated tensor.
    """
    d = X.shape[-1]
    signs = _generate_signs(d, seed).to(X.device, X.dtype)
    return _fwht(X * signs) / math.sqrt(d)


def hadamard_rotate_inverse(Y: torch.Tensor, seed: int) -> torch.Tensor:
    """Inverse Hadamard rotation: equivalent to Y @ Pi with Pi = H D / sqrt(d).

    Steps: apply normalised FWHT, then multiply by the same signs.

    Args:
        Y: (..., d) tensor, d must be a power of 2.
        seed: random seed (must match forward).
    Returns:
        (..., d) tensor.
    """
    d = Y.shape[-1]
    signs = _generate_signs(d, seed).to(Y.device, Y.dtype)
    return _fwht(Y) / math.sqrt(d) * signs
