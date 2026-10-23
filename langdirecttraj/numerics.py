from __future__ import annotations

import torch


def safe_atan2(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Evaluate atan2 with finite gradients for zero-length geometry vectors."""

    if not (y.is_floating_point() and x.is_floating_point()):
        raise TypeError("safe_atan2 expects floating-point tensors")
    dtype = torch.promote_types(y.dtype, x.dtype)
    threshold = max(float(eps), float(torch.finfo(dtype).eps))
    degenerate = (y.abs() + x.abs()) < threshold
    safe_y = torch.where(degenerate, torch.zeros_like(y), y)
    safe_x = torch.where(degenerate, torch.full_like(x, threshold), x)
    return torch.atan2(safe_y, safe_x)
