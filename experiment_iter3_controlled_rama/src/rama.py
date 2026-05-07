"""RAMA: Rank-Aware Moment Aggregation for PyG.

Three aggregation variants:
  1. RAMAAggregation  - full RAMA with rank + cardinality gating
  2. RAMANoRankAggregation - ablation with fixed rho=0.5 (cardinality-only gating)

Both are zero-initialized so they start as pure mean aggregation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation


class RAMAAggregation(Aggregation):
    """Rank-Aware Moment Aggregation (full version).

    Enriches mean pooling with gated per-dimension variance,
    where gate is conditioned on effective spectral rank proxy and cardinality.
    Zero-initialized so RAMA starts as pure mean aggregation.
    """

    def __init__(self, channels: int, gate_hidden: int = 16, eps: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.gate_hidden = gate_hidden
        self.eps = eps

        # Variance transform: Linear(d, d)
        self.W_sigma = nn.Linear(channels, channels, bias=False)
        # Gate MLP: [log(N+1), rho] -> hidden -> d
        self.W_g = nn.Linear(2, gate_hidden)
        self.gate_out = nn.Linear(gate_hidden, channels)

        self.reset_parameters()

    def reset_parameters(self):
        """Zero-init so RAMA = mean at start."""
        nn.init.zeros_(self.W_sigma.weight)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:
        # Step 1: Mean [dim_size, d]
        mu = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")

        # Step 2: Variance via E[X^2] - E[X]^2 [dim_size, d]
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce="mean")
        var = (mean_x2 - mu * mu).clamp(min=0)

        # Step 3: Cardinality N per parent [dim_size, 1]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        N = self.reduce(ones, index, ptr, dim_size, dim, reduce="sum")

        # Step 4: Effective rank proxy (variance entropy) [dim_size, 1]
        var_sum = var.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        p = (var / var_sum).clamp(min=self.eps)  # CRITICAL: clamp BEFORE log
        H = -(p * p.log()).sum(dim=-1, keepdim=True)
        log_d = math.log(max(self.channels, 2))
        rho = (H / log_d).clamp(0, 1)  # CRITICAL: clamp output

        # Step 5: NaN-safe gate [dim_size, d]
        log_N = torch.log1p(N)
        gate_input = torch.nan_to_num(
            torch.cat([log_N, rho], dim=-1), nan=0.0
        )
        g = torch.sigmoid(self.gate_out(F.relu(self.W_g(gate_input))))

        # Step 6: Output = mean + gated variance transform
        out = mu + g * self.W_sigma(var)
        return out


class RAMANoRankAggregation(Aggregation):
    """RAMA ablation: fixed rho=0.5, cardinality-only gating.

    Same architecture and parameter count as full RAMA, but replaces the
    variance entropy rank proxy with a fixed constant of 0.5. This tests
    whether the rank conditioning signal matters.
    """

    def __init__(self, channels: int, gate_hidden: int = 16, eps: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.gate_hidden = gate_hidden
        self.eps = eps

        self.W_sigma = nn.Linear(channels, channels, bias=False)
        self.W_g = nn.Linear(2, gate_hidden)
        self.gate_out = nn.Linear(gate_hidden, channels)

        self.reset_parameters()

    def reset_parameters(self):
        """Zero-init so starts as mean."""
        nn.init.zeros_(self.W_sigma.weight)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:
        # Step 1: Mean
        mu = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")

        # Step 2: Variance
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce="mean")
        var = (mean_x2 - mu * mu).clamp(min=0)

        # Step 3: Cardinality
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        N = self.reduce(ones, index, ptr, dim_size, dim, reduce="sum")

        # Step 4: Fixed rho=0.5 (no rank computation)
        fixed_rho = torch.full_like(N, 0.5)

        # Step 5: Gate with cardinality-only conditioning
        log_N = torch.log1p(N)
        gate_input = torch.cat([log_N, fixed_rho], dim=-1)
        g = torch.sigmoid(self.gate_out(F.relu(self.W_g(gate_input))))

        # Step 6: Output
        out = mu + g * self.W_sigma(var)
        return out
