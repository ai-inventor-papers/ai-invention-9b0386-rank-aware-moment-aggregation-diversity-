"""RAMA: Rank-Aware Moment Aggregation for PyG.

Enriches mean pooling with gated per-dimension variance,
where gate is conditioned on effective spectral rank proxy and cardinality.
Zero-initialized so RAMA starts as pure mean aggregation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation


class RAMAAggregation(Aggregation):
    """Rank-Aware Moment Aggregation.

    Enriches mean pooling with gated per-dimension variance,
    where gate is conditioned on effective spectral rank and cardinality.
    Zero-initialized so RAMA starts as pure mean aggregation.
    """

    def __init__(self, channels: int, gate_hidden: int = 16, eps: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.gate_hidden = gate_hidden
        self.eps = eps

        # Variance transform
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
        # Small random init for W_g to break symmetry
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
        p = var / var_sum
        p = p.clamp(min=self.eps)
        H = -(p * p.log()).sum(dim=-1, keepdim=True)
        log_d = math.log(max(self.channels, 2))  # avoid log(1)=0
        rho = H / log_d  # normalized to ~[0, 1]

        # Step 5: Gate [dim_size, d]
        gate_input = torch.cat([torch.log1p(N), rho], dim=-1)
        g = torch.sigmoid(self.gate_out(F.relu(self.W_g(gate_input))))

        # Step 6: Output = mean + gated variance transform
        out = mu + g * self.W_sigma(var)
        return out
