import torch
import torch.nn as nn
from typing import Union, List


class LayerNorm(nn.Module):
    """
    Custom LayerNorm implementation without using nn.LayerNorm or F.layer_norm.

    Normalizes over the last len(normalized_shape) dimensions of the input.
    For a tensor of shape [B, C, L] with normalized_shape=[C, L], normalization
    is computed over the [C, L] slice of each batch element independently.

    y = (x - mean) / sqrt(var + eps) * weight + bias
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = list(normalized_shape)
        self.eps = eps

        if len(self.normalized_shape) != 1:
            raise ValueError(
                "LayerNorm expects a single channel dimension; "
                f"got normalized_shape={self.normalized_shape}"
            )

        channels = self.normalized_shape[0]
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
