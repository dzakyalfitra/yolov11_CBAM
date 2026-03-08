"""Full CBAM (Convolutional Block Attention Module) from the original paper.

Implements the complete CBAM with MLP bottleneck channel attention and dual-pooling
spatial attention, as described in: https://arxiv.org/abs/1807.06521

This version is adapted for seamless integration with Ultralytics YOLO parse_model,
which calls CBAM(c1) with only the input channel count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module with shared MLP bottleneck (from CBAM paper).

    Uses both average-pooled and max-pooled features through a shared two-layer MLP,
    then combines them with element-wise summation before sigmoid activation.
    """

    def __init__(self, channels, reduction_ratio=16):
        """Initialize channel attention with bottleneck MLP.

        Args:
            channels (int): Number of input channels.
            reduction_ratio (int): Reduction ratio for the bottleneck MLP.
        """
        super().__init__()
        mid_channels = max(channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
        )

    def forward(self, x):
        """Apply channel attention: AvgPool + MaxPool → shared MLP → sigmoid."""
        avg_out = self.shared_mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.shared_mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module (from CBAM paper).

    Concatenates average-pooled and max-pooled features along channel axis,
    then applies a convolution to produce a spatial attention map.
    """

    def __init__(self, kernel_size=7):
        """Initialize spatial attention with conv layer.

        Args:
            kernel_size (int): Kernel size for spatial conv (must be odd, typically 7).
        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        """Apply spatial attention: channel AvgPool + MaxPool → Conv → sigmoid."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (full paper version).

    Sequentially applies channel attention then spatial attention to refine features.
    Compatible with Ultralytics parse_model: called as CBAM(c1).

    Reference: https://arxiv.org/abs/1807.06521
    """

    def __init__(self, c1, reduction_ratio=16, kernel_size=7):
        """Initialize full CBAM module.

        Args:
            c1 (int): Number of input/output channels (channel-preserving).
            reduction_ratio (int): Reduction ratio for channel attention MLP bottleneck.
            kernel_size (int): Kernel size for spatial attention convolution.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel attention then spatial attention sequentially."""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x