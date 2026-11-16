from __future__ import annotations

import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,
                 groups: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c: int, shortcut: bool = True, k: int = 3):
        super().__init__()
        self.cv1 = Conv(c, c, k)
        self.cv2 = Conv(c, c, k)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class C3k(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 2,
                 shortcut: bool = True):
        super().__init__()
        hidden = int(c2 * 0.5)
        self.cv1 = Conv(c1, hidden, 1)
        self.cv2 = Conv(c1, hidden, 1)
        self.blocks = nn.Sequential(*[
            Bottleneck(hidden, shortcut=shortcut) for _ in range(n)
        ])
        self.cv3 = Conv(hidden * 2, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), dim=1))


class PSAAttention(nn.Module):
    """Spatial multi-head attention, with a depthwise positional encoding."""
    def __init__(self, dim: int, ratio: float = 0.5):
        super().__init__()
        self.num_heads = max(dim // 64, 1)
        if dim % self.num_heads:
            raise ValueError("Attention channels must be divisible by heads.")
        self.head_dim = dim // self.num_heads
        self.key_dim = int(self.head_dim * ratio)
        self.scale = self.key_dim ** -0.5
        self.qkv = Conv(dim, dim + 2 * self.num_heads * self.key_dim, 1, act=False)
        self.positional = Conv(dim, dim, 3, groups=dim, act=False)
        self.projection = Conv(dim, dim, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x).reshape(
            b, self.num_heads, 2 * self.key_dim + self.head_dim, n
        )
        q, k, v = qkv.split(
            (self.key_dim, self.key_dim, self.head_dim), dim=2
        )
        weights = ((q * self.scale).transpose(-2, -1) @ k).softmax(-1)
        y = (v @ weights.transpose(-2, -1)).reshape(b, c, h, w)
        y = y + self.positional(v.reshape(b, c, h, w))
        return self.projection(y)


class PSABlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.attn = PSAAttention(c)
        self.ffn = nn.Sequential(
            Conv(c, 2 * c, 1), Conv(2 * c, c, 1, act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.ffn(x)


class C2PSA(nn.Module):
    def __init__(self, channels: int, n: int = 1):
        super().__init__()
        hidden = channels // 2
        self.cv1 = Conv(channels, hidden * 2, 1)
        self.blocks = nn.Sequential(*[PSABlock(hidden) for _ in range(n)])
        self.cv2 = Conv(hidden * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, dim=1)
        return self.cv2(torch.cat([a, self.blocks(b)], dim=1))


class C3k2(nn.Module):
    """CSP-style C2f block with a C3k, Bottleneck or PSA inner module."""
    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False,
                 expansion: float = 0.5, attn: bool = False):
        super().__init__()
        hidden = int(c2 * expansion)
        self.cv1 = Conv(c1, 2 * hidden, 1)
        self.blocks = nn.ModuleList()
        for _ in range(n):
            if attn:
                self.blocks.append(nn.Sequential(Bottleneck(hidden), PSABlock(hidden)))
            elif c3k:
                self.blocks.append(C3k(hidden, hidden, n=2))
            else:
                self.blocks.append(Bottleneck(hidden))
        self.cv2 = Conv((n + 2) * hidden, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, dim=1)
        outputs = [a, b]
        for block in self.blocks:
            outputs.append(block(outputs[-1]))
        return self.cv2(torch.cat(outputs, dim=1))


class SPPF(nn.Module):
    """Three 5x5 max-pooling iterations and optional YOLO26 residual."""
    def __init__(self, channels: int, k: int = 5, n: int = 3,
                 shortcut: bool = True):
        super().__init__()
        hidden = channels // 2
        self.cv1 = Conv(channels, hidden, 1, act=False)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)
        self.cv2 = Conv((n + 1) * hidden, channels, 1)
        self.n = n
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [self.cv1(x)]
        for _ in range(self.n):
            outputs.append(self.pool(outputs[-1]))
        y = self.cv2(torch.cat(outputs, dim=1))
        return x + y if self.shortcut else y
