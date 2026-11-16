"""RGB-D segmentation model.

Architecture attribution:
The perception architecture uses design concepts including C3k2, SPPF,
C2PSA and PAN-FPN from the Ultralytics YOLO family.

Architecture reference: https://github.com/ultralytics/ultralytics
Upstream licensing information: https://www.ultralytics.com/license
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import C2PSA, C3k2, Conv, SPPF


class RGBBackbone(nn.Module):
    """YOLO26s-inspired backbone, with channel plan 32/64/128/256/512."""
    def __init__(self):
        super().__init__()
        self.p1 = Conv(3, 32, 3, 2)
        self.p2_down = Conv(32, 64, 3, 2)
        self.p2 = C3k2(64, 64, n=1, c3k=False, expansion=0.25)
        self.p3_down = Conv(64, 128, 3, 2)
        self.p3 = C3k2(128, 128, n=1, c3k=False, expansion=0.25)
        self.p4_down = Conv(128, 256, 3, 2)
        self.p4 = C3k2(256, 256, n=1, c3k=True)
        self.p5_down = Conv(256, 512, 3, 2)
        self.p5 = C3k2(512, 512, n=1, c3k=True)
        self.sppf = SPPF(512, 5, 3, shortcut=True)
        self.psa = C2PSA(512, n=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.p1(image)
        x = self.p2(self.p2_down(x))
        p3 = self.p3(self.p3_down(x))
        p4 = self.p4(self.p4_down(p3))
        p5 = self.psa(self.sppf(self.p5(self.p5_down(p4))))
        return p3, p4, p5


class SqueezeExcitation(nn.Module):
    def __init__(self, expanded: int, hidden: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(expanded, hidden, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, expanded, 1, bias=True)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))


class MNv3Block(nn.Module):
    """Inverted residual: expansion -> 3x3 DWConv -> SE -> projection."""
    def __init__(self, channels: int, expansion: int, se_hidden: int):
        super().__init__()
        expanded = channels * expansion
        self.expand = Conv(channels, expanded, 1)
        self.depthwise = Conv(expanded, expanded, 3, 1, groups=expanded)
        self.se = SqueezeExcitation(expanded, se_hidden)
        self.project = nn.Sequential(
            nn.Conv2d(expanded, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.project(self.se(self.depthwise(self.expand(x))))


class DepthEncoder(nn.Module):
    """D1-D5: 16/32/64/128/256; e=4/4/4/4, SE=64/128/256/512."""
    def __init__(self):
        super().__init__()
        self.d1 = Conv(1, 16, 3, 2)
        self.d2 = nn.Sequential(Conv(16, 32, 3, 2), MNv3Block(32, 4, 64))
        self.d3 = nn.Sequential(Conv(32, 64, 3, 2), MNv3Block(64, 4, 128))
        self.d4 = nn.Sequential(Conv(64, 128, 3, 2), MNv3Block(128, 4, 256))
        self.d5 = nn.Sequential(Conv(128, 256, 3, 2), MNv3Block(256, 4, 512))

    def forward(self, depth: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.d2(self.d1(depth))
        d3 = self.d3(x)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        return d3, d4, d5


class CMSCA(nn.Module):
    """Depth-driven channel-spatial attention on an RGB feature map."""
    def __init__(self, depth_channels: int, rgb_channels: int, hidden: int):
        super().__init__()
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(depth_channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(depth_channels, depth_channels, 3, padding=1,
                      groups=depth_channels, bias=True),
            nn.Conv2d(depth_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError("RGB and depth feature maps must be spatially aligned.")
        return rgb * self.channel(depth) * self.spatial(depth)


class PANFPN(nn.Module):
    """Top-down FPN + bottom-up PAN; outputs P3/P4/P5=128/256/512."""
    def __init__(self):
        super().__init__()
        self.fpn_p4 = C3k2(512 + 256, 256, n=1, c3k=True)
        self.fpn_p3 = C3k2(256 + 128, 128, n=1, c3k=True)
        self.down_p4 = Conv(128, 128, 3, 2)
        self.pan_p4 = C3k2(128 + 256, 256, n=1, c3k=True)
        self.down_p5 = Conv(256, 256, 3, 2)
        self.pan_p5 = C3k2(256 + 512, 512, n=1, c3k=True,
                            expansion=0.5, attn=True)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor,
                p5: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n4 = self.fpn_p4(torch.cat([
            F.interpolate(p5, size=p4.shape[-2:], mode="nearest"), p4
        ], dim=1))
        n3 = self.fpn_p3(torch.cat([
            F.interpolate(n4, size=p3.shape[-2:], mode="nearest"), p3
        ], dim=1))
        out4 = self.pan_p4(torch.cat([self.down_p4(n3), n4], dim=1))
        out5 = self.pan_p5(torch.cat([self.down_p5(out4), p5], dim=1))
        return n3, out4, out5


class SegHead(nn.Module):
    """Aligned three-scale fusion followed by three 2x deconvolutions."""
    def __init__(self):
        super().__init__()
        self.fuse = Conv(128 + 256 + 512, 128, 1)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2),
                                     nn.BatchNorm2d(64), nn.SiLU(inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, 2),
                                     nn.BatchNorm2d(32), nn.SiLU(inplace=True))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, 2),
                                     nn.BatchNorm2d(16), nn.SiLU(inplace=True))
        self.classifier = nn.Conv2d(16, 1, 1)

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        p3, p4, p5 = features
        size = p3.shape[-2:]
        x = torch.cat([
            p3,
            F.interpolate(p4, size=size, mode="bilinear", align_corners=False),
            F.interpolate(p5, size=size, mode="bilinear", align_corners=False),
        ], dim=1)
        x = self.fuse(x)
        x = self.deconv3(self.deconv2(self.deconv1(x)))
        return self.classifier(x)


class LiteCMNet(nn.Module):
    """RGB [B,3,H,W] + aligned depth [B,1,H,W] -> binary logits [B,1,H,W]."""
    def __init__(self):
        super().__init__()
        self.rgb_backbone = RGBBackbone()
        self.depth_encoder = DepthEncoder()
        self.cmsca_p3 = CMSCA(64, 128, 128)
        self.cmsca_p4 = CMSCA(128, 256, 256)
        self.cmsca_p5 = CMSCA(256, 512, 512)
        self.pan_fpn = PANFPN()
        self.seg_head = SegHead()

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B, 3, H, W]")
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError("depth must have shape [B, 1, H, W]")
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError("rgb/depth batch size and spatial dimensions must match")
        h, w = rgb.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError("input height and width must be divisible by 32")
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_encoder(depth)
        fused = (
            self.cmsca_p3(rgb_features[0], depth_features[0]),
            self.cmsca_p4(rgb_features[1], depth_features[1]),
            self.cmsca_p5(rgb_features[2], depth_features[2]),
        )
        mask_logits = self.seg_head(self.pan_fpn(*fused))
        if mask_logits.shape[-2:] != (h, w):
            raise RuntimeError("segmentation output must match input resolution")
        return mask_logits
