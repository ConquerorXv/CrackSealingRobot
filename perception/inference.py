from __future__ import annotations

import cv2
import numpy as np
import torch

from .model import LiteCMNet
from .utils import choose_device, load_weights
from .data import depth_to_float


class Predictor:
    def __init__(self, weights, device='auto', threshold=.5):
        self.device = choose_device(device)
        if not 0 < threshold < 1:
            raise ValueError('Threshold must be between zero and one')
        self.threshold = threshold
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = LiteCMNet().to(self.device).eval()
        load_weights(self.model, weights, self.device)

    @torch.inference_mode()
    def predict(self, observation):
        if observation.rgb.shape != (736, 1280, 3) or observation.depth_mm.shape != (736, 1280):
            raise ValueError('The deployed model expects aligned 1280x736 RGB-D observations')
        depth = depth_to_float(observation.depth_mm, normalization='per_image_max')
        rgb = torch.from_numpy(np.ascontiguousarray(observation.rgb.transpose(2, 0, 1))).float()
        depth = torch.from_numpy(np.ascontiguousarray(depth[None, None]))
        logits = self.model(rgb[None].to(self.device)/255., depth.to(self.device))
        pred = (logits.sigmoid()[0, 0] >= self.threshold).cpu().numpy()
        pred &= observation.valid
        pred[:8] = False
        pred[-8:] = False
        return pred.astype(np.uint8)*255


def overlay(rgb, mask, alpha=.5):
    result = rgb.copy()
    foreground = mask > 0
    result[foreground] = np.rint((1-alpha)*rgb[foreground].astype(float)
                                 + alpha*np.array([255, 0, 0])).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
