import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from library.config import Config


class HPF(nn.Module):
    """
    Fixed High-Pass Filter layer using the KV kernel.
    """

    def __init__(self):
        super().__init__()
        # KV Kernel (5x5)
        # Cite solution_lesson_node_00001
        kv_kernel = (
            np.array(
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                dtype=np.float32,
            )
            / 12.0
        )

        self.weight = nn.Parameter(
            torch.from_numpy(kv_kernel).view(1, 1, 5, 5), requires_grad=False
        )
        self.gray_weights = nn.Parameter(
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
            requires_grad=False,
        )

    def forward(self, x):
        # Scale to [0, 255]
        x = x * 255.0
        # RGB to Gray
        x = F.conv2d(x, self.gray_weights)
        # Apply HPF
        x = F.conv2d(x, self.weight, padding=2)
        return x


class HPF_EfficientNet(nn.Module):
    """
    Single-Stream HPF-CNN using EfficientNet-B0.
    Cite solution_lesson_node_00001
    """

    def __init__(self):
        super().__init__()
        self.hpf = HPF()
        # EfficientNet-B0 with 1 input channel (grayscale residuals)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=True,
            in_chans=1,
            num_classes=Config.NUM_CLASSES,
        )

    def forward(self, x):
        x = self.hpf(x)
        x = self.backbone(x)
        return x
