import os
import torch
import timm
import numpy as np
import pandas as pd
from library.config import DEVICE, MODEL_NAME


import torch.nn as nn


class SkinLesionModel(nn.Module):
    """
    End-to-end model combining a MobileNetV3 backbone and tabular data.
    """

    def __init__(self, tabular_dim):
        super().__init__()
        # Create the model using timm.
        try:
            self.backbone = timm.create_model(
                MODEL_NAME, pretrained=True, num_classes=0, global_pool="avg"
            )
        except Exception:
            self.backbone = timm.create_model(
                "mobilenetv3_large_100",
                pretrained=True,
                num_classes=0,
                global_pool="avg",
            )

        # We do NOT freeze the backbone. Fine-tuning is required.
        # Cite solution_lesson_node_00001

        # Feature dimension from backbone + tabular dimension
        self.head = nn.Sequential(
            nn.Linear(1280 + tabular_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, images, tabular):
        """
        Args:
            images (torch.Tensor): (B, C, H, W)
            tabular (torch.Tensor): (B, Tabular_Dim)
        Returns:
            torch.Tensor: Logits (B, 1)
        """
        img_feats = self.backbone(images)
        combined = torch.cat([img_feats, tabular], dim=1)
        return self.head(combined)
