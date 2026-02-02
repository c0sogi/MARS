import os
import torch
import timm
import numpy as np
import pandas as pd
from library.config import DEVICE, MODEL_NAME


class ISICNet(torch.nn.Module):
    """
    End-to-end trainable model combining MobileNetV3 backbone and tabular features.
    Cite solution_lesson_node_00001: Fine-tuning backbone is mandatory for medical domains.
    """

    def __init__(self, num_tabular_features=0):
        super().__init__()
        # Create the model using timm.
        # num_classes=0 removes the final classification layer.
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

        # Unfreeze backbone (default is requires_grad=True for created models, but explicit is good)
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Classification Head
        # Input: Backbone Features (1280) + Tabular Features
        input_dim = 1280 + num_tabular_features

        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),  # Output logit
        )

    def forward(self, images, tabular):
        """
        Args:
            images (torch.Tensor): (B, C, H, W)
            tabular (torch.Tensor): (B, Num_Tabular)
        """
        img_feats = self.backbone(images)
        combined = torch.cat([img_feats, tabular], dim=1)
        return self.head(combined)
