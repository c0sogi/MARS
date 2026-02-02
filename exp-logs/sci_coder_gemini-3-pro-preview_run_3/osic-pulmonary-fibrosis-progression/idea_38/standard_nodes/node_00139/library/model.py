import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

from library.config import Config
from library.utils import laplace_log_likelihood

# =========================================================================
# Model Architecture
# =========================================================================


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPR-Net).
    Cite solution_lesson_node_00052: Use a linear residual stream for strong autoregressive signals.
    """

    def __init__(self):
        super().__init__()

        # Stream A: Linear Residual Stream (Baseline FVC + Time)
        # Cite solution_lesson_node_00060: Over-parameterize the linear stream.
        self.linear_stream = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64)
        )

        # Stream B: Deep Interaction Stream
        # Visual Backbone
        self.backbone = timm.create_model(
            Config.BACKBONE_ARCH,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )
        self._freeze_backbone()

        # Clinical Encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(Config.CLINICAL_INPUT_DIM, 64), nn.ReLU()
        )

        # Deep Fusion Layer
        # Input: Visual (1408) + Clinical (64)
        self.deep_fusion = nn.Linear(self.backbone.num_features + 64, 64)

        # Shared Head
        # Cite solution_lesson_node_00126: No dropout on residual branch.
        self.head = nn.Linear(64, 2)

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, child in self.backbone.named_children():
            if name == "blocks":
                for i, block in enumerate(child):
                    if i >= 5:
                        for param in block.parameters():
                            param.requires_grad = True
            elif name in ["conv_head", "bn2"]:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, image, clinical, linear_input):
        # Stream A
        h_lin = self.linear_stream(linear_input)

        # Stream B
        feat_img = self.backbone(image)
        feat_clin = self.clinical_encoder(clinical)
        feat_deep = torch.cat([feat_img, feat_clin], dim=1)
        h_deep = self.deep_fusion(feat_deep)

        # Latent Fusion (Summation)
        h_fused = h_lin + h_deep

        # Prediction
        out = self.head(h_fused)
        mu = out[:, 0].unsqueeze(1)
        sigma = F.softplus(out[:, 1].unsqueeze(1)) + 1e-6

        return mu, sigma


# =========================================================================
# Training & Evaluation Logic
# =========================================================================


def evaluate_model(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    Performs denormalization to ensure metric is calculated on original scale.
    """
    model.eval()
    total_metric = 0.0

    # Retrieve normalization stats from dataset
    fvc_mean = loader.dataset.stats.get("fvc_mean", 2500.0)
    fvc_std = loader.dataset.stats.get("fvc_std", 500.0)

    with torch.no_grad():
        for images, clinical, linear_input, targets, _ in loader:
            images = images.to(device)
            clinical = clinical.to(device)
            linear_input = linear_input.to(device)
            targets = targets.to(device)  # Normalized targets

            mu_final, sigma_final = model(images, clinical, linear_input)

            # Denormalize predictions and targets
            mu_denorm = mu_final * fvc_std + fvc_mean
            sigma_denorm = sigma_final * fvc_std
            targets_denorm = targets * fvc_std + fvc_mean

            # Calculate metric (LLL)
            batch_metric = laplace_log_likelihood(
                targets_denorm, mu_denorm, sigma_denorm
            )
            total_metric += batch_metric.item() * images.size(0)

    return total_metric / len(loader.dataset)


# train_model function removed as it is superseded by library/train.py logic.
# Keeping file clean.


# predict function removed as it is superseded by library/train.py logic.
