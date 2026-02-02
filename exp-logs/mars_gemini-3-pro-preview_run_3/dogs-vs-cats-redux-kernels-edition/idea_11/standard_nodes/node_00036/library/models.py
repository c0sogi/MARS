import torch
import torch.nn as nn
import timm
from typing import List
from library.config import ModelConfig


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout (MSD).

    This module applies multiple dropout masks with different rates to the same
    input features, passes them through a shared linear layer, and averages
    the outputs. This acts as an internal ensemble, reducing overfitting and
    improving generalization, specifically useful for Log Loss optimization.
    """

    def __init__(self, in_features: int, out_features: int, dropout_rates: List[float]):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, In_Features).

        Returns:
            torch.Tensor: Averaged logits of shape (Batch_Size, Out_Features).
        """
        # Collect logits from each dropout path
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout then linear projection
            out = self.fc(dropout(x))
            logits_list.append(out)

        # Stack along a new dimension (Batch, Num_Drops, Out_Features)
        stacked_logits = torch.stack(logits_list, dim=1)

        # Average across the dropout dimension
        return torch.mean(stacked_logits, dim=1)


class CustomEnsembleModel(nn.Module):
    """
    Wrapper model that combines a TIMM backbone with a custom classification head.

    Supports:
    - Dynamic backbone loading via TIMM.
    - Multi-Sample Dropout head (Idea 11).
    - Standard Linear head (fallback).
    """

    def __init__(
        self, config: ModelConfig, num_classes: int = 1, pretrained: bool = True
    ):
        """
        Args:
            config (ModelConfig): Configuration object containing model name and dropout settings.
            num_classes (int): Number of output classes (1 for binary classification).
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super().__init__()
        self.config = config

        # Initialize backbone using timm
        # num_classes=0 removes the default classifier and applies global pooling,
        # returning a feature vector (Batch_Size, Num_Features).
        self.backbone = timm.create_model(
            config.model_name, pretrained=pretrained, num_classes=0
        )

        # Determine the input feature dimension for the head
        in_features = self.backbone.num_features

        # Initialize the classification head based on configuration
        if config.use_multi_sample_dropout:
            self.head = MultiSampleDropout(
                in_features=in_features,
                out_features=num_classes,
                dropout_rates=config.dropout_rates,
            )
        else:
            self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch_Size, Channels, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, Num_Classes).
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Pass through classification head
        logits = self.head(features)

        return logits
