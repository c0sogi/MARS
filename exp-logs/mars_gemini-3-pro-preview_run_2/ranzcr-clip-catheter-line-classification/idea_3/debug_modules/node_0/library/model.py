import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean over the spatial dimensions.
    f(X) = (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6, p_trainable=True):
        super(GeM, self).__init__()
        # Initialize p. If trainable, wrap in nn.Parameter.
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid numerical instability with pow() near zero
        x = x.clamp(min=self.eps)

        # Apply power p
        x_pow = x.pow(self.p)

        # Compute average over spatial dimensions (H, W)
        # F.avg_pool2d with kernel size equal to input size effectively computes the mean
        # Output shape: (Batch, Channels, 1, 1)
        avg_x_pow = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Apply inverse power 1/p
        gem_features = avg_x_pow.pow(1.0 / self.p)

        # Flatten to (Batch, Channels)
        return gem_features.flatten(1)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class CatheterModel(nn.Module):
    """
    Catheter and Line Position Detection Model.
    Backbone: EfficientNetV2-Small
    Head: GeM Pooling -> Dropout -> Linear
    """

    def __init__(self, pretrained=True):
        super(CatheterModel, self).__init__()

        # --- Backbone ---
        # Create EfficientNetV2-S model using timm
        # num_classes=0 and global_pool='' ensures we get the spatial feature map (B, C, H, W)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Get the number of output features from the backbone
        in_features = self.backbone.num_features

        # --- Head ---
        # Pooling Layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM(p=Config.GEM_P, p_trainable=Config.GEM_LEARNABLE)
        else:
            # Fallback to standard Global Average Pooling if GeM is disabled
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Dropout Layer
        self.drop = nn.Dropout(p=Config.DROP_RATE)

        # Final Classification Layer
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, NUM_CLASSES)
        """
        # Extract features from backbone
        # Shape: (B, C, H_feat, W_feat)
        features = self.backbone(x)

        # Apply pooling (GeM or Avg)
        # Shape: (B, C) if flattened inside pooling, or (B, C, 1, 1) -> flatten
        pooled_features = self.pooling(features)

        # Ensure it is flattened (GeM implementation above flattens, but AdaptiveAvgPool might not)
        if len(pooled_features.shape) > 2:
            pooled_features = pooled_features.flatten(1)

        # Apply Dropout
        dropped_features = self.drop(pooled_features)

        # Final prediction logits
        logits = self.fc(dropped_features)

        return logits
