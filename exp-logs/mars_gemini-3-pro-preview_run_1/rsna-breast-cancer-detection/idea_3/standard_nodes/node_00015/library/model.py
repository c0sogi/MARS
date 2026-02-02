import torch
import torch.nn as nn
import timm
from library import config


class StochasticModalityDropout(nn.Module):
    """
    Custom dropout layer that randomly zeroes out the metadata channels (Age and Implant)
    during training while keeping the image channel intact.

    This forces the model to learn visual features independently of the strong demographic priors,
    preventing shortcut learning where the model becomes a simple age regressor.
    """

    def __init__(self, p=0.5):
        """
        Args:
            p (float): Probability of dropping the metadata channels.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
                              Expects C=3: [Image, Age, Implant].
        Returns:
            torch.Tensor: The processed tensor with metadata channels potentially zeroed out.
        """
        # Only apply during training and if probability > 0
        if not self.training or self.p == 0.0:
            return x

        B = x.shape[0]

        # Probability of keeping the metadata is 1 - p
        keep_prob = 1.0 - self.p

        # Generate a Bernoulli mask for the metadata block
        # Shape (B, 1, 1, 1) allows broadcasting across the 2 metadata channels and spatial dims
        # 1.0 = Keep, 0.0 = Drop
        mask_metadata = torch.bernoulli(
            torch.full((B, 1, 1, 1), keep_prob, device=x.device)
        )

        # Clone input to avoid in-place modification errors
        out = x.clone()

        # Apply mask to channels 1 (Age) and 2 (Implant)
        # Channel 0 (Image) is left untouched
        out[:, 1:, :, :] = out[:, 1:, :, :] * mask_metadata

        return out


class MetadataEfficientNet(nn.Module):
    """
    EfficientNet-B2 based model that uses Early Fusion (Spatial Channel Expansion)
    to integrate mammograms with patient metadata.
    """

    def __init__(self):
        super().__init__()

        # Create Backbone
        # We use in_chans=3 to accommodate the composite input [Image, Age, Implant]
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=config.PRETRAINED,
            in_chans=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
        )

        # Initialize Modality Dropout
        self.modality_dropout = StochasticModalityDropout(
            p=config.MODALITY_DROPOUT_PROB
        )

    def forward(self, img, age, implant):
        """
        Forward pass constructing the 3-channel input and passing through backbone.

        Args:
            img (torch.Tensor): Mammogram image tensor, shape (B, 1, H, W).
            age (torch.Tensor): Normalized age tensor, shape (B,) or (B, 1).
            implant (torch.Tensor): Implant binary tensor, shape (B,) or (B, 1).

        Returns:
            torch.Tensor: Logits, shape (B, 1).
        """
        B, _, H, W = img.shape

        # 1. Reshape metadata for spatial broadcasting
        # Ensure shape is (B, 1, 1, 1)
        if age.dim() == 1:
            age = age.view(B, 1, 1, 1)
        else:
            age = age.view(B, 1, 1, 1)

        if implant.dim() == 1:
            implant = implant.view(B, 1, 1, 1)
        else:
            implant = implant.view(B, 1, 1, 1)

        # 2. Spatially Broadcast Metadata
        # Expand scalar values to match image spatial dimensions (H, W)
        age_map = age.expand(-1, -1, H, W)
        implant_map = implant.expand(-1, -1, H, W)

        # 3. Channel Concatenation (Early Fusion)
        # Resulting shape: (B, 3, H, W)
        x = torch.cat([img, age_map, implant_map], dim=1)

        # 4. Stochastic Modality Dropout
        # Randomly zeroes out Age and Implant channels during training
        x = self.modality_dropout(x)

        # 5. Backbone Forward Pass
        logits = self.backbone(x)

        return logits
