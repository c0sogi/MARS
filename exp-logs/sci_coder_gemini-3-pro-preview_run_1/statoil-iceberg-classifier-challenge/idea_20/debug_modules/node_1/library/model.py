import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class SymmetrizedResNet18(nn.Module):
    """
    A ResNet-18 based architecture that implements a Symmetrized Forward Pass.
    It enforces geometric invariance by averaging predictions over the Klein Four-Group
    (Original, FlipLR, FlipUD, Rotate180) directly within the forward method.
    """

    def __init__(self):
        super(SymmetrizedResNet18, self).__init__()

        # Load pretrained ResNet18
        # Using V1 weights as standard for ImageNet pretraining
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        original_model = models.resnet18(weights=weights)

        # Extract backbone: keep everything up to the Average Pooling layer
        # ResNet structure: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> fc
        # list(children)[:-1] removes the final 'fc' layer
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])

        # Feature dimension from ResNet18 GAP is 512
        self.feature_dim = 512

        # Angle normalization statistics (derived from dataset analysis)
        # Mean: 39.2829, Std: 3.8362
        self.angle_mean = 39.2829
        self.angle_std = 3.8362

        # Minimalist Head for Late Fusion
        # Input: 512 (Image Features) + 1 (Normalized Angle)
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim + 1),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.feature_dim + 1, 1),
        )

    def forward(self, x, angle):
        """
        Forward pass with internal symmetry expansion and averaging.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, 224, 224)
            angle (torch.Tensor): Incidence angles of shape (B,) or (B, 1)

        Returns:
            torch.Tensor: Averaged logits of shape (B, 1)
        """
        batch_size = x.size(0)

        # ---------------------------------------------------------------------
        # 1. Symmetrized Input Generation (Klein Four-Group)
        # ---------------------------------------------------------------------
        # View 1: Original
        x0 = x
        # View 2: Flip Left-Right (Horizontal) - Flip on Width (dim 3)
        x1 = torch.flip(x, dims=[3])
        # View 3: Flip Up-Down (Vertical) - Flip on Height (dim 2)
        x2 = torch.flip(x, dims=[2])
        # View 4: Rotate 180 (Equivalent to Flip UD + Flip LR)
        x3 = torch.flip(x, dims=[2, 3])

        # Stack all views to process in parallel: (4*B, 3, H, W)
        x_sym = torch.cat([x0, x1, x2, x3], dim=0)

        # ---------------------------------------------------------------------
        # 2. Backbone Feature Extraction
        # ---------------------------------------------------------------------
        # Pass through ResNet backbone
        # Output shape: (4*B, 512, 1, 1)
        features = self.backbone(x_sym)

        # Flatten to (4*B, 512)
        features = features.view(features.size(0), -1)

        # ---------------------------------------------------------------------
        # 3. Angle Processing & Late Fusion
        # ---------------------------------------------------------------------
        # Ensure angle is (B, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Normalize angle using global stats
        angle_norm = (angle - self.angle_mean) / self.angle_std

        # Repeat angle for each symmetry view to match the features batch size
        # We constructed x_sym as [batch, batch_flip_lr, batch_flip_ud, batch_rot180]
        # So we repeat the angle batch 4 times
        angle_sym = angle_norm.repeat(4, 1)  # (4*B, 1)

        # Concatenate image features and angle
        fused_features = torch.cat([features, angle_sym], dim=1)  # (4*B, 513)

        # ---------------------------------------------------------------------
        # 4. Classification Head
        # ---------------------------------------------------------------------
        # Pass through BN -> Dropout -> Linear
        logits_sym = self.head(fused_features)  # (4*B, 1)

        # ---------------------------------------------------------------------
        # 5. Logit Averaging (Ensemble Aggregation)
        # ---------------------------------------------------------------------
        # Reshape to separate views: (4, B, 1)
        logits_reshaped = logits_sym.view(4, batch_size, 1)

        # Average across the 4 views
        final_logits = torch.mean(logits_reshaped, dim=0)  # (B, 1)

        return final_logits
