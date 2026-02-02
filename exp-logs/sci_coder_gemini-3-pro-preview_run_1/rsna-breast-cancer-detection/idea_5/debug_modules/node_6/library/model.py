import torch
import torch.nn as nn
import timm
from library.config import MODEL_NAME, IN_CHANNELS, NUM_CLASSES


class SiameseEfficientNet(nn.Module):
    """
    Siamese Network with EfficientNet-B2 backbone.

    Architecture:
    1. Shared Backbone (EfficientNet-B2) processes Target and Contralateral images.
    2. Features are extracted: f_target, f_contra.
    3. Difference feature is computed: |f_target - f_contra|.
    4. Combined feature vector: Concat(f_target, |f_target - f_contra|).
    5. Classification Head: Linear layer -> Logit.
    """

    def __init__(self):
        super(SiameseEfficientNet, self).__init__()

        # Initialize shared backbone
        # num_classes=0 with default global_pool='avg' returns the pooled feature vector
        self.backbone = timm.create_model(
            MODEL_NAME, pretrained=True, in_chans=IN_CHANNELS, num_classes=0
        )

        # Get feature dimension from the backbone
        # For EfficientNet-B2, this is typically 1408
        self.num_features = self.backbone.num_features

        # Classification Head
        # Input: f_target (num_features) + diff (num_features) = 2 * num_features
        # Output: NUM_CLASSES (1)
        self.classifier = nn.Linear(self.num_features * 2, NUM_CLASSES)

    def forward_features(self, x):
        """
        Passes a single image tensor through the backbone.
        """
        return self.backbone(x)

    def forward(self, target_img, contra_img):
        """
        Forward pass for the Siamese Network.

        Args:
            target_img (Tensor): Target breast image tensor (B, C, H, W).
            contra_img (Tensor): Contralateral breast image tensor (B, C, H, W).

        Returns:
            Tensor: Logits (B, 1).
        """
        # 1. Extract features using shared backbone
        f_target = self.forward_features(target_img)
        f_contra = self.forward_features(contra_img)

        # 2. Compute absolute difference
        # Captures asymmetry between breasts
        diff = torch.abs(f_target - f_contra)

        # 3. Concatenate Target features with Difference features
        # We keep Target features to retain local context of the breast in question
        combined = torch.cat([f_target, diff], dim=1)

        # 4. Classification
        out = self.classifier(combined)

        return out
