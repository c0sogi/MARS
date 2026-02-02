import torch
import torch.nn as nn
import timm
from library.config import Config


class EfficientNetV2Classifier(nn.Module):
    """
    EfficientNetV2-based classifier for Breast Cancer Detection.
    Supports Two-Stage Calibration strategy via backbone freezing and head resetting.
    """

    def __init__(self, backbone_name=Config.BACKBONE_NAME, pretrained=True):
        """
        Args:
            backbone_name (str): Name of the timm model to load.
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(EfficientNetV2Classifier, self).__init__()

        # Create the model using timm
        # num_classes=1 for binary classification (output is raw logit)
        # drop_rate controls the dropout before the classifier
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROPOUT_RATE,
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input images (B, C, H, W).
        Returns:
            torch.Tensor: Raw logits (B, 1).
        """
        return self.model(x)

    def freeze_backbone(self):
        """
        Freezes the feature extractor (backbone) parameters.
        Only the classifier head remains trainable.
        Used for Stage 2 (Calibration).
        """
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier head
        # In timm's EfficientNet implementation, the final layer is named 'classifier'
        if hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            # Fallback check for other architectures (e.g., 'fc' or 'head')
            # though tf_efficientnetv2_s uses 'classifier'
            for name, module in self.model.named_modules():
                if name in ["head", "fc"]:
                    for param in module.parameters():
                        param.requires_grad = True

    def unfreeze_backbone(self):
        """
        Unfreezes all parameters.
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def reset_classifier(self):
        """
        Re-initializes the weights of the classifier head.
        Used at the start of Stage 2 to learn calibration from scratch
        on the natural distribution.
        """
        if hasattr(self.model, "classifier"):
            self._init_weights(self.model.classifier)
        else:
            # Fallback for other naming conventions
            for name, module in self.model.named_modules():
                if name in ["head", "fc"]:
                    self._init_weights(module)

    def _init_weights(self, module):
        """
        Applies Xavier initialization to Linear layers.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
