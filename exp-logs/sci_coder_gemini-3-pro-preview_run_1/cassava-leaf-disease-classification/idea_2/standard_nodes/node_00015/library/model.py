import torch
import torch.nn as nn
import timm
from library.config import CFG


class CassavaModel(nn.Module):
    """
    Cassava Leaf Disease Classification Model.

    Architecture:
    - Backbone: EfficientNet-B4 (Noisy Student weights)
    - Head: Multi-Sample Dropout (5 parallel dropout paths)

    The Multi-Sample Dropout technique accelerates training convergence and
    improves generalization by computing multiple dropout masks per sample
    and averaging the resulting logits.
    """

    def __init__(self, model_name=CFG.model_name, pretrained=CFG.pretrained):
        """
        Args:
            model_name (str): Name of the timm model to load.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(CassavaModel, self).__init__()

        # Load the backbone
        # num_classes=0 removes the default classifier layer
        # global_pool='avg' ensures the output is a flattened feature vector (Batch, NumFeatures)
        # drop_path_rate controls Stochastic Depth within the backbone blocks
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=CFG.drop_path_rate,
        )

        # Determine input feature dimension from the backbone
        self.in_features = self.backbone.num_features

        # Multi-Sample Dropout Head
        # We use 5 parallel dropout layers. Each layer will generate a different
        # random mask during the forward pass.
        self.num_samples = 5
        self.dropouts = nn.ModuleList(
            [nn.Dropout(CFG.drop_rate) for _ in range(self.num_samples)]
        )

        # Final classification layer (Shared weights across all dropout samples)
        self.fc = nn.Linear(self.in_features, CFG.target_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Averaged logits of shape (B, NumClasses).
        """
        # Extract features from backbone
        features = self.backbone(x)  # Shape: (B, in_features)

        # Apply Multi-Sample Dropout
        # We pass the features through each dropout layer and then the shared FC layer.
        logits_list = []
        for dropout in self.dropouts:
            # Each dropout(features) call generates a unique mask
            logits_list.append(self.fc(dropout(features)))

        # Stack logits to shape: (B, NumSamples, NumClasses)
        stacked_logits = torch.stack(logits_list, dim=1)

        # Average the logits across the sample dimension
        # This effectively ensembles the predictions from different dropout masks
        mean_logits = torch.mean(stacked_logits, dim=1)

        return mean_logits
