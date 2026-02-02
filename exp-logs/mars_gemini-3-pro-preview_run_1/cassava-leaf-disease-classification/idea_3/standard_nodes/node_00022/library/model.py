import torch
import torch.nn as nn
import timm
from library.config import CFG


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout module.
    Applies multiple dropout masks to the input features and averages the predictions
    from the shared fully connected layer to improve generalization.
    """

    def __init__(self, in_features, out_features, p=0.5, num_samples=5):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: [Batch, in_features]
        # Iterate over dropout masks, compute logits, and accumulate
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))

        # Average the logits
        return out / len(self.dropouts)


class CassavaConvNeXt(nn.Module):
    """
    Cassava Leaf Disease Classification Model using ConvNeXt backbone.
    Integrates a Multi-Sample Dropout head for robust classification.
    """

    def __init__(self, model_name=CFG.model_name, pretrained=True):
        super().__init__()

        # Load the backbone model
        # num_classes=0 removes the original classification head
        # global_pool='avg' ensures the output is a pooled feature vector (Batch, Channels)
        # Cite solution_lesson_node_00019: Disable Stochastic Depth to prevent over-regularization
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=CFG.drop_path_rate,
        )

        # Determine input features for the head
        # timm models typically expose num_features
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            # Fallback mechanism if num_features is not directly available
            # (though standard for timm ConvNeXt)
            raise AttributeError(
                f"Model {model_name} does not have 'num_features' attribute."
            )

        # Initialize the Multi-Sample Dropout Head
        self.head = MultiSampleDropout(
            in_features=self.in_features,
            out_features=CFG.num_classes,
            p=CFG.dropout_rate,
            num_samples=CFG.num_dropout_samples,
        )

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input images of shape (Batch, Channels, Height, Width)
        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        # Extract features from backbone
        # Shape: (Batch, in_features)
        features = self.backbone(x)

        # Pass through classification head with multi-sample dropout
        logits = self.head(features)

        return logits
