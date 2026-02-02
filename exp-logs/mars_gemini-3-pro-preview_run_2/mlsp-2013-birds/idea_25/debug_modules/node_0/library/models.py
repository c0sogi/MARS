import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout head.

    Input features are passed through multiple dropout layers with different masks,
    then through a single shared linear layer. The results are averaged.
    This technique acts as a form of ensembling within the network, promoting
    faster convergence and better generalization.
    """

    def __init__(self, in_features, out_features, num_samples=5, p=0.5):
        super().__init__()
        # Create multiple dropout instances.
        # Since they are separate objects, they generate independent masks.
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])

        # Shared fully connected layer
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (Batch_Size, In_Features)

        # Apply each dropout mask and pass through the shared linear layer
        # Output of each is (Batch_Size, Out_Features)
        logits_list = [self.fc(dropout(x)) for dropout in self.dropouts]

        # Stack results to shape (Batch_Size, Num_Samples, Out_Features)
        logits = torch.stack(logits_list, dim=1)

        # Average over the samples dimension
        # Final shape: (Batch_Size, Out_Features)
        return torch.mean(logits, dim=1)


class BirdClassifier(nn.Module):
    """
    Main classifier class supporting heterogeneous backbones.

    Wraps a timm backbone with the MultiSampleDropout head.
    """

    def __init__(self, backbone_name, pretrained=True):
        """
        Args:
            backbone_name (str): Name of the timm model (e.g., 'resnet18').
            pretrained (bool): Whether to load ImageNet weights.
        """
        super().__init__()

        # Create the backbone
        # num_classes=0 removes the default FC layer
        # global_pool='avg' ensures the output is a 1D feature vector per sample
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Dynamically determine the number of input features for the head
        # timm models usually expose this via .num_features
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback inspection if specific attribute is missing
            # (Create a dummy input to check output shape)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)
                features = self.backbone(dummy)
                in_features = features.shape[1]

        # Initialize the Multi-Sample Dropout Head
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=Config.NUM_CLASSES,
            num_samples=5,  # As per strategy
            p=0.5,
        )

    def forward(self, x):
        # Forward pass through backbone
        # x shape: (Batch, 3, H, W) -> features shape: (Batch, Num_Features)
        features = self.backbone(x)

        # Forward pass through head
        # features -> logits shape: (Batch, Num_Classes)
        logits = self.head(features)

        return logits
