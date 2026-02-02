import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout (MSD).

    Instead of a single dropout layer, this module applies multiple dropout masks
    with different rates to the same input features. The results are passed
    through a shared fully connected layer and averaged. This technique accelerates
    convergence and improves generalization.
    """

    def __init__(self, in_features, out_features, dropout_rates):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature tensor of shape (Batch, Features).

        Returns:
            torch.Tensor: Averaged logits of shape (Batch, Out_Features).
        """
        logits_list = []
        for dropout in self.dropouts:
            # Apply specific dropout mask
            out = dropout(x)
            # Pass through shared linear layer
            out = self.fc(out)
            logits_list.append(out)

        # Stack results: (Batch, Num_Dropouts, Out_Features)
        logits = torch.stack(logits_list, dim=1)

        # Average across the dropout samples dimension
        return torch.mean(logits, dim=1)


class BirdClassifier(nn.Module):
    """
    Main classifier class for Bird Species Detection.

    Wraps a backbone (ResNet, EfficientNet, or DenseNet) with a
    Multi-Sample Dropout head.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): Name of the architecture to load via timm
                              (e.g., 'resnet18', 'efficientnet_b0', 'densenet121').
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super().__init__()
        self.model_name = model_name

        # Load backbone using timm
        # num_classes=0 removes the default classification head
        # global_pool='avg' ensures the output is a flattened feature vector
        # in_chans=3 matches the Pseudo-RGB input strategy
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
            global_pool="avg",
        )

        # Retrieve the number of output features from the backbone
        in_features = self.backbone.num_features

        # Initialize the custom Multi-Sample Dropout head
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=Config.NUM_SPECIES,
            dropout_rates=Config.DROPOUT_RATES,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Species).
        """
        # Extract features using the backbone
        # Output shape: (Batch, num_features)
        features = self.backbone(x)

        # Pass features through the MSD head
        logits = self.head(features)

        return logits
