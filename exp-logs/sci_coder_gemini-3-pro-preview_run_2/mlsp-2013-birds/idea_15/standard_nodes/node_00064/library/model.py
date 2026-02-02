import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements the Multi-Sample Dropout head.

    Instead of a single dropout -> fc path, this module applies dropout multiple times
    (with different masks) to the input features, passes each result through a
    shared Fully Connected layer, and averages the outputs.

    This technique acts as an ensemble within a single model, reducing overfitting
    and accelerating convergence.
    """

    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        """
        Args:
            in_features (int): Dimension of input feature vector.
            out_features (int): Dimension of output (number of classes).
            num_samples (int): Number of dropout samples to average.
            dropout_rate (float): Probability of an element to be zeroed.
        """
        super(MultiSampleDropout, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features, out_features)
        self.num_samples = num_samples

    def forward(self, x):
        # x shape: (Batch_Size, In_Features)
        logits_list = []

        for _ in range(self.num_samples):
            # Apply dropout (mask is randomized each call)
            out = self.dropout(x)
            # Pass through shared linear layer
            out = self.fc(out)
            logits_list.append(out)

        # Stack results: (Num_Samples, Batch_Size, Out_Features)
        logits = torch.stack(logits_list)

        # Return the average of the logits: (Batch_Size, Out_Features)
        return torch.mean(logits, dim=0)


class BirdClassifier(nn.Module):
    """
    The main classification model for the Bird Species task.

    It utilizes a backbone from the `timm` library for feature extraction
    and a custom Multi-Sample Dropout head for classification.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): Name of the timm model (e.g., 'resnet18', 'densenet121').
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(BirdClassifier, self).__init__()
        self.model_name = model_name

        # Initialize the backbone using timm.
        # num_classes=0 removes the default classification head.
        # global_pool='avg' ensures the output is a pooled feature vector.
        # in_chans=3 is specified as we convert spectrograms to pseudo-RGB.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=3,
        )

        # Retrieve the feature dimension size from the backbone
        in_features = self.backbone.num_features

        # Initialize the classification head
        if Config.USE_MULTI_SAMPLE_DROPOUT:
            self.head = MultiSampleDropout(
                in_features=in_features,
                out_features=Config.NUM_CLASSES,
                num_samples=Config.DROPOUT_SAMPLES,
                dropout_rate=Config.DROPOUT_RATE,
            )
        else:
            # Standard head fallback
            self.head = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE),
                nn.Linear(in_features, Config.NUM_CLASSES),
            )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features from the backbone
        features = self.backbone(x)

        # Pass features through the custom head
        logits = self.head(features)

        return logits
