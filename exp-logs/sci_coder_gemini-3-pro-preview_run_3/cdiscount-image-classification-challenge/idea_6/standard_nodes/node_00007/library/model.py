import torch
import torch.nn as nn
import timm
from library.config import BACKBONE_MODEL, EMBEDDING_DIM, NUM_CLASSES, SEED

# Set fixed seed for reproducibility
torch.manual_seed(SEED)


class ProductFeatureExtractor(nn.Module):
    """
    A wrapper around EfficientNet-B0 to extract and aggregate features from product images.

    This model is designed to be used in the feature extraction phase (offline).
    It takes a batch of images corresponding to a single product, passes them through
    the frozen backbone, and aggregates them via mean pooling to create a single
    embedding per product.
    """

    def __init__(self, model_name=BACKBONE_MODEL, pretrained=True):
        super(ProductFeatureExtractor, self).__init__()
        # Load the backbone model (EfficientNet-B0)
        # num_classes=0 ensures we get the feature vector after global pooling (1280 dim)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Freeze all parameters as per the strategy
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.output_dim = self.backbone.num_features

    def forward(self, x):
        """
        Forward pass to extract and aggregate features.

        Args:
            x (torch.Tensor): Tensor of shape (N_images, 3, H, W).
                              Represents all images for a specific product.

        Returns:
            torch.Tensor: Aggregated feature vector of shape (Embedding_Dim,).
        """
        # Handle edge case of empty input (though dataset should prevent this)
        if x.numel() == 0:
            return torch.zeros(self.output_dim, device=x.device)

        # Extract features: (N_images, 1280)
        # We explicitly use no_grad to save memory during extraction
        with torch.no_grad():
            features = self.backbone(x)

        # Aggregate via Mean Pooling across the images of the product: (1280,)
        aggregated_features = torch.mean(features, dim=0)

        return aggregated_features


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Classifier.

    Designed to take pre-computed, aggregated feature vectors (1280-dim) and output class probabilities.
    Includes Batch Normalization, ReLU, and Dropout for regularization.
    """

    def __init__(
        self,
        input_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        hidden_dims=[1024, 512],
        dropout_rate=0.5,
    ):
        super(MLPClassifier, self).__init__()

        layers = []
        current_dim = input_dim

        # Build hidden layers dynamically
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        # Final output layer (Logits)
        layers.append(nn.Linear(current_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch_Size, Input_Dim).

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, Num_Classes).
        """
        return self.network(x)
