import torch
import torch.nn as nn
import timm
from library.config import Config


class PetModel(nn.Module):
    """
    PetModel class implementing the backbone and custom head for the
    Tri-Model Stacking strategy.

    Supported Architectures (via Config):
    - Swin Transformer Large
    - ConvNeXt Large
    - BEiT Large

    The model replaces the default classifier with a 'Narrow MLP Head':
    Backbone -> Global Average Pooling -> Dropout -> Linear -> Sigmoid.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        """
        Args:
            model_name (str): Key from Config.models (e.g., 'swin') or specific timm model name.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(PetModel, self).__init__()

        # Resolve the timm model name from Config if a short key is provided
        if model_name in Config.models:
            self.model_name = Config.models[model_name]
        else:
            self.model_name = model_name

        # Create the backbone using timm
        # num_classes=0: Removes the default classification layer (fc).
        # global_pool='avg': Enforces Global Average Pooling.
        #   - For CNNs (ConvNeXt), this pools spatial maps to a vector.
        #   - For Transformers (Swin, BEiT), this averages the output tokens,
        #     satisfying the GAP requirement in the "Narrow MLP Head".
        self.backbone = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the feature dimension (e.g., 1536 for Large variants)
        self.n_features = self.backbone.num_features
        self.meta_dim = len(Config.meta_cols)

        # Define the Narrow MLP Head with Metadata Fusion (Cite solution_lesson_node_00007)
        # Structure: Dropout -> Linear(Image + Meta) -> Sigmoid
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.n_features + self.meta_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, meta):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).
            meta (torch.Tensor): Metadata tensor of shape (Batch_Size, Meta_Dim).

        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, 1) with values in [0, 1].
        """
        # Extract features using the backbone (includes GAP)
        features = self.backbone(x)

        # Concatenate image features with metadata
        features = torch.cat([features, meta], dim=1)

        # Pass features through the custom head
        output = self.head(features)

        return output
