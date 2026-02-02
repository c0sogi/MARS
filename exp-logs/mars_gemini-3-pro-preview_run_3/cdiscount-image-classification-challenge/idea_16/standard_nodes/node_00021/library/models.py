import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class DualBackboneExtractor(nn.Module):
    """
    Wraps frozen ResNet50 and EfficientNet-B0 models to extract features.
    Outputs concatenated feature vectors.
    """

    def __init__(self):
        super(DualBackboneExtractor, self).__init__()

        # Load Pretrained ResNet50
        # Use new weights API if available, fallback to legacy pretrained=True
        try:
            weights_resnet = models.ResNet50_Weights.DEFAULT
            self.resnet = models.resnet50(weights=weights_resnet)
        except AttributeError:
            self.resnet = models.resnet50(pretrained=True)

        # Remove the classification head (fc)
        # ResNet structure: ... -> avgpool -> fc
        # We keep everything up to avgpool (inclusive)
        self.resnet_modules = nn.Sequential(*list(self.resnet.children())[:-1])

        # Load Pretrained EfficientNet-B0
        try:
            weights_effnet = models.EfficientNet_B0_Weights.DEFAULT
            self.effnet = models.efficientnet_b0(weights=weights_effnet)
        except AttributeError:
            self.effnet = models.efficientnet_b0(pretrained=True)

        # EfficientNet structure: features -> avgpool -> classifier
        # We keep features and avgpool
        self.effnet_modules = nn.Sequential(self.effnet.features, self.effnet.avgpool)

        # Freeze parameters to ensure this is purely a feature extractor
        for param in self.resnet_modules.parameters():
            param.requires_grad = False
        for param in self.effnet_modules.parameters():
            param.requires_grad = False

        # Set to eval mode (important for BatchNorm behavior)
        self.eval()

    def forward(self, x, sizes=None):
        """
        Args:
            x: Tensor of shape (N_images, 3, H, W).
            sizes: Optional Tensor/List of shape (Batch_Size,) indicating num images per product.
                   If provided, performs mean pooling per product.

        Returns:
            features: Tensor.
                      If sizes is None: (N_images, DIM_RESNET + DIM_EFFNET)
                      If sizes is not None: (Batch_Size, DIM_RESNET + DIM_EFFNET)
        """
        with torch.no_grad():
            # ResNet Features
            r_feat = self.resnet_modules(x)
            r_feat = torch.flatten(r_feat, 1)  # (N, 2048)

            # EfficientNet Features
            e_feat = self.effnet_modules(x)
            e_feat = torch.flatten(e_feat, 1)  # (N, 1280)

            # Concatenate
            features = torch.cat([r_feat, e_feat], dim=1)  # (N, 3328)

            if sizes is not None:
                # Perform Mean Pooling per product
                if isinstance(sizes, torch.Tensor):
                    split_sizes = sizes.cpu().tolist()
                else:
                    split_sizes = sizes

                # Split the batch of images into chunks per product
                chunks = torch.split(features, split_sizes, dim=0)

                # Compute mean for each chunk
                pooled_features = [chunk.mean(dim=0) for chunk in chunks]
                features = torch.stack(pooled_features, dim=0)

        return features


class DeepFeatureCascade(nn.Module):
    """
    Hierarchical Deep Feature Cascading Network (DFC).

    Architecture:
    Input (Product Embed) -> [Block 1] -> L1 Logits + Hidden 1
                                |
    Input + Hidden 1      -> [Block 2] -> L2 Logits + Hidden 2
                                |
    Input + Hidden 2      -> [Block 3] -> L3 Logits
    """

    def __init__(self):
        super(DeepFeatureCascade, self).__init__()

        self.input_dim = Config.DIM_INPUT
        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout_rate = Config.DROPOUT_RATE

        # --- Level 1 Block ---
        # Input: Raw Features
        self.block1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.head_l1 = nn.Linear(self.hidden_dim, Config.NUM_CLASSES_L1)

        # --- Level 2 Block ---
        # Input: Raw Features + Hidden State from L1
        # This allows L2 to see both the raw visual signal and the high-level semantic concept from L1
        self.block2 = nn.Sequential(
            nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.head_l2 = nn.Linear(self.hidden_dim, Config.NUM_CLASSES_L2)

        # --- Level 3 Block ---
        # Input: Raw Features + Hidden State from L2
        self.block3 = nn.Sequential(
            nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.head_l3 = nn.Linear(self.hidden_dim, Config.NUM_CLASSES_L3)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Kaiming Normal for ReLU layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input features (Batch, DIM_INPUT)

        Returns:
            logits_l1, logits_l2, logits_l3
        """
        # Level 1
        h1 = self.block1(x)
        out_l1 = self.head_l1(h1)

        # Level 2
        # Concatenate original input with previous hidden state
        x2 = torch.cat([x, h1], dim=1)
        h2 = self.block2(x2)
        out_l2 = self.head_l2(h2)

        # Level 3
        # Concatenate original input with previous hidden state
        x3 = torch.cat([x, h2], dim=1)
        h3 = self.block3(x3)
        out_l3 = self.head_l3(h3)

        return out_l1, out_l2, out_l3
