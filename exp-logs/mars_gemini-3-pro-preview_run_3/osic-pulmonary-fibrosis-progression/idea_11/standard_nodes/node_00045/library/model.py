import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ImageEncoder(nn.Module):
    """
    EfficientNet-B2 backbone with specific layer freezing logic.
    Extracts deep semantic features from the 3-channel input image (Apical, Middle, Basal slices).
    """

    def __init__(self, model_name=Config.BACKBONE, pretrained=Config.PRETRAINED):
        super().__init__()
        # Load the backbone without the classifier and global pooling
        # We use 'tf_efficientnet_b2_ns' as specified in Config
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=Config.IN_CHANS,
        )

        # Determine the output feature dimension dynamically by running a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, Config.IN_CHANS, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.backbone(dummy)
            self.num_features = features.shape[1]

        # --- Freezing Logic ---
        # 1. Freeze all parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze the top layers for Domain Adaptation
        # Unfreeze Final Conv Head and BN if they exist
        for name, param in self.backbone.named_parameters():
            if "conv_head" in name or "bn2" in name:
                param.requires_grad = True

        # 3. Unfreeze the last two blocks of the 'blocks' container
        # timm EfficientNet implementation uses a 'blocks' Sequential container
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze the last 2 blocks
            for i in range(num_blocks - 2, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

    def forward(self, x):
        # x shape: (Batch, 3, Height, Width)
        x = self.backbone(x)
        # x shape: (Batch, Channels, H_feat, W_feat)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        # x shape: (Batch, Channels)
        return x


class TCDSNet(nn.Module):
    """
    Time-Conditioned Deep-Semantic Network (TCDS-Net).
    A hybrid CNN-MLP architecture that predicts FVC and Confidence based on
    CT scans and time-conditioned tabular metadata.
    """

    def __init__(self):
        super().__init__()

        # --- Image Branch ---
        self.image_encoder = ImageEncoder()

        # Projection layer to balance modalities (Backbone Dim -> 128)
        self.img_projector = nn.Linear(
            self.image_encoder.num_features, Config.N_FEATURES
        )

        # --- Tabular Branch ---
        # Input Features: [Base_FVC_Scaled, Age_Scaled, Sex_Code, Smoke_Code, Relative_Time]
        self.tabular_dim = 5

        # --- Fusion Head ---
        # Concatenation of Projected Image Features (128) and Raw Tabular Features (5)
        fusion_input_dim = Config.N_FEATURES + self.tabular_dim

        # Mixing MLP
        # Structure: Linear(512) -> ReLU -> Linear(256) -> ReLU -> Linear(2)
        # We explicitly exclude Dropout to preserve the precision of the linear signal
        self.mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.HIDDEN_DIM),  # 512
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # Outputs: mu (FVC), raw_sigma (Confidence)
        )

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor of shape (Batch, 3, Height, Width)
            tabular: Tensor of shape (Batch, 5) containing tabular features

        Returns:
            mu: Predicted FVC (Standardized)
            sigma: Predicted Confidence (Standardized, > 0)
        """
        # 1. Image Feature Extraction & Projection
        img_feats = self.image_encoder(image)  # (Batch, Backbone_Dim)
        img_emb = self.img_projector(img_feats)  # (Batch, 128)

        # 2. Fusion
        # Concatenate image embeddings with tabular features.
        # Tabular features include Relative Time, allowing the MLP to model
        # time-dependent non-linearities (e.g., fast decline vs slow decline).
        combined = torch.cat([img_emb, tabular], dim=1)  # (Batch, 133)

        # 3. Prediction Head
        out = self.mlp(combined)  # (Batch, 2)

        mu = out[:, 0]
        sigma_raw = out[:, 1]

        # 4. Uncertainty Activation
        # Ensure sigma is positive using softplus + epsilon
        sigma = F.softplus(sigma_raw) + 1e-6

        return mu, sigma
