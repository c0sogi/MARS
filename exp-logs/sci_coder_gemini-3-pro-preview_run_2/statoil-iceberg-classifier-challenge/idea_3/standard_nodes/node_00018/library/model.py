import torch
import torch.nn as nn
import timm
from library.config import Config


class RHTN(nn.Module):
    """
    Residual Hybrid Transfer Network (RHTN).

    This architecture fuses a visual branch based on a pre-trained ResNet-18 backbone
    with a scalar metadata branch processing the incidence angle.
    """

    def __init__(self):
        super(RHTN, self).__init__()

        # ==========================================
        # 1. Visual Branch (ResNet-18 Backbone)
        # ==========================================
        # We use timm to create the backbone.
        # in_chans=3: Expects 3-channel input (Band 1, Band 2, Mean).
        # num_classes=0: Removes the final fully connected classification layer.
        # global_pool='avg': Applies Global Average Pooling to the features.
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
            global_pool="avg",
        )

        # ==========================================
        # 2. Metadata Branch (Incidence Angle)
        # ==========================================
        # Processes the scalar incidence angle.
        self.meta_mlp = nn.Sequential(
            nn.Linear(Config.METADATA_INPUT_DIM, Config.METADATA_HIDDEN_DIM),
            nn.BatchNorm1d(Config.METADATA_HIDDEN_DIM),
            nn.ReLU(),
        )

        # ==========================================
        # 3. Fusion Head
        # ==========================================
        # Concatenates visual features (512) and metadata features (64).
        fusion_input_dim = Config.BACKBONE_OUT_DIM + Config.METADATA_HIDDEN_DIM

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(Config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.FUSION_HIDDEN_DIM, Config.NUM_CLASSES),
        )

    def forward(self, x_img, x_meta):
        """
        Forward pass of the RHTN.

        Args:
            x_img (torch.Tensor): Image tensor of shape (Batch, 3, 75, 75).
            x_meta (torch.Tensor): Metadata tensor of shape (Batch, 1).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # 1. Process Image
        # Output shape: (Batch, 512)
        img_features = self.backbone(x_img)

        # 2. Process Metadata
        # Output shape: (Batch, 64)
        meta_features = self.meta_mlp(x_meta)

        # 3. Feature Fusion
        # Concatenate along the feature dimension (dim=1)
        combined_features = torch.cat((img_features, meta_features), dim=1)

        # 4. Classification
        # Output shape: (Batch, 1)
        # We return logits. Sigmoid should be applied during inference or
        # handled by BCEWithLogitsLoss during training.
        logits = self.fusion_head(combined_features)

        return logits
