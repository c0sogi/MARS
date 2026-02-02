import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class CalibratedSymmetricDualAxisNetwork(nn.Module):
    """
    A multi-modal neural network for lung function decline prediction.

    Architecture:
    1. Independent Visual Backbones: Two EfficientNet-B0 encoders for Axial and Coronal views.
    2. Up-Projected Tabular Embedding: Maps clinical features to visual dimensionality.
    3. Symmetric Attention Fusion: Transformer Encoder processes [Axial, Coronal, Tabular] sequence.
    4. Clinical Calibration Layer: Gates visual features using strong clinical priors.
    5. Prior-Anchored Head: Predicts trajectory parameters (alpha, sigma_base, sigma_growth).
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Independent Visual Backbones
        # ==========================================
        # We use EfficientNet-B0. With num_classes=0 and global_pool='avg',
        # timm returns the uncompressed feature vector (1280 dimensions).
        self.axial_backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        self.coronal_backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # ==========================================
        # 2. Up-Projected Tabular Embedding
        # ==========================================
        # Projects 7-dim clinical features UP to 1280-dim to match visual fidelity.
        self.tabular_embedding = nn.Sequential(
            nn.Linear(Config.TABULAR_INPUT_DIM, Config.TABULAR_HIDDEN_DIM),
            nn.LayerNorm(Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.TABULAR_HIDDEN_DIM, Config.FUSION_DIM),
            nn.LayerNorm(Config.FUSION_DIM),
        )

        # ==========================================
        # 3. Symmetric Attention Fusion
        # ==========================================
        # Processes the sequence [V_ax, V_cor, V_tab] symmetrically.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.FUSION_DIM,
            nhead=Config.NUM_ATTENTION_HEADS,
            dim_feedforward=Config.FUSION_DIM * 2,
            dropout=Config.DROPOUT_RATE,
            activation="gelu",
            batch_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # ==========================================
        # 4. Clinical Calibration Layer (Gating)
        # ==========================================
        # Generates a scaling vector from raw clinical features to filter visual noise.
        self.calibration_gate = nn.Sequential(
            nn.Linear(Config.TABULAR_INPUT_DIM, Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.TABULAR_HIDDEN_DIM, Config.FUSION_DIM),
            nn.Sigmoid(),  # Output range [0, 1] for gating
        )

        # ==========================================
        # 5. Prior-Anchored Head
        # ==========================================
        # Concatenates the calibrated high-dim vector with the raw tabular priors.
        # Input: 1280 (Calibrated Visual) + 7 (Raw Tabular) = 1287
        head_input_dim = Config.FUSION_DIM + Config.TABULAR_INPUT_DIM

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.TABULAR_HIDDEN_DIM, 3),
            # Output nodes: [alpha, sigma_base, sigma_growth]
        )

    def forward(self, images_axial, images_coronal, tabular):
        """
        Forward pass of the network.

        Args:
            images_axial (torch.Tensor): Batch of Axial Tri-Slab images (B, 3, 224, 224).
            images_coronal (torch.Tensor): Batch of Coronal Tri-Slab images (B, 3, 224, 224).
            tabular (torch.Tensor): Batch of normalized clinical features (B, 7).

        Returns:
            tuple: (alpha, sigma_base, sigma_growth)
                alpha (torch.Tensor): Slope of decline (B, 1).
                sigma_base (torch.Tensor): Base confidence (B, 1).
                sigma_growth (torch.Tensor): Confidence growth rate (B, 1).
        """
        # --- 1. Feature Extraction ---
        # Extract global descriptors (B, 1280)
        v_ax = self.axial_backbone(images_axial)
        v_cor = self.coronal_backbone(images_coronal)

        # Embed tabular features (B, 1280)
        v_tab = self.tabular_embedding(tabular)

        # --- 2. Symmetric Fusion ---
        # Stack into sequence: [Axial, Coronal, Tabular] -> Shape (B, 3, 1280)
        sequence = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # Apply Self-Attention
        transformed_sequence = self.fusion_transformer(sequence)

        # Holistic Pooling: Average across the sequence dimension -> (B, 1280)
        v_fused = transformed_sequence.mean(dim=1)

        # --- 3. Clinical Calibration ---
        # Calculate gating vector from raw tabular input
        gate = self.calibration_gate(tabular)  # (B, 1280)

        # Apply calibration (Late FiLM-like mechanism)
        v_calibrated = v_fused * gate

        # --- 4. Prior-Anchored Prediction ---
        # Skip connection: Concatenate raw tabular features for direct access by the head
        head_input = torch.cat([v_calibrated, tabular], dim=1)  # (B, 1287)

        raw_output = self.head(head_input)  # (B, 3)

        # Split outputs
        alpha = raw_output[:, 0:1]  # Linear slope (can be negative)
        sigma_base_raw = raw_output[:, 1:2]
        sigma_growth_raw = raw_output[:, 2:3]

        # Enforce positivity constraints for uncertainty
        sigma_base = F.softplus(sigma_base_raw)
        sigma_growth = F.softplus(sigma_growth_raw)

        return alpha, sigma_base, sigma_growth
