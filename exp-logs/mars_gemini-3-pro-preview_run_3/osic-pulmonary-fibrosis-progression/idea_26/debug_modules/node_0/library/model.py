import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GCRNet(nn.Module):
    """
    Gated Clinical-Residual Network (GCR-Net).

    A hybrid CNN-MLP architecture that uses a clinical anchor stream to establish
    a robust baseline trajectory and a gated visual stream to integrate
    fine-grained image features without destabilizing the prediction.
    """

    def __init__(self):
        super(GCRNet, self).__init__()

        # --- Image Backbone (Fine-Tuned Content-Adaptive 2.5D) ---
        # EfficientNet-B2
        # We load with num_classes=0 to get features, but we will use forward_features explicitly.
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, in_chans=3, features_only=False
        )

        # Determine backbone output dimension
        # For EfficientNet-B2, final conv channel count is typically 1408
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.backbone.forward_features(dummy)
            self.backbone_dim = features.shape[1]

        # Pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection to lower dimension to prevent overfitting
        self.img_project = nn.Linear(self.backbone_dim, Config.IMG_EMBED_DIM)

        # --- Freezing Strategy ---
        # Freeze all layers initially to preserve robust edge/texture detection
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top two convolutional stages (Blocks 5 and 6 for EfficientNet)
        # and the final conv head / bn to allow domain adaptation
        for name, child in self.backbone.named_children():
            if name == "blocks":
                # EfficientNet blocks are usually indexed 0-6
                for i, block in enumerate(child):
                    if i >= 5:
                        for param in block.parameters():
                            param.requires_grad = True
            elif name in ["conv_head", "bn2"]:
                for param in child.parameters():
                    param.requires_grad = True

        # --- Stream A: Clinical Anchor (The Controller) ---
        # Input: [Base_FVC, Base_Pct, Rel_Time, Age, Sex, Smoke] (6 features)
        self.clinical_input_dim = 6
        self.clinical_mlp = nn.Sequential(
            nn.Linear(self.clinical_input_dim, Config.CLINICAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.CLINICAL_HIDDEN_DIM, Config.LATENT_DIM),
        )

        # --- Stream B: Visual Interaction Stream ---
        # Input: Image Projection + Clinical Features
        self.visual_input_dim = Config.IMG_EMBED_DIM + self.clinical_input_dim
        self.visual_mlp = nn.Sequential(
            nn.Linear(self.visual_input_dim, Config.VISUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.VISUAL_HIDDEN_DIM, Config.LATENT_DIM),
        )

        # --- Gating Mechanism ---
        # Dynamic scaling based on clinical context
        # Gate = Sigmoid(Linear(H_clin))
        self.gate_layer = nn.Linear(Config.LATENT_DIM, Config.LATENT_DIM)

        # --- Shared Head ---
        # Projects fused latent to Mu and Sigma
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor (Batch, 3, H, W)
            tabular: Tensor (Batch, 6) -> [Base_FVC, Base_Pct, Rel_Time, Age, Sex, Smoke]
        Returns:
            Tensor (Batch, 2) -> [FVC_Mean, Confidence_Sigma]
        """
        # 1. Image Feature Extraction
        x = self.backbone.forward_features(image)  # (B, C, H', W')
        x = self.global_pool(x).flatten(1)  # (B, C)
        img_embed = self.img_project(x)  # (B, 64)

        # 2. Clinical Stream (Stream A)
        # Learns the "Expected Clinical Trajectory"
        h_clin = self.clinical_mlp(tabular)  # (B, 64)

        # 3. Visual Stream (Stream B)
        # Captures non-linear cross-modal interactions
        vis_input = torch.cat([img_embed, tabular], dim=1)
        h_vis = self.visual_mlp(vis_input)  # (B, 64)

        # 4. Gated Fusion
        # Generate gating scalar/vector from clinical stream
        gate = torch.sigmoid(self.gate_layer(h_clin))

        # Residual Connection: H_final = H_clin + (Gate * H_vis)
        # Allows suppression of visual stream if noisy or irrelevant
        h_final = h_clin + (gate * h_vis)

        # 5. Prediction Head
        out = self.head(h_final)

        mu = out[:, 0].unsqueeze(1)
        raw_sigma = out[:, 1].unsqueeze(1)

        # Constraint: Sigma must be positive.
        # We use softplus + epsilon for numerical stability.
        # Note: The metric clipping (70ml) is applied in post-processing, not here.
        sigma = F.softplus(raw_sigma) + 1e-6

        return torch.cat([mu, sigma], dim=1)
