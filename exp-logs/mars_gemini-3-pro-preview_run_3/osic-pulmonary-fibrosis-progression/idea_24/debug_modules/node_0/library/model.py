import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ImageEncoder(nn.Module):
    """
    Fine-Tuned Content-Adaptive 2.5D Image Encoder.
    Backbone: EfficientNet-B2
    """

    def __init__(self):
        super(ImageEncoder, self).__init__()
        # Load EfficientNet B2
        # in_chans=3 matches our 3-slice input (Anchor + 2 boundaries)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            in_chans=Config.NUM_SLICES,
            num_classes=0,  # Remove classifier
            global_pool="",  # Remove default pooling, we handle it
        )

        # Feature dimension for EfficientNet-B2 is 1408
        self.n_features = self.backbone.num_features

        # ----------------------------------------------------------------------
        # Freezing Logic
        # ----------------------------------------------------------------------
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the top two convolutional stages and the head
        # In timm EfficientNet, the main blocks are in 'blocks' (nn.Sequential)
        # We unfreeze the last two blocks, the conv_head, and bn2

        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True

        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Unfreeze last 2 blocks of the sequential container
        # blocks is usually a list of Sequential blocks
        if hasattr(self.backbone, "blocks"):
            for param in self.backbone.blocks[-1].parameters():
                param.requires_grad = True
            for param in self.backbone.blocks[-2].parameters():
                param.requires_grad = True

        # ----------------------------------------------------------------------
        # Projection
        # ----------------------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(self.n_features, Config.IMG_EMBED_DIM)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.backbone.forward_features(x)  # (B, 1408, H', W')
        x = self.global_pool(x).flatten(1)  # (B, 1408)
        x = self.projection(x)  # (B, 64)
        return x


class ClinicalEncoder(nn.Module):
    """
    Stream A: Over-Parameterized Clinical Anchor.
    Learns the expected clinical trajectory (Baseline + Decay) from tabular data.
    """

    def __init__(self):
        super(ClinicalEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.CLINICAL_INPUT_DIM, Config.CLINICAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.CLINICAL_HIDDEN_DIM, Config.CLINICAL_LATENT_DIM),
        )

    def forward(self, x):
        # x: (B, 6)
        return self.net(x)  # (B, 64)


class InteractionEncoder(nn.Module):
    """
    Stream B: Cascaded Visual Interaction.
    Forces visual stream to learn as a conditional modifier of the clinical state.
    """

    def __init__(self):
        super(InteractionEncoder, self).__init__()
        # Input: Concatenation of [Image Projection (64), Clinical Latent (64)]
        input_dim = Config.IMG_EMBED_DIM + Config.CLINICAL_LATENT_DIM

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(
                128, Config.CLINICAL_LATENT_DIM
            ),  # Output dimension matches residual
        )

    def forward(self, img_embed, clin_embed):
        # Concatenate features
        x = torch.cat([img_embed, clin_embed], dim=1)
        return self.net(x)


class MACLINet(nn.Module):
    """
    Metric-Aligned Cascaded Latent-Interaction Network.
    """

    def __init__(self):
        super(MACLINet, self).__init__()
        self.image_encoder = ImageEncoder()
        self.clinical_encoder = ClinicalEncoder()
        self.interaction_encoder = InteractionEncoder()

        # Shared Head for Prediction
        # Projects the fused latent vector to mu (FVC) and raw sigma (Confidence)
        self.head = nn.Linear(Config.CLINICAL_LATENT_DIM, 2)

    def forward(self, image, clinical):
        """
        Args:
            image (torch.Tensor): (B, 3, 260, 260)
            clinical (torch.Tensor): (B, 6)
        Returns:
            mu (torch.Tensor): Predicted FVC (normalized scale)
            sigma (torch.Tensor): Predicted Confidence (normalized scale)
        """
        # 1. Stream A: Clinical Anchor
        # Get the expected state based purely on clinical history
        h_clin = self.clinical_encoder(clinical)  # (B, 64)

        # 2. Image Feature Extraction
        h_img = self.image_encoder(image)  # (B, 64)

        # 3. Stream B: Cascaded Interaction
        # Calculate visual correction conditioned on clinical state
        visual_correction = self.interaction_encoder(h_img, h_clin)  # (B, 64)

        # 4. Residual Fusion
        # H_final = H_clin + StreamB_output
        h_final = h_clin + visual_correction

        # 5. Prediction Head
        out = self.head(h_final)

        mu = out[:, 0]
        raw_sigma = out[:, 1]

        # Uncertainty Constraint
        # Use softplus to ensure positivity, add epsilon for stability
        sigma = F.softplus(raw_sigma) + 1e-6

        return mu, sigma
