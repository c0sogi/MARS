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
        # Load EfficientNet-B2
        # num_classes=0 returns the pooled feature vector (1408 dim for B2)
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=True, num_classes=0
        )
        self.in_features = self.backbone.num_features

        # Bottleneck Projection: 1408 -> 64
        self.projection = nn.Linear(self.in_features, Config.IMG_EMBED_DIM)

        self._freeze_stages()

    def _freeze_stages(self):
        """
        Freezes the entire backbone, then unfreezes the top two convolutional stages
        and the head to allow domain adaptation.
        """
        # 1. Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze Head (Conv Head + BN)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # 3. Unfreeze top two blocks of the main stage
        # EfficientNet in timm stores blocks in a Sequential list `blocks`
        # B2 has 7 blocks (indices 0-6). We unfreeze 5 and 6.
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze last 2 blocks
            for i in range(num_blocks - 2, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

    def forward(self, x):
        # x: (Batch, 3, 260, 260)
        features = self.backbone(x)  # (Batch, 1408)
        embedding = self.projection(features)  # (Batch, 64)
        return embedding


class ClinicalStream(nn.Module):
    """
    Stream A: Supervised Clinical Anchor.
    Learns the baseline disease trajectory from clinical scalars.
    """

    def __init__(self):
        super(ClinicalStream, self).__init__()
        input_dim = Config.N_CLINICAL_FEATURES
        hidden_dim = 128
        output_dim = Config.IMG_EMBED_DIM  # 64

        # Over-Parameterized MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Auxiliary Head for Deep Supervision
        # Projects latent features to mu_aux, sigma_aux
        self.aux_head = nn.Linear(output_dim, 2)

    def forward(self, x):
        # x: (Batch, 5)
        features = self.mlp(x)  # (Batch, 64)
        aux_raw = self.aux_head(features)  # (Batch, 2)
        return features, aux_raw


class ResidualStream(nn.Module):
    """
    Stream B: Context-Injected Visual Residual.
    Learns patient-specific residuals conditioned on clinical context.
    """

    def __init__(self):
        super(ResidualStream, self).__init__()
        # Input: Projected Image (64) + Raw Clinical (5)
        input_dim = Config.IMG_EMBED_DIM + Config.N_CLINICAL_FEATURES
        hidden_dim = 128
        output_dim = Config.IMG_EMBED_DIM  # 64

        # MLP without Dropout
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, img_embed, clinical):
        # img_embed: (Batch, 64)
        # clinical: (Batch, 5)

        # Context Injection
        combined = torch.cat([img_embed, clinical], dim=1)  # (Batch, 69)
        residual = self.mlp(combined)  # (Batch, 64)
        return residual


class GPCRNet(nn.Module):
    """
    Global-Normed Projected-Context Residual Network.
    Synthesizes Dual-Stream topology with Residual Learning.
    """

    def __init__(self):
        super(GPCRNet, self).__init__()
        self.image_encoder = ImageEncoder()
        self.stream_a = ClinicalStream()
        self.stream_b = ResidualStream()

        # Shared Head
        # Projects the fused latent representation to mu, sigma
        self.head = nn.Linear(Config.IMG_EMBED_DIM, 2)

    def _process_output(self, raw_output):
        """
        Converts raw linear output to mu and sigma.
        Enforces positivity constraint on sigma using Softplus.
        """
        mu = raw_output[:, 0]
        raw_sigma = raw_output[:, 1]

        # Sigma constraint: softplus(raw) + epsilon
        sigma = F.softplus(raw_sigma) + 1e-6

        return mu, sigma

    def forward(self, image, clinical):
        """
        Args:
            image: (Batch, 3, H, W)
            clinical: (Batch, 5) [Baseline_FVC_Scaled, Relative_Time, Age_Scaled, Sex_Code, Smoking_Code]

        Returns:
            (mu, sigma): Main prediction
            (aux_mu, aux_sigma): Auxiliary prediction from Stream A
        """
        # 1. Image Branch
        img_embed = self.image_encoder(image)

        # 2. Stream A (Clinical Anchor)
        feat_a, aux_raw = self.stream_a(clinical)

        # 3. Stream B (Residual)
        feat_b = self.stream_b(img_embed, clinical)

        # 4. Latent Fusion (Summation)
        # Enforces residual learning: Final = Anchor + Residual
        feat_final = feat_a + feat_b

        # 5. Prediction Heads
        main_raw = self.head(feat_final)

        # Process outputs
        mu, sigma = self._process_output(main_raw)
        aux_mu, aux_sigma = self._process_output(aux_raw)

        return (mu, sigma), (aux_mu, aux_sigma)
