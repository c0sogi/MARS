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


class LinearTrendStream(nn.Module):
    """
    Stream A: Linear Trend.
    Captures the dominant autoregressive linear trend (Baseline + Time).
    Cite solution_lesson_node_00052: Dual-Stream Residuals.
    Cite solution_lesson_node_00060: Over-Parameterization of Linear Baselines.
    """

    def __init__(self):
        super(LinearTrendStream, self).__init__()
        # Input: Baseline_FVC_Scaled, Relative_Time
        input_dim = 2
        output_dim = Config.IMG_EMBED_DIM  # 64

        # Linear Projection (Over-parameterized linear stream)
        self.linear = nn.Linear(input_dim, output_dim)

        # Auxiliary Head for Deep Supervision
        self.aux_head = nn.Linear(output_dim, 2)

    def forward(self, x):
        # x: (Batch, 2)
        features = self.linear(x)
        aux_raw = self.aux_head(features)
        return features, aux_raw


class DeepInteractionStream(nn.Module):
    """
    Stream B: Deep Interaction.
    Captures non-linear residuals from Image + All Clinical Context.
    Cite solution_lesson_node_00139: Context Visibility.
    Cite solution_lesson_node_00126: Preserving Weak Residual Signals (No Dropout).
    """

    def __init__(self):
        super(DeepInteractionStream, self).__init__()
        input_dim = Config.IMG_EMBED_DIM + Config.N_CLINICAL_FEATURES
        hidden_dim = 128
        output_dim = Config.IMG_EMBED_DIM

        # MLP without Dropout
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, img_embed, clinical):
        # Context Injection
        combined = torch.cat([img_embed, clinical], dim=1)
        return self.mlp(combined)


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network.
    Cite solution_lesson_node_00052.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()
        self.image_encoder = ImageEncoder()
        self.stream_a = LinearTrendStream()
        self.stream_b = DeepInteractionStream()

        # Shared Head
        self.head = nn.Linear(Config.IMG_EMBED_DIM, 2)

    def _process_output(self, raw_output):
        mu = raw_output[:, 0]
        raw_sigma = raw_output[:, 1]
        sigma = F.softplus(raw_sigma) + 1e-6
        return mu, sigma

    def forward(self, image, clinical):
        """
        Args:
            image: (Batch, 3, H, W)
            clinical: (Batch, 5) [Baseline_FVC_Scaled, Relative_Time, Age_Scaled, Sex_Code, Smoking_Code]
        """
        # 1. Image Branch
        img_embed = self.image_encoder(image)

        # 2. Stream A (Linear Trend: Baseline + Time)
        # Indices 0 and 1 correspond to Baseline_FVC_Scaled and Relative_Time
        linear_input = clinical[:, :2]
        feat_a, aux_raw = self.stream_a(linear_input)

        # 3. Stream B (Deep Interaction: Image + All Clinical)
        feat_b = self.stream_b(img_embed, clinical)

        # 4. Latent Fusion (Summation)
        feat_final = feat_a + feat_b

        # 5. Prediction Heads
        main_raw = self.head(feat_final)

        mu, sigma = self._process_output(main_raw)
        aux_mu, aux_sigma = self._process_output(aux_raw)

        return (mu, sigma), (aux_mu, aux_sigma)
