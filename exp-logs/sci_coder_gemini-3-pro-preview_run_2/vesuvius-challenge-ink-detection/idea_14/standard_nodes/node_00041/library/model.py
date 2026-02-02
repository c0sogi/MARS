import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel


class MLP(nn.Module):
    """
    Linear Embedding layer for SegFormer Decoder.
    Projects features from (B, C, H, W) to (B, Embedding_Dim, H, W)
    via a Linear layer applied to the flattened channel dimension.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x shape: (B, C, H, W)
        n, c, h, w = x.shape
        # Flatten spatial dims and transpose for Linear layer: (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        # Reshape back to spatial: (B, C_out, H, W) (handled in parent usually, but here we return flat)
        return x


class SiameseSegFormer(nn.Module):
    """
    Siamese Multi-View SegFormer.

    Architecture:
    1. Shared Encoder (MiT-B2): Processes 3 input views (High, Center, Low) independently.
    2. Feature Fusion: Element-wise Max Pooling across the 3 views at each feature scale.
    3. Decoder: MLP-based decoder (standard SegFormer design) to aggregate scales and predict mask.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super(SiameseSegFormer, self).__init__()

        # --- Backbone ---
        # Using MiT-B2 (approx 25M params)
        model_name = "nvidia/mit-b2"
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(model_name)
        else:
            from transformers import SegformerConfig

            config = SegformerConfig.from_pretrained(model_name)
            self.encoder = SegformerModel(config)

        # Ensure encoder returns hidden states for all stages
        self.encoder.config.output_hidden_states = True

        # MiT-B2 Feature Channels: [64, 128, 320, 512]
        # These correspond to the outputs of the 4 Transformer blocks
        c1, c2, c3, c4 = self.encoder.config.hidden_sizes

        # --- Decoder ---
        # Standard SegFormer MLP Decoder parameters
        self.embedding_dim = 256

        # MLP layers to unify channel dimensions of different scales
        self.linear_c1 = MLP(c1, self.embedding_dim)
        self.linear_c2 = MLP(c2, self.embedding_dim)
        self.linear_c3 = MLP(c3, self.embedding_dim)
        self.linear_c4 = MLP(c4, self.embedding_dim)

        # Fusion of concatenated scales
        # Input channels = 4 * embedding_dim (concatenation of 4 scales)
        self.linear_fuse = nn.Conv2d(
            self.embedding_dim * 4, self.embedding_dim, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(self.embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

        # Final Classifier
        self.classifier = nn.Conv2d(self.embedding_dim, num_classes, kernel_size=1)

    def forward(self, view_1, view_2, view_3):
        """
        Forward pass for Siamese Network.

        Args:
            view_1 (torch.Tensor): Input tensor for View 1 (High). Shape (B, 3, H, W).
            view_2 (torch.Tensor): Input tensor for View 2 (Center). Shape (B, 3, H, W).
            view_3 (torch.Tensor): Input tensor for View 3 (Low). Shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits mask. Shape (B, num_classes, H, W).
        """

        # --- 1. Shared Encoder Pass ---
        # We process each view through the same encoder instance.
        # hidden_states is a tuple. For MiT, usually:
        # [0]: Output of Patch Embeddings (often ignored or treated as stage 1 input)
        # [-4], [-3], [-2], [-1]: Outputs of the 4 Transformer stages.

        outputs_1 = self.encoder(view_1).hidden_states
        outputs_2 = self.encoder(view_2).hidden_states
        outputs_3 = self.encoder(view_3).hidden_states

        # Extract the 4 multi-scale features
        # Shapes for 512x512 input:
        # 0: (B, 64, 128, 128)
        # 1: (B, 128, 64, 64)
        # 2: (B, 320, 32, 32)
        # 3: (B, 512, 16, 16)
        features_1 = outputs_1[-4:]
        features_2 = outputs_2[-4:]
        features_3 = outputs_3[-4:]

        # --- 2. Feature-Space MIP Fusion ---
        # Element-wise Max Pooling across the 3 views for each scale
        fused_features = []
        for i in range(4):
            # Max across the view dimension
            f = torch.max(features_1[i], features_2[i])
            f = torch.max(f, features_3[i])
            fused_features.append(f)

        c1, c2, c3, c4 = fused_features

        # --- 3. MLP Decoder ---

        # Get batch size and spatial dimensions of the largest feature map (c1)
        n, _, h, w = c1.shape

        # Process Stage 4 (Smallest, Deepest)
        _c4 = (
            self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        )
        _c4 = F.interpolate(_c4, size=(h, w), mode="bilinear", align_corners=False)

        # Process Stage 3
        _c3 = (
            self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        )
        _c3 = F.interpolate(_c3, size=(h, w), mode="bilinear", align_corners=False)

        # Process Stage 2
        _c2 = (
            self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        )
        _c2 = F.interpolate(_c2, size=(h, w), mode="bilinear", align_corners=False)

        # Process Stage 1 (Largest)
        _c1 = (
            self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        )

        # Concatenate all upsampled features
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)

        # Fuse and refine
        x = self.linear_fuse(_c)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Predict logits
        logits = self.classifier(x)

        # Final Upsample to original input resolution (4x upsample)
        # c1 is H/4, W/4 relative to input
        logits = F.interpolate(
            logits, scale_factor=4, mode="bilinear", align_corners=False
        )

        return logits
