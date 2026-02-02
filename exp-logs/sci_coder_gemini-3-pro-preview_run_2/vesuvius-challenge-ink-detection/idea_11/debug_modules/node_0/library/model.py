import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel
from library.config import Config


class SegFormer(nn.Module):
    """
    SegFormer (MiT-B3) with a lightweight MLP Decoder for Binary Segmentation.

    Architecture:
    - Backbone: MiT-B3 (Mix Transformer) pretrained on ImageNet.
    - Decoder: All-MLP decoder that aggregates multi-scale features.
    - Input: 3-channel RGB (or pseudo-RGB) images.
    - Output: 1-channel binary logits.
    """

    def __init__(self):
        super(SegFormer, self).__init__()

        # 1. Load Encoder (MiT-B3)
        # We use output_hidden_states=True to retrieve features from all 4 stages.
        self.encoder = SegformerModel.from_pretrained(
            "nvidia/mit-b3", output_hidden_states=True, output_attentions=False
        )

        # MiT-B3 Feature Channels for the 4 stages
        # [Stage 1, Stage 2, Stage 3, Stage 4]
        self.enc_channels = [64, 128, 320, 512]

        # Decoder Hyperparameters
        # We use a lightweight embedding dimension as per the "Expert" strategy
        self.embed_dim = 256

        # 2. MLP Decoder Layers
        # Project features from each stage to the common embedding dimension.
        # In the paper, this is a Linear layer. For 2D spatial features,
        # a Conv2d with kernel_size=1 is equivalent and handles spatial dims.
        self.mlp_c1 = nn.Conv2d(self.enc_channels[0], self.embed_dim, 1)
        self.mlp_c2 = nn.Conv2d(self.enc_channels[1], self.embed_dim, 1)
        self.mlp_c3 = nn.Conv2d(self.enc_channels[2], self.embed_dim, 1)
        self.mlp_c4 = nn.Conv2d(self.enc_channels[3], self.embed_dim, 1)

        # 3. Fusion Layer
        # Concatenates the 4 upsampled feature maps (4 * embed_dim) and fuses them.
        self.fuse = nn.Sequential(
            nn.Conv2d(self.embed_dim * 4, self.embed_dim, 1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
        )

        # 4. Classification Head
        # Projects fused features to the number of classes (1 for binary ink detection).
        self.head = nn.Conv2d(self.embed_dim, Config.NUM_CLASSES, 1)

        # 5. Normalization Buffers
        # The backbone expects ImageNet normalized data.
        # We register these as buffers so they move to device automatically.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).
                              Values should be in [0, 1].

        Returns:
            torch.Tensor: Logits of shape (B, 1, H, W).
        """
        # 1. Normalize Input
        # Apply ImageNet mean/std normalization
        x_norm = (x - self.mean) / self.std

        # 2. Encoder Pass
        # Get features from the backbone
        outputs = self.encoder(x_norm)

        # Retrieve hidden states.
        # We take the last 4 tensors to ensure we get the outputs of the 4 transformer blocks,
        # handling cases where embeddings might be included in the tuple.
        features = outputs.hidden_states[-4:]
        c1, c2, c3, c4 = features

        # 3. Decoder Projection
        # Project all features to embedding dimension
        c1_emb = self.mlp_c1(c1)  # Shape: (B, 256, H/4, W/4)
        c2_emb = self.mlp_c2(c2)  # Shape: (B, 256, H/8, W/8)
        c3_emb = self.mlp_c3(c3)  # Shape: (B, 256, H/16, W/16)
        c4_emb = self.mlp_c4(c4)  # Shape: (B, 256, H/32, W/32)

        # 4. Upsampling
        # Upsample all features to the resolution of the largest feature map (c1 -> H/4)
        h_4, w_4 = c1_emb.shape[2], c1_emb.shape[3]

        c2_up = F.interpolate(
            c2_emb, size=(h_4, w_4), mode="bilinear", align_corners=False
        )
        c3_up = F.interpolate(
            c3_emb, size=(h_4, w_4), mode="bilinear", align_corners=False
        )
        c4_up = F.interpolate(
            c4_emb, size=(h_4, w_4), mode="bilinear", align_corners=False
        )

        # 5. Fusion
        # Concatenate along channel dimension
        fused = torch.cat([c1_emb, c2_up, c3_up, c4_up], dim=1)
        fused = self.fuse(fused)

        # 6. Prediction
        logits = self.head(fused)

        # 7. Final Upsample
        # Upsample logits to original input resolution (H, W)
        h, w = x.shape[2], x.shape[3]
        logits = F.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )

        return logits
