import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig

from library.config import Config


class StratifiedSegFormer(nn.Module):
    """
    SegFormer architecture with MiT-B1 encoder and Lightweight MLP decoder.
    Designed for 2.5D stratified input (3 channels).
    """

    def __init__(self, pretrained=True):
        super(StratifiedSegFormer, self).__init__()

        # ----------------------------------------------------------------
        # Encoder: Mix Transformer (MiT)
        # ----------------------------------------------------------------
        # We use the HuggingFace implementation.
        model_name = Config.MODEL_ARCH

        try:
            if pretrained:
                self.encoder = SegformerModel.from_pretrained(model_name)
            else:
                config = SegformerConfig.from_pretrained(model_name)
                self.encoder = SegformerModel(config)
        except Exception as e:
            print(f"Warning: Could not load pretrained model {model_name}. Error: {e}")
            print("Initializing random weights based on default config (MiT-B2).")
            # Fallback configuration matching MiT-B2
            config = SegformerConfig(
                num_channels=3,
                num_encoder_blocks=4,
                depths=[3, 4, 6, 3],
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[64, 128, 320, 512],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
            )
            self.encoder = SegformerModel(config)

        # Handle input channel mismatch if Config changes (Adapter)
        # The encoder expects 3 channels (RGB)
        if Config.IN_CHANNELS != 3:
            self.input_adapter = nn.Conv2d(
                Config.IN_CHANNELS, 3, kernel_size=1, bias=False
            )
        else:
            self.input_adapter = None

        # ----------------------------------------------------------------
        # Decoder: Lightweight All-MLP
        # ----------------------------------------------------------------
        # Retrieve feature channels dynamically from the encoder config
        self.encoder_channels = self.encoder.config.hidden_sizes
        self.embedding_dim = 256  # Common embedding dimension for decoder

        # MLP layers to project encoder features to the same embedding dimension
        # We use Conv2d with kernel_size=1 which is equivalent to a Linear layer per pixel
        self.linear_c1 = nn.Conv2d(
            self.encoder_channels[0], self.embedding_dim, kernel_size=1
        )
        self.linear_c2 = nn.Conv2d(
            self.encoder_channels[1], self.embedding_dim, kernel_size=1
        )
        self.linear_c3 = nn.Conv2d(
            self.encoder_channels[2], self.embedding_dim, kernel_size=1
        )
        self.linear_c4 = nn.Conv2d(
            self.encoder_channels[3], self.embedding_dim, kernel_size=1
        )

        # Fuse layer: Concatenation of 4 feature maps -> 4 * embedding_dim
        self.linear_fuse = nn.Conv2d(
            self.embedding_dim * 4, self.embedding_dim, kernel_size=1
        )
        self.dropout = nn.Dropout(0.1)

        # Classifier: Embedding Dim -> Num Classes (1 for binary)
        self.linear_pred = nn.Conv2d(
            self.embedding_dim, Config.NUM_CLASSES, kernel_size=1
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, IN_CHANNELS, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, NUM_CLASSES, H, W)
        """
        # Adapter for input channels if necessary
        if self.input_adapter is not None:
            x = self.input_adapter(x)

        # 1. Encoder Forward Pass
        # output_hidden_states=True ensures we get the feature maps from all stages
        outputs = self.encoder(x, output_hidden_states=True)

        # hidden_states is a tuple containing outputs from the PatchEmbed and the 4 Transformer blocks.
        # We take the last 4 elements corresponding to the 4 hierarchical stages.
        # Shapes for 512x512 input:
        # c1: (B, 64, 128, 128)  - Stride 4
        # c2: (B, 128, 64, 64)   - Stride 8
        # c3: (B, 320, 32, 32)   - Stride 16
        # c4: (B, 512, 16, 16)   - Stride 32
        features = outputs.hidden_states[-4:]
        c1, c2, c3, c4 = features[0], features[1], features[2], features[3]

        # 2. Decoder Forward Pass
        # Target shape for fusion is c1 shape (H/4, W/4)
        n, _, h, w = c1.shape

        # Stage 4 (Stride 32) -> Project & Upsample to Stride 4
        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=(h, w), mode="bilinear", align_corners=False)

        # Stage 3 (Stride 16) -> Project & Upsample to Stride 4
        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=(h, w), mode="bilinear", align_corners=False)

        # Stage 2 (Stride 8) -> Project & Upsample to Stride 4
        _c2 = self.linear_c2(c2)
        _c2 = F.interpolate(_c2, size=(h, w), mode="bilinear", align_corners=False)

        # Stage 1 (Stride 4) -> Project only
        _c1 = self.linear_c1(c1)

        # Concatenate along channel dimension
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)

        # Fuse
        x = self.linear_fuse(_c)
        x = self.dropout(x)

        # Predict
        x = self.linear_pred(x)

        # 3. Final Upsample
        # Upsample from H/4 to H (Stride 4 to Stride 1)
        # For 512x512 input, x is currently 128x128. Output becomes 512x512.
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)

        return x
