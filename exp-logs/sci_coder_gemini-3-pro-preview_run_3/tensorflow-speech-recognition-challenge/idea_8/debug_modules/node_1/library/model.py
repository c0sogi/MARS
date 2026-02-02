import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio
from library.config import Config


class MultiHeadAttentionPooling(nn.Module):
    """
    Aggregates temporal features using multiple attention heads.
    Allows the model to focus on different parts of the audio clip (e.g., start, middle, end)
    simultaneously.
    """

    def __init__(self, in_dim, num_heads):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads

        # Projects input to 'num_heads' attention scores
        self.attn_linear = nn.Linear(in_dim, num_heads)

    def forward(self, x):
        # Input x: (Batch, Time, Dim)

        # 1. Compute Attention Scores -> (Batch, Time, NumHeads)
        attn_logits = self.attn_linear(x)

        # 2. Normalize across Time -> (Batch, Time, NumHeads)
        attn_weights = F.softmax(attn_logits, dim=1)

        # 3. Weighted Sum
        # Transpose weights to (Batch, NumHeads, Time) for matrix multiplication
        attn_weights = attn_weights.transpose(1, 2)

        # BMM: (B, H, T) x (B, T, D) -> (B, H, D)
        weighted_sum = torch.bmm(attn_weights, x)

        # 4. Flatten Heads -> (Batch, NumHeads * Dim)
        out = weighted_sum.reshape(x.size(0), -1)

        return out


class SKResNetConformer(nn.Module):
    """
    Hybrid Architecture:
    1. Multi-Resolution Input (handled by dataset/feature_extractor)
    2. SK-ResNet34 Backbone (Spatial/Spectral Feature Extraction) with Modified Strides
    3. Conformer Encoder (Temporal/Global Context)
    4. Multi-Head Attention Pooling (Aggregation)
    """

    def __init__(self):
        super().__init__()

        # --- 1. Backbone: SK-ResNet34 ---
        # We load the model without features_only=True initially to easily access layers
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, in_chans=Config.IN_CHANNELS
        )

        # --- 2. Modify Strides ---
        # Standard ResNet34 has 32x downsampling (stride 2 in stem, layer2, layer3, layer4).
        # We change layer3 and layer4 strides to 1 to keep temporal resolution high (8x total downsampling).
        self._modify_strides()

        # --- 3. Bridge / Projection ---
        # ResNet34 Layer 4 output channels = 512
        self.backbone_dim = 512

        # If Conformer dim differs from Backbone dim, project it.
        if self.backbone_dim != Config.CONFORMER_DIM:
            self.projection = nn.Linear(self.backbone_dim, Config.CONFORMER_DIM)
        else:
            self.projection = nn.Identity()

        # --- 4. Conformer Neck ---
        # Uses torchaudio's optimized implementation
        self.conformer = torchaudio.models.Conformer(
            input_dim=Config.CONFORMER_DIM,
            num_heads=Config.CONFORMER_HEADS,
            ffn_dim=Config.CONFORMER_DIM * 4,
            num_layers=Config.CONFORMER_LAYERS,
            depthwise_conv_kernel_size=3,  # Small kernel for local context in feature space
            dropout=Config.CONFORMER_DROPOUT,
        )

        # --- 5. Pooling Head ---
        self.pooling = MultiHeadAttentionPooling(
            in_dim=Config.CONFORMER_DIM, num_heads=Config.POOLING_HEADS
        )

        # --- 6. Classifier ---
        # Input dimension is (NumHeads * ConformerDim)
        self.classifier = nn.Linear(
            Config.POOLING_HEADS * Config.CONFORMER_DIM, Config.NUM_CLASSES
        )

    def _modify_strides(self):
        """
        Iterates over layer3 and layer4 to set strides to 1.
        This preserves the time dimension in the feature map.
        """
        layers_to_modify = ["layer3", "layer4"]

        for layer_name in layers_to_modify:
            if not hasattr(self.backbone, layer_name):
                continue

            layer = getattr(self.backbone, layer_name)
            # ResNet layers are Sequential containers of Blocks.
            # The first block handles the downsampling.
            first_block = layer[0]

            # 1. Modify Main Conv Stride
            if hasattr(first_block, "conv1"):
                first_block.conv1.stride = (1, 1)
            elif hasattr(first_block, "conv_dw"):  # Handle some variants
                first_block.conv_dw.stride = (1, 1)

            # 2. Modify Downsample Layer Stride (Shortcut path)
            if (
                hasattr(first_block, "downsample")
                and first_block.downsample is not None
            ):
                # downsample is typically Sequential(Conv, BN)
                if isinstance(first_block.downsample, nn.Sequential):
                    # The conv is usually the first item
                    for module in first_block.downsample:
                        if isinstance(module, nn.Conv2d):
                            module.stride = (1, 1)
                            break

    def forward(self, x):
        # Input: (Batch, 3, 64, 101)

        # 1. Backbone Feature Extraction
        # forward_features returns (B, 512, H, W)
        x = self.backbone.forward_features(x)

        # 2. Frequency Pooling & Reshape
        # Current shape: (B, 512, F', T')
        # We average pool the frequency dimension to collapse it
        x = torch.mean(x, dim=2)  # -> (B, 512, T')

        # Permute to (B, T', 512) for Conformer
        x = x.permute(0, 2, 1)

        # 3. Projection
        x = self.projection(x)

        # 4. Conformer
        # Conformer expects lengths argument. Since we have fixed size batches/padding,
        # we assume full length valid or let the model attend to padding (it's robust enough).
        # Construct lengths tensor
        b, t, d = x.shape
        lengths = torch.full((b,), t, dtype=torch.int32, device=x.device)

        # Returns (output, lengths)
        x, _ = self.conformer(x, lengths)

        # 5. Attention Pooling
        # (B, T, D) -> (B, Heads*D)
        x = self.pooling(x)

        # 6. Classifier
        logits = self.classifier(x)

        return logits
