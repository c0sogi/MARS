import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config
from library.model_components import ContextGatingBlock, AttentionPooling


class WhaleConvNeXt(nn.Module):
    """
    WhaleConvNeXt: A Time-Preserving Context-Gated Hierarchical CRNN.

    Backbone: ConvNeXt-Pico (Pretrained)
    Modifications: Asymmetric Strides (2, 1) in deeper stages to preserve time.
    Neck: Context Gating + Multi-Scale Fusion.
    Head: Bi-GRU + Attention Pooling.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    ):
        super(WhaleConvNeXt, self).__init__()

        # 1. Initialize Backbone
        # We use features_only=False to access and modify the internal stages directly.
        # in_chans=1 allows timm to adapt the first projection layer.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=False,
        )

        # 2. Modify Downsampling for Time Preservation
        # "Stages 3 & 4: Modify the downsampling layers to use Asymmetric Strides (2, 1)"
        # In timm's ConvNeXt, stages are indexed 0, 1, 2, 3.
        # This corresponds to indices 2 and 3.
        self._modify_downsample(self.backbone.stages[2])
        self._modify_downsample(self.backbone.stages[3])

        # 3. Determine Feature Dimensions Dynamically
        # We perform a dummy forward pass to get exact channel counts.
        dummy_input = torch.randn(1, in_channels, 224, 224)
        with torch.no_grad():
            x = self.backbone.stem(dummy_input)
            feats = []
            for stage in self.backbone.stages:
                x = stage(x)
                feats.append(x)

        # We use Stage 2 (idx 1), Stage 3 (idx 2), and Stage 4 (idx 3)
        c_stage2 = feats[1].shape[1]
        c_stage3 = feats[2].shape[1]
        c_stage4 = feats[3].shape[1]

        # 4. Context Gating Components
        # Use Stage 4 (Deepest) as context for Stage 2 and Stage 3
        self.gate_stage2 = ContextGatingBlock(c_stage2, c_stage4)
        self.gate_stage3 = ContextGatingBlock(c_stage3, c_stage4)

        # 5. Fusion Components
        # We concatenate Gated S2, Gated S3, and Upsampled S4
        fusion_in_channels = c_stage2 + c_stage3 + c_stage4
        fusion_out_channels = 256
        self.fusion_bottleneck = nn.Conv2d(
            fusion_in_channels, fusion_out_channels, kernel_size=1
        )

        # 6. RNN & Head Components
        # Calculate RNN input dimension:
        # Input Spectrogram: 128 Mels.
        # Stage 2 Stride: 8 (Standard). Freq Dim = 128 / 8 = 16.
        # We upsample all features to Stage 2 resolution (F=16).
        # We flatten Frequency into Channels for the RNN.
        rnn_input_dim = fusion_out_channels * 16
        rnn_hidden_size = 128

        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.attn_pool = AttentionPooling(rnn_hidden_size * 2)
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def _modify_downsample(self, stage):
        """
        Replaces the Conv2d in the downsampling block with one using stride (2, 1).
        This preserves temporal resolution (stride 1 in time) while downsampling frequency.
        """
        # In timm ConvNeXt, stage.downsample is a Sequential(LayerNorm, Conv2d)
        if hasattr(stage, "downsample") and stage.downsample is not None:
            for name, module in stage.downsample.named_children():
                if isinstance(module, nn.Conv2d):
                    # Create new conv with asymmetric stride
                    new_conv = nn.Conv2d(
                        module.in_channels,
                        module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=(2, 1),  # (Freq, Time)
                        padding=module.padding,
                        bias=module.bias is not None,
                    )

                    # Copy pretrained weights
                    new_conv.weight.data = module.weight.data
                    if module.bias is not None:
                        new_conv.bias.data = module.bias.data

                    # Replace in-place
                    setattr(stage.downsample, name, new_conv)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrogram of shape (B, 1, F, T)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # 1. Backbone Forward
        x = self.backbone.stem(x)

        feats = []
        for stage in self.backbone.stages:
            x = stage(x)
            feats.append(x)

        # Extract specific stages
        f2 = feats[1]  # Stage 2
        f3 = feats[2]  # Stage 3
        f4 = feats[3]  # Stage 4 (Context)

        # 2. Context Gating
        gated_f2 = self.gate_stage2(f2, f4)
        gated_f3 = self.gate_stage3(f3, f4)

        # 3. Upsampling & Fusion
        # Target resolution is that of f2 (High-Res)
        target_size = f2.shape[2:]

        # Upsample f3
        if gated_f3.shape[2:] != target_size:
            gated_f3 = F.interpolate(
                gated_f3, size=target_size, mode="bilinear", align_corners=False
            )

        # Upsample f4
        if f4.shape[2:] != target_size:
            f4_up = F.interpolate(
                f4, size=target_size, mode="bilinear", align_corners=False
            )
        else:
            f4_up = f4

        # Concatenate
        fused = torch.cat([gated_f2, gated_f3, f4_up], dim=1)

        # Bottleneck
        fused = self.fusion_bottleneck(fused)  # (B, 256, F', T')

        # 4. Prepare for RNN
        # Collapse Frequency dimension
        B, C, F_dim, T_dim = fused.shape
        # Permute to (B, T, C, F) -> Reshape to (B, T, C*F)
        rnn_input = fused.permute(0, 3, 1, 2).reshape(B, T_dim, C * F_dim)

        # 5. RNN & Head
        rnn_out, _ = self.gru(rnn_input)
        pooled = self.attn_pool(rnn_out)
        logits = self.fc(pooled)

        return logits
