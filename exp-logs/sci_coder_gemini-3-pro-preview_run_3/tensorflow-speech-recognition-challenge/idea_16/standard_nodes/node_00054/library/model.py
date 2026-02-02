import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class FrequencyAttention(nn.Module):
    """
    Learns to weight frequency bands dynamically before pooling them.
    Input: (B, C, F, T)
    Output: (B, C, T)
    """

    def __init__(self, channels):
        super().__init__()
        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C, F, T)

        # Calculate attention scores over the Frequency dimension (dim=2)
        # score: (B, 1, F, T)
        score = self.attn_conv(x)
        weights = F.softmax(score, dim=2)

        # Weighted sum over Frequency
        # out: (B, C, T)
        out = torch.sum(x * weights, dim=2)
        return out


class AttnPoolingHead(nn.Module):
    """
    Multi-Head Attention Pooling to aggregate temporal features.
    Input: (B, T, D)
    Output: (B, Num_Heads * D)
    """

    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim

        # Project input to get attention scores for each head
        # We use a simple self-attention mechanism where the query is learned implicitly
        self.attn_linear = nn.Linear(input_dim, num_heads)

    def forward(self, x):
        # x: (B, T, D)

        # Compute scores: (B, T, num_heads)
        scores = self.attn_linear(x)

        # Normalize scores over Time (dim 1)
        weights = F.softmax(scores, dim=1)

        # Weighted Aggregation
        # Transpose weights to (B, num_heads, T)
        weights_t = weights.transpose(1, 2)

        # Batch Matrix Multiplication
        # (B, H, T) @ (B, T, D) -> (B, H, D)
        out = torch.bmm(weights_t, x)

        # Flatten heads
        # (B, H * D)
        out = out.reshape(out.size(0), -1)
        return out


class ResNeStBackbone(nn.Module):
    """
    ResNeSt50 backbone with modified strides to preserve temporal resolution.
    """

    def __init__(self, model_name, pretrained=True, in_channels=3):
        super().__init__()
        # Load model with features_only=True to get intermediate feature maps
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(4,),  # We only need the output of the last stage
        )

        # Modify strides to prevent excessive downsampling in the temporal dimension
        self._modify_strides()

        # Determine output channels dynamically
        dummy = torch.randn(2, in_channels, 224, 224)
        with torch.no_grad():
            # features is a list of tensors
            features = self.model(dummy)
            last_feat = features[-1]

        self.out_channels = last_feat.shape[1]

    def _modify_strides(self):
        """
        Iterates through layers to set downsampling strides to 1.
        This increases the spatial resolution of the final feature map.
        Cite Lesson 17: Preserve Temporal Resolution
        """
        # Added layer2 to ensure sufficient temporal resolution with native inputs
        layers_to_modify = ["layer2", "layer3", "layer4"]

        for layer_name in layers_to_modify:
            if not hasattr(self.model, layer_name):
                continue

            layer = getattr(self.model, layer_name)

            # Cite debug_lesson_7: Use recursive traversal to modify attributes in nested model architectures
            for m in layer.modules():
                if hasattr(m, "stride"):
                    if m.stride == (2, 2) or m.stride == 2:
                        m.stride = (1, 1)

                        # Cite debug_lesson_12: Adjust Pooling Kernels When Removing Strides in Residual Networks
                        if isinstance(m, nn.AvgPool2d):
                            if m.kernel_size == (2, 2) or m.kernel_size == 2:
                                m.kernel_size = (1, 1)
                                m.padding = (0, 0)

    def forward(self, x):
        # Returns the last feature map
        return self.model(x)[-1]


class FreqAttnResNeStCRNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Backbone
        self.backbone = ResNeStBackbone(
            model_name=Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            in_channels=Config.IN_CHANNELS,
        )

        backbone_dim = self.backbone.out_channels

        # 2. Neck: Frequency Attention
        self.freq_attn = FrequencyAttention(backbone_dim)

        # 3. Sequence Modeling: BiGRU
        self.rnn = nn.GRU(
            input_size=backbone_dim,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.RNN_DROPOUT if Config.RNN_LAYERS > 1 else 0,
        )

        rnn_out_dim = (
            Config.RNN_HIDDEN_SIZE * 2
            if Config.BIDIRECTIONAL
            else Config.RNN_HIDDEN_SIZE
        )

        # 4. Head: Multi-Head Attention Pooling
        self.attn_pooling = AttnPoolingHead(rnn_out_dim, Config.ATTN_NUM_HEADS)

        # 5. Classifier
        # Input dim is (Num_Heads * RNN_Out_Dim)
        clf_in_dim = Config.ATTN_NUM_HEADS * rnn_out_dim
        self.classifier = nn.Linear(clf_in_dim, Config.NUM_CLASSES)

    def forward(self, x):
        # Input: (B, 3, 224, 224)

        # Backbone Extraction
        # feat: (B, C, F, T) -> e.g., (B, 2048, 28, 28)
        feat = self.backbone(x)

        # Frequency Attention Pooling
        # Collapses Frequency dimension while weighting important bands
        # x_seq: (B, C, T)
        x_seq = self.freq_attn(feat)

        # Permute for RNN: (B, T, C)
        x_seq = x_seq.permute(0, 2, 1)

        # RNN Processing
        self.rnn.flatten_parameters()
        # rnn_out: (B, T, D)
        rnn_out, _ = self.rnn(x_seq)

        # Temporal Aggregation (Attention Pooling)
        # pooled: (B, H * D)
        pooled = self.attn_pooling(rnn_out)

        # Classification
        logits = self.classifier(pooled)

        return logits
