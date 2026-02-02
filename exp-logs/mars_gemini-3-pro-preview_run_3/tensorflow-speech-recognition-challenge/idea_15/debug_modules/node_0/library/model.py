import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class FrequencyAttentivePooling(nn.Module):
    """
    Learns to pool the frequency dimension by calculating an attention map
    over frequencies for each time step.

    Input: (Batch, Channels, Freq, Time)
    Output: (Batch, Channels, Time)
    """

    def __init__(self, channels):
        super().__init__()
        # 1x1 Conv to compute attention scores from features
        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C, F, T)

        # Calculate attention scores: (B, 1, F, T)
        attn_logits = self.attn_conv(x)

        # Softmax over Frequency dimension (dim=2) to get weights summing to 1
        attn_weights = F.softmax(attn_logits, dim=2)

        # Apply weights: (B, C, F, T) * (B, 1, F, T) -> (B, C, F, T)
        weighted_x = x * attn_weights

        # Sum over Frequency dimension to aggregate: (B, C, T)
        x_pooled = weighted_x.sum(dim=2)

        return x_pooled


class MultiHeadAttentionPooling(nn.Module):
    """
    Aggregates a sequence of temporal features into a single vector using
    Multi-Head Attention with a learnable Query vector.

    Input: (Batch, Time, Input_Dim)
    Output: (Batch, Input_Dim)
    """

    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim

        # Learnable Query vector: (1, 1, Input_Dim)
        # This represents the "prototype" of relevant features the model looks for.
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))

        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        # x: (Batch, Time, Input_Dim)
        B = x.size(0)

        # Expand query for the batch: (B, 1, Input_Dim)
        query = self.query.expand(B, -1, -1)

        # Multihead Attention
        # Query: Learnable vector
        # Key, Value: Input sequence x
        # Output: (B, 1, Input_Dim), weights
        attn_output, _ = self.mha(query, x, x)

        # Squeeze time dimension to get final embedding: (B, Input_Dim)
        return attn_output.squeeze(1)


class FrequencyAttentiveResNeStCRNN(nn.Module):
    """
    Frequency-Attentive Multi-Resolution ResNeSt-CRNN.

    Pipeline:
    1. ResNeSt50 Backbone (Output Stride 8 for high temporal resolution)
    2. Frequency Attentive Pooling (Aggregates Freq dim dynamically)
    3. BiGRU (Sequence Modeling)
    4. Multi-Head Attention Pooling (Aggregates Time dim)
    5. Linear Classifier
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone: ResNeSt50
        # We use output_stride=8. This changes the strides of the last two stages
        # to 1 and applies dilation, preserving spatial (temporal) resolution.
        # Input: (B, 3, 224, 224) -> Output: (B, 2048, 28, 28)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(4,),  # Get the output of the final stage
            output_stride=8,
        )

        # Determine backbone output channels dynamically
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            # features is a list because features_only=True
            last_feat = features[-1]
            backbone_channels = last_feat.shape[1]

        # 2. Neck: Frequency Attentive Pooling
        self.freq_pooling = FrequencyAttentivePooling(backbone_channels)

        # 3. RNN: BiGRU
        self.rnn = nn.GRU(
            input_size=backbone_channels,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.RNN_LAYERS > 1 else 0,
        )

        rnn_out_dim = Config.RNN_HIDDEN_SIZE * 2  # Bidirectional

        # 4. Head: Multi-Head Attention Pooling
        self.attn_pooling = MultiHeadAttentionPooling(
            rnn_out_dim, num_heads=Config.ATTENTION_HEADS
        )

        # 5. Classifier
        self.fc = nn.Linear(rnn_out_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, 3, 224, 224)
        Returns:
            Logits of shape (Batch, Num_Classes)
        """
        # Backbone Feature Extraction
        features_list = self.backbone(x)
        x = features_list[-1]  # (B, 2048, 28, 28) -> (B, C, F, T)

        # Frequency Pooling
        # Collapses Frequency (H) dimension using attention
        x = self.freq_pooling(x)  # (B, 2048, 28) where 28 is Time

        # Prepare for RNN: Permute to (Batch, Time, Channels)
        x = x.permute(0, 2, 1)  # (B, 28, 2048)

        # RNN Sequence Modeling
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # (B, 28, 512)

        # Attention Pooling (Temporal Aggregation)
        x = self.attn_pooling(x)  # (B, 512)

        # Classification
        x = self.fc(x)  # (B, 12)

        return x
