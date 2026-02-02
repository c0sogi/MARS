import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class Encoder(nn.Module):
    """
    EfficientNet-B0 Encoder.
    Outputs:
        - Spatial features for Attention (B, C, H, W)
        - Global features for Attribute Branch & Decoder Init (B, C)
    """

    def __init__(self, config: Config):
        super().__init__()
        # Load pretrained EfficientNet-B0
        # features_only=True returns a list of feature maps from different stages
        self.backbone = timm.create_model(
            config.ENCODER_NAME, pretrained=True, features_only=True
        )

        # Get the channel count of the final feature map
        # EfficientNet-B0 typically has 1280 channels at the last stage
        self.out_channels = self.backbone.feature_info[-1]["num_chs"]

    def forward(self, x):
        # x: (B, 3, H, W)
        features_list = self.backbone(x)

        # Take the last feature map: (B, 1280, H/32, W/32)
        spatial_features = features_list[-1]

        # Global Average Pooling to get a vector representation
        # (B, C, H', W') -> (B, C, 1, 1) -> (B, C)
        global_features = F.adaptive_avg_pool2d(spatial_features, (1, 1))
        global_features = global_features.view(spatial_features.size(0), -1)

        return spatial_features, global_features


class AttributeBranch(nn.Module):
    """
    MLP Regression Head for predicting atom counts and sequence length.
    """

    def __init__(self, input_dim, num_attributes, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_attributes),
        )

    def forward(self, x):
        # x: (B, encoder_dim)
        # out: (B, num_attributes)
        return self.net(x)


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.W_features = nn.Linear(encoder_dim, attention_dim)
        self.W_hidden = nn.Linear(decoder_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden):
        # features: (B, L, encoder_dim) where L = H*W
        # hidden: (B, decoder_dim)

        # Score calculation
        # (B, L, attn_dim)
        features_score = self.W_features(features)
        # (B, attn_dim) -> (B, 1, attn_dim)
        hidden_score = self.W_hidden(hidden).unsqueeze(1)

        # (B, L, 1)
        energy = self.V(torch.tanh(features_score + hidden_score))

        # Attention weights: (B, L)
        attention_weights = F.softmax(energy.squeeze(2), dim=1)

        # Context vector: Weighted sum of features
        # (B, 1, L) @ (B, L, encoder_dim) -> (B, 1, encoder_dim)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), features)

        # (B, encoder_dim)
        context_vector = context_vector.squeeze(1)

        return context_vector, attention_weights


class Decoder(nn.Module):
    """
    LSTM Decoder with Persistent Attribute Conditioning.
    """

    def __init__(self, config: Config, vocab_size, encoder_dim):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = config.DECODER_DIM
        self.embed_dim = config.EMBED_DIM
        self.num_attributes = config.NUM_ATTRIBUTES

        # Embedding layer for characters
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)

        # Attention Mechanism
        self.attention = BahdanauAttention(
            encoder_dim, self.decoder_dim, config.ATTENTION_DIM
        )

        # Initialization layers
        # Projects concatenated [Global Features + Attributes] to Decoder state size
        self.init_h = nn.Linear(encoder_dim + self.num_attributes, self.decoder_dim)
        self.init_c = nn.Linear(encoder_dim + self.num_attributes, self.decoder_dim)

        # LSTM Cell
        # Input: [Embedding (256) + Context (1280) + Attributes (10)]
        self.lstm_input_dim = self.embed_dim + encoder_dim + self.num_attributes
        self.lstm = nn.LSTMCell(self.lstm_input_dim, self.decoder_dim)

        # Output Classifier
        self.classifier = nn.Linear(self.decoder_dim, vocab_size)
        self.dropout = nn.Dropout(config.DROPOUT)

        # Define SOS index based on Tokenizer logic (PAD=0, SOS=1, EOS=2, UNK=3)
        self.sos_idx = 1

    def init_hidden_state(self, global_features, attributes):
        """
        Initializes LSTM hidden and cell states using global features and attributes.
        """
        combined = torch.cat([global_features, attributes], dim=1)
        h = torch.tanh(self.init_h(combined))
        c = torch.tanh(self.init_c(combined))
        return h, c

    def forward(
        self,
        features,
        global_features,
        attributes,
        targets=None,
        teacher_forcing_ratio=0.0,
    ):
        """
        Args:
            features: (B, C, H, W) - Spatial features from Encoder
            global_features: (B, C) - Global features from Encoder
            attributes: (B, num_attributes) - Predicted attributes
            targets: (B, max_len) - Ground truth token indices (optional)
            teacher_forcing_ratio: Float between 0 and 1
        """
        batch_size = features.size(0)

        # Flatten spatial features for attention: (B, C, H, W) -> (B, H*W, C)
        # Permute to (B, H, W, C) first, then flatten spatial dims
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch_size, -1, self.encoder_dim)

        # Initialize hidden states
        h, c = self.init_hidden_state(global_features, attributes)

        # Determine sequence length
        if targets is not None:
            max_len = targets.size(1)
        else:
            max_len = self.config.MAX_LEN

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(
            self.config.DEVICE
        )

        # Initialize input with SOS token
        current_input = torch.full(
            (batch_size,), self.sos_idx, dtype=torch.long, device=self.config.DEVICE
        )

        for t in range(max_len):
            # 1. Calculate Attention Context
            context, _ = self.attention(features, h)

            # 2. Embed current character
            embedded = self.embedding(current_input)

            # 3. Concatenate: Embedding + Context + Attributes (Persistent Conditioning)
            lstm_input = torch.cat([embedded, context, attributes], dim=1)

            # 4. LSTM Step
            h, c = self.lstm(lstm_input, (h, c))

            # 5. Predict next token
            output = self.classifier(self.dropout(h))
            outputs[:, t, :] = output

            # 6. Select input for next step
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher Forcing: Use ground truth
                # If t is the last step, we don't need next input
                if t + 1 < max_len:
                    current_input = targets[:, t + 1]
            else:
                # Greedy Decoding: Use predicted token
                current_input = output.argmax(1)

        return outputs


class AttributeAugmentedAttnNet(nn.Module):
    """
    Main Model: Idea 5 (Attribute-Augmented Attention Network)
    Combines Encoder, Attribute Branch, and Decoder.
    """

    def __init__(self, config: Config, vocab_size):
        super().__init__()
        self.encoder = Encoder(config)

        self.attribute_branch = AttributeBranch(
            input_dim=self.encoder.out_channels,
            num_attributes=config.NUM_ATTRIBUTES,
            dropout_rate=config.DROPOUT,
        )

        self.decoder = Decoder(config, vocab_size, self.encoder.out_channels)

    def forward(self, images, targets=None, teacher_forcing_ratio=0.0):
        """
        Forward pass for Multi-Task Learning.
        Returns:
            seq_logits: (B, max_len, vocab_size)
            attributes_pred: (B, num_attributes)
        """
        # 1. Encode Images
        spatial_features, global_features = self.encoder(images)

        # 2. Predict Attributes (Auxiliary Task)
        attributes_pred = self.attribute_branch(global_features)

        # 3. Decode Sequence (Main Task)
        # The decoder uses the predicted attributes for conditioning
        seq_logits = self.decoder(
            spatial_features,
            global_features,
            attributes_pred,
            targets,
            teacher_forcing_ratio,
        )

        return seq_logits, attributes_pred
