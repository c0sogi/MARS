import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class EncoderCNN(nn.Module):
    """
    Encoder network using MobileNetV3-Small backbone.
    Extracts spatial feature maps from input images.
    """

    def __init__(self, encoder_dim=Config.ENCODER_DIM):
        super(EncoderCNN, self).__init__()
        # Load pre-trained MobileNetV3 Small
        # features_only=True returns a list of feature maps. We take the last one.
        # in_chans=Config.IMAGE_CHANNELS (1) for grayscale input.
        self.backbone = timm.create_model(
            Config.ENCODER_NAME,
            pretrained=True,
            in_chans=Config.IMAGE_CHANNELS,
            features_only=True,
        )

        # Determine the number of output channels from the backbone
        # For mobilenetv3_small_100, the last feature map typically has 576 channels.
        self.feature_info = self.backbone.feature_info
        last_channel_dim = self.feature_info[-1]["num_chs"]

        # 1x1 Convolution to project backbone features to the desired embedding dimension
        self.conv_proj = nn.Conv2d(last_channel_dim, encoder_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(encoder_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images):
        """
        Forward pass of the encoder.
        Args:
            images: Input tensor of shape (B, C, H, W)
        Returns:
            features: Spatial features of shape (B, L, encoder_dim), where L = H*W
        """
        # Extract features from backbone
        features_list = self.backbone(images)
        features = features_list[-1]  # Shape: (B, last_channel_dim, H_feat, W_feat)

        # Project dimensions
        features = self.conv_proj(features)
        features = self.bn(features)
        features = self.relu(features)

        # Rearrange to (B, H*W, C) for attention mechanism
        # Permute: (B, C, H, W) -> (B, H, W, C)
        features = features.permute(0, 2, 3, 1)
        B, H, W, C = features.shape
        features = features.view(
            B, H * W, C
        )  # Flatten spatial dimensions to sequence L

        return features


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    Computes context vector as a weighted sum of encoder features based on decoder hidden state.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden):
        """
        Args:
            features: Encoder output features (B, L, encoder_dim)
            hidden: Decoder hidden state (B, decoder_dim)
        Returns:
            context_vector: Weighted sum of features (B, encoder_dim)
            alpha: Attention weights (B, L)
        """
        # Calculate attention scores
        # att1: (B, L, attention_dim)
        att1 = self.encoder_att(features)
        # att2: (B, attention_dim) -> Unsqueeze to (B, 1, attention_dim) for broadcasting
        att2 = self.decoder_att(hidden)

        # Additive attention: score = v^T * tanh(W1*features + W2*hidden)
        combined = self.relu(att1 + att2.unsqueeze(1))
        energy = self.full_att(combined)  # (B, L, 1)

        # Calculate probability weights
        alpha = self.softmax(energy)  # (B, L, 1)

        # Calculate context vector
        # Element-wise multiplication (broadcasting alpha) and sum over L dimension
        context_vector = (features * alpha).sum(dim=1)  # (B, encoder_dim)

        return context_vector, alpha.squeeze(2)


class DecoderRNN(nn.Module):
    """
    GRU-based Decoder with Attention.
    Generates the sequence one token at a time.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        decoder_dim,
        encoder_dim,
        attention_dim,
        dropout=0.5,
    ):
        super(DecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)

        # GRU Input: Concatenation of (Embedding of prev token) and (Context Vector)
        self.gru = nn.GRU(embed_dim + encoder_dim, decoder_dim, batch_first=True)

        # Output classifier
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, features):
        """
        Single step forward pass.
        Args:
            x: Input token indices (B,)
            hidden: Previous hidden state (1, B, decoder_dim)
            features: Encoder features (B, L, encoder_dim)
        Returns:
            preds: Logits for next token (B, vocab_size)
            hidden: Updated hidden state (1, B, decoder_dim)
            attention_weights: Attention weights used (B, L)
        """
        # 1. Calculate Attention
        # Use the hidden state from the previous time step (squeeze layer dim)
        # hidden is (1, B, D), we need (B, D) for attention calculation
        hidden_squeezed = hidden.squeeze(0)
        context_vector, attention_weights = self.attention(features, hidden_squeezed)

        # 2. Embedding
        # (B) -> (B, embed_dim)
        embed = self.embedding(x)
        embed = self.dropout(embed)

        # 3. Concatenate context and embedding
        # (B, embed_dim + encoder_dim)
        gru_input = torch.cat((embed, context_vector), dim=1)

        # 4. GRU Step
        # Add sequence dimension for GRU: (B, 1, input_dim)
        gru_input = gru_input.unsqueeze(1)
        output, hidden = self.gru(gru_input, hidden)

        # 5. Prediction
        # output: (B, 1, decoder_dim) -> squeeze to (B, decoder_dim)
        preds = self.fc(output.squeeze(1))

        return preds, hidden, attention_weights

    def init_hidden(self, batch_size, device):
        """Initializes the hidden state with zeros."""
        return torch.zeros(1, batch_size, self.decoder_dim).to(device)


class ShowAttendTell(nn.Module):
    """
    Wrapper class combining Encoder and Decoder.
    Handles the full forward pass for sequence generation.
    """

    def __init__(self, vocab_size):
        super(ShowAttendTell, self).__init__()
        self.vocab_size = vocab_size

        self.encoder = EncoderCNN(encoder_dim=Config.ENCODER_DIM)

        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            embed_dim=Config.EMBED_DIM,
            decoder_dim=Config.DECODER_DIM,
            encoder_dim=Config.ENCODER_DIM,
            attention_dim=Config.ATTENTION_DIM,
            dropout=Config.DROPOUT,
        )

    def forward(self, images, captions=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for training or inference.

        Args:
            images: Input images (B, C, H, W)
            captions: Ground truth sequences (B, max_len). If None, runs in inference mode.
            teacher_forcing_ratio: Probability of using ground truth as next input during training.

        Returns:
            outputs: Tensor of logits (B, max_len, vocab_size)
        """
        batch_size = images.size(0)
        device = images.device

        # 1. Encode Images
        features = self.encoder(images)  # (B, L, encoder_dim)

        # 2. Initialize Decoder
        hidden = self.decoder.init_hidden(batch_size, device)

        # 3. Sequence Generation Loop
        if captions is not None:
            # Training/Validation Mode
            max_len = captions.size(1)
            outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(device)

            # First input is the <SOS> token (index 1)
            # Assuming captions are padded and start with SOS
            input_token = captions[:, 0]

            # Loop starts from 1 because we predict the token at t based on t-1
            # We fill outputs[:, t, :] which corresponds to the prediction for captions[:, t]
            for t in range(1, max_len):
                preds, hidden, _ = self.decoder(input_token, hidden, features)
                outputs[:, t, :] = preds

                # Teacher Forcing decision
                use_teacher_forcing = (
                    True if torch.rand(1).item() < teacher_forcing_ratio else False
                )

                if use_teacher_forcing:
                    input_token = captions[:, t]
                else:
                    input_token = preds.argmax(1)
        else:
            # Inference Mode (Greedy Decoding)
            max_len = Config.MAX_SEQUENCE_LENGTH
            outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(device)

            # Start with <SOS> token (Index 1 based on tokenizer convention)
            input_token = torch.full((batch_size,), 1, dtype=torch.long).to(device)

            for t in range(1, max_len):
                preds, hidden, _ = self.decoder(input_token, hidden, features)
                outputs[:, t, :] = preds
                input_token = preds.argmax(1)

                # Note: We do not break on EOS here to maintain batch tensor shape.
                # EOS handling is done during post-processing of the output indices.

        return outputs
