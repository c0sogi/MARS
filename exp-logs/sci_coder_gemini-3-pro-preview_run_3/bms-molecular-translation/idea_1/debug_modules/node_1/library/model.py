import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class EncoderCNN(nn.Module):
    """
    Encoder Network: MobileNetV3-Large backbone.
    Encodes the image into a fixed global vector to initialize the Decoder RNN.
    """

    def __init__(
        self,
        model_name=Config.ENCODER_NAME,
        pretrained=Config.ENCODER_PRETRAINED,
        hidden_size=Config.DECODER_HIDDEN_SIZE,
    ):
        super(EncoderCNN, self).__init__()

        # Load pre-trained backbone
        # num_classes=0 and global_pool='avg' ensures we get the pooled feature vector
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Determine the output dimension of the backbone dynamically.
        # timm's num_features attribute (often 960 for MobileNetV3) can be inconsistent
        # with the actual output dimension (1280) when num_classes=0, as it may include
        # a projection layer in the head.
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            dummy_out = self.backbone(dummy_input)
            self.feature_dim = dummy_out.shape[1]

        # Linear projections to map image features to Decoder's hidden and cell states
        # We use separate projections for h_0 and c_0
        self.bn_h = nn.Linear(self.feature_dim, hidden_size)
        self.bn_c = nn.Linear(self.feature_dim, hidden_size)

        # Batch Norm to help with training stability
        self.bn = nn.BatchNorm1d(self.feature_dim)

        self.activation = nn.Tanh()

    def forward(self, images):
        """
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]

        Returns:
            tuple: (h_0, c_0) initialized states for the LSTM
                   Shapes: [num_layers, B, hidden_size]
        """
        # Extract features [B, feature_dim]
        features = self.backbone(images)
        features = self.bn(features)

        # Project to hidden size
        h = self.bn_h(features)
        c = self.bn_c(features)

        # Apply activation
        h = self.activation(h)
        c = self.activation(c)

        # Reshape for LSTM: [num_layers, batch_size, hidden_size]
        # Config.DECODER_LAYERS is used here. Assuming 1 layer for now based on Config.
        # If num_layers > 1, we might need to repeat or have more projections.
        # For simplicity and efficiency (baseline), we expand dims.
        h = h.unsqueeze(0).repeat(Config.DECODER_LAYERS, 1, 1)
        c = c.unsqueeze(0).repeat(Config.DECODER_LAYERS, 1, 1)

        return h, c


class DecoderRNN(nn.Module):
    """
    Decoder Network: LSTM.
    Generates the InChI string sequence character by character.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=Config.EMBEDDING_DIM,
        hidden_size=Config.DECODER_HIDDEN_SIZE,
        num_layers=Config.DECODER_LAYERS,
        dropout=Config.DROPOUT,
    ):
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, captions, h, c):
        """
        Forward pass for training (Teacher Forcing).

        Args:
            captions (torch.Tensor): Input sequence indices [B, seq_len]
            h (torch.Tensor): Initial hidden state [num_layers, B, hidden_size]
            c (torch.Tensor): Initial cell state [num_layers, B, hidden_size]

        Returns:
            torch.Tensor: Logits [B, seq_len, vocab_size]
            tuple: (h, c) final states
        """
        # Embed the captions
        # [B, seq_len, embed_dim]
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)

        # Pass through LSTM
        # output shape: [B, seq_len, hidden_size]
        output, (h_out, c_out) = self.lstm(embeddings, (h, c))

        # Classify
        # [B, seq_len, vocab_size]
        logits = self.linear(output)

        return logits, (h_out, c_out)


class ShowAndTell(nn.Module):
    """
    Combined Encoder-Decoder Model.
    """

    def __init__(self, vocab_size):
        super(ShowAndTell, self).__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(vocab_size)

    def forward(self, images, captions):
        """
        Forward pass for training.

        Args:
            images (torch.Tensor): [B, C, H, W]
            captions (torch.Tensor): [B, max_len] containing indices.
                                     Should include <SOS> and <EOS>.

        Returns:
            torch.Tensor: Logits for the sequence prediction.
        """
        # 1. Encode Image
        h0, c0 = self.encoder(images)

        # 2. Decode
        # During training, we feed the ground truth captions (Teacher Forcing).
        # We exclude the last token (<EOS> or <PAD>) from the input,
        # because we want to predict the next token given the current one.
        # The targets will be captions[:, 1:].
        decoder_input = captions[:, :-1]

        logits, _ = self.decoder(decoder_input, h0, c0)

        return logits
