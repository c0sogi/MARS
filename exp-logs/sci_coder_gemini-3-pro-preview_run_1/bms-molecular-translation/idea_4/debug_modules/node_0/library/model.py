import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class Encoder(nn.Module):
    """
    MobileNetV3-Small backbone for visual feature extraction.
    """

    def __init__(self):
        super(Encoder, self).__init__()
        # Load pre-trained MobileNetV3 Small
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = models.mobilenet_v3_small(weights=weights)

        # Extract feature extractor and pooling layer
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Encoder output dimension is fixed for MobileNetV3-Small (576)
        self.out_dim = Config.ENCODER_OUT_DIM

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (Batch, 3, 256, 256)
        Returns:
            torch.Tensor: Global visual features (Batch, 576)
        """
        x = self.features(x)  # (B, 576, H/32, W/32)
        x = self.avgpool(x)  # (B, 576, 1, 1)
        x = torch.flatten(x, 1)  # (B, 576)
        return x


class AttributeHead(nn.Module):
    """
    MLP Head for predicting chemical attributes (atom counts + length).
    """

    def __init__(self, input_dim, output_dim):
        super(AttributeHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Visual features (Batch, Input_Dim)
        Returns:
            torch.Tensor: Predicted attributes (Batch, Output_Dim)
        """
        return self.net(x)


class Decoder(nn.Module):
    """
    GRU Decoder for sequence generation.
    Hidden state is initialized via projection of visual features + attributes.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim, attr_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=Config.PAD_IDX)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Projection layer to map concatenated (Visual + Attributes) to GRU Hidden State
        self.init_proj = nn.Linear(visual_dim + attr_dim, hidden_dim)
        self.tanh = nn.Tanh()

    def init_hidden(self, visual_features, attributes):
        """
        Initializes the hidden state.
        """
        # Concatenate visual features and predicted attributes
        combined = torch.cat([visual_features, attributes], dim=1)

        # Project to hidden dimension
        hidden = self.init_proj(combined)
        hidden = self.tanh(hidden)

        # Reshape for GRU (Num_Layers=1, Batch, Hidden)
        return hidden.unsqueeze(0)

    def forward(self, x, hidden):
        """
        Args:
            x (torch.Tensor): Input token indices (Batch, Seq_Len)
            hidden (torch.Tensor): Previous hidden state (1, Batch, Hidden)
        Returns:
            torch.Tensor: Logits (Batch, Seq_Len, Vocab)
            torch.Tensor: New hidden state
        """
        x = self.embedding(x)  # (B, Seq, Embed)
        output, hidden = self.gru(x, hidden)  # (B, Seq, Hidden)
        logits = self.fc_out(output)  # (B, Seq, Vocab)
        return logits, hidden


class AttributeConditionedModel(nn.Module):
    """
    Multi-Task Model:
    1. Encodes image.
    2. Predicts attributes (Regression).
    3. Conditions Decoder on Visual + Attributes to predict InChI (Generation).
    """

    def __init__(self):
        super(AttributeConditionedModel, self).__init__()

        self.encoder = Encoder()

        self.attribute_head = AttributeHead(
            input_dim=Config.ENCODER_OUT_DIM, output_dim=Config.ATTRIBUTE_DIM
        )

        self.decoder = Decoder(
            vocab_size=Config.VOCAB_SIZE,
            embed_dim=Config.EMBED_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            visual_dim=Config.ENCODER_OUT_DIM,
            attr_dim=Config.ATTRIBUTE_DIM,
        )

        self.max_len = Config.MAX_SEQ_LEN

    def forward(self, images, target_seqs=None):
        """
        Forward pass.

        Args:
            images (torch.Tensor): Input images.
            target_seqs (torch.Tensor, optional): Ground truth sequences for Teacher Forcing.
                                                  Expected shape (Batch, Max_Len).

        Returns:
            seq_logits (torch.Tensor): Logits for sequence generation.
            pred_attributes (torch.Tensor): Predictions for attribute regression.
        """
        # 1. Encode
        visual_features = self.encoder(images)

        # 2. Predict Attributes
        pred_attributes = self.attribute_head(visual_features)

        # 3. Initialize Decoder with Visual + Predicted Attributes
        hidden = self.decoder.init_hidden(visual_features, pred_attributes)

        # 4. Decode
        if target_seqs is not None:
            # Training mode: Teacher Forcing
            # Input to decoder is target_seqs excluding the last token (EOS)
            # Target for loss will be target_seqs excluding the first token (SOS)
            decoder_input = target_seqs[:, :-1]
            seq_logits, _ = self.decoder(decoder_input, hidden)
            return seq_logits, pred_attributes
        else:
            # Without targets, we cannot perform teacher forcing in this forward signature.
            # Use predict() for inference.
            return None, pred_attributes

    def predict(self, images, device=None):
        """
        Inference method using Greedy Decoding.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        batch_size = images.size(0)

        with torch.no_grad():
            # Encode and Predict Attributes
            visual_features = self.encoder(images)
            pred_attributes = self.attribute_head(visual_features)

            # Init Hidden
            hidden = self.decoder.init_hidden(visual_features, pred_attributes)

            # Start Token
            current_input = torch.full(
                (batch_size, 1), Config.SOS_IDX, dtype=torch.long, device=device
            )

            predictions = []

            # Generation Loop
            for _ in range(self.max_len):
                logits, hidden = self.decoder(current_input, hidden)

                # Greedy selection
                probs = torch.softmax(logits, dim=-1)
                predicted_token = torch.argmax(probs, dim=-1)  # (B, 1)

                predictions.append(predicted_token)

                # Next input is current prediction
                current_input = predicted_token

            # Stack predictions
            output_seqs = torch.cat(predictions, dim=1)  # (B, Max_Len)

        return output_seqs
