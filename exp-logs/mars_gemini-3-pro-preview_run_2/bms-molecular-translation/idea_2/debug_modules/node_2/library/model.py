import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class Encoder(nn.Module):
    """
    MobileNetV2 Encoder with Global Average Pooling.
    Encodes an image into a fixed-size 1D feature vector.
    """

    def __init__(self, config: Config):
        super(Encoder, self).__init__()
        # Load pre-trained MobileNetV2
        # We assume weights are available or allowed to be downloaded.
        backbone = models.mobilenet_v2(pretrained=True)

        # We only need the feature extractor (CNN part)
        self.features = backbone.features

        # Output channels of MobileNetV2 features is 1280
        self.out_channels = 1280

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (Batch, 3, 224, 224)
        Returns:
            torch.Tensor: Encoded features (Batch, 1280)
        """
        # Extract features
        x = self.features(x)  # (Batch, 1280, 7, 7)

        # Global Average Pooling
        # Averages over spatial dimensions (H, W) -> (Batch, 1280)
        x = x.mean(dim=[2, 3])

        return x


class Decoder(nn.Module):
    """
    LSTM Decoder.
    Generates a sequence of characters from image features.
    """

    def __init__(self, config: Config, vocab_size: int):
        super(Decoder, self).__init__()
        self.config = config

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.decoder_hidden_dim,
            num_layers=config.decoder_layers,
            batch_first=True,
            dropout=config.dropout if config.decoder_layers > 1 else 0,
        )

        # Output layer (Logits for vocabulary)
        self.fc = nn.Linear(config.decoder_hidden_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, hidden):
        """
        Args:
            x (torch.Tensor): Input token indices (Batch, Seq_Len)
            hidden (tuple): Previous hidden and cell states (h, c)
        Returns:
            output (torch.Tensor): Logits (Batch, Seq_Len, Vocab_Size)
            hidden (tuple): Updated hidden and cell states
        """
        # Embed inputs
        x = self.embedding(x)  # (Batch, Seq_Len, Embed_Dim)
        x = self.dropout(x)

        # LSTM forward
        output, hidden = self.lstm(x, hidden)  # output: (Batch, Seq_Len, Hidden_Dim)

        # Project to vocabulary
        output = self.fc(output)  # (Batch, Seq_Len, Vocab_Size)

        return output, hidden


class Image2Seq(nn.Module):
    """
    Global Context Image-to-Sequence Network.
    Combines Encoder and Decoder for image captioning style prediction.
    """

    def __init__(self, config: Config, vocab_size: int):
        super(Image2Seq, self).__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.encoder = Encoder(config)
        self.decoder = Decoder(config, vocab_size)

        # Project encoder output to decoder hidden state size
        self.init_h = nn.Linear(self.encoder.out_channels, config.decoder_hidden_dim)
        self.init_c = nn.Linear(self.encoder.out_channels, config.decoder_hidden_dim)

    def forward(self, images, captions):
        """
        Training forward pass.

        Args:
            images (torch.Tensor): Input images (Batch, 3, H, W)
            captions (torch.Tensor): Target sequences (Batch, Max_Len) containing <SOS>...<EOS>

        Returns:
            torch.Tensor: Logits for the sequence (Batch, Max_Len-1, Vocab_Size)
        """
        # 1. Encode Image
        features = self.encoder(images)  # (Batch, 1280)

        # 2. Initialize LSTM states
        # Project features to hidden dim
        h0 = self.init_h(features)  # (Batch, Hidden_Dim)
        c0 = self.init_c(features)  # (Batch, Hidden_Dim)

        # Expand for number of layers if needed (assuming 1 layer for now based on config)
        # LSTM expects (Num_Layers, Batch, Hidden_Dim)
        h0 = h0.unsqueeze(0).repeat(self.config.decoder_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.config.decoder_layers, 1, 1)
        hidden = (h0, c0)

        # 3. Prepare Decoder Input
        # Teacher Forcing: Feed the ground truth sequence (excluding the last token <EOS>)
        # The model predicts the next token for each input.
        # Input: <SOS> A B C
        # Target: A B C <EOS>
        decoder_input = captions[:, :-1]

        # 4. Decode
        outputs, _ = self.decoder(decoder_input, hidden)

        return outputs

    def predict(self, images, tokenizer):
        """
        Inference forward pass using Greedy Decoding.

        Args:
            images (torch.Tensor): Input images (Batch, 3, H, W)
            tokenizer (Tokenizer): Tokenizer instance for special token IDs and decoding.

        Returns:
            list[str]: Predicted InChI strings.
        """
        self.eval()
        batch_size = images.size(0)
        device = images.device

        with torch.no_grad():
            # 1. Encode
            features = self.encoder(images)

            # 2. Init States
            h = (
                self.init_h(features)
                .unsqueeze(0)
                .repeat(self.config.decoder_layers, 1, 1)
            )
            c = (
                self.init_c(features)
                .unsqueeze(0)
                .repeat(self.config.decoder_layers, 1, 1)
            )
            hidden = (h, c)

            # 3. Start Token
            # Shape: (Batch, 1)
            current_input = torch.full(
                (batch_size, 1),
                tokenizer.sos_token_id,
                dtype=torch.long,
                device=device,
            )

            # Storage for predictions
            predictions = []

            # 4. Generation Loop
            for _ in range(self.config.max_length):
                # Pass current token through decoder
                output, hidden = self.decoder(current_input, hidden)
                # output: (Batch, 1, Vocab_Size)

                # Greedy selection: argmax
                predicted_indices = torch.argmax(output, dim=2)  # (Batch, 1)

                # Store prediction
                predictions.append(predicted_indices)

                # Update input for next step
                current_input = predicted_indices

            # 5. Convert to Text
            # Concatenate along sequence dimension: (Batch, Max_Len)
            predictions = torch.cat(predictions, dim=1)

            result_strings = []
            for i in range(batch_size):
                seq = predictions[i]
                text = tokenizer.sequence_to_text(seq)
                result_strings.append(text)

            return result_strings
