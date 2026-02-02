import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class Encoder(nn.Module):
    """
    EfficientNet-B0 Encoder extracting spatial features.
    """

    def __init__(self):
        super().__init__()
        # Load pre-trained EfficientNet-B0
        # We use the backbone to extract features.
        # EfficientNet-B0 output channels at the final conv layer is 1280.
        self.cnn = timm.create_model(Config.ENCODER_NAME, pretrained=True)

    def forward(self, images):
        """
        Args:
            images (torch.Tensor): Input images of shape (Batch, 3, H, W)
        Returns:
            features (torch.Tensor): Spatial features of shape (Batch, Seq_Len, Encoder_Dim)
        """
        # Extract features: (Batch, 1280, H/32, W/32)
        # For 256x256 input, output is (Batch, 1280, 8, 8)
        features = self.cnn.forward_features(images)

        batch_size, channels, height, width = features.size()

        # Permute to (Batch, H, W, Channels) -> (Batch, H*W, Channels)
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch_size, height * width, channels)

        return features


class Attention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Args:
            encoder_out (torch.Tensor): Encoder output (Batch, Num_Pixels, Encoder_Dim)
            decoder_hidden (torch.Tensor): Decoder hidden state (Batch, Decoder_Dim)
        Returns:
            context (torch.Tensor): Context vector (Batch, Encoder_Dim)
            alpha (torch.Tensor): Attention weights (Batch, Num_Pixels)
        """
        # Calculate attention scores
        att1 = self.encoder_att(encoder_out)  # (Batch, Num_Pixels, Att_Dim)
        att2 = self.decoder_att(decoder_hidden)  # (Batch, Att_Dim)

        # Broadcast add: (Batch, Num_Pixels, Att_Dim) + (Batch, 1, Att_Dim)
        att = self.full_att(
            self.relu(att1 + att2.unsqueeze(1))
        )  # (Batch, Num_Pixels, 1)

        # Softmax over pixels
        alpha = self.softmax(att)  # (Batch, Num_Pixels, 1)

        # Weighted sum
        context = (encoder_out * alpha).sum(dim=1)  # (Batch, Encoder_Dim)

        return context, alpha.squeeze(2)


class Decoder(nn.Module):
    """
    GRU Decoder with Attention.
    """

    def __init__(
        self,
        vocab_size,
        encoder_dim,
        decoder_dim,
        embed_dim,
        attention_dim,
        dropout=0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # GRU Input: Concatenation of Embedding and Context Vector
        self.gru = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)

        # Initialize hidden state from encoder features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)

        # Output layer
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        """
        Initializes the hidden state as the tanh of the linear projection of the mean encoder features.
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        hidden = self.init_h(mean_encoder_out)
        return torch.tanh(hidden)

    def forward(self, input_token, decoder_hidden, encoder_out):
        """
        One step of the decoder.
        Args:
            input_token (torch.Tensor): Previous token indices (Batch,)
            decoder_hidden (torch.Tensor): Previous hidden state (Batch, Decoder_Dim)
            encoder_out (torch.Tensor): Encoder features (Batch, Num_Pixels, Encoder_Dim)
        Returns:
            preds (torch.Tensor): Logits for next token (Batch, Vocab_Size)
            new_hidden (torch.Tensor): Updated hidden state (Batch, Decoder_Dim)
            alpha (torch.Tensor): Attention weights
        """
        # Embedding
        embedded = self.embedding(input_token)  # (Batch, Embed_Dim)
        embedded = self.dropout(embedded)

        # Calculate Attention and Context
        context, alpha = self.attention(encoder_out, decoder_hidden)

        # GRU Step
        gru_input = torch.cat([embedded, context], dim=1)
        new_hidden = self.gru(gru_input, decoder_hidden)

        # Prediction
        preds = self.fc(new_hidden)

        return preds, new_hidden, alpha


class Seq2Seq(nn.Module):
    """
    Main model class connecting Encoder and Decoder.
    """

    def __init__(self, tokenizer_len):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(
            vocab_size=tokenizer_len,
            encoder_dim=Config.ENCODER_DIM,
            decoder_dim=Config.DECODER_DIM,
            embed_dim=Config.EMBED_DIM,
            attention_dim=Config.ATTENTION_DIM,
            dropout=Config.DROPOUT,
        )
        self.vocab_size = tokenizer_len
        self.device = Config.DEVICE
        self.max_len = Config.MAX_TEXT_LEN

    def forward(self, images, targets=None):
        """
        Forward pass for training.
        Args:
            images (torch.Tensor): (Batch, 3, H, W)
            targets (torch.Tensor, optional): (Batch, Max_Len) Ground truth sequences.
        Returns:
            outputs (torch.Tensor): (Batch, Seq_Len, Vocab_Size)
        """
        batch_size = images.size(0)

        # Encode
        encoder_out = self.encoder(images)

        # Init Decoder
        decoder_hidden = self.decoder.init_hidden_state(encoder_out)

        # Determine sequence length
        seq_len = targets.size(1) if targets is not None else self.max_len

        # Tensor to store outputs
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(self.device)

        # First input: <SOS> token (assumed to be at index 0 of targets)
        # If targets is None (unlikely in forward training), we'd need SOS index.
        # But forward is mostly for training where targets exist.
        decoder_input = targets[:, 0]

        # Loop through sequence
        for t in range(1, seq_len):
            predictions, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_out
            )
            outputs[:, t, :] = predictions

            # Teacher Forcing
            if (
                targets is not None
                and torch.rand(1).item() < Config.TEACHER_FORCING_RATIO
            ):
                decoder_input = targets[:, t]
            else:
                decoder_input = predictions.argmax(1)

        return outputs

    def predict(self, images, tokenizer):
        """
        Inference using Greedy Decoding.
        Args:
            images (torch.Tensor): (Batch, 3, H, W)
            tokenizer (Tokenizer): For SOS/EOS indices and decoding.
        Returns:
            decoded_strings (list[str]): List of predicted InChI strings.
        """
        self.eval()
        batch_size = images.size(0)

        with torch.no_grad():
            encoder_out = self.encoder(images)
            decoder_hidden = self.decoder.init_hidden_state(encoder_out)

            sos_idx = tokenizer.char_to_idx[Config.SOS_TOKEN]
            eos_idx = tokenizer.char_to_idx[Config.EOS_TOKEN]

            # Start with SOS
            decoder_input = torch.full(
                (batch_size,), sos_idx, dtype=torch.long, device=self.device
            )

            predictions = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            for t in range(self.max_len):
                preds_logits, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_out
                )

                # Greedy selection
                top1 = preds_logits.argmax(1)

                predictions.append(top1)
                decoder_input = top1

                # Check for EOS
                is_eos = top1 == eos_idx
                finished = finished | is_eos

                if finished.all():
                    break

            # Stack predictions: (Batch, Seq_Len)
            predictions = torch.stack(predictions, dim=1)

            # Decode sequences to text
            decoded_strings = []
            for i in range(batch_size):
                seq = predictions[i]
                text = tokenizer.sequence_to_text(seq)
                decoded_strings.append(text)

            return decoded_strings
