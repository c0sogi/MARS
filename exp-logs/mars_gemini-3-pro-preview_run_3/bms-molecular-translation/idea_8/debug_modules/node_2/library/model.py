import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from library.config import Config
from library.model_components import PatchEmbed, MixerBlock, Attention


class MlpMixerEncoder(nn.Module):
    """
    Isotropic MLP-Mixer Encoder.
    Processes the image as a sequence of patches using dense layers for mixing
    spatial and channel information.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.embed_dim = config.encoder_dim

        # Patch Embedding Layer
        self.patch_embed = PatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.in_channels,
            embed_dim=self.embed_dim,
        )

        num_patches = self.patch_embed.n_patches

        # Stack of Mixer Blocks
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    num_tokens=num_patches,
                    dim=self.embed_dim,
                    token_mixing_dim=config.token_mixing_dim,
                    channel_mixing_dim=config.channel_mixing_dim,
                    dropout=0.0,
                )
                for _ in range(config.encoder_depth)
            ]
        )

        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)  # (B, N, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x


class GruDecoder(nn.Module):
    """
    GRU-based Decoder with Dot-Product Attention.
    Generates the InChI string character by character.
    """

    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.hidden_dim = config.decoder_dim
        self.embed_dim = config.embedding_dim
        self.vocab_size = vocab_size

        # Embedding for text tokens
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(config.decoder_dropout)

        # Attention Mechanism
        self.attention = Attention(
            enc_dim=config.encoder_dim,
            dec_dim=self.hidden_dim,
            attention_dim=config.attention_dim,
        )

        # GRU Cell
        # Input to GRU is concatenation of embedding and context vector
        input_dim = self.embed_dim + config.encoder_dim
        self.gru_cell = nn.GRUCell(input_dim, self.hidden_dim)

        # Classifier
        self.fc_out = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        """
        Single step forward pass for the decoder.

        Args:
            input_token: (B,) tensor of indices for the current step
            hidden: (B, dec_dim) previous hidden state
            encoder_outputs: (B, N, enc_dim)

        Returns:
            output: (B, vocab_size) logits
            hidden: (B, dec_dim) updated hidden state
            attn_weights: (B, N, 1) attention weights
        """
        # Embed input token: (B, embed_dim)
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)

        # Calculate attention context: (B, enc_dim)
        context, attn_weights = self.attention(hidden, encoder_outputs)

        # Concatenate embedding and context: (B, embed_dim + enc_dim)
        gru_input = torch.cat([embedded, context], dim=1)

        # Update hidden state
        hidden = self.gru_cell(gru_input, hidden)

        # Predict output logits
        output = self.fc_out(hidden)

        return output, hidden, attn_weights


class InChIModel(nn.Module):
    """
    End-to-End Image-to-InChI Model.
    Wraps the MlpMixerEncoder and GruDecoder.
    """

    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.encoder = MlpMixerEncoder(config)
        self.decoder = GruDecoder(config, vocab_size)
        self.device = config.device

    def forward(self, images, text_seqs, teacher_forcing_ratio=None):
        """
        Forward pass for training.

        Args:
            images: (B, C, H, W)
            text_seqs: (B, max_len) Ground truth sequences (indices)
            teacher_forcing_ratio: float, probability of using ground truth input

        Returns:
            outputs: (B, max_len, vocab_size) Logits
        """
        batch_size = images.size(0)
        max_len = text_seqs.size(1)
        vocab_size = self.decoder.vocab_size

        # Use config ratio if not provided
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.config.teacher_forcing_ratio

        # 1. Encode Image
        encoder_outputs = self.encoder(images)  # (B, N, enc_dim)

        # 2. Initialize Decoder Hidden State
        # We average encoder outputs to get an initial global context
        decoder_hidden = encoder_outputs.mean(
            dim=1
        )  # (B, enc_dim) -> (B, dec_dim) assuming dims match

        # Prepare outputs tensor
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)

        # First input is <sos> token
        input_token = text_seqs[:, 0]

        # Loop through the sequence
        # Note: We predict from t=1 to max_len.
        # text_seqs[:, t] is the target for the step taking input text_seqs[:, t-1] (or prediction)
        for t in range(1, max_len):
            output_logits, decoder_hidden, _ = self.decoder(
                input_token, decoder_hidden, encoder_outputs
            )

            outputs[:, t, :] = output_logits

            # Teacher Forcing logic
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                input_token = text_seqs[:, t]
            else:
                input_token = output_logits.argmax(1)

        return outputs

    def generate(self, images, max_len=None, sos_token_idx=1, eos_token_idx=2):
        """
        Inference / Generation method (Greedy Decoding).

        Args:
            images: (B, C, H, W)
            max_len: Maximum sequence length to generate
            sos_token_idx: Index of <sos> token
            eos_token_idx: Index of <eos> token

        Returns:
            predictions: (B, L) List of predicted indices
        """
        if max_len is None:
            max_len = self.config.max_text_length

        batch_size = images.size(0)

        with torch.no_grad():
            # 1. Encode
            encoder_outputs = self.encoder(images)
            decoder_hidden = encoder_outputs.mean(dim=1)

            # 2. Prepare decoding
            input_token = torch.full((batch_size,), sos_token_idx, dtype=torch.long).to(
                self.device
            )
            predictions = []

            # Track finished sequences to stop early if needed (batch-wise optimization omitted for simplicity)
            # We will generate fixed length or until max_len and handle EOS in post-processing

            for _ in range(max_len):
                output_logits, decoder_hidden, _ = self.decoder(
                    input_token, decoder_hidden, encoder_outputs
                )

                # Greedy selection
                top1 = output_logits.argmax(1)  # (B,)
                predictions.append(top1)

                input_token = top1

            # Stack predictions: (L, B) -> (B, L)
            predictions = torch.stack(predictions, dim=1)

        return predictions
