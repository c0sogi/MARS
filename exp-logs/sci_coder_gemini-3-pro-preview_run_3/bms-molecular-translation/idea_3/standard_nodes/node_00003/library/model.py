import torch
import torch.nn as nn
import math
import numpy as np
from library.config import Config


class PatchEmbedding(nn.Module):
    """
    Projects raw image patches into embedding vectors.
    """

    def __init__(self, in_channels, d_model, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        # Using a convolution to extract and project patches is efficient
        # Input: (B, C, H, W) -> Output: (B, D, H/P, W/P)
        self.proj = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2)  # (B, D, N_Patches)
        x = x.transpose(1, 2)  # (B, N_Patches, D)
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Transformer Decoder-Only model for Image-to-Text generation.
    Treats image patches and text tokens as a single unified sequence.
    """

    def __init__(self, vocab_size):
        super().__init__()

        self.d_model = Config.D_MODEL
        self.num_patches = Config.NUM_PATCHES
        self.max_text_len = Config.MAX_TEXT_LEN
        # Total max sequence length: visual tokens + text tokens
        self.max_seq_len = self.num_patches + self.max_text_len

        # 1. Embeddings
        self.patch_embed = PatchEmbedding(
            in_channels=Config.IN_CHANNELS,
            d_model=self.d_model,
            patch_size=Config.PATCH_SIZE,
        )
        self.text_embed = nn.Embedding(vocab_size, self.d_model, padding_idx=0)

        # Learnable positional embedding for the entire sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))

        self.dropout = nn.Dropout(Config.DROPOUT)

        # 2. Transformer
        # We use TransformerEncoder because we process a single sequence with a custom mask.
        # This is functionally equivalent to a Decoder-only architecture when masked correctly.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=Config.N_HEADS,
            dim_feedforward=Config.D_FF,
            dropout=Config.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN is generally more stable for deep transformers
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.N_LAYERS
        )

        # 3. Output Head
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def generate_mask(self, text_len, device):
        """
        Generates the attention mask for the unified sequence.

        Structure of the mask (Total_Len x Total_Len):
        [ Visual-Visual (0) | Visual-Text (-inf) ]
        [ Text-Visual   (0) | Text-Text (Causal) ]

        Visual tokens can attend to all other visual tokens (Bidirectional).
        Text tokens can attend to all visual tokens and preceding text tokens (Causal).
        """
        total_len = self.num_patches + text_len
        mask = torch.full((total_len, total_len), float("-inf"), device=device)

        # 1. Visual-Visual: Bidirectional (0)
        mask[: self.num_patches, : self.num_patches] = 0

        # 2. Text-Visual: Bidirectional (0) - Text attends to all image patches
        mask[self.num_patches :, : self.num_patches] = 0

        # 3. Text-Text: Causal (Lower triangular 0, Upper -inf)
        # Create a causal mask for the text portion
        causal_mask = torch.triu(
            torch.full((text_len, text_len), float("-inf"), device=device), diagonal=1
        )
        mask[self.num_patches :, self.num_patches :] = causal_mask

        return mask

    def forward(self, images, text_input_ids):
        """
        Args:
            images: (B, C, H, W)
            text_input_ids: (B, L) - padded sequences of text tokens (usually starting with SOS)
        Returns:
            logits: (B, L, Vocab_Size) - predictions for the next token at each position
        """
        B, L = text_input_ids.shape
        device = images.device

        # 1. Get Embeddings
        visual_embeds = self.patch_embed(images)  # (B, N, D)
        text_embeds = self.text_embed(text_input_ids)  # (B, L, D)

        # 2. Concatenate
        x = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, N+L, D)

        # 3. Add Positional Embeddings
        # Slice pos_embed to current sequence length (in case L < max_text_len)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)

        # 4. Create Mask
        mask = self.generate_mask(L, device)

        # 5. Transformer Pass
        # src_key_padding_mask: Mask PAD tokens in text input.
        # Visual tokens are never padded.
        # text_input_ids == 0 assumes PAD_TOKEN index is 0.
        visual_padding = torch.zeros(
            (B, self.num_patches), dtype=torch.bool, device=device
        )
        text_padding = text_input_ids == 0  # (B, L)
        padding_mask = torch.cat([visual_padding, text_padding], dim=1)  # (B, N+L)

        output = self.transformer(x, mask=mask, src_key_padding_mask=padding_mask)

        # 6. Output Head
        # We only care about the output corresponding to the text part for next-token prediction.
        # output[:, self.num_patches:, :] corresponds to the processed text tokens.
        text_output = output[:, self.num_patches :, :]
        text_output = self.norm(text_output)
        logits = self.head(text_output)

        return logits

    @torch.no_grad()
    def generate(self, images, tokenizer, max_len=None):
        """
        Autoregressive generation for inference.

        Args:
            images: (B, C, H, W)
            tokenizer: Instance of library.tokenizer.Tokenizer
            max_len: Maximum length of generated text

        Returns:
            predictions: List of strings
        """
        if max_len is None:
            max_len = self.max_text_len

        self.eval()
        device = images.device
        B = images.shape[0]

        # 1. Encode Image
        visual_embeds = self.patch_embed(images)  # (B, N, D)

        # 2. Initialize Text Sequence with SOS
        sos_idx = tokenizer.stoi[Config.SOS_TOKEN]
        eos_idx = tokenizer.stoi[Config.EOS_TOKEN]

        # Current sequence of token indices (B, 1)
        generated = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)

        # Keep track of finished sequences
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            L = generated.shape[1]

            # Embed text
            text_embeds = self.text_embed(generated)

            # Concat
            x = torch.cat([visual_embeds, text_embeds], dim=1)

            # Positional Embedding
            seq_len = x.shape[1]
            x = x + self.pos_embed[:, :seq_len, :]

            # Mask
            mask = self.generate_mask(L, device)

            # Forward
            # Note: We recompute the whole sequence here.
            # Optimization (KV-caching) is possible but complex to implement with standard nn.TransformerEncoder.
            # Given the constraints, this is the robust baseline approach.
            output = self.transformer(x, mask=mask)

            # Get last token output
            last_token_out = output[:, -1, :]  # (B, D)
            last_token_out = self.norm(last_token_out)
            logits = self.head(last_token_out)  # (B, Vocab)

            # Greedy decode
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

            # Update generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check EOS
            is_eos = next_token.squeeze(-1) == eos_idx
            finished = finished | is_eos

            if finished.all():
                break

        # Decode to strings
        predictions = []
        generated = generated.cpu().numpy()
        for seq in generated:
            # sequence_to_text handles SOS/EOS removal
            pred = tokenizer.sequence_to_text(seq)
            predictions.append(pred)

        return predictions
