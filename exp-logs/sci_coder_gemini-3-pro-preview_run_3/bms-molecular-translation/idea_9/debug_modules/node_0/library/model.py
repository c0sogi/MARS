import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from library.modules import (
    Mlp,
    MixerBlock,
    PatchEmbed,
    PositionalEncoding,
    TransformerDecoderLayer,
)
from library.utils import ATOM_VOCAB


class MixerTransformer(nn.Module):
    """
    Isotropic Mixer-Transformer with Stoichiometric Guidance.

    Architecture:
    - Encoder: MLP-Mixer (PatchEmbed -> Mixer Blocks)
    - Decoder: Transformer Decoder (Masked Self-Attn + Cross-Attn)
    - Aux Head: Linear layer for Atom Count prediction
    """

    def __init__(
        self,
        img_size=384,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        encoder_depth=8,
        token_dim=256,
        channel_dim=2048,
        decoder_depth=6,
        nhead=8,
        vocab_size=200,  # Placeholder, should be set based on tokenizer
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()

        # --- Encoder (MLP-Mixer) ---
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.encoder_blocks = nn.ModuleList(
            [
                MixerBlock(
                    dim=embed_dim,
                    num_patches=self.num_patches,
                    token_dim=token_dim,
                    channel_dim=channel_dim,
                    drop=dropout,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # --- Auxiliary Head (Stoichiometry) ---
        # Predicts counts for each atom type in ATOM_VOCAB
        self.aux_head = nn.Linear(embed_dim, len(ATOM_VOCAB))

        # --- Decoder (Transformer) ---
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=max_len)

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=embed_dim,
                    nhead=nhead,
                    dim_feedforward=2048,
                    dropout=dropout,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # Initialize patch embedding like nn.Linear (often helps convergence)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize linear layers
        torch.nn.init.xavier_uniform_(self.aux_head.weight)
        torch.nn.init.constant_(self.aux_head.bias, 0)
        torch.nn.init.xavier_uniform_(self.output_head.weight)
        torch.nn.init.constant_(self.output_head.bias, 0)

    def forward_encoder(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.encoder_norm(x)
        return x

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, image, text_input_ids=None, padding_mask=None):
        """
        Args:
            image: (B, C, H, W)
            text_input_ids: (B, Seq_Len) - Target tokens for teacher forcing.
                            If None, only encoder and aux outputs are computed.
            padding_mask: (B, Seq_Len) - Boolean mask where True indicates padding.

        Returns:
            logits: (B, Seq_Len, Vocab_Size) - Next token predictions
            aux_preds: (B, Num_Atoms) - Predicted atom counts
        """
        # 1. Encode
        memory = self.forward_encoder(image)  # (B, num_patches, embed_dim)

        # 2. Auxiliary Task
        # Global Average Pooling over patches
        pooled = memory.mean(dim=1)  # (B, embed_dim)
        aux_preds = self.aux_head(pooled)  # (B, num_atoms)

        logits = None
        if text_input_ids is not None:
            # 3. Decode
            # Embed and add position info
            tgt = self.embedding(text_input_ids)  # (B, Seq_Len, embed_dim)
            tgt = self.pos_encoder(tgt)

            # Create Causal Mask
            seq_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(seq_len, image.device)

            x = tgt
            for block in self.decoder_blocks:
                x = block(
                    tgt=x,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=padding_mask,
                )

            x = self.decoder_norm(x)
            logits = self.output_head(x)  # (B, Seq_Len, Vocab_Size)

        return logits, aux_preds

    @torch.no_grad()
    def generate(self, image, tokenizer, max_len=300, device="cuda"):
        """
        Greedy decoding for inference.
        """
        self.eval()
        bs = image.size(0)

        # Encode
        memory = self.forward_encoder(image)

        # Start with <SOS>
        sos_idx = tokenizer.stoi["<SOS>"]
        eos_idx = tokenizer.stoi["<EOS>"]

        # (B, 1)
        input_ids = torch.full((bs, 1), sos_idx, dtype=torch.long, device=device)

        # Store finished sequences
        finished = torch.zeros(bs, dtype=torch.bool, device=device)

        for _ in range(max_len):
            # Embed current sequence
            tgt = self.embedding(input_ids)
            tgt = self.pos_encoder(tgt)

            # Pass through decoder layers
            # Note: We re-process the whole sequence each step (standard transformer inference)
            # Caching K/V is an optimization not implemented here for simplicity
            x = tgt
            for block in self.decoder_blocks:
                x = block(tgt=x, memory=memory)

            x = self.decoder_norm(x)
            logits = self.output_head(x)  # (B, Seq_Len, Vocab)

            # Get last token prediction
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)  # (B,)

            # Update input_ids
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

            # Check for EOS
            is_eos = next_token == eos_idx
            finished = finished | is_eos

            if finished.all():
                break

        # Decode to strings
        preds = []
        for i in range(bs):
            seq = input_ids[i]
            text = tokenizer.decode(seq)
            preds.append(text)

        return preds
