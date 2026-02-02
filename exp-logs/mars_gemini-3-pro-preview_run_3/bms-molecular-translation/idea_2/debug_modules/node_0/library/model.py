import torch
import torch.nn as nn
import math
import timm
import numpy as np
from library.config import Config


class PositionalEncoding1D(nn.Module):
    """
    Standard sinusoidal 1D positional encoding for the Transformer Decoder.
    """

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encodings added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ResNetEncoder(nn.Module):
    """
    ResNet-18 Encoder that outputs a flattened sequence of spatial features
    with 2D positional encodings.
    """

    def __init__(self, model_name=Config.ENCODER_NAME, pretrained=True):
        super().__init__()
        # Load ResNet backbone, removing classification head and pooling
        # Output shape for resnet18 with 384x384 input is usually (B, 512, 12, 12)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get the number of channels in the last feature map
        # We can run a dummy forward pass or inspect feature_info
        dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        self.feature_channels = features.shape[1]
        self.feature_h = features.shape[2]
        self.feature_w = features.shape[3]

        # 1x1 Convolution to project to DECODER_DIM
        self.conv_project = nn.Conv2d(
            self.feature_channels, Config.DECODER_DIM, kernel_size=1
        )

        # Learnable 2D Positional Encodings
        # Shape: (1, DECODER_DIM, H, W)
        self.pos_embed = nn.Parameter(
            torch.randn(1, Config.DECODER_DIM, self.feature_h, self.feature_w)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: Images (B, 3, H, W)
        Returns:
            features: Flattened features (B, Seq_Len_Img, Decoder_Dim)
        """
        # Extract features (B, 512, 12, 12)
        x = self.backbone(x)

        # Project dimensions (B, 256, 12, 12)
        x = self.conv_project(x)

        # Add 2D positional embedding
        x = x + self.pos_embed

        # Flatten spatial dimensions: (B, 256, 12, 12) -> (B, 256, 144)
        x = x.flatten(2)

        # Permute to (B, Seq_Len, Dim) -> (B, 144, 256)
        x = x.transpose(1, 2)

        return x


class TransformerDecoder(nn.Module):
    """
    Vanilla Transformer Decoder.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, Config.DECODER_DIM)
        self.pos_encoder = PositionalEncoding1D(
            Config.DECODER_DIM, max_len=Config.MAX_LEN, dropout=Config.DROPOUT
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=Config.DECODER_DIM,
            nhead=Config.NUM_HEADS,
            dim_feedforward=Config.FF_DIM,
            dropout=Config.DROPOUT,
            batch_first=True,  # Inputs are (Batch, Seq, Dim)
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=Config.NUM_LAYERS
        )

        self.output_head = nn.Linear(Config.DECODER_DIM, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """Generates a causal mask to prevent attending to future tokens."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, tgt, memory, tgt_key_padding_mask=None):
        """
        Args:
            tgt: Target sequence indices (B, Seq_Len)
            memory: Encoder features (B, Img_Seq_Len, Dim)
            tgt_key_padding_mask: Mask for padding tokens in tgt (B, Seq_Len)
        Returns:
            logits: (B, Seq_Len, Vocab_Size)
        """
        # Embed and add position info
        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Create causal mask
        seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)

        # Pass through Decoder
        # memory is (B, Img_Seq, Dim)
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Project to vocabulary
        logits = self.output_head(output)
        return logits


class InChiModel(nn.Module):
    """
    End-to-End Model: ResNet Encoder + Transformer Decoder.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = TransformerDecoder(vocab_size)

    def forward(self, images, text, pad_token_id=None):
        """
        Training forward pass.

        Args:
            images: Input images (B, 3, H, W)
            text: Target text sequences (B, Max_Len)
            pad_token_id: ID of the padding token to create mask.

        Returns:
            logits: (B, Seq_Len, Vocab_Size)
        """
        # 1. Encode Image
        # (B, Img_Seq_Len, Dim)
        memory = self.encoder(images)

        # 2. Create Padding Mask for Text
        # True where value is pad_token_id
        tgt_key_padding_mask = None
        if pad_token_id is not None:
            tgt_key_padding_mask = text == pad_token_id

        # 3. Decode
        logits = self.decoder(text, memory, tgt_key_padding_mask=tgt_key_padding_mask)

        return logits

    @torch.no_grad()
    def predict(self, images, tokenizer, max_len=Config.MAX_LEN):
        """
        Inference using greedy decoding.

        Args:
            images: Input images (B, 3, H, W)
            tokenizer: Tokenizer instance for special token IDs.
            max_len: Maximum generation length.

        Returns:
            predictions: List of predicted token sequences (B, Generated_Len)
        """
        self.eval()
        device = images.device
        batch_size = images.size(0)

        # 1. Encode
        memory = self.encoder(images)

        # 2. Initialize Decoder Input with SOS
        # Shape: (B, 1)
        start_token = tokenizer.sos_token_id
        end_token = tokenizer.eos_token_id

        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # Keep track of finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 3. Greedy Loop
        for _ in range(max_len):
            # Forward pass through decoder
            # We pass the current partial sequence 'tgt'
            # Note: Optimally, we would cache KV pairs, but for baseline, re-forwarding is acceptable
            logits = self.decoder(tgt, memory)

            # Get the logits for the last token
            last_token_logits = logits[:, -1, :]

            # Greedy choice
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(1)  # (B, 1)

            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Check for EOS
            just_finished = next_token.squeeze() == end_token
            finished = finished | just_finished

            if finished.all():
                break

        return tgt
