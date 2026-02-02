import torch
import torch.nn as nn
import math
import timm
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for the Transformer.
    """

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        # x shape: (Seq_Len, Batch_Size, D_Model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class VisualEncoder(nn.Module):
    """
    EfficientNet-B0 backbone for feature extraction.
    Returns both spatial features (for Transformer) and global features (for Attributes).
    """

    def __init__(self, pretrained=True):
        super(VisualEncoder, self).__init__()
        self.backbone = timm.create_model(
            Config.ENCODER_NAME, pretrained=pretrained, features_only=True
        )

        # EfficientNet-B0 output channels at the last stage is 1280
        self.out_channels = self.backbone.feature_info.channels()[-1]

    def forward(self, x):
        # x: (B, 3, 256, 256)
        # features: list of feature maps. We take the last one.
        # Shape: (B, 1280, 8, 8) for 256x256 input
        features = self.backbone(x)[-1]

        # Global Average Pooling for Attribute Branch
        # Shape: (B, 1280)
        global_features = features.mean(dim=[2, 3])

        return features, global_features


class AttributeBranch(nn.Module):
    """
    MLP to predict normalized molecular attributes from global visual features.
    """

    def __init__(self, input_dim, hidden_dim=512):
        super(AttributeBranch, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, Config.NUM_ATTRIBUTES),
        )

    def forward(self, x):
        return self.net(x)


class AttributeContextualizedTransformer(nn.Module):
    """
    Main model architecture combining CNN, Attribute MLP, and Transformer Decoder.
    """

    def __init__(self):
        super(AttributeContextualizedTransformer, self).__init__()

        # 1. Visual Encoder
        self.encoder = VisualEncoder(pretrained=Config.ENCODER_PRETRAINED)

        # 2. Attribute Branch
        self.attribute_branch = AttributeBranch(input_dim=self.encoder.out_channels)

        # 3. Projections and Embeddings
        # Project visual features (1280 -> d_model)
        self.visual_projection = nn.Conv2d(
            self.encoder.out_channels, Config.D_MODEL, kernel_size=1
        )

        # Project predicted attributes (NUM_ATTRIBUTES -> d_model)
        self.attribute_projection = nn.Linear(Config.NUM_ATTRIBUTES, Config.D_MODEL)

        # Text Embedding
        self.embedding = nn.Embedding(
            Config.VOCAB_SIZE, Config.D_MODEL, padding_idx=Config.PAD_IDX
        )
        self.pos_encoder = PositionalEncoding(
            Config.D_MODEL, dropout=Config.DROPOUT, max_len=Config.MAX_LEN
        )

        # Learnable positional embedding for visual tokens (8x8 = 64 tokens)
        # We add 1 for the attribute summary token -> 65 tokens
        self.num_visual_tokens = (Config.IMAGE_SIZE // 32) ** 2  # 64
        self.visual_pos_embed = nn.Parameter(
            torch.randn(self.num_visual_tokens + 1, 1, Config.D_MODEL)
        )

        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=Config.NUM_DECODER_LAYERS
        )

        # 5. Prediction Head
        self.fc_out = nn.Linear(Config.D_MODEL, Config.VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self):
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Visual positional embeddings initialization
        nn.init.normal_(self.visual_pos_embed, mean=0, std=0.1)

    def encode_image_context(self, images):
        """
        Runs the encoder and attribute branch, fuses features into memory.

        Returns:
            memory: (Source_Seq_Len, Batch, D_Model)
            predicted_attrs: (Batch, Num_Attributes)
        """
        batch_size = images.size(0)

        # 1. Extract Features
        # spatial_feats: (B, 1280, 8, 8)
        # global_feats: (B, 1280)
        spatial_feats, global_feats = self.encoder(images)

        # 2. Predict Attributes
        predicted_attrs = self.attribute_branch(global_feats)

        # 3. Process Visual Features
        # Project: (B, 1280, 8, 8) -> (B, 256, 8, 8)
        visual_tokens = self.visual_projection(spatial_feats)
        # Flatten: (B, 256, 64) -> Permute to (B, 64, 256)
        visual_tokens = visual_tokens.flatten(2).permute(0, 2, 1)

        # 4. Process Attribute Token
        # Project: (B, Num_Attr) -> (B, 256) -> Unsqueeze (B, 1, 256)
        attr_token = self.attribute_projection(predicted_attrs).unsqueeze(1)

        # 5. Fuse Context (Memory)
        # Prepend attribute token to visual tokens
        # Shape: (B, 65, 256)
        context = torch.cat([attr_token, visual_tokens], dim=1)

        # Permute for Transformer: (Seq_Len, Batch, D_Model) -> (65, B, 256)
        context = context.permute(1, 0, 2)

        # Add visual positional embeddings
        context = context + self.visual_pos_embed

        return context, predicted_attrs

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, images, target_seq):
        """
        Forward pass for training.

        Args:
            images: (B, 3, H, W)
            target_seq: (B, Seq_Len) - Input sequence (indices)

        Returns:
            logits: (B, Seq_Len, Vocab_Size)
            predicted_attrs: (B, Num_Attributes)
        """
        # 1. Encode
        # memory: (S, B, E)
        memory, predicted_attrs = self.encode_image_context(images)

        # 2. Prepare Target
        # Permute target for Transformer: (T, B)
        tgt = target_seq.permute(1, 0)

        # Create masks
        # Causal mask for self-attention
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

        # Padding mask (True where padding exists)
        tgt_padding_mask = target_seq == Config.PAD_IDX

        # 3. Decode
        # Embed and add position info
        tgt_emb = self.embedding(tgt) * math.sqrt(Config.D_MODEL)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Transformer Decoder forward pass
        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        # 4. Project to Vocab
        # output: (T, B, E) -> (B, T, E)
        output = output.permute(1, 0, 2)
        logits = self.fc_out(output)

        return logits, predicted_attrs

    def predict(self, images, max_len=None):
        """
        Inference using greedy decoding.

        Args:
            images: (B, 3, H, W)
            max_len: int, max sequence length

        Returns:
            predictions: (B, Seq_Len) - Indices
        """
        if max_len is None:
            max_len = Config.MAX_LEN

        batch_size = images.size(0)
        device = images.device

        # 1. Encode
        memory, _ = self.encode_image_context(images)

        # 2. Initialize Decoding
        # Start with SOS token
        # Shape: (1, B)
        input_seq = torch.full(
            (1, batch_size), Config.SOS_IDX, dtype=torch.long, device=device
        )

        # Store finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 3. Autoregressive Loop
        for _ in range(max_len):
            # Embed input
            tgt_emb = self.embedding(input_seq) * math.sqrt(Config.D_MODEL)
            tgt_emb = self.pos_encoder(tgt_emb)

            # Pass through decoder
            # We don't need a causal mask here because we are only looking at past tokens effectively
            # by passing the full generated sequence so far.
            # However, PyTorch decoder processes the whole sequence.
            # For efficiency in a loop, one might cache, but standard implementation re-processes.
            output = self.decoder(tgt=tgt_emb, memory=memory)

            # Get output for the last token
            # output: (Seq_Len, B, E) -> Take last step: (B, E)
            last_output = output[-1, :, :]

            # Project to vocab
            logits = self.fc_out(last_output)

            # Greedy choice: Argmax
            next_token = torch.argmax(logits, dim=-1)  # (B,)

            # Update input sequence
            # (Seq_Len + 1, B)
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=0)

            # Check for EOS
            is_eos = next_token == Config.EOS_IDX
            finished = finished | is_eos

            if finished.all():
                break

        # Return sequence (B, T)
        return input_seq.permute(1, 0)
