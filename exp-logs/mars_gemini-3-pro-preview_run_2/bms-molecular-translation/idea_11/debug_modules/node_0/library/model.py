import math
import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class PositionEmbedding2D(nn.Module):
    """
    Learnable 2D positional embedding for the feature map.
    Generates unique embeddings for each grid location (h, w) by summing
    learnable row and column embeddings.
    """

    def __init__(self, d_model, max_h=100, max_w=1000):
        super().__init__()
        # We split the dimension or sum them. Here we use two embeddings of size d_model/2
        # and concatenate them to form d_model, or use full d_model and sum.
        # Concatenation is often preferred to keep dimensions independent.
        self.h_dim = d_model // 2
        self.w_dim = d_model - self.h_dim

        self.h_embed = nn.Embedding(max_h, self.h_dim)
        self.w_embed = nn.Embedding(max_w, self.w_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Create grids
        y_pos = torch.arange(H, device=x.device)
        x_pos = torch.arange(W, device=x.device)

        # Look up embeddings
        # [H, h_dim]
        h_emb = self.h_embed(y_pos)
        # [W, w_dim]
        w_emb = self.w_embed(x_pos)

        # Broadcast to [H, W, dim]
        # h_emb: [H, 1, h_dim] -> [H, W, h_dim]
        # w_emb: [1, W, w_dim] -> [H, W, w_dim]
        h_emb_expanded = h_emb.unsqueeze(1).expand(H, W, -1)
        w_emb_expanded = w_emb.unsqueeze(0).expand(H, W, -1)

        # Concatenate: [H, W, C]
        pos_emb = torch.cat([h_emb_expanded, w_emb_expanded], dim=-1)

        # Permute to [1, C, H, W] for broadcasting addition to features
        pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0)

        return pos_emb


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for the 1D target sequence.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [Seq_len, Batch, Dim]
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class AnisotropicResNetTransformer(nn.Module):
    """
    Encoder-Decoder architecture with:
    1. Anisotropic ResNet-50 Encoder (preserving horizontal resolution).
    2. 2D Positional Embeddings.
    3. Transformer Decoder.
    """

    def __init__(self, vocab_size):
        super().__init__()

        # --- Encoder Setup ---
        # Load ResNet50
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            resnet = models.resnet50(weights=weights)
        except:
            resnet = models.resnet50(pretrained=True)

        # Modify input layer for 1-channel grayscale
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # --- Anisotropic Modifications ---
        # Standard ResNet downsamples by 2 in layer3 and layer4 (stride=2).
        # We change stride from (2, 2) to (2, 1) to preserve width resolution for dense text.

        # Layer 3 modification
        resnet.layer3[0].conv2.stride = (2, 1)
        resnet.layer3[0].downsample[0].stride = (2, 1)

        # Layer 4 modification
        resnet.layer4[0].conv2.stride = (2, 1)
        resnet.layer4[0].downsample[0].stride = (2, 1)

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.encoder_dim = 2048  # ResNet50 bottleneck output channels
        self.d_model = Config.DECODER_DIM

        # Projection layer to match transformer dimension
        self.projection = nn.Conv2d(self.encoder_dim, self.d_model, kernel_size=1)

        # 2D Positional Embedding
        # Max Height: 320px / 32 = 10. Set max_h=50 safe.
        # Max Width: ~3840px / 8 = 480. Set max_w=1000 safe.
        self.pos_emb_2d = PositionEmbedding2D(self.d_model, max_h=50, max_w=1000)

        # --- Decoder Setup ---
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoder_1d = PositionalEncoding(
            self.d_model, max_len=Config.MAX_TEXT_LEN + 50
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=Config.NUM_HEADS,
            dim_feedforward=Config.FF_DIM,
            dropout=Config.DROPOUT,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=Config.NUM_LAYERS
        )

        self.fc_out = nn.Linear(self.d_model, vocab_size)

    def forward(self, images, tgt_seqs=None):
        """
        Args:
            images: [Batch, 1, H, W]
            tgt_seqs: [Batch, Seq_Len] (optional, for training)

        Returns:
            If tgt_seqs is provided: Logits [Batch, Seq_Len, Vocab]
            If tgt_seqs is None: Memory features [Seq_Len_Src, Batch, Dim]
        """
        # 1. Encode
        features = self.encoder(images)  # [B, 2048, H', W']
        features = self.projection(features)  # [B, 512, H', W']

        # 2. Add 2D Positional Embeddings
        pos_2d = self.pos_emb_2d(features)  # [1, 512, H', W']
        features = features + pos_2d

        # 3. Flatten for Transformer
        # Transformer expects [Seq_Len, Batch, Dim]
        B, C, H, W = features.shape
        memory = features.flatten(2).permute(2, 0, 1)  # [H*W, B, C]

        # 4. Decode
        if tgt_seqs is not None:
            # Training mode with Teacher Forcing

            # Embed targets
            # tgt_seqs: [B, L]
            tgt_emb = self.embedding(tgt_seqs)  # [B, L, Dim]
            tgt_emb = tgt_emb.permute(1, 0, 2)  # [L, B, Dim]
            tgt_emb = self.pos_encoder_1d(tgt_emb)

            # Generate masks
            L = tgt_emb.size(0)
            tgt_mask = self.generate_square_subsequent_mask(L).to(images.device)

            # Padding mask: True where value is 0 (PAD_IDX)
            # Assuming PAD_IDX is 0 based on tokenizer
            tgt_padding_mask = tgt_seqs == 0

            output = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

            logits = self.fc_out(output)  # [L, B, Vocab]
            return logits.permute(1, 0, 2)  # [B, L, Vocab]

        else:
            # Inference mode: return memory for beam search
            return memory

    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
