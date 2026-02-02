import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A Residual Block with Dilated 1D Convolutions.
    Maintains sequence length via padding = dilation (for kernel_size=3).
    """

    def __init__(self, channels, dilation, kernel_size=3, dropout=0.1):
        super(DilatedResidualBlock, self).__init__()

        # For kernel_size=3, padding=dilation ensures input length == output length
        padding = dilation

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class RNADilatedNet(nn.Module):
    """
    Hybrid Dilated ResNet-BiGRU Architecture for RNA Degradation Prediction.

    Structure:
    1. Multi-channel Embeddings (Seq, Struct, Loop)
    2. Dilated ResNet Encoder (Exponentially increasing receptive field)
    3. Bidirectional GRU (Global context)
    4. Linear Output Head
    """

    def __init__(self, config=Config):
        super(RNADilatedNet, self).__init__()

        # 1. Embeddings
        self.embed_seq = nn.Embedding(config.VOCAB_SIZE_SEQ, config.EMBED_DIM)
        self.embed_struct = nn.Embedding(config.VOCAB_SIZE_STRUCT, config.EMBED_DIM)
        self.embed_loop = nn.Embedding(config.VOCAB_SIZE_LOOP, config.EMBED_DIM)

        # Total embedding dimension after concatenation
        total_embed_dim = config.EMBED_DIM * 3

        # 2. Adapter / Projection
        # Projects concatenated embeddings to the ResNet filter size
        self.adapter = nn.Conv1d(
            in_channels=total_embed_dim,
            out_channels=config.RESNET_FILTERS,
            kernel_size=1,
        )

        # 3. Dilated ResNet Encoder
        layers = []
        for dilation in config.DILATION_RATES:
            layers.append(
                DilatedResidualBlock(
                    channels=config.RESNET_FILTERS,
                    dilation=dilation,
                    kernel_size=config.RESNET_KERNEL_SIZE,
                    dropout=config.RESNET_DROPOUT,
                )
            )
        self.resnet_encoder = nn.Sequential(*layers)

        # 4. Bidirectional GRU
        self.gru = nn.GRU(
            input_size=config.RESNET_FILTERS,
            hidden_size=config.GRU_HIDDEN_SIZE,
            num_layers=config.GRU_LAYERS,
            dropout=config.GRU_DROPOUT if config.GRU_LAYERS > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # 5. Output Head
        # Input dim is 2 * hidden_size because of bidirectionality
        self.head = nn.Linear(config.GRU_HIDDEN_SIZE * 2, config.NUM_TARGETS)

    def forward(self, seq, struct, loop):
        """
        Args:
            seq: (Batch, SeqLen) - LongTensor
            struct: (Batch, SeqLen) - LongTensor
            loop: (Batch, SeqLen) - LongTensor

        Returns:
            logits: (Batch, SeqLen, 5)
        """
        # 1. Embed and Concatenate
        # Shape: (Batch, SeqLen, EmbedDim)
        emb_s = self.embed_seq(seq)
        emb_st = self.embed_struct(struct)
        emb_l = self.embed_loop(loop)

        # Concatenate along feature dimension
        # Shape: (Batch, SeqLen, 3 * EmbedDim)
        x = torch.cat([emb_s, emb_st, emb_l], dim=2)

        # 2. ResNet Encoder
        # Conv1d expects (Batch, Channels, SeqLen)
        x = x.permute(0, 2, 1)

        # Project and Encode
        x = self.adapter(x)
        x = self.resnet_encoder(x)

        # 3. BiGRU
        # GRU expects (Batch, SeqLen, Channels)
        x = x.permute(0, 2, 1)

        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        # 4. Output Head
        # x shape: (Batch, SeqLen, 2 * HiddenSize)
        logits = self.head(x)

        return logits
