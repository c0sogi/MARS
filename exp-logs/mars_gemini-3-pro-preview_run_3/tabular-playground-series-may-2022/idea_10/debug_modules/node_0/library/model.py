import torch
import torch.nn as nn
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block adapted for 1D/Tabular data (Fully Connected layers).
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [Batch, Channel]
        y = self.fc(x)
        return x * y


class MultiGranularityNet(nn.Module):
    """
    Multi-Granularity Embedding Network with Squeeze-and-Excitation blocks.

    Architecture:
    1. Embeddings:
       - Unigrams (f_27 chars): High dim (16)
       - Bigrams (f_27 sliding window): Low dim (8)
       - Other Categorical (f_29, f_30): High dim (16)
    2. Concatenation: Continuous features + Flattened Embeddings
    3. Backbone: Funnel MLP with optional SE Blocks
    4. Head: Binary classification
    """

    def __init__(self, vocab_sizes, num_cont_features):
        """
        Args:
            vocab_sizes (list[int]): List of vocabulary sizes for each categorical feature.
                                     Order: Unigrams (10) -> Bigrams (9) -> Others (2).
            num_cont_features (int): Number of continuous input features.
        """
        super(MultiGranularityNet, self).__init__()

        # ==========================================
        # 1. Embedding Layers
        # ==========================================
        self.embeddings = nn.ModuleList()

        # We need to assign embedding dimensions based on the feature type.
        # According to data_processing.py and Config:
        # First F27_SEQ_LEN (10) are Unigrams
        # Next F27_BIGRAM_LEN (9) are Bigrams
        # Remaining (2) are f_29, f_30

        n_unigrams = Config.F27_SEQ_LEN
        n_bigrams = Config.F27_BIGRAM_LEN

        total_embed_dim = 0

        for i, vocab_size in enumerate(vocab_sizes):
            if i < n_unigrams:
                # Unigram
                dim = Config.UNIGRAM_EMBED_DIM
            elif i < n_unigrams + n_bigrams:
                # Bigram
                dim = Config.BIGRAM_EMBED_DIM
            else:
                # Other Categorical (f_29, f_30) treated same as unigrams/entities
                dim = Config.UNIGRAM_EMBED_DIM

            self.embeddings.append(nn.Embedding(vocab_size, dim))
            total_embed_dim += dim

        # ==========================================
        # 2. Backbone (Funnel MLP + SE)
        # ==========================================
        input_dim = num_cont_features + total_embed_dim

        layers = []
        in_features = input_dim

        for hidden_dim in Config.HIDDEN_LAYERS:
            # Linear
            layers.append(nn.Linear(in_features, hidden_dim))

            # Squeeze-and-Excitation
            if Config.USE_SE_BLOCK:
                # Ensure reduction doesn't make hidden dim 0
                reduction = 16
                if hidden_dim // reduction < 1:
                    reduction = 1
                layers.append(SEBlock(hidden_dim, reduction=reduction))

            # Activation
            layers.append(nn.ReLU(inplace=True))

            # Dropout
            layers.append(nn.Dropout(Config.DROPOUT))

            in_features = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # ==========================================
        # 3. Output Head
        # ==========================================
        self.head = nn.Linear(in_features, 1)

    def forward(self, x_cont, x_cat):
        """
        Args:
            x_cont: Continuous features [Batch, Num_Cont]
            x_cat: Categorical features [Batch, Num_Cat] (LongTensor)
        """
        # 1. Process Embeddings
        embed_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            # x_cat[:, i] is shape [Batch], output is [Batch, Embed_Dim]
            embed_outputs.append(emb_layer(x_cat[:, i]))

        # Concatenate all embeddings: [Batch, Total_Embed_Dim]
        x_emb = torch.cat(embed_outputs, dim=1)

        # 2. Concatenate with Continuous
        x = torch.cat([x_cont, x_emb], dim=1)

        # 3. Backbone
        x = self.backbone(x)

        # 4. Head
        logits = self.head(x)

        return logits
