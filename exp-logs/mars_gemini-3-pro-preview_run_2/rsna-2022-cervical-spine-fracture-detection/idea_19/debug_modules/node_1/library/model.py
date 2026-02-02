import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class MultiLevelFeatureExtractor(nn.Module):
    """
    Extracts and aggregates features from P4 and P5 blocks of EfficientNet-B4.
    """

    def __init__(self, backbone_name=Config.BACKBONE, pretrained=True):
        super().__init__()
        # Load model with features_only=True to access intermediate layers
        # out_indices=(3, 4) corresponds to P4 (stride 16) and P5 (stride 32)
        self.encoder = timm.create_model(
            backbone_name,
            features_only=True,
            out_indices=(3, 4),
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
        )

        # Get channel counts for P4 and P5 from the model info
        infos = self.encoder.feature_info.info
        p4_ch = infos[0]["num_chs"]
        p5_ch = infos[1]["num_chs"]

        self.out_dim = p4_ch + p5_ch

    def forward(self, x):
        # x: (B * Seq, C, H, W)
        features = self.encoder(x)
        # features is a list [P4, P5]
        p4 = features[0]
        p5 = features[1]

        # Global Average Pooling
        p4_gap = F.adaptive_avg_pool2d(p4, (1, 1)).flatten(1)
        p5_gap = F.adaptive_avg_pool2d(p5, (1, 1)).flatten(1)

        # Concatenate features to preserve both fine and coarse details
        return torch.cat([p4_gap, p5_gap], dim=1)


class SequenceEncoder(nn.Module):
    """
    Bi-LSTM with learnable positional embeddings to model Z-axis continuity.
    """

    def __init__(
        self,
        input_dim,
        hidden_size=Config.LSTM_HIDDEN_SIZE,
        layers=Config.LSTM_LAYERS,
        dropout=Config.LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        # Output dimension is hidden_size * 2 due to bidirectionality
        self.out_dim = hidden_size * 2

        # Learnable Positional Embeddings: (1, Seq_Len, Out_Dim)
        # Allows the model to distinguish C1 from C7 based on relative position in the stack
        self.pos_embed = nn.Parameter(torch.zeros(1, Config.SEQ_LENGTH, self.out_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, Seq, Input_Dim)
        lstm_out, _ = self.lstm(x)  # (B, Seq, Hidden*2)

        # Inject positional information via addition
        return lstm_out + self.pos_embed


class AttentionHead(nn.Module):
    """
    Computes a weighted average of the sequence embeddings (Context Vector).
    """

    def __init__(self, input_dim):
        super().__init__()
        # Linear projection to calculate attention score per time step
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (B, Seq, Dim)
        scores = self.attention(x)  # (B, Seq, 1)
        weights = F.softmax(scores, dim=1)  # (B, Seq, 1)

        # Weighted sum: (B, Seq, Dim) * (B, Seq, 1) -> sum over Seq -> (B, Dim)
        context = torch.sum(x * weights, dim=1)
        return context


class HierarchicalHeads(nn.Module):
    """
    7 Independent Attention Heads for vertebrae + 1 Fusion Head for Patient Overall.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.vertebrae = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

        # 1. Independent Attention Heads for each level
        self.attention_heads = nn.ModuleDict(
            {v: AttentionHead(input_dim) for v in self.vertebrae}
        )

        # 2. Independent Classifiers for each level
        self.classifiers = nn.ModuleDict(
            {v: nn.Linear(input_dim, 1) for v in self.vertebrae}
        )

        # 3. Fusion Head for Patient Overall
        # Input is the concatenation of the 7 specific embeddings
        self.overall_classifier = nn.Linear(input_dim * 7, 1)

    def forward(self, x):
        # x: (B, Seq, Dim)

        logits = {}
        embeddings = []

        # Compute local predictions and collect embeddings
        for v in self.vertebrae:
            emb = self.attention_heads[v](x)  # (B, Dim)
            embeddings.append(emb)
            logits[v] = self.classifiers[v](emb)  # (B, 1)

        # Compute global prediction via embedding fusion
        # This enforces logical consistency: Global = f(Local_Features)
        overall_emb = torch.cat(embeddings, dim=1)  # (B, Dim * 7)
        logits["patient_overall"] = self.overall_classifier(overall_emb)

        return logits


class CalibratedHierarchicalSeqModel(nn.Module):
    """
    End-to-End Architecture:
    2.5D Input -> Multi-Level CNN -> Bi-LSTM + Pos Embed -> Hierarchical Attention Heads
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.feature_extractor = MultiLevelFeatureExtractor(pretrained=pretrained)
        self.sequence_encoder = SequenceEncoder(
            input_dim=self.feature_extractor.out_dim
        )
        self.heads = HierarchicalHeads(input_dim=self.sequence_encoder.out_dim)

    def forward(self, x):
        # x: (B, Seq, C, H, W)
        b, s, c, h, w = x.shape

        # 1. Flatten Batch and Sequence dimensions for CNN processing
        x = x.view(b * s, c, h, w)

        # 2. Extract Multi-Level Features
        features = self.feature_extractor(x)  # (B*S, Feat_Dim)

        # 3. Restore Sequence dimension
        features = features.view(b, s, -1)  # (B, S, Feat_Dim)

        # 4. Sequence Modeling
        seq_features = self.sequence_encoder(features)  # (B, S, LSTM_Dim)

        # 5. Hierarchical Prediction
        logits_dict = self.heads(seq_features)

        # 6. Format Output
        # Return tensor matching the order in Config.TARGET_COLS
        output_list = [logits_dict[col] for col in Config.TARGET_COLS]
        return torch.cat(output_list, dim=1)  # (B, 8)
