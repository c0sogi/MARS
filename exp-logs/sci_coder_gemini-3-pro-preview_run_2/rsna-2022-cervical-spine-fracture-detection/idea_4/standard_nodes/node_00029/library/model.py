import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class AttentionHead(nn.Module):
    """
    Class-Specific Attention Head.
    Computes a weighted average of the sequence features based on a learned internal attention mechanism.

    Args:
        input_dim (int): Dimension of input features (LSTM output).
        hidden_dim (int): Dimension of the internal attention layer (Config.EMBED_DIM).
    """

    def __init__(self, input_dim, hidden_dim):
        super(AttentionHead, self).__init__()
        # Project input to hidden dimension for score calculation
        self.W = nn.Linear(input_dim, hidden_dim)
        # Project hidden dimension to scalar score
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)

        # Calculate attention scores
        # u: (Batch, Seq_Len, Hidden_Dim)
        u = torch.tanh(self.W(x))

        # scores: (Batch, Seq_Len, 1)
        scores = self.V(u)

        # Calculate weights via softmax along sequence dimension
        # weights: (Batch, Seq_Len, 1)
        weights = F.softmax(scores, dim=1)

        # Weighted aggregation (Embedding-Space Aggregation)
        # context: (Batch, Input_Dim)
        context = torch.sum(x * weights, dim=1)

        return context


class AnatomicallyAwareModel(nn.Module):
    """
    2.5D Anatomically-Aware Multi-Head Attention Network.

    Structure:
    1. 2.5D CNN Backbone (EfficientNet-B4) -> Extracts features per slice.
    2. Bidirectional LSTM -> Models Z-axis continuity.
    3. Positional Embeddings -> Injects anatomical location info.
    4. Multi-Head Attention -> 8 independent heads aggregate features for each target.
    5. Classifiers -> 8 independent linear layers produce logits.
    """

    def __init__(self):
        super(AnatomicallyAwareModel, self).__init__()

        # --- 1. Backbone ---
        # Load EfficientNet-B4 from timm
        # num_classes=0 ensures we get the feature vector before the final classifier
        # in_chans=3 corresponds to the 2.5D stacking (z-1, z, z+1)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            num_classes=0,
            in_chans=Config.IN_CHANS,
        )
        self.backbone.set_grad_checkpointing(True)

        # Determine backbone feature dimension (e.g., 1792 for B4)
        self.feature_dim = self.backbone.num_features

        # --- 2. Sequential Encoding ---
        # Bidirectional LSTM to capture context from above and below
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_LAYERS > 1 else 0.0,
        )

        # LSTM output dimension (Bidirectional means 2 * hidden_size)
        self.lstm_out_dim = Config.LSTM_HIDDEN_SIZE * 2

        # --- 3. Positional Embeddings ---
        # Learnable vector added to LSTM outputs to distinguish vertebrae levels
        # Shape: (1, Seq_Len, LSTM_Out_Dim)
        # This allows the model to learn that the start of the sequence is C1 and end is C7
        self.pos_embed = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, self.lstm_out_dim))

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- 4. Attention Heads ---
        # Create one attention head for each class (C1-C7, patient_overall)
        # This disentangles the prediction tasks
        self.attention_heads = nn.ModuleList(
            [
                AttentionHead(self.lstm_out_dim, Config.EMBED_DIM)
                for _ in range(Config.NUM_CLASSES)
            ]
        )

        # --- 5. Classifiers ---
        # Create one linear classifier for each class
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.lstm_out_dim, 1) for _ in range(Config.NUM_CLASSES)]
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Channels, Height, Width)
        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        b, s, c, h, w = x.shape

        # --- Feature Extraction ---
        # Flatten Batch and Sequence dimensions to process all slices in parallel through CNN
        # Shape: (Batch * Seq_Len, C, H, W)
        x = x.view(b * s, c, h, w)

        # Pass through backbone
        # Shape: (Batch * Seq_Len, Feature_Dim)
        features = self.backbone(x)

        # Reshape back to sequence format
        # Shape: (Batch, Seq_Len, Feature_Dim)
        features = features.view(b, s, -1)

        # --- Sequential Modeling ---
        # Pass through LSTM
        # lstm_out: (Batch, Seq_Len, LSTM_Out_Dim)
        lstm_out, _ = self.lstm(features)

        # --- Positional Injection ---
        # Add learnable positional embeddings (broadcasted over batch)
        # x_seq: (Batch, Seq_Len, LSTM_Out_Dim)
        x_seq = lstm_out + self.pos_embed

        # --- Multi-Head Aggregation & Prediction ---
        logits_list = []

        # Iterate over each specific target class
        for i in range(Config.NUM_CLASSES):
            # 1. Aggregate: Weighted sum of features specific to this class
            # context: (Batch, LSTM_Out_Dim)
            context = self.attention_heads[i](x_seq)

            # 2. Classify: Linear projection to logit
            # logit: (Batch, 1)
            logit = self.classifiers[i](context)
            logits_list.append(logit)

        # Concatenate all logits
        # Output: (Batch, Num_Classes)
        output = torch.cat(logits_list, dim=1)

        return output
