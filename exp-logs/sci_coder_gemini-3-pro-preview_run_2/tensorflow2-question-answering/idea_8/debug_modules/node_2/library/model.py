import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import config
from library.data_utils import load_embeddings
from library.model_components import (
    CosineInteraction,
    RBFKernelLayer,
    DepthwiseSeparableConv1D,
)


class KernelPoolingNetwork(nn.Module):
    """
    Kernel-Pooling Interaction Network for Natural Questions.

    This model uses fixed RBF kernels to rank long answer candidates based on
    soft-matching histograms and uses a lightweight depthwise separable
    convolutional encoder to extract short answer spans.
    """

    def __init__(self, vocab):
        """
        Args:
            vocab (dict): Vocabulary mapping token to index. Used to initialize embeddings.
        """
        super(KernelPoolingNetwork, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Embedding Layer
        # ---------------------------------------------------------------------
        # Load pre-trained embeddings (randomly initialized if cache not found/configured)
        embedding_matrix = load_embeddings(vocab, load_cached_data=True)
        num_embeddings, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False  # Freeze embeddings for efficiency

        # ---------------------------------------------------------------------
        # 2. Interaction & Ranking Module
        # ---------------------------------------------------------------------
        self.cosine_interaction = CosineInteraction()

        # RBF Kernels for soft counting
        self.rbf_layer = RBFKernelLayer(
            means=config.KERNEL_MEANS, sigmas=config.KERNEL_SIGMAS
        )

        # Ranking Head: Projects aggregated kernel features to a scalar score
        # Input dim is NUM_KERNELS (one count per kernel type)
        self.ranking_linear = nn.Linear(config.NUM_KERNELS, 1)

        # ---------------------------------------------------------------------
        # 3. Span Prediction Module
        # ---------------------------------------------------------------------
        # Input features: Word Embedding (D) + Max Similarity Score (1)
        span_input_dim = embedding_dim + 1
        hidden_dim = config.HIDDEN_DIM

        layers = []
        # First layer projects input to hidden dim
        layers.append(
            DepthwiseSeparableConv1D(
                in_channels=span_input_dim,
                out_channels=hidden_dim,
                kernel_size=config.CONV_KERNEL_SIZE,
                padding=config.CONV_KERNEL_SIZE // 2,
                activation=True,
            )
        )
        layers.append(nn.Dropout(config.DROPOUT_RATE))

        # Stack subsequent layers
        for _ in range(config.ENCODER_LAYERS - 1):
            layers.append(
                DepthwiseSeparableConv1D(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=config.CONV_KERNEL_SIZE,
                    padding=config.CONV_KERNEL_SIZE // 2,
                    activation=True,
                )
            )
            layers.append(nn.Dropout(config.DROPOUT_RATE))

        self.span_encoder = nn.Sequential(*layers)

        # Span Heads
        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)

        # ---------------------------------------------------------------------
        # 4. Yes/No Classification Head
        # ---------------------------------------------------------------------
        # Uses the same aggregated features as ranking
        self.yesno_head = nn.Sequential(
            nn.Linear(config.NUM_KERNELS, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, config.NUM_YES_NO_CLASSES),
        )

    def forward(self, question, candidate):
        """
        Forward pass of the network.

        Args:
            question: Tensor [Batch, Q_Len]
            candidate: Tensor [Batch, C_Len]

        Returns:
            dict containing:
                - long_score: [Batch, 1] (Logits for long answer ranking)
                - start_logits: [Batch, C_Len]
                - end_logits: [Batch, C_Len]
                - yesno_logits: [Batch, Num_Classes]
        """
        # 1. Embeddings
        # Shapes: [B, Q, D], [B, C, D]
        q_emb = self.embedding(question)
        c_emb = self.embedding(candidate)

        # 2. Interaction Matrix
        # Shape: [B, Q, C]
        interaction = self.cosine_interaction(q_emb, c_emb)

        # ---------------------------------------------------------------------
        # Ranking Path (Kernel Pooling)
        # ---------------------------------------------------------------------
        # Apply RBF Kernels -> [B, Q, K]
        # Each vector at q represents the log-count of matches for that query word
        # across the document for different similarity thresholds (kernels).
        log_pooled_features = self.rbf_layer(interaction)

        # Aggregate over Query dimension (Summing log-counts ~ multiplying probabilities)
        # Shape: [B, K]
        aggregated_features = torch.sum(log_pooled_features, dim=1)

        # Long Answer Score
        # Shape: [B, 1]
        long_score = self.ranking_linear(aggregated_features)

        # ---------------------------------------------------------------------
        # Span Prediction Path
        # ---------------------------------------------------------------------
        # Compute Max Similarity for each candidate word w.r.t question words
        # Interaction is [B, Q, C], max over Q dim -> [B, C]
        # This tells us how relevant each candidate word is to *any* part of the question.
        max_sim, _ = torch.max(interaction, dim=1)

        # Concatenate embedding and similarity feature
        # c_emb: [B, C, D]
        # max_sim: [B, C] -> [B, C, 1]
        span_input = torch.cat([c_emb, max_sim.unsqueeze(-1)], dim=2)

        # Permute for Conv1d: [B, Channels, Length]
        # [B, C, D+1] -> [B, D+1, C]
        span_input = span_input.permute(0, 2, 1)

        # Encode
        # Shape: [B, Hidden, C]
        span_features = self.span_encoder(span_input)

        # Permute back for Linear heads: [B, C, Hidden]
        span_features = span_features.permute(0, 2, 1)

        # Predict Start/End logits
        # Shape: [B, C, 1] -> Squeeze to [B, C]
        start_logits = self.start_head(span_features).squeeze(-1)
        end_logits = self.end_head(span_features).squeeze(-1)

        # ---------------------------------------------------------------------
        # Yes/No Path
        # ---------------------------------------------------------------------
        # Uses the document-level matching features
        # Shape: [B, Num_Classes]
        yesno_logits = self.yesno_head(aggregated_features)

        return {
            "long_score": long_score,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "yesno_logits": yesno_logits,
        }
