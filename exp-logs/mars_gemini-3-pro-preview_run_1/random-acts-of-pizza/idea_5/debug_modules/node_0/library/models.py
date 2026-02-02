import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import ModelConfig, FeatureConfig


class TripleBranchMLP(nn.Module):
    """
    Triple-Branch Neural Network for Pizza Request Success Prediction.

    This model fuses three distinct data streams:
    1. Semantic Branch: Processes pre-computed SBERT embeddings.
    2. Metadata Branch: Processes engineered numerical and meta-features.
    3. Community Branch: Processes sequences of subreddit IDs using an EmbeddingBag.

    The architecture is designed to balance the high-dimensionality of text embeddings
    with the sparse but high-signal nature of community history and metadata.
    """

    def __init__(self, meta_dim: int):
        """
        Args:
            meta_dim (int): The number of features in the metadata input vector.
                            This is required to initialize the input layer of the metadata branch.
        """
        super(TripleBranchMLP, self).__init__()

        # --- Configuration ---
        # Vocabulary size includes Top K subreddits + PAD(0) + UNK(1)
        self.vocab_size = FeatureConfig.TOP_K_SUBREDDITS + 2

        # --- Branch 1: Semantic (SBERT) ---
        # Input: (Batch, 384)
        # High dropout to regularize the dense semantic information
        self.semantic_layer = nn.Linear(
            ModelConfig.SEMANTIC_INPUT_DIM, ModelConfig.BRANCH_SEMANTIC_HIDDEN
        )
        self.semantic_dropout = nn.Dropout(ModelConfig.DROPOUT_HIGH)

        # --- Branch 2: Metadata (Numerical) ---
        # Input: (Batch, meta_dim)
        # Low dropout to preserve explicit signal from engineered features
        self.meta_layer = nn.Linear(meta_dim, ModelConfig.BRANCH_META_HIDDEN)
        self.meta_dropout = nn.Dropout(ModelConfig.DROPOUT_LOW)

        # --- Branch 3: Community (Subreddits) ---
        # Input: (Batch, Sequence_Length)
        # EmbeddingBag averages embeddings across the sequence dimension efficiently
        self.community_embedding = nn.EmbeddingBag(
            num_embeddings=self.vocab_size,
            embedding_dim=ModelConfig.SUBREDDIT_EMBED_DIM,
            padding_idx=0,
            mode="mean",
        )
        self.community_dense = nn.Linear(
            ModelConfig.SUBREDDIT_EMBED_DIM, ModelConfig.BRANCH_COMMUNITY_HIDDEN
        )

        # --- Fusion Head ---
        # Concatenates outputs from all three branches
        fusion_input_dim = (
            ModelConfig.BRANCH_SEMANTIC_HIDDEN
            + ModelConfig.BRANCH_META_HIDDEN
            + ModelConfig.BRANCH_COMMUNITY_HIDDEN
        )

        self.fusion_layer = nn.Linear(fusion_input_dim, ModelConfig.FUSION_HIDDEN)
        self.fusion_dropout = nn.Dropout(ModelConfig.DROPOUT_MEDIUM)

        # Output Layer: Single logit for binary classification
        self.output_layer = nn.Linear(ModelConfig.FUSION_HIDDEN, 1)

    def forward(self, semantic_input, community_input, meta_input):
        """
        Forward pass of the Triple-Branch MLP.

        Args:
            semantic_input (Tensor): Tensor of shape (Batch, 384) containing SBERT embeddings.
            community_input (Tensor): LongTensor of shape (Batch, Seq_Len) containing subreddit IDs.
            meta_input (Tensor): Tensor of shape (Batch, Meta_Dim) containing numerical features.

        Returns:
            Tensor: Logits of shape (Batch, 1).
        """
        # 1. Semantic Branch Processing
        sem_out = self.semantic_layer(semantic_input)
        sem_out = F.relu(sem_out)
        sem_out = self.semantic_dropout(sem_out)

        # 2. Metadata Branch Processing
        meta_out = self.meta_layer(meta_input)
        meta_out = F.relu(meta_out)
        meta_out = self.meta_dropout(meta_out)

        # 3. Community Branch Processing
        # EmbeddingBag takes (Batch, Seq_Len) and produces (Batch, Embedding_Dim)
        comm_emb = self.community_embedding(community_input)
        comm_out = self.community_dense(comm_emb)
        comm_out = F.relu(comm_out)

        # 4. Fusion
        combined = torch.cat([sem_out, meta_out, comm_out], dim=1)

        fusion_out = self.fusion_layer(combined)
        fusion_out = F.relu(fusion_out)
        fusion_out = self.fusion_dropout(fusion_out)

        logits = self.output_layer(fusion_out)

        return logits
