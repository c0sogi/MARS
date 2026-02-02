import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import ModelConfig, FeatureConfig


class DualBranchMLP(nn.Module):
    """
    Dual-Branch Neural Network (Cite solution_lesson_node_00012).
    Reverts to using only Semantic and Metadata branches with Batch Normalization (Cite solution_lesson_node_00013).
    """

    def __init__(self, meta_dim: int):
        super(DualBranchMLP, self).__init__()

        # --- Branch 1: Semantic (SBERT) ---
        self.semantic_layer = nn.Linear(
            ModelConfig.SEMANTIC_INPUT_DIM, ModelConfig.BRANCH_SEMANTIC_HIDDEN
        )
        self.semantic_bn = nn.BatchNorm1d(ModelConfig.BRANCH_SEMANTIC_HIDDEN)
        self.semantic_dropout = nn.Dropout(ModelConfig.DROPOUT_HIGH)

        # --- Branch 2: Metadata (Numerical) ---
        self.meta_layer = nn.Linear(meta_dim, ModelConfig.BRANCH_META_HIDDEN)
        self.meta_bn = nn.BatchNorm1d(ModelConfig.BRANCH_META_HIDDEN)
        self.meta_dropout = nn.Dropout(ModelConfig.DROPOUT_LOW)

        # --- Fusion Head ---
        fusion_input_dim = (
            ModelConfig.BRANCH_SEMANTIC_HIDDEN + ModelConfig.BRANCH_META_HIDDEN
        )

        self.fusion_layer = nn.Linear(fusion_input_dim, ModelConfig.FUSION_HIDDEN)
        self.fusion_bn = nn.BatchNorm1d(ModelConfig.FUSION_HIDDEN)
        self.fusion_dropout = nn.Dropout(ModelConfig.DROPOUT_MEDIUM)

        self.output_layer = nn.Linear(ModelConfig.FUSION_HIDDEN, 1)

    def forward(self, semantic_input, meta_input):
        # 1. Semantic Branch
        sem_out = self.semantic_layer(semantic_input)
        sem_out = self.semantic_bn(sem_out)
        sem_out = F.relu(sem_out)
        sem_out = self.semantic_dropout(sem_out)

        # 2. Metadata Branch
        meta_out = self.meta_layer(meta_input)
        meta_out = self.meta_bn(meta_out)
        meta_out = F.relu(meta_out)
        meta_out = self.meta_dropout(meta_out)

        # 3. Fusion
        combined = torch.cat([sem_out, meta_out], dim=1)

        fusion_out = self.fusion_layer(combined)
        fusion_out = self.fusion_bn(fusion_out)
        fusion_out = F.relu(fusion_out)
        fusion_out = self.fusion_dropout(fusion_out)

        logits = self.output_layer(fusion_out)

        return logits
