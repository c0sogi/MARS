import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class MeanPooling(nn.Module):
    """
    Applies Mean Pooling on the last hidden state.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: Tensor (batch, seq_len, hidden)
            attention_mask: Tensor (batch, seq_len)

        Returns:
            Tensor (batch, hidden_size)
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask


class SiameseDeberta(nn.Module):
    """
    Siamese Network using DeBERTa-v3-base backbone.
    Processes two branches (A and B) and combines them with meta-features
    for preference prediction.
    """

    def __init__(self):
        super(SiameseDeberta, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True

        # Initialize Backbone
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Initialize Pooling Layer
        self.pooler = MeanPooling()

        # Feature Dimensions
        # u, v, |u-v|, u*v -> 4 vectors
        embedding_dim = 4 * self.config.hidden_size
        meta_dim = 3  # Prompt len, Res A len, Res B len
        input_dim = embedding_dim + meta_dim

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_size),
            nn.BatchNorm1d(self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.config.hidden_size, Config.NUM_LABELS),
        )

        # Initialize weights for the custom head
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """
        Initialize weights for the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm1d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_one_branch(self, input_ids, attention_mask):
        """
        Passes one branch (Prompt + Response) through backbone and pooler.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use last_hidden_state for MeanPooling
        pooled_output = self.pooler(outputs.last_hidden_state, attention_mask)
        return pooled_output

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        meta_features,
    ):
        """
        Forward pass for the Siamese network.

        Args:
            input_ids_a, attention_mask_a: Inputs for Branch A
            input_ids_b, attention_mask_b: Inputs for Branch B
            meta_features: Normalized length features (batch, 3)

        Returns:
            logits: (batch, 3)
        """
        # 1. Get Embeddings for both branches
        u = self.forward_one_branch(input_ids_a, attention_mask_a)
        v = self.forward_one_branch(input_ids_b, attention_mask_b)

        # 2. Compute Interaction Features
        diff_uv = torch.abs(u - v)
        prod_uv = u * v

        # 3. Concatenate all features
        # [u, v, |u-v|, u*v, meta_features]
        features = torch.cat([u, v, diff_uv, prod_uv, meta_features], dim=1)

        # 4. Classification
        logits = self.classifier(features)

        return logits
