import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class SiameseDeberta(nn.Module):
    """
    Siamese Network with DeBERTa-v3-base backbone and Meta-Feature Fusion.

    Architecture:
    1. Shared DeBERTa backbone processes (Prompt, Response A) and (Prompt, Response B).
    2. Mean Pooling extracts sentence embeddings (u, v).
    3. Interaction terms computed: |u-v|, u*v.
    4. Concatenation: [u, v, |u-v|, u*v, meta_features].
    5. Classification Head: Linear -> ReLU -> Dropout -> Linear.
    """

    def __init__(self, model_name=Config.MODEL_NAME, num_classes=3, meta_dim=3):
        """
        Args:
            model_name (str): HuggingFace model identifier.
            num_classes (int): Number of output classes (3: A, B, Tie).
            meta_dim (int): Number of meta-features (3: len_prompt, len_a, len_b).
        """
        super(SiameseDeberta, self).__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Feature Dimensions
        self.hidden_size = self.config.hidden_size

        # Input dimension for the classifier:
        # u (hidden) + v (hidden) + |u-v| (hidden) + u*v (hidden) + meta_features (meta_dim)
        self.combined_dim = (self.hidden_size * 4) + meta_dim

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_classes),
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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Performs mean pooling on token embeddings, accounting for the attention mask.

        Args:
            token_embeddings (torch.Tensor): Shape (batch, seq_len, hidden_size)
            attention_mask (torch.Tensor): Shape (batch, seq_len)

        Returns:
            torch.Tensor: Pooled sentence embeddings of shape (batch, hidden_size)
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        meta_features,
    ):
        """
        Forward pass of the Siamese network.

        Args:
            input_ids_a (torch.Tensor): Input IDs for Branch A.
            attention_mask_a (torch.Tensor): Attention Mask for Branch A.
            input_ids_b (torch.Tensor): Input IDs for Branch B.
            attention_mask_b (torch.Tensor): Attention Mask for Branch B.
            meta_features (torch.Tensor): Normalized meta-features (batch, meta_dim).

        Returns:
            torch.Tensor: Logits for the 3 classes (batch, 3).
        """
        # --- Branch A ---
        outputs_a = self.backbone(
            input_ids=input_ids_a, attention_mask=attention_mask_a
        )
        u = self.mean_pooling(outputs_a.last_hidden_state, attention_mask_a)

        # --- Branch B ---
        outputs_b = self.backbone(
            input_ids=input_ids_b, attention_mask=attention_mask_b
        )
        v = self.mean_pooling(outputs_b.last_hidden_state, attention_mask_b)

        # --- Interaction Terms ---
        diff_emb = torch.abs(u - v)
        prod_emb = u * v

        # --- Feature Fusion ---
        # Concatenate: [u, v, |u-v|, u*v, meta_features]
        combined_features = torch.cat([u, v, diff_emb, prod_emb, meta_features], dim=1)

        # --- Classification ---
        logits = self.classifier(combined_features)

        return logits
