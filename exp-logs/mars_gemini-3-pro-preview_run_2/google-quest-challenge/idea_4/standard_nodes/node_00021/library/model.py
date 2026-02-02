import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MeanPooling(nn.Module):
    """
    Performs mean pooling on the token embeddings, accounting for the attention mask.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # Expand attention mask to match the embedding dimensions
        # Mask shape: [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings over the sequence length where mask is 1
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum the mask values to get the count of valid tokens
        sum_mask = input_mask_expanded.sum(1)

        # Avoid division by zero by clamping the divisor
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Compute mean
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class UnifiedDebertaSiamese(nn.Module):
    """
    Unified Context DeBERTa-v3 Siamese Network.

    Features:
    - Shared DeBERTa-v3 backbone for Question and Answer streams.
    - Mean Pooling for robust sentence embeddings.
    - Explicit interaction features: [u, v, |u-v|, u*v].
    - Unified MLP head for predicting all 30 targets simultaneously.
    """

    def __init__(self):
        super(UnifiedDebertaSiamese, self).__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Pooling Layer
        self.pooler = MeanPooling()

        # Calculate input dimension for the head
        # We concatenate u, v, |u-v|, u*v -> 4 vectors of size hidden_size
        self.hidden_size = self.config.hidden_size
        interaction_dim = self.hidden_size * 4

        # Unified Multi-Target MLP Head
        # Structure: Linear -> LayerNorm -> GELU -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.Linear(interaction_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, Config.NUM_LABELS),
        )

        # Initialize the weights of the custom head
        self._init_head_weights()

    def _init_head_weights(self):
        """
        Initialize the weights of the MLP head.
        """
        for module in self.head:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        """
        Forward pass of the Siamese Network.

        Args:
            q_input_ids: Input IDs for the Question stream.
            q_attention_mask: Attention Mask for the Question stream.
            a_input_ids: Input IDs for the Answer stream.
            a_attention_mask: Attention Mask for the Answer stream.

        Returns:
            logits: Raw output scores for the 30 target labels (before sigmoid).
        """
        # 1. Process Question Stream
        q_outputs = self.backbone(
            input_ids=q_input_ids, attention_mask=q_attention_mask
        )
        u = self.pooler(q_outputs.last_hidden_state, q_attention_mask)

        # 2. Process Answer Stream
        a_outputs = self.backbone(
            input_ids=a_input_ids, attention_mask=a_attention_mask
        )
        v = self.pooler(a_outputs.last_hidden_state, a_attention_mask)

        # 3. Compute Interaction Features
        # Absolute difference captures distance/disagreement
        diff = torch.abs(u - v)
        # Element-wise product captures alignment/similarity
        prod = u * v

        # 4. Feature Fusion
        # Concatenate: [u, v, |u-v|, u*v]
        features = torch.cat([u, v, diff, prod], dim=1)

        # 5. Prediction
        logits = self.head(features)

        return logits
