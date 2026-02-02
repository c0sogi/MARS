import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class LayerAggregator(nn.Module):
    """
    Computes a learnable weighted average of the last N hidden layers.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # Initialize weights to be equal
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, hidden_states):
        # hidden_states is a tuple of tensors (B, L, H)
        # We take the last num_layers
        # Note: hidden_states contains (embeddings, layer_1, ..., layer_12)
        # So [-4:] gets layers 9, 10, 11, 12
        layers_to_agg = hidden_states[-self.num_layers :]

        # Stack along a new dimension: (B, L, H, N)
        stacked = torch.stack(layers_to_agg, dim=-1)

        # Compute normalized weights: (N,)
        norm_weights = torch.softmax(self.weights, dim=0)

        # Weighted sum: (B, L, H, N) * (1, 1, 1, N) -> sum over N -> (B, L, H)
        weighted_sum = (stacked * norm_weights.view(1, 1, 1, -1)).sum(dim=-1)

        return weighted_sum


class ResidualHead(nn.Module):
    """
    Residual Projection Block as described:
    Output = Linear(Concat(F, Dropout(ReLU(Linear(F)))))
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.1):
        super().__init__()
        self.linear_inner = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        # The input to the final linear layer is the concatenation of original input F and transformed H
        self.linear_out = nn.Linear(input_dim + hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, input_dim)

        # Path: Linear -> ReLU -> Dropout
        h = self.linear_inner(x)
        h = self.activation(h)
        h = self.dropout(h)

        # Concatenate original input and transformed features
        concat = torch.cat([x, h], dim=-1)

        # Final projection
        logits = self.linear_out(concat)

        return logits


class MultiScaleDualEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Load Backbone
        config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=config)

        # Layer Aggregator
        self.aggregator = LayerAggregator(
            num_layers=Config.NUM_HIDDEN_LAYERS_TO_AGGREGATE
        )

        # Feature Dimensions
        self.hidden_size = Config.HIDDEN_SIZE

        # We concatenate:
        # 1. u_title (avg)
        # 2. u_body (avg)
        # 3. v_avg (avg)
        # 4. u_max (max)
        # 5. v_max (max)
        # 6. I_intent (prod)
        # 7. I_intent (diff)
        # 8. I_context (prod)
        # 9. I_context (diff)
        # 10. I_salience (prod)
        # 11. I_salience (diff)
        # Total = 11 vectors of size hidden_size
        self.fusion_dim = 11 * self.hidden_size

        # Layer Norm before head
        self.layer_norm = nn.LayerNorm(self.fusion_dim)

        # Residual Head
        # Using hidden_size (768) for the internal projection block
        self.head = ResidualHead(
            input_dim=self.fusion_dim,
            hidden_dim=self.hidden_size,
            output_dim=Config.NUM_LABELS,
        )

    def _pool_avg(self, hidden, mask):
        """
        Computes masked average pooling.
        hidden: (B, L, H)
        mask: (B, L) (1 for tokens to keep, 0 for others)
        """
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
        sum_embeddings = torch.sum(hidden * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def _pool_max(self, hidden, mask):
        """
        Computes masked max pooling.
        hidden: (B, L, H)
        mask: (B, L)
        """
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
        # Set masked positions to a very small number so they aren't selected by max
        hidden_masked = hidden.masked_fill(mask_expanded == 0, -1e9)
        max_embeddings, _ = torch.max(hidden_masked, dim=1)
        return max_embeddings

    def forward(self, batch):
        # Unpack batch
        q_input_ids = batch["q_input_ids"]
        q_attention_mask = batch["q_attention_mask"]
        q_title_mask = batch["q_title_mask"]
        q_body_mask = batch["q_body_mask"]

        a_input_ids = batch["a_input_ids"]
        a_attention_mask = batch["a_attention_mask"]

        # ==========================
        # Question Branch
        # ==========================
        q_out = self.backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        # Aggregate layers
        q_agg = self.aggregator(q_out.hidden_states)

        # Pooling
        u_title = self._pool_avg(q_agg, q_title_mask)
        u_body = self._pool_avg(q_agg, q_body_mask)
        u_max = self._pool_max(q_agg, q_attention_mask)

        # ==========================
        # Answer Branch
        # ==========================
        a_out = self.backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        # Aggregate layers
        a_agg = self.aggregator(a_out.hidden_states)

        # Pooling
        v_avg = self._pool_avg(a_agg, a_attention_mask)
        v_max = self._pool_max(a_agg, a_attention_mask)

        # ==========================
        # Interactions
        # ==========================
        # Intent Matching: u_title vs v_avg
        i_intent_prod = u_title * v_avg
        i_intent_diff = torch.abs(u_title - v_avg)

        # Context Matching: u_body vs v_avg
        i_context_prod = u_body * v_avg
        i_context_diff = torch.abs(u_body - v_avg)

        # Salience Matching: u_max vs v_max
        i_salience_prod = u_max * v_max
        i_salience_diff = torch.abs(u_max - v_max)

        # ==========================
        # Fusion & Head
        # ==========================
        # Concatenate all features
        features = torch.cat(
            [
                u_title,
                u_body,
                v_avg,
                u_max,
                v_max,
                i_intent_prod,
                i_intent_diff,
                i_context_prod,
                i_context_diff,
                i_salience_prod,
                i_salience_diff,
            ],
            dim=1,
        )

        # Normalize
        features = self.layer_norm(features)

        # Predict
        logits = self.head(features)

        return logits
