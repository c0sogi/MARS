import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class LayerWeightedSum(nn.Module):
    """
    Computes a learnable weighted sum of the last N hidden layers from the backbone.
    """

    def __init__(self, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, hidden_states):
        # hidden_states is a tuple of tensor embeddings from all layers
        # We take the last num_layers
        layers_to_agg = hidden_states[-self.num_layers :]

        # Stack them: (num_layers, batch, seq_len, hidden_size)
        stacked = torch.stack(layers_to_agg, dim=0)

        # Compute normalized weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum: (batch, seq_len, hidden_size)
        # Broadcasting weights: (num_layers, 1, 1, 1) * (num_layers, batch, seq, hidden)
        weighted_sum = torch.sum(norm_weights.view(-1, 1, 1, 1) * stacked, dim=0)

        return weighted_sum


class GranularSiameseDeBERTa(nn.Module):
    def __init__(self, cat_cardinalities=None):
        super().__init__()

        # Configuration
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True
        self.hidden_size = Config.HIDDEN_SIZE

        # Backbone
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Layer Aggregation
        self.layer_weighting = LayerWeightedSum(
            num_layers=Config.NUM_LAYERS_TO_AGGREGATE
        )

        # Categorical Embeddings
        # Default cardinalities based on EDA if not provided: Category ~5, Host ~63.
        # We add a buffer for unknown/padding.
        if cat_cardinalities is None:
            cat_cardinalities = {"category": 10, "host": 100}

        self.emb_category = nn.Embedding(cat_cardinalities["category"], 16)
        self.emb_host = nn.Embedding(cat_cardinalities["host"], 32)
        cat_feat_dim = 16 + 32

        # Interaction Dimensions
        # For each pair (Title-Answer, Body-Answer), we have:
        # [u, v, |u-v|, u*v] -> 4 * hidden_size
        self.inter_dim = 4 * self.hidden_size

        # Total Input Dimension for MLP
        # (Title-Answer Interaction) + (Body-Answer Interaction) + Categorical
        self.mlp_input_dim = (self.inter_dim * 2) + cat_feat_dim

        # Prediction Head
        self.dense = nn.Linear(self.mlp_input_dim, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.activation = nn.GELU()

        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

        self.classifier = nn.Linear(self.hidden_size, Config.NUM_LABELS)

        self._init_weights(self.dense)
        self._init_weights(self.classifier)
        self._init_weights(self.emb_category)
        self._init_weights(self.emb_host)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _pool_segments(self, hidden_states, mask):
        """
        Computes mean pooling for tokens where mask == 1.
        mask shape: (batch, seq_len)
        hidden_states: (batch, seq_len, hidden_size)
        """
        # Expand mask
        mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # Sum embeddings
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)

        # Sum mask (count of tokens)
        sum_mask = mask_expanded.sum(1)

        # Safe division
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask

    def _compute_interaction(self, u, v):
        """
        Computes interaction vector: [u, v, |u-v|, u*v]
        """
        return torch.cat([u, v, torch.abs(u - v), u * v], dim=1)

    def forward(
        self,
        q_input_ids,
        q_attention_mask,
        q_segment_ids,
        a_input_ids,
        a_attention_mask,
        cats,
    ):

        # --- Stream 1: Question ---
        q_out = self.backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        q_hidden = self.layer_weighting(q_out.hidden_states)

        # Segment Pooling
        # Title: q_segment_ids == 1
        # Body: q_segment_ids == 2
        u_title = self._pool_segments(q_hidden, (q_segment_ids == 1).long())
        u_body = self._pool_segments(q_hidden, (q_segment_ids == 2).long())

        # --- Stream 2: Answer ---
        a_out = self.backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        a_hidden = self.layer_weighting(a_out.hidden_states)

        # Answer Pooling (Standard attention mask)
        v_answer = self._pool_segments(a_hidden, a_attention_mask)

        # --- Interactions ---
        inter_title_ans = self._compute_interaction(u_title, v_answer)
        inter_body_ans = self._compute_interaction(u_body, v_answer)

        # --- Categorical Features ---
        # cats shape: (batch, 2) -> [category, host]
        cat_emb = self.emb_category(cats[:, 0])
        host_emb = self.emb_host(cats[:, 1])

        # --- Concatenation ---
        features = torch.cat(
            [inter_title_ans, inter_body_ans, cat_emb, host_emb], dim=1
        )

        # --- Prediction Head ---
        x = self.dense(features)
        x = self.layer_norm(x)
        x = self.activation(x)

        # Multi-Sample Dropout
        logits = torch.mean(
            torch.stack(
                [self.classifier(dropout(x)) for dropout in self.dropouts], dim=0
            ),
            dim=0,
        )

        return torch.sigmoid(logits)
