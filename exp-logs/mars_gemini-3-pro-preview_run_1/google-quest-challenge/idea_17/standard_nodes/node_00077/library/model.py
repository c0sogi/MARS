import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class ResidualInteractionHead(nn.Module):
    """
    Residual Projection Block for classification.

    Structure:
    - Path A: Linear -> ReLU -> Dropout
    - Path B: Identity (Skip Connection)
    - Output: Linear(Concat(Path A, Path B))
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.1):
        super().__init__()
        self.projector = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        # The final classifier takes the original input + the projected hidden representation
        self.classifier = nn.Linear(input_dim + hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize linear layers with Normal distribution (mu=0, sigma=0.02)
        to match the scale of the pre-trained backbone.
        """
        nn.init.normal_(self.projector.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projector.bias)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # Path A: Non-linear projection
        h = self.projector(x)
        h = self.activation(h)
        h = self.dropout(h)

        # Path B: Skip connection (Raw features)
        # Concatenate raw interaction features with learned abstractions
        combined = torch.cat([x, h], dim=-1)

        # Final prediction
        logits = self.classifier(combined)
        return logits


class DualEncoder(nn.Module):
    """
    Dual-Encoder architecture.
    Processes Question and Answer streams independently, then fuses them
    using explicit interaction features and a residual head.
    """

    def __init__(self):
        super().__init__()
        model_name = Config.MODEL_NAME
        config = AutoConfig.from_pretrained(model_name)

        # Instantiate two separate backbones for independent processing
        self.q_backbone = AutoModel.from_pretrained(model_name)
        self.a_backbone = AutoModel.from_pretrained(model_name)

        self.hidden_size = config.hidden_size  # 768

        # Fusion Vector F construction:
        # Components: u_avg, u_max, v_avg, v_max, u_avg*v_avg, |u_avg - v_avg|
        # Total: 6 vectors of size 768
        fusion_dim = self.hidden_size * 6

        # Normalization before the head
        self.layer_norm = nn.LayerNorm(fusion_dim)

        # Custom Residual Head
        self.head = ResidualInteractionHead(
            input_dim=fusion_dim,
            hidden_dim=self.hidden_size,
            output_dim=Config.NUM_LABELS,
        )

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        # --- Stream A: Question ---
        q_out = self.q_backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        q_hidden = q_out.last_hidden_state  # (Batch, Seq_Len, Hidden)

        # --- Stream B: Answer ---
        a_out = self.a_backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        a_hidden = a_out.last_hidden_state  # (Batch, Seq_Len, Hidden)

        # --- Pooling ---
        u_avg = self._mean_pooling(q_hidden, q_attention_mask)
        u_max = self._max_pooling(q_hidden, q_attention_mask)

        v_avg = self._mean_pooling(a_hidden, a_attention_mask)
        v_max = self._max_pooling(a_hidden, a_attention_mask)

        # --- Interaction ---
        # Compute interactions only on Average Pooled vectors
        uv_prod = u_avg * v_avg
        uv_diff = torch.abs(u_avg - v_avg)

        # --- Fusion ---
        # F = [u_avg, u_max, v_avg, v_max, u_avg * v_avg, |u_avg - v_avg|]
        features = torch.cat([u_avg, u_max, v_avg, v_max, uv_prod, uv_diff], dim=1)

        # Normalize
        features = self.layer_norm(features)

        # --- Prediction ---
        logits = self.head(features)

        return logits

    def _mean_pooling(self, last_hidden_state, attention_mask):
        """
        Computes the average of hidden states, ignoring padding tokens.
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def _max_pooling(self, last_hidden_state, attention_mask):
        """
        Computes the max of hidden states, ignoring padding tokens.
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).bool()
        )
        # Clone to avoid in-place modification errors during backprop if used multiple times
        embeddings = last_hidden_state.clone()
        # Set padding tokens to a very small number so they are not selected as max
        embeddings[~input_mask_expanded] = -1e9
        max_embeddings = torch.max(embeddings, 1)[0]
        return max_embeddings

    def freeze_backbone(self):
        """Freezes the parameters of both backbones."""
        for param in self.q_backbone.parameters():
            param.requires_grad = False
        for param in self.a_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the parameters of both backbones."""
        for param in self.q_backbone.parameters():
            param.requires_grad = True
        for param in self.a_backbone.parameters():
            param.requires_grad = True
