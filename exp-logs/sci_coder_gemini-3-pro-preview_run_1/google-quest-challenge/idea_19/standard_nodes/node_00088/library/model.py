import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class SymmetricDualEncoder(nn.Module):
    """
    Symmetric Independent RoBERTa Dual-Encoder with Siamese Alignment Bridge.

    This architecture uses two independent encoders to specialize in Question and Answer
    distributions respectively. A Siamese Alignment Bridge projects their representations
    into a shared subspace for valid geometric comparison, mitigating latent space misalignment.
    """

    def __init__(self):
        super().__init__()

        # Load Configuration
        self.model_name = Config.MODEL_NAME
        config = AutoConfig.from_pretrained(self.model_name)
        self.hidden_size = config.hidden_size

        # ==========================================
        # 1. Backbone Architecture: Independent Encoders
        # ==========================================
        # We use separate encoders for Questions and Answers to allow full specialization.
        self.q_encoder = AutoModel.from_pretrained(self.model_name)
        self.a_encoder = AutoModel.from_pretrained(self.model_name)

        # ==========================================
        # 2. Siamese Alignment Bridge
        # ==========================================
        # A shared linear projection to map specialized embeddings into a unified
        # "Interaction Subspace" before computing geometric interactions.
        self.alignment_bridge = nn.Linear(self.hidden_size, self.hidden_size)

        # ==========================================
        # 3. Interaction-Aware Fusion Dimensions
        # ==========================================
        # The fusion vector F consists of:
        # - Raw Signals: u_avg, u_max, v_avg, v_max (4 vectors)
        # - Aligned Interactions: u_aligned * v_aligned, |u_aligned - v_aligned| (2 vectors)
        # Total dimension = 6 * hidden_size
        self.fusion_dim = self.hidden_size * 6

        self.layer_norm = nn.LayerNorm(self.fusion_dim)

        # ==========================================
        # 4. Residual Interaction Head
        # ==========================================
        # Architecture: Output = Linear(Concat(F, Dropout(ReLU(Linear(F)))))
        self.head_hidden_dim = 512  # Bottleneck dimension for feature mixing

        self.head_proj = nn.Linear(self.fusion_dim, self.head_hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # Final projection to 30 target labels
        # Input: Concat(Fusion Vector, Residual Projection)
        self.final_proj = nn.Linear(
            self.fusion_dim + self.head_hidden_dim, Config.NUM_LABELS
        )

        # Initialize custom layers
        self._init_weights(self.alignment_bridge)
        self._init_weights(self.head_proj)
        self._init_weights(self.final_proj)

    def _init_weights(self, module):
        """Initialize weights for new layers using Xavier Uniform."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _pool(self, last_hidden_states, attention_mask, pool_type="avg"):
        """
        Pools hidden states based on attention mask.

        Args:
            last_hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len]
            pool_type: 'avg' or 'max'
        """
        # Expand mask to match hidden state dimensions: [batch, seq_len, 1]
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        )

        if pool_type == "avg":
            # Sum valid token embeddings
            sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
            # Count valid tokens (clamp to avoid division by zero)
            sum_mask = mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask

        elif pool_type == "max":
            # Replace padded tokens with a very small number so they aren't selected by max
            last_hidden_states_masked = last_hidden_states.clone()
            last_hidden_states_masked[mask_expanded == 0] = -1e9
            max_embeddings = torch.max(last_hidden_states_masked, 1)[0]
            return max_embeddings

        else:
            raise ValueError("Invalid pool_type. Choose 'avg' or 'max'.")

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a):
        """
        Forward pass of the Symmetric Dual Encoder.
        """
        # --- 1. Encode ---
        # Question Stream
        q_out = self.q_encoder(input_ids=input_ids_q, attention_mask=attention_mask_q)
        q_hidden = q_out.last_hidden_state

        # Answer Stream
        a_out = self.a_encoder(input_ids=input_ids_a, attention_mask=attention_mask_a)
        a_hidden = a_out.last_hidden_state

        # --- 2. Pooling ---
        # Extract global representations
        u_avg = self._pool(q_hidden, attention_mask_q, "avg")
        u_max = self._pool(q_hidden, attention_mask_q, "max")

        v_avg = self._pool(a_hidden, attention_mask_a, "avg")
        v_max = self._pool(a_hidden, attention_mask_a, "max")

        # --- 3. Interaction-Aware Fusion ---
        # Cite solution_lesson_node_00085: Compute interactions directly on pooled vectors
        # without uninitialized projection.
        interaction_prod = u_avg * v_avg
        interaction_diff = torch.abs(u_avg - v_avg)

        # Concatenate: [Raw Signals (4x) + Interactions (2x)]
        features = torch.cat(
            [u_avg, u_max, v_avg, v_max, interaction_prod, interaction_diff], dim=1
        )

        # Normalize fusion vector
        features = self.layer_norm(features)

        # --- 4. Standard MLP Head ---
        x = self.head_proj(features)
        x = self.activation(x)
        x = self.dropout(x)

        # Concatenate fusion features with residual projection to match final_proj input dim
        x = torch.cat([features, x], dim=1)
        logits = self.final_proj(x)

        return logits
