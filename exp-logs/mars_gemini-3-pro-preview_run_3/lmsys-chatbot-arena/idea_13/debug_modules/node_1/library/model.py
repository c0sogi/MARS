import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Learned Attention Pooling layer.
    Computes a weighted average of token embeddings based on learned attention scores.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, mask):
        """
        Args:
            x: Hidden states (batch_size, seq_len, hidden_size)
            mask: Attention mask (batch_size, seq_len) - 1 for valid, 0 for pad/ignored
        """
        # Compute attention scores
        # (batch, seq, 1)
        scores = self.attention(x)

        # Mask padding/ignored tokens with a large negative value
        # mask is (batch, seq), scores is (batch, seq, 1)
        # We need to broadcast mask to scores
        expanded_mask = mask.unsqueeze(-1)
        scores = scores.float().masked_fill(expanded_mask == 0, -1e9)

        # Softmax over sequence dimension
        weights = torch.softmax(scores, dim=1)

        # Weighted sum
        # (batch, seq, hidden) * (batch, seq, 1) -> (batch, seq, hidden) -> sum -> (batch, hidden)
        pooled_output = torch.sum(x * weights, dim=1)

        return pooled_output


class DualStreamSiameseModel(nn.Module):
    """
    Siamese DeBERTa-v3-Base with Dual-Stream Multi-Layer Aggregation.
    Decouples Prompt and Response processing to maximize signal clarity.
    """

    def __init__(self):
        super(DualStreamSiameseModel, self).__init__()

        # 1. Load Backbone
        config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        config.output_hidden_states = True
        config.hidden_dropout_prob = Config.DROPOUT
        config.attention_probs_dropout_prob = Config.DROPOUT

        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=config)
        self.hidden_size = Config.HIDDEN_SIZE

        # 2. Pooling Streams
        # Stream 1: Response (Content) - Last N layers
        self.response_layers_count = Config.RESPONSE_LAYERS
        self.response_poolers = nn.ModuleList(
            [
                AttentionPooling(self.hidden_size)
                for _ in range(self.response_layers_count)
            ]
        )

        # Stream 2: Prompt (Context) - Last M layers (Config says 1)
        self.context_layers_count = Config.CONTEXT_LAYERS
        self.prompt_poolers = nn.ModuleList(
            [
                AttentionPooling(self.hidden_size)
                for _ in range(self.context_layers_count)
            ]
        )

        # 3. Feature Dimensions
        # Response Vector R: Concatenation of N layers
        r_dim = self.hidden_size * self.response_layers_count

        # Context Vector P: Concatenation of M layers (usually 1)
        p_dim = self.hidden_size * self.context_layers_count

        # Interaction Features: R_A, R_B, |R_A - R_B|, R_A * R_B -> 4 * r_dim
        # Context Feature: P -> p_dim
        # Scalars: 3
        input_dim = (4 * r_dim) + p_dim + 3

        # 4. Classification Head
        self.head = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.hidden_size, Config.NUM_CLASSES),
        )

        # Initialize weights for head and poolers (backbone is pretrained)
        self._init_weights(self.head)
        self._init_weights(self.response_poolers)
        self._init_weights(self.prompt_poolers)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.ModuleList):
            for m in module:
                self._init_weights(m)
        elif isinstance(module, nn.Sequential):
            for m in module:
                self._init_weights(m)

    def _process_branch(self, hidden_states, attention_mask, token_type_ids):
        """
        Extracts Response Vector (R) and Prompt Vector (P) from a single branch.
        """
        # Create Masks
        # token_type_ids: 0 for Context (Prompt), 1 for Content (Response)
        # We must also respect the original attention_mask (padding)

        # Prompt Mask: type 0 AND valid token
        prompt_mask = (token_type_ids == 0) & (attention_mask == 1)

        # Response Mask: type 1 AND valid token
        response_mask = (token_type_ids == 1) & (attention_mask == 1)

        # --- Stream 1: Response Aggregation ---
        # Extract last N layers
        # hidden_states is tuple of (embeddings, layer_1, ... layer_12)
        # We want the last N.
        response_vectors = []
        for i in range(self.response_layers_count):
            # Index from end: -1, -2, etc.
            layer_idx = -(i + 1)
            layer_hidden = hidden_states[layer_idx]

            # Pool
            pooled = self.response_poolers[i](layer_hidden, response_mask)
            response_vectors.append(pooled)

        # Concatenate to form deep response vector R
        # Order: [Layer -1, Layer -2, ...]
        R = torch.cat(response_vectors, dim=1)

        # --- Stream 2: Prompt Aggregation ---
        prompt_vectors = []
        for i in range(self.context_layers_count):
            layer_idx = -(i + 1)
            layer_hidden = hidden_states[layer_idx]

            pooled = self.prompt_poolers[i](layer_hidden, prompt_mask)
            prompt_vectors.append(pooled)

        P = torch.cat(prompt_vectors, dim=1)

        return R, P

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        token_type_ids_a,
        input_ids_b,
        attention_mask_b,
        token_type_ids_b,
        scalars,
    ):
        """
        Args:
            input_ids_[a/b]: (batch, seq_len)
            attention_mask_[a/b]: (batch, seq_len)
            token_type_ids_[a/b]: (batch, seq_len)
            scalars: (batch, 3) -> [log_len_p, log_len_a, log_len_b]
        """

        # 1. Backbone Forward Pass
        # Branch A
        outputs_a = self.backbone(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            token_type_ids=token_type_ids_a,
        )
        # Branch B
        outputs_b = self.backbone(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,
            token_type_ids=token_type_ids_b,
        )

        # 2. Dual-Stream Processing
        R_a, P_a = self._process_branch(
            outputs_a.hidden_states, attention_mask_a, token_type_ids_a
        )
        R_b, P_b = self._process_branch(
            outputs_b.hidden_states, attention_mask_b, token_type_ids_b
        )

        # 3. Aggregation
        # Average Context Vector (Prompt is identical, but embeddings context-dependent)
        P = (P_a + P_b) / 2.0

        # Interaction Features
        diff_sim = torch.abs(R_a - R_b)
        prod_sim = R_a * R_b

        # Concatenate all features
        # [R_a, R_b, |Ra-Rb|, Ra*Rb, P, scalars]
        combined_features = torch.cat([R_a, R_b, diff_sim, prod_sim, P, scalars], dim=1)

        # 4. Classification
        logits = self.head(combined_features)

        return logits
