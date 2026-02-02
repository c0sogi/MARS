import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class LearnedAttentionPooling(nn.Module):
    """
    Applies learned attention pooling to a sequence of hidden states.
    Projects hidden states to a scalar score, applies softmax, and computes weighted sum.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) - Boolean or 0/1 mask indicating valid tokens.
        """
        # Compute attention scores: (batch, seq_len, 1)
        weights = self.attention(last_hidden_state)

        # Broadcast mask to (batch, seq_len, 1)
        mask = attention_mask.unsqueeze(-1)

        # Set masked scores to -inf so softmax results in 0 (ignoring padding/masked tokens)
        # Works for both boolean (False=0) and int (0) masks
        # Cast to float32 to avoid overflow with -1e9 in FP16
        weights = weights.float()
        weights = weights.masked_fill(mask == 0, -1e9)

        # Softmax over sequence length
        weights = F.softmax(weights, dim=1)

        # Weighted sum: (batch, hidden_dim)
        weighted_average = torch.sum(last_hidden_state * weights, dim=1)

        return weighted_average


class SiameseDebertaHierarchical(nn.Module):
    """
    Siamese DeBERTa-v3-Base with Disentangled Hierarchical Pooling.

    Architecture:
    1. Shared Backbone: Encodes Branch A and Branch B.
    2. Disentanglement: Dynamically masks Prompt vs Response tokens.
    3. Stream 1 (Response): Pools last 4 layers of Response tokens -> Concatenates.
    4. Stream 2 (Context): Pools last layer of Prompt tokens -> Averages A & B context.
    5. Fusion: Combines Response features, Interaction terms, Context, and Scalars.
    6. Head: MLP Classifier.
    """

    def __init__(self, config=Config, sep_token_id=None):
        super().__init__()
        self.model_name = config.MODEL_NAME
        self.num_classes = config.NUM_CLASSES
        self.pool_layers = config.POOLING_NUM_LAYERS
        self.sep_token_id = sep_token_id

        # Load Backbone
        model_config = AutoConfig.from_pretrained(self.model_name)
        model_config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(self.model_name, config=model_config)
        self.hidden_dim = model_config.hidden_size

        # Pooling Modules
        # Stream 1: Response Content (Last 4 layers)
        # Independent attention poolers for each layer to capture hierarchical features
        self.response_poolers = nn.ModuleList(
            [LearnedAttentionPooling(self.hidden_dim) for _ in range(self.pool_layers)]
        )

        # Stream 2: Prompt Context (Last layer only)
        self.context_pooler = LearnedAttentionPooling(self.hidden_dim)

        # Classification Head
        # Response Vector R dim = 4 * hidden_dim (Concatenation of 4 layers)
        r_dim = self.pool_layers * self.hidden_dim

        # Input Feature Dimension Construction:
        # 1. R_A (r_dim)
        # 2. R_B (r_dim)
        # 3. |R_A - R_B| (r_dim)
        # 4. R_A * R_B (r_dim)
        # 5. Context P (hidden_dim)
        # 6. Scalars (3)
        input_dim = (4 * r_dim) + self.hidden_dim + 3

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, self.num_classes),
        )

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize weights for the classification head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, input_ids, attention_mask, token_type_ids=None, scalar_features=None
    ):
        """
        Args:
            input_ids: (batch, 2, seq_len)
            attention_mask: (batch, 2, seq_len)
            token_type_ids: (batch, 2, seq_len) - Optional
            scalar_features: (batch, 3) - [log_len_prompt, log_len_resp_a, log_len_resp_b]
        """
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(2)

        # Flatten input for backbone processing: (batch * 2, seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)

        # Backbone Forward
        outputs = self.backbone(
            input_ids=flat_input_ids, attention_mask=flat_attention_mask
        )
        # outputs.hidden_states is a tuple: (embeddings, layer_1, ..., layer_12)
        all_hidden_states = outputs.hidden_states

        # --- Create Disentanglement Masks ---
        # Format: [CLS] Prompt [SEP] Response [SEP]
        sep_token_id = (
            self.sep_token_id
            if self.sep_token_id is not None
            else self.backbone.config.sep_token_id
        )
        is_sep = flat_input_ids == sep_token_id

        # Cumulative sum of SEPs helps identify segments
        sep_cumsum = torch.cumsum(is_sep.long(), dim=1)

        # Logic:
        # Prompt: Up to and including the first [SEP]
        # Response: After first [SEP], up to and including second [SEP]
        prompt_mask_binary = (sep_cumsum == 0) | ((sep_cumsum == 1) & is_sep)
        response_mask_binary = (sep_cumsum == 1) & (~is_sep) | (
            (sep_cumsum == 2) & is_sep
        )

        # Apply padding mask (ensure we don't unmask padding)
        prompt_mask = prompt_mask_binary & flat_attention_mask.bool()
        response_mask = response_mask_binary & flat_attention_mask.bool()

        # --- Stream 1: Response Content ---
        # Extract last N layers and pool independently
        response_vectors = []

        for i in range(self.pool_layers):
            # Calculate layer index (e.g., -4, -3, -2, -1)
            layer_idx = -(self.pool_layers - i)
            hidden = all_hidden_states[layer_idx]

            # Pool using Response Mask
            pooled = self.response_poolers[i](hidden, response_mask)
            response_vectors.append(pooled)

        # Concatenate pooled layers: (batch*2, 4 * hidden_dim)
        r_raw = torch.cat(response_vectors, dim=1)

        # --- Stream 2: Prompt Context ---
        # Extract last layer only
        last_hidden = all_hidden_states[-1]
        # Pool using Prompt Mask
        p_raw = self.context_pooler(last_hidden, prompt_mask)  # (batch*2, hidden_dim)

        # --- Reconstruct Pairs and Compute Features ---
        # Unflatten: (batch, 2, dim)
        r_unflat = r_raw.view(batch_size, 2, -1)
        p_unflat = p_raw.view(batch_size, 2, -1)

        r_a = r_unflat[:, 0, :]  # (batch, r_dim)
        r_b = r_unflat[:, 1, :]

        p_a = p_unflat[:, 0, :]  # (batch, hidden_dim)
        p_b = p_unflat[:, 1, :]

        # Shared Context Vector P (Average of both branches' view of the prompt)
        p_vec = (p_a + p_b) / 2.0

        # Interaction Features
        r_diff = torch.abs(r_a - r_b)
        r_prod = r_a * r_b

        # Handle Scalars
        if scalar_features is None:
            scalar_features = torch.zeros(batch_size, 3, device=input_ids.device)

        # Concatenate All Features
        # [R_A, R_B, Diff, Prod, P, Scalars]
        combined_features = torch.cat(
            [r_a, r_b, r_diff, r_prod, p_vec, scalar_features], dim=1
        )

        # --- Classifier ---
        logits = self.classifier(combined_features)

        return logits
