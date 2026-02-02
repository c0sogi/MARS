import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Learned Attention Pooling layer.
    Computes a weighted average of token embeddings where weights are learned via a projection.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) - 1 for valid tokens, 0 for masked.
        Returns:
            pooled_output: (batch_size, hidden_size)
        """
        # Calculate attention scores
        w = self.attention(last_hidden_state)  # (batch, seq, 1)
        w = w.squeeze(-1)  # (batch, seq)

        # Apply mask: set masked tokens to a very small number so softmax makes them zero
        # attention_mask is 1 for keep, 0 for ignore
        w = w.float().masked_fill(attention_mask == 0, -1e9)

        # Softmax to get probabilities
        w = torch.softmax(w, dim=-1)  # (batch, seq)
        w = w.unsqueeze(-1)  # (batch, seq, 1)

        # Weighted sum of hidden states
        embeddings = torch.sum(last_hidden_state * w, dim=1)  # (batch, hidden)
        return embeddings


class SiameseDeberta(nn.Module):
    """
    Siamese DeBERTa-v3-Large with Disentangled Hierarchical Pooling.

    Architecture:
    1. Shared Backbone: DeBERTa-v3-Large.
    2. Disentangled Pooling:
       - Context Stream: Extracts prompt features from the final layer.
       - Response Stream: Extracts response features from the last 4 layers.
    3. Interaction Head: Combines features (A, B, |A-B|, A*B, Context, Scalars) for classification.
    """

    def __init__(self):
        super().__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True

        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Enable Gradient Checkpointing to save memory with Large model
        self.backbone.gradient_checkpointing_enable()

        self.hidden_size = self.config.hidden_size

        # Retrieve special token IDs from the tokenizer, not the config
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.sep_token_id = tokenizer.sep_token_id
        self.cls_token_id = tokenizer.cls_token_id

        # --- Pooling Layers ---

        # Response Stream: Independent pooling for last 4 layers
        self.response_poolers = nn.ModuleList(
            [AttentionPooling(self.hidden_size) for _ in range(4)]
        )

        # Context Stream: Single pooling for the final layer
        self.context_pooler = AttentionPooling(self.hidden_size)

        # --- Feature Dimensions ---

        # Response Feature: Concatenation of 4 layers -> 4 * hidden
        self.response_dim = 4 * self.hidden_size

        # Interaction Features: R_A, R_B, |R_A - R_B|, R_A * R_B -> 4 * response_dim
        self.interaction_dim = 4 * self.response_dim

        # Context Feature: 1 * hidden
        self.context_dim = self.hidden_size

        # Scalar Features: 3 (log lengths)
        self.scalar_dim = 3

        # Total Input Dimension for Classifier
        self.total_dim = self.interaction_dim + self.context_dim + self.scalar_dim

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(self.total_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 3),  # Output: [Win_A, Win_B, Tie]
        )

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize weights for the classification head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _get_masks(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Generates binary masks to separate Prompt (Context) from Response (Content).
        Structure: [CLS] Prompt [SEP] Response [SEP]
        """
        # Identify SEP tokens
        sep_mask = (input_ids == self.sep_token_id).long()

        # Cumulative sum to identify segments.
        # [CLS] P P [SEP] R R [SEP]
        # sep:  0 0 0  1  0 0  1
        # cum:  0 0 0  1  1 1  2
        segment_ids = torch.cumsum(sep_mask, dim=1)

        # Prompt Mask: Segment 0 (before first SEP), excluding CLS
        # cumsum == 0 implies we are before the first SEP.
        prompt_mask = (
            (segment_ids == 0)
            & (input_ids != self.cls_token_id)
            & (attention_mask == 1)
        )

        # Response Mask: Segment 1 (after first SEP, before second SEP)
        # cumsum == 1 implies we have passed the first SEP.
        # We exclude the SEP token itself from the content mask.
        response_mask = (
            (segment_ids == 1)
            & (input_ids != self.sep_token_id)
            & (attention_mask == 1)
        )

        return prompt_mask.float(), response_mask.float()

    def forward_branch(self, input_ids, attention_mask):
        """Process a single branch (A or B) through backbone and pooling."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # Tuple of tensors

        # Generate specific masks
        prompt_mask, response_mask = self._get_masks(input_ids, attention_mask)

        # 1. Context Stream (Prompt)
        # Use final hidden layer only
        ctx_feat = self.context_pooler(hidden_states[-1], prompt_mask)

        # 2. Response Stream (Content)
        # Use last 4 hidden layers, pool independently, then concatenate
        resp_feats = []
        for i in range(4):
            # Access layers from end: -1, -2, -3, -4
            layer_idx = -(i + 1)
            feat = self.response_poolers[i](hidden_states[layer_idx], response_mask)
            resp_feats.append(feat)

        resp_feat = torch.cat(resp_feats, dim=-1)  # (B, 4 * hidden)

        return ctx_feat, resp_feat

    def forward(
        self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, scalars
    ):
        """
        Forward pass for the Siamese model.
        Args:
            input_ids_a, attention_mask_a: Inputs for Branch A
            input_ids_b, attention_mask_b: Inputs for Branch B
            scalars: (B, 3) Log-transformed lengths
        """
        # Process Branch A
        ctx_a, resp_a = self.forward_branch(input_ids_a, attention_mask_a)

        # Process Branch B
        ctx_b, resp_b = self.forward_branch(input_ids_b, attention_mask_b)

        # --- Feature Combination ---

        # Context: Average the context vectors from both branches (Prompt is identical)
        context_vector = (ctx_a + ctx_b) / 2.0

        # Response Interactions
        diff = torch.abs(resp_a - resp_b)
        prod = resp_a * resp_b

        # Concatenate all features
        # [Resp_A, Resp_B, |A-B|, A*B, Context, Scalars]
        combined = torch.cat(
            [resp_a, resp_b, diff, prod, context_vector, scalars], dim=1
        )

        # Classification
        logits = self.classifier(combined)

        return logits
