import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class ContextualAttentionPooling(nn.Module):
    """
    Learned Attention Pooling layer.
    Computes a weighted average of hidden states based on a learned attention score.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch, seq_len, hidden_size)
        # attention_mask: (batch, seq_len) - 1 for keep, 0 for ignore

        # Compute raw attention scores: (batch, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Apply mask: set ignored positions to a very large negative value
        # (1.0 - mask) * -10000.0 makes valid positions 0 and invalid -10000
        extended_mask = (1.0 - attention_mask.unsqueeze(-1)) * -10000.0
        w = w + extended_mask

        # Normalize scores
        scores = torch.softmax(w, dim=1)

        # Weighted sum: (batch, seq_len, hidden) * (batch, seq_len, 1) -> sum dim 1
        context_vector = torch.sum(last_hidden_state * scores, dim=1)
        return context_vector


class SiameseDebertaModel(nn.Module):
    """
    Siamese Architecture with Decoupled Contextual Pooling using DeBERTa-v3-Large.
    """

    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Enable gradient checkpointing for memory efficiency
        if Config.grad_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.hidden_size = self.config.hidden_size
        self.num_pooling_layers = Config.num_pooling_layers

        # Pooling Layer
        self.pooler = ContextualAttentionPooling(self.hidden_size)

        # Calculate Input Dimension for Classification Head
        # We pool 'num_pooling_layers' layers for both Prompt (P) and Response (R).
        # Each vector (P or R) has size: num_pooling_layers * hidden_size
        # We concatenate 7 such vectors: R_a, R_b, |Ra-Rb|, Ra*Rb, P, P*Ra, P*Rb
        # Plus 3 scalar features.

        vector_dim = self.num_pooling_layers * self.hidden_size
        input_dim = (7 * vector_dim) + 3

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(Config.dropout),
            nn.Linear(self.hidden_size, Config.num_classes),
        )

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _get_masks(self, input_ids, attention_mask):
        """
        Dynamically separates Prompt and Response tokens based on [SEP] token location.
        Assumes input structure: [CLS] Prompt [SEP] Response [SEP] (Padding)
        """
        sep_token_id = self.config.sep_token_id
        cls_token_id = self.config.cls_token_id

        # Identify SEP locations
        is_sep = (input_ids == sep_token_id).long()

        # Cumulative sum identifies segments:
        # 0: Before 1st SEP (Prompt)
        # 1: After 1st SEP, before 2nd SEP (Response)
        # 2: After 2nd SEP (Padding)
        segments = torch.cumsum(is_sep, dim=1)

        # Prompt Mask: Segment 0, excluding CLS, respecting original padding
        prompt_mask = (
            (segments == 0) & (input_ids != cls_token_id) & (attention_mask == 1)
        )

        # Response Mask: Segment 1, excluding the SEP token itself, respecting original padding
        response_mask = (
            (segments == 1) & (input_ids != sep_token_id) & (attention_mask == 1)
        )

        return prompt_mask.float(), response_mask.float()

    def forward_branch(self, input_ids, attention_mask):
        """
        Processes a single branch (Prompt + Response X) through the backbone and pooling.
        Returns decoupled Prompt Vector (P) and Response Vector (R).
        """
        # Pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Extract last N hidden layers
        # outputs.hidden_states is a tuple of (embeddings, layer_1, ..., layer_N)
        # We want the last 'num_pooling_layers'
        hidden_states = outputs.hidden_states[-self.num_pooling_layers :]

        # Derive masks for Prompt and Response
        prompt_mask, response_mask = self._get_masks(input_ids, attention_mask)

        p_vectors = []
        r_vectors = []

        # Apply pooling to each layer
        for layer_hidden in hidden_states:
            # Pool Prompt tokens
            p_v = self.pooler(layer_hidden, prompt_mask)
            # Pool Response tokens
            r_v = self.pooler(layer_hidden, response_mask)

            p_vectors.append(p_v)
            r_vectors.append(r_v)

        # Concatenate pooled representations from all selected layers
        # Result shape: (Batch, num_pooling_layers * hidden_size)
        P = torch.cat(p_vectors, dim=1)
        R = torch.cat(r_vectors, dim=1)

        return P, R

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        features,
        target=None,
    ):
        # Process Branch A
        P_a, R_a = self.forward_branch(input_ids_a, attention_mask_a)

        # Process Branch B
        P_b, R_b = self.forward_branch(input_ids_b, attention_mask_b)

        # Context Vector: Average of P_a and P_b (should be identical ideally, averaging adds robustness)
        P = (P_a + P_b) / 2.0

        # --- Feature Engineering ---

        # 1. Interaction Features
        diff = torch.abs(R_a - R_b)
        prod = R_a * R_b

        # 2. Context Modulation
        # Modulate response vectors by the prompt context
        ctx_a = P * R_a
        ctx_b = P * R_b

        # 3. Concatenate All Features
        # [R_a, R_b, Diff, Prod, Ctx_a, Ctx_b, P, Scalars]
        combined = torch.cat([R_a, R_b, diff, prod, ctx_a, ctx_b, P, features], dim=1)

        # Classification
        logits = self.classifier(combined)

        loss = None
        if target is not None:
            loss = F.cross_entropy(logits, target)

        return logits, loss
