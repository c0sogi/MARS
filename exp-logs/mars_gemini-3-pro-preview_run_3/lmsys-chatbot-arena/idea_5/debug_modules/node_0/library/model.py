import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class ResponseIsolatedPooling(nn.Module):
    """
    Pooling layer that attends strictly to response tokens, masking out the prompt.
    Uses a learnable attention mechanism.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, response_mask):
        """
        Args:
            last_hidden_state: (batch, seq_len, hidden_size)
            response_mask: (batch, seq_len) - 1 for response tokens, 0 for prompt/padding
        """
        # Calculate raw attention scores
        # w: (batch, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Create mask for softmax (broadcastable)
        # mask: (batch, seq_len, 1)
        mask = response_mask.unsqueeze(-1)

        # Apply mask: set scores of non-response tokens to a very small number
        # We use -1e4 instead of -inf to avoid NaNs in fp16
        w = w.masked_fill(mask == 0, -1e4)

        # Softmax to get normalized weights
        att_weights = torch.softmax(w, dim=1)

        # Weighted sum of hidden states
        # (batch, seq_len, 1) * (batch, seq_len, hidden) -> sum over seq_len
        pooled_output = torch.sum(att_weights * last_hidden_state, dim=1)

        return pooled_output


class SiameseDebertaModel(nn.Module):
    """
    Siamese architecture using DeBERTa-v3-base backbone with response-isolated pooling.
    """

    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Enable Gradient Checkpointing for memory efficiency
        if Config.USE_GRADIENT_CHECKPOINTING:
            self.backbone.gradient_checkpointing_enable()

        # Pooling Layer
        self.pooling = ResponseIsolatedPooling(self.config.hidden_size)

        # Classification Head
        # Features: u, v, |u-v|, u*v (4 * hidden) + scalars (3)
        input_dim = (self.config.hidden_size * 4) + 3

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(Config.HIDDEN_DROPOUT_PROB),
            nn.Linear(self.config.hidden_size, Config.NUM_CLASSES),
        )

        # Initialize classifier weights
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, response_mask, scalars):
        """
        Args:
            input_ids: (batch, 2, seq_len)
            attention_mask: (batch, 2, seq_len)
            response_mask: (batch, 2, seq_len)
            scalars: (batch, 3) - [log(prompt_len), log(resp_a_len), log(resp_b_len)]

        Returns:
            logits: (batch, num_classes)
        """
        batch_size = input_ids.size(0)

        # Flatten batch and branch dimensions to process in parallel
        # shape: (batch * 2, seq_len)
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_response_mask = response_mask.view(-1, response_mask.size(-1))

        # Backbone Forward Pass
        outputs = self.backbone(
            input_ids=flat_input_ids, attention_mask=flat_attention_mask
        )
        last_hidden_state = outputs.last_hidden_state

        # Apply Response-Isolated Pooling
        # shape: (batch * 2, hidden_size)
        pooled_output = self.pooling(last_hidden_state, flat_response_mask)

        # Reshape back to (batch, 2, hidden_size) to separate branches
        pooled_output = pooled_output.view(batch_size, 2, -1)

        # Extract embeddings for Branch A (u) and Branch B (v)
        u = pooled_output[:, 0, :]
        v = pooled_output[:, 1, :]

        # Compute Interaction Features
        diff_feat = torch.abs(u - v)
        prod_feat = u * v

        # Concatenate all features
        # u, v, |u-v|, u*v, scalars
        combined_features = torch.cat([u, v, diff_feat, prod_feat, scalars], dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits
