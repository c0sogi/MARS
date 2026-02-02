import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config
from library.utils import get_logger

logger = get_logger("model")


class AttentionPooling(nn.Module):
    """
    Learned Attention Pooling layer.
    Computes a weighted average of token embeddings, where weights are learned
    and masked to exclude padding and non-target tokens (e.g., prompts).
    """

    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.Tanh(), nn.Linear(in_dim, 1)
        )

    def forward(self, x, mask):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            mask: [batch_size, seq_len] - Binary mask (1 for keep, 0 for ignore)
        """
        # Calculate raw attention scores
        # w: [batch, seq_len, 1]
        w = self.attention(x)

        # Apply mask: set score of masked tokens to a very large negative number
        # so softmax becomes 0.
        # mask is 1 for valid tokens, 0 for invalid.
        # (1.0 - mask) makes invalid tokens 1.
        extended_mask = (1.0 - mask.unsqueeze(-1)) * -1e9
        w = w + extended_mask

        # Normalize weights
        weights = torch.softmax(w, dim=1)

        # Compute weighted sum
        # [batch, seq_len, hidden] * [batch, seq_len, 1] -> sum over seq_len
        weighted_sum = torch.sum(x * weights, dim=1)

        return weighted_sum


class SiameseDeberta(nn.Module):
    """
    Siamese Architecture using DeBERTa-v3-base backbone.
    Features:
    - Shared encoder weights.
    - Multi-layer extraction (last N layers).
    - Response-isolated pooling (masks out prompt).
    - Interaction features (u, v, |u-v|, u*v).
    - Scalar feature integration.
    """

    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True
        self.config.hidden_dropout_prob = Config.DROPOUT
        self.config.attention_probs_dropout_prob = Config.DROPOUT

        # Load Backbone
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Enable Gradient Checkpointing for memory efficiency
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        self.hidden_size = self.config.hidden_size
        self.n_last_layers = Config.N_LAST_LAYERS_POOLING

        # Independent Attention Poolers for each of the last N layers
        self.poolers = nn.ModuleList(
            [AttentionPooling(self.hidden_size) for _ in range(self.n_last_layers)]
        )

        # Calculate Input Dimension for Classifier
        # Each branch produces a vector of size: n_last_layers * hidden_size
        branch_dim = self.n_last_layers * self.hidden_size

        # Interaction features: u, v, |u-v|, u*v
        # Total interaction dim = 4 * branch_dim
        # Scalar features dim = 3
        input_dim = (4 * branch_dim) + 3

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(1024, Config.NUM_CLASSES),
        )

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize weights for the classifier head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_branch(self, input_ids, attention_mask, token_type_ids):
        """
        Process a single branch (Prompt + Response).
        Extracts last N layers, masks out the prompt, and pools the response tokens.
        """
        # Forward pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract hidden states from all layers
        all_hidden_states = outputs.hidden_states

        # Select the last N layers
        selected_layers = all_hidden_states[-self.n_last_layers :]

        # Construct Response Mask
        # We want to pool ONLY the response tokens.
        # token_type_ids: 0 for Prompt, 1 for Response.
        # attention_mask: 1 for Real tokens, 0 for Padding.
        # response_mask = attention_mask AND token_type_ids
        if token_type_ids is not None:
            response_mask = attention_mask * token_type_ids
        else:
            # Fallback: if token_type_ids missing, pool everything (including prompt)
            response_mask = attention_mask

        # Apply pooling to each layer independently
        pooled_outputs = []
        for i, layer_hidden in enumerate(selected_layers):
            # layer_hidden: [batch, seq_len, hidden_size]
            pooled = self.poolers[i](layer_hidden, response_mask)
            pooled_outputs.append(pooled)

        # Concatenate pooled vectors from all selected layers
        # Result: [batch, n_last_layers * hidden_size]
        branch_vector = torch.cat(pooled_outputs, dim=1)
        return branch_vector

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        scalars,
        token_type_ids_a=None,
        token_type_ids_b=None,
        target=None,
    ):
        """
        Main forward pass.
        Args:
            input_ids_a, ...: Inputs for Branch A
            input_ids_b, ...: Inputs for Branch B
            scalars: [batch, 3] - Log lengths
            token_type_ids_*: Optional, used for masking prompt
            target: Ignored (loss calculated externally)
        """

        # 1. Process Branch A
        u = self.forward_branch(input_ids_a, attention_mask_a, token_type_ids_a)

        # 2. Process Branch B
        v = self.forward_branch(input_ids_b, attention_mask_b, token_type_ids_b)

        # 3. Compute Interaction Features
        diff = torch.abs(u - v)
        prod = u * v

        # 4. Concatenate All Features
        # [u, v, |u-v|, u*v, scalars]
        features = torch.cat([u, v, diff, prod, scalars], dim=1)

        # 5. Classification
        logits = self.classifier(features)

        return logits
