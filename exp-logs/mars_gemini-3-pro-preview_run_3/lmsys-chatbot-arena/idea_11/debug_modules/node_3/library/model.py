import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("model")


class ContextualAttentionPooling(nn.Module):
    """
    Learned Attention Pooling layer.
    Computes a weighted sum of hidden states: sum(alpha_i * h_i).
    Weights alpha_i are computed via a small MLP: softmax(v^T * tanh(W * h)).
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) - 1 for valid tokens, 0 for padding
        Returns:
            context_vector: (batch_size, hidden_size)
        """
        # Compute raw attention scores: (batch_size, seq_len, 1)
        scores = self.attention(hidden_states)

        # Mask padding tokens. We set their score to a very large negative number
        # so that softmax drives their weight to zero.
        # attention_mask needs to be broadcastable: (batch_size, seq_len, 1)
        expanded_mask = attention_mask.unsqueeze(-1)
        scores = scores.float().masked_fill(expanded_mask == 0, -1e9)

        # Normalize scores to probabilities
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)

        # Compute weighted sum
        context_vector = torch.sum(hidden_states * weights, dim=1)

        return context_vector


class SiameseDeberta(nn.Module):
    """
    Siamese Architecture with Decoupled Contextual Pooling.
    Uses DeBERTa-v3-Large as the backbone.
    """

    def __init__(self):
        super().__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True
        self.config.hidden_dropout_prob = 0.0  # We handle dropout in the head
        self.config.attention_probs_dropout_prob = 0.0

        # Load Backbone
        logger.info(f"Loading backbone: {Config.MODEL_NAME}")
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Enable Gradient Checkpointing for memory efficiency with Large models
        if Config.GRAD_CHECKPOINTING:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.hidden_size = self.config.hidden_size

        # Pooling Layers
        # We use separate pooling operations for each of the last N layers.
        # These poolers are shared between Prompt and Response extraction to learn
        # a consistent notion of "importance" at each abstraction level.
        self.poolers = nn.ModuleList(
            [
                ContextualAttentionPooling(self.hidden_size)
                for _ in range(Config.NUM_POOLING_LAYERS)
            ]
        )

        # Calculate Feature Dimension
        # We extract: R_A, R_B, P (averaged)
        # Each is a concatenation of N layers -> dim = N * hidden_size
        # Features: R_A, R_B, |R_A-R_B|, R_A*R_B, P, P*R_A, P*R_B
        # Total vector blocks: 7
        self.vector_dim = Config.NUM_POOLING_LAYERS * self.hidden_size
        self.total_feature_dim = 7 * self.vector_dim

        # Add Scalar Features
        if Config.USE_SCALARS:
            self.total_feature_dim += Config.NUM_SCALARS

        logger.info(f"Total Feature Dimension for Classifier: {self.total_feature_dim}")

        # Classification Head
        # A projection layer is often helpful when the input dimension is very large
        self.classifier = nn.Sequential(
            nn.Linear(self.total_feature_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.hidden_size, Config.NUM_LABELS),
        )

        # Initialize head weights
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward_branch(self, input_ids, attention_mask, token_type_ids):
        """
        Processes one branch of the Siamese network.
        Extracts decoupled Prompt and Response vectors.
        """
        # Pass through backbone
        # We do not pass token_type_ids to the backbone to rely on its default behavior
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Get hidden states from all layers
        all_hidden_states = outputs.hidden_states

        # Select the last N layers
        # Tuple of tensors -> List of tensors
        selected_layers = all_hidden_states[-Config.NUM_POOLING_LAYERS :]

        # Create Decoupled Masks
        # token_type_ids: 0 for Prompt, 1 for Response
        # We must also respect the padding in attention_mask

        # Mask for Prompt: Keep tokens where type is 0 AND not padding
        prompt_mask = attention_mask * (token_type_ids == 0)

        # Mask for Response: Keep tokens where type is 1 AND not padding
        resp_mask = attention_mask * (token_type_ids == 1)

        prompt_vectors = []
        resp_vectors = []

        # Apply pooling per layer
        for i, layer_hidden in enumerate(selected_layers):
            # layer_hidden: (bs, seq, hidden_size)

            # Pool Prompt
            p_vec = self.poolers[i](layer_hidden, prompt_mask)
            prompt_vectors.append(p_vec)

            # Pool Response
            r_vec = self.poolers[i](layer_hidden, resp_mask)
            resp_vectors.append(r_vec)

        # Concatenate layer outputs to form the final representation
        # Shape: (bs, NUM_POOLING_LAYERS * hidden_size)
        P = torch.cat(prompt_vectors, dim=1)
        R = torch.cat(resp_vectors, dim=1)

        return P, R

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        token_type_ids_a,
        input_ids_b,
        attention_mask_b,
        token_type_ids_b,
        scalars=None,
    ):
        """
        Forward pass for the Siamese model.
        """
        # Process Branch A
        P_a, R_a = self.forward_branch(input_ids_a, attention_mask_a, token_type_ids_a)

        # Process Branch B
        P_b, R_b = self.forward_branch(input_ids_b, attention_mask_b, token_type_ids_b)

        # Context Aggregation
        # Since the Prompt is identical (textually) in both branches, we average the
        # extracted representations to reduce noise.
        P = (P_a + P_b) / 2.0

        # Feature Engineering
        # 1. Response Interaction
        diff = torch.abs(R_a - R_b)
        prod = R_a * R_b

        # 2. Context Modulation
        # Condition the response representation on the prompt context
        ctx_a = P * R_a
        ctx_b = P * R_b

        # Concatenate all features
        features = torch.cat([R_a, R_b, diff, prod, P, ctx_a, ctx_b], dim=1)

        # Append Scalars if used
        if scalars is not None and Config.USE_SCALARS:
            features = torch.cat([features, scalars], dim=1)

        # Classification
        logits = self.classifier(features)

        return logits
