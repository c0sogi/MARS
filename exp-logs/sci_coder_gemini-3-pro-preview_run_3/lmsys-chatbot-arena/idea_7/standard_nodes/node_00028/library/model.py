import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Learned Attention Pooling layer.
    Computes a weighted average of sequence tokens, masking out specific tokens (e.g., Prompt/Pad).
    """

    def __init__(self, hidden_size):
        super().__init__()
        # A small MLP to compute attention scores
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (Batch, Seq, Hidden)
            mask: (Batch, Seq) - 1 for tokens to keep, 0 for tokens to ignore.
        """
        # Compute raw attention scores: (Batch, Seq, 1)
        scores = self.attn(x)

        # Cast to float32 to avoid overflow with -1e9 and ensure softmax stability
        scores = scores.float()

        if mask is not None:
            # Expand mask to match scores dimension: (Batch, Seq, 1)
            expanded_mask = mask.unsqueeze(-1)
            # Set scores of masked-out tokens to a very large negative number
            # so softmax becomes 0.
            scores = scores.masked_fill(expanded_mask == 0, -1e9)

        # Normalize scores to probabilities
        weights = torch.softmax(scores, dim=1)

        # Cast weights back to input dtype (FP16) for the weighted sum
        weights = weights.to(x.dtype)

        # Compute weighted sum: (Batch, Hidden)
        # Sum over sequence dimension
        pooled = torch.sum(x * weights, dim=1)
        return pooled


class SiameseDebertaMultiLayer(nn.Module):
    """
    Siamese Network using DeBERTa-v3-base with Multi-Layer Response-Isolated Pooling.
    """

    def __init__(self):
        super().__init__()

        # 1. Load Backbone
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.config.output_hidden_states = True
        # Ensure dropout is consistent if needed, though usually handled by training loop mode
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Enable Gradient Checkpointing for memory efficiency
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        self.hidden_size = self.config.hidden_size
        self.pooling_layers = Config.pooling_layers

        # 2. Define Pooling Layers
        # We use separate attention pooling weights for each of the extracted layers
        self.poolers = nn.ModuleList(
            [AttentionPooling(self.hidden_size) for _ in range(self.pooling_layers)]
        )

        # 3. Calculate Feature Dimensions
        # Each branch produces a vector of size: pooling_layers * hidden_size
        branch_dim = self.pooling_layers * self.hidden_size

        # We combine branches using: u, v, |u-v|, u*v
        combined_dim = branch_dim * 4

        # Add scalar features
        if Config.use_scalar_features:
            combined_dim += 3

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, Config.num_classes),
        )

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize weights for the classification head."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_branch(self, input_ids, attention_mask, response_mask):
        """
        Processes one branch (Prompt + Response X) through the backbone and pooling.
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.hidden_states is a tuple of (Batch, Seq, Hidden)
        # We take the last 'pooling_layers'
        # Note: hidden_states[0] is embeddings, hidden_states[-1] is last layer
        selected_hidden_states = outputs.hidden_states[-self.pooling_layers :]

        pooled_vectors = []
        for i, hidden_state in enumerate(selected_hidden_states):
            # Apply Attention Pooling with Response Mask
            # This ensures we only aggregate information from the Response tokens
            pooled = self.poolers[i](hidden_state, mask=response_mask)
            pooled_vectors.append(pooled)

        # Concatenate pooled vectors from all selected layers
        # Result shape: (Batch, pooling_layers * Hidden)
        return torch.cat(pooled_vectors, dim=1)

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        response_mask_a,
        input_ids_b,
        attention_mask_b,
        response_mask_b,
        scalars=None,
    ):
        """
        Forward pass for the Siamese network.
        """
        # Process Branch A
        u = self.forward_branch(input_ids_a, attention_mask_a, response_mask_a)

        # Process Branch B
        v = self.forward_branch(input_ids_b, attention_mask_b, response_mask_b)

        # Interaction Features
        diff_feat = torch.abs(u - v)
        prod_feat = u * v

        # Concatenate all features
        features = torch.cat([u, v, diff_feat, prod_feat], dim=1)

        # Append Scalar Features (Log Lengths)
        if scalars is not None and Config.use_scalar_features:
            features = torch.cat([features, scalars], dim=1)

        # Classification
        logits = self.classifier(features)

        return logits
