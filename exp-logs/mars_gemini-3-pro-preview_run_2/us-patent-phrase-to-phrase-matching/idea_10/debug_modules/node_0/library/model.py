import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Applies attention-based pooling to the last hidden state of the transformer.
    Computes a weighted average of token embeddings.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # Calculate attention weights
        w = self.attention(last_hidden_state)  # (Batch, Seq_Len, 1)

        # Mask padding tokens (set to very small value before softmax)
        float_mask = attention_mask.unsqueeze(-1).float()
        w = w.masked_fill(float_mask == 0, -1e9)

        # Normalize weights
        w = torch.softmax(w, dim=1)  # (Batch, Seq_Len, 1)

        # Weighted sum
        return torch.sum(w * last_hidden_state, dim=1)  # (Batch, Hidden_Size)


class CustomDeberta(nn.Module):
    """
    Custom DeBERTa model with Attention Pooling and Structural Feature Fusion.
    """

    def __init__(self):
        super(CustomDeberta, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.config.attention_probs_dropout_prob = Config.dropout
        self.config.hidden_dropout_prob = Config.dropout

        # Load Backbone
        self.model = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Initialize Pooling Layer
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Determine dimension for the final classification layer
        # Fusion: Backbone Hidden Size + Number of Structural Features
        self.num_structural_features = len(Config.structural_features)
        self.fc_input_dim = self.config.hidden_size + self.num_structural_features

        # Classification Head
        self.dropout = nn.Dropout(Config.dropout)
        self.fc = nn.Linear(self.fc_input_dim, Config.num_classes)

        # Initialize custom weights
        self._init_weights(self.pooler)
        self._init_weights(self.fc)

        # Enable Gradient Checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

    def _init_weights(self, module):
        """
        Initializes weights for the custom layers (Head and Pooler).
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for submodule in module:
                self._init_weights(submodule)

    def forward(
        self, input_ids, attention_mask, token_type_ids=None, structural_features=None
    ):
        """
        Forward pass of the model.

        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor indicating non-padding tokens
            token_type_ids: Tensor indicating segment ids (optional)
            structural_features: Tensor of shape (Batch, Num_Features)

        Returns:
            logits: Tensor of shape (Batch, Num_Classes)
        """
        # Pass through Transformer Backbone
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = outputs.last_hidden_state

        # Apply Attention Pooling
        pooled_output = self.pooler(last_hidden_state, attention_mask)

        # Feature Fusion
        if structural_features is not None:
            # Concatenate semantic embedding with structural features
            pooled_output = torch.cat([pooled_output, structural_features], dim=1)

        # Classification Head
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)

        return logits
