import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Computes a weighted average of token embeddings using a learned attention mechanism.
    Allows the model to focus on specific tokens (e.g., nouns, modifiers) rather than
    treating all tokens equally or relying solely on the [CLS] token.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, hidden_size)
        """
        # Calculate attention scores
        # w: (batch_size, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Mask padding tokens so they don't contribute to the softmax
        # attention_mask is 1 for valid tokens, 0 for padding.
        # We create a mask where 0 becomes -inf.
        mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        w = w.masked_fill(mask == 0, -1e9)

        # Normalize scores to sum to 1
        w = torch.softmax(w, dim=1)

        # Compute weighted sum of hidden states
        # (batch_size, seq_len, 1) * (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        # Sum over seq_len -> (batch_size, hidden_size)
        pooled_output = torch.sum(w * last_hidden_state, dim=1)

        return pooled_output


class CustomModel(nn.Module):
    """
    Deberta-V3-Large Cross-Encoder with Attention Pooling.
    """

    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Enable gradient checkpointing to save memory with the Large model
        # self.backbone.gradient_checkpointing_enable()

        # Custom Head components
        self.pooler = AttentionPooling(self.config.hidden_size)
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the custom head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            logits: (batch_size, 1)
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply Attention Pooling
        feature = self.pooler(last_hidden_state, attention_mask)

        # Final Regression Layer
        output = self.fc(feature)

        return output
