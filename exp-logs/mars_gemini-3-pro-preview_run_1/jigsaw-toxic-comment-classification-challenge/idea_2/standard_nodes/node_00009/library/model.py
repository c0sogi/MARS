import torch
import torch.nn as nn
from transformers import RobertaModel
from library.config import Config


class LinearAttentionPooling(nn.Module):
    """
    Implementation of Linear Attention Pooling.
    Computes a weighted sum of hidden states using learned attention weights.
    """

    def __init__(self, hidden_size):
        super(LinearAttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_len)
        Returns:
            context_vector: Tensor of shape (batch_size, hidden_size)
        """
        # Calculate attention scores: (batch_size, seq_len, 1)
        scores = self.attention(last_hidden_state)

        # Squeeze to (batch_size, seq_len)
        scores = scores.squeeze(-1)

        # Mask padding tokens (where attention_mask is 0) with a large negative value
        # so they result in ~0 weight after softmax.
        scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Calculate softmax weights: (batch_size, seq_len)
        weights = torch.softmax(scores, dim=-1)

        # Expand weights for broadcasting: (batch_size, seq_len, 1)
        weights = weights.unsqueeze(-1)

        # Compute weighted sum: (batch_size, hidden_size)
        context_vector = torch.sum(last_hidden_state * weights, dim=1)

        return context_vector


class ToxicityRoBERTa(nn.Module):
    """
    Context-Aware Transformer with Hybrid Signal Aggregation.
    Backbone: RoBERTa-base
    Aggregation: Concatenation of Linear Attention Pooling and Global Max Pooling.
    """

    def __init__(self):
        super(ToxicityRoBERTa, self).__init__()

        # Load pre-trained RoBERTa backbone
        self.roberta = RobertaModel.from_pretrained(Config.MODEL_NAME)

        # Initialize custom pooling layer
        self.attention_pooling = LinearAttentionPooling(Config.HIDDEN_SIZE)

        # Dropout for regularization
        self.dropout = nn.Dropout(Config.DROPOUT)

        # Classification head
        # Input dimension is doubled because we concatenate two pooling outputs
        self.classifier = nn.Linear(Config.HIDDEN_SIZE * 2, Config.NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
        Returns:
            logits: Tensor of shape (batch_size, num_labels)
        """
        # Pass through RoBERTa
        # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 1. Linear Attention Pooling
        attn_pool = self.attention_pooling(last_hidden_state, attention_mask)

        # 2. Global Max Pooling
        # We must mask padding tokens to ensure they are not selected as the max value.
        # Create a mask with the same shape as hidden state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        )

        # Clone hidden state to avoid in-place modification issues
        hidden_masked = last_hidden_state.clone()
        # Set padding positions to a very small number
        hidden_masked = hidden_masked.masked_fill(input_mask_expanded == 0, -1e9)

        # Perform max pooling over the sequence dimension (dim=1)
        max_pool = torch.max(hidden_masked, dim=1)[0]

        # 3. Concatenate Representations
        combined_vector = torch.cat((attn_pool, max_pool), dim=1)

        # 4. Classification Head
        logits = self.classifier(self.dropout(combined_vector))

        return logits
