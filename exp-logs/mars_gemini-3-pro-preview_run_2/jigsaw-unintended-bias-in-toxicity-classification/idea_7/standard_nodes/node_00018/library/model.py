import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops entire channels across the sequence dimension.
    Input shape: (Batch, Seq_Len, Hidden_Dim)
    """

    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob
        # Dropout2d expects (N, C, H, W). We will map Hidden_Dim to C.
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        # x: (Batch, Seq_Len, Hidden_Dim)
        # Permute to (Batch, Hidden_Dim, Seq_Len)
        x = x.permute(0, 2, 1)
        # Unsqueeze to (Batch, Hidden_Dim, Seq_Len, 1) to fit Dropout2d
        x = x.unsqueeze(3)
        x = self.dropout(x)
        # Squeeze and permute back
        x = x.squeeze(3)
        x = x.permute(0, 2, 1)
        return x


class AttentionPooling(nn.Module):
    """
    Computes a weighted average of the hidden states based on learnable attention scores.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (Batch, Seq_Len, Hidden_Dim)
        # attention_mask: (Batch, Seq_Len)

        # Calculate attention weights: (Batch, Seq_Len, 1)
        w = self.attention(last_hidden_state)

        # Apply mask to ignore padding tokens (set their weight to -inf)
        # attention_mask is 1 for tokens, 0 for padding.
        if attention_mask is not None:
            # Expand mask to (Batch, Seq_Len, 1)
            mask = attention_mask.unsqueeze(-1)
            w = w.masked_fill(mask == 0, -1e9)

        # Softmax to get probabilities
        w = torch.softmax(w, dim=1)

        # Weighted sum: (Batch, Hidden_Dim)
        # Sum over sequence dimension
        c = torch.sum(last_hidden_state * w, dim=1)
        return c


class MultiTaskRoBERTa(nn.Module):
    """
    RoBERTa-Large model with Spatial Dropout, Attention Pooling,
    and Multi-Task heads (Toxicity + Identity).
    """

    def __init__(self, config=Config):
        super(MultiTaskRoBERTa, self).__init__()
        self.config = config

        # Load Pre-trained Transformer
        # We use AutoConfig to ensure we match the architecture details
        transformer_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        transformer_config.output_hidden_states = True
        self.roberta = AutoModel.from_pretrained(
            config.MODEL_NAME, config=transformer_config
        )

        # Custom Layers
        self.spatial_dropout = SpatialDropout(config.DROPOUT)
        self.pooling = AttentionPooling(config.HIDDEN_SIZE)

        # Task Heads
        # Primary Head: Toxicity (1 output)
        self.toxicity_head = nn.Linear(config.HIDDEN_SIZE, config.NUM_CLASSES)

        # Auxiliary Head: Identity Attributes (NUM_AUX_CLASSES outputs)
        self.identity_head = nn.Linear(config.HIDDEN_SIZE, config.NUM_AUX_CLASSES)

        # Initialize weights for custom layers
        self._init_weights(self.toxicity_head)
        self._init_weights(self.identity_head)

    def _init_weights(self, module):
        """
        Initialize the weights of the custom heads.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.MODEL_NAME == "roberta-large" and 0.02 or 0.02
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        Returns:
            toxicity_logit: (Batch, 1)
            identity_logits: (Batch, Num_Identities)
        """
        # Transformer Output
        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        # We use the last hidden state sequence: (Batch, Seq_Len, Hidden_Dim)
        last_hidden_state = outputs.last_hidden_state

        # Apply Spatial Dropout
        embeddings = self.spatial_dropout(last_hidden_state)

        # Apply Attention Pooling
        pooled_output = self.pooling(embeddings, attention_mask)

        # Prediction Heads
        toxicity_logit = self.toxicity_head(pooled_output)
        identity_logits = self.identity_head(pooled_output)

        return toxicity_logit, identity_logits
