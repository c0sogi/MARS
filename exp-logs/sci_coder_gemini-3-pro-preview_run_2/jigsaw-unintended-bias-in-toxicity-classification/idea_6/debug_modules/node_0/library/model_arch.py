import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops entire channels (feature dimensions) across the full sequence length
    instead of dropping individual elements. This is effective for NLP tasks to prevent
    co-adaptation of feature maps.
    """

    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, hidden_size)
        if not self.training or self.drop_prob == 0:
            return inputs

        # Permute to (batch_size, hidden_size, seq_len, 1) to use dropout2d
        # dropout2d expects (N, C, H, W)
        inputs = inputs.permute(0, 2, 1).unsqueeze(3)

        # Apply dropout
        inputs = F.dropout2d(inputs, self.drop_prob, training=self.training)

        # Restore shape: (batch_size, seq_len, hidden_size)
        inputs = inputs.squeeze(3).permute(0, 2, 1)

        return inputs


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Computes a weighted average of hidden states where weights are learned dynamically.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len) - 1 for token, 0 for pad

        # Calculate raw attention scores
        # w shape: (batch_size, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Mask padding positions so they don't contribute to the softmax
        # Expand mask to match w dimensions
        mask = attention_mask.unsqueeze(-1).float()

        # Set padding scores to a very large negative number
        w = w.masked_fill(mask == 0, -1e9)

        # Compute attention weights via softmax
        att_weights = F.softmax(w, dim=1)

        # Compute weighted sum (context vector)
        # shape: (batch_size, hidden_size)
        context_vector = torch.sum(att_weights * last_hidden_state, dim=1)

        return context_vector


class ToxicityModel(nn.Module):
    """
    Multi-Task RoBERTa-Large Model for Toxicity Classification.

    Structure:
    1. RoBERTa-Large Backbone
    2. Spatial Dropout
    3. Attention Pooling
    4. Two Output Heads:
       - Toxicity Head (Primary): Predicts toxicity score.
       - Identity Head (Auxiliary): Predicts presence of identity attributes.
    """

    def __init__(self, checkpoint=Config.MODEL_NAME):
        super(ToxicityModel, self).__init__()

        # Load Transformer Configuration and Backbone
        self.config = AutoConfig.from_pretrained(checkpoint)
        self.backbone = AutoModel.from_pretrained(checkpoint, config=self.config)

        # Feature processing layers
        self.spatial_dropout = SpatialDropout(Config.DROPOUT)
        self.pooling = AttentionPooling(self.config.hidden_size)

        # Output Heads
        # Primary task: Binary classification (Toxic vs Non-Toxic) -> 1 output
        self.toxicity_head = nn.Linear(self.config.hidden_size, 1)

        # Auxiliary task: Multi-label classification for identities -> Num Identity Cols
        self.identity_head = nn.Linear(self.config.hidden_size, Config.NUM_AUX_CLASSES)

        # Initialize weights for new heads
        self._init_weights(self.toxicity_head)
        self._init_weights(self.identity_head)

    def _init_weights(self, module):
        """Initialize weights for the classification heads."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Token indices (Batch, Seq_Len)
            attention_mask (torch.Tensor): Attention masks (Batch, Seq_Len)

        Returns:
            toxicity_logits (torch.Tensor): Logits for toxicity (Batch, 1)
            identity_logits (torch.Tensor): Logits for identities (Batch, Num_Aux)
        """
        # Get backbone outputs
        # We use the sequence of hidden states, not just the [CLS] token
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # Shape: (Batch, Seq, Hidden)

        # Apply Spatial Dropout to the sequence
        embeddings = self.spatial_dropout(last_hidden_state)

        # Aggregate sequence into a single vector using Attention Pooling
        pooled_output = self.pooling(embeddings, attention_mask)

        # Pass through classification heads
        toxicity_logits = self.toxicity_head(pooled_output)
        identity_logits = self.identity_head(pooled_output)

        return toxicity_logits, identity_logits
