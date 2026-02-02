import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch_scatter import scatter_mean
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer that computes a weighted average of token embeddings.
    Allows the model to dynamically focus on important tokens (e.g., non-padding, content words).
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: (Batch_Size, Seq_Len, Hidden_Size)
            attention_mask: (Batch_Size, Seq_Len)
        Returns:
            pooled_output: (Batch_Size, Hidden_Size)
        """
        # Compute attention scores
        w = self.attention(last_hidden_state)  # (B, L, 1)
        weights = w.squeeze(-1)  # (B, L)

        # Mask padding tokens so they don't contribute to the average
        # Using a large negative number for softmax stability
        weights.masked_fill_(attention_mask == 0, -1e9)

        # Normalize weights
        scores = torch.softmax(weights, dim=-1)  # (B, L)

        # Compute weighted sum of embeddings
        # scores: (B, L) -> (B, 1, L)
        # last_hidden_state: (B, L, H)
        # bmm output: (B, 1, H)
        weighted_embeddings = torch.bmm(scores.unsqueeze(1), last_hidden_state)

        return weighted_embeddings.squeeze(1)


class EssayModel(nn.Module):
    """
    Main model class for Essay Scoring.

    Flow:
    1. Input (Chunks) -> DeBERTa Backbone -> Token Embeddings
    2. Token Embeddings -> Attention Pooling -> Chunk Embeddings
    3. Chunk Embeddings -> Scatter Mean (Aggregation) -> Essay Embeddings
    4. Essay Embeddings -> Linear Head -> Score
    """

    def __init__(self):
        super().__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)

        # Update config for efficiency and specific task requirements
        self.config.update(
            {
                "output_hidden_states": False,
                "hidden_dropout_prob": 0.0,  # Disable dropout for regression stability
                "attention_probs_dropout_prob": 0.0,
                "add_pooling_layer": False,
            }
        )

        # Initialize Backbone
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Enable Gradient Checkpointing to save memory
        # This is crucial for training 'large' models on limited GPU memory
        self.backbone.gradient_checkpointing_enable()

        # Pooling Layer
        self.pooling = AttentionPooling(self.config.hidden_size)

        # Regression Head
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the custom head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the linear head using a normal distribution.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, batch_ids=None, labels=None):
        """
        Args:
            input_ids: (Total_Chunks, Seq_Len)
            attention_mask: (Total_Chunks, Seq_Len)
            batch_ids: (Total_Chunks,) - Maps each chunk to its original sample index in the batch.
            labels: Optional, not used inside forward but kept for API consistency.

        Returns:
            dict: {
                "logits": (Batch_Size, 1),
                "embeddings": (Batch_Size, Hidden_Size)
            }
        """
        # 1. Backbone Forward Pass
        # Processes all chunks in parallel as a single large batch
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = (
            outputs.last_hidden_state
        )  # (Total_Chunks, Seq_Len, Hidden_Size)

        # 2. Pooling
        # Aggregates token embeddings into a single vector per chunk
        chunk_embeddings = self.pooling(
            last_hidden_state, attention_mask
        )  # (Total_Chunks, Hidden_Size)

        # 3. Aggregation (Chunk -> Document)
        if batch_ids is not None:
            # Aggregate chunks belonging to the same essay using mean
            # batch_ids maps chunks to the sample index (0 to Batch_Size-1)
            # Result shape: (Batch_Size, Hidden_Size)
            sample_embeddings = scatter_mean(chunk_embeddings, batch_ids, dim=0)
        else:
            # Fallback if no batch_ids provided (e.g., if input is not chunked)
            sample_embeddings = chunk_embeddings

        # 4. Regression Head
        logits = self.fc(sample_embeddings)  # (Batch_Size, 1)

        return {"logits": logits, "embeddings": sample_embeddings}
