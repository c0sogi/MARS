import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import CFG


class AttentionPooling(nn.Module):
    """
    Implementation of Attention Pooling.
    Projects the hidden states to calculate attention weights, then computes
    a weighted average of the hidden states.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len)

        # Calculate weights
        weights = self.attention(last_hidden_state)  # (batch_size, seq_len, 1)
        weights = weights.squeeze(-1)  # (batch_size, seq_len)

        # Mask padding tokens
        # Create a large negative value tensor
        min_value = -1e4
        weights = weights.masked_fill(attention_mask == 0, min_value)

        # Softmax to get probabilities
        scores = torch.softmax(weights, dim=-1).unsqueeze(
            -1
        )  # (batch_size, seq_len, 1)

        # Weighted sum
        # (batch_size, seq_len, hidden_size) * (batch_size, seq_len, 1) -> sum over seq_len
        feature = torch.sum(
            last_hidden_state * scores, dim=1
        )  # (batch_size, hidden_size)

        return feature


class EssayModel(nn.Module):
    """
    DeBERTa-v3-Large based model for essay scoring.
    Handles sliding window inputs by processing chunks and averaging their embeddings.
    """

    def __init__(self, model_name=None, pretrained=True):
        super().__init__()
        if model_name is None:
            model_name = CFG.model_name

        # Load Configuration
        config = AutoConfig.from_pretrained(model_name)
        config.attention_probs_dropout_prob = CFG.attention_probs_dropout_prob
        config.hidden_dropout_prob = CFG.hidden_dropout_prob
        config.output_hidden_states = True  # Ensure we get hidden states

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=config)
        else:
            self.backbone = AutoModel.from_config(config)

        # Gradient Checkpointing
        if CFG.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # Pooling Layer
        self.pool = AttentionPooling(config.hidden_size)

        # Regression Head
        self.fc = nn.Linear(config.hidden_size, CFG.target_size)

        # Initialize weights for custom layers
        self._init_weights(self.pool)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, chunk_mask=None, **kwargs):
        """
        Args:
            input_ids: (batch_size, num_chunks, seq_len)
            attention_mask: (batch_size, num_chunks, seq_len)
            chunk_mask: (batch_size, num_chunks) - 1 for valid chunks, 0 for padded chunks
        """
        bs, n_chunks, seq_len = input_ids.shape

        # Flatten inputs to (batch_size * num_chunks, seq_len) for the backbone
        input_ids_flat = input_ids.view(bs * n_chunks, seq_len)
        attention_mask_flat = attention_mask.view(bs * n_chunks, seq_len)

        # Pass through backbone
        outputs = self.backbone(
            input_ids=input_ids_flat, attention_mask=attention_mask_flat
        )
        last_hidden_state = (
            outputs.last_hidden_state
        )  # (bs * n_chunks, seq_len, hidden_size)

        # Apply Attention Pooling to get chunk embeddings
        chunk_embeddings = self.pool(
            last_hidden_state, attention_mask_flat
        )  # (bs * n_chunks, hidden_size)

        # Reshape back to (batch_size, num_chunks, hidden_size)
        chunk_embeddings = chunk_embeddings.view(bs, n_chunks, -1)

        # Aggregate chunk embeddings to get document embedding
        if chunk_mask is not None:
            # chunk_mask is (batch_size, num_chunks)
            # Expand mask for broadcasting: (batch_size, num_chunks, 1)
            mask = chunk_mask.unsqueeze(-1).float()

            # Sum embeddings of valid chunks
            sum_embeddings = torch.sum(
                chunk_embeddings * mask, dim=1
            )  # (batch_size, hidden_size)

            # Count valid chunks
            sum_mask = torch.sum(mask, dim=1)  # (batch_size, 1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # Prevent division by zero

            # Average
            essay_embedding = sum_embeddings / sum_mask
        else:
            # Simple average if no mask provided (assumes all chunks valid)
            essay_embedding = torch.mean(chunk_embeddings, dim=1)

        # Predict Score
        logits = self.fc(essay_embedding)  # (batch_size, 1)

        return {"logits": logits, "embedding": essay_embedding}
