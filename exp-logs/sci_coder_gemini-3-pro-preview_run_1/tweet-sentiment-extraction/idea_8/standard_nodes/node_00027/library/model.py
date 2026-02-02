import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling strategy.
    Aggregates the hidden states from the last N layers using learnable weights.
    """

    def __init__(self, num_hidden_layers=4):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        # Initialize learnable weights for the layers
        self.layer_weights = nn.Parameter(torch.ones(num_hidden_layers))

    def forward(self, all_hidden_states):
        """
        Args:
            all_hidden_states (tuple): Tuple of tensors containing hidden states from the backbone.
                                       Shape of each tensor: (batch_size, seq_len, hidden_size)
        Returns:
            torch.Tensor: Weighted average of the selected hidden states.
                          Shape: (batch_size, seq_len, hidden_size)
        """
        # Select the last N layers
        # all_hidden_states usually includes embeddings as the first element, so we check the length
        # But standard HF output_hidden_states includes all layers + embedding.
        # We just take the last 'num_hidden_layers' elements.
        selected_layers = all_hidden_states[-self.num_hidden_layers :]

        # Stack them: (num_layers, batch_size, seq_len, hidden_size)
        stacked_layers = torch.stack(selected_layers)

        # Compute softmax weights: (num_layers, 1, 1, 1) for broadcasting
        weights = torch.softmax(self.layer_weights, dim=0).view(
            self.num_hidden_layers, 1, 1, 1
        )

        # Weighted sum
        weighted_output = torch.sum(weights * stacked_layers, dim=0)

        return weighted_output


class TweetModel(nn.Module):
    """
    DeBERTa-v3 for Tweet Sentiment Extraction.
    Uses Weighted Layer Pooling and a CNN-enhanced head for direct span prediction.
    """

    def __init__(self):
        super(TweetModel, self).__init__()
        self.config = Config

        # 1. Load Backbone
        model_config = AutoConfig.from_pretrained(self.config.MODEL_NAME)
        model_config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(
            self.config.MODEL_NAME, config=model_config
        )

        # 2. Weighted Layer Pooling (Last 4 layers)
        self.pooling = WeightedLayerPooling(num_hidden_layers=4)

        # 3. Primary Span Head Components
        # Input: Pooled Embeddings (Hidden)

        # 1D Convolution to capture local context boundaries
        # Kernel size 3, padding 1 preserves sequence length
        self.conv = nn.Conv1d(
            in_channels=self.config.HIDDEN_SIZE,
            out_channels=self.config.HIDDEN_SIZE,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()

        # Final projection to Start/End logits
        self.span_head = nn.Linear(self.config.HIDDEN_SIZE, 2)

        # Regularization
        self.dropout = nn.Dropout(self.config.DROPOUT)

        # Initialize weights for new layers
        self._init_weights(self.conv)
        self._init_weights(self.span_head)

    def _init_weights(self, module):
        """Standard Hugging Face weight initialization."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.backbone.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.
        """
        # 1. Backbone Forward
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 2. Pooling
        # outputs.hidden_states contains all layers
        pooled_output = self.pooling(outputs.hidden_states)
        pooled_output = self.dropout(pooled_output)  # (Batch, Seq, Hidden)

        # 3. Span Prediction (Conv1d)
        # Conv1d expects (Batch, Channels, Seq), so we permute
        # (Batch, Seq, Hidden) -> (Batch, Hidden, Seq)
        conv_input = pooled_output.permute(0, 2, 1)

        conv_output = self.conv(conv_input)  # (Batch, Hidden, Seq)
        conv_output = self.relu(conv_output)

        # Permute back for Linear layer
        conv_output = conv_output.permute(0, 2, 1)  # (Batch, Seq, Hidden)

        # 4. Final Projection
        logits = self.span_head(conv_output)  # (Batch, Seq, 2)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (Batch, Seq)
        end_logits = end_logits.squeeze(-1)  # (Batch, Seq)

        return start_logits, end_logits
