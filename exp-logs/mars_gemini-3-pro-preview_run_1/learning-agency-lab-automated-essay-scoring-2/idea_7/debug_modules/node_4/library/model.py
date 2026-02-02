import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer that dynamically weights the input sequence tokens.
    This allows the model to focus on critical parts of the essay (sentences/phrases)
    when forming the final representation, rather than treating all tokens equally
    (as in MeanPooling) or only taking the [CLS] token.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_len)
        Returns:
            context_vector: Tensor of shape (batch_size, hidden_size)
        """
        # Calculate raw attention scores
        # w shape: (batch_size, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Mask padding tokens so they don't contribute to the softmax
        # attention_mask is 1 for tokens, 0 for padding.
        # We set padding positions to a very large negative number (-1e4)
        w = w.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4)

        # Apply softmax to get normalized attention weights
        # weights shape: (batch_size, seq_len, 1)
        weights = torch.softmax(w, dim=1)

        # Compute weighted sum of hidden states
        # (batch_size, seq_len, 1) * (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        # Summing over seq_len dimension -> (batch_size, hidden_size)
        context_vector = torch.sum(weights * last_hidden_state, dim=1)

        return context_vector


class CustomModel(nn.Module):
    """
    The main model architecture wrapping the DeBERTa backbone with a custom
    Attention Pooling layer and a Regression Head.
    """

    def __init__(self, model_name=Config.model_name, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)

        # Load the transformer backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing to save memory during training
        # This is crucial for fitting 'Large' models with long sequences (1024) into GPU memory
        # Cite debug_lesson_3: Distinguish Preconditions from Runtime Errors in Gradient Checkpointing
        # Disabling gradient checkpointing to resolve "RuntimeError: Trying to backward through the graph a second time"
        # if pretrained:
        #     self.backbone.gradient_checkpointing_enable()
        #     self.backbone.enable_input_require_grads()

        # Initialize Custom Pooling Layer
        self.pooling = AttentionPooling(self.config.hidden_size)

        # Initialize Regression Head (Linear Layer)
        # Projects hidden size to a single scalar score
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the new layers
        self._init_weights(self.fc)
        self._init_weights(self.pooling)

    def _init_weights(self, module):
        """
        Initialize weights for Linear and LayerNorm layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            output: (batch_size, 1) - The predicted scalar score
        """
        # Pass inputs through the backbone
        # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply Attention Pooling to aggregate sequence into a single vector
        # feature shape: (batch_size, hidden_size)
        feature = self.pooling(last_hidden_state, attention_mask)

        # Pass through regression head
        # output shape: (batch_size, 1)
        output = self.fc(feature)

        return output
