import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout (MSD).
    Applies multiple dropout masks to the input and passes each through the same
    linear layer, averaging the results. This acts as a strong regularizer.
    """

    def __init__(self, dropout_rates, input_dim, output_dim):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(rate) for rate in dropout_rates])
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, hidden_dim)
        # Apply first dropout and classifier
        logits = self.classifier(self.dropouts[0](x))

        # Accumulate results from remaining dropouts
        for i in range(1, len(self.dropouts)):
            logits += self.classifier(self.dropouts[i](x))

        # Average the logits
        return logits / len(self.dropouts)


class MultiTaskRoBERTa(nn.Module):
    """
    RoBERTa-based model for Toxicity Classification with Multi-Task Learning.

    Architecture:
    1. RoBERTa Backbone (roberta-base)
    2. Extraction of [CLS] token embedding
    3. Multi-Sample Dropout
    4. Single Linear Projection to 7 outputs (1 Main + 6 Aux)
    """

    def __init__(self, pretrained=True):
        super().__init__()

        config_name = Config.MODEL_NAME

        # Load configuration
        self.config = AutoConfig.from_pretrained(config_name)

        # Load backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(config_name, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        self.hidden_size = self.config.hidden_size

        # Initialize Multi-Sample Dropout Head
        # Projects [CLS] embedding to TOTAL_OUTPUTS (Target + Aux Tasks)
        self.multi_sample_dropout = MultiSampleDropout(
            dropout_rates=Config.DROPOUT_RATES,
            input_dim=self.hidden_size,
            output_dim=Config.TOTAL_OUTPUTS,
        )

        # Initialize weights for the classifier head
        self._init_weights(self.multi_sample_dropout.classifier)

    def _init_weights(self, module):
        """
        Initialize the weights of the linear layer using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size, TOTAL_OUTPUTS).
                          Index 0 is the main toxicity target.
                          Indices 1-6 are the auxiliary toxicity subtypes.
        """
        # Pass through RoBERTa backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token embedding (first token of the last hidden state)
        # Shape: (batch_size, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]

        # Pass through Multi-Sample Dropout and Projection Layer
        logits = self.multi_sample_dropout(cls_embedding)

        return logits
