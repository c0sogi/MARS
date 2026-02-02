import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MeanPooling(nn.Module):
    """
    Mean Pooling - Averaging the last hidden states of the transformer backbone,
    taking the attention mask into account to ignore padding tokens.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # Expand attention mask to match hidden state dimensions: [batch, seq_len, hidden]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum hidden states of valid tokens
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum mask to get count of valid tokens
        sum_mask = input_mask_expanded.sum(1)

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Compute mean
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class RegressionHead(nn.Module):
    """
    Custom Head for Regression.
    Consists of MeanPooling followed by a Linear layer.
    """

    def __init__(self, hidden_size, num_labels):
        super(RegressionHead, self).__init__()
        self.pooler = MeanPooling()
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden_state, attention_mask):
        feature = self.pooler(last_hidden_state, attention_mask)
        logits = self.fc(feature)
        return logits


class EssayModel(nn.Module):
    """
    Regression Model using DeBERTa backbone.
    Predicts a single continuous score.
    """

    def __init__(
        self,
        model_name: str = Config.model_name,
        num_labels: int = Config.num_labels,
        pretrained: bool = True,
    ):
        super(EssayModel, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Apply dropout settings from Config
        self.config.hidden_dropout_prob = Config.hidden_dropout_prob
        self.config.attention_probs_dropout_prob = Config.attention_probs_dropout_prob

        # Initialize Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Initialize Regression Head
        # Maps hidden_size -> num_labels (1)
        self.head = RegressionHead(self.config.hidden_size, num_labels)

        # Initialize weights for the new head
        self._init_weights(self.head.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the specific module (Linear layer).
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids: Tensor of token ids.
            attention_mask: Tensor of attention masks.

        Returns:
            logits: Raw output from the linear layer [batch_size, num_labels].
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Get last hidden states
        last_hidden_state = outputs.last_hidden_state

        # Pass through Head (Pooling + Linear)
        logits = self.head(last_hidden_state, attention_mask)

        return logits
