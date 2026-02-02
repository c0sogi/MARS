import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MeanPooling(nn.Module):
    """
    Mean Pooling strategy to aggregate token embeddings.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state (torch.Tensor): Output from the transformer backbone (batch, seq_len, hidden_size).
            attention_mask (torch.Tensor): Attention mask (batch, seq_len).

        Returns:
            torch.Tensor: Pooled embeddings (batch, hidden_size).
        """
        # Expand attention mask to match hidden state dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings while masking padding tokens
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum mask values to get token counts (clamp to avoid division by zero)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class InsultModel(nn.Module):
    """
    Transformer-based model for Insult Detection with Mean Pooling and Multi-Sample Dropout.
    """

    def __init__(self, model_name, config):
        """
        Args:
            model_name (str): Name of the transformer backbone (e.g., 'microsoft/deberta-v3-large').
            config (Config): Configuration object containing hyperparameters.
        """
        super(InsultModel, self).__init__()
        self.config = config

        # Load Configuration to access hidden_size
        self.model_config = AutoConfig.from_pretrained(model_name)

        # Load Backbone
        self.backbone = AutoModel.from_pretrained(model_name, config=self.model_config)

        # Enable Gradient Checkpointing for memory efficiency if available
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        # Pooling Layer
        self.pooler = MeanPooling()

        # Multi-Sample Dropout (MSD)
        # Create multiple dropout layers with different masks
        self.dropouts = nn.ModuleList(
            [nn.Dropout(config.dropout_rate) for _ in range(config.num_msd)]
        )

        # Classification Head
        self.fc = nn.Linear(self.model_config.hidden_size, config.num_classes)

        # Initialize weights for the classification head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the linear layer.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits (batch, num_classes).
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply Mean Pooling
        pooled_output = self.pooler(last_hidden_state, attention_mask)

        # Apply Multi-Sample Dropout
        # Pass pooled output through each dropout layer and then the shared FC layer
        logits_list = []
        for dropout in self.dropouts:
            logits_list.append(self.fc(dropout(pooled_output)))

        # Average the predictions across the multiple dropout samples
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)

        return logits
