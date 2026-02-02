import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MeanPooling(nn.Module):
    """
    Performs Mean Pooling on the last hidden state of the transformer.
    Takes the attention mask into account to ignore padding tokens.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # Expand mask to match hidden state dimensions: (batch, seq_len, hidden_size)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings across the sequence length
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum the mask values to get the count of valid tokens
        sum_mask = input_mask_expanded.sum(1)

        # Clamp to avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class InsultModel(nn.Module):
    """
    DeBERTa-v3-Large based model for Insult Detection.
    Features:
    - Pre-trained Backbone
    - Mean Pooling
    - Multi-Sample Dropout Head
    """

    def __init__(self, pretrained=True):
        super(InsultModel, self).__init__()

        # Load configuration
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.config.output_hidden_states = False

        # Load backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(
                Config.model_name, config=self.config
            )
        else:
            self.model = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing to save memory with Large models
        # self.model.gradient_checkpointing_enable()

        # Feature dimension
        self.hidden_size = self.config.hidden_size

        # Pooling Layer
        self.pool = MeanPooling()

        # Multi-Sample Dropout: Create multiple dropout layers with different probabilities
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in Config.dropout_rates])

        # Classification Head
        self.fc = nn.Linear(self.hidden_size, Config.num_classes)

        # Initialize weights for the new head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, target=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            target (torch.Tensor, optional): Targets (not used in forward, but kept for signature compatibility).

        Returns:
            torch.Tensor: Logits (averaged across dropout samples).
        """
        # 1. Backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 2. Pooling
        feature = self.pool(last_hidden_state, attention_mask)

        # 3. Multi-Sample Dropout & Classification
        # Pass the feature through each dropout layer and then the shared FC layer
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                output = self.fc(dropout(feature))
            else:
                output += self.fc(dropout(feature))

        # Average the predictions
        output /= len(self.dropouts)

        return output
