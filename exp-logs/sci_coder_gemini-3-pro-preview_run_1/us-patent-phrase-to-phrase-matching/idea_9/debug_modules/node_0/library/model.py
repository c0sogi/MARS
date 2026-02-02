import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class CustomModel(nn.Module):
    """
    Custom model class for Patent Phrase Matching using DeBERTa-v3-Large.
    Implements a Multi-Sample Dropout (MSD) head for stabilized regression.
    """

    def __init__(self, pretrained: bool = True):
        """
        Initializes the model architecture.

        Args:
            pretrained (bool): Whether to load pre-trained weights for the backbone.
                               If False, initializes with random weights (useful for debugging/config loading).
        """
        super().__init__()

        # Load configuration from the backbone
        self.config = AutoConfig.from_pretrained(Config.model_name)

        # Initialize backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                Config.model_name, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Enable gradient checkpointing to save memory
        self.backbone.gradient_checkpointing_enable()

        # Multi-Sample Dropout settings
        self.num_msd = Config.num_msd
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.fc_dropout) for _ in range(self.num_msd)]
        )

        # Regression Head
        # The output of DeBERTa-v3-Large hidden states is config.hidden_size
        self.fc = nn.Linear(self.config.hidden_size, Config.target_size)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the specific module.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids=None, label=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs (segment IDs).
            label (torch.Tensor, optional): Ground truth scores for loss calculation.

        Returns:
            dict: Dictionary containing 'logits' and optionally 'loss'.
        """
        # Pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract the [CLS] token representation (first token of the last hidden state)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]

        # Multi-Sample Dropout Head
        # Apply multiple dropout masks and average the predictions
        logits_sum = 0
        for i, dropout in enumerate(self.dropouts):
            logits_sum += self.fc(dropout(cls_embedding))

        logits = logits_sum / self.num_msd

        # Prepare output
        output = {"logits": logits}

        # Calculate loss if labels are provided
        if label is not None:
            # Flatten logits to match label shape [batch_size]
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), label.view(-1))
            output["loss"] = loss

        return output
