import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CustomModel(nn.Module):
    """
    Custom Semantic Similarity Model based on DeBERTa-v3-Large.

    Features:
    - Backbone: microsoft/deberta-v3-large
    - Head: Multi-Sample Dropout (MSD) for regression
    - Loss: Mean Squared Error (MSE)
    """

    def __init__(self, cfg=Config, pretrained=True):
        """
        Args:
            cfg: Configuration class with model settings.
            pretrained (bool): Whether to load pre-trained backbone weights.
        """
        super().__init__()
        self.cfg = cfg

        # Load Configuration for the backbone
        self.model_config = AutoConfig.from_pretrained(cfg.model_name)
        self.model_config.update(
            {
                "output_hidden_states": False,
                "hidden_dropout_prob": 0.0,  # Disable standard dropout in backbone
                "attention_probs_dropout_prob": 0.0,
                "num_labels": cfg.num_classes,
            }
        )

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                cfg.model_name, config=self.model_config
            )
        else:
            self.backbone = AutoModel.from_config(self.model_config)

        # Gradient Checkpointing (Optional, helps with memory on Large models)
        # self.backbone.gradient_checkpointing_enable()

        # Multi-Sample Dropout (MSD) Head
        # We use a ModuleList of Dropout layers and a single Linear layer
        self.dropouts = nn.ModuleList(
            [nn.Dropout(cfg.msd_dropout) for _ in range(cfg.msd_samples)]
        )

        # Regression Head
        self.fc = nn.Linear(self.model_config.hidden_size, cfg.num_classes)

        # Initialize weights of the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the specific module using standard Transformer initialization.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor, optional): Target similarity scores.

        Returns:
            dict: Dictionary containing 'logits' and optionally 'loss'.
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] representation (index 0)
        # Shape: (batch_size, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]

        # Multi-Sample Dropout
        # Apply multiple dropout masks and average the predictions
        logits_list = []
        for dropout in self.dropouts:
            x = dropout(cls_embedding)
            logits_list.append(self.fc(x))

        # Stack and average
        # Shape: (batch_size, num_classes)
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)

        # Squeeze if regression (num_classes=1) to match label shape (batch_size,)
        if self.cfg.num_classes == 1:
            logits = logits.squeeze(-1)

        output = {"logits": logits}

        # Calculate Loss if labels are provided
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
            output["loss"] = loss

        return output
