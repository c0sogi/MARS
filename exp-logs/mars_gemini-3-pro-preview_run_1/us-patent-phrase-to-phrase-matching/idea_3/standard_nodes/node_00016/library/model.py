import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from library.config import CFG


class PhraseModel(nn.Module):
    """
    The PhraseModel class wraps the Hugging Face AutoModelForSequenceClassification
    to provide a regression output for the phrase similarity task.
    It uses the DeBERTa-v3-large backbone as defined in the configuration.
    """

    def __init__(self, cfg=CFG, pretrained=True):
        """
        Initializes the model architecture.

        Args:
            cfg: Configuration object containing model settings (model_name, target_size, fc_dropout).
            pretrained (bool): Whether to load pretrained weights. Defaults to True.
        """
        super().__init__()
        self.cfg = cfg

        # Load the configuration from the model name
        self.config = AutoConfig.from_pretrained(cfg.model_name)

        # Configure for regression (1 output label)
        self.config.num_labels = cfg.target_size
        self.config.classifier_dropout = cfg.fc_dropout

        # Initialize the model
        if pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model_name,
                config=self.config,
            )
        else:
            # Initialize with random weights (useful for debugging or custom pretraining)
            self.model = AutoModelForSequenceClassification.from_config(self.config)

        # Enable gradient checkpointing if configured
        if hasattr(cfg, "gradient_checkpointing") and cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs (segment IDs).
            labels (torch.Tensor, optional): Ground truth labels. Accepted for compatibility
                                             but not used for internal loss calculation here.

        Returns:
            torch.Tensor: The predicted scores (logits) of shape (batch_size,).
        """
        # Pass inputs to the Hugging Face model
        # return_dict=True ensures we get a ModelOutput object
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        # Extract logits. Shape is (batch_size, num_labels) -> (batch_size, 1)
        logits = outputs.logits

        # Squeeze the last dimension to get shape (batch_size,)
        # This prepares the output for MSELoss against a 1D target tensor
        return logits.squeeze(-1)
