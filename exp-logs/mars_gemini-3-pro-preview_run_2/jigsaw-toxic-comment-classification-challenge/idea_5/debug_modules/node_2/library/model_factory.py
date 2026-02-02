import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class ToxicityModel(nn.Module):
    """
    Toxicity Classification Model wrapping a Transformer backbone.

    Features:
    - Configurable backbone (DeBERTa-v3-Large, RoBERTa-Large, etc.)
    - Multi-Sample Dropout for improved generalization and faster convergence.
    - Gradient Checkpointing for memory efficiency with Large models.
    """

    def __init__(self, model_name=Config.MODEL_A_NAME, pretrained=True):
        """
        Args:
            model_name (str): Hugging Face model identifier.
            pretrained (bool): Whether to load pre-trained weights.
        """
        super(ToxicityModel, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.return_dict = True

        # Load Backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing to save VRAM
        # Essential for training Large models on GPU
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Multi-Sample Dropout Configuration
        # We use 5 dropout layers with p=0.5
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

        # Classification Head
        self.fc = nn.Linear(self.config.hidden_size, Config.NUM_LABELS)

        # Initialize the classification head weights
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the classification head using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs (segment IDs).

        Returns:
            torch.Tensor: Logits for the 6 toxicity classes.
        """
        # Prepare arguments for the backbone
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Pass token_type_ids only if provided (DeBERTa uses them, RoBERTa usually ignores/doesn't need them)
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids

        # Get backbone outputs
        outputs = self.model(**model_kwargs)

        # Extract the representation of the [CLS] token (first token)
        # Shape: (batch_size, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # Apply Multi-Sample Dropout
        # Pass the CLS embedding through multiple dropout masks and the same linear layer
        final_output = None
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                final_output = self.fc(dropout(cls_output))
            else:
                final_output += self.fc(dropout(cls_output))

        # Average the predictions
        final_output = final_output / len(self.dropouts)

        return final_output
