import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class CustomModel(nn.Module):
    """
    Custom DeBERTa-v3-large model for Semantic Similarity Regression.
    Features a Multi-Sample Dropout (MSD) head for improved generalization.
    """

    def __init__(self, cfg: Config, config_path=None, pretrained=False):
        """
        Initializes the model architecture.

        Args:
            cfg (Config): Configuration object containing model settings.
            config_path (str, optional): Path to a saved configuration file.
            pretrained (bool): Whether to load pre-trained backbone weights.
        """
        super().__init__()
        self.cfg = cfg

        # Load Configuration
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                cfg.model_name, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)

        # Load Backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing if configured (saves memory)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Multi-Sample Dropout (MSD) Head
        # Instantiates multiple dropout layers to create an ensemble effect within the network
        self.dropouts = nn.ModuleList(
            [nn.Dropout(self.cfg.msd_dropout) for _ in range(self.cfg.msd_samples)]
        )

        # Regression Output Layer
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the new head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initializes weights for the custom head using standard transformer initialization.
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

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs (segment IDs).

        Returns:
            torch.Tensor: Predicted similarity scores of shape (batch_size,).
        """
        # Pass inputs through the backbone
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract [CLS] token embedding (index 0)
        last_hidden_state = outputs.last_hidden_state
        cls_embeddings = last_hidden_state[:, 0, :]

        if self.cfg.use_msd and self.training:
            # Multi-Sample Dropout Logic
            # Pass embeddings through each dropout mask, project, and then average
            output_list = []
            for dropout in self.dropouts:
                output_list.append(self.fc(dropout(cls_embeddings)))

            # Stack outputs: (n_samples, batch_size, 1)
            stacked_outputs = torch.stack(output_list, dim=0)
            # Average across samples: (batch_size, 1)
            output = torch.mean(stacked_outputs, dim=0)
        else:
            # Standard inference (Dropout is identity in eval mode)
            output = self.fc(cls_embeddings)

        # Squeeze to return shape (batch_size,) matching the labels
        return output.squeeze(-1)
