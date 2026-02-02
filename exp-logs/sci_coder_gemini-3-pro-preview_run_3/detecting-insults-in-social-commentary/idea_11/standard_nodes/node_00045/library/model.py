import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class InsultModel(nn.Module):
    """
    InsultModel class implementing a transformer-based architecture for insult detection.
    Supports RoBERTa and DeBERTa backbones with Mean Pooling and Layer Freezing.
    """

    def __init__(self, model_name, config=None, pretrained=True):
        """
        Args:
            model_name (str): Name of the transformer model to load (e.g., 'roberta-large').
            config (Config, optional): Configuration object. Defaults to None (creates new Config).
            pretrained (bool): Whether to load pretrained weights. Defaults to True.
        """
        super().__init__()
        self.config = config if config is not None else Config()
        self.model_name = model_name

        # Load AutoConfig to determine hidden size
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.output_hidden_states = True
        self.hidden_size = model_config.hidden_size

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=model_config)
        else:
            self.backbone = AutoModel.from_config(model_config)

        # Classification Head
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(self.hidden_size, self.config.num_classes)

        # Initialize Head Weights
        self._init_weights(self.fc)

        # Freeze Layers if specified in Config
        if self.config.freeze_layers > 0:
            self._freeze_layers()

    def _init_weights(self, module):
        """
        Initialize the weights of the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def _freeze_layers(self):
        """
        Freezes the embeddings and the bottom N layers of the encoder
        as specified in config.freeze_layers.
        """
        frozen_layers = 0
        frozen_embeddings = False

        for name, param in self.backbone.named_parameters():
            # Freeze Embeddings
            if "embeddings" in name:
                param.requires_grad = False
                frozen_embeddings = True

            # Freeze Encoder Layers
            # Common structure: model.encoder.layer.{i}. ...
            if "encoder.layer" in name:
                parts = name.split(".")
                for part in parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        if layer_idx < self.config.freeze_layers:
                            param.requires_grad = False
                        break

    def feature(self, input_ids, attention_mask):
        """
        Extracts features from the backbone using Mean Pooling.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Mean Pooling
        # Expand attention mask to match hidden state dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum hidden states masking out padding
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum mask to get count of non-padding tokens
        sum_mask = input_mask_expanded.sum(1)

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        """
        # Get pooled features
        feature = self.feature(input_ids, attention_mask)

        # Apply Dropout
        output = self.dropout(feature)

        # Classification Head
        logits = self.fc(output)

        return logits
