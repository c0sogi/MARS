import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.configuration import Config


class TransformerModel(nn.Module):
    def __init__(self, model_name, dropout=0.2, freeze_layers=0):
        """
        Transformer Model with Mean Pooling and Custom Head.

        Args:
            model_name (str): HuggingFace model identifier.
            dropout (float): Dropout probability.
            freeze_layers (int): Number of bottom encoder layers to freeze (including embeddings).
        """
        super(TransformerModel, self).__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Apply Layer Freezing
        self._freeze_layers(freeze_layers)

        # Classification Head
        # Mean Pooling does not change dimension (hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the linear layer
        self._init_weights(self.fc)

    def _freeze_layers(self, num_layers):
        """
        Freezes the embeddings and the bottom N encoder layers.
        """
        if num_layers <= 0:
            return

        for name, param in self.backbone.named_parameters():
            # Freeze Embeddings
            if "embeddings" in name:
                param.requires_grad = False

            # Freeze Encoder Layers
            # Structure is usually model.encoder.layer.X...
            elif "encoder.layer" in name:
                # Extract the layer index
                parts = name.split(".")
                layer_idx = -1

                # Find the index following the word 'layer'
                for i, part in enumerate(parts):
                    if part == "layer" and i + 1 < len(parts):
                        if parts[i + 1].isdigit():
                            layer_idx = int(parts[i + 1])
                            break

                # If layer index found and is within the freeze range
                if layer_idx != -1 and layer_idx < num_layers:
                    param.requires_grad = False

    def _init_weights(self, module):
        """
        Initialize weights for the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass with Mean Pooling.

        Args:
            input_ids (torch.Tensor): (Batch, SeqLen)
            attention_mask (torch.Tensor): (Batch, SeqLen)

        Returns:
            torch.Tensor: Logits (Batch, 1)
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (Batch, SeqLen, HiddenDim)

        # Mean Pooling
        # Expand attention mask to match hidden state dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings (masking out padding)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum mask (count non-padding tokens), clamp to avoid division by zero
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Average
        mean_embeddings = sum_embeddings / sum_mask  # (Batch, HiddenDim)

        # Classification Head
        x = self.dropout(mean_embeddings)
        logits = self.fc(x)

        return logits
