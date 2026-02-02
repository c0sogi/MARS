import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import ModelConfig


class InsultModel(nn.Module):
    """
    DeBERTa-v3-Large based model for Insult Detection.
    Includes Mean Pooling and specific layer freezing for regularization.
    """

    def __init__(self, config: ModelConfig):
        """
        Args:
            config: ModelConfig object containing hyperparameters like model_name,
                    dropout, and freeze_layers.
        """
        super().__init__()
        self.config = config

        # Load Configuration to access hidden_size
        model_config = AutoConfig.from_pretrained(config.model_name)

        # Load Backbone
        # We use AutoModel to get the raw hidden states
        self.backbone = AutoModel.from_pretrained(
            config.model_name, config=model_config
        )

        # Apply Layer Freezing
        self._freeze_layers()

        # Classification Head
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(model_config.hidden_size, 1)

    def _freeze_layers(self):
        """
        Freezes the embeddings and the bottom N encoder layers based on config.
        """
        if self.config.freeze_layers == 0:
            return

        # 1. Freeze Embeddings
        # DeBERTa V3 usually stores embeddings in 'embeddings' or 'deberta.embeddings'
        # AutoModel usually returns the main model (DebertaV2Model) which has .embeddings
        if hasattr(self.backbone, "embeddings"):
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False

        # 2. Freeze Encoder Layers
        # DeBERTa V3 structure: backbone.encoder.layer is a ModuleList
        if hasattr(self.backbone, "encoder") and hasattr(
            self.backbone.encoder, "layer"
        ):
            layers = self.backbone.encoder.layer
            num_layers_to_freeze = min(self.config.freeze_layers, len(layers))

            for i in range(num_layers_to_freeze):
                for param in layers[i].parameters():
                    param.requires_grad = False
        else:
            print(
                "Warning: Could not find encoder layers to freeze. Check model architecture."
            )

    def mean_pooling(self, last_hidden_state, attention_mask):
        """
        Performs mean pooling on the last hidden state, ignoring padding tokens.

        Args:
            last_hidden_state: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, hidden_size)
        """
        # Expand attention mask to match hidden state dimensions
        # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings (masked)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum mask (count of non-padding tokens)
        sum_mask = input_mask_expanded.sum(1)

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            labels: Optional labels (not used in forward, but kept for signature compatibility).

        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        # 1. Backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 2. Mean Pooling
        feature = self.mean_pooling(last_hidden_state, attention_mask)

        # 3. Head
        feature = self.dropout(feature)
        logits = self.fc(feature)

        return logits
