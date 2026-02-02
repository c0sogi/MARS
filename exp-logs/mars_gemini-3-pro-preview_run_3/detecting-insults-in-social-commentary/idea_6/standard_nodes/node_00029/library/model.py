import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class DebertaV3Classifier(nn.Module):
    """
    DeBERTa-v3-Large based classifier for Insult Detection.

    Architecture:
    - Backbone: microsoft/deberta-v3-large
    - Aggregation: Mean Pooling
    - Head: Dropout -> Linear

    Regularization:
    - Freezes Embeddings and bottom N encoder layers based on Config.
    """

    def __init__(self, pretrained_model_name: str = Config.model_name):
        """
        Initializes the model, loads pretrained weights, and sets up the classification head.

        Args:
            pretrained_model_name (str): HuggingFace model identifier.
        """
        super().__init__()

        # Load Configuration and Backbone Model
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name, config=self.config
        )

        # Apply Layer Freezing Strategy
        self._freeze_layers()

        # Classification Head
        self.dropout = nn.Dropout(Config.dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the custom linear head
        self._init_weights(self.fc)

    def _freeze_layers(self):
        """
        Freezes the gradients of the embeddings and the bottom N layers of the encoder
        to prevent overfitting and improve training efficiency.
        """
        # Freeze Embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        # Freeze Encoder Layers
        # DeBERTa V3 uses 'encoder.layer' as the ModuleList for layers
        if hasattr(self.backbone, "encoder") and hasattr(
            self.backbone.encoder, "layer"
        ):
            for i in range(Config.freeze_layers):
                if i < len(self.backbone.encoder.layer):
                    for param in self.backbone.encoder.layer[i].parameters():
                        param.requires_grad = False
                else:
                    # Safety check if config requests freezing more layers than exist
                    pass

    def _init_weights(self, module):
        """
        Initialize weights for the linear layer using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def mean_pooling(self, last_hidden_state, attention_mask):
        """
        Performs mean pooling on the last hidden state, strictly respecting the attention mask.

        Args:
            last_hidden_state: (Batch, Seq_Len, Hidden_Size)
            attention_mask: (Batch, Seq_Len)

        Returns:
            Pooled embeddings: (Batch, Hidden_Size)
        """
        # Expand attention mask to match hidden state dimensions
        # attention_mask: (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings over the sequence length where mask is 1
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum mask values (count of valid tokens)
        sum_mask = input_mask_expanded.sum(1)

        # Avoid division by zero by clamping the denominator
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids: Tensor of token ids (Batch, Seq_Len).
            attention_mask: Tensor of attention masks (Batch, Seq_Len).
            labels: Optional labels (not used for loss calculation inside model).

        Returns:
            logits: Raw output from the linear layer (Batch, 1).
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Aggregate features via Mean Pooling
        feature = self.mean_pooling(last_hidden_state, attention_mask)

        # Apply Dropout
        feature = self.dropout(feature)

        # Classification Head
        logits = self.fc(feature)

        return logits
