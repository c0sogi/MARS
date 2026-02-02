import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MeanPooling(nn.Module):
    """
    Mean Pooling mechanism.
    Averages the hidden states of tokens, respecting the attention mask.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            mean_embeddings: (batch_size, hidden_size)
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class InsultModel(nn.Module):
    """
    Main model architecture for Insult Detection.
    Backbone: RoBERTa-Large
    Head: Mean Pooling + Dropout + Linear
    """

    def __init__(self):
        super(InsultModel, self).__init__()

        # Load Transformer Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.model_path)
        self.backbone = AutoModel.from_pretrained(Config.model_path, config=self.config)

        # Freezing Layers (Cite solution_lesson_node_00010)
        # Freeze embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        # Freeze bottom N layers
        if hasattr(self.backbone, "encoder") and hasattr(
            self.backbone.encoder, "layer"
        ):
            for i in range(Config.freeze_layers):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = False

        # Initialize Custom Components
        self.pooling = MeanPooling()
        self.dropout = nn.Dropout(Config.dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for custom layers
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for custom modules using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        Returns:
            logits: (batch_size, 1) - Raw scores before sigmoid
        """
        # Pass through Backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply Mean Pooling (Cite solution_lesson_node_00007)
        feature = self.pooling(last_hidden_state, attention_mask)

        # Apply Dropout and Classification
        feature = self.dropout(feature)
        logits = self.fc(feature)

        return logits
