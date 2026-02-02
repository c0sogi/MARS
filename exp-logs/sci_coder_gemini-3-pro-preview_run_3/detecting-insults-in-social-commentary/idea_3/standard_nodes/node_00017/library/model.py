import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class InsultModel(nn.Module):
    """
    Main model architecture for Insult Detection.
    Backbone: RoBERTa-Large
    Head: Mean Pooling + Standard Dropout
    """

    def __init__(self):
        super(InsultModel, self).__init__()

        # Load Transformer Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.model_path)
        self.backbone = AutoModel.from_pretrained(Config.model_path, config=self.config)

        # Freeze Embeddings and Bottom 6 Layers
        self._freeze_layers()

        # Final Classification Head
        self.dropout = nn.Dropout(Config.dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights
        self._init_weights(self.fc)

    def _freeze_layers(self):
        # Freeze Embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        # Freeze bottom 6 encoder layers
        # RoBERTa encoder layers are accessible via backbone.encoder.layer
        for i in range(6):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # Pass through Backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Mean Pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # Dropout and Classification
        feature = self.dropout(mean_embeddings)
        logits = self.fc(feature)

        return logits
