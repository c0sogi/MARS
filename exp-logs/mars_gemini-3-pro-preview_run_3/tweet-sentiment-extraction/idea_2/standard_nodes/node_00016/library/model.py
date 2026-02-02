import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class TweetModel(nn.Module):
    """
    Neural Network for Tweet Sentiment Extraction.

    Architecture:
    1. Backbone: DeBERTa-v3-base (pretrained).
    2. Head: Multi-Sample Dropout + Linear Layer.

    The model predicts the start and end logits for the selected text span.
    """

    def __init__(self):
        super(TweetModel, self).__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(
            Config.MODEL_NAME, output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Dropout Layer
        self.dropout = nn.Dropout(0.1)

        # Classification Head
        self.classifier = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Simple Head: Dropout + Linear
        # Cite solution_lesson_node_00009: Decouple Backbone Upgrades from Architectural Changes
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
