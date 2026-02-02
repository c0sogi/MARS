import torch
import torch.nn as nn
from transformers import AutoModel
from library.config import Config


class CrossEncoder(nn.Module):
    """
    A Cross-Encoder model that processes the context, anchor, and target jointly.
    Cite solution_lesson_node_00001: Addresses the limitation of Bi-Encoders by allowing
    full token-level interaction.
    """

    def __init__(self, model_name=None):
        super(CrossEncoder, self).__init__()

        if model_name is None:
            model_name = Config.model_name

        self.backbone = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.backbone.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the Cross-Encoder.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token embedding (index 0)
        cls_emb = outputs.last_hidden_state[:, 0, :]

        # Predict score
        logits = self.linear(cls_emb)
        score = self.sigmoid(logits).squeeze(1)

        return score
