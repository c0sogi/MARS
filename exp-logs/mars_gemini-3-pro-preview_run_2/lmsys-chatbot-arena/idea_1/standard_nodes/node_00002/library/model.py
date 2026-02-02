import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from library.config import Config


class ChatbotTransformer(nn.Module):
    """
    Transformer-based Cross-Encoder for Chatbot Preference Prediction.

    Uses a pre-trained transformer (e.g., DeBERTa) to jointly encode the prompt
    and both responses, allowing for deep interaction and quality comparison.
    Cite solution_lesson_node_00001: Switching from frozen embeddings to a Cross-Encoder
    architecture to capture quality nuances.
    """

    def __init__(
        self, model_name=Config.TRANSFORMER_MODEL, num_classes=Config.NUM_CLASSES
    ):
        super(ChatbotTransformer, self).__init__()

        config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits
