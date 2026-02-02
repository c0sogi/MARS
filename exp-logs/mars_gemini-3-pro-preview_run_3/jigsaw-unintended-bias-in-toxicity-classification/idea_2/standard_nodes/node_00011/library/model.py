import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from library.config import Config


class DistilBertWithBiasHead(nn.Module):
    """
    DistilBERT-based model with a Multi-Sample Dropout head for toxicity classification.

    This architecture uses a pre-trained Transformer backbone to capture contextual
    semantics and a Multi-Sample Dropout mechanism in the classification head to
    improve generalization and training stability.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT,
        hidden_size=Config.HIDDEN_SIZE,
        num_dropout_samples=5,
    ):
        """
        Args:
            model_name (str): Name of the pre-trained model to load.
            num_classes (int): Number of output classes (1 for binary regression).
            dropout_rate (float): Dropout probability.
            hidden_size (int): Hidden size of the transformer output.
            num_dropout_samples (int): Number of dropout samples for Multi-Sample Dropout.
        """
        super(DistilBertWithBiasHead, self).__init__()

        # Load configuration
        config = DistilBertConfig.from_pretrained(model_name)
        config.output_hidden_states = False

        # Load Backbone
        self.distilbert = DistilBertModel.from_pretrained(model_name, config=config)

        # Multi-Sample Dropout: Multiple dropout layers with the same rate.
        # This creates an ensemble-like effect within a single model during training.
        self.dropout_ops = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_dropout_samples)]
        )

        # Shared Classification Layer
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Auxiliary Classification Layer for Multi-Task Learning
        # Cite solution_lesson_node_00008
        self.aux_classifier = nn.Linear(hidden_size, Config.NUM_AUX_CLASSES)

        # Weight Initialization for the head
        self._init_weights(self.classifier)
        self._init_weights(self.aux_classifier)

    def _init_weights(self, module):
        """
        Initialize weights for the classification head using Xavier Normal.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits (before sigmoid).
        """
        # Get transformer outputs
        # DistilBERT outputs: (last_hidden_state, hidden_states, attentions)
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token embedding (index 0)
        # Shape: (batch_size, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]

        # Multi-Sample Dropout Logic
        # Pass the embedding through multiple dropout masks -> shared linear layer -> average
        logits_list = []
        aux_logits_list = []

        for dropout_op in self.dropout_ops:
            dropped_embedding = dropout_op(cls_embedding)
            logits_list.append(self.classifier(dropped_embedding))
            aux_logits_list.append(self.aux_classifier(dropped_embedding))

        # Stack and average the outputs from the different dropout masks
        # Shape: (batch_size, num_classes)
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        aux_logits = torch.mean(torch.stack(aux_logits_list, dim=0), dim=0)

        return logits, aux_logits
