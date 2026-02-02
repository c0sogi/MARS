import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class ToxicityModel(nn.Module):
    """
    DeBERTa-v3-Large based model with Multi-Task Learning heads.

    Structure:
    - Backbone: microsoft/deberta-v3-large
    - Head 1: Primary Toxicity (1 output)
    - Head 2: Identity Attributes (N outputs)
    - Head 3: Identity Attack Subtype (1 output)
    """

    def __init__(
        self,
        model_name: str,
        num_identity_classes: int,
        num_aux_attack_classes: int = 1,
    ):
        """
        Args:
            model_name (str): HuggingFace model identifier (e.g., "microsoft/deberta-v3-large").
            num_identity_classes (int): Number of identity attributes to predict.
            num_aux_attack_classes (int): Number of identity attack subtypes (default 1).
        """
        super().__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # --- Multi-Task Heads ---

        # 1. Primary Toxicity Head
        # Predicts the main 'target' toxicity score
        self.toxicity_head = nn.Linear(self.config.hidden_size, 1)

        # 2. Auxiliary Identity Head
        # Predicts the presence of specific identity attributes (e.g., male, female, etc.)
        self.identity_aux_head = nn.Linear(
            self.config.hidden_size, num_identity_classes
        )

        # 3. Auxiliary Identity Attack Head
        # Predicts the 'identity_attack' subtype to help distinguish attacks from mentions
        self.identity_attack_head = nn.Linear(
            self.config.hidden_size, num_aux_attack_classes
        )

        # Initialize weights for the new heads
        self._init_weights(self.toxicity_head)
        self._init_weights(self.identity_aux_head)
        self._init_weights(self.identity_attack_head)

    def _init_weights(self, module):
        """
        Initialize the weights of the linear layers to match the backbone's initialization.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            dict: Dictionary containing logits for 'toxicity', 'identity', and 'attack'.
        """
        # Pass through DeBERTa backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token representation (first token in the sequence)
        # Shape: (batch_size, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        x = self.dropout(cls_embedding)

        # Pass the embedding through the three distinct heads
        toxicity_logits = self.toxicity_head(x)
        identity_logits = self.identity_aux_head(x)
        attack_logits = self.identity_attack_head(x)

        return {
            "toxicity": toxicity_logits,
            "identity": identity_logits,
            "attack": attack_logits,
        }
