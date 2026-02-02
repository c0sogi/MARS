import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class BiasAwareDeberta(nn.Module):
    """
    Neural network architecture for Toxicity Classification with Bias Mitigation.
    Uses DeBERTa-v3-base as the backbone with three distinct heads:
    1. Primary Toxicity Head (Binary Classification)
    2. Auxiliary Identity Head (Multi-label Classification)
    3. Auxiliary Identity Attack Head (Binary Classification)
    """

    def __init__(self):
        super(BiasAwareDeberta, self).__init__()

        # Load Configuration and Backbone
        self.model_config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(
            Config.MODEL_NAME, config=self.model_config
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(Config.DROPOUT)

        # 1. Primary Toxicity Head
        # Predicts the main 'target' (toxicity score)
        self.toxicity_head = nn.Linear(self.model_config.hidden_size, 1)

        # 2. Auxiliary Identity Head
        # Predicts the presence of specific identity attributes (Multi-label)
        # Used to force the encoder to retain identity information distinct from toxicity
        num_identities = len(Config.IDENTITY_COLUMNS)
        self.identity_head = nn.Linear(self.model_config.hidden_size, num_identities)

        # 3. Auxiliary Identity Attack Head
        # Predicts the 'identity_attack' subtype
        # Used to help the model distinguish between mere mention and attack
        self.identity_attack_head = nn.Linear(self.model_config.hidden_size, 1)

        # Initialize weights for the new heads
        self._init_weights(self.toxicity_head)
        self._init_weights(self.identity_head)
        self._init_weights(self.identity_attack_head)

    def _init_weights(self, module):
        """
        Initialize the weights of the linear layers following the backbone's initialization scheme.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            dict: A dictionary containing logits for all three heads:
                  - 'logits': Primary toxicity logits (Batch, 1)
                  - 'aux_identity_logits': Identity attribute logits (Batch, Num_Identities)
                  - 'aux_attack_logits': Identity attack logits (Batch, 1)
        """
        # Pass through DeBERTa backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the representation of the [CLS] token (first token)
        # DeBERTa-v3 uses the first token as the sequence representative
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        features = self.dropout(cls_embedding)

        # Pass features through the three distinct heads
        toxicity_logits = self.toxicity_head(features)
        identity_logits = self.identity_head(features)
        attack_logits = self.identity_attack_head(features)

        return {
            "logits": toxicity_logits,
            "aux_identity_logits": identity_logits,
            "aux_attack_logits": attack_logits,
        }
