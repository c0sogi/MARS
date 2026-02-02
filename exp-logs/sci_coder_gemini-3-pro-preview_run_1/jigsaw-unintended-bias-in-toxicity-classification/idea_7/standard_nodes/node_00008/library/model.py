import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class DebertaV3MultiTask(nn.Module):
    """
    Multi-Task DeBERTa-v3-Large model for Toxicity Classification.

    This architecture implements the 'Hybrid Pairwise Ranking' strategy by using a shared
    encoder backbone with three distinct heads:
    1. Toxicity Head (Primary): Predicts the main binary toxicity target.
    2. Identity Head (Auxiliary): Predicts the presence of specific identity attributes.
       This forces the encoder to disentangle identity features from toxicity features.
    3. Attack Type Head (Auxiliary): Predicts the 'identity_attack' subtype.

    Attributes:
        backbone: The pretrained DeBERTa-v3-Large model.
        toxicity_head: Linear layer for the primary task.
        identity_head: Linear layer for the identity prediction auxiliary task.
        attack_type_head: Linear layer for the identity attack auxiliary task.
    """

    def __init__(self, pretrained_model_name=Config.MODEL_NAME):
        super().__init__()

        # Load Configuration from pretrained model
        self.config = AutoConfig.from_pretrained(pretrained_model_name)

        # Initialize Backbone
        # We use the AutoModel to get the raw hidden states
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name, config=self.config
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(Config.DROPOUT)

        # Define Dimensions based on Config
        hidden_size = self.config.hidden_size
        num_identities = len(Config.IDENTITY_COLS)
        num_aux_attack = len(Config.AUX_COLS)

        # 1. Primary Toxicity Head
        # Output: 1 logit (Binary Classification)
        self.toxicity_head = nn.Linear(hidden_size, 1)

        # 2. Auxiliary Identity Head
        # Output: N logits corresponding to identity columns (Multi-label Classification)
        self.identity_head = nn.Linear(hidden_size, num_identities)

        # 3. Auxiliary Identity Attack Head
        # Output: 1 logit (Binary Classification for 'identity_attack' subtype)
        self.attack_type_head = nn.Linear(hidden_size, num_aux_attack)

        # Initialize weights for the new heads
        self._init_weights(self.toxicity_head)
        self._init_weights(self.identity_head)
        self._init_weights(self.attack_type_head)

    def _init_weights(self, module):
        """
        Initialize weights for linear layers using the backbone's initializer range.
        This ensures the new heads start with a scale compatible with the pretrained model.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, ids, mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).
            token_type_ids (torch.Tensor, optional): Token type IDs.

        Returns:
            dict: A dictionary containing logits for all three heads:
                - 'toxicity_logits': Tensor of shape (batch_size, 1)
                - 'identity_logits': Tensor of shape (batch_size, num_identities)
                - 'attack_logits': Tensor of shape (batch_size, num_aux_attack)
        """
        # Pass inputs through the DeBERTa backbone
        outputs = self.backbone(
            input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids
        )

        # Extract the CLS token representation (first token in the sequence)
        # Shape: (Batch_Size, Hidden_Size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply Dropout
        features = self.dropout(cls_embedding)

        # Compute logits for each task
        toxicity_logits = self.toxicity_head(features)
        identity_logits = self.identity_head(features)
        attack_logits = self.attack_type_head(features)

        return {
            "toxicity_logits": toxicity_logits,
            "identity_logits": identity_logits,
            "attack_logits": attack_logits,
        }
