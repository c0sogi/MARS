import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class TriangulationDeberta(nn.Module):
    """
    DeBERTa-v3 based model with Semantic Triangulation.

    This architecture uses a shared encoder backbone with three distinct heads:
    1. Primary Head: Predicts the main toxicity target.
    2. Identity Head: Predicts the presence of specific identity attributes.
    3. Attack Head: Predicts the 'identity_attack' subtype to help disentangle
       identity mentions from identity attacks.
    """

    def __init__(self, pretrained_model_name=Config.MODEL_NAME):
        super(TriangulationDeberta, self).__init__()

        # Load Configuration to get hidden size
        self.config = AutoConfig.from_pretrained(pretrained_model_name)

        # Load the pre-trained backbone
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)

        # Hidden size of the backbone (e.g., 768 for base)
        hidden_size = self.config.hidden_size

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # --- Classification Heads ---

        # 1. Primary Toxicity Head (Binary Classification -> 1 logit)
        self.primary_head = nn.Linear(hidden_size, 1)

        # 2. Auxiliary Identity Head (Multi-label Classification -> N logits)
        num_identities = len(Config.IDENTITY_COLS)
        self.identity_head = nn.Linear(hidden_size, num_identities)

        # 3. Auxiliary Identity Attack Head (Binary Classification -> 1 logit)
        self.attack_head = nn.Linear(hidden_size, 1)

        # Initialize weights for the new heads
        self._init_weights(self.primary_head)
        self._init_weights(self.identity_head)
        self._init_weights(self.attack_head)

    def _init_weights(self, module):
        """
        Initialize the weights of the linear layers using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the Semantic Triangulation model.

        Args:
            input_ids: Tensor of token ids (Batch, Seq_Len)
            attention_mask: Tensor of attention masks (Batch, Seq_Len)

        Returns:
            Dictionary containing logits for 'primary', 'identity', and 'attack' heads.
        """
        # Pass inputs through the backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the CLS token representation.
        # For DeBERTa-v3, we use the first token of the last hidden state.
        # Shape: (Batch, Hidden_Size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        x = self.dropout(cls_embedding)

        # Route to heads
        primary_logits = self.primary_head(x)
        identity_logits = self.identity_head(x)
        attack_logits = self.attack_head(x)

        return {
            "primary": primary_logits,
            "identity": identity_logits,
            "attack": attack_logits,
        }
