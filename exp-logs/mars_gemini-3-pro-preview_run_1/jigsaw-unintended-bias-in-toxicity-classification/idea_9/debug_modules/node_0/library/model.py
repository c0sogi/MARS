import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import CFG


class MeanPooling(nn.Module):
    """
    Performs Mean Pooling on the last hidden state of the transformer backbone.
    This averages the token embeddings, taking the attention mask into account
    to ignore padding tokens.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # Expand attention mask to match the size of last_hidden_state
        # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings across the sequence dimension, masking padding
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum the mask to get the count of non-padding tokens
        sum_mask = input_mask_expanded.sum(1)

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class JigsawModel(nn.Module):
    """
    Custom DeBERTa-v3-Large model with Triangulated Multi-Task Heads.

    Architecture:
    1. Backbone: DeBERTa-v3-Large (with Gradient Checkpointing)
    2. Pooling: Mean Pooling
    3. Heads:
       - Primary: Toxicity (1 output)
       - Aux 1: Identity Attributes (9 outputs)
       - Aux 2: Identity Attack (1 output)
    """

    def __init__(self, pretrained=True):
        super(JigsawModel, self).__init__()

        self.config = AutoConfig.from_pretrained(CFG.model_name)

        # Initialize backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(CFG.model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing for memory efficiency with Large models
        if CFG.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.pool = MeanPooling()

        # --- Triangulated Multi-Task Heads ---

        # 1. Primary Toxicity Head
        # Predicts the main target (toxicity score)
        self.toxicity_head = nn.Linear(self.config.hidden_size, 1)

        # 2. Auxiliary Identity Head
        # Predicts the presence of specific identity attributes (e.g., male, muslim)
        self.identity_head = nn.Linear(
            self.config.hidden_size, CFG.num_identity_classes
        )

        # 3. Auxiliary Identity Attack Head
        # Predicts the 'identity_attack' subtype to help disentangle attack from mention
        self.identity_attack_head = nn.Linear(
            self.config.hidden_size, CFG.num_identity_attack_classes
        )

        # Initialize weights for new layers
        self._init_weights(self.toxicity_head)
        self._init_weights(self.identity_head)
        self._init_weights(self.identity_attack_head)

    def _init_weights(self, module):
        """
        Initialize the weights of the linear layers using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def feature(self, input_ids, attention_mask):
        """
        Extract features from the backbone.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, attention_mask)
        return feature

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the backbone and all three heads.

        Returns:
            dict: Contains 'logits' (toxicity), 'aux_identity', and 'aux_attack'.
        """
        # Get pooled sentence representation
        feature = self.feature(input_ids, attention_mask)

        # Pass through heads
        toxicity_logits = self.toxicity_head(feature)
        identity_logits = self.identity_head(feature)
        attack_logits = self.identity_attack_head(feature)

        return {
            "logits": toxicity_logits,
            "aux_identity": identity_logits,
            "aux_attack": attack_logits,
        }
