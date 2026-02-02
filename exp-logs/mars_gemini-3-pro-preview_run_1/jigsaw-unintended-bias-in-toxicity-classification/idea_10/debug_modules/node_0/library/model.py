import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import CFG


class MeanPooling(nn.Module):
    """
    Performs mean pooling on the token embeddings, ignoring padding tokens.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # Expand attention mask to match the size of last_hidden_state
        # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings over the sequence dimension, masking out padding
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum the mask values to get the count of non-padding tokens
        sum_mask = input_mask_expanded.sum(1)

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class JigsawModel(nn.Module):
    """
    DeBERTa-v3-Large based model with Multi-Task Learning heads for
    Toxicity, Identity detection, and Identity Attack detection.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(CFG.model_name)
        self.config.update(
            {
                "output_hidden_states": False,
                "hidden_dropout_prob": CFG.dropout,
                "attention_probs_dropout_prob": CFG.dropout,
            }
        )

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                CFG.model_name, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Enable gradient checkpointing for memory efficiency with Large models
        self.backbone.gradient_checkpointing_enable()

        self.pool = MeanPooling()
        self.dropout = nn.Dropout(CFG.dropout)

        # Hidden size of DeBERTa-v3-Large (usually 1024)
        self.hidden_size = self.config.hidden_size

        # --- Multi-Task Heads ---

        # 1. Primary Toxicity Head (Binary Classification)
        self.toxicity_head = nn.Linear(self.hidden_size, 1)

        # 2. Auxiliary Identity Head (Multi-label Classification)
        # Predicts which identities are mentioned in the text
        self.identity_head = nn.Linear(self.hidden_size, len(CFG.identity_cols))

        # 3. Auxiliary Identity Attack Head (Binary Classification)
        # Explicitly models the "Identity Attack" subtype
        self.attack_head = nn.Linear(self.hidden_size, 1)

        # Initialize weights for the new heads
        self._init_weights(self.toxicity_head)
        self._init_weights(self.identity_head)
        self._init_weights(self.attack_head)

    def _init_weights(self, module):
        """
        Initialize the weights of the classification heads.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, seq_len)
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len)

        Returns:
            dict: Dictionary containing logits for all three heads.
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Pooling
        feature = self.pool(last_hidden_state, attention_mask)

        # Dropout
        feature = self.dropout(feature)

        # Multi-Task Predictions
        logits = self.toxicity_head(feature)
        aux_identity_logits = self.identity_head(feature)
        aux_attack_logits = self.attack_head(feature)

        return {
            "logits": logits,
            "aux_identity_logits": aux_identity_logits,
            "aux_attack_logits": aux_attack_logits,
        }
