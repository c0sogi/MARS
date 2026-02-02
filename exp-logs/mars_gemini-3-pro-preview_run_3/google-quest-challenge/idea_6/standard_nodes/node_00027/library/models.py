import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class SegmentAwareNet(nn.Module):
    """
    A Transformer-based model that leverages segment-specific pooling for Question and Answer.

    Architecture:
    1. Transformer Backbone (e.g., DeBERTa, MPNet, RoBERTa)
    2. Segment Pooling:
       - h_CLS: The [CLS] token representation.
       - h_Q: Mean pooling of tokens belonging to the Question segment.
       - h_A: Mean pooling of tokens belonging to the Answer segment.
       - h_diff: Absolute difference |h_Q - h_A|.
    3. Concatenation: [h_CLS, h_Q, h_A, h_diff] (Size: 4 * Hidden)
    4. Classification Head: Linear projection to target labels.
    """

    def __init__(self, model_name, num_labels=None, pretrained=True):
        """
        Args:
            model_name (str): HuggingFace model identifier.
            num_labels (int, optional): Number of output targets. Defaults to Config.NUM_LABELS.
            pretrained (bool): Whether to load pretrained weights for the backbone.
        """
        super().__init__()

        if num_labels is None:
            num_labels = Config.NUM_LABELS

        self.config = AutoConfig.from_pretrained(model_name)

        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Feature dimension: CLS + Q + A + Diff = 4 * Hidden Size
        self.feature_dim = self.config.hidden_size * 4

        self.fc = nn.Linear(self.feature_dim, num_labels)

        # Initialize the classification head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """Initialize the weights of the classification head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, q_mask, a_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): (Batch, SeqLen)
            attention_mask (torch.Tensor): (Batch, SeqLen)
            q_mask (torch.Tensor): Binary mask for Question tokens (Batch, SeqLen)
            a_mask (torch.Tensor): Binary mask for Answer tokens (Batch, SeqLen)
            labels (torch.Tensor, optional): Targets (Batch, NumLabels)

        Returns:
            dict: Contains 'logits', 'features', and optionally 'loss'.
        """
        # 1. Backbone Forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (Batch, SeqLen, Hidden)

        # 2. Extract Features

        # h_CLS: (Batch, Hidden)
        h_cls = last_hidden_state[:, 0, :]

        # Helper for masked mean pooling
        def masked_mean_pool(hidden, mask):
            # mask: (Batch, SeqLen) -> (Batch, SeqLen, 1)
            mask_expanded = mask.unsqueeze(-1).expand(hidden.size()).float()

            # Sum valid tokens
            sum_embeddings = torch.sum(hidden * mask_expanded, dim=1)

            # Count valid tokens
            sum_mask = mask_expanded.sum(dim=1)

            # Avoid division by zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            return sum_embeddings / sum_mask

        # h_Q: Mean pool of Question tokens
        h_q = masked_mean_pool(last_hidden_state, q_mask)

        # h_A: Mean pool of Answer tokens
        h_a = masked_mean_pool(last_hidden_state, a_mask)

        # h_diff: Interaction feature
        h_diff = torch.abs(h_q - h_a)

        # 3. Concatenate
        # Shape: (Batch, 4 * Hidden)
        features = torch.cat([h_cls, h_q, h_a, h_diff], dim=1)

        # 4. Classification Head
        logits = self.fc(features)

        output = {"logits": logits, "features": features}

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            output["loss"] = loss

        return output
