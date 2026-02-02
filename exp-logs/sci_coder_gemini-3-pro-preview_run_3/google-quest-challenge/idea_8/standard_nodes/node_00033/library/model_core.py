import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BackboneWrapper(nn.Module):
    """
    A PyTorch module that wraps a Hugging Face transformer backbone.
    It extracts topology-aware embeddings (Question-specific, Answer-specific, and Joint)
    using custom masking provided by the data loader.
    """

    def __init__(self, model_name, num_labels=30):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Enable gradient checkpointing for memory efficiency with large models
        # This allows training larger batch sizes or larger models on limited VRAM
        self.backbone.gradient_checkpointing_enable()

        # Ensure input embeddings require gradients for checkpointing to work
        # This prevents "RuntimeError: Trying to backward through the graph a second time"
        self.backbone.enable_input_require_grads()

        self.hidden_size = self.config.hidden_size
        self.num_labels = num_labels

        # Temporary linear head for the fine-tuning phase (Phase 2)
        # We use a simple projection from the CLS token.
        # This head is used to drive the gradients into the backbone during fine-tuning
        # but is replaced by Ridge Regressors in Phase 3.
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        # Initialize classifier weights
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """
        Initialize the weights of the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, q_mask=None, a_mask=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): (batch, seq_len)
            attention_mask (torch.Tensor): (batch, seq_len)
            q_mask (torch.Tensor): (batch, seq_len) - Binary mask for Question tokens
            a_mask (torch.Tensor): (batch, seq_len) - Binary mask for Answer tokens
            labels (torch.Tensor, optional): (batch, num_labels)

        Returns:
            dict: Contains 'logits', 'loss' (if labels provided), and 'features' dict.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 1. Extract CLS Embedding (Joint Context)
        # Shape: (batch, hidden_size)
        h_cls = last_hidden_state[:, 0, :]

        features = {"h_cls": h_cls}

        # 2. Extract Question Embedding (h_Q) via Mean Pooling with q_mask
        if q_mask is not None:
            h_q = self._mean_pooling(last_hidden_state, q_mask)
            features["h_q"] = h_q

        # 3. Extract Answer Embedding (h_A) via Mean Pooling with a_mask
        if a_mask is not None:
            h_a = self._mean_pooling(last_hidden_state, a_mask)
            features["h_a"] = h_a

        # Compute logits for fine-tuning supervision using the Joint Context (CLS)
        logits = self.classifier(h_cls)

        result = {"logits": logits, "features": features}

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            result["loss"] = loss

        return result

    def _mean_pooling(self, hidden_state, mask):
        """
        Performs mean pooling on the hidden state using the provided binary mask.

        Args:
            hidden_state: (batch, seq_len, hidden_size)
            mask: (batch, seq_len)

        Returns:
            pooled_output: (batch, hidden_size)
        """
        # Expand mask to match hidden_state dimensions
        # mask: (batch, seq_len) -> (batch, seq_len, 1)
        mask_expanded = mask.unsqueeze(-1).expand(hidden_state.size()).float()

        # Sum embeddings where mask is 1
        sum_embeddings = torch.sum(hidden_state * mask_expanded, 1)

        # Sum mask values to get the count of tokens
        sum_mask = mask_expanded.sum(1)

        # Avoid division by zero by clamping the divisor
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask
