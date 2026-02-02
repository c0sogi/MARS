import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Computes a learnable weighted average of the last N transformer layers.
    """

    def __init__(self, num_hidden_layers):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        # Initialize weights to be equal (logits for softmax)
        self.weights = nn.Parameter(torch.ones(num_hidden_layers))

    def forward(self, all_hidden_states):
        """
        Args:
            all_hidden_states: Tuple of tensors from the backbone.
                               Usually (embeddings, layer_1, ..., layer_N).
        Returns:
            weighted_output: Tensor of shape (batch, seq_len, hidden_size)
        """
        # Select the last N layers
        # We assume all_hidden_states contains at least num_hidden_layers
        if len(all_hidden_states) < self.num_hidden_layers:
            selected_layers = all_hidden_states
        else:
            selected_layers = all_hidden_states[-self.num_hidden_layers :]

        # Stack layers: (batch, seq_len, hidden_size, num_selected_layers)
        stacked_layers = torch.stack(selected_layers, dim=-1)

        # Compute softmax weights: (num_selected_layers)
        softmax_weights = torch.nn.functional.softmax(self.weights, dim=0)

        # Reshape weights for broadcasting: (1, 1, 1, num_selected_layers)
        softmax_weights = softmax_weights.view(1, 1, 1, -1)

        # Weighted sum: (batch, seq_len, hidden_size)
        weighted_output = (stacked_layers * softmax_weights).sum(dim=-1)

        return weighted_output


class CustomXLMRoberta(nn.Module):
    """
    XLM-RoBERTa with Weighted Layer Pooling and Multi-Task Heads (QA + Answerability).
    """

    def __init__(self, config_path=None, pretrained=True):
        super(CustomXLMRoberta, self).__init__()

        model_name = config_path if config_path else Config.MODEL_CHECKPOINT
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True

        if pretrained:
            self.roberta = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.roberta = AutoModel.from_config(self.config)

        # Weighted Layer Pooling
        self.pooler = WeightedLayerPooling(Config.N_LAST_HIDDEN)

        # Heads
        # QA Head: Predicts start and end logits for each token
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # Answerability Head: Predicts if the span contains an answer (Binary)
        # Applied on the [CLS] token representation (index 0)
        self.answerability_classifier = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for new layers
        self._init_weights(self.qa_outputs)
        self._init_weights(self.answerability_classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        answerable=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass.
        Returns:
            start_logits: (batch, seq_len)
            end_logits: (batch, seq_len)
            answerability_logits: (batch, 1)
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Force True to get hidden states for pooling
            return_dict=True,
        )

        # outputs.hidden_states is a tuple of (embeddings, layer_1, ..., layer_N)
        hidden_states = outputs.hidden_states

        # Apply Weighted Layer Pooling to get sequence representation
        sequence_output = self.pooler(hidden_states)

        # 1. QA Logits
        # (batch, seq_len, 2)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch, seq_len)
        end_logits = end_logits.squeeze(-1)  # (batch, seq_len)

        # 2. Answerability Logits
        # Use the [CLS] token (index 0) from the pooled output
        cls_output = sequence_output[:, 0, :]
        answerability_logits = self.answerability_classifier(cls_output)  # (batch, 1)

        return start_logits, end_logits, answerability_logits
