import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CustomXLMRoberta(nn.Module):
    """
    Custom XLM-RoBERTa model with Weighted Layer Pooling and Multi-Task Heads.

    Architecture:
    1. Backbone: XLM-RoBERTa Large
    2. Pooling: Weighted average of the last N hidden layers (learnable weights).
    3. Head 1 (QA): Predicts start and end logits for span extraction.
    4. Head 2 (Answerability): Predicts binary logit for answer presence on [CLS] token.
    """

    def __init__(self):
        super(CustomXLMRoberta, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.model_name)
        # Ensure we get all hidden states for pooling
        self.config.output_hidden_states = True

        # Load Pre-trained Backbone
        self.roberta = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Weighted Layer Pooling Configuration
        self.n_pool_layers = Config.n_pool_layers

        # Learnable weights for the selected layers.
        # Initialized to ones (softmax will make them equal initially).
        self.layer_weights = nn.Parameter(torch.ones(self.n_pool_layers))

        # ---------------------------------------------------------------------
        # Task Heads
        # ---------------------------------------------------------------------

        # 1. QA Head: Maps hidden size to 2 logits (Start, End)
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # 2. Answerability Head: Maps hidden size to 1 logit (Binary Classification)
        self.answerable_classifier = nn.Linear(self.config.hidden_size, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Initialize weights for the new heads
        self._init_weights(self.qa_outputs)
        self._init_weights(self.answerable_classifier)

    def _init_weights(self, module):
        """
        Initialize the weights of the task-specific heads following
        the standard BERT/RoBERTa initialization pattern.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input indices of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
                                           Shape (batch_size, seq_len).

        Returns:
            start_logits (torch.Tensor): Logits for start position (batch_size, seq_len).
            end_logits (torch.Tensor): Logits for end position (batch_size, seq_len).
            answerable_logits (torch.Tensor): Logits for answerability (batch_size, 1).
        """
        # Pass through backbone
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.hidden_states is a tuple of (batch, seq, hidden_size)
        # We extract the last n_pool_layers
        all_hidden_states = outputs.hidden_states

        # Safety check
        if len(all_hidden_states) < self.n_pool_layers:
            raise ValueError(
                f"Backbone has fewer layers ({len(all_hidden_states)}) than requested for pooling ({self.n_pool_layers})."
            )

        # Select last N layers
        selected_hidden_states = all_hidden_states[-self.n_pool_layers :]

        # Stack to shape: (n_pool_layers, batch_size, seq_len, hidden_size)
        stacked_layers = torch.stack(selected_hidden_states, dim=0)

        # Calculate softmax weights: shape (n_pool_layers)
        weights = torch.softmax(self.layer_weights, dim=0)

        # Apply Weighted Pooling
        # Reshape weights to (n_pool_layers, 1, 1, 1) for broadcasting
        # Result shape: (batch_size, seq_len, hidden_size)
        weighted_sum = (weights.view(-1, 1, 1, 1) * stacked_layers).sum(dim=0)

        # Apply Dropout
        sequence_output = self.dropout(weighted_sum)

        # ---------------------------------------------------------------------
        # Head Outputs
        # ---------------------------------------------------------------------

        # 1. QA Logits
        logits = self.qa_outputs(sequence_output)  # (batch_size, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1)  # (batch_size, seq_len)

        # 2. Answerability Logits
        # Use the [CLS] token representation (index 0) from the weighted sequence
        cls_token_output = sequence_output[:, 0, :]
        answerable_logits = self.answerable_classifier(
            cls_token_output
        )  # (batch_size, 1)

        return start_logits, end_logits, answerable_logits
