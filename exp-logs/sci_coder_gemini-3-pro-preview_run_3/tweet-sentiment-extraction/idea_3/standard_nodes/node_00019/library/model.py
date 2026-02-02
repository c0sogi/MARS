import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class TweetModel(nn.Module):
    """
    TweetModel architecture for Sentiment Extraction.

    Simplified to use the final hidden layer directly, as empirical evidence
    suggests layer aggregation (WLP) reduces performance for this specific
    span extraction task. Cite solution_lesson_node_00018.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.hf_config = AutoConfig.from_pretrained(
            config.model_name, output_hidden_states=True
        )

        self.backbone = AutoModel.from_pretrained(
            config.model_name, config=self.hf_config
        )

        self.dropout = nn.Dropout(config.dropout)

        # Projects hidden_size -> 2 (start_logit, end_logit)
        self.fc = nn.Linear(self.hf_config.hidden_size, 2)

        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use the final hidden state directly
        # Shape: (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)

        return start_logits.squeeze(-1), end_logits.squeeze(-1)
