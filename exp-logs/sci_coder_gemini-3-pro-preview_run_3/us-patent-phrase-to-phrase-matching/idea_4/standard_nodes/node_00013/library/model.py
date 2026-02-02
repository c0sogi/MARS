import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Computes a weighted average of token embeddings where weights are learned.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len)

        # Calculate attention scores
        w = self.attention(last_hidden_state)  # (batch, seq, 1)
        w = w.squeeze(-1)  # (batch, seq)

        # Mask padding tokens so they don't contribute to the average
        if attention_mask is not None:
            w = w.masked_fill(attention_mask == 0, -1e9)

        # Normalize weights
        w = torch.softmax(w, dim=-1)  # (batch, seq)
        w = w.unsqueeze(-1)  # (batch, seq, 1)

        # Weighted sum
        # (batch, seq, hidden) * (batch, seq, 1) -> (batch, seq, hidden) -> sum -> (batch, hidden)
        return torch.sum(last_hidden_state * w, dim=1)


class CustomModel(nn.Module):
    """
    Multi-Task Deberta-V3-Large with Attention Pooling and Multi-Sample Dropout.
    """

    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.model = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Enable Gradient Checkpointing if configured
        if Config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Pooling Layer
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Multi-Sample Dropout (MSD)
        # We use multiple dropout masks to create an ensemble-within-a-model effect
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.msd_dropout_rate) for _ in range(Config.num_msd_rounds)]
        )

        # Output Heads
        # 1. Regression Head for continuous similarity score
        self.fc_regressor = nn.Linear(self.config.hidden_size, 1)

        # 2. Classification Head for discrete score buckets
        self.fc_classifier = nn.Linear(self.config.hidden_size, Config.num_classes)

        # Initialize weights for custom layers
        self._init_weights(self.pooler)
        self._init_weights(self.fc_regressor)
        self._init_weights(self.fc_classifier)

    def _init_weights(self, module):
        """
        Initialize weights for the custom heads and pooling layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, **kwargs):
        # Pass through backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply Attention Pooling
        feature = self.pooler(last_hidden_state, attention_mask)

        # Multi-Sample Dropout Logic
        # Apply dropout K times, pass through heads, and average the logits
        reg_logits_list = []
        cls_logits_list = []

        for i in range(Config.num_msd_rounds):
            # Apply distinct dropout mask
            dropped_feature = self.dropouts[i](feature)

            # Pass through Regression Head
            reg_logits_list.append(self.fc_regressor(dropped_feature))

            # Pass through Classification Head
            cls_logits_list.append(self.fc_classifier(dropped_feature))

        # Average the predictions (Logit Averaging)
        reg_output = torch.mean(torch.stack(reg_logits_list), dim=0)
        cls_output = torch.mean(torch.stack(cls_logits_list), dim=0)

        return {
            "score": reg_output,  # Shape: (batch_size, 1)
            "logits": cls_output,  # Shape: (batch_size, num_classes)
        }
