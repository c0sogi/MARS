import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer (Weighted Layer Pooling context) that dynamically
    weighs the importance of tokens in the sequence.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]

        # Calculate attention scores
        w = self.attention(last_hidden_state)  # [batch_size, seq_len, 1]

        # Mask padding tokens (set to very small value so softmax becomes 0)
        mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        w = w.masked_fill(mask == 0, -1e4)

        # Calculate softmax probabilities
        att_weights = torch.softmax(w, dim=1)  # [batch_size, seq_len, 1]

        # Weighted sum of hidden states
        pooled = torch.sum(
            last_hidden_state * att_weights, dim=1
        )  # [batch_size, hidden_size]

        return pooled


class EssayModel(nn.Module):
    """
    Essay Scoring Model using DeBERTa-v3 backbone with Attention Pooling and a Linear Head.
    Supports loading from base pre-trained weights or domain-adapted MLM checkpoints.
    """

    def __init__(self, checkpoint_path=None, pretrained=True):
        super().__init__()

        # Determine model path: use provided checkpoint (MLM) or default base model
        model_path = checkpoint_path if checkpoint_path else Config.model_name

        # Load configuration
        self.config = AutoConfig.from_pretrained(model_path)

        # Apply dropout settings from Config
        self.config.hidden_dropout_prob = Config.dropout
        self.config.attention_probs_dropout_prob = Config.dropout

        # Load Backbone
        # If loading from MLM checkpoint, this will load encoder weights and ignore LM head
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_path, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing to save memory
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # Cite debug_lesson_2: Explicitly enable input gradients for custom loop compatibility
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()

        # Pooling Layer
        self.pool = AttentionPooling(self.config.hidden_size)

        # Regression Head
        self.fc = nn.Linear(self.config.hidden_size, Config.num_labels)

        # Initialize custom layers
        self._init_weights(self.pool)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the custom head and pooling layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for sub_module in module:
                self._init_weights(sub_module)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor of attention masks
            labels: Optional, not used in forward but kept for API consistency

        Returns:
            logits: Tensor of predicted scores [batch_size]
        """
        # Backbone forward pass
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Get sequence output
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Apply Attention Pooling
        feature = self.pool(last_hidden_state, attention_mask)  # [batch, hidden]

        # Regression Head
        logits = self.fc(feature)  # [batch, 1]

        # Squeeze to [batch] to match target shape
        return logits.squeeze(-1)
