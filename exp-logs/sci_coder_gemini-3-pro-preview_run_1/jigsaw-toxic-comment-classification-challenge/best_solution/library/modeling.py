import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class LinearAttentionPooling(nn.Module):
    """
    Implementation of Linear Attention Pooling.
    Computes a weighted average of the hidden states using a learned scalar attention score.
    Formula: v = sum(softmax(w^T h_t) * h_t)
    """

    def __init__(self, hidden_size):
        super().__init__()
        # Simple Linear Attention: projects hidden_size -> 1 scalar score per token
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) - 1 for token, 0 for padding
        Returns:
            context_vector: (batch_size, hidden_size)
        """
        # Calculate attention scores
        # (batch, seq, 1)
        logits = self.attention(last_hidden_state)

        # Mask padding tokens so they don't contribute to the softmax
        # attention_mask is 1 for tokens, 0 for padding
        # (batch, seq, 1)
        mask_expanded = attention_mask.unsqueeze(-1)
        # Set logits of padding tokens to -infinity
        # Use -1e4 instead of -1e9 to prevent float16 overflow in mixed precision
        logits = logits.masked_fill(mask_expanded == 0, -1e4)

        # Calculate weights
        # (batch, seq, 1)
        weights = torch.softmax(logits, dim=1)

        # Weighted sum of hidden states
        # (batch, seq, 1) * (batch, seq, hidden) -> (batch, seq, hidden)
        weighted_states = weights * last_hidden_state

        # Sum over sequence dimension
        # (batch, hidden)
        context_vector = torch.sum(weighted_states, dim=1)

        return context_vector


class ToxicityModel(nn.Module):
    """
    Context-Aware DeBERTa-v3-Base model with Hybrid Pooling (Max + Linear Attention).
    """

    def __init__(self, cfg: Config, pretrained: bool = True):
        super().__init__()
        self.cfg = cfg

        # Load Configuration and Backbone
        if pretrained:
            # Load config first to override dropout settings
            model_config = AutoConfig.from_pretrained(cfg.model_name)
            model_config.hidden_dropout_prob = cfg.hidden_dropout_prob
            model_config.attention_probs_dropout_prob = cfg.attention_probs_dropout_prob

            self.backbone = AutoModel.from_pretrained(
                cfg.model_name, config=model_config
            )
        else:
            model_config = AutoConfig.from_pretrained(cfg.model_name)
            model_config.hidden_dropout_prob = cfg.hidden_dropout_prob
            model_config.attention_probs_dropout_prob = cfg.attention_probs_dropout_prob
            self.backbone = AutoModel.from_config(model_config)

        self.hidden_size = self.backbone.config.hidden_size

        # Pooling Layers
        self.attn_pool = LinearAttentionPooling(self.hidden_size)

        # Classification Head
        # We concatenate Max Pooling (hidden_size) and Attention Pooling (hidden_size)
        self.fc = nn.Linear(self.hidden_size * 2, cfg.num_classes)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

        # Initialize head weights
        self._init_weights(self.fc)
        self._init_weights(self.attn_pool)

    def _init_weights(self, module):
        """Initialize weights for the custom head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.backbone.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, num_classes) optional

        Returns:
            dict containing 'logits' and optionally 'loss'
        """
        # Backbone Forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq, hidden)

        # 1. Global Max Pooling
        # Mask padding tokens before max operation
        mask_expanded = attention_mask.unsqueeze(-1)  # (batch, seq, 1)
        # Set padding positions to a very small number
        # Use -1e4 instead of -1e9 to prevent float16 overflow in mixed precision
        masked_hidden = last_hidden_state.masked_fill(mask_expanded == 0, -1e4)
        # Max over sequence dimension
        max_pool_vec = torch.max(masked_hidden, dim=1)[0]  # (batch, hidden)

        # 2. Linear Attention Pooling
        attn_pool_vec = self.attn_pool(
            last_hidden_state, attention_mask
        )  # (batch, hidden)

        # 3. Concatenate
        concat_vec = torch.cat(
            [max_pool_vec, attn_pool_vec], dim=1
        )  # (batch, 2*hidden)

        # 4. Classification Head
        feature = self.dropout(concat_vec)
        logits = self.fc(feature)  # (batch, num_classes)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {"logits": logits, "loss": loss}


class AWP:
    """
    Adversarial Weight Perturbation (AWP) utility.
    Perturbs model weights to maximize loss during training, improving robustness.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1.0,
        adv_eps=0.01,
        start_epoch=0,
        scaler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Performs the adversarial attack on the weights.
        Calculates perturbation based on gradients and applies it.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Calculate perturbation direction
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Perturbation formula: lr * grad / |grad| * |weight|
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Project back to epsilon ball if needed (clipping)
                    # Note: We rely on restore() to reset, but clipping ensures stability
                    # during the adversarial step if multiple steps were taken (here usually 1).
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def save(self):
        """
        Saves the original weights before perturbation.
        Also calculates the epsilon bounds for clipping.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        """
        Restores the original weights after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
