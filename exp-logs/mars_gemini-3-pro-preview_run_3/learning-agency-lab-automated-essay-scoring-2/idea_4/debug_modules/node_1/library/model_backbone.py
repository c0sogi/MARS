import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer to dynamically weight token embeddings.
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
        # Calculate attention weights
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        w = self.attention(last_hidden_state)  # [batch_size, seq_len, 1]

        # Mask padding tokens so they don't contribute to the average
        float_mask = attention_mask.unsqueeze(-1).float()
        w = w + (1.0 - float_mask) * -10000.0

        weights = torch.softmax(w, dim=1)

        # Weighted sum
        # [batch_size, seq_len, 1] * [batch_size, seq_len, hidden_size] -> sum over dim 1
        return torch.sum(weights * last_hidden_state, dim=1)


class EssayBackbone(nn.Module):
    """
    Main model architecture wrapping DeBERTa-v3 with custom pooling and regression head.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.model_name)

        # Load backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(
                Config.model_name, config=self.config
            )
        else:
            self.model = AutoModel.from_config(self.config)

        # Gradient Checkpointing for memory efficiency
        if Config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Custom Pooling
        self.pool = AttentionPooling(self.config.hidden_size)

        # Regression Head
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize Head Weights
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, return_embedding=False):
        """
        Forward pass.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            return_embedding (bool): If True, returns the pooled feature vector instead of the score.
                                     Used for OOF generation for stacking.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply pooling
        feature = self.pool(last_hidden_state, attention_mask)

        if return_embedding:
            return feature

        # Regression output
        logits = self.fc(feature)
        return logits


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    Perturbs model weights to maximize loss, forcing the optimizer to find a flatter minimum.
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1.0, adv_eps=0.01):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Perturbs the weights based on the gradient direction.
        Should be called after backward() and before optimizer.step().
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Calculate perturbation
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Save original weights
                    self.backup[name] = param.data.clone()

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Clamp perturbation to epsilon ball
                    param.data = torch.min(
                        torch.max(param.data, self.backup[name] - self.adv_eps),
                        self.backup[name] + self.adv_eps,
                    )

    def restore(self):
        """
        Restores the original weights.
        Should be called after optimizer.step() or before the next forward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
