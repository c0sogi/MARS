import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling: Learns a weighted average of the last N hidden layers.
    """

    def __init__(
        self, num_hidden_layers: int = 4, layer_start: int = 4, hidden_size: int = 768
    ):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.weights = nn.Parameter(
            torch.tensor([1] * num_hidden_layers, dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors (batch, seq_len, hidden)
        # We take the last 'num_hidden_layers' layers
        all_layer_embedding = all_hidden_states[-self.num_hidden_layers :]

        # Stack to (batch, seq_len, hidden, num_layers)
        all_layer_embedding = torch.stack(all_layer_embedding, dim=-1)

        # Calculate softmax weights: (num_layers,)
        weight_factor = F.softmax(self.weights, dim=0)

        # Weighted sum: (batch, seq_len, hidden)
        weighted_embedding = (all_layer_embedding * weight_factor).sum(dim=-1)

        return weighted_embedding


class LinearAttentionPooling(nn.Module):
    """
    Linear Attention Pooling: Computes a context-aware weighted sum of token embeddings.
    """

    def __init__(self, hidden_size: int = 768):
        super(LinearAttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch, seq_len, hidden)
        # attention_mask: (batch, seq_len)

        # Calculate raw attention scores: (batch, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Mask padding tokens (set to -inf so softmax becomes 0)
        # attention_mask is 1 for tokens, 0 for padding
        # We want to add a large negative number where mask is 0
        w = w.squeeze(-1)  # (batch, seq_len)
        w.masked_fill_(attention_mask == 0, -1e4)

        # Softmax to get probabilities
        alpha = torch.softmax(w, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)

        # Weighted sum
        context_vector = torch.sum(last_hidden_state * alpha, dim=1)  # (batch, hidden)

        return context_vector


class CustomModel(nn.Module):
    """
    DeBERTa-v3 based model with Weighted Layer Pooling, Hybrid Head, and Multi-Sample Dropout.
    """

    def __init__(self, config: Config, pretrained: bool = True):
        super(CustomModel, self).__init__()
        self.config = config

        # Load Backbone
        if pretrained:
            self.model_config = AutoConfig.from_pretrained(config.model_name)
        else:
            self.model_config = AutoConfig.from_pretrained(config.model_name)

        self.model_config.update({"output_hidden_states": True})

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                config.model_name, config=self.model_config
            )
        else:
            self.backbone = AutoModel.from_config(self.model_config)

        # Feature Aggregation
        self.layer_pooler = WeightedLayerPooling(
            num_hidden_layers=config.aggregation_layers,
            layer_start=config.aggregation_layers,
            hidden_size=config.hidden_size,
        )

        # Pooling Heads
        self.attention_pooler = LinearAttentionPooling(hidden_size=config.hidden_size)

        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in config.msd_dropout_rates])

        # Final Classifier
        # Input size is hidden_size * 2 because we concat (Attention Pool + Max Pool)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_labels)

        # Initialize weights for custom layers
        self._init_weights(self.layer_pooler)
        self._init_weights(self.attention_pooler)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels=None):
        # Backbone Forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Get all hidden states
        all_hidden_states = outputs.hidden_states

        # 1. Weighted Layer Pooling
        # Aggregates info from last N layers into a single sequence representation
        sequence_output = self.layer_pooler(all_hidden_states)  # (batch, seq, hidden)

        # 2. Hybrid Pooling
        # A. Linear Attention Pooling (Weighted Average)
        avg_pool = self.attention_pooler(
            sequence_output, attention_mask
        )  # (batch, hidden)

        # B. Global Max Pooling (Strongest Signal)
        # Mask padding for max pooling
        # Set padding tokens to a very small number
        sequence_output_masked = sequence_output.clone()
        sequence_output_masked[attention_mask == 0] = -1e4
        max_pool = torch.max(sequence_output_masked, dim=1)[0]  # (batch, hidden)

        # Concatenate
        pooled_output = torch.cat([avg_pool, max_pool], dim=1)  # (batch, hidden * 2)

        # 3. Multi-Sample Dropout & Classification
        logits_list = []
        for dropout in self.dropouts:
            logits_list.append(self.fc(dropout(pooled_output)))

        # Average the logits
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)

        return logits


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    Perturbs weights to maximize loss, improving robustness.
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1e-4, adv_eps=1e-2):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Performs the adversarial attack on the weights.
        Should be called after loss.backward() and before optimizer.step().
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            # Apply perturbation only to parameters that require grad and match the target name (e.g., 'weight')
            # Typically we avoid perturbing LayerNorm or Bias
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):

                # Calculate perturbation direction
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Perturbation formula
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Clamp perturbation to epsilon
                    # Note: Simple clamping might not be strictly correct for 'weight' AWP which scales by norm,
                    # but here we follow the standard implementation logic often used in Kaggle.
                    # Ideally: r_at = min(max(r_at, -eps), eps) if absolute, or scaled.
                    # Here we trust adv_lr controls the scale sufficiently relative to weights.

                    # Save original data
                    self.backup[name] = param.data.clone()

                    # Apply perturbation
                    param.data.add_(r_at)
                    self.backup_eps[name] = r_at

    def restore(self):
        """
        Restores the original weights.
        Should be called after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        self.backup = {}
        self.backup_eps = {}
