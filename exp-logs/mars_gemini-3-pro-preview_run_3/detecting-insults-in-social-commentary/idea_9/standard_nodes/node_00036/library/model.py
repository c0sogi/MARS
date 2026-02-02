import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import ModelConfig


class InsultModel(nn.Module):
    """
    Insult Detection Model based on Hugging Face Transformers.
    Implements Mean Pooling and specific layer freezing strategies.
    """

    def __init__(self, model_name, config=ModelConfig):
        super().__init__()
        self.config = config

        # Load AutoConfig and AutoModel
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.model_config.output_hidden_states = True
        self.model = AutoModel.from_pretrained(model_name, config=self.model_config)

        # Freezing Strategy
        self._freeze_layers(config.freeze_layers)

        # Architecture Head
        self.drop = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.model_config.hidden_size, 1)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _freeze_layers(self, freeze_layers_count):
        """
        Freezes embeddings and the specified number of bottom encoder layers.
        """
        if freeze_layers_count == 0:
            return

        for name, param in self.model.named_parameters():
            # Freeze Embeddings
            if "embeddings" in name:
                param.requires_grad = False

            # Freeze Encoder Layers
            # Common patterns: "encoder.layer.0", "layers.0", "encoder.layers.0"
            if "layer" in name:
                # Extract layer index
                try:
                    parts = name.split(".")
                    # Find the part that is a number
                    layer_idx = -1
                    for part in parts:
                        if part.isdigit():
                            layer_idx = int(part)
                            break

                    if layer_idx != -1 and layer_idx < freeze_layers_count:
                        param.requires_grad = False
                except Exception:
                    # If parsing fails, skip freezing for safety
                    pass

    def _init_weights(self, module):
        """
        Initialize weights for the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def feature(self, input_ids, attention_mask):
        """
        Extracts features from the backbone using Mean Pooling.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Mean Pooling
        # Expand attention mask to match hidden state dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings where mask is 1
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum mask to get count of tokens (avoid div by zero)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        """
        feature = self.feature(input_ids, attention_mask)
        out = self.drop(feature)
        logits = self.fc(out)
        return logits


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    Perturbs model weights to flatten the loss landscape and improve generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=ModelConfig.awp_lr,
        adv_eps=ModelConfig.awp_eps,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Saves original weights and applies adversarial perturbation.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            # Only perturb parameters that require gradients and match the target (usually weights)
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):

                # Save original weights
                self.backup[name] = param.data.clone()

                # Calculate perturbation
                grad = param.grad
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data)

                # Formula: delta = adv_lr * (grad / |grad|) * |weight|
                if norm_grad != 0 and not torch.isnan(norm_grad):
                    r_at = self.adv_lr * grad / (norm_grad + e) * (norm_data + e)

                    # Apply perturbation to weights
                    param.data.add_(r_at)

                    # Clamp perturbation magnitude if needed (optional based on implementation,
                    # but here we rely on the scale factor logic. Some implementations explicitly
                    # project back to epsilon ball. We store the perturbation to ensure we can restore.)

                    # For stricter epsilon constraint (Projected Gradient Descent style):
                    # We can clamp the total change. Here we follow a standard AWP implementation.
                    # To strictly respect adv_eps as a max magnitude of perturbation:
                    # perturbation = torch.clamp(r_at, -self.adv_eps, self.adv_eps)
                    # param.data = self.backup[name] + perturbation

                    # Using the simpler scaling approach often used in competition kernels:
                    # We check if the perturbation exceeds eps relative to weight or absolute.
                    # Let's stick to the simplest effective form:
                    # param.data.add_(r_at)

                    # To ensure stability, we will clamp the weight update to be within epsilon range
                    # relative to the original weight if needed, but usually the LR controls this.
                    # Let's add a min/max clamp based on eps for safety.
                    diff = param.data - self.backup[name]
                    diff = torch.clamp(diff, -self.adv_eps, self.adv_eps)
                    param.data = self.backup[name] + diff

    def restore(self):
        """
        Restores the original weights from backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
