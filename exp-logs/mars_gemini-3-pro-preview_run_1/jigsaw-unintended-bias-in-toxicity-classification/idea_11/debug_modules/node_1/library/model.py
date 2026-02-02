import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class ToxicityModel(nn.Module):
    """
    Multi-Task Learning Model for Toxicity Classification.

    Backbone: DeBERTa-v3 (Large or Base)
    Heads:
      1. Primary Toxicity Head (Binary Classification)
      2. Identity Head (Multi-label Classification for Bias Mitigation)
      3. Auxiliary Head (Multi-label Classification for Toxicity Subtypes)
    """

    def __init__(self, model_name=None, pretrained=True):
        super(ToxicityModel, self).__init__()

        # Use config model name if not provided (allows overriding for Scout model)
        if model_name is None:
            model_name = Config.model_name

        print(f"Initializing ToxicityModel with backbone: {model_name}")

        # Load Configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # 1. Primary Toxicity Head
        self.toxicity_head = nn.Linear(self.config.hidden_size, 1)

        # 2. Identity Head (for Bias Mitigation/Mining)
        self.identity_head = nn.Linear(
            self.config.hidden_size, len(Config.identity_cols)
        )

        # 3. Auxiliary Head (for Toxicity Subtypes)
        self.aux_head = nn.Linear(self.config.hidden_size, len(Config.aux_cols))

        # Initialize weights for heads
        self._init_weights(self.toxicity_head)
        self._init_weights(self.identity_head)
        self._init_weights(self.aux_head)

    def _init_weights(self, module):
        """Initialize the weights of the heads."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs (if used).

        Returns:
            tuple: (toxicity_logits, identity_logits, aux_logits)
        """
        # Pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        # Use CLS token representation (first token)
        # Shape: (Batch_Size, Hidden_Size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        features = self.dropout(cls_embedding)

        # Pass through heads
        toxicity_logits = self.toxicity_head(features)
        identity_logits = self.identity_head(features)
        aux_logits = self.aux_head(features)

        return toxicity_logits, identity_logits, aux_logits


class AWP:
    """
    Adversarial Weight Perturbation (AWP).

    Injects adversarial perturbations into the model weights to flatten the loss landscape
    and improve generalization/robustness.
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
        """
        Args:
            model (nn.Module): The model to attack.
            optimizer (optim.Optimizer): The optimizer used for training.
            adv_param (str): Name of parameters to attack (default: "weight").
            adv_lr (float): Magnitude of the attack step (relative to gradient).
            adv_eps (float): Maximum perturbation norm (epsilon).
            start_epoch (int): Epoch to start applying AWP.
            scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler

        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, criterion, epoch):
        """
        Performs the AWP attack step and backward pass.

        This method encapsulates the logic:
        1. If epoch < start_epoch, do nothing.
        2. Save current weights.
        3. Perturb weights based on gradients (attack).
        4. Forward + Backward on perturbed weights.
        5. Restore original weights.

        Note: This assumes the standard backward() has already been called once
        to populate gradients for the attack direction.

        Args:
            inputs (dict): Dictionary of model inputs (input_ids, etc.).
                           Must also contain targets for loss calculation.
            criterion (callable): Loss function.
            epoch (int): Current training epoch.

        Returns:
            float: The adversarial loss value (or 0.0 if not active).
        """
        if epoch < self.start_epoch:
            return 0.0

        self._save()
        self._attack_step()

        # Forward pass with perturbed weights
        # We assume inputs contains everything needed for the model and loss
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Targets
        targets = inputs["target"]
        aux_targets = inputs["aux_target"]
        # Identity targets might be needed depending on loss signature,
        # but usually contained in aux or separate.
        # Based on Config/Losses, we need specific targets.
        # We extract them assuming standard batch structure.

        # Re-run forward
        tox_logits, ident_logits, aux_logits = self.model(input_ids, attention_mask)

        # Calculate adversarial loss
        # Note: We don't need sample weights for the AWP step usually,
        # or we reuse them if available. Here we assume simple mean or reuse.
        # To be safe and consistent with library.losses, we pass what is available.
        # If sample_weights are in inputs, use them.
        sample_weights = inputs.get("weight", None)

        adv_loss, _ = criterion(
            tox_logits,
            targets,
            torch.cat(
                [ident_logits, aux_logits], dim=1
            ),  # Combined aux logits for loss signature if needed?
            # Actually library.losses.JigsawLoss takes:
            # toxicity_logits, toxicity_targets, aux_logits, aux_targets, sample_weights
            # But aux_logits in loss signature is one tensor.
            # The model returns ident and aux separately.
            # We must concatenate them to match JigsawLoss expectation if it treats them as one block,
            # OR pass them correctly.
            # Looking at JigsawLoss in description:
            # "aux_logits (torch.Tensor): Logits from auxiliary heads (Identities + Subtypes)."
            # So we concatenate.
            torch.cat([inputs["identity_target"], aux_targets], dim=1),
            sample_weights,
        )

        # Backward pass for adversarial loss
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(adv_loss).backward()
        else:
            adv_loss.backward()

        self._restore()

        return adv_loss.item()

    def _attack_step(self):
        """Perturbs the model weights."""
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Exclude LayerNorm and Bias from attack for stability
                if "LayerNorm" in name or "bias" in name:
                    continue

                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data)

                if norm1 != 0 and not torch.isnan(norm1):
                    # Calculate perturbation
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Clamp perturbation magnitude to epsilon
                    # We want to add r_at such that it doesn't exceed adv_eps relative to data?
                    # Standard AWP implementation:
                    # param.data.add_(r_at)
                    # Then clamp relative to backup if needed, but usually just adding scaled grad is enough
                    # if adv_lr is small.
                    # However, strict implementation often does:
                    # r_at = self.adv_lr * param.grad / (norm1 + e)
                    # param.data.add_(r_at)
                    # min/max clamp usually done if we track perturbation tensor.

                    # Here we follow a standard implementation:
                    param.data.add_(r_at)

                    # Store perturbation size for clamping (optional, simplified here)
                    # If we wanted strict epsilon ball projection:
                    # diff = param.data - self.backup[name]
                    # diff = torch.clamp(diff, -self.adv_eps, self.adv_eps) # simplified
                    # param.data = self.backup[name] + diff

    def _save(self):
        """Saves the current model weights."""
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if "LayerNorm" not in name and "bias" not in name:
                    self.backup[name] = param.data.clone()

    def _restore(self):
        """Restores the saved model weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
