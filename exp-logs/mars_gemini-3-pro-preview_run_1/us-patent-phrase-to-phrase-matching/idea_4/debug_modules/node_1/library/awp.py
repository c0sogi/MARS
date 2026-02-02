import torch
import torch.nn as nn


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Regularizes the model by injecting worst-case perturbations into the weights
    and optimizing against them.
    """

    def __init__(self, model, optimizer, adv_lr, adv_eps, start_epoch=0, scaler=None):
        """
        Args:
            model (nn.Module): The model to attack.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_lr (float): The magnitude of the perturbation step (learning rate for attack).
            adv_eps (float): The maximum allowed perturbation (epsilon constraint).
            start_epoch (float): The epoch to start applying AWP.
            scaler (torch.cuda.amp.GradScaler, optional): Scaler for mixed precision training.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}

    def attack_backward(self, inputs, epoch):
        """
        Performs the adversarial attack and backward pass.

        Steps:
        1. Saves current weights.
        2. Perturbs weights based on current gradients.
        3. Computes loss on perturbed weights (Forward).
        4. Updates gradients (Backward) on the adversarial loss.
        5. Restores original weights.

        Args:
            inputs (dict): Dictionary of inputs for the model (must include 'labels').
            epoch (float): Current training epoch.
        """
        # Only execute if AWP is active for this epoch
        if epoch < self.start_epoch:
            return

        # 1. Save original weights
        self._save()

        # 2. Perturb weights
        self._attack_step()

        # 3. Forward & 4. Backward on perturbed model
        # We assume inputs contains 'labels' so the model returns loss
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            # We zero out the 'clean' gradients and replace them with adversarial gradients
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            self.optimizer.zero_grad()
            loss.backward()

        # 5. Restore original weights
        self.restore()

    def restore(self):
        """
        Restores the original weights from the backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to free memory
        self.backup = {}

    def _save(self):
        """
        Saves the current values of target parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_lr != 0:
                if not self._is_target_param(name):
                    continue
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def _attack_step(self):
        """
        Calculates and applies the adversarial perturbation.
        Formula: w_adv = w + adv_lr * (grad / |grad|) * |w|
        Constrained by: |w_adv - w| <= adv_eps
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_lr != 0:
                if not self._is_target_param(name):
                    continue

                # Calculate norms
                grad_norm = torch.norm(param.grad)
                weight_norm = torch.norm(param.data.detach())

                if grad_norm != 0 and not torch.isnan(grad_norm):
                    # Compute perturbation direction and scale
                    # delta = eta * (g / ||g||) * ||w||
                    r_at = (
                        self.adv_lr * param.grad / (grad_norm + e) * (weight_norm + e)
                    )

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Enforce Epsilon Constraint (Projection)
                    # w = min(max(w, w_orig - eps), w_orig + eps)
                    if self.adv_eps > 0:
                        backup_w = self.backup[name]
                        param.data = torch.min(
                            torch.max(param.data, backup_w - self.adv_eps),
                            backup_w + self.adv_eps,
                        )

    def _is_target_param(self, name):
        """
        Determines if a parameter should be perturbed.
        Target: Weights only.
        Exclude: Bias, LayerNorm, and Embeddings (optional, but usually kept for Transformers).
        Here we exclude Bias and LayerNorm for stability.
        """
        return "weight" in name and "LayerNorm" not in name and "bias" not in name
