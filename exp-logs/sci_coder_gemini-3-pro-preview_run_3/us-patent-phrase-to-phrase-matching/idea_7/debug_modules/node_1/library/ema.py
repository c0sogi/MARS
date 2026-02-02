import torch


class ModelEMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    Maintains a shadow copy of the model weights and updates them using the formula:
    theta_EMA = decay * theta_EMA + (1 - decay) * theta_train
    """

    def __init__(self, model, decay):
        """
        Args:
            model: The model to track.
            decay (float): The decay factor for EMA (e.g., 0.999).
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Register the initial model parameters
        self._register(model)

    def _register(self, model):
        """
        Registers the current model parameters as the initial shadow parameters.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Clone to detach from computation graph and store in separate memory
                    self.shadow[name] = param.detach().clone()

    def update(self, model):
        """
        Updates the shadow parameters using the current model parameters.
        Formula: shadow = decay * shadow + (1 - decay) * current
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name not in self.shadow:
                        # If a parameter was not registered (e.g., dynamic graph), register it now
                        self.shadow[name] = param.detach().clone()
                        continue

                    # Ensure shadow parameter is on the same device as the model parameter
                    if self.shadow[name].device != param.device:
                        self.shadow[name] = self.shadow[name].to(param.device)

                    # Apply EMA update
                    # shadow = decay * shadow + (1 - decay) * param
                    self.shadow[name].mul_(self.decay).add_(
                        param.detach(), alpha=(1.0 - self.decay)
                    )

    def apply_shadow(self, model):
        """
        Loads the smoothed (shadow) weights into the provided model for validation or inference.
        Backs up the current model weights to allow restoration.
        """
        self.backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # Backup current parameter
                    self.backup[name] = param.detach().clone()

                    # Load shadow parameter into model
                    # Ensure device compatibility before copy
                    if self.shadow[name].device != param.device:
                        self.shadow[name] = self.shadow[name].to(param.device)

                    param.data.copy_(self.shadow[name])

    def restore(self, model):
        """
        Restores the original model weights from the backup.
        Should be called after validation/inference is complete if training is to continue.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.data.copy_(self.backup[name])

        # Clear backup to free memory
        self.backup = {}
