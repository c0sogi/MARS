import copy
import torch
import torch.nn as nn
from library.config import Config


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA) for a PyTorch model.
    Maintains a running average of model parameters (weights and biases),
    ignoring buffers (like BatchNorm statistics) which are handled separately.
    """

    def __init__(self, model):
        """
        Initialize the SWA Handler.

        Args:
            model (torch.nn.Module): The base model architecture. A deep copy is created
                                     to store the averaged weights.
        """
        # Create a deep copy of the model to hold the averaged parameters.
        # We move it to the configured device immediately.
        self.swa_model = copy.deepcopy(model).to(Config.DEVICE)
        self.n_averaged = 0

    def update(self, model):
        """
        Update the running average with the parameters from the current model state.

        Args:
            model (torch.nn.Module): The current training model.
        """
        self.n_averaged += 1
        alpha = 1.0 / self.n_averaged

        # Iterate over parameters (weights/biases) only, ignoring buffers.
        for swa_param, param in zip(self.swa_model.parameters(), model.parameters()):
            # Ensure the new parameter is on the same device as the stored SWA parameter
            param_data = param.data.to(swa_param.device)

            if self.n_averaged == 1:
                # For the first update, simply copy the weights
                swa_param.data.copy_(param_data)
            else:
                # Update running average: avg = avg * (1 - alpha) + new * alpha
                swa_param.data.mul_(1.0 - alpha).add_(param_data, alpha=alpha)

    def get_averaged_model(self):
        """
        Retrieve the model with the averaged weights.

        Returns:
            torch.nn.Module: The SWA model.
        """
        return self.swa_model


def update_bn_statistics(model, data_loader, device=None):
    """
    Recalibrates the BatchNorm statistics (running_mean, running_var) of the model
    by performing a forward pass on the provided training data.

    This is required because the averaged weights from SWA do not have corresponding
    valid statistics in the BatchNorm layers.

    Args:
        model (torch.nn.Module): The SWA model to update.
        data_loader (torch.utils.data.DataLoader): Loader containing training data.
        device (str, optional): Device to perform computations on. Defaults to Config.DEVICE.
    """
    if device is None:
        device = Config.DEVICE

    model.to(device)
    model.train()  # Set to train mode to allow updating of BN stats

    # 1. Reset BatchNorm Statistics
    # We reset running stats and set momentum to None to use simple cumulative average
    # during the recalibration pass.
    momenta = {}
    bn_layers = []

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            module.num_batches_tracked = torch.tensor(
                0, dtype=torch.long, device=device
            )

            momenta[module] = module.momentum
            module.momentum = None  # Cumulative average
            bn_layers.append(module)

    if not bn_layers:
        # If no BN layers, just return
        model.eval()
        return

    # 2. Forward Pass
    # We pass data through the model. We don't need gradients.
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # Unpack batch based on BirdDataset structure: image, target, rec_id
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images = batch[0]
            else:
                images = batch

            images = images.to(device)

            # Forward pass triggers BN updates
            model(images)

    # 3. Restore Momentum
    # Restore original momentum values for future consistency (though usually model is used for eval)
    for module in bn_layers:
        module.momentum = momenta[module]

    # Set model to evaluation mode for inference
    model.eval()
