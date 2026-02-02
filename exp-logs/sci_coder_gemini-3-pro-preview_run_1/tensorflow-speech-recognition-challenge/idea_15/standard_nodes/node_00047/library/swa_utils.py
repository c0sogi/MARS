import os
import torch
from torch.optim.swa_utils import AveragedModel, update_bn


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA) for a PyTorch model.
    Encapsulates the creation, updating, and batch normalization statistics
    re-calculation for the averaged model.
    """

    def __init__(self, model, device=None):
        """
        Initialize the SWA Handler.

        Args:
            model (torch.nn.Module): The base model to be averaged.
            device (torch.device, optional): The device to run computations on.
                                           If None, detects CUDA/CPU.
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Create the AveragedModel.
        # By default, this uses a simple average (1/(n+1) weighting).
        # We move it to the specified device immediately.
        self.swa_model = AveragedModel(model).to(self.device)

        # Counter for number of updates (optional tracking)
        self.n_averaged = 0

    def update_average(self, model):
        """
        Update the averaged model parameters with the current model's parameters.
        This should be called at the end of each epoch during the SWA phase.

        Args:
            model (torch.nn.Module): The current training model.
        """
        # Ensure the source model is on the correct device or CPU before update if needed.
        # AveragedModel.update_parameters handles the update logic.
        self.swa_model.update_parameters(model)
        self.n_averaged += 1

    def update_bn_statistics(self, loader):
        """
        Update the Batch Normalization statistics (mean and variance) of the
        averaged model by performing a forward pass over the training data.

        This corresponds to Phase 3 of the SWA pipeline.

        Args:
            loader (torch.utils.data.DataLoader): The training data loader.
        """
        # Set model to training mode to enable BN statistics tracking
        self.swa_model.train()

        # update_bn performs a forward pass on the data in the loader.
        # It assumes the loader yields either x or (x, y).
        # It updates the running_mean and running_var of BN layers.
        # Note: This operation can be time-consuming as it runs over the dataset.
        update_bn(loader, self.swa_model, device=self.device)

    def get_averaged_model(self):
        """
        Retrieve the underlying averaged model module.

        Returns:
            torch.nn.Module: The averaged model (ready for inference or saving).
        """
        # AveragedModel wraps the actual model in a 'module' attribute.
        return self.swa_model.module

    def save_model(self, path):
        """
        Save the state dictionary of the averaged model to a file.

        Args:
            path (str): The file path to save the model.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the underlying module's state dict
        torch.save(self.swa_model.module.state_dict(), path)

    def load_model(self, path):
        """
        Load weights into the averaged model from a file.

        Args:
            path (str): The file path to load the model from.
        """
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            self.swa_model.module.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {path}")
