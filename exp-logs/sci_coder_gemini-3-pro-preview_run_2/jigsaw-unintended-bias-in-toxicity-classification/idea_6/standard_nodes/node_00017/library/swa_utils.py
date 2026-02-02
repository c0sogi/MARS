import torch
from torch.optim.swa_utils import AveragedModel
from library.config import Config


class SWAHandler:
    """
    Handles Stochastic Weight Averaging (SWA) for model training.

    This class manages the accumulation of model weights starting from a specific epoch
    and provides functionality to swap the current model weights with the averaged weights.
    """

    def __init__(
        self, model, swa_start_epoch=Config.SWA_START_EPOCH, swa_lr=Config.SWA_LR
    ):
        """
        Initialize the SWA Handler.

        Args:
            model (torch.nn.Module): The model to track.
            swa_start_epoch (int): The epoch number to start averaging weights.
            swa_lr (float): The learning rate to be used during the SWA phase.
        """
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr

        # Initialize AveragedModel which maintains a running average of parameters.
        # It creates a deep copy of the model on the same device.
        self.swa_model = AveragedModel(model)
        self.n_averaged = 0

    def update_average(self, model, current_epoch):
        """
        Update the SWA model parameters if the current epoch is within the SWA schedule.

        Args:
            model (torch.nn.Module): The current model state.
            current_epoch (int): The current training epoch index.
        """
        if current_epoch >= self.swa_start_epoch:
            # Update the running average of parameters
            self.swa_model.update_parameters(model)
            self.n_averaged += 1

    def swap_swa_params(self, model):
        """
        Replace the parameters of the given model with the averaged parameters.

        This is typically called at the end of training before final evaluation or inference.
        Note: BatchNorm statistics update is skipped because RoBERTa uses LayerNorm.

        Args:
            model (torch.nn.Module): The model to update with averaged weights.
        """
        if self.n_averaged > 0:
            # Load the averaged state dict into the provided model.
            # AveragedModel stores the averaged parameters in its 'module' attribute.
            model.load_state_dict(self.swa_model.module.state_dict())
