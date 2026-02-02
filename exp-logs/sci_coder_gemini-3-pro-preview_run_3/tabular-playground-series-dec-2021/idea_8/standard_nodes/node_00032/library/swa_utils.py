import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn


class SWAHandler:
    """
    Encapsulates Stochastic Weight Averaging (SWA) logic.
    Wraps torch.optim.swa_utils.AveragedModel and manages the SWA learning rate scheduler.
    """

    def __init__(self, model, optimizer, swa_lr):
        """
        Initialize the SWA Handler.

        Args:
            model (torch.nn.Module): The base model to be averaged.
            optimizer (torch.optim.Optimizer): The optimizer used during training.
            swa_lr (float): The constant learning rate to be used during the SWA phase.
        """
        # Initialize the AveragedModel which maintains a running average of parameters
        self.swa_model = AveragedModel(model)

        # Initialize the SWA Learning Rate Scheduler
        # This scheduler sets the learning rate to a fixed value (swa_lr)
        # or cycles it, depending on configuration. Here we use the standard SWALR.
        self.swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    def update_average(self, model):
        """
        Update the averaged model parameters and step the SWA scheduler.
        This should be called at the end of each epoch during the SWA phase.

        Args:
            model (torch.nn.Module): The current state of the model to be incorporated into the average.
        """
        # Update the running average of parameters
        self.swa_model.update_parameters(model)

        # Step the SWA scheduler
        self.swa_scheduler.step()

    def update_bn_statistics(self, train_loader, device):
        """
        Perform a forward pass over the training data to recalibrate
        Batch Normalization statistics for the averaged model.
        This is crucial because the averaged weights do not track running_mean/var
        during training.

        Args:
            train_loader (DataLoader): The data loader for the training set.
            device (torch.device): The device (CPU/GPU) to perform the computation on.
        """
        print("Updating SWA Batch Normalization statistics...")
        update_bn(train_loader, self.swa_model, device=device)

    def get_averaged_model(self):
        """
        Retrieve the final averaged model.

        Returns:
            torch.nn.Module: The averaged model (AveragedModel instance).
        """
        return self.swa_model
