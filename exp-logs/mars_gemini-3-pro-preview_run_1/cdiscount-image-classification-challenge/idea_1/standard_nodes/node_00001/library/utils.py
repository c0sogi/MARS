import os
import shutil
import torch
import torchvision.transforms as transforms
from library.config import Config


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(output, target):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    For this task, we focus on Top-1 accuracy.
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the training checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer, etc.
        is_best (bool): Whether this is the best model seen so far.
        filename (str): Filename to save the checkpoint to.
    """
    # Save the current state to the working directory
    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    # If this is the best model, copy it to the designated best model path
    if is_best:
        shutil.copyfile(filepath, Config.MODEL_CHECKPOINT)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Loads a checkpoint into the model and optional optimizer/scheduler.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.

    Returns:
        epoch (int): The epoch to resume from.
        best_acc (float): The best accuracy recorded in the checkpoint.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")

    checkpoint = torch.load(filename, map_location=Config.DEVICE)

    # Load model state
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)
    best_acc = checkpoint.get("best_acc", 0.0)

    return epoch, best_acc


def get_transforms(phase):
    """
    Returns the image transformations for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=Config.MEAN, std=Config.STD)

    if phase == "train":
        return transforms.Compose(
            [
                # Images are already 180x180, so no resize needed usually.
                # However, RandomCrop could be used if we had larger images.
                # Here we stick to simple augmentation for speed.
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        # Validation and Test
        return transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
