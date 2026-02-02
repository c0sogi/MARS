import torch
import torch.nn as nn
import copy
from library.config import Config
from library.utils import AverageMeter, get_score


class ModelEMA:
    """
    Model Exponential Moving Average.
    Maintains a moving average of model parameters for better generalization.
    """

    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.model = model
        # Create a deep copy of the model for EMA
        self.ema = copy.deepcopy(model)
        self.ema.eval()

        # Move to device if specified, otherwise use model's device
        if device is not None:
            self.ema.to(device)

        # Disable gradients for EMA model
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update EMA parameters.
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for k, v in esd.items():
                if k in msd:
                    model_v = msd[k].detach()
                    if v.dtype.is_floating_point:
                        v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
                    else:
                        # Copy non-floating point parameters (e.g. buffers)
                        v.copy_(model_v)


def train_one_epoch(
    model, optimizer, scheduler, data_loader, device, epoch, ema_model=None
):
    """
    Trains the model for one epoch.
    """
    model.train()

    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model handles Multi-Sample Dropout internally during training
        logits = model(images)

        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()

        # Scheduler step (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        # Update EMA
        if ema_model is not None:
            ema_model.update(model)

        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch} Train Loss: {losses.avg}")

    return losses.avg


def validate(model, data_loader, device):
    """
    Validates the model on the validation set.
    """
    model.eval()

    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid for scoring
            probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu())
            targets_list.append(labels.cpu())

    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    score = get_score(targets, preds)

    # Print full precision as requested
    print(f"Validation Loss: {losses.avg}")
    print(f"Validation AUC: {score}")

    return losses.avg, score
