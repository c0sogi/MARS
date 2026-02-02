import torch
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from library.config import Config
from library.utils import AverageMeter, calculate_auc


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss (weighted sum of losses).
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Executes one training epoch with Mixup and Gradient Accumulation.
    """
    model.train()
    losses = AverageMeter()

    use_mixup = Config.USE_MIXUP
    mixup_alpha = Config.MIXUP_ALPHA
    accum_steps = Config.GRADIENT_ACCUMULATION_STEPS

    optimizer.zero_grad()

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, mixup_alpha, device
            )
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Gradient Accumulation
        # Normalize loss to account for accumulation
        loss = loss / accum_steps
        loss.backward()

        # Step optimizer every 'accum_steps' iterations
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Record unscaled loss
        losses.update(loss.item() * accum_steps, images.size(0))

    # Process remaining gradients if batch count is not divisible by accum_steps
    if len(loader) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return losses.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    losses = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

            # Convert logits to probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    if len(all_preds) == 0:
        return 0.0, 0.5

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    auc = calculate_auc(all_targets, all_preds)

    return losses.avg, auc


class SWAManager:
    """
    Manages Stochastic Weight Averaging (SWA) transition and updates.
    """

    def __init__(self, model, optimizer):
        # Initialize Config to access dynamic properties
        self.config = Config()
        self.swa_start_epoch = self.config.SWA_START_EPOCH
        self.swa_lr = Config.SWA_LR

        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(optimizer, swa_lr=self.swa_lr)
        self.is_swa_phase = False

    def step(self, epoch, model, base_scheduler=None):
        """
        Steps the appropriate scheduler and updates SWA model if in SWA phase.
        """
        if epoch >= self.swa_start_epoch:
            if not self.is_swa_phase:
                print(
                    f"Epoch {epoch}: Switching to SWA Phase. LR fixed at {self.swa_lr}"
                )
                self.is_swa_phase = True

            # Update SWA model parameters with current model
            self.swa_model.update_parameters(model)
            # Step SWA scheduler (keeps LR constant)
            self.swa_scheduler.step()
        else:
            self.is_swa_phase = False
            # Step standard scheduler
            if base_scheduler:
                base_scheduler.step()

    def finalize(self, loader, device):
        """
        Performs the final Batch Normalization update for the SWA model.
        Returns the finalized SWA model.
        """
        print("SWA: Updating Batch Normalization statistics...")
        update_bn(loader, self.swa_model, device=device)
        return self.swa_model
