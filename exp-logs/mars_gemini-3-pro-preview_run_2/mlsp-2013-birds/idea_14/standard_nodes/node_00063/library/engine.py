import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config


class EarlyStopping:
    """
    Early stopping utility to stop training when the monitored metric does not improve.
    """

    def __init__(self, patience=5, min_delta=0.0, mode="max"):
        """
        Args:
            patience (int): How many epochs to wait before stopping.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): 'max' for metrics where higher is better (e.g., AUC), 'min' for loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "max":
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        elif self.mode == "min":
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0


def mixup_data(x, y, alpha=0.4, device=None):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs and mixed targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device is None:
        device = x.device

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    # Mix targets directly (works for both hard one-hot and soft probabilities)
    mixed_y = lam * y + (1 - lam) * y[index, :]

    return mixed_x, mixed_y


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply Mixup
        inputs, targets = mixup_data(
            inputs, targets, alpha=Config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Macro ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Robust Column-wise AUC calculation
    # This handles cases where a fold might miss a specific class entirely
    aucs = []
    for i in range(all_targets.shape[1]):
        # Only calculate AUC if there are at least two classes (0 and 1) present
        if len(np.unique(all_targets[:, i])) > 1:
            try:
                score = roc_auc_score(all_targets[:, i], all_preds[:, i])
                aucs.append(score)
            except ValueError:
                pass

    if len(aucs) > 0:
        auc_score = np.mean(aucs)
    else:
        # Fallback if validation set is too small or homogeneous
        auc_score = 0.5

    return epoch_loss, auc_score


def inference(model, dataloader, device):
    """
    Generates predictions for the test set.
    Returns an array of probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
