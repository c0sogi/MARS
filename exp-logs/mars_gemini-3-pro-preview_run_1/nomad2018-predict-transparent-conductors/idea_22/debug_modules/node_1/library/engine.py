import torch
import numpy as np
import os
import pandas as pd
from library.utils import rmsle


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to train on.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        atomic = batch["atomic"].to(device)
        glob = batch["global"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        outputs = model(atomic, glob, mask)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * atomic.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        val_loss: Average loss on the validation set.
        rmsle_form: RMSLE for formation energy.
        rmsle_band: RMSLE for bandgap energy.
    """
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            atomic = batch["atomic"].to(device)
            glob = batch["global"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(atomic, glob, mask)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * atomic.size(0)

            # Inverse transform for metric calculation: exp(x) - 1
            # Clamp to 0 to avoid negative energies which are physically impossible/invalid for RMSLE
            preds_orig = torch.expm1(outputs).cpu().numpy()
            preds_orig = np.maximum(preds_orig, 0.0)

            targets_orig = torch.expm1(targets).cpu().numpy()
            targets_orig = np.maximum(targets_orig, 0.0)

            preds_list.append(preds_orig)
            targets_list.append(targets_orig)

    val_loss = running_loss / len(dataloader.dataset)

    all_preds = np.concatenate(preds_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)

    # Calculate RMSLE for each column
    rmsle_form = rmsle(all_targets[:, 0], all_preds[:, 0])
    rmsle_band = rmsle(all_targets[:, 1], all_preds[:, 1])

    return val_loss, rmsle_form, rmsle_band


def generate_predictions(model, test_loader, device, output_path=None):
    """
    Generates predictions for the test set.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for test data.
        device: Device to run inference on.
        output_path: Optional path to save the submission CSV.

    Returns:
        DataFrame containing IDs and predictions.
    """
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            atomic = batch["atomic"].to(device)
            glob = batch["global"].to(device)
            mask = batch["mask"].to(device)
            batch_ids = batch["id"]

            outputs = model(atomic, glob, mask)

            # Inverse transform: exp(x) - 1
            preds_orig = torch.expm1(outputs).cpu().numpy()
            preds_orig = np.maximum(preds_orig, 0.0)

            ids_list.extend(batch_ids)
            preds_list.append(preds_orig)

    all_preds = np.concatenate(preds_list, axis=0)

    df = pd.DataFrame(
        {
            "id": ids_list,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Sort by ID to ensure correct order
    df = df.sort_values("id")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
