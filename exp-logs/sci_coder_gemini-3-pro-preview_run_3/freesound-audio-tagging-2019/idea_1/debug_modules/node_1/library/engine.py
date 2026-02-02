import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config


class EarlyStopping:
    """
    Early stopping to stop the training when the metric does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0, mode="max"):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                quantity monitored has stopped decreasing; in 'max' mode it will stop when the
                quantity monitored has stopped increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        else:
            if self.mode == "max":
                improvement = score - self.best_score
            else:
                improvement = self.best_score - score

            if improvement > self.min_delta:
                self.best_score = score
                self.best_model_state = model.state_dict()
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True


def calculate_lwlrap(truth, scores):
    """
    Calculates Label-Weighted Label-Ranking Average Precision (LWLRAP).

    Args:
        truth (np.array): Binary ground truth matrix [n_samples, n_classes].
        scores (np.array): Predicted probability matrix [n_samples, n_classes].

    Returns:
        float: The overall LWLRAP score.
    """
    if isinstance(truth, torch.Tensor):
        truth = truth.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape

    # Sort scores in descending order
    # idx contains the indices of the classes sorted by score
    idx = np.argsort(-scores, axis=1)

    # Rearrange truth to match the sorted score order
    # sorted_truth[i, j] is the ground truth label for the j-th highest scored class in sample i
    sorted_truth = np.take_along_axis(truth, idx, axis=1)

    # Calculate cumulative hits (number of true positives found up to rank k)
    cum_hits = np.cumsum(sorted_truth, axis=1)

    # Calculate precision at each rank (1-based)
    ranks = np.arange(1, num_classes + 1)
    precisions = cum_hits / ranks

    # We only care about the precision at the ranks where the label is actually True
    relevant_precisions = precisions * sorted_truth

    # Sum relevant precisions for each class across all samples
    # We map the values back to their original class indices using idx
    per_class_prec_sum = np.zeros(num_classes)
    np.add.at(per_class_prec_sum, idx.flatten(), relevant_precisions.flatten())

    # Count total occurrences of each class in the ground truth
    per_class_counts = np.sum(truth, axis=0)

    # Calculate LWLRAP per class
    # Handle division by zero for classes that don't appear in the batch/dataset
    lwlrap_per_class = np.zeros(num_classes)
    mask = per_class_counts > 0
    lwlrap_per_class[mask] = per_class_prec_sum[mask] / per_class_counts[mask]

    # The final score is the average of per-class scores (label-weighted)
    if mask.sum() > 0:
        return np.mean(lwlrap_per_class[mask])
    else:
        return 0.0


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # Accumulate loss (multiply by batch size to get total, then divide later)
        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation data loader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (average_loss, lwlrap_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid to get probabilities for metric calculation
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        lrap = calculate_lwlrap(all_targets, all_outputs)
    else:
        lrap = 0.0

    return epoch_loss, lrap


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    patience=5,
):
    """
    Runs the full training loop with early stopping.

    Args:
        model: The model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device.
        num_epochs: Max epochs.
        patience: Early stopping patience.

    Returns:
        model: The model with the best weights loaded.
    """
    early_stopping = EarlyStopping(patience=patience, mode="max")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_lrap = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val LRAP: {val_lrap:.6f}"
        )

        early_stopping(val_lrap, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model weights
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    return model


def predict(model, dataloader, device):
    """
    Generates predictions for a dataset.

    Args:
        model: Trained model.
        dataloader: DataLoader (test mode).
        device: Device.

    Returns:
        tuple: (list of filenames, np.array of probabilities)
    """
    model.eval()
    predictions = []
    fnames = []

    with torch.no_grad():
        for inputs, batch_fnames in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            predictions.append(probs.cpu().numpy())
            fnames.extend(batch_fnames)

    if len(predictions) > 0:
        predictions = np.concatenate(predictions, axis=0)
    else:
        predictions = np.array([])

    return fnames, predictions


def generate_submission(model, test_loader, device, output_path):
    """
    Generates and saves the submission CSV.

    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        device: Device.
        output_path: Path to save the CSV.
    """
    print("Generating submission...")
    fnames, preds = predict(model, test_loader, device)

    # Get class names from sample submission to ensure correct column order
    ss = pd.read_csv(Config.SAMPLE_SUBMISSION)
    class_names = ss.columns[1:].tolist()

    # Create DataFrame
    sub_df = pd.DataFrame(preds, columns=class_names)
    sub_df.insert(0, "fname", fnames)

    # Align with sample submission order (just to be safe)
    ss_fnames = ss["fname"].values

    # Set index to fname for reindexing
    sub_df = sub_df.set_index("fname")

    # Reindex to match sample submission file order
    # Fill missing (if any) with 0, though there shouldn't be any
    sub_df = sub_df.reindex(ss_fnames, fill_value=0)
    sub_df = sub_df.reset_index()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
