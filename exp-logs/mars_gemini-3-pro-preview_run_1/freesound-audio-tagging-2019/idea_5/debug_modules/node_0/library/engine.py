import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, calculate_lwlrap, get_logger

logger = get_logger("engine")


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to inputs and targets.
    Returns mixed inputs, pairs of targets, and lambda.
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
    Computes the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch using Mixup.
    """
    model.train()
    losses = AverageMeter()

    # Ensure Mixup is applied as per config
    use_mixup = Config.MIXUP and (np.random.rand() < Config.MIXUP_PROB)

    for batch_idx, (data, target, _) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if use_mixup:
            mixed_data, target_a, target_b, lam = mixup_data(
                data, target, Config.MIXUP_ALPHA, device
            )
            output = model(mixed_data)
            loss = mixup_criterion(
                nn.BCEWithLogitsLoss(), output, target_a, target_b, lam
            )
        else:
            output = model(data)
            loss = nn.BCEWithLogitsLoss()(output, target)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), data.size(0))

    if scheduler is not None:
        scheduler.step()

    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and LWLRAP score.
    """
    model.eval()
    losses = AverageMeter()

    # Containers for full dataset evaluation
    all_targets = []
    all_scores = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data, target, _ in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            losses.update(loss.item(), data.size(0))

            # Apply sigmoid for metric calculation
            scores = torch.sigmoid(output)

            all_targets.append(target.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_scores = np.concatenate(all_scores)

    # Calculate metric
    score_lwlrap = calculate_lwlrap(all_targets, all_scores)

    return losses.avg, score_lwlrap


def inference(model, loader, device):
    """
    Runs inference on a dataset.
    Returns a dictionary mapping fname -> prediction vector (probabilities).
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for data, _, fnames in loader:
            data = data.to(device)
            output = model(data)
            scores = torch.sigmoid(output).cpu().numpy()

            for fname, score in zip(fnames, scores):
                results[fname] = score

    return results


def get_or_compute_teacher_predictions(
    model, loader, device, cache_path, load_cached_data=True
):
    """
    Retrieves soft labels for the noisy dataset.
    Implements caching mechanism:
    1. If load_cached_data is True and file exists, load it.
    2. Otherwise, compute using the model and save to cache.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached teacher predictions from {cache_path}")
        try:
            preds_dict = np.load(cache_path, allow_pickle=True).item()
            return preds_dict
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    logger.info("Computing teacher predictions (Soft Labels)...")
    preds_dict = inference(model, loader, device)

    logger.info(f"Saving teacher predictions to {cache_path}")
    np.save(cache_path, preds_dict)

    return preds_dict


class Trainer:
    """
    Manages the training process, including Early Stopping and Checkpointing.
    """

    def __init__(self, model, optimizer, scheduler, device, save_path, patience=5):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.patience = patience

        self.best_score = -np.inf
        self.counter = 0
        self.early_stop = False

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = train_one_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.scheduler,
                self.device,
                epoch,
            )

            # Validate
            val_loss, val_score = validate(self.model, val_loader, self.device)

            elapsed = time.time() - start_time

            logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Time: {elapsed:.1f}s - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val LWLRAP: {val_score:.9f}"  # Full precision
            )

            # Checkpoint & Early Stopping
            if val_score > self.best_score:
                self.best_score = val_score
                self.counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(f"New best model saved to {self.save_path}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    logger.info("Early stopping triggered.")
                    self.early_stop = True
                    break

        # Load best model state before returning
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
        return self.best_score


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    logger.info("Generating submission...")
    preds_dict = inference(model, test_loader, device)

    # Load sample submission to get correct column order
    sample_sub = pd.read_csv(os.path.join(Config.INPUT_ROOT, "sample_submission.csv"))
    columns = list(sample_sub.columns)
    label_cols = columns[1:]  # Skip fname

    # Prepare data for DataFrame
    fnames = sample_sub["fname"].values
    data = []

    for fname in fnames:
        if fname in preds_dict:
            data.append(preds_dict[fname])
        else:
            # Should not happen if loader is correct
            data.append(np.zeros(len(label_cols)))

    df_sub = pd.DataFrame(data, columns=label_cols)
    df_sub.insert(0, "fname", fnames)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
