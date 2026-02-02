import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import calculate_weighted_log_loss, save_checkpoint, load_checkpoint
from library.loss import TriLevelFractureLoss
from library.dataset import CervicalSpineDataset, get_slice_cache


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    loss_fn: nn.Module,
    epoch: int,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Device to run training on (CPU/GPU).
        loss_fn: Instance of TriLevelFractureLoss.
        epoch: Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Initialize GradScaler for AMP
    scaler = GradScaler()

    # Disable tqdm for cleaner output as requested
    bar = tqdm(loader, disable=True)

    optimizer.zero_grad()

    for step, (images, targets) in enumerate(bar):
        images = images.to(device)

        # Move targets to device
        targets = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in targets.items()
        }

        batch_size = images.size(0)

        # Mixed Precision Forward Pass
        with autocast():
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            # Normalize loss for gradient accumulation
            loss = loss / Config.ACCUMULATION_STEPS

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        if (step + 1) % Config.ACCUMULATION_STEPS == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * Config.ACCUMULATION_STEPS * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Evaluates the model on the validation set using the competition metric.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        device: Device to run evaluation on.

    Returns:
        float: Weighted Multi-label Logarithmic Loss.
    """
    model.eval()

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for images, targets in tqdm(loader, disable=True):
            images = images.to(device)

            # Get Ground Truth (Study Level)
            # targets['study_labels'] is (B, 8)
            y_true = targets["study_labels"].numpy()
            targets_all.append(y_true)

            # Forward pass (Mixed Precision optional for inference, but good for speed)
            with autocast():
                outputs = model(images)
                # Get Predictions (Study Level)
                # Apply sigmoid to logits
                study_logits = outputs["study_logits"]
                y_pred = torch.sigmoid(study_logits).cpu().float().numpy()
                preds_all.append(y_pred)

    if len(preds_all) == 0:
        return 0.0

    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    # Calculate Metric
    score = calculate_weighted_log_loss(targets_all, preds_all)
    return score


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epochs: int = Config.EPOCHS,
    patience: int = Config.PATIENCE,
    save_dir: str = Config.OUTPUT_DIR,
):
    """
    Orchestrates the training process with early stopping.
    """
    loss_fn = TriLevelFractureLoss(
        lambda_study=1.0,
        lambda_slice=Config.LAMBDA_SLICE,
        lambda_spatial=Config.LAMBDA_SPATIAL,
    )

    best_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn, epoch
        )

        # Validate
        val_score = evaluate(model, val_loader, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.10f} | Val Score: {val_score:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_score, best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Training complete. Best Val Score: {best_score:.10f}")


def generate_submission(model: nn.Module, device: torch.device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")

    # Load Test Metadata
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Ensure slice cache exists for test set
    slice_cache = get_slice_cache(test_meta, load_cached_data=True)

    # Create Dataset and Loader
    test_dataset = CervicalSpineDataset(
        metadata_df=test_meta,
        study_to_slices=slice_cache,
        is_train=False,
        seq_len=Config.SEQ_LEN,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    results = []
    target_cols = Config.TARGET_COLS  # ["C1", ..., "C7", "patient_overall"]

    # Track metadata index to map predictions back to UIDs
    # DataLoader preserves order because shuffle=False
    meta_idx = 0

    with torch.no_grad():
        for images, _ in tqdm(test_loader, disable=True):
            images = images.to(device)
            batch_size = images.size(0)

            with autocast():
                outputs = model(images)
                # Apply sigmoid to get probabilities
                probs = (
                    torch.sigmoid(outputs["study_logits"]).cpu().float().numpy()
                )  # (B, 8)

            for b in range(batch_size):
                study_uid = test_meta.iloc[meta_idx]["StudyInstanceUID"]
                study_probs = probs[b]

                # Generate rows for each target column
                for i, col_name in enumerate(target_cols):
                    row_id = f"{study_uid}_{col_name}"
                    prob = study_probs[i]
                    results.append({"row_id": row_id, "fractured": prob})

                meta_idx += 1

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
