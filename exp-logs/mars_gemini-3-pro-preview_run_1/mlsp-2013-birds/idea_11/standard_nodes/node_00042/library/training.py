import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, update_bn

from library.utils import save_model, set_seed, save_submission
from library.data import Mixup


def train_one_epoch(loader, model, optimizer, criterion, device, mixup_fn):
    """
    Trains the model for one epoch using Mixup augmentation.

    Args:
        loader (DataLoader): Training dataloader.
        model (nn.Module): The model to train.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (str): Device to train on.
        mixup_fn (Mixup): Mixup augmentation object.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)
        dataset_size += batch_size

        # Apply Mixup
        images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    return running_loss / dataset_size if dataset_size > 0 else 0.0


def validate(loader, model, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        loader (DataLoader): Validation dataloader.
        model (nn.Module): The model to evaluate.
        criterion (Loss): The loss function.
        device (str): Device to evaluate on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)
            dataset_size += batch_size

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_targets) > 0:
        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)

        # Calculate Macro AUC
        try:
            auc_score = roc_auc_score(all_targets, all_preds, average="macro")
        except ValueError:
            auc_score = float("nan")

        # Cite {debug_lesson_2}: Explicitly Handle NaN Returns in Metric Calculations
        if np.isnan(auc_score):
            valid_scores = []
            for i in range(all_targets.shape[1]):
                # Only calculate if valid targets exist (both 0 and 1)
                if len(np.unique(all_targets[:, i])) > 1:
                    valid_scores.append(
                        roc_auc_score(all_targets[:, i], all_preds[:, i])
                    )

            if valid_scores:
                auc_score = np.mean(valid_scores)
            else:
                auc_score = 0.5
    else:
        auc_score = 0.0

    return avg_loss, auc_score


def run_swa_training(
    cfg, model, train_loader, val_loader, epochs, swa_start_epoch, save_path
):
    """
    Runs the training loop with Stochastic Weight Averaging (SWA).

    Args:
        cfg (Config): Configuration object.
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        epochs (int): Total number of epochs.
        swa_start_epoch (int): Epoch to start SWA collection.
        save_path (str): Path to save the final SWA model.

    Returns:
        nn.Module: The trained SWA model.
    """
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()
    mixup_fn = Mixup(alpha=cfg.MIXUP_ALPHA)

    # Initialize SWA Model
    swa_model = AveragedModel(model)

    best_val_auc = 0.0

    print(
        f"Starting training: {epochs} epochs total. SWA starts at epoch {swa_start_epoch}."
    )

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            train_loader, model, optimizer, criterion, device, mixup_fn
        )
        val_loss, val_auc = validate(val_loader, model, criterion, device)

        # SWA Update
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_status = "Active"
        else:
            swa_status = "Inactive"

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} [SWA: {swa_status}] "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f} "
            f"| Val AUC: {val_auc:.10f} "
            f"| Time: {elapsed:.2f}s"
        )

        # Save best base model (useful for analysis or fallback)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            base_save_path = save_path.replace("_swa.pth", "_base_best.pth")
            if base_save_path != save_path:
                save_model(model, base_save_path)

    print("Training finished. Updating SWA BatchNorm statistics...")
    # Update BN statistics using the training loader (without mixup logic applied by loader)
    update_bn(train_loader, swa_model, device=device)

    # Final Validation of SWA Model
    swa_val_loss, swa_val_auc = validate(val_loader, swa_model, criterion, device)
    print(
        f"Final SWA Model | Val Loss: {swa_val_loss:.6f} | Val AUC: {swa_val_auc:.10f}"
    )

    save_model(swa_model, save_path)
    print(f"SWA Model saved to {save_path}")

    return swa_model


def predict(loader, model, device):
    """
    Generates predictions for a dataset.

    Args:
        loader (DataLoader): Dataloader for inference.
        model (nn.Module): Trained model.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Array of predicted probabilities (N_samples, N_classes).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())

    if len(all_preds) > 0:
        return np.vstack(all_preds)
    return np.array([])


def generate_submission(cfg, model, loader, output_path):
    """
    Generates predictions for the test set and saves them in the submission format.

    Args:
        cfg (Config): Configuration object.
        model (nn.Module): Trained model.
        loader (DataLoader): Test dataloader.
        output_path (str): Path to save the submission CSV.
    """
    print("Generating submission predictions...")
    device = cfg.DEVICE

    # Get raw probability matrix (N_samples, N_classes)
    probs = predict(loader, model, device)

    # Load test metadata to get rec_ids
    # We assume the loader preserves the order of the CSV (shuffle=False)
    df_test = pd.read_csv(cfg.TEST_CSV)

    if cfg.MAX_SAMPLES:
        df_test = df_test.head(cfg.MAX_SAMPLES)

    if len(df_test) != len(probs):
        raise ValueError(
            f"Mismatch between test set size ({len(df_test)}) and predictions ({len(probs)})"
        )

    submission_ids = []
    submission_probs = []

    # Iterate through samples and classes to flatten the output
    # Format: Id = rec_id * 100 + species_id
    num_classes = probs.shape[1]

    for idx, row in df_test.iterrows():
        rec_id = int(row["rec_id"])
        sample_probs = probs[idx]

        for species_id in range(num_classes):
            submission_id = rec_id * 100 + species_id
            prob = sample_probs[species_id]

            submission_ids.append(submission_id)
            submission_probs.append(prob)

    save_submission(submission_ids, submission_probs, output_path)
    print(f"Submission saved to {output_path}")
