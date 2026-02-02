import os
import torch
import numpy as np
import scipy.ndimage
import library.config as config
from library.utils import set_seed, compute_levenshtein
from library.data_loader import get_loaders
from library.model import DCSGCN
from library.loss import DCSGCNLoss


def decode_predictions(probs, lengths):
    """
    Decodes frame-wise probabilities into a list of gesture sequences.
    Applies Median Filtering and collapses repeated labels.

    Args:
        probs (np.ndarray): Shape (B, T, C)
        lengths (torch.Tensor): Shape (B,)

    Returns:
        list of list of int: Decoded gesture sequences.
    """
    predictions = []
    # Get class indices
    labels_pred = np.argmax(probs, axis=2)  # (B, T)

    for i in range(len(lengths)):
        length = lengths[i]
        seq = labels_pred[i, :length]

        # Apply Median Filter to smooth noise
        # Kernel size 7 is heuristic based on 10fps data (~0.7s window)
        seq = scipy.ndimage.median_filter(seq, size=7, mode="nearest")

        # Collapse repeats and remove background (class 0)
        decoded_seq = []
        prev_label = -1
        for label in seq:
            if label != prev_label:
                if label != 0:
                    decoded_seq.append(int(label))
                prev_label = label
        predictions.append(decoded_seq)

    return predictions


def decode_targets(labels, lengths):
    """
    Decodes frame-wise target tensors into a list of gesture sequences.

    Args:
        labels (torch.Tensor): Shape (B, T)
        lengths (torch.Tensor): Shape (B,)

    Returns:
        list of list of int: Target gesture sequences.
    """
    targets = []
    labels_np = labels.cpu().numpy()

    for i in range(len(lengths)):
        length = lengths[i]
        seq = labels_np[i, :length]

        decoded_seq = []
        prev_label = -1
        for label in seq:
            if label != prev_label:
                if label != 0:
                    decoded_seq.append(int(label))
                prev_label = label
        targets.append(decoded_seq)

    return targets


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        boundaries = batch["boundaries"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward pass returns dict of stages
        outputs = model(features, mask)

        # Compute loss
        loss = criterion(outputs, labels, boundaries, mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            boundaries = batch["boundaries"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]

            outputs = model(features, mask)

            # Compute loss
            loss = criterion(outputs, labels, boundaries, mask)
            total_loss += loss.item()

            # Decode predictions using Stage 3 output
            # outputs['stage3'] is tuple (cls_probs, bnd_probs)
            cls_probs, _ = outputs["stage3"]
            cls_probs_np = cls_probs.cpu().numpy()

            batch_preds = decode_predictions(cls_probs_np, lengths)
            batch_targets = decode_targets(labels, lengths)

            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)

    # Compute Levenshtein Error Rate
    # Metric = Sum(Dist) / Sum(TargetLen)
    # compute_levenshtein handles the summation internally
    lev_score = compute_levenshtein(all_preds, all_targets)

    return total_loss / len(loader), lev_score


def generate_submission(model, loader, device, output_path):
    print("Generating submission...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            sample_ids = batch["sample_ids"]
            lengths = batch["lengths"]

            outputs = model(features, mask)
            cls_probs, _ = outputs["stage3"]
            cls_probs_np = cls_probs.cpu().numpy()

            batch_preds = decode_predictions(cls_probs_np, lengths)

            for i, seq in enumerate(batch_preds):
                sid = sample_ids[i]
                seq_str = ",".join(map(str, seq))
                predictions.append(f"{sid},{seq_str}")

    with open(output_path, "w") as f:
        for line in predictions:
            f.write(line + "\n")
    print(f"Submission saved to {output_path}")


def run_training(epochs=None, batch_size=None):
    if epochs is None:
        epochs = config.HYPERPARAMS["num_epochs"]
    if batch_size is None:
        batch_size = config.HYPERPARAMS["batch_size"]

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, test_loader = get_loaders(batch_size=batch_size)

    # Initialize Model & Loss
    model = DCSGCN().to(device)
    criterion = DCSGCNLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.HYPERPARAMS["lr"],
        weight_decay=config.HYPERPARAMS["weight_decay"],
    )

    best_val_loss = float("inf")
    patience = config.HYPERPARAMS["patience"]
    patience_counter = 0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_lev = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Levenshtein: {val_lev}"
        )

        # Early Stopping based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model for submission
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for submission.")

    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    generate_submission(model, test_loader, device, submission_path)
