import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import mcrmse
from library.loss import MaskedMSELoss


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()

        # Stabilization: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Slice to scored length (68)
            outputs_scored = outputs[:, : Config.SCORED_LEN, :]
            targets_scored = targets[:, : Config.SCORED_LEN, :]

            # Flatten to (N_samples * Seq_len, N_targets)
            outputs_flat = outputs_scored.reshape(-1, Config.NUM_TARGETS)
            targets_flat = targets_scored.reshape(-1, Config.NUM_TARGETS)

            all_preds.append(outputs_flat.cpu().numpy())
            all_targets.append(targets_flat.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    score = mcrmse(all_targets, all_preds)
    return score


def train_model(model, train_loader, val_loader):
    """
    Main training loop with Early Stopping based on MCRMSE.
    """
    device = Config.DEVICE
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    criterion = MaskedMSELoss()

    best_score = float("inf")

    print(f"Starting training on {device} for {Config.EPOCHS} epochs.")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"Saved Best Model (Score: {best_score:.6f})")

    print(f"Training finished. Best Score: {best_score:.6f}")


def generate_submission(model, test_loader):
    """
    Generates submission file using the trained model.
    """
    device = Config.DEVICE
    # Load best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: Best model not found, using current weights.")

    model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # (B, 107, 3)
            all_preds.append(outputs.cpu().numpy())

    # Concatenate all batches: (Total_Samples, 107, 3)
    all_preds = np.concatenate(all_preds, axis=0)

    # Retrieve IDs
    ids = test_loader.dataset.ids
    seq_len = Config.SEQ_LEN

    # Prepare columns for submission
    # We need to expand IDs and create seqpos indices
    # id_col: [id1, id1, ..., id2, id2, ...]
    id_col = np.repeat(ids, seq_len)

    # seqpos_col: [0, 1, ..., 106, 0, 1, ..., 106, ...]
    seqpos_col = np.tile(np.arange(seq_len), len(ids))

    id_seqpos = [f"{i}_{p}" for i, p in zip(id_col, seqpos_col)]

    # Flatten predictions to match rows
    # (N, 107, 3) -> (N*107, 3)
    flat_preds = all_preds.reshape(-1, 3)

    # Construct DataFrame
    # Target Cols: reactivity, deg_Mg_pH10, deg_Mg_50C
    # Submission Cols: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission = pd.DataFrame()
    submission["id_seqpos"] = id_seqpos
    submission["reactivity"] = flat_preds[:, 0]
    submission["deg_Mg_pH10"] = flat_preds[:, 1]
    submission["deg_pH10"] = 0.0
    submission["deg_Mg_50C"] = flat_preds[:, 2]
    submission["deg_50C"] = 0.0

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
