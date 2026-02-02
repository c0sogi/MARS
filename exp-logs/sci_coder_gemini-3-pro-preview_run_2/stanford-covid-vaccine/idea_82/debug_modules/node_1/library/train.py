import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.data import get_dataset, RNADataset
from library.modules import AHCHDN
from library.loss import AnchoredMCRMSELoss
from library.utils import seed_everything, calculate_global_rmse


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Handles the training logic for a single epoch using the two-pass strategy.
    """
    model.train()
    running_loss = 0.0

    for features, pair_map, targets in loader:
        features = features.to(device)
        pair_map = pair_map.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Pass 1: Initial prediction (Zero Feedback)
        y1 = model(features, pair_map, y_prev=None)

        # Pass 2: Refinement (Feedback from Pass 1)
        # We detach y1 to stop gradients from flowing back through the feedback generation of Pass 1
        # during the optimization of Pass 2's output.
        y2 = model(features, pair_map, y_prev=y1.detach())

        # Anchored Loss: Calculated over full sequence length (0-107)
        # Targets for tail (68-107) are 0.0, anchoring the model.
        loss = criterion(y2, targets) + 0.5 * criterion(y1, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the Correct Global RMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, pair_map, targets in loader:
            features = features.to(device)
            pair_map = pair_map.to(device)

            # Inference Strategy: 2 Passes
            y1 = model(features, pair_map, y_prev=None)
            y2 = model(features, pair_map, y_prev=y1)

            all_preds.append(y2.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metric only on valid scored positions (0-68) and columns
    score = calculate_global_rmse(
        all_preds,
        all_targets,
        scored_length=Config.SCORED_LENGTH,
        scored_cols_indices=Config.SCORED_COLS_INDICES,
    )

    return score


def run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False):
    """
    Main training loop with Early Stopping and Best Model Checkpointing.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Ensure directories exist
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Load Data
    train_data = get_dataset("train", load_cached_data=True)
    val_data = get_dataset("val", load_cached_data=True)

    train_ds = RNADataset(train_data, "train")
    val_ds = RNADataset(val_data, "val")

    if debug:
        print("Debug mode: utilizing a small subset of data.")
        subset_indices = range(min(len(train_ds), 64))
        train_ds = Subset(train_ds, subset_indices)
        val_ds = Subset(val_ds, range(min(len(val_ds), 64)))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model Setup
    model = AHCHDN().to(device)
    criterion = AnchoredMCRMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=Config.PATIENCE // 2
    )

    best_score = float("inf")
    best_path = os.path.join(Config.IDEA_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Best Val Score: {best_score}")


def run_inference(batch_size=Config.BATCH_SIZE):
    """
    Runs inference on the test set using the best saved model and generates submission.csv.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting Inference...")

    model_path = os.path.join(Config.IDEA_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load Model
    model = AHCHDN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Test Data
    test_data = get_dataset("test", load_cached_data=True)
    test_ds = RNADataset(test_data, "test")
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    preds_list = []
    with torch.no_grad():
        for features, pair_map in test_loader:
            features = features.to(device)
            pair_map = pair_map.to(device)

            # Inference Strategy: 2 Passes
            y1 = model(features, pair_map, y_prev=None)
            y2 = model(features, pair_map, y_prev=y1)

            preds_list.append(y2.cpu().numpy())

    # Concatenate predictions: (N_test, 107, 5)
    preds = np.concatenate(preds_list, axis=0)
    ids = test_data["ids"]

    # Flatten for submission
    submission_data = []
    for i, sample_id in enumerate(ids):
        for j in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{j}"
            row_vals = preds[i, j, :]
            row = [row_id] + row_vals.tolist()
            submission_data.append(row)

    columns = ["id_seqpos"] + Config.TARGET_COLS
    sub_df = pd.DataFrame(submission_data, columns=columns)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
