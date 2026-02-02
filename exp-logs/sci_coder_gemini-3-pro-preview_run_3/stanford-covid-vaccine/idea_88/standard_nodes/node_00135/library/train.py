import os
import time
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, mcrmse_loss, mcrmse_metric, save_checkpoint
from library.model import HighCapacityBiGRU
from library.data import get_dataloaders


def train_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (features, pair_indices, pair_masks, targets) in enumerate(loader):
        # Move data to device
        features = features.to(device)
        pair_indices = pair_indices.to(device)
        pair_masks = pair_masks.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, pair_indices, pair_masks)

        # Calculate loss (MCRMSE on all 5 targets, sliced to seq_scored)
        loss = mcrmse_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns the MCRMSE score on the 3 scored columns.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, pair_indices, pair_masks, targets in loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)
            pair_masks = pair_masks.to(device)

            outputs = model(features, pair_indices, pair_masks)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metric
    score = mcrmse_metric(all_preds, all_targets)
    return score


def inference(model, loader, device):
    """
    Generates predictions for the test set and creates the submission file.
    """
    print("Starting inference on test set...")
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for features, pair_indices, pair_masks, sample_ids in loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)
            pair_masks = pair_masks.to(device)

            # outputs shape: (Batch, Seq_Len, 5)
            outputs = model(features, pair_indices, pair_masks)

            # Move to CPU numpy
            batch_preds = outputs.cpu().numpy()

            preds_list.append(batch_preds)
            ids_list.extend(sample_ids)

    # Concatenate all predictions: (Total_Samples, Seq_Len, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare data for submission DataFrame
    # We need to flatten: (Total_Samples * Seq_Len, Columns)
    # And generate id_seqpos keys

    flat_ids = []
    flat_preds = []

    seq_len = Config.SEQ_LEN  # 107

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(seq_len):
            flat_ids.append(f"{sample_id}_{seqpos}")
            flat_preds.append(sample_preds[seqpos])

    flat_preds = np.array(flat_preds)

    # Columns required for submission
    # Note: The model outputs 5 targets in this order:
    # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    submission_df = pd.DataFrame(flat_preds, columns=cols)
    submission_df.insert(0, "id_seqpos", flat_ids)

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


def run_training():
    """
    Main execution function for training, validation, and submission generation.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model Initialization
    model = HighCapacityBiGRU().to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "  # Full precision
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint & Early Stopping
        is_best = val_score < best_score
        if is_best:
            best_score = val_score
            patience_counter = 0
            print(f"New best model found! Score: {val_score}")
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_score": best_score,
                "optimizer": optimizer.state_dict(),
            },
            is_best=is_best,
            checkpoint_dir=Config.WORKING_DIR,
        )

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")

    # 6. Inference
    # Load best model
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Warning: Best model not found, using current model state.")

    inference(model, test_loader, device)
