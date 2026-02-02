import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, MCRMSELoss, compute_val_metric
from library.model import RNAModel
from library.data import get_dataloaders


def train_fn(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move data to device
        inputs = batch["inputs"].to(device)
        bpp_indices = batch["bpp_indices"].to(device)
        bpp_masks = batch["bpp_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (Batch, 107, 5)
        outputs = model(inputs, bpp_indices, bpp_masks)

        # Compute loss
        # Criterion expects (Batch, 107, 5) and (Batch, 68, 5)
        # It slices internally.
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns the official MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_masks = batch["bpp_masks"].to(device)
            targets = batch[
                "targets"
            ]  # Keep on CPU for metric calculation later if preferred,
            # but compute_val_metric handles device movement.

            # Forward pass
            outputs = model(inputs, bpp_indices, bpp_masks)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute metric
    score = compute_val_metric(preds, targets)
    return score


def predict_fn(model, dataloader, device):
    """
    Generates predictions for the test set.
    Returns a dictionary mapping 'id' to prediction arrays (107, 5).
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_masks = batch["bpp_masks"].to(device)
            ids = batch["ids"]

            outputs = model(inputs, bpp_indices, bpp_masks)
            outputs = outputs.cpu().numpy()

            for i, sample_id in enumerate(ids):
                predictions[sample_id] = outputs[i]

    return predictions


def generate_submission(predictions, output_path):
    """
    Formats predictions into the competition CSV format.
    """
    data = []
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Sort keys to ensure deterministic order if needed, though not strictly required
    sample_ids = sorted(predictions.keys())

    for sample_id in sample_ids:
        pred = predictions[sample_id]  # Shape (107, 5)
        seq_len = pred.shape[0]

        for i in range(seq_len):
            row_id = f"{sample_id}_{i}"
            row_data = [row_id] + pred[i].tolist()
            data.append(row_data)

    columns = ["id_seqpos"] + target_cols
    df = pd.DataFrame(data, columns=columns)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main execution function.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    Config.setup_directories()

    print(f"Device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model
    print("Initializing model...")
    model = RNAModel(Config).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    criterion = MCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_score = eval_fn(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved! Score: {best_score:.6f}")
        else:
            patience_counter += 1
            # print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating predictions on test set...")
    test_preds = predict_fn(model, test_loader, device)

    print("Creating submission file...")
    generate_submission(test_preds, Config.SUBMISSION_PATH)

    print("Done.")
