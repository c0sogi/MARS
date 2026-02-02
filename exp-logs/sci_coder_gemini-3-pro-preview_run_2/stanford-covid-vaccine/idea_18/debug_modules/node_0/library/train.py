import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, MetricTracker
from library.loss import MaskedMCRMSELoss
from library.data import get_loader
from library.model import InteractionEnrichedDenseNet


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Training logic for one epoch.
    """
    model.train()
    running_loss = 0.0

    for i, (inputs, partner_indices, targets) in enumerate(loader):
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, partner_indices)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Validation logic using MetricTracker for global MCRMSE calculation.
    """
    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for inputs, partner_indices, targets in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            outputs = model(inputs, partner_indices)

            # Update tracker with ground truth and predictions
            tracker.update(targets, outputs)

    return tracker.result()


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")

    # Load test data
    test_loader = get_loader(
        mode="test", batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    model.eval()
    all_preds = []

    # Inference
    with torch.no_grad():
        for inputs, partner_indices, _ in test_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Output shape: (Batch, SeqLen, 5)
            outputs = model(inputs, partner_indices)
            all_preds.append(outputs.cpu().numpy())

    # Concatenate all batches
    # Shape: (N_samples, SeqLen, 5)
    preds_array = np.concatenate(all_preds, axis=0)

    # Get IDs from the dataset
    ids = test_loader.dataset.ids

    # Prepare submission data
    submission_data = []

    # Iterate through each sample and each position
    # We need to output rows for id_seqpos
    for idx, sample_id in enumerate(ids):
        sample_preds = preds_array[idx]  # (SeqLen, 5)

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            # Get the 5 predicted values
            # Order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # This matches Config.TARGET_COLS order
            vals = sample_preds[seqpos]

            row_dict = {
                "id_seqpos": row_id,
                "reactivity": vals[0],
                "deg_Mg_pH10": vals[1],
                "deg_pH10": vals[2],
                "deg_Mg_50C": vals[3],
                "deg_50C": vals[4],
            }
            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Save
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def train_model(debug_limit=None):
    """
    Main training pipeline.

    Args:
        debug_limit (int, optional): Limit dataset size for debugging.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Using device: {device}")

    # 2. Data Loaders
    train_loader = get_loader(
        mode="train",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        limit_size=debug_limit,
    )
    val_loader = get_loader(
        mode="val",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        limit_size=debug_limit,
    )

    # 3. Model
    model = InteractionEnrichedDenseNet()
    model.to(device)

    # 4. Optimization
    criterion = MaskedMCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # 5. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, device)
        val_mcrmse = val_metrics["mcrmse"]

        # Scheduler Step
        scheduler.step(val_mcrmse)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            print(
                f"Validation MCRMSE improved from {best_mcrmse} to {val_mcrmse}. Saving model..."
            )
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse}")

    # 6. Generate Submission
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(model, device)
