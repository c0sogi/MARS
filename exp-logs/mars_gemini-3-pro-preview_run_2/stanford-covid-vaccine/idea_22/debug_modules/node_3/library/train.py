import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, MetricTracker
from library.loss import MaskedMCRMSELoss
from library.data import get_loader
from library.model import DecoupledDenseNet


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        inputs = batch["inputs"].to(device)
        partner_indices = batch["partner_indices"].to(device)
        targets = batch["targets"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, partner_indices)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the global MCRMSE metric.
    """
    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(inputs, partner_indices)

            # Update metric tracker
            # We pass the raw outputs and targets; tracker handles flattening and scoring indices
            tracker.update(targets, outputs)

    score = tracker.result()
    return score


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves the submission CSV.
    """
    print("Generating submission...")

    # Load Test Loader
    test_loader = get_loader(split="test", shuffle=False, load_cached_data=True)

    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            batch_ids = batch["id"]

            # Forward pass
            outputs = model(inputs, partner_indices)

            # Move to CPU numpy
            outputs_np = outputs.cpu().numpy()

            ids_list.extend(batch_ids)
            preds_list.append(outputs_np)

    # Concatenate all predictions: (N_samples, Seq_Len, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare data for DataFrame
    # We need to flatten to (N_samples * Seq_Len, 5)
    # And create corresponding id_seqpos keys

    submission_ids = []
    submission_data = []

    seq_len = Config.SEQ_LEN
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)

        for pos in range(seq_len):
            row_id = f"{sample_id}_{pos}"
            submission_ids.append(row_id)
            submission_data.append(sample_preds[pos])

    submission_data = np.array(submission_data)

    # Create DataFrame
    df_sub = pd.DataFrame(submission_data, columns=target_cols)
    df_sub.insert(0, "id_seqpos", submission_ids)

    # Save
    save_path = Config.SUBMISSION_PATH
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loaders
    print("Initializing DataLoaders...")
    train_loader = get_loader(split="train", shuffle=True, load_cached_data=True)
    val_loader = get_loader(split="val", shuffle=False, load_cached_data=True)

    # 3. Model, Loss, Optimizer
    print("Initializing Model...")
    model = DecoupledDenseNet().to(device)
    criterion = MaskedMCRMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.8f} | Val MCRMSE: {val_score:.8f}"
        )

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with MCRMSE: {best_score:.8f}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    print(f"Training complete. Best Validation MCRMSE: {best_score:.8f}")

    # 5. Inference
    # Load best model
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    generate_submission(model, device)


if __name__ == "__main__":
    run_training()
