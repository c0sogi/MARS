import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.data import get_dataloaders
from library.model import OffsetGuidedDualStreamModel
from library.utils import seed_everything, kl_divergence_score, get_logger


def train_one_epoch(epoch, model, loader, optimizer, scheduler, device, logger):
    """
    Handles the training of a single epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (spec, eeg, guidance, targets) in enumerate(loader):
        spec = spec.to(device)
        eeg = eeg.to(device)
        guidance = guidance.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(spec, eeg, guidance)

        # Loss calculation (KL Divergence)
        loss = kl_divergence_score(logits, targets, from_logits=True)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Scheduler step (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        batch_size = spec.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    return avg_loss


def validate(model, loader, device, logger):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for spec, eeg, guidance, targets in loader:
            spec = spec.to(device)
            eeg = eeg.to(device)
            guidance = guidance.to(device)
            targets = targets.to(device)

            logits = model(spec, eeg, guidance)
            loss = kl_divergence_score(logits, targets, from_logits=True)

            batch_size = spec.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    return avg_loss


def inference(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    predictions = []
    eeg_ids = []

    # Load test metadata to get eeg_ids corresponding to loader order
    # The loader preserves order of the dataframe it was created from
    test_df = pd.read_csv(Config.TEST_CSV)
    if Config.DEBUG:
        # Match the subsetting logic in get_dataloaders
        test_df = test_df.sample(
            n=min(len(test_df), 100), random_state=Config.SEED
        ).reset_index(drop=True)

    loader_eeg_ids = test_df["eeg_id"].values

    idx_counter = 0
    with torch.no_grad():
        for spec, eeg, guidance in loader:
            spec = spec.to(device)
            eeg = eeg.to(device)
            guidance = guidance.to(device)

            logits = model(spec, eeg, guidance)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            predictions.append(probs)

            # Track IDs
            batch_size = spec.size(0)
            current_ids = loader_eeg_ids[idx_counter : idx_counter + batch_size]
            eeg_ids.extend(current_ids)
            idx_counter += batch_size

    predictions = np.concatenate(predictions, axis=0)

    # Create submission DataFrame
    submission_df = pd.DataFrame(predictions, columns=Config.CLASS_NAMES)
    submission_df.insert(0, "eeg_id", eeg_ids)

    # Ensure eeg_id is integer
    submission_df["eeg_id"] = submission_df["eeg_id"].astype(int)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main function to orchestrate training, validation, and inference.
    """
    # 1. Setup
    Config.setup(debug=Config.DEBUG)
    seed_everything(Config.SEED)
    logger = get_logger("training_log")
    device = torch.device(Config.DEVICE)

    logger.info(f"Starting execution. Device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        Config, load_cached_data=True
    )
    logger.info("DataLoaders initialized.")

    # 3. Model
    model = OffsetGuidedDualStreamModel(Config).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, scheduler, device, logger
        )

        # Validate
        val_loss = validate(model, val_loader, device, logger)

        # Log (Full precision as requested)
        logger.info(
            f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with Val Loss: {best_val_loss}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # 6. Inference
    logger.info("Starting inference on test set using best model...")

    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logger.warning("Best model not found, using current model state.")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    inference(model, test_loader, device, submission_path)

    logger.info("Process complete.")


if __name__ == "__main__":
    run_training()
