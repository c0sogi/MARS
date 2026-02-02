import os
import sys
import torch
import numpy as np
import pandas as pd

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_loaders
from library.model import CatheterModel
from library.engine import ModelEMA, train_one_epoch, validate


def run_demo():
    # --- 1. Setup & Configuration Overrides ---
    # Set seed for reproducibility
    seed_everything(42)

    print("Configuring for fast demonstration run...")
    # Override Config defaults to run a quick test
    Config.epochs = 1
    Config.batch_size = 4
    Config.valid_batch_size = 4
    Config.num_workers = 2
    Config.debug = True  # Triggers data subsetting in get_loaders

    # Ensure working directory exists (Config.setup() runs on import, but good practice if we changed paths)
    os.makedirs(Config.working_dir, exist_ok=True)

    # --- 2. Data Loading ---
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_loaders(debug=Config.debug)

    # Verification: Check Data Loaders
    assert len(train_loader) > 0, "Train loader should not be empty."

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}")

    # Assertions for data integrity
    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), f"Expected image shape {(Config.batch_size, 3, Config.image_size, Config.image_size)}, got {images.shape}"
    assert labels.shape == (
        Config.batch_size,
        Config.num_classes,
    ), f"Expected label shape {(Config.batch_size, Config.num_classes)}, got {labels.shape}"

    # --- 3. Model Initialization ---
    print("Initializing Model...")
    device = Config.device
    model = CatheterModel(pretrained=True)
    model.to(device)

    # Verification: Dummy Forward Pass
    with torch.no_grad():
        # Pass the batch fetched earlier
        logits = model(images.to(device))

    assert logits.shape == (
        Config.batch_size,
        Config.num_classes,
    ), f"Model output shape mismatch. Expected {(Config.batch_size, Config.num_classes)}, got {logits.shape}"
    print("Model initialized and verified successfully.")

    # --- 4. Training Setup ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Calculate steps for OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.epochs,
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # Initialize EMA
    ema_model = ModelEMA(model, decay=Config.ema_decay, device=device)

    # --- 5. Training Loop (1 Epoch) ---
    print("\nStarting Training...")
    train_loss = train_one_epoch(
        model, optimizer, scheduler, train_loader, device, epoch=1, ema_model=ema_model
    )

    # Verification: Check loss validity
    assert np.isfinite(train_loss), "Training loss is not finite (NaN or Inf)."
    print(f"Training completed. Loss: {train_loss:.4f}")

    # --- 6. Validation ---
    print("\nStarting Validation...")
    # Validate using the EMA model weights for better stability
    val_loss, val_auc = validate(ema_model.ema, val_loader, device)

    # Verification: Check validation metrics
    assert np.isfinite(val_loss), "Validation loss is not finite."
    assert 0.0 <= val_auc <= 1.0, f"AUC score {val_auc} is out of bounds [0, 1]."
    print(f"Validation completed. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # --- 7. Inference & Submission ---
    print("\nGenerating Submission...")
    ema_model.ema.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Forward pass
            logits = ema_model.ema(images)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)

    # Create submission DataFrame
    submission = pd.DataFrame(all_preds, columns=Config.target_cols)
    submission.insert(0, "StudyInstanceUID", all_ids)

    # Verification: Check submission format
    assert len(submission) == len(all_ids), "Submission row count mismatch."
    assert (
        submission.shape[1] == len(Config.target_cols) + 1
    ), "Submission column count mismatch."

    # Save submission
    submission_path = os.path.join(Config.working_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    run_demo()
