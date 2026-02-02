import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library files
from library.utils import seed_everything, custom_weight_init
from library.dataset import get_dataloaders, get_test_ids
from library.network import SustainedHybridModel
from library.engine import Trainer, generate_submission
from library.config import DEVICE

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 42
BATCH_SIZE = 4096  # Large batch size for faster execution on A100
EPOCHS = 1  # Single epoch for demonstration speed
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
OUTPUT_DIR = "./working/demo_execution"
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")


def main():
    print(f"Running on device: {DEVICE}")

    # 1. Setup Reproducibility
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=4, load_cached_data=True
    )

    # Logic Verification: Check Data Shapes
    print("Verifying data shapes...")
    sample_cat, sample_cont, sample_y = next(iter(train_loader))

    # Expected: cat (B, 10), cont (B, 30), y (B)
    assert (
        sample_cat.dim() == 2 and sample_cat.shape[1] == 10
    ), f"Unexpected categorical shape: {sample_cat.shape}"
    assert (
        sample_cont.dim() == 2 and sample_cont.shape[1] == 30
    ), f"Unexpected continuous shape: {sample_cont.shape}"
    assert sample_y.dim() == 1, f"Unexpected target shape: {sample_y.shape}"

    print("Data shapes verified successfully.")

    # 3. Model Initialization
    print("Initializing Model...")
    model = SustainedHybridModel().to(DEVICE)

    # 4. Custom Weight Initialization
    custom_weight_init(model)

    # Logic Verification: Forward Pass Sanity Check
    print("Performing forward pass sanity check...")
    model.eval()
    with torch.no_grad():
        # Move sample to device
        s_cat = sample_cat.to(DEVICE)
        s_cont = sample_cont.to(DEVICE)

        logits = model(s_cat, s_cont)

        # Check output shape (B, 1)
        assert logits.shape == (
            sample_cat.shape[0],
            1,
        ), f"Model output shape mismatch. Expected {(sample_cat.shape[0], 1)}, got {logits.shape}"

        # Check for NaNs
        assert not torch.isnan(
            logits
        ).any(), "Model produced NaN values in forward pass."

    print("Forward pass verified successfully.")

    # 5. Training Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        patience=3,
        save_path=MODEL_PATH,
    )

    # 6. Run Training
    print("Starting training loop...")
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    # 7. Inference & Submission
    print("Retrieving Test IDs...")
    test_ids = get_test_ids(load_cached_data=True)

    print("Generating Submission...")
    # Load best model weights if saved (Trainer saves if validation improves)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    generate_submission(model, test_loader, test_ids, DEVICE, SUBMISSION_PATH)

    # 8. Final Output Verification
    print("Verifying submission file...")
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(SUBMISSION_PATH)

    # Check dimensions
    assert (
        len(df_sub) == 100000
    ), f"Submission has incorrect number of rows: {len(df_sub)}"
    assert list(df_sub.columns) == [
        "id",
        "target",
    ], f"Submission has incorrect columns: {df_sub.columns}"

    # Check value range
    preds = df_sub["target"].values
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions contain values outside [0, 1] range."

    print("Submission verified successfully.")
    print("Demonstration complete.")


if __name__ == "__main__":
    main()
