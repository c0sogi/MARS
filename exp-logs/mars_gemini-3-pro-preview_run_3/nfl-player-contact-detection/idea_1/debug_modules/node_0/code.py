import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import (
    SEED,
    DEVICE,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    setup_reproducibility,
    WORKING_DIR,
)
from library.dataset import get_dataloaders
from library.model import ContactMLP
from library.trainer import Trainer, optimize_threshold, generate_submission
from library.inference import predict_probs


def main():
    print("=== Starting Demonstration of NFL Contact Detection Pipeline ===")

    # 1. Setup Reproducibility
    setup_reproducibility(SEED)
    print(f"Random seed set to {SEED}. Device: {DEVICE}")

    # 2. Data Loading
    print("\n[Step 1] Loading Data...")
    # We use a slightly smaller batch size for the demo to ensure it fits easily in memory
    # and load_cached_data=True to use existing processed files if available.
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=256, num_workers=2, load_cached_data=True
    )

    # Verification: Check DataLoaders
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."
    assert len(test_ids) > 0, "Test IDs are missing."

    # Inspect one batch to verify shapes
    X_batch, y_batch = next(iter(train_loader))
    input_dim = X_batch.shape[1]
    print(f"Data loaded successfully.")
    print(f" - Input Feature Dimension: {input_dim}")
    print(f" - Batch Shape: {X_batch.shape}")
    print(f" - Target Shape: {y_batch.shape}")

    # 3. Model Initialization
    print("\n[Step 2] Initializing Model...")
    model = ContactMLP(input_dim=input_dim).to(DEVICE)

    # Verification: Forward pass with dummy data
    dummy_input = torch.randn(10, input_dim).to(DEVICE)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        10,
        1,
    ), f"Model output shape mismatch. Expected (10, 1), got {dummy_output.shape}"
    assert (dummy_output >= 0).all() and (
        dummy_output <= 1
    ).all(), "Model output not in [0, 1] range (Sigmoid check)."
    print("Model initialized and forward pass verified.")

    # 4. Training Setup
    print("\n[Step 3] Setting up Trainer...")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
    )

    # 5. Training Execution
    # limiting to 1 epoch for demonstration speed
    print("\n[Step 4] Training Model (1 Epoch for Demo)...")

    # Ensure model save path directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    trainer.fit(
        num_epochs=1,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        model_save_path=MODEL_SAVE_PATH,
    )

    # Verification: Check if model file was created
    assert os.path.exists(
        MODEL_SAVE_PATH
    ), f"Model file not found at {MODEL_SAVE_PATH} after training."
    print("Training complete. Best model saved.")

    # 6. Threshold Optimization
    print("\n[Step 5] Optimizing Threshold...")
    # Load the best model state
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    best_threshold = optimize_threshold(model, val_loader, DEVICE)

    # Verification: Threshold sanity check
    assert (
        0.0 < best_threshold < 1.0
    ), f"Optimized threshold {best_threshold} is out of expected range (0, 1)."
    print(f"Threshold optimization complete. Best Threshold: {best_threshold}")

    # 7. Inference (Probability Prediction)
    print("\n[Step 6] Running Inference (Probabilities)...")
    probs = predict_probs(model, test_loader, device=DEVICE)

    # Verification: Check predictions shape matches test IDs
    assert len(probs) == len(
        test_ids
    ), f"Prediction count ({len(probs)}) does not match Test ID count ({len(test_ids)})."
    print(f"Inference complete. Generated {len(probs)} predictions.")

    # 8. Submission Generation
    print("\n[Step 7] Generating Submission File...")
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    generate_submission(
        model, test_loader, test_ids, best_threshold, SUBMISSION_PATH, DEVICE
    )

    # Verification: Check submission file
    assert os.path.exists(
        SUBMISSION_PATH
    ), f"Submission file not found at {SUBMISSION_PATH}."

    df_sub = pd.read_csv(SUBMISSION_PATH)
    assert df_sub.shape[0] == len(test_ids), "Submission file row count mismatch."
    assert (
        "contact_id" in df_sub.columns and "contact" in df_sub.columns
    ), "Submission file missing required columns."
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Submission contains non-binary values."

    print(f"Submission generated successfully at {SUBMISSION_PATH}")
    print(f"Top 5 rows:\n{df_sub.head().to_string()}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
