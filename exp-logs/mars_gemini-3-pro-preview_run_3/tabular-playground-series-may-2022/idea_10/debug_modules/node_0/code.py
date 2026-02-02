import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data_processing import get_data_loaders
from library.model import MultiGranularityNet
from library.train_eval import train_one_epoch, evaluate, predict


def main():
    print("Starting demonstration of Manufacturing Control pipeline...")

    # ==========================================
    # 1. Configuration Setup for Demo
    # ==========================================
    # We modify the Config class attributes directly to optimize for speed and demonstration.
    print("Configuring parameters for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 2000  # Small subset for speed
    Config.BATCH_SIZE = 32  # Smaller batch size for the small subset
    Config.EPOCHS = 2  # Only run 2 epochs to prove the loop works
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.create_dirs()

    # Set reproducibility
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Processing Demonstration
    # ==========================================
    print("\n[Step 1] Loading and Processing Data...")

    # Note: We force load_cached_data=False to demonstrate the processing logic
    # or rely on the debug flag logic in get_data_loaders which usually re-processes or subsamples.
    # Given the implementation of get_data_loaders, if debug is True, it processes from scratch.
    train_loader, val_loader, test_loader, vocab_sizes = get_data_loaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force processing to verify feature engineering
        debug=Config.DEBUG,
        debug_samples=Config.DEBUG_SAMPLES,
    )

    # Validation: Check DataLoaders
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Val loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."
    assert isinstance(vocab_sizes, list), "vocab_sizes should be a list."
    assert len(vocab_sizes) > 0, "vocab_sizes should not be empty."

    # Inspect a single batch
    print("Inspecting a training batch...")
    sample_cont, sample_cat, sample_y = next(iter(train_loader))

    # Assert shapes
    # Continuous features: 29 original - 1 (f_27) + 1 (unique_count) = 29 expected
    # Categorical features: 10 (unigrams) + 9 (bigrams) + 2 (f_29, f_30) = 21 expected
    num_cont_features = sample_cont.shape[1]
    num_cat_features = sample_cat.shape[1]

    print(f"Continuous features: {num_cont_features}")
    print(f"Categorical features: {num_cat_features}")

    assert (
        num_cont_features == 29
    ), f"Expected 29 continuous features, got {num_cont_features}"
    assert (
        num_cat_features == 21
    ), f"Expected 21 categorical features, got {num_cat_features}"
    assert sample_y.shape == (Config.BATCH_SIZE, 1), "Target shape mismatch."

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n[Step 2] Initializing Model...")
    model = MultiGranularityNet(
        vocab_sizes=vocab_sizes, num_cont_features=num_cont_features
    )
    model.to(device)

    print("Verifying forward pass...")
    sample_cont = sample_cont.to(device)
    sample_cat = sample_cat.to(device)

    with torch.no_grad():
        logits = model(sample_cont, sample_cat)

    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch. Expected ({Config.BATCH_SIZE}, 1), got {logits.shape}"
    print("Forward pass successful.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n[Step 3] Running Training Loop...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Run for Config.EPOCHS (set to 2)
    best_auc = 0.0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_auc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        # Basic Assertions
        assert not np.isnan(train_loss), "Training loss is NaN."
        assert 0 <= val_auc <= 1, "AUC score out of range [0, 1]."

        # Save Checkpoint
        if val_auc >= best_auc:
            best_auc = val_auc
            save_checkpoint(
                {"state_dict": model.state_dict(), "best_auc": best_auc},
                filename=Config.MODEL_PATH,
            )
            print("Checkpoint saved.")

    # ==========================================
    # 5. Checkpoint Loading & Inference
    # ==========================================
    print("\n[Step 4] Testing Checkpoint Loading & Inference...")

    # Re-initialize model to ensure we are loading weights correctly
    model_inference = MultiGranularityNet(
        vocab_sizes=vocab_sizes, num_cont_features=num_cont_features
    ).to(device)

    checkpoint = load_checkpoint(
        model_inference, filename=Config.MODEL_PATH, device=Config.DEVICE
    )
    assert checkpoint is not None, "Failed to load checkpoint."
    assert "state_dict" in checkpoint, "Checkpoint missing state_dict."

    # Generate Predictions
    print("Generating predictions on test set...")
    predictions = predict(model_inference, test_loader, device)

    # Validate Predictions
    expected_test_samples = len(test_loader.dataset)
    assert (
        len(predictions) == expected_test_samples
    ), f"Prediction count mismatch. Expected {expected_test_samples}, got {len(predictions)}"

    # Check probability range
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Predictions out of probability range [0, 1]."

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n[Step 5] Creating Submission File...")

    # We need the IDs. In a real run, we read the full test CSV.
    # Since we used DEBUG mode, the loader subsampled the data.
    # We must ensure we grab the corresponding IDs.
    test_df_full = pd.read_csv(Config.TEST_CSV)
    test_df_debug = test_df_full.iloc[: Config.DEBUG_SAMPLES].copy()

    submission = pd.DataFrame(
        {
            Config.ID_COL: test_df_debug[Config.ID_COL],
            Config.TARGET_COL: predictions.flatten(),
        }
    )

    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify file existence and content
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert df_sub.shape == (
        Config.DEBUG_SAMPLES,
        2,
    ), f"Submission shape mismatch. Expected ({Config.DEBUG_SAMPLES}, 2), got {df_sub.shape}"

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
