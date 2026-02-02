import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library import config, utils, data, model, train, inference


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("1. Setting up configuration for fast demonstration...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    utils.seed_everything(42)

    # Override config parameters for speed and debugging
    # We modify the module variables directly so they propagate to other modules
    config.DEBUG = True
    config.DEBUG_SIZE = 10  # Use only 10 samples for demonstration
    config.EPOCHS = 1  # Train for only 1 epoch
    config.BATCH_SIZE = 2  # Small batch size
    config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Ensure output directories exist
    os.makedirs(config.IDEA_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    print(f"   Debug Mode: {config.DEBUG}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Epochs: {config.EPOCHS}")

    # ==========================================
    # 2. Data Module Demonstration
    # ==========================================
    print("\n2. Demonstrating Data Module...")

    # Load a small subset of training metadata manually to test Dataset class
    train_df = pd.read_csv(config.TRAIN_CSV).head(config.DEBUG_SIZE)

    # Instantiate the EEGDataset
    dataset = data.EEGDataset(train_df, mode="train", augment=False)
    print(f"   Dataset initialized with {len(dataset)} samples.")

    # Fetch a single sample
    img, target = dataset[0]
    print(f"   Sample Image Shape: {img.shape}")
    print(f"   Sample Target Shape: {target.shape}")

    # Validation assertions
    # Image should be (3, 512, 512) as per config.IMG_SIZE and replication
    assert img.shape == (
        3,
        512,
        512,
    ), f"Expected image shape (3, 512, 512), got {img.shape}"
    # Target should be (6,) for the 6 voting classes
    assert target.shape == (6,), f"Expected target shape (6,), got {target.shape}"
    # Target probabilities should sum to approximately 1.0
    assert torch.isclose(
        target.sum(), torch.tensor(1.0), atol=1e-5
    ), f"Target probabilities sum to {target.sum()}, expected 1.0"

    # Test DataLoaders generation
    print("   Generating DataLoaders...")
    train_loader, val_loader, test_loader = data.get_dataloaders(
        train_batch_size=config.BATCH_SIZE, val_batch_size=config.BATCH_SIZE, debug=True
    )

    # Fetch a batch from the loader
    batch_imgs, batch_targets = next(iter(train_loader))
    print(f"   Batch Input Shape: {batch_imgs.shape}")

    assert batch_imgs.shape == (
        config.BATCH_SIZE,
        3,
        512,
        512,
    ), "Batch image shape mismatch"
    assert batch_targets.shape == (config.BATCH_SIZE, 6), "Batch target shape mismatch"

    # ==========================================
    # 3. Model Module Demonstration
    # ==========================================
    print("\n3. Demonstrating Model Module...")

    # Instantiate the model (pretrained=False for speed/offline safety in this demo)
    net = model.EEGEfficientNet(pretrained=False)
    net.eval()

    # Perform a dummy forward pass
    with torch.no_grad():
        logits = net(batch_imgs)

    print(f"   Model Output (Logits) Shape: {logits.shape}")

    # Validate output shape (Batch_Size, Num_Classes)
    assert logits.shape == (
        config.BATCH_SIZE,
        6,
    ), f"Expected output shape ({config.BATCH_SIZE}, 6), got {logits.shape}"

    # ==========================================
    # 4. Loss Function Demonstration
    # ==========================================
    print("\n4. Demonstrating Loss Function...")

    criterion = utils.KL_Loss()

    # Calculate loss between model logits and targets
    loss = criterion(logits, batch_targets)
    print(f"   Calculated KL Loss: {loss.item():.4f}")

    # Validate loss
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # ==========================================
    # 5. Training Pipeline Demonstration
    # ==========================================
    print("\n5. Demonstrating Training Pipeline...")
    print("   Running training loop (1 Epoch, Debug Subset)...")

    # Execute the training routine provided in the library
    # This will train, validate, and save the best model to config.MODEL_PATH
    train.run_training(debug=True)

    # Verify model artifact creation
    assert os.path.exists(
        config.MODEL_PATH
    ), f"Model file was not saved at {config.MODEL_PATH}"
    print(f"   Model successfully saved to {config.MODEL_PATH}")

    # ==========================================
    # 6. Inference Pipeline Demonstration
    # ==========================================
    print("\n6. Demonstrating Inference Pipeline...")

    # Run the standalone inference script
    # This loads the model saved in step 5 and predicts on the test set
    inference.predict(debug=True)

    # Verify submission file creation
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), f"Submission file was not created at {config.SUBMISSION_PATH}"

    # Load and inspect submission
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"   Submission File generated with {len(sub_df)} rows.")
    print("   Head of Submission:")
    print(sub_df.head())

    # Validate Submission Structure
    expected_cols = [
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Validate Row Count (should match DEBUG_SIZE)
    assert (
        len(sub_df) == config.DEBUG_SIZE
    ), f"Expected {config.DEBUG_SIZE} rows in submission, found {len(sub_df)}"

    # Validate Probability Sum Constraint
    # Sum of vote columns must equal 1.0
    vote_cols = expected_cols[1:]
    row_sums = sub_df[vote_cols].sum(axis=1)
    assert np.allclose(
        row_sums, 1.0, atol=1e-4
    ), "Submission probabilities do not sum to 1.0"

    print("\nSUCCESS: All library components demonstrated and verified.")


if __name__ == "__main__":
    main()
