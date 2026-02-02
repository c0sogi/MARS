import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, calculate_roc_auc, calculate_pos_weights
from library.data import process_folds, get_loaders, get_test_loader
from library.models import get_model
from library.training import train_fold


def main():
    print("Starting Library Usage Demonstration...")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    print("\n[1] Configuring environment for demo execution...")

    # Override Config constants to run a fast demo instead of full training
    Config.MAX_STEPS = 10  # Run only 10 training steps
    Config.VAL_CHECK_INTERVAL = 5  # Validate every 5 steps
    Config.DEBUG = True  # Use a subset of data
    Config.DEBUG_SUBSET_SIZE = 40  # Small subset size
    Config.BATCH_SIZE = 4  # Small batch size
    Config.N_FOLDS = 2  # Reduce folds for stratification logic check
    Config.WORKING_DIR = "./working/demo_execution"  # Separate working dir

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated for speed.")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n[2] Verifying Data Pipeline...")

    # Step 2.1: Process Folds
    # This reads metadata, performs iterative stratification, and caches the result
    df_folds = process_folds(load_cached_data=False)
    print(f"Folds processed. Shape: {df_folds.shape}")
    assert "fold" in df_folds.columns, "Fold column missing from processed dataframe"
    assert not df_folds.empty, "Processed dataframe is empty"

    # Step 2.2: Data Loaders
    # Get loaders for Fold 0
    train_loader, val_loader = get_loaders(
        fold=0, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )
    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")

    # Fetch one batch to verify shapes and types
    images, labels, rec_ids = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")  # Should be (B, 3, H, W)
    print(f"Batch Label Shape: {labels.shape}")  # Should be (B, 19)

    # Assertions
    assert images.dim() == 4, "Images must be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images must have 3 channels (Pseudo-RGB)"
    assert (
        labels.shape[1] == Config.NUM_CLASSES
    ), f"Labels must have {Config.NUM_CLASSES} classes"
    assert (
        images.shape[2:] == Config.IMAGE_SIZE
    ), f"Image size mismatch. Expected {Config.IMAGE_SIZE}, got {images.shape[2:]}"

    print("Data loading verified successfully.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    device = Config.DEVICE
    backbone = "resnet18"

    # Instantiate model
    model = get_model(
        backbone_name=backbone, num_classes=Config.NUM_CLASSES, pretrained=True
    )
    model = model.to(device)

    # Forward pass with the batch fetched earlier
    images = images.to(device)
    with torch.no_grad():
        logits = model(images)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Output logits shape mismatch"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("Model architecture verified successfully.")

    # ==========================================
    # 4. Metric Calculation Verification
    # ==========================================
    print("\n[4] Verifying Metrics...")

    # Create synthetic ground truth and predictions
    # Case: Perfect predictions
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    y_pred_good = np.array(
        [[0.9, 0.1, 0.9], [0.1, 0.9, 0.1], [0.8, 0.8, 0.2], [0.1, 0.1, 0.9]]
    )

    auc_score = calculate_roc_auc(y_true, y_pred_good)
    print(f"Calculated AUC (Synthetic Good): {auc_score:.4f}")

    assert auc_score > 0.9, "AUC calculation failed for good predictions"

    # Verify Positive Weights calculation
    # Using the dataframe from the dataset
    pos_weights = calculate_pos_weights(train_loader.dataset.df, device="cpu")
    print(f"Positive Weights Shape: {pos_weights.shape}")
    assert pos_weights.shape[0] == Config.NUM_CLASSES, "Positive weights shape mismatch"

    print("Metrics verified successfully.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n[5] Running Training Loop Demo...")

    # Run training for Fold 0 using the library function
    # This function handles the loop, optimization, validation, and model saving
    best_auc = train_fold(fold_idx=0, backbone_name=backbone)

    print(f"Training demo complete. Best AUC: {best_auc:.4f}")

    # Verify model file was created
    expected_model_path = os.path.join(
        Config.WORKING_DIR, f"model_{backbone}_fold_0.pth"
    )
    assert os.path.exists(
        expected_model_path
    ), f"Model file not found at {expected_model_path}"

    print("Training loop verified successfully.")

    # ==========================================
    # 6. Inference and Submission Demo
    # ==========================================
    print("\n[6] Running Inference Demo...")

    # Load the best model
    model = get_model(
        backbone_name=backbone, num_classes=Config.NUM_CLASSES, pretrained=False
    )
    state_dict = torch.load(expected_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Get Test Loader
    test_loader = get_test_loader(batch_size=Config.BATCH_SIZE)
    print(f"Test Loader batches: {len(test_loader)}")

    predictions = []
    rec_ids_list = []

    # Run inference on a few batches (or all, since it's fast)
    with torch.no_grad():
        for i, (images, _, rec_ids) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            predictions.append(probs.cpu().numpy())
            rec_ids_list.append(rec_ids.numpy())

            # Limit inference for demo speed if dataset was large (it's small here)
            if i >= 2:
                break

    predictions = np.concatenate(predictions, axis=0)
    rec_ids_list = np.concatenate(rec_ids_list, axis=0)

    print(f"Inference output shape: {predictions.shape}")

    # Generate Submission DataFrame
    # Flattening logic: Id = rec_id * 100 + species_id
    submission_rows = []
    for i, rec_id in enumerate(rec_ids_list):
        for species_idx in range(Config.NUM_CLASSES):
            row_id = rec_id * 100 + species_idx
            prob = predictions[i, species_idx]
            submission_rows.append({"Id": int(row_id), "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Save submission
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(submission_df.head())

    # Assertions
    assert not submission_df.empty, "Submission dataframe is empty"
    assert (
        "Id" in submission_df.columns and "Probability" in submission_df.columns
    ), "Submission columns mismatch"

    print("\nLibrary usage demonstration completed successfully!")


if __name__ == "__main__":
    main()
