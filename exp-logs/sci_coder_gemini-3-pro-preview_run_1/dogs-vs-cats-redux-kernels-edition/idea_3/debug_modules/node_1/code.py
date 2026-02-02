import os
import sys
import torch
import pandas as pd
import numpy as np
import glob

# Import from the provided library files
from library.config import CFG
from library.utils import seed_everything
from library.data import prepare_data, get_loaders, get_test_loader
from library.modeling import get_model
from library.engine import train_fold, inference


def run_demonstration():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[Step 1] Configuring environment for fast demonstration...")
    seed_everything(CFG.seed)

    # Override CFG settings to ensure the script runs quickly (Debug Mode)
    CFG.debug = True  # Use small subset of data (100 train, 50 val)
    CFG.epochs = 1  # Train for only 1 epoch
    CFG.n_fold = 2  # We will only run 1 fold, but setup for 2
    CFG.model_names = ["tf_efficientnetv2_m.in21k_ft_in1k"]  # Use one model
    CFG.output_dir = "./working/demo_execution"
    CFG.batch_size = 8  # Smaller batch size for demo

    # Ensure output directory exists
    os.makedirs(CFG.output_dir, exist_ok=True)
    print(f"Debug mode: {CFG.debug}")
    print(f"Epochs: {CFG.epochs}")
    print(f"Output Directory: {CFG.output_dir}")

    # 2. Data Preparation and Verification
    print("\n[Step 2] Preparing and Verifying Data Pipeline...")
    # prepare_data loads metadata and creates folds
    df = prepare_data(load_cached_data=False)

    # Validation: Check if folds are assigned
    if "fold" not in df.columns:
        raise AssertionError("Data preparation failed: 'fold' column missing.")
    print(f"Data prepared. Total samples: {len(df)}")

    # Test Data Loaders
    print("Testing DataLoaders...")
    fold_idx = 0
    train_loader, val_loader = get_loaders(fold=fold_idx, df=df)

    # Fetch a batch to verify shapes
    try:
        images, labels = next(iter(train_loader))
        print(f"Batch Image Shape: {images.shape}")
        print(f"Batch Label Shape: {labels.shape}")

        # Assertions
        expected_shape = (CFG.batch_size, 3, CFG.image_size, CFG.image_size)
        if images.shape != expected_shape:
            raise AssertionError(
                f"Image shape mismatch. Expected {expected_shape}, got {images.shape}"
            )
        if labels.shape[0] != CFG.batch_size:
            raise AssertionError(
                f"Label batch size mismatch. Expected {CFG.batch_size}, got {labels.shape[0]}"
            )

    except StopIteration:
        raise AssertionError("Train loader is empty.")

    # 3. Model Instantiation and Forward Pass
    print("\n[Step 3] Verifying Model Architecture...")
    model_name = CFG.model_names[0]
    model = get_model(model_name, pretrained=True)
    model.to(CFG.device)

    # Run a dummy forward pass
    with torch.no_grad():
        dummy_input = images.to(CFG.device)
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions: Output should be (Batch_Size, 1) for binary classification logits
    if output.shape != (CFG.batch_size, 1):
        raise AssertionError(
            f"Model output shape incorrect. Expected {(CFG.batch_size, 1)}, got {output.shape}"
        )

    # 4. Training Loop Execution
    print("\n[Step 4] Executing Training Loop (Fold 0)...")
    # This function handles optimizer, scheduler, loop, and saving
    trained_model, best_score = train_fold(df, fold=fold_idx, model_name=model_name)

    print(f"Training completed. Best Validation Loss: {best_score:.4f}")

    # Verify model file was saved
    model_save_path = os.path.join(
        CFG.output_dir, model_name, f"model_fold_{fold_idx}.pth"
    )
    if not os.path.exists(model_save_path):
        raise AssertionError(f"Model checkpoint not found at {model_save_path}")
    print(f"Verified checkpoint exists at: {model_save_path}")

    # 5. Inference and Submission Generation
    print("\n[Step 5] Running Inference and Generating Submission...")

    # Get test loader (Note: get_test_loader loads the full test set defined in metadata)
    test_loader = get_test_loader()

    # Run inference
    # Note: trained_model is already loaded with best weights from train_fold
    predictions = inference(trained_model, test_loader)

    print(f"Inference completed. Predictions shape: {predictions.shape}")

    # Load test metadata to align IDs
    test_df = pd.read_csv(CFG.test_csv)

    # Assertions
    if len(predictions) != len(test_df):
        raise AssertionError(
            f"Prediction count mismatch. Expected {len(test_df)}, got {len(predictions)}"
        )

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {"id": test_df["id"], "label": predictions.flatten()}  # Ensure 1D array
    )

    # Verify values are probabilities (0-1) - inference function applies sigmoid
    if submission["label"].min() < 0 or submission["label"].max() > 1:
        raise AssertionError(
            "Predictions are not valid probabilities (must be between 0 and 1)."
        )

    # Save submission
    submission_path = os.path.join(CFG.output_dir, "submission_demo.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Verify file content
    saved_df = pd.read_csv(submission_path)
    print("First 5 rows of submission:")
    print(saved_df.head())

    if list(saved_df.columns) != ["id", "label"]:
        raise AssertionError(
            f"Submission columns incorrect. Expected ['id', 'label'], got {list(saved_df.columns)}"
        )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
