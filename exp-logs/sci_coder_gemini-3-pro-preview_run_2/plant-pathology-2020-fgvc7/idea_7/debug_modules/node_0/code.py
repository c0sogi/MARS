import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import process_train_data, get_loaders
from library.train import run_fold
from library.inference import generate_submission, predict_model


def run_demo():
    print("=== Apple Disease Detection Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Enable Debug mode to use a tiny subset of data (50 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50

    # Reduce training duration to a single epoch
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8

    # Set Folds to 2 (Minimum for StratifiedKFold logic), but we will only actively train Fold 0
    Config.N_FOLDS = 2

    # Use a lightweight model available in timm for speed instead of the heavy EfficientNet/ConvNeXt
    Config.MODEL_A_NAME = "resnet18"
    Config.MODEL_B_NAME = (
        "resnet18"  # Pointing both to same architecture for simplicity
    )

    # Adjust Image Sizes for the smaller model
    Config.IMG_SIZE_EFFNET = 224
    Config.IMG_SIZE_CONVNEXT = 224

    # Redirect output to a demo directory to avoid cluttering main working dir
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure clean state
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Model Architecture: {Config.MODEL_A_NAME}")
    print(f"    Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Processing Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Processing...")

    # Force re-computation of training cache to verify logic
    train_df = process_train_data(load_cached_data=False)

    # Validations
    assert not train_df.empty, "Training DataFrame is empty."
    assert "target_rust" in train_df.columns, "Target 'rust' missing."
    assert "target_scab" in train_df.columns, "Target 'scab' missing."

    print(f"    Train DataFrame Shape (Full): {train_df.shape}")

    # Verify DataLoaders
    # This triggers the StratifiedKFold and Debug subsetting logic
    train_loader, val_loader, pos_weights = get_loaders(
        fold_idx=0, img_size=Config.IMG_SIZE_EFFNET, load_cached_data=True
    )

    # Fetch one batch to verify shapes
    imgs, targets = next(iter(train_loader))
    print(f"    Batch Image Shape: {imgs.shape}")
    print(f"    Batch Target Shape: {targets.shape}")

    assert imgs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch."
    assert imgs.shape[2] == Config.IMG_SIZE_EFFNET, "Image resolution mismatch."
    assert targets.shape[1] == 2, "Target should have 2 columns (Rust, Scab)."

    # -------------------------------------------------------------------------
    # 3. Training Demonstration (Fold 0)
    # -------------------------------------------------------------------------
    print("\n[3] Executing Training Loop (Fold 0)...")

    # Run training for 1 epoch. This function creates the model, optimizer, loss,
    # runs the training/validation loop, and saves the best model.
    best_auc = run_fold(
        fold_idx=0, model_name=Config.MODEL_A_NAME, img_size=Config.IMG_SIZE_EFFNET
    )

    # Verify Model Artifact Creation
    safe_model_name = Config.MODEL_A_NAME.replace(".", "_")
    model_path = os.path.join(
        Config.WORKING_DIR, f"best_model_{safe_model_name}_fold_0.pth"
    )

    assert os.path.exists(model_path), f"Model file was not saved at {model_path}"
    print(f"    Training finished. Best AUC: {best_auc:.4f}")
    print(f"    Model saved to: {model_path}")

    # -------------------------------------------------------------------------
    # 4. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Executing Inference (Fold 0)...")

    # Predict using the model we just trained
    preds = predict_model(
        Config.MODEL_A_NAME, Config.IMG_SIZE_EFFNET, fold_idx=0, device=device
    )

    assert preds is not None, "Prediction returned None."
    assert len(preds) > 0, "No predictions generated."

    # Check a sample prediction structure
    sample_id = next(iter(preds))
    sample_prob = preds[sample_id]

    print(f"    Sample Prediction ({sample_id}): {sample_prob}")
    assert len(sample_prob) == 2, "Prediction vector must have length 2 (Rust, Scab)."
    assert np.all(
        (sample_prob >= 0) & (sample_prob <= 1)
    ), "Probabilities must be between 0 and 1."

    # -------------------------------------------------------------------------
    # 5. Submission Generation (Ensemble)
    # -------------------------------------------------------------------------
    print("\n[5] Generating Submission...")

    # This function iterates through Config.MODEL_CONFIGS and Config.N_FOLDS.
    # It will find our trained model (Model A, Fold 0) and use it.
    # It will also attempt to load Model B Fold 0 (which we aliased to the same name),
    # effectively ensembling the same model twice (valid for demo).
    # It will fail to find Fold 1 models (since we didn't train them), printing a warning and continuing.
    generate_submission()

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found."

    # Validate Submission File Content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission Shape: {sub_df.shape}")
    print(f"    Submission Head:\n{sub_df.head()}")

    expected_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    for col in expected_cols:
        assert col in sub_df.columns, f"Missing column: {col}"

    # Check value ranges for probabilities
    numeric_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    for col in numeric_cols:
        assert sub_df[col].min() >= 0, f"Column {col} contains negative values."
        assert sub_df[col].max() <= 1.00001, f"Column {col} contains values > 1."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
