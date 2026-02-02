import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    get_binary_targets,
    reconstruct_4_class_probabilities,
    get_class_weights,
    get_device,
)
from library.data import (
    AppleDataset,
    get_transforms,
    get_folds_data,
    get_train_val_loaders,
    get_test_loader,
)
from library.models import AppleNet
from library.training import run_fold
from library.inference import (
    get_oof_predictions,
    get_test_predictions_raw,
    run_inference,
)
from library.stacking import run_stacking, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=" * 50)
    print("STARTING DEMO: Apple Disease Detection Pipeline")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # -------------------------------------------------------------------------
    print("\n[1] Configuring Environment...")

    # Set up a separate working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.create_dirs()

    # Enable Debug mode to use a small subset of data (20 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20

    # Reduce training duration
    Config.EPOCHS = 1
    Config.N_FOLDS = 2  # We will only run Fold 0 for the demo

    # Use a lightweight model for demonstration speed
    Config.MODELS = [
        {
            "name": "resnet18",
            "img_size": 224,
            "batch_size": 4,
            "dropout_rates": [0.0, 0.1],
        }
    ]

    # Set seed
    seed_everything(Config.SEED)
    print("Configuration updated for demo (Debug Mode: ON, Model: ResNet18).")

    # -------------------------------------------------------------------------
    # 2. Validate Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Validating Utility Functions...")

    # Test: Reconstruct 4-class probabilities
    # Case: Rust=1.0, Scab=0.0 -> Should be Rust Only
    r_prob = np.array([1.0, 0.0])
    s_prob = np.array([0.0, 1.0])

    # Expected Output columns: [healthy, multiple, rust, scab]
    # 1. Rust=1, Scab=0 -> Healthy=0, Multi=0, Rust=1, Scab=0
    # 2. Rust=0, Scab=1 -> Healthy=0, Multi=0, Rust=0, Scab=1
    recon = reconstruct_4_class_probabilities(r_prob, s_prob)

    assert np.isclose(recon[0, 2], 1.0), "Reconstruction logic failed for Rust"
    assert np.isclose(recon[1, 3], 1.0), "Reconstruction logic failed for Scab"
    print("Utility functions validated successfully.")

    # -------------------------------------------------------------------------
    # 3. Validate Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Validating Data Loading...")

    # Force fresh fold generation
    folds_cache = os.path.join(Config.WORKING_DIR, "folds_data.parquet")
    if os.path.exists(folds_cache):
        os.remove(folds_cache)

    df_folds = get_folds_data(load_cached_data=False)
    print(f"Folds data generated. Shape: {df_folds.shape}")

    # Check binary targets generation
    targets = get_binary_targets(df_folds.head(5))
    assert targets.shape == (5, 2), "Binary targets shape mismatch"

    # Instantiate Dataset
    dataset = AppleDataset(
        df_folds.head(5), transform=get_transforms(224, mode="train"), mode="train"
    )

    img, target = dataset[0]
    assert img.shape == (3, 224, 224), f"Image tensor shape incorrect: {img.shape}"
    assert target.shape == (2,), f"Target tensor shape incorrect: {target.shape}"
    print("Dataset and Transforms validated successfully.")

    # -------------------------------------------------------------------------
    # 4. Validate Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Validating Model Architecture...")

    device = get_device()
    model = AppleNet(model_name="resnet18", pretrained=False, dropout_rates=[0.1]).to(
        device
    )

    # Create dummy batch
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 2), f"Model output shape incorrect: {output.shape}"
    print("Model forward pass validated successfully.")

    # Clean up
    del model, dummy_input
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 5. Run Training (Single Fold)
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Demo (Fold 0)...")

    # Run training for Fold 0 using the first model config
    # This covers: get_train_val_loaders, AppleNet, train_one_epoch, validate, save_checkpoint
    auc_score = run_fold(fold_idx=0, model_config=Config.MODELS[0])

    print(f"Training completed. Validation AUC: {auc_score:.4f}")

    # Verify checkpoint creation
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_resnet18_fold_0.pth")
    assert os.path.exists(ckpt_path), "Checkpoint file was not created."
    print("Checkpoint verified.")

    # -------------------------------------------------------------------------
    # 6. Run Inference & Stacking
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference and Stacking Demo...")

    # A. Generate OOF Predictions
    # Note: Since we only trained Fold 0, OOF predictions for Fold 1 will be 0.
    # The function handles missing checkpoints gracefully.
    oof_preds = get_oof_predictions(load_cached_data=False)
    assert oof_preds.shape[1] == len(
        Config.MODELS
    ), "OOF preds architecture dim mismatch"
    assert oof_preds.shape[2] == 2, "OOF preds target dim mismatch"
    print("OOF predictions generated.")

    # B. Generate Test Predictions
    test_preds = get_test_predictions_raw(load_cached_data=False)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    assert test_preds.shape[0] == len(test_df), "Test preds sample count mismatch"
    print("Test predictions generated.")

    # C. Run Stacking
    # We need y_train for the meta-learner
    y_train = get_binary_targets(df_folds)

    # To make stacking work effectively in this demo where we only trained Fold 0,
    # we need to ensure the meta-learner doesn't fail due to zero-vectors in OOF.
    # However, LogisticRegression handles zeros fine (it just learns from what's there).
    # We proceed with the provided function.

    final_probs = run_stacking(oof_preds, test_preds, y_train, load_cached_data=False)

    assert final_probs.shape == (len(test_df), 4), "Final probabilities shape mismatch"
    print("Stacking completed.")

    # -------------------------------------------------------------------------
    # 7. Generate Submission
    # -------------------------------------------------------------------------
    print("\n[7] Generating Submission...")

    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    generate_submission(final_probs, output_path=submission_path)

    assert os.path.exists(submission_path), "Submission file not found."

    # Verify content format
    sub_df = pd.read_csv(submission_path)
    expected_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"
    assert len(sub_df) == len(test_df), "Submission row count mismatch"

    print("Submission file generated and verified.")
    print("=" * 50)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
