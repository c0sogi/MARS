import os
import numpy as np
import pandas as pd
import torch
import shutil
import sys

# Import provided library modules
from library import config, data_processing, dataset, models, training, evaluation


def run_demo():
    print("Initializing Demo Pipeline...")

    # =========================================================================
    # 1. CONFIGURATION OVERRIDES (Optimize for Speed)
    # =========================================================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Create a separate working directory for this demo to avoid conflicts
    DEMO_WORKING_DIR = "./working/demo_run_v1"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override Global Config Paths
    config.WORKING_DIR = DEMO_WORKING_DIR
    config.MODEL_SAVE_PATH = os.path.join(DEMO_WORKING_DIR, "best_model.pth")
    config.SCALER_SAVE_PATH = os.path.join(DEMO_WORKING_DIR, "scaler.joblib")
    config.SUBMISSION_PATH = os.path.join(DEMO_WORKING_DIR, "submission.csv")

    # Override Training Hyperparameters for speed
    config.TRAIN_PARAMS["epochs"] = 2
    config.TRAIN_PARAMS["batch_size"] = 128  # Smaller batch for demo
    config.TRAIN_PARAMS["debug_sample_size"] = 2000  # Process only 2000 rows
    config.TRAIN_PARAMS["num_workers"] = (
        0  # Avoid multiprocessing overhead for small data
    )

    # Set Seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # =========================================================================
    # 2. DATA PROCESSING DEMO
    # =========================================================================
    print("\n[2] Testing Data Processing (with debug sampling)...")

    # Force regeneration of data (load_cached_data=False) to apply debug sampling
    X_kin, X_vis, y, ids = data_processing.get_train_data(load_cached_data=False)

    # Verification
    print(
        f"    Generated Train Data Shapes: Kinematic {X_kin.shape}, Visual {X_vis.shape}, Target {y.shape}"
    )

    # Assertions
    assert (
        len(X_kin) == len(X_vis) == len(y) == len(ids)
    ), "Data array lengths mismatch."
    assert (
        len(X_kin) <= config.TRAIN_PARAMS["debug_sample_size"]
    ), "Debug sampling failed."
    assert (
        X_kin.shape[1] == config.NUM_KINEMATIC_FEATURES
    ), f"Kinematic feature dim mismatch. Expected {config.NUM_KINEMATIC_FEATURES}, got {X_kin.shape[1]}"
    assert (
        X_vis.shape[1] == config.NUM_VISUAL_FEATURES
    ), f"Visual feature dim mismatch. Expected {config.NUM_VISUAL_FEATURES}, got {X_vis.shape[1]}"
    assert not np.isnan(X_kin).any(), "NaNs found in Kinematic features."
    assert not np.isnan(X_vis).any(), "NaNs found in Visual features."

    print("    -> Data Processing Logic Verified.")

    # =========================================================================
    # 3. DATASET & DATALOADER DEMO
    # =========================================================================
    print("\n[3] Testing Dataset and DataLoader...")

    # Initialize loaders (this triggers validation data processing as well)
    train_loader, val_loader = dataset.get_train_val_loaders(batch_size=32)

    # Fetch one batch
    x_kin_batch, x_vis_batch, y_batch = next(iter(train_loader))

    # Verification
    print(
        f"    Batch Shapes: Kinematic {x_kin_batch.shape}, Visual {x_vis_batch.shape}, Target {y_batch.shape}"
    )

    # Assertions
    assert x_kin_batch.shape[0] == 32, "Batch size mismatch."
    assert (
        x_kin_batch.shape[1] == config.NUM_KINEMATIC_FEATURES
    ), "Batch feature dimension mismatch."
    assert torch.is_tensor(x_kin_batch) and torch.is_tensor(
        y_batch
    ), "DataLoader did not return Tensors."

    print("    -> DataLoader Logic Verified.")

    # =========================================================================
    # 4. MODEL DEMO
    # =========================================================================
    print("\n[4] Testing GRV-Net Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.GRVNet().to(device)

    # Forward pass check
    x_kin_dev = x_kin_batch.to(device)
    x_vis_dev = x_vis_batch.to(device)

    with torch.no_grad():
        logits = model(x_kin_dev, x_vis_dev)

    # Verification
    print(f"    Model Output Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (32, 1), "Model output shape mismatch."
    assert not torch.isnan(logits).any(), "Model produced NaNs."

    print("    -> Model Architecture Verified.")

    # =========================================================================
    # 5. TRAINING LOOP DEMO
    # =========================================================================
    print("\n[5] Testing Training Loop...")

    # Initialize Trainer
    trainer = training.Trainer(model, train_loader, val_loader, device)

    # Run Training (Fast due to small epochs and data)
    trainer.fit()

    # Verification
    assert os.path.exists(config.MODEL_SAVE_PATH), "Best model file was not saved."
    print("    -> Training Loop Completed and Model Saved.")

    # =========================================================================
    # 6. EVALUATION & INFERENCE DEMO
    # =========================================================================
    print("\n[6] Testing Evaluation and Submission Generation...")

    # We need to ensure test data generation works.
    # Note: Test data generation reads sample_submission.csv which is large.
    # For this demo, we will allow it to run normally as we can't easily subsample
    # the input file without modifying the library code, but the inference is fast.
    # However, to keep it strictly within time limits if the test set is huge,
    # we rely on the fact that processing ~400k rows (test set size) is reasonably fast (<5 min).

    # Run full evaluation pipeline
    best_mcc = evaluation.generate_predictions(model_path=config.MODEL_SAVE_PATH)

    print(f"    Best Validation MCC achieved: {best_mcc:.4f}")

    # Verify Submission
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"    Submission File Shape: {df_sub.shape}")

    # Assertions
    expected_cols = ["contact_id", "contact"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Submission contains non-binary predictions."

    # Check against sample submission length
    df_sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    assert len(df_sub) == len(
        df_sample
    ), f"Submission length mismatch. Expected {len(df_sample)}, got {len(df_sub)}"

    print("    -> Evaluation and Submission Logic Verified.")

    print("\n========================================================")
    print("DEMO COMPLETED SUCCESSFULLY")
    print("========================================================")


if __name__ == "__main__":
    run_demo()
