import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.model import TriangulationDeberta
from library.loss import TriangulationLoss
from library.engine import run_training, predict_and_submit
from library.metrics import calculate_score


def main():
    # ==========================================
    # 1. Setup & Configuration for Fast Demo
    # ==========================================
    print(">>> Setting up configuration for fast demonstration...")

    # Set seed for reproducibility
    seed_everything(42)

    # Modify Config for speed (Runtime overrides)
    Config.DEBUG = True  # Limits data to 5000 rows
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 16

    # Clear cache to ensure DEBUG settings take effect (re-process data)
    if os.path.exists(Config.CACHE_DIR):
        print(
            f"Cleaning cache at {Config.CACHE_DIR} to ensure fresh data processing..."
        )
        shutil.rmtree(Config.CACHE_DIR)

    # Ensure working directories exist
    Config.setup_directories()

    device = get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Verify Metric Logic (Unit Test)
    # ==========================================
    print("\n>>> Verifying Metric Logic (calculate_score)...")

    # Create synthetic data covering different bias scenarios
    # Columns: target, prediction, identity_col (e.g., 'male')
    # We need at least one identity column from Config.IDENTITY_COLS
    ident_col = Config.IDENTITY_COLS[0]  # 'male'

    data = {
        "id": [1, 2, 3, 4],
        "target": [0.0, 1.0, 0.0, 1.0],  # 0.5 threshold -> [0, 1, 0, 1]
        "prediction": [0.1, 0.9, 0.8, 0.2],  # Good, Good, Bad (FP), Bad (FN)
        ident_col: [0.0, 0.0, 1.0, 1.0],  # Mentions of identity
        # Add other identity columns as 0.0 to avoid key errors if code expects them
    }
    for col in Config.IDENTITY_COLS:
        if col != ident_col:
            data[col] = [0.0] * 4

    df_metric_test = pd.DataFrame(data)

    # Calculate score
    score, metrics = calculate_score(df_metric_test, "prediction")

    print(f"Synthetic Score: {score:.4f}")
    print(f"Metrics Breakdown: {metrics}")

    # Assertions
    assert isinstance(score, float), "Score should be a float"
    assert "overall_auc" in metrics, "Metrics should contain overall_auc"
    assert (
        metrics["overall_auc"] == 0.5
    ), "Expected AUC of 0.5 for this synthetic case (2 correct, 2 wrong)"

    # ==========================================
    # 3. Verify Data Pipeline
    # ==========================================
    print("\n>>> Verifying Data Pipeline...")

    # Load dataloaders
    # This triggers load_and_preprocess which respects Config.DEBUG
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
        load_cached_data=False,  # Force reload
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Inspect one batch
    batch = next(iter(train_loader))

    # Check keys
    expected_keys = [
        "input_ids",
        "attention_mask",
        "target",
        "identity_targets",
        "attack_target",
        "sample_weight",
    ]
    for k in expected_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Check shapes
    input_ids = batch["input_ids"]
    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)

    print(f"Batch Size: {batch_size}, Seq Len: {seq_len}")

    assert batch["target"].size(0) == batch_size, "Target batch size mismatch"
    assert batch["identity_targets"].size(1) == len(
        Config.IDENTITY_COLS
    ), "Identity targets dimension mismatch"

    # Check Weights Logic (Bias Trap)
    # Weights should be either 1.0 or Config.BIAS_WEIGHT_MULTIPLIER (5.0)
    weights = batch["sample_weight"]
    unique_weights = torch.unique(weights).tolist()
    print(f"Unique sample weights in batch: {unique_weights}")
    for w in unique_weights:
        assert (
            w == 1.0 or w == Config.BIAS_WEIGHT_MULTIPLIER
        ), f"Unexpected weight value: {w}"

    # ==========================================
    # 4. Verify Model & Loss
    # ==========================================
    print("\n>>> Verifying Model and Loss...")

    model = TriangulationDeberta(Config.MODEL_NAME).to(device)
    criterion = TriangulationLoss()

    # Move batch to device
    batch_gpu = {k: v.to(device) for k, v in batch.items()}

    # Forward Pass
    outputs = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])

    # Check Output Keys
    assert "primary" in outputs
    assert "identity" in outputs
    assert "attack" in outputs

    # Check Output Shapes
    assert outputs["primary"].shape == (batch_size, 1), "Primary logits shape mismatch"
    assert outputs["identity"].shape == (
        batch_size,
        len(Config.IDENTITY_COLS),
    ), "Identity logits shape mismatch"
    assert outputs["attack"].shape == (batch_size, 1), "Attack logits shape mismatch"

    # Calculate Loss
    loss, loss_dict = criterion(outputs, batch_gpu)

    print(f"Calculated Loss: {loss.item():.4f}")
    print(f"Loss Components: {loss_dict}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert (
        loss_dict["loss_total"] > loss_dict["loss_primary"]
    ), "Total loss should be sum of components"

    # ==========================================
    # 5. Run Training Loop (Engine)
    # ==========================================
    print("\n>>> Running Training Loop (Shortened)...")

    # We use the run_training function from engine.py
    # Since we set Config.EPOCHS = 1 and Config.DEBUG = True, this will be fast.
    best_model_path = run_training(train_loader, val_loader)

    assert os.path.exists(
        best_model_path
    ), f"Model checkpoint not found at {best_model_path}"
    print(f"Training complete. Best model saved to {best_model_path}")

    # ==========================================
    # 6. Run Prediction & Submission
    # ==========================================
    print("\n>>> Running Prediction and Submission...")

    predict_and_submit(best_model_path, test_loader)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print(df_sub.head())

    # Check rows (Test set size is 97320, but we might have padded if batch size didn't align perfectly,
    # though predict_and_submit handles truncation to sample_submission length)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    assert len(df_sub) == len(
        sample_sub
    ), f"Submission length mismatch. Got {len(df_sub)}, expected {len(sample_sub)}"
    assert "prediction" in df_sub.columns, "Submission missing 'prediction' column"
    assert df_sub["prediction"].dtype == float, "Prediction column is not float"

    print("\n>>> Demonstration Complete Successfully.")


if __name__ == "__main__":
    main()
