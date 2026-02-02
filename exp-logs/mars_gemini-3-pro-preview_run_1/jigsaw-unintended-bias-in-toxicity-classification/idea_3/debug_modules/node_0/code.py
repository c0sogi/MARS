import os
import shutil
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_data_loaders, calculate_sample_weights
from library.model import ToxicityClassifier
from library.trainer import Trainer
from library.metrics import compute_final_metric


def run_demonstration():
    print("==================================================")
    print("   TOXICITY CLASSIFICATION LIBRARY DEMONSTRATION   ")
    print("==================================================")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup for Fast Demo
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast execution...")

    # Enable Debug mode to use a small subset of data (5000 samples by default, we reduce to 1000)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 1000

    # Reduce training parameters for speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 16
    Config.VALID_BATCH_SIZE = 32

    # Clean up cache to ensure we demonstrate processing logic
    if os.path.exists(Config.CACHE_DIR):
        print(f"    Cleaning cache directory: {Config.CACHE_DIR}")
        shutil.rmtree(Config.CACHE_DIR)

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration complete. Debug mode: ON.")

    # ------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # ------------------------------------------------------------------------
    print("\n[2] Testing Data Loading...")

    # Load data loaders (this triggers tokenization and caching)
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=False)

    # Verify Train Loader
    print(f"    Train Loader batches: {len(train_loader)}")
    batch = next(iter(train_loader))

    # Check keys
    required_keys = ["input_ids", "attention_mask", "target", "weight"]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Check shapes
    input_shape = batch["input_ids"].shape
    expected_shape = (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)
    assert (
        input_shape == expected_shape
    ), f"Shape mismatch. Got {input_shape}, expected {expected_shape}"

    print("    Data structure verified successfully.")

    # ------------------------------------------------------------------------
    # 3. Bias Mitigation Logic (Sample Weights)
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Bias Mitigation (Sample Weighting)...")

    # Create a synthetic dataframe to test the 4 conditions
    # 1. Toxic + Identity Mentioned (BNSP Trap) -> High Weight
    # 2. Non-Toxic + Identity Mentioned (BPSN Trap) -> High Weight
    # 3. Toxic + No Identity -> Standard Weight
    # 4. Non-Toxic + No Identity -> Standard Weight

    dummy_data = {
        Config.TARGET_COL: [0.8, 0.1, 0.8, 0.1],  # 0.8 is Toxic, 0.1 is Non-Toxic
        "male": [0.6, 0.6, 0.0, 0.0],  # 0.6 is Mentioned
        # Fill other identities with 0
    }
    for col in Config.IDENTITY_COLUMNS:
        if col != "male":
            dummy_data[col] = [0.0] * 4

    dummy_df = pd.DataFrame(dummy_data)

    # Calculate weights
    weights = calculate_sample_weights(dummy_df)

    print(f"    Calculated Weights: {weights}")

    # Expected: [5.0, 5.0, 1.0, 1.0]
    expected_weights = np.array(
        [Config.BIAS_LOSS_WEIGHT, Config.BIAS_LOSS_WEIGHT, 1.0, 1.0], dtype=np.float32
    )
    np.testing.assert_array_almost_equal(weights, expected_weights)
    print("    Sample weighting logic verified.")

    # ------------------------------------------------------------------------
    # 4. Metric Calculation
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Metric Calculation...")

    # Create a perfect prediction scenario
    metric_df = dummy_df.copy()
    # Predictions match binary targets perfectly
    # Target 0.8 -> Pred 0.9 (Toxic)
    # Target 0.1 -> Pred 0.1 (Non-Toxic)
    metric_df["prediction"] = [0.9, 0.1, 0.9, 0.1]

    # Compute score
    # With perfect separation, all AUCs (Overall, Subgroup, BPSN, BNSP) should be 1.0
    score = compute_final_metric(
        metric_df, "prediction", Config.TARGET_COL, verbose=False
    )

    print(f"    Perfect Prediction Score: {score}")
    assert score == 1.0, f"Expected score 1.0 for perfect predictions, got {score}"
    print("    Metric calculation verified.")

    # ------------------------------------------------------------------------
    # 5. Model Training (Debug Run)
    # ------------------------------------------------------------------------
    print("\n[5] Running Training Loop (Debug Mode)...")

    # Initialize Model
    model = ToxicityClassifier(Config.MODEL_NAME)

    # Initialize Trainer
    trainer = Trainer(model, device=Config.DEVICE)

    # Prepare Validation DataFrame
    # Note: Since get_data_loaders performs random sampling in DEBUG mode,
    # we must replicate that sampling to pass the correct dataframe to trainer.fit()
    # for evaluation.
    val_df_full = pd.read_csv(Config.VAL_PATH)
    val_df_sampled = val_df_full.sample(
        n=min(len(val_df_full), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
    ).reset_index(drop=True)

    # Run Training
    # This will run for 1 epoch on 1000 samples (very fast)
    trainer.fit(train_loader, val_loader, val_df_sampled)

    # Verify Model Checkpoint
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint not found after training."
    print(f"    Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # ------------------------------------------------------------------------
    # 6. Inference & Submission
    # ------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    trainer.generate_submission(test_loader)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission shape: {sub_df.shape}")

    assert "id" in sub_df.columns and "prediction" in sub_df.columns
    # In debug mode, test set is also sampled to DEBUG_SAMPLE_SIZE
    assert len(sub_df) == Config.DEBUG_SAMPLE_SIZE

    print("    Submission generated successfully.")
    print("\n==================================================")
    print("       DEMONSTRATION COMPLETED SUCCESSFULLY       ")
    print("==================================================")


if __name__ == "__main__":
    run_demonstration()
