import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.data_processing import get_data, NFLDataset
from library.model import WDPIRVModel, FocalLoss, train_model, predict
from library.training import run_training_pipeline
from library.inference import optimize_threshold, predict_and_submit

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting NFL Contact Detection Code Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    print("\n[1] Configuring Environment for Demo...")

    # Override paths to use a dedicated demo directory
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Update cache paths to point to the demo directory
    Config.CACHE_TRAIN_PARQUET = os.path.join(DEMO_DIR, "train_features.parquet")
    Config.CACHE_VAL_PARQUET = os.path.join(DEMO_DIR, "val_features.parquet")
    Config.CACHE_TEST_PARQUET = os.path.join(DEMO_DIR, "test_features.parquet")
    Config.CACHE_SCALER = os.path.join(DEMO_DIR, "scaler.joblib")
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.THRESHOLD_PATH = os.path.join(DEMO_DIR, "best_threshold.npy")

    # Speed optimizations
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 256
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print("    Configuration updated for speed and isolation.")

    # -------------------------------------------------------------------------
    # 2. Data Processing Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Processing (get_data)...")

    # Run get_data with debug=True to use a small subset and force processing from scratch
    # This tests: Metadata loading, Tracking/Helmet merging, Feature Engineering, Scaling
    train_ds, val_ds, test_ds, feature_dim = get_data(
        load_cached_data=False, debug=True
    )

    # Assertions
    assert isinstance(
        train_ds, NFLDataset
    ), "Train dataset is not an instance of NFLDataset"
    assert len(train_ds) > 0, "Train dataset is empty"
    assert feature_dim > 0, f"Feature dimension is invalid: {feature_dim}"
    assert os.path.exists(Config.CACHE_TRAIN_PARQUET), "Train parquet cache not created"
    assert os.path.exists(Config.CACHE_SCALER), "Scaler cache not created"

    print(f"    Success: Processed {len(train_ds)} training samples.")
    print(f"    Feature Dimension: {feature_dim}")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture (WDPIRVModel)...")

    # Instantiate model
    model = WDPIRVModel(input_dim=feature_dim)
    model.to(Config.DEVICE)
    model.train()

    # Create dummy batch
    batch_size = 32
    dummy_input = torch.randn(batch_size, feature_dim).to(Config.DEVICE)
    dummy_target = torch.randint(0, 2, (batch_size, 1)).float().to(Config.DEVICE)

    # Forward pass
    output = model(dummy_input)

    # Assertions
    assert output.shape == (
        batch_size,
        1,
    ), f"Output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"

    # Loss calculation check
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    loss = criterion(output, dummy_target)
    loss.backward()

    assert not torch.isnan(loss), "Loss is NaN"
    print("    Success: Model forward/backward pass verified.")

    # -------------------------------------------------------------------------
    # 4. Full Training Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Training Pipeline (run_training_pipeline)...")

    # Run the pipeline using the cached data we just generated
    # This tests: DataLoader creation, Training Loop, Validation, Early Stopping, Inference
    run_training_pipeline(debug=True, load_cached_data=True, epochs=1, batch_size=256)

    # Assertions
    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint not saved"
    assert os.path.exists(Config.THRESHOLD_PATH), "Threshold file not saved"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Validate submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "contact_id" in sub_df.columns and "contact" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) > 0, "Submission file is empty"

    print(
        f"    Success: Pipeline completed. Submission saved to {Config.SUBMISSION_PATH}"
    )

    # -------------------------------------------------------------------------
    # 5. Inference Tools Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Inference Tools...")

    # Test optimize_threshold
    # Create synthetic ground truth and probabilities
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.1, 0.6, 0.95])

    best_thresh = optimize_threshold(y_true, y_probs, step=0.05)

    # Assertions
    assert 0.0 < best_thresh < 1.0, f"Optimized threshold {best_thresh} out of bounds"

    # Test predict_and_submit using the trained model and test loader
    # We need to recreate the test loader since run_training_pipeline consumes it locally
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Load metadata for predict_and_submit
    # In debug mode, we need to match the subset logic.
    # Since we can't easily replicate the random sample indices from get_data without reloading,
    # we will mock the metadata dataframe to match the test_ds length.
    mock_meta = pd.DataFrame(
        {
            "contact_id": [f"id_{i}" for i in range(len(test_ds))],
            "contact": [0] * len(test_ds),
        }
    )

    custom_sub_path = os.path.join(DEMO_DIR, "custom_submission.csv")

    predict_and_submit(
        model=model,
        test_loader=test_loader,
        device=Config.DEVICE,
        threshold=best_thresh,
        save_path=custom_sub_path,
        metadata_df=mock_meta,
    )

    assert os.path.exists(custom_sub_path), "Custom submission file not created"
    print("    Success: Inference utilities verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
