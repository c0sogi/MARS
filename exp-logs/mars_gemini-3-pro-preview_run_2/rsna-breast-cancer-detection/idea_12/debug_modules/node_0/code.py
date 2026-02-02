import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
from library import config
from library import utils
from library import data
from library import model
from library import trainer


def run_demo():
    print("Initializing Demo...")

    # 1. Setup and Configuration Overrides
    # We override config settings to ensure the script runs quickly for demonstration purposes.
    utils.seed_everything(42)

    # Create a specific directory for this demo execution
    demo_dir = os.path.join(config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)

    # Override config paths and parameters
    config.WORKING_DIR = demo_dir
    config.CACHE_DIR = demo_dir
    config.SUBMISSION_DIR = demo_dir
    config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Speed optimizations
    config.DEBUG = True  # Use a small subset of data (1000 rows)
    config.DEBUG_SAMPLE_SIZE = 50  # Even smaller for this specific demo script
    config.EPOCHS = 1  # Train for only 1 epoch
    config.BATCH_SIZE = 4  # Small batch size
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    config.IMAGE_SIZE = (256, 256)  # Reduce image size for faster processing in demo

    print(f"Configuration set. Working directory: {config.WORKING_DIR}")

    # 2. Data Loading Demonstration
    print("\n--- Step 1: Data Loading ---")
    # Force reload to ensure we use the debug subset and new cache location
    train_loader, val_loader, test_loader, feature_meta = data.get_dataloaders(
        load_cached_data=False, debug=config.DEBUG
    )

    # Assertions to verify data loading
    assert len(train_loader) > 0, "Train loader is empty"
    assert "vocab_sizes" in feature_meta, "Feature metadata missing vocab_sizes"

    # Fetch a single batch to verify shapes
    images, (cat_feats, num_feats), labels = next(iter(train_loader))

    print(f"Batch loaded successfully.")
    print(f"Image shape: {images.shape}")
    print(f"Categorical features shape: {cat_feats.shape}")
    print(f"Numerical features shape: {num_feats.shape}")
    print(f"Labels shape: {labels.shape}")

    # Verify dimensions
    # Images: (B, C, H, W) -> (4, 3, 256, 256)
    assert images.shape == (
        config.BATCH_SIZE,
        config.NUM_CHANNELS,
        config.IMAGE_SIZE[0],
        config.IMAGE_SIZE[1],
    )
    # Categorical: (B, Num_Cats) -> (4, 5)
    assert cat_feats.shape == (config.BATCH_SIZE, len(config.CATEGORICAL_COLS))
    # Numerical: (B, Num_Nums) -> (4, 1)
    assert num_feats.shape == (config.BATCH_SIZE, len(config.NUMERICAL_COLS))

    # 3. Model Initialization and Forward Pass
    print("\n--- Step 2: Model Initialization ---")
    net = model.MRHNModel(feature_meta["vocab_sizes"])
    net.to(config.DEVICE)
    net.eval()

    # Move batch to device
    images = images.to(config.DEVICE)
    cat_feats = cat_feats.to(config.DEVICE)
    num_feats = num_feats.to(config.DEVICE)

    # Run forward pass
    with torch.no_grad():
        # Using mixed precision as per training script
        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            logits = net(images, (cat_feats, num_feats))

    print(f"Forward pass output shape: {logits.shape}")
    assert logits.shape == (config.BATCH_SIZE, 1), "Model output shape mismatch"

    # 4. Metric Verification
    print("\n--- Step 3: Metric Verification (pF1) ---")
    # Test Case 1: Perfect prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 0.0, 1.0, 0.0])
    score_perfect = utils.pf1_score(y_true, y_pred)
    print(f"Perfect Score: {score_perfect}")
    assert np.isclose(
        score_perfect, 1.0
    ), "pF1 calculation failed for perfect predictions"

    # Test Case 2: Zero prediction
    y_pred_zero = np.array([0.0, 0.0, 0.0, 0.0])
    score_zero = utils.pf1_score(y_true, y_pred_zero)
    print(f"Zero Score: {score_zero}")
    assert score_zero == 0.0, "pF1 calculation failed for zero predictions"

    # 5. Training Loop Demonstration
    print("\n--- Step 4: Training Loop ---")
    # Run the trainer
    best_model_path = trainer.run_training(
        train_loader, val_loader, feature_meta, epochs=config.EPOCHS, patience=1
    )

    print(f"Training complete. Best model saved at: {best_model_path}")
    assert os.path.exists(best_model_path), "Best model file was not created"

    # 6. Inference and Submission
    print("\n--- Step 5: Inference and Submission ---")
    model.predict_and_submit(best_model_path, test_loader, feature_meta)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

    # Verify submission content
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(sub_df)}")
    print(sub_df.head())

    required_cols = {"prediction_id", "cancer"}
    assert required_cols.issubset(
        sub_df.columns
    ), f"Submission missing columns. Found: {sub_df.columns}"
    assert not sub_df.isnull().values.any(), "Submission contains NaN values"

    print("\nDemo execution completed successfully!")


if __name__ == "__main__":
    # Suppress specific warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    run_demo()
