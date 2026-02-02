import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings


# Ensure reproducible results
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
warnings.filterwarnings("ignore")

# Import library components
# We assume the library files are in the python path.
# Since the prompt structure implies they are in ./library, we import from there.
from library.config import Config
from library.architecture import AsymmetricParallelNet, VectorCrossLayer
from library.data_processor import get_dataloaders, FeatureEngineer
from library.trainer import Trainer


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # --------------------------------------------------------------------------
    print("\n[Step 1] Overriding Configuration for Demo...")

    # Use mini datasets available in working directory for speed
    # These files were identified in the file listing
    mini_train_path = "./working/mini_train.parquet"
    mini_val_path = "./working/mini_val.parquet"
    mini_test_path = "./working/mini_test.parquet"

    # Verify mini files exist, otherwise fallback (though they should exist based on prompt)
    if os.path.exists(mini_train_path):
        Config.TRAIN_DATA_PATH = mini_train_path
        Config.VAL_DATA_PATH = mini_val_path
        Config.TEST_DATA_PATH = mini_test_path
        print(f"  Using mini datasets: {mini_train_path}")
    else:
        print("  Mini datasets not found. Using full metadata (this might be slower).")

    # Modify hyperparameters for quick execution
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.N_CLUSTERS = 5  # Reduce KMeans clusters for faster feature engineering
    Config.BATCH_SIZE = 128  # Smaller batch size for mini dataset
    Config.CACHE_DIR = "./working/demo_cache"  # Separate cache for demo
    Config.DEBUG = True

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Data Processing Demonstration
    # --------------------------------------------------------------------------
    print("\n[Step 2] Processing Data and Creating Loaders...")

    # Force reload to ensure we process the mini dataset and not load old cache
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        load_cached_data=False
    )

    # Verification
    print(f"  Input Dimension: {input_dim}")
    print(f"  Train Batches: {len(train_loader)}")

    # Assertions to verify data integrity
    assert input_dim > 0, "Input dimension should be positive."
    assert len(train_loader) > 0, "Training loader should not be empty."

    # Inspect one batch
    inputs, targets = next(iter(train_loader))
    print(f"  Sample Batch Shape: Inputs {inputs.shape}, Targets {targets.shape}")
    assert inputs.shape[1] == input_dim, "Batch feature dimension mismatch."

    # --------------------------------------------------------------------------
    # 3. Architecture Logic Verification
    # --------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    # Test VectorCrossLayer
    layer = VectorCrossLayer(input_dim)
    x_dummy = torch.randn(10, input_dim)
    x_out = layer(x_dummy, x_dummy)
    assert x_out.shape == x_dummy.shape, "VectorCrossLayer output shape mismatch."
    print("  VectorCrossLayer check passed.")

    # Test Full Model
    model = AsymmetricParallelNet(input_dim, Config.NUM_CLASSES)
    model.to(Config.DEVICE)

    # Forward pass with dummy data
    dummy_input = torch.randn(2, input_dim).to(Config.DEVICE)
    dummy_output = model(dummy_input)

    print(f"  Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch."
    print("  AsymmetricParallelNet instantiation and forward pass passed.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop...")

    trainer = Trainer(model)

    # Fit the model
    # Since we set EPOCHS=1, this will be quick
    best_acc = trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    print(f"  Training finished. Best Validation Accuracy: {best_acc:.4f}")
    assert 0 <= best_acc <= 1.0, "Accuracy should be between 0 and 1."

    # --------------------------------------------------------------------------
    # 5. Inference and Submission
    # --------------------------------------------------------------------------
    print("\n[Step 5] Generating Predictions and Submission...")

    predictions = trainer.predict(test_loader)

    print(f"  Predictions Shape: {predictions.shape}")
    assert len(predictions) == len(
        test_ids
    ), "Number of predictions must match number of test IDs."

    # Remap predictions back to 1-7 range (Model predicts 0-6)
    final_predictions = predictions + 1

    # Create submission DataFrame
    submission = pd.DataFrame({"Id": test_ids, "Cover_Type": final_predictions})

    # Save submission
    submission_path = "./working/demo_submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"  Submission saved to {submission_path}")
    print("  First 5 rows:")
    print(submission.head())

    # Verify file exists
    assert os.path.exists(submission_path), "Submission file was not created."

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
