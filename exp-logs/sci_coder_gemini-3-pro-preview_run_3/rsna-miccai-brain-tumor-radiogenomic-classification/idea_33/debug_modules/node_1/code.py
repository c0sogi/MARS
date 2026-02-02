import sys
import os
import types
import torch
import pandas as pd
import numpy as np


# ==========================================
# 1. Patch and Import Library Modules
# ==========================================
def patch_and_import_model():
    """
    Reads library/model.py, comments out the top-level run() call
    to prevent auto-execution, and loads it as a module.
    """
    model_path = "./library/model.py"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find {model_path}")

    with open(model_path, "r") as f:
        source = f.read()

    # Disable the automatic execution at the end of the file
    # We look for the specific pattern "run()" at the end of the script
    if source.strip().endswith("run()"):
        # Use rsplit to replace ONLY the last occurrence (the call),
        # preserving the 'def run():' definition earlier in the file.
        parts = source.rsplit("run()", 1)
        if len(parts) == 2:
            source = parts[0] + "# run()" + parts[1]

    # Create a new module object
    module = types.ModuleType("library.model")
    module.__file__ = model_path

    # Register in sys.modules so other imports (like in library.train) find this patched version
    sys.modules["library.model"] = module

    # Execute the source code within the module's namespace
    try:
        exec(source, module.__dict__)
    except Exception as e:
        raise RuntimeError(f"Failed to patch and load library.model: {e}")


# Apply patch before importing other libraries that depend on library.model
patch_and_import_model()

# Now import the rest of the library
from library.utils import seed_everything, get_device
from library.data import process_dataset
from library.model import VAMSNet
from library.train import train_model
from library.predict import generate_submission

# ==========================================
# 2. Main Execution
# ==========================================
if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    print(">>> Starting Task Demonstration")

    # -------------------------------------------------
    # Step A: Data Processing Demonstration
    # -------------------------------------------------
    print("\n[Step A] Verifying Data Processing Logic...")

    # Load training metadata
    train_meta_path = "./metadata/train.parquet"
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata not found: {train_meta_path}")

    df_train = pd.read_parquet(train_meta_path)

    # Process a tiny subset (10 samples) from scratch
    # We verify that X (images), y (labels), and ids are returned correctly
    debug_n = 10
    X_demo, y_demo, ids_demo = process_dataset(
        df_train, split_name="demo_train", load_cached_data=False, debug_limit=debug_n
    )

    # Assertions
    # Expected Shape: (N, 64, 256, 256) -> 64 channels = 4 modalities * 16 slices
    assert len(X_demo) == debug_n, f"Expected {debug_n} samples, got {len(X_demo)}"
    assert X_demo.shape[1] == 64, f"Expected 64 channels, got {X_demo.shape[1]}"
    assert (
        X_demo.shape[2] == 256 and X_demo.shape[3] == 256
    ), "Image dimensions mismatch"
    assert len(y_demo) == debug_n, "Label count mismatch"
    assert len(ids_demo) == debug_n, "ID count mismatch"

    print(f"Data processing successful. Output shape: {X_demo.shape}")

    # -------------------------------------------------
    # Step B: Model Architecture Verification
    # -------------------------------------------------
    print("\n[Step B] Verifying Model Architecture...")

    device = get_device()
    model = VAMSNet(drop_path_rate=0.0).to(device)
    model.eval()

    # Create dummy input: Batch Size 2, 64 Channels, 256x256
    dummy_input = torch.randn(2, 64, 256, 256).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Assertions
    # Output should be logits with shape (Batch, 1)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    print("Model forward pass successful.")

    # -------------------------------------------------
    # Step C: Training Loop Demonstration
    # -------------------------------------------------
    print("\n[Step C] Verifying Training Loop...")

    # Train for 1 epoch on a tiny subset
    # This verifies the loss computation, backprop, and model saving
    try:
        best_model_path = train_model(
            epochs=1,
            batch_size=2,  # Small batch size for demo
            learning_rate=1e-4,
            debug_limit=10,  # Only use 10 samples
            load_cached_data=False,  # Force processing
        )
    except Exception as e:
        raise RuntimeError(f"Training loop failed: {e}")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Training completed but model file missing at {best_model_path}"
        )

    print(f"Training successful. Model saved to: {best_model_path}")

    # -------------------------------------------------
    # Step D: Prediction & Submission Demonstration
    # -------------------------------------------------
    print("\n[Step D] Verifying Inference and Submission...")

    # Run inference on a small subset of test data
    submission_file = "./submission/submission.csv"

    # Ensure clean state
    if os.path.exists(submission_file):
        os.remove(submission_file)

    generate_submission(
        model_path=best_model_path,
        batch_size=2,
        load_cached_data=False,
        debug_limit=5,  # Predict on 5 test samples
    )

    # Assertions
    if not os.path.exists(submission_file):
        raise FileNotFoundError("Submission file was not generated.")

    sub_df = pd.read_csv(submission_file)
    assert len(sub_df) == 5, f"Expected 5 predictions, got {len(sub_df)}"
    assert "BraTS21ID" in sub_df.columns, "Missing BraTS21ID column"
    assert "MGMT_value" in sub_df.columns, "Missing MGMT_value column"

    # Check probability range
    preds = sub_df["MGMT_value"].values
    assert np.all(preds >= 0.0) and np.all(
        preds <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("Submission generation successful.")
    print(sub_df.head())

    print("\n>>> Demonstration Complete. All checks passed.")
