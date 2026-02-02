import sys
import os
import torch
import pandas as pd
import numpy as np
import random

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.dataset import ManufacturingDataset
from library.model import RPFEModel
from library.engine import Engine


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    # 1. Setup and Configuration
    print("Initializing demonstration...")
    set_seed(Config.SEED)

    # Define a small sample size for quick demonstration
    DEMO_SAMPLE_SIZE = 500

    # -------------------------------------------------------------------------
    # 2. Component Demo: Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[1/5] Demonstrating Feature Engineering...")
    fe = FeatureEngineer()

    # Process a tiny subset of data without caching to verify logic
    train_df, val_df, test_df, vocab_sizes = fe.process_data(
        load_cached_data=False, debug_sample_size=100
    )

    # Validation
    print("Validating Feature Engineering outputs...")
    # Check if f_27 was decomposed into 10 columns (f_27_0 ... f_27_9)
    for i in range(10):
        col_name = f"f_27_{i}"
        if col_name not in train_df.columns:
            raise AssertionError(f"Feature Engineering failed: {col_name} missing.")

    # Check if unique_character_count was created
    if "unique_character_count" not in train_df.columns:
        raise AssertionError(
            "Feature Engineering failed: 'unique_character_count' missing."
        )

    # Check shapes
    if len(train_df) != 100:
        raise AssertionError(f"Expected 100 training samples, got {len(train_df)}")

    print("Feature Engineering validation passed.")

    # -------------------------------------------------------------------------
    # 3. Component Demo: Dataset
    # -------------------------------------------------------------------------
    print("\n[2/5] Demonstrating Dataset Class...")

    # Identify columns based on FE output
    cat_cols = [f"f_27_{i}" for i in range(10)]
    exclude_cols = {"id", "target", "source_path"} | set(cat_cols)
    cont_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Instantiate Dataset
    ds = ManufacturingDataset(train_df, cat_cols, cont_cols, is_test=False)

    # Fetch one sample
    sample = ds[0]

    # Validation
    print("Validating Dataset outputs...")
    if not isinstance(sample["continuous"], torch.Tensor):
        raise AssertionError("Dataset continuous data is not a Tensor.")
    if not isinstance(sample["categorical"], torch.Tensor):
        raise AssertionError("Dataset categorical data is not a Tensor.")
    if not isinstance(sample["target"], torch.Tensor):
        raise AssertionError("Dataset target data is not a Tensor.")

    # Check dimensions
    if sample["continuous"].shape[0] != len(cont_cols):
        raise AssertionError(
            f"Continuous feature dimension mismatch. Expected {len(cont_cols)}, got {sample['continuous'].shape[0]}"
        )
    if sample["categorical"].shape[0] != len(cat_cols):
        raise AssertionError(
            f"Categorical feature dimension mismatch. Expected {len(cat_cols)}, got {sample['categorical'].shape[0]}"
        )

    print("Dataset validation passed.")

    # -------------------------------------------------------------------------
    # 4. Component Demo: Model Architecture
    # -------------------------------------------------------------------------
    print("\n[3/5] Demonstrating RPFE Model...")

    model = RPFEModel(vocab_sizes, len(cont_cols))

    # Prepare a dummy batch (Batch Size = 2)
    dummy_cont = torch.randn(2, len(cont_cols))
    # Generate random integers within vocab range for categorical
    dummy_cat_list = []
    for col in cat_cols:
        v_size = vocab_sizes[col]
        dummy_cat_list.append(torch.randint(0, v_size, (2, 1)))
    dummy_cat = torch.cat(dummy_cat_list, dim=1)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_cont, dummy_cat)

    # Validation
    print("Validating Model output shapes...")
    # Expected output shape: (batch_size, num_streams) -> (2, 5)
    expected_shape = (2, Config.NUM_STREAMS)
    if output.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
        )

    print("Model validation passed.")

    # -------------------------------------------------------------------------
    # 5. Integration Demo: Full Engine Execution
    # -------------------------------------------------------------------------
    print("\n[4/5] Executing Full Engine Pipeline (Debug Mode)...")

    engine = Engine()

    # Run the engine with a small sample size and 1 epoch to ensure speed
    # This handles data processing, training, evaluation, and submission generation
    engine.run(debug_sample_size=DEMO_SAMPLE_SIZE, epochs=1)

    print("Engine execution complete.")

    # -------------------------------------------------------------------------
    # 6. Submission Verification
    # -------------------------------------------------------------------------
    print("\n[5/5] Verifying Submission File...")

    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    sub_df = pd.read_csv(submission_path)

    # Validation
    print("Checking submission format...")
    if list(sub_df.columns) != ["id", "target"]:
        raise AssertionError(
            f"Submission columns incorrect. Expected ['id', 'target'], got {list(sub_df.columns)}"
        )

    # In debug mode, the engine slices the submission file to match the debug_sample_size
    if len(sub_df) != DEMO_SAMPLE_SIZE:
        raise AssertionError(
            f"Submission length mismatch. Expected {DEMO_SAMPLE_SIZE}, got {len(sub_df)}"
        )

    # Check probability range
    if sub_df["target"].min() < 0 or sub_df["target"].max() > 1:
        raise AssertionError(
            "Predictions contain values outside [0, 1] probability range."
        )

    print(f"Submission verified. File located at: {submission_path}")
    print("\nAll demonstrations completed successfully.")
