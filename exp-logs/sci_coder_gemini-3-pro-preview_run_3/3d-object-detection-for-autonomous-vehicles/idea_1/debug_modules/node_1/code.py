import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.dataset import NuScenesDataset
from library.model import PointPillars
from library.train import train_model, set_seed
from library.inference import generate_submission


def main():
    print("Initializing 3D Object Detection Demo...")

    # 1. Setup & Configuration Overrides for Speed
    # =========================================================
    set_seed(42)

    # Override Config to run a tiny, fast experiment
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Create a subset of test metadata for fast submission generation
    # The full test set takes too long for a demo run.
    full_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    subset_test_meta_path = os.path.join(Config.WORKING_DIR, "test_metadata_subset.csv")
    full_test_meta.head(10).to_csv(subset_test_meta_path, index=False)

    # Point Config to this subset
    Config.TEST_METADATA_PATH = subset_test_meta_path

    print(f"Configuration updated. Test subset saved to {subset_test_meta_path}")

    # 2. Verify Dataset and Voxelization
    # =========================================================
    print("\n--- Verifying Dataset Component ---")
    # Load a small slice of training data
    dataset = NuScenesDataset(split="train", load_cached_data=True)

    # Check length
    assert len(dataset) > 0, "Dataset should not be empty."

    # Get one sample
    sample = dataset[0]

    # Verify keys
    required_keys = [
        "pillars",
        "coors",
        "n_points",
        "sample_token",
        "cls_map",
        "reg_map",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key {key} in dataset sample."

    # Verify shapes
    # Pillars: (P, 32, 9)
    pillars = sample["pillars"]
    assert (
        pillars.ndim == 3 and pillars.shape[1] == 32 and pillars.shape[2] == 9
    ), f"Incorrect pillars shape: {pillars.shape}"

    print("Dataset verification passed.")

    # 3. Verify Model Forward Pass
    # =========================================================
    print("\n--- Verifying Model Component ---")
    device = torch.device(Config.DEVICE)
    model = PointPillars().to(device)

    # Create a batch using collate_fn
    batch_list = [dataset[i] for i in range(min(2, len(dataset)))]
    batch = NuScenesDataset.collate_fn(batch_list)

    # Move to device
    input_dict = {
        "pillars": batch["pillars"].to(device),
        "coors": batch["coors"].to(device),
        "n_points": batch["n_points"].to(device),
        "sample_tokens": batch["sample_tokens"],
        "cls_targets": batch["cls_targets"].to(device),
        "reg_targets": batch["reg_targets"].to(device),
    }

    # Forward pass
    model.train()
    output = model(input_dict)

    # Check outputs
    assert "loss" in output, "Model output missing 'loss' key during training."
    assert "cls_preds" in output, "Model output missing 'cls_preds'."
    assert "reg_preds" in output, "Model output missing 'reg_preds'."

    # Check loss is scalar
    assert output["loss"].dim() == 0, "Loss should be a scalar."

    print(f"Model forward pass successful. Loss: {output['loss'].item():.4f}")

    # 4. Run Integration Test (Train + Val + Submission)
    # =========================================================
    print("\n--- Running Integration Pipeline (Train/Val/Infer) ---")

    # We use a very small max_samples to ensure this finishes in seconds
    # train_model handles the loop, validation, saving, and submission generation
    train_model(epochs=1, batch_size=2, max_samples=10)

    # 5. Verify Outputs
    # =========================================================
    print("\n--- Verifying Outputs ---")

    # Check Model Checkpoint
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
        )
    print(f"Verified model checkpoint exists at {Config.MODEL_SAVE_PATH}")

    # Check Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Validate Submission Format
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "Id" in submission_df.columns, "Submission missing 'Id' column"
    assert (
        "PredictionString" in submission_df.columns
    ), "Submission missing 'PredictionString' column"

    # Check that we have rows corresponding to our subset (10 samples)
    assert (
        len(submission_df) == 10
    ), f"Expected 10 rows in submission, found {len(submission_df)}"

    print(f"Verified submission file format. First row:\n{submission_df.iloc[0]}")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
