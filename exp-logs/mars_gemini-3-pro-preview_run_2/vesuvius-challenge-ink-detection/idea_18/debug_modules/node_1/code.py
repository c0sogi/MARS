import os
import torch
import pandas as pd
import numpy as np
import sys

# Import from the provided library files
from library.config import Config
from library.dataset import InkDataset, TestInkDataset
from library.model import SegFormerB2
from library.train import train_model, set_seed
from library.inference import run_inference


def main():
    print("=== Starting Vesuvius Ink Detection Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("[1] Configuring parameters for rapid execution...")

    # Reduce dataset size to minimal amount for demo
    Config.MAX_TRAIN_SAMPLES = 16
    Config.MAX_VAL_SAMPLES = 8

    # Reduce training parameters
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.PATIENCE = 1

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("    Configuration updated: 1 Epoch, Batch Size 4, Limited Samples.")

    # -------------------------------------------------------------------------
    # 2. Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Dataset classes...")

    # Test Training Dataset
    try:
        train_ds = InkDataset(mode="train", limit=4, load_cached_data=True)
        sample_img, sample_label = train_ds[0]

        # Verify shapes
        # Image: (3, 512, 512) -> 3 channels (Config.IN_CHANNELS), Tile Size
        assert sample_img.shape == (
            3,
            512,
            512,
        ), f"Expected image shape (3, 512, 512), got {sample_img.shape}"

        # Label: (1, 512, 512) -> Binary mask
        assert sample_label.shape == (
            1,
            512,
            512,
        ), f"Expected label shape (1, 512, 512), got {sample_label.shape}"

        print("    InkDataset (Train) verification passed.")
    except Exception as e:
        print(f"    InkDataset verification failed: {e}")
        raise e

    # Test Inference Dataset (Fragment 'a' from test set)
    try:
        # We use fragment 'a' as it is in the test metadata
        test_ds = TestInkDataset(fragment_id="a", view="B", load_cached_data=True)
        t_img, t_coords, t_size, t_id = test_ds[0]

        assert t_img.shape == (
            3,
            512,
            512,
        ), f"Expected test image shape (3, 512, 512), got {t_img.shape}"
        assert t_coords.shape == (2,), "Expected coords shape (2,)"
        assert t_id == "a", "Fragment ID mismatch"

        print("    TestInkDataset verification passed.")
    except Exception as e:
        print(f"    TestInkDataset verification failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 3. Model Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying SegFormerB2 Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegFormerB2().to(device)

    # Create dummy input: Batch=2, Channels=3, H=512, W=512
    dummy_input = torch.randn(2, 3, 512, 512).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Verify output shape: (B, 1, 512, 512)
    expected_shape = (2, 1, 512, 512)
    assert (
        output.shape == expected_shape
    ), f"Expected model output shape {expected_shape}, got {output.shape}"

    print(f"    Model forward pass successful. Output shape: {output.shape}")

    # -------------------------------------------------------------------------
    # 4. Training Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[4] Executing Training Pipeline (train_model)...")

    # train_model() uses the global Config we modified earlier.
    # It handles data loading, training loop, validation, and saving 'best_model.pth'.
    # It also triggers an inference run at the end, but we will run inference explicitly
    # in step 5 to demonstrate the standalone inference module.
    train_model()

    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Model checkpoint was not saved."
    print("    Training completed. Checkpoint verified.")

    # -------------------------------------------------------------------------
    # 5. Inference Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[5] Executing Inference Pipeline (run_inference)...")

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Run inference explicitly
    # This generates submission.csv using Multi-View Ensemble Scanning
    run_inference(model, device)

    assert os.path.exists(Config.SUBMISSION_PATH), "submission.csv was not generated."
    print(f"    Inference completed. Submission saved to {Config.SUBMISSION_PATH}")

    # -------------------------------------------------------------------------
    # 6. Output Validation
    # -------------------------------------------------------------------------
    print("\n[6] Validating Submission File...")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["Id", "Predicted"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df_sub.columns)}"

    # Check that we have rows for the test fragments (fragment 'a')
    # Note: The test metadata provided in the environment usually contains 'a'.
    assert len(df_sub) > 0, "Submission file is empty."
    assert "a" in df_sub["Id"].values, "Fragment 'a' not found in submission."

    # Check RLE format (string)
    rle_sample = df_sub.iloc[0]["Predicted"]
    if pd.isna(rle_sample) or rle_sample == "":
        print(
            "    Note: RLE is empty (no ink detected), which is valid but check logic if unexpected."
        )
    else:
        assert isinstance(rle_sample, str), "Predicted column should be string (RLE)."
        # Basic check: space delimited numbers
        parts = rle_sample.split(" ")
        assert (
            len(parts) % 2 == 0
        ), "RLE string must have even number of elements (start length pairs)."

    print("    Submission file format verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
