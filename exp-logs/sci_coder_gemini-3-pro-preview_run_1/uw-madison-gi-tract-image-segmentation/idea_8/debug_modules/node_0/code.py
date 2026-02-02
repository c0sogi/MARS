import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from unittest.mock import patch

# Import library components
import library.config as config
import library.utils as utils
from library.trainer import Trainer
from library.inference import InferenceEngine

# -----------------------------------------------------------------------------
# 1. Setup & Data Mocking Strategy
# -----------------------------------------------------------------------------
# To ensure the demo runs quickly, we intercept pandas.read_csv calls and
# return small subsets of the real metadata. This prevents processing thousands of images.


def run_demo():
    print("=== Starting High-Fidelity Recurrent U-Net Pipeline Demo ===")

    # Set seeds for reproducibility
    utils.set_seed(config.SEED)

    # Load real metadata but keep only a tiny subset (e.g., 1 case, few slices)
    # We use keep_default_na=False to match the dataset class behavior
    print("Preparing data subsets...")

    full_train = pd.read_csv(config.TRAIN_CSV, keep_default_na=False)
    # Filter for just one case to ensure we have a valid sequence
    train_case = full_train["case"].unique()[0]
    subset_train = full_train[full_train["case"] == train_case].head(32).copy()

    full_val = pd.read_csv(config.VAL_CSV, keep_default_na=False)
    val_case = full_val["case"].unique()[0]
    subset_val = full_val[full_val["case"] == val_case].head(32).copy()

    full_test = pd.read_csv(config.TEST_CSV, keep_default_na=False)
    test_case = full_test["case"].unique()[0]
    subset_test = full_test[full_test["case"] == test_case].head(32).copy()

    # Store original read_csv to allow fallback if needed
    original_read_csv = pd.read_csv

    def mock_read_csv(filepath, *args, **kwargs):
        path_str = str(filepath)
        if "train.csv" in path_str:
            return subset_train
        elif "val.csv" in path_str:
            return subset_val
        elif "test.csv" in path_str:
            return subset_test
        return original_read_csv(filepath, *args, **kwargs)

    # Apply the patch context for the duration of the demo
    with patch("pandas.read_csv", side_effect=mock_read_csv):

        # -------------------------------------------------------------------------
        # 2. Verify Utility Functions
        # -------------------------------------------------------------------------
        print("\n[1/4] Verifying Utility Functions...")

        # Test RLE Encoding/Decoding
        mask_shape = (100, 100)
        dummy_mask = np.zeros(mask_shape, dtype=np.uint8)
        dummy_mask[10:20, 10:20] = 1  # Create a square

        rle_str = utils.rle_encode(dummy_mask)
        decoded_mask = utils.rle_decode(rle_str, mask_shape)

        assert np.array_equal(dummy_mask, decoded_mask), "RLE Decode mismatch"
        print(" - RLE Encode/Decode: OK")

        # Test Metrics
        # Perfect overlap
        d_score = utils.dice_coef(dummy_mask, dummy_mask)
        assert np.isclose(d_score, 1.0), f"Dice should be 1.0, got {d_score}"

        # 3D Hausdorff (Create 3D volume 1x100x100)
        vol_a = dummy_mask[np.newaxis, ...]
        vol_b = dummy_mask[np.newaxis, ...]
        h_dist = utils.hausdorff_3d_distance(vol_a, vol_b)
        assert h_dist == 0.0, f"Hausdorff distance should be 0.0, got {h_dist}"
        print(" - Metrics (Dice/Hausdorff): OK")

        # -------------------------------------------------------------------------
        # 3. Training Demonstration
        # -------------------------------------------------------------------------
        print("\n[2/4] Initializing Trainer...")
        # Initialize Trainer (this will trigger dataset processing with our mocked subset)
        trainer = Trainer(load_cached_data=False)  # Force processing to test pipeline

        print(" - Running Train Epoch (1 epoch, subset data)...")
        # Run one epoch
        loss = trainer.train_epoch(epoch=0)
        assert not np.isnan(loss), "Training loss is NaN"
        print(f" - Train Loss: {loss:.4f}")

        print(" - Running Validation...")
        # Run validation
        val_score = trainer.validate(epoch=0)
        assert not np.isnan(val_score), "Validation score is NaN"
        print(f" - Validation Score: {val_score:.4f}")

        # Manually save the model to ensure a checkpoint exists for the inference step
        # (Trainer.fit usually handles this, but we are running steps manually)
        print(" - Saving checkpoint for inference...")
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        torch.save(
            trainer.model.state_dict(),
            os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
        )

        # -------------------------------------------------------------------------
        # 4. Inference Demonstration
        # -------------------------------------------------------------------------
        print("\n[3/4] Initializing Inference Engine...")
        inference_engine = InferenceEngine(load_cached_data=False)

        # Load the checkpoint we just saved
        inference_engine.load_checkpoint("best_model.pth")

        print(" - Running Prediction on Test Subset...")
        inference_engine.predict_volume()

        # -------------------------------------------------------------------------
        # 5. Result Verification
        # -------------------------------------------------------------------------
        print("\n[4/4] Verifying Submission...")
        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")

        if os.path.exists(submission_path):
            sub_df = pd.read_csv(submission_path)
            print(f" - Submission file found at {submission_path}")
            print(f" - Rows generated: {len(sub_df)}")
            print(" - Sample rows:")
            print(sub_df.head())

            # Basic checks
            assert "id" in sub_df.columns
            assert "class" in sub_df.columns
            assert "predicted" in sub_df.columns
            assert len(sub_df) > 0, "Submission file is empty"
        else:
            raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
