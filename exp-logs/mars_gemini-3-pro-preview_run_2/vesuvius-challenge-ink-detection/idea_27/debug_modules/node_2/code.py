import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
import cv2

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_encode, fbeta_score
from library.dataset import InkDataset
from library.model import SegFormerB2
from library.train import run_training
from library.inference import inference


def main():
    # 1. Setup
    print("=== Starting Demonstration ===")
    set_seed(42)

    # Create a directory for our outputs
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config for speed and demo purposes
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # 2. Verify Utilities
    print("\n[1/5] Verifying Utilities...")

    # Test RLE Encoding
    # Mask: 0 0 1 1 1 0 0 1 0
    # Indices: 1 2 3 4 5 6 7 8 9
    # Runs: Start 3, Len 3; Start 8, Len 1
    dummy_mask = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    # Flattened: 0 0 1 1 1 0 0 1 0
    expected_rle = "3 3 8 1"
    encoded_rle = rle_encode(dummy_mask)
    assert (
        encoded_rle == expected_rle
    ), f"RLE Failed. Expected {expected_rle}, got {encoded_rle}"
    print("  RLE Encode: OK")

    # Test F-Beta Score
    # Pred: 0.8, 0.1 | Target: 1, 0 -> Perfect
    preds = torch.tensor([0.8, 0.1])
    targets = torch.tensor([1, 0])
    score = fbeta_score(preds, targets, beta=0.5, threshold=0.5)
    assert np.isclose(score, 1.0), f"F-Beta Failed. Expected 1.0, got {score}"
    print("  F-Beta Score: OK")

    # 3. Verify Data Loading and Model Forward Pass
    print("\n[2/5] Verifying Data Pipeline & Model...")

    # Initialize Dataset (Train)
    # Note: This relies on existing metadata in ./metadata/train.csv and files in ./input
    try:
        train_ds = InkDataset(mode="train")
        print(f"  Dataset initialized with {len(train_ds)} samples.")

        # Fetch one sample
        sample = train_ds[0]
        image = sample["image"]
        label = sample["label"]

        # Check shapes
        # Image: (3, 512, 512), Label: (1, 512, 512)
        assert image.shape == (3, 512, 512), f"Image shape mismatch: {image.shape}"
        assert label.shape == (1, 512, 512), f"Label shape mismatch: {label.shape}"
        print("  Data Loading: OK")

        # Initialize Model
        model = SegFormerB2()
        model.to(Config.DEVICE)

        # Create a batch
        images_batch = image.unsqueeze(0).to(Config.DEVICE)  # (1, 3, 512, 512)
        labels_batch = label.unsqueeze(0).to(Config.DEVICE)  # (1, 1, 512, 512)

        # Forward Pass
        output = model(images_batch, labels_batch)

        assert "logits" in output, "Model output missing logits"
        assert "loss" in output, "Model output missing loss"
        assert output["logits"].shape == (
            1,
            1,
            512,
            512,
        ), f"Logits shape mismatch: {output['logits'].shape}"
        print("  Model Forward Pass: OK")

    except Exception as e:
        print(f"  FAILED: {e}")
        raise e

    # 4. Verify Training Loop
    print("\n[3/5] Verifying Training Loop (Debug Mode)...")
    # Run training on a tiny subset
    best_score = run_training(debug=True, num_epochs=1, batch_size=2)

    # Check if model was saved
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(model_path):
        print(f"  Training completed. Model saved to {model_path}")
    else:
        # If score didn't improve over baseline (0.598), it might not save.
        # For demo, we just ensure it ran without error.
        print(
            "  Training ran successfully (no model saved, likely didn't beat baseline)."
        )

    # 5. Verify Inference (Mocked)
    print("\n[4/5] Verifying Inference Pipeline (Mocked Data)...")

    # Create Mock Data Structure to avoid processing large real test files
    mock_root = os.path.join(demo_dir, "mock_input")
    mock_frag_dir = os.path.join(mock_root, "test", "mock_frag")
    mock_vol_dir = os.path.join(mock_frag_dir, "surface_volume")
    os.makedirs(mock_vol_dir, exist_ok=True)

    # Create Mock Mask (512x512)
    # Make it small so inference is instant
    mock_h, mock_w = 512, 512
    mock_mask = np.zeros((mock_h, mock_w), dtype=np.uint8)
    # Add a valid region
    mock_mask[100:200, 100:200] = 255
    cv2.imwrite(os.path.join(mock_frag_dir, "mask.png"), mock_mask)

    # Create Mock Volume Slices (00.tif to 40.tif)
    # Random noise
    for i in range(45):
        slice_img = np.random.randint(0, 255, (mock_h, mock_w), dtype=np.uint8)
        cv2.imwrite(os.path.join(mock_vol_dir, f"{i:02d}.tif"), slice_img)

    # Create Mock Metadata
    mock_meta_path = os.path.join(demo_dir, "mock_test.csv")
    mock_df = pd.DataFrame(
        [
            {
                "fragment_id": "mock_frag",
                "mask_path": "test/mock_frag/mask.png",
                "volume_path": "test/mock_frag/surface_volume",
            }
        ]
    )
    mock_df.to_csv(mock_meta_path, index=False)

    # Override Config Paths
    original_input_dir = Config.INPUT_DIR
    original_test_meta = Config.TEST_METADATA_PATH

    Config.INPUT_DIR = mock_root
    Config.TEST_METADATA_PATH = mock_meta_path

    try:
        # Run Inference
        inference(
            checkpoint_path=model_path if os.path.exists(model_path) else None,
            batch_size=2,
        )

        # Verify Submission
        if os.path.exists(Config.SUBMISSION_PATH):
            sub_df = pd.read_csv(Config.SUBMISSION_PATH)
            print("  Submission file generated.")
            print(f"  Rows: {len(sub_df)}")
            print(f"  Columns: {sub_df.columns.tolist()}")

            # Check if we have a prediction for mock_frag
            row = sub_df[sub_df["Id"] == "mock_frag"]
            assert len(row) == 1, "Missing prediction for mock_frag"
            print("  Inference Logic: OK")
        else:
            raise FileNotFoundError("Submission file not created.")

    finally:
        # Restore Config
        Config.INPUT_DIR = original_input_dir
        Config.TEST_METADATA_PATH = original_test_meta
        # Cleanup Mock Data (Optional, but good practice)
        shutil.rmtree(mock_root)

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
