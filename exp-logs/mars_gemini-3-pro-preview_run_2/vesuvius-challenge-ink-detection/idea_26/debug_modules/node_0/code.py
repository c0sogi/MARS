import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2

# Import from the provided library
from library.config import Config
from library.utils import sigmoid, rle_encoding, dice_coefficient
from library.model import InkSegFormer
from library.train import train_model
from library.inference import predict_and_submit


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demo by overriding Config
    and creating subset metadata to ensure the script runs quickly.
    """
    print(">>> Setting up demo environment...")

    # 1. Define paths
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_run")
    demo_cache_dir = os.path.join(demo_dir, "cache")
    demo_meta_dir = os.path.join(demo_dir, "metadata")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_meta_dir, exist_ok=True)

    # 2. Subset Metadata
    # We read the original metadata and take only a few rows to speed up data loading
    # and volume caching (only relevant fragments will be loaded).

    # Train
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    train_subset = train_df.head(4).copy()  # 4 samples = 1 batch if batch_size=4
    train_subset_path = os.path.join(demo_meta_dir, "train.csv")
    train_subset.to_csv(train_subset_path, index=False)

    # Validation
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_subset = val_df.head(4).copy()
    val_subset_path = os.path.join(demo_meta_dir, "validation.csv")
    val_subset.to_csv(val_subset_path, index=False)

    # Test
    # Test metadata usually contains fragment-level info. We keep it as is
    # or limit it if there are many fragments (usually just 'a' and 'b').
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    test_subset = test_df.head(1).copy()
    test_subset_path = os.path.join(demo_meta_dir, "test.csv")
    test_subset.to_csv(test_subset_path, index=False)

    # 3. Override Config
    # We modify the class attributes directly.
    Config.CACHE_DIR = demo_cache_dir
    Config.TRAIN_METADATA_PATH = train_subset_path
    Config.VAL_METADATA_PATH = val_subset_path
    Config.TEST_METADATA_PATH = test_subset_path
    Config.SUBMISSION_FILE = os.path.join(demo_dir, "submission.csv")

    # Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.TRAIN_Z_RANGE = (16, 17)  # Reduce randomness
    Config.INFERENCE_Z_STARTS = [16]  # Single view inference for speed

    print("    Config updated for demo (Epochs=1, Batch=2, Subset Data).")
    return demo_dir


def test_utilities():
    """
    Validates functions in library/utils.py
    """
    print("\n>>> Testing Utilities...")

    # 1. Sigmoid
    x = np.array([0.0, 100.0, -100.0])
    res = sigmoid(x)
    assert np.isclose(res[0], 0.5), "Sigmoid(0) should be 0.5"
    assert np.isclose(res[1], 1.0), "Sigmoid(100) should be close to 1"
    assert np.isclose(res[2], 0.0), "Sigmoid(-100) should be close to 0"
    print("    Sigmoid check passed.")

    # 2. RLE Encoding
    # Mask: 0 1 1 0 0 1 0
    # Indices (1-based): 2,3 are 1s (start 2, len 2), 6 is 1 (start 6, len 1)
    mask = np.array([[0, 1, 1, 0, 0, 1, 0]], dtype=np.uint8)
    rle = rle_encoding(mask)
    expected_rle = "2 2 6 1"
    assert rle == expected_rle, f"RLE failed. Expected '{expected_rle}', got '{rle}'"
    print("    RLE Encoding check passed.")

    # 3. Dice Coefficient
    # Perfect match
    pred = torch.tensor([10.0, 10.0, -10.0])  # Sigmoid -> ~1, ~1, ~0
    target = torch.tensor([1.0, 1.0, 0.0])
    score = dice_coefficient(pred, target, threshold=0.5)
    assert np.isclose(score, 1.0, atol=1e-4), f"Perfect Dice should be 1.0, got {score}"

    # No match
    pred_bad = torch.tensor([-10.0, -10.0, 10.0])  # Sigmoid -> 0, 0, 1
    score_bad = dice_coefficient(pred_bad, target, threshold=0.5)
    # TP=0, FP=1, FN=2. F0.5 = (1.25 * 0) / (1.25*0 + 0.25*2 + 1) = 0
    assert np.isclose(
        score_bad, 0.0, atol=1e-4
    ), f"Mismatch Dice should be 0.0, got {score_bad}"
    print("    Dice Coefficient check passed.")


def test_model_architecture():
    """
    Validates InkSegFormer in library/model.py
    """
    print("\n>>> Testing Model Architecture...")

    model = InkSegFormer()
    model.eval()

    # Dummy input: (Batch, Channels, Height, Width)
    # SegFormer expects 3 channels, 512x512 is standard tile size
    dummy_input = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (Batch, 1, 512, 512)
    expected_shape = (2, 1, 512, 512)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("    Model forward pass successful. Output shape verified.")


def run_demo_training():
    """
    Runs the training loop using the subset metadata.
    """
    print("\n>>> Running Demo Training...")

    # This will load volumes (caching them), run 1 epoch, and save best_model.pth if valid.
    # Note: train_model saves to Config.CACHE_DIR/best_model.pth
    train_model(load_cached_data=False)

    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if os.path.exists(model_path):
        print("    Training completed. 'best_model.pth' found.")
    else:
        # It's possible validation score didn't beat baseline in 1 epoch with random weights.
        # For the sake of the demo, we force create a dummy model file if not created,
        # so inference can run.
        print(
            "    Training completed but baseline not beaten (expected for dummy run). Saving current model manually."
        )
        model = InkSegFormer()
        torch.save(model.state_dict(), model_path)


def run_demo_inference():
    """
    Runs the inference pipeline.
    """
    print("\n>>> Running Demo Inference...")

    predict_and_submit(load_cached_data=True)

    if os.path.exists(Config.SUBMISSION_FILE):
        print(
            f"    Inference completed. Submission file found at {Config.SUBMISSION_FILE}"
        )

        # Validate content
        df = pd.read_csv(Config.SUBMISSION_FILE)
        print("    Submission Head:")
        print(df.head())

        assert (
            "Id" in df.columns and "Predicted" in df.columns
        ), "Submission columns missing."
        assert len(df) > 0, "Submission file is empty."
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    try:
        # 1. Setup
        demo_dir = setup_demo_environment()

        # 2. Test Utils
        test_utilities()

        # 3. Test Model
        test_model_architecture()

        # 4. Run Training
        run_demo_training()

        # 5. Run Inference
        run_demo_inference()

        print("\n>>> All demo steps completed successfully.")

    except Exception as e:
        print(f"\n>>> Error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
