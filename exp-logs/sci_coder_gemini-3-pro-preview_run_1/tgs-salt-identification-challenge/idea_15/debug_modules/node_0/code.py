import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, calculate_iou_batch
from library.model import DeepResUNet
from library.losses import BCEDiceLoss, BCELovaszLoss
from library.dataset import SaltDataset
from library.train import Trainer
from library.predict import Predictor


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    # We override Config attributes to use a temporary directory and fast settings.
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Create subdirectories
    DEMO_META_DIR = os.path.join(DEMO_DIR, "metadata")
    os.makedirs(DEMO_META_DIR, exist_ok=True)

    # Override Config paths and params
    Config.WORKING_DIR = DEMO_DIR
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "demo_submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Point to our new mini metadata files (to be created)
    Config.TRAIN_CSV = os.path.join(DEMO_META_DIR, "train.csv")
    Config.VAL_CSV = os.path.join(DEMO_META_DIR, "val.csv")
    Config.TEST_CSV = os.path.join(DEMO_META_DIR, "test.csv")

    # Fast training settings
    Config.NUM_EPOCHS = 1
    Config.EPOCHS_PER_CYCLE = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.setup()  # Create the directories defined in Config

    set_seed(42)

    # -------------------------------------------------------------------------
    # 2. Create Mini Metadata (Subset of Data)
    # -------------------------------------------------------------------------
    print("\n[1/6] Creating mini datasets for speed...")

    # Read original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Take top 10 samples for train/val, 6 for test
    mini_train = orig_train.head(10).copy()
    mini_val = orig_val.head(6).copy()
    mini_test = orig_test.head(6).copy()

    # Save to demo location
    mini_train.to_csv(Config.TRAIN_CSV, index=False)
    mini_val.to_csv(Config.VAL_CSV, index=False)
    mini_test.to_csv(Config.TEST_CSV, index=False)

    print(
        f"Created mini train ({len(mini_train)}), val ({len(mini_val)}), test ({len(mini_test)})"
    )

    # -------------------------------------------------------------------------
    # 3. Verify Utilities (RLE & IoU)
    # -------------------------------------------------------------------------
    print("\n[2/6] Verifying Utilities...")

    # Test RLE Encode/Decode
    # Create a simple 4x4 mask
    # Shape (4,4). Let's put a 2x2 block in the top-left.
    # Pixels: (0,0), (0,1), (1,0), (1,1) are 1.
    # Flattened (column-major order):
    # Col 0: 1, 1, 0, 0
    # Col 1: 1, 1, 0, 0
    # Col 2: 0...
    # Col 3: 0...
    # Flat sequence: 1, 1, 0, 0, 1, 1, 0...
    # Indices (1-based): 1, 2 (run 1 start 1 len 2), 5, 6 (run 2 start 5 len 2)
    dummy_mask = np.zeros((4, 4), dtype=np.uint8)
    dummy_mask[0:2, 0:2] = 1

    encoded = rle_encode(dummy_mask)
    # Expected: "1 2 5 2"
    assert encoded == "1 2 5 2", f"RLE Encode failed. Got {encoded}, expected '1 2 5 2'"

    decoded = rle_decode(encoded, shape=(4, 4))
    assert np.array_equal(dummy_mask, decoded), "RLE Decode failed to reconstruct mask."
    print("RLE Encode/Decode verified.")

    # Test IoU Calculation
    # Pred: 1 1 0 0, True: 1 0 1 0
    # Intersection: 1 (first pixel). Union: 3 (first, second, third). IoU: 1/3.
    y_pred = np.array([[[1, 1], [0, 0]]], dtype=np.float32)  # (1, 2, 2)
    y_true = np.array([[[1, 0], [1, 0]]], dtype=np.float32)  # (1, 2, 2)

    iou = calculate_iou_batch(y_pred, y_true)
    assert np.isclose(
        iou[0], 1 / 3
    ), f"IoU Calculation failed. Got {iou[0]}, expected 0.333"
    print("IoU Calculation verified.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[3/6] Verifying DeepResUNet Model...")
    model = DeepResUNet()
    model.eval()

    # Input shape: (Batch, Channels=2, Height=128, Width=128)
    # Note: Config.IMG_SIZE is 128
    dummy_input = torch.randn(2, 2, 128, 128)

    with torch.no_grad():
        output = model(dummy_input)

    # In eval mode, output is just logits (B, 1, 128, 128)
    assert output.shape == (
        2,
        1,
        128,
        128,
    ), f"Model output shape mismatch: {output.shape}"
    print("Model forward pass (eval) verified.")

    # Test Deep Supervision output in train mode
    model.train()
    out_main, out_aux2, out_aux1 = model(dummy_input)
    assert out_main.shape == (2, 1, 128, 128)
    assert out_aux2.shape == (2, 1, 128, 128)  # Upsampled
    assert out_aux1.shape == (2, 1, 128, 128)  # Upsampled
    print("Model forward pass (train/deep supervision) verified.")

    # -------------------------------------------------------------------------
    # 5. Verify Dataset Loading
    # -------------------------------------------------------------------------
    print("\n[4/6] Verifying SaltDataset...")
    # Force reload to ensure we use the mini csvs
    dataset = SaltDataset(mode="train", load_cached_data=False)

    assert (
        len(dataset) == 10
    ), f"Dataset length mismatch. Expected 10, got {len(dataset)}"

    img, mask, img_id = dataset[0]

    # Check shapes
    # Image: (2, 128, 128) -> Channel 0 is seismic, Channel 1 is depth
    assert img.shape == (2, 128, 128), f"Image tensor shape wrong: {img.shape}"
    # Mask: (1, 128, 128)
    assert mask.shape == (1, 128, 128), f"Mask tensor shape wrong: {mask.shape}"
    # Depth channel check: Should be constant per image (dense depth map)
    depth_map = img[1, :, :].numpy()
    assert np.allclose(
        depth_map, depth_map[0, 0]
    ), "Depth channel is not constant as expected."

    print("Dataset loading and processing verified.")

    # -------------------------------------------------------------------------
    # 6. Verify Training Loop
    # -------------------------------------------------------------------------
    print("\n[5/6] Verifying Trainer...")
    trainer = Trainer()

    # Run 1 epoch
    print("Running 1 epoch of training...")
    train_loss = trainer.train_one_epoch(0)
    print(f"Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Train loss should be positive."

    # Run validation
    val_loss, val_map = trainer.validate()
    print(f"Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}")

    # Save a dummy checkpoint to simulate a successful run
    # (Trainer saves based on improvement, but we force save for the next step)
    trainer.save_checkpoint("best_model.pth")
    trainer.save_checkpoint("best_cycle_2.pth")

    # Verify checkpoint exists
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(ckpt_path), "Checkpoint file was not created."
    print("Training loop and checkpointing verified.")

    # -------------------------------------------------------------------------
    # 7. Verify Prediction Pipeline
    # -------------------------------------------------------------------------
    print("\n[6/6] Verifying Predictor...")
    predictor = Predictor()

    # Run prediction (disable TTA for speed)
    predictor.predict(use_tta=False)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        len(df_sub) == 6
    ), f"Submission should have 6 rows (mini test set), got {len(df_sub)}"
    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Submission columns missing."

    print("Prediction pipeline verified.")

    print("\n=== Demonstration Complete: All Systems Go ===")


if __name__ == "__main__":
    main()
