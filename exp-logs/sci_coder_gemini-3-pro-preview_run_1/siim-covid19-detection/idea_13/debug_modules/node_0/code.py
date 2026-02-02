import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config, seed_everything
from library.dataset import SIIMDataset
from library.model import ResNet18D_UNet
from library.loss import HybridLoss
from library.utils import mask2bbox, calculate_map
from library.training import run_training
from library.inference import predict_and_submit


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo by creating a working directory
    and creating small subsets of the metadata to speed up execution.
    """
    print(">>> Setting up demo environment...")

    # Define new working directory
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Monkey-patch Config to use this directory and reduce load
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_METADATA = os.path.join(demo_dir, "train.csv")
    Config.VAL_METADATA = os.path.join(demo_dir, "val.csv")
    Config.TEST_METADATA = os.path.join(demo_dir, "test.csv")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Update BEST_MODEL_PATH manually since it was defined at class level
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model.pth")

    # Reduce hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2

    # Create subsets of metadata
    # We read from the original metadata provided in the task environment
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample subsets (ensure we have enough for a batch)
    subset_train = orig_train.head(16)
    subset_val = orig_val.head(8)
    subset_test = orig_test.head(8)

    # Save subsets to the demo directory
    subset_train.to_csv(Config.TRAIN_METADATA, index=False)
    subset_val.to_csv(Config.VAL_METADATA, index=False)
    subset_test.to_csv(Config.TEST_METADATA, index=False)

    print(f"    Created subset metadata in {demo_dir}")
    print(
        f"    Train: {len(subset_train)}, Val: {len(subset_val)}, Test: {len(subset_test)}"
    )


def test_utils():
    """
    Verifies the correctness of utility functions.
    """
    print("\n>>> Testing Utilities...")

    # 1. Test mask2bbox
    # Create a 100x100 mask with a 10x10 square at (10, 10)
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[10:20, 10:20] = 1.0

    boxes = mask2bbox(mask, threshold=0.5)
    # Expected: one box [10, 10, 20, 20] (approximate depending on contour finding)
    # cv2.boundingRect returns x, y, w, h. mask2bbox converts to x1, y1, x2, y2.
    assert len(boxes) == 1, "mask2bbox should find exactly one box"
    x1, y1, x2, y2 = boxes[0]
    # Allow 1 pixel tolerance for contour approximation
    assert 9 <= x1 <= 11 and 9 <= y1 <= 11, f"Box coordinates incorrect: {boxes[0]}"
    assert 19 <= x2 <= 21 and 19 <= y2 <= 21, f"Box coordinates incorrect: {boxes[0]}"
    print("    mask2bbox passed.")

    # 2. Test calculate_map
    # Perfect match case
    pred_boxes = [[[10, 10, 50, 50]]]
    pred_scores = [[0.9]]
    pred_labels = [[1]]
    gt_boxes = [[[10, 10, 50, 50]]]
    gt_labels = [[1]]

    map_score = calculate_map(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    assert np.isclose(
        map_score, 1.0
    ), f"Perfect match mAP should be 1.0, got {map_score}"

    # No match case
    pred_boxes_bad = [[[100, 100, 150, 150]]]  # No overlap
    map_score_bad = calculate_map(
        pred_boxes_bad, pred_scores, pred_labels, gt_boxes, gt_labels
    )
    assert np.isclose(
        map_score_bad, 0.0
    ), f"No match mAP should be 0.0, got {map_score_bad}"
    print("    calculate_map passed.")


def test_dataset_and_model():
    """
    Verifies Dataset loading and Model forward pass.
    """
    print("\n>>> Testing Dataset and Model...")

    # 1. Test Dataset
    ds = SIIMDataset("train", load_cached_data=False)  # Force process to verify logic
    item = ds[0]

    # Check keys
    assert "image" in item and "mask" in item and "label" in item

    # Check shapes
    img = item["image"]
    mask = item["mask"]
    # Image: (3, 512, 512) - Channel first from ToTensorV2
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch: {img.shape}"
    # Mask: (1, 512, 512)
    assert mask.shape == (
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Mask shape mismatch: {mask.shape}"

    print(f"    Dataset loaded successfully. Image shape: {img.shape}")

    # 2. Test Model
    device = torch.device("cpu")  # Use CPU for simple shape check
    model = ResNet18D_UNet(num_classes=4, pretrained=False)
    model.eval()

    # Create dummy batch (B=2)
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE)

    with torch.no_grad():
        cls_logits, seg_logits = model(dummy_input)

    # Check output shapes
    # cls_logits: (B, 4)
    assert cls_logits.shape == (
        2,
        4,
    ), f"Class logits shape mismatch: {cls_logits.shape}"
    # seg_logits: (B, 1, H, W)
    assert seg_logits.shape == (
        2,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Seg logits shape mismatch: {seg_logits.shape}"

    print("    Model forward pass successful.")

    # 3. Test Loss
    criterion = HybridLoss()
    dummy_labels = torch.tensor([0, 1], dtype=torch.long)  # Classes
    dummy_masks = torch.randn(2, 1, Config.IMG_SIZE, Config.IMG_SIZE)  # Logits/Targets

    loss = criterion(cls_logits, seg_logits, dummy_labels, dummy_masks)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    print("    Loss calculation successful.")


def run_full_pipeline():
    """
    Runs the actual training and inference functions provided in the library
    using the subset data.
    """
    print("\n>>> Running Full Pipeline (Train + Inference)...")

    # 1. Run Training
    # This will use the monkey-patched Config paths and subset data
    print("    Starting Training Loop...")
    run_training()

    # Verify model was saved
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError("Training did not produce a best_model.pth checkpoint.")
    print("    Training complete. Checkpoint verified.")

    # 2. Run Inference
    print("    Starting Inference Loop...")
    predict_and_submit(load_cached_data=False)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Inference did not produce a submission.csv file.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Inference complete. Submission shape: {df_sub.shape}")

    # Basic check on submission content
    assert "Id" in df_sub.columns and "PredictionString" in df_sub.columns
    # We expect rows for study and image.
    # Test subset has 8 images -> 8 study rows + 8 image rows = 16 rows
    assert len(df_sub) == 16, f"Expected 16 rows in submission, got {len(df_sub)}"

    print(">>> Pipeline Verification Successful.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Unit Tests
        test_utils()

        # 3. Component Tests
        test_dataset_and_model()

        # 4. Integration Test
        run_full_pipeline()

        print("\nAll demonstrations passed successfully.")

    except Exception as e:
        print(f"\n!!! Error during demonstration: {e}")
        # Fail explicitly
        sys.exit(1)
