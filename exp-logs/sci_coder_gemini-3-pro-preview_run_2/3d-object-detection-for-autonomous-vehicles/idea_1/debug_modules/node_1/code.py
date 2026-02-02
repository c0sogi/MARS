import os
import torch
import numpy as np
import shutil
from torch.utils.data import DataLoader, Subset

# Import library modules
from library.config import Config, set_random_seed
from library.geometry import compute_iou_bev, quaternion_to_matrix
from library.anchors import AnchorGenerator
from library.data_loader import NuScenesDataset, collate_fn
from library.model import PointPillars
from library.loss import LossModule
from library.trainer import Trainer
from library.inference import Inference


# ==========================================
# 1. Configuration & Setup
# ==========================================
class FastConfig(Config):
    """
    Configuration optimized for a fast demonstration run.
    """

    # Reduce training duration
    EPOCHS = 1
    BATCH_SIZE = 2

    # Reduce model complexity/memory for demo
    MAX_PILLARS_TRAIN = 1000
    MAX_PILLARS_TEST = 1000

    # Use a specific working directory for this demo
    WORKING_DIR = "./working/demo_run"

    # Debug flag
    DEBUG = True


def run_demo():
    print("=== Starting 3D Object Detection Pipeline Demo ===")

    # Set seed
    set_random_seed(FastConfig.SEED)

    # Ensure working directory exists
    os.makedirs(FastConfig.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Geometry & Anchor Verification
    # ==========================================
    print("\n[1] Verifying Geometry and Anchors...")

    # Test IoU Calculation
    # Box: [x, y, z, w, l, h, yaw]
    # Box A: 2x2 square at origin
    box_a = np.array([0, 0, 0, 2, 2, 2, 0])
    # Box B: 2x2 square shifted by 1 in x (overlap should be 1*2 = 2, union = 4+4-2=6, IoU=0.333)
    box_b = np.array([1, 0, 0, 2, 2, 2, 0])

    iou = compute_iou_bev(box_a, box_b)
    print(f"  Computed IoU: {iou:.4f}")
    assert (
        abs(iou - (1.0 / 3.0)) < 1e-4
    ), f"IoU calculation failed. Expected 0.333, got {iou}"

    # Test Anchor Generator
    anchor_gen = AnchorGenerator(FastConfig)
    # Feature map size corresponding to 512x512 input grid with stride 2 backbone (256x256 output)
    # Actually backbone has 3 blocks with stride 2 each, but upsampling restores resolution.
    # Let's check config: Backbone strides [2, 2, 2].
    # Block 1: /2. Block 2: /4. Block 3: /8.
    # Neck upsamples: Block 1 (x1) -> /2. Block 2 (x2) -> /2. Block 3 (x4) -> /2.
    # So final output stride is 2 relative to input grid.
    # Input Grid: 512x512. Output Map: 256x256.
    feat_map_size = (256, 256)
    anchors = anchor_gen.generate(feat_map_size)

    expected_anchors = (
        256 * 256 * len(FastConfig.CLASS_NAMES) * len(FastConfig.ANCHOR_ROTATIONS)
    )
    # Shape: (H, W, Num_Types, 7)
    print(f"  Anchor Shape: {anchors.shape}")
    assert anchors.shape == (
        256,
        256,
        18,
        7,
    ), f"Incorrect anchor shape: {anchors.shape}"
    print("  Geometry and Anchors Verified.")

    # ==========================================
    # 3. Data Loading & Subsetting
    # ==========================================
    print("\n[2] Initializing Data Loaders (Subset)...")

    # Initialize Dataset
    # We use 'train' split. Metadata is already generated in ./metadata/train_metadata.csv
    full_dataset = NuScenesDataset(
        metadata_path=FastConfig.TRAIN_METADATA_PATH,
        split="train",
        config=FastConfig,
        load_cached_data=False,  # Force processing to demonstrate it works
    )

    # Create a small subset for speed
    subset_indices = list(range(min(10, len(full_dataset))))
    subset_dataset = Subset(full_dataset, subset_indices)

    demo_loader = DataLoader(
        subset_dataset,
        batch_size=FastConfig.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Fetch one batch to verify
    batch = next(iter(demo_loader))
    print(f"  Batch Keys: {batch.keys()}")
    print(f"  Num Points in Sample 0: {batch['points'][0].shape}")
    print(f"  Num Boxes in Sample 0: {batch['boxes'][0].shape}")

    assert "points" in batch and "boxes" in batch
    assert len(batch["points"]) == FastConfig.BATCH_SIZE
    print("  Data Loading Verified.")

    # ==========================================
    # 4. Model Initialization & Forward Pass
    # ==========================================
    print("\n[3] Model Initialization & Forward Pass...")

    model = PointPillars(FastConfig).to(FastConfig.DEVICE)
    model.train()

    # Move batch to device
    points = [p.to(FastConfig.DEVICE) for p in batch["points"]]

    # Forward
    cls_preds, reg_preds = model(points)

    print(f"  Cls Preds Shape: {cls_preds.shape}")
    print(f"  Reg Preds Shape: {reg_preds.shape}")

    # Expected: (B, 256, 256, 18, 9) and (B, 256, 256, 18, 7)
    # 18 anchors (9 classes * 2 rotations)
    # 9 classes
    assert cls_preds.shape[1:] == (256, 256, 18, 9)
    assert reg_preds.shape[1:] == (256, 256, 18, 7)
    print("  Model Forward Pass Verified.")

    # ==========================================
    # 5. Loss Calculation
    # ==========================================
    print("\n[4] Loss Calculation...")

    criterion = LossModule(FastConfig).to(FastConfig.DEVICE)

    gt_boxes = [b.to(FastConfig.DEVICE) for b in batch["boxes"]]
    gt_labels = [l.to(FastConfig.DEVICE) for l in batch["labels"]]

    loss, loss_dict = criterion(cls_preds, reg_preds, gt_boxes, gt_labels)

    print(f"  Total Loss: {loss.item():.4f}")
    print(f"  Breakdown: {loss_dict}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("  Loss Calculation Verified.")

    # ==========================================
    # 6. Training Loop Demonstration
    # ==========================================
    print("\n[5] Running Trainer (1 Epoch on Subset)...")

    trainer = Trainer(config=FastConfig, load_cached_data=True)

    # Inject our subset loader to make training fast
    trainer.loaders["train"] = demo_loader
    trainer.loaders["val"] = demo_loader  # Use same for val just to test pipeline

    # Run fit
    trainer.fit()

    # Check if checkpoint was saved
    checkpoint_path = os.path.join(FastConfig.WORKING_DIR, "idea_1", "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"  Checkpoint saved at {checkpoint_path}")
    else:
        # It might not save if validation loss doesn't improve (infinity start),
        # but first epoch usually improves over inf.
        # However, if batch size is small and data is random, it might fluctuate.
        # We just verify the code ran without error.
        print("  Trainer finished (checkpoint check optional depending on loss).")

    print("  Training Loop Verified.")

    # ==========================================
    # 7. Inference Components
    # ==========================================
    print("\n[6] Inference Pipeline Demonstration...")

    # Instantiate Inference
    # We point to the checkpoint we (hopefully) just created, or it will use random weights if not found
    inference = Inference(config=FastConfig, checkpoint_path=checkpoint_path)

    # We will manually run the inference steps on our batch to show usage
    inference.model.eval()
    with torch.no_grad():
        # Forward
        cls_preds, reg_preds = inference.model(points)

        # Decode
        decoded = inference.decode_predictions(cls_preds, reg_preds)
        print(f"  Decoded {len(decoded)} samples.")

        # NMS
        results = inference.apply_nms(decoded)
        print(f"  Post-NMS boxes for sample 0: {results[0]['boxes'].shape[0]}")

        # Format string
        # Transform back to global (using token from batch)
        token = batch["tokens"][0]
        boxes_sensor = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels = results[0]["labels"].cpu().numpy()

        boxes_global = inference.transform_to_global(boxes_sensor, token)
        pred_str = inference.format_prediction_string(boxes_global, scores, labels)

        print(f"  Prediction String (Truncated): {pred_str[:100]}...")

    print("  Inference Pipeline Verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
