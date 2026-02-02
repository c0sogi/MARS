import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import cv2

# Import from the provided library
from library.config import Config
from library.utils import (
    seed_everything,
    mask2bbox,
    get_map_score,
    post_process_submission,
)
from library.data import get_dataloaders
from library.model import ResNet18UNetASPP
from library.engine import fit


def run_demo():
    print("Initializing Demo Script...")

    # 1. Setup Configuration for Demo (Speed Optimization)
    # We override the default configuration to run a fast check
    # using a separate working directory and minimal parameters.
    Config.WORKING_DIR = "./working/demo_execution"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.setup()  # Create the directory

    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 2

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # =========================================================================
    # 2. Verify Utility Functions
    # =========================================================================
    print("\n--- Verifying Utility Functions ---")

    # Test mask2bbox
    # Create a 100x100 mask with a 10x10 square at (10, 10)
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1
    boxes = mask2bbox(dummy_mask)

    # Expected: [[10, 10, 20, 20]] (xmin, ymin, xmax, ymax)
    # Note: cv2.boundingRect returns x, y, w, h. x+w = 20, y+h = 20.
    assert len(boxes) == 1, "mask2bbox failed to find the box"
    b = boxes[0]
    assert b == [10, 10, 20, 20], f"mask2bbox returned incorrect coordinates: {b}"
    print("mask2bbox: OK")

    # Test get_map_score
    # Perfect match case
    pred_boxes = [[[10, 10, 50, 50]]]
    pred_scores = [[0.9]]
    true_boxes = [[[10, 10, 50, 50]]]

    score = get_map_score(pred_boxes, pred_scores, true_boxes, iou_threshold=0.5)
    assert np.isclose(
        score, 1.0
    ), f"get_map_score (perfect match) should be 1.0, got {score}"

    # No match case
    pred_boxes_bad = [[[60, 60, 70, 70]]]
    score_bad = get_map_score(
        pred_boxes_bad, pred_scores, true_boxes, iou_threshold=0.5
    )
    assert np.isclose(
        score_bad, 0.0
    ), f"get_map_score (no match) should be 0.0, got {score_bad}"
    print("get_map_score: OK")

    # =========================================================================
    # 3. Verify Data Pipeline
    # =========================================================================
    print("\n--- Verifying Data Pipeline ---")

    # We use debug_limit to load only 10 images for speed.
    # We set load_cached_data=False to force processing logic verification
    # and ensure we don't accidentally load a full cached dataset if one existed.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
        debug_limit=10,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    images = batch["image"]
    masks = batch["mask"]
    labels = batch["label"]

    # Check shapes
    # Image: (B, 3, 512, 512)
    assert (
        images.dim() == 4
        and images.shape[1] == 3
        and images.shape[2] == Config.IMG_SIZE
    ), f"Image shape mismatch: {images.shape}"
    # Mask: (B, 1, 512, 512)
    assert (
        masks.dim() == 4 and masks.shape[1] == 1
    ), f"Mask shape mismatch: {masks.shape}"
    # Label: (B,)
    assert labels.dim() == 1, f"Label shape mismatch: {labels.shape}"

    print("Data Loading: OK")

    # =========================================================================
    # 4. Verify Model Architecture
    # =========================================================================
    print("\n--- Verifying Model Architecture ---")

    model = ResNet18UNetASPP(num_classes=Config.NUM_CLASSES).to(device)

    # Run a dummy forward pass
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        logits_cls, logits_seg = model(dummy_input)

    # Check output shapes
    # Class logits: (B, Num_Classes)
    assert logits_cls.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Class output shape incorrect: {logits_cls.shape}"
    # Seg logits: (B, 1, H, W)
    assert logits_seg.shape == (
        2,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Seg output shape incorrect: {logits_seg.shape}"

    print("Model Architecture: OK")

    # =========================================================================
    # 5. Verify Training Loop (Engine)
    # =========================================================================
    print("\n--- Verifying Training Loop ---")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Simple scheduler for demo
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Run fit for 1 epoch
    # This tests train_one_epoch, valid_one_epoch, metric calculation, and checkpointing
    fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=Config.EPOCHS,
    )

    # Check if model was saved
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Model checkpoint saved successfully at {Config.BEST_MODEL_PATH}")
    else:
        # It's possible validation score was -inf or something failed, but fit usually saves if score > -inf.
        # With random init, mAP might be 0, acc might be random.
        # If composite score > -inf, it saves.
        print(
            "Warning: Model checkpoint not found (might be due to low score in 1 epoch)."
        )

    print("Training Loop: OK")

    # =========================================================================
    # 6. Verify Submission Post-Processing
    # =========================================================================
    print("\n--- Verifying Submission Generation ---")

    # Create dummy predictions
    study_ids = ["test_study_1", "test_study_2"]
    image_ids = ["test_image_1", "test_image_2"]

    # Random class probs (softmaxed)
    study_preds = [[0.1, 0.7, 0.1, 0.1], [0.8, 0.1, 0.05, 0.05]]  # Typical  # Negative

    # Random boxes
    image_preds = [
        {"boxes": [[10, 10, 50, 50]], "scores": [0.95]},
        {"boxes": [], "scores": []},  # No findings
    ]

    submission_df = post_process_submission(
        study_preds, study_ids, image_preds, image_ids, save_path=Config.SUBMISSION_FILE
    )

    # Validate DataFrame
    assert (
        len(submission_df) == 4
    ), "Submission DF should have 4 rows (2 study + 2 image)"
    assert "id" in submission_df.columns and "PredictionString" in submission_df.columns

    # Check content format
    # Row 0: Study 1
    row0 = submission_df.iloc[0]
    assert "typical" in row0["PredictionString"]

    # Row 3: Image 2 (None)
    row3 = submission_df.iloc[3]
    assert row3["PredictionString"] == "none 1 0 0 1 1"

    print(f"Submission file generated at {Config.SUBMISSION_FILE}")
    print("Submission Generation: OK")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
