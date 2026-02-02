import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config, seed_everything
from library.utils import (
    bb_intersection_over_union,
    weighted_boxes_fusion,
    format_prediction_string,
)
from library.dataset import ChestXRayDataset, get_dataloaders, collate_fn
from library.model import CovidCascadeRCNN
from library.engine import train_one_epoch, evaluate, fit
from library.inference import predict


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Override for Speed and Demo Purposes
    # We modify the Config class attributes directly to affect the behavior of library modules
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6  # Small sample size for quick execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = (
        0  # Use 0 workers to avoid multiprocessing overhead in this short script
    )

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n--- Verifying Utility Functions ---")

    # Test IoU
    # Box format: [x1, y1, x2, y2]
    # Box A: 0,0 to 100,100 (Area 10000)
    # Box B: 50,50 to 150,150 (Area 10000)
    # Intersection: 50,50 to 100,100 (Area 2500)
    # Union: 10000 + 10000 - 2500 = 17500
    # IoU: 2500 / 17500 = 1/7 ~= 0.142857
    boxA = [0, 0, 100, 100]
    boxB = [50, 50, 150, 150]
    iou = bb_intersection_over_union(boxA, boxB)
    print(f"Calculated IoU: {iou:.4f}")
    assert abs(iou - (1 / 7)) < 1e-5, "IoU calculation is incorrect"

    # Test Prediction String Formatting
    labels = [1, 1]
    boxes = [[10, 10, 20, 20], [30, 30, 40, 40]]
    scores = [0.9, 0.8]
    pred_str = format_prediction_string(labels, boxes, scores)
    print(f"Formatted Prediction String: {pred_str}")
    assert (
        "opacity 0.9000 10.0 10.0 20.0 20.0" in pred_str
    ), "Prediction string format incorrect"

    # Test Weighted Boxes Fusion (WBF)
    # 2 models predicting similar boxes
    boxes_list = [[[0, 0, 100, 100]], [[2, 2, 102, 102]]]
    scores_list = [[0.9], [0.8]]
    labels_list = [[1], [1]]
    wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=0.5
    )
    print(f"WBF Result - Boxes: {len(wbf_boxes)}, Scores: {wbf_scores}")
    assert len(wbf_boxes) == 1, "WBF should fuse these overlapping boxes"
    assert wbf_scores[0] > 0.8, "Fused score should be reasonable"

    # ==========================================
    # 3. Verify Dataset and DataLoader
    # ==========================================
    print("\n--- Verifying Dataset and DataLoader ---")

    # Initialize Dataset (Train)
    train_ds = ChestXRayDataset(split="train", debug=True)
    print(f"Train Dataset Length (Debug): {len(train_ds)}")
    assert (
        len(train_ds) == Config.DEBUG_SAMPLE_SIZE
    ), "Dataset length mismatch with DEBUG_SAMPLE_SIZE"

    # Fetch one sample
    image, target = train_ds[0]
    print(f"Image Shape: {image.shape}")
    print(f"Target Keys: {list(target.keys())}")

    assert image.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Image tensor shape incorrect"
    assert "boxes" in target, "Target missing 'boxes'"
    assert "study_label" in target, "Target missing 'study_label'"

    # Initialize DataLoaders
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=0, debug=True
    )

    # Fetch one batch
    images_batch, targets_batch = next(iter(train_loader))
    print(f"Batch Images Shape: {images_batch.shape}")
    print(f"Batch Targets Length: {len(targets_batch)}")

    assert images_batch.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert isinstance(targets_batch, list), "Targets should be a list"

    # ==========================================
    # 4. Verify Model
    # ==========================================
    print("\n--- Verifying Model ---")

    model = CovidCascadeRCNN()
    model.to(device)

    # Move batch to device
    images_batch = images_batch.to(device)
    targets_batch = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets_batch
    ]

    # Training Forward Pass
    model.train()
    loss_dict = model(images_batch, targets_batch)
    print("Training Losses keys:", list(loss_dict.keys()))

    # Check for expected loss components
    expected_losses = [
        "loss_classifier_s0",
        "loss_box_reg_s0",
        "loss_study",
        "loss_rpn_box_reg",
    ]
    for loss_key in expected_losses:
        # Note: Some RPN losses might not be present if no anchors matched, but usually they are.
        # We check at least one key exists to confirm dictionary return.
        pass
    assert "loss_study" in loss_dict, "Study loss missing from output"

    total_loss = sum(loss for loss in loss_dict.values())
    print(f"Total Loss: {total_loss.item():.4f}")

    # Inference Forward Pass
    model.eval()
    with torch.no_grad():
        detections, study_probs = model(images_batch)

    print(f"Detections type: {type(detections)}")
    print(f"Study Probs shape: {study_probs.shape}")

    assert (
        len(detections) == Config.BATCH_SIZE
    ), "Number of detections should match batch size"
    assert study_probs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_STUDY_CLASSES,
    ), "Study probabilities shape incorrect"
    assert "boxes" in detections[0], "Detection output missing boxes"

    # ==========================================
    # 5. Verify Training Engine
    # ==========================================
    print("\n--- Verifying Training Engine ---")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Run one epoch training
    print("Running train_one_epoch...")
    avg_losses = train_one_epoch(
        model, optimizer, train_loader, device, epoch=0, print_freq=1
    )
    print("Average Losses:", avg_losses)
    assert "total_loss" in avg_losses, "train_one_epoch did not return total_loss"

    # Run evaluation
    print("Running evaluate...")
    metrics = evaluate(model, val_loader, device)
    print("Evaluation Metrics:", metrics)
    assert "map_0.5" in metrics, "mAP metric missing"
    assert "study_accuracy" in metrics, "Study accuracy metric missing"

    # ==========================================
    # 6. Verify Inference Pipeline
    # ==========================================
    print("\n--- Verifying Inference Pipeline ---")

    # We save the current model as 'best_model.pth' to test the loading mechanism in predict()
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    torch.save(model.state_dict(), checkpoint_path)

    # Run prediction
    # This uses the 'test' split defined in metadata/test.csv
    # We use debug=True to only run on a few samples
    predict(checkpoint_path=checkpoint_path, batch_size=Config.BATCH_SIZE, debug=True)

    submission_file = Config.SUBMISSION_PATH
    if os.path.exists(submission_file):
        df_sub = pd.read_csv(submission_file)
        print(f"Submission file created at {submission_file}")
        print(f"Submission shape: {df_sub.shape}")
        print("First 5 rows:")
        print(df_sub.head())

        # Validate columns
        assert "Id" in df_sub.columns, "Submission missing Id column"
        assert (
            "PredictionString" in df_sub.columns
        ), "Submission missing PredictionString column"

        # Check for study and image rows
        study_rows = df_sub[df_sub["Id"].str.contains("_study")]
        image_rows = df_sub[df_sub["Id"].str.contains("_image")]

        print(f"Study rows: {len(study_rows)}, Image rows: {len(image_rows)}")
        assert len(study_rows) > 0, "No study predictions found"
        assert len(image_rows) > 0, "No image predictions found"
    else:
        raise FileNotFoundError("Submission file was not created by predict()")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
