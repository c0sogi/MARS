import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import cv2

# Import from the provided library files
from library.config import Config
from library.dataset import KuzushijiDataset, get_label_map
from library.model import get_model
from library.engine import train_one_epoch, evaluate, inference
from library.utils import (
    get_train_transform,
    get_valid_transform,
    collate_fn,
    calculate_f1_score,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Kuzushiji Library Demo ===")

    # 1. Setup and Configuration
    # Set seed for reproducibility
    Config.set_seed(42)

    # Override Config for speed in this demo
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = (
        0  # Use main process for data loading to avoid overhead in demo
    )
    Config.EPOCHS = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Verify Metric Logic (Unit Test)
    print("\n--- Verifying F1 Score Metric Logic ---")
    # Create synthetic ground truth: One box [10, 10, 50, 50] with label 1
    gt_boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.int64)
    target = [{"boxes": gt_boxes, "labels": gt_labels}]

    # Case A: Perfect Center Match
    # Center of [10, 10, 50, 50] is (30, 30).
    # Prediction box [20, 20, 40, 40] -> Center (30, 30). Inside GT.
    pred_boxes_a = torch.tensor([[20, 20, 40, 40]], dtype=torch.float32)
    pred_labels_a = torch.tensor([1], dtype=torch.int64)
    pred_scores_a = torch.tensor([0.9], dtype=torch.float32)
    pred_a = [{"boxes": pred_boxes_a, "labels": pred_labels_a, "scores": pred_scores_a}]

    metric_a = calculate_f1_score(pred_a, target, score_threshold=0.5)
    assert (
        metric_a["f1"] == 1.0
    ), f"Expected F1=1.0 for perfect match, got {metric_a['f1']}"
    print("Case A (Perfect Match): Passed")

    # Case B: Wrong Label
    pred_b = [
        {"boxes": pred_boxes_a, "labels": torch.tensor([2]), "scores": pred_scores_a}
    ]
    metric_b = calculate_f1_score(pred_b, target, score_threshold=0.5)
    assert (
        metric_b["f1"] == 0.0
    ), f"Expected F1=0.0 for wrong label, got {metric_b['f1']}"
    print("Case B (Wrong Label): Passed")

    # Case C: Center Outside
    # GT is [10, 10, 50, 50]. Prediction center at (100, 100)
    pred_boxes_c = torch.tensor([[90, 90, 110, 110]], dtype=torch.float32)
    pred_c = [{"boxes": pred_boxes_c, "labels": pred_labels_a, "scores": pred_scores_a}]
    metric_c = calculate_f1_score(pred_c, target, score_threshold=0.5)
    assert (
        metric_c["f1"] == 0.0
    ), f"Expected F1=0.0 for outside box, got {metric_c['f1']}"
    print("Case C (Spatial Mismatch): Passed")

    # 3. Data Loading
    print("\n--- Initializing Dataset and DataLoader ---")
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Subset for speed
    train_subset = train_df.head(10).copy()
    val_subset = val_df.head(4).copy()

    print(f"Training subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")

    # Instantiate Datasets
    train_dataset = KuzushijiDataset(
        dataframe=train_subset,
        image_dir=Config.INPUT_DIR,
        transforms=get_train_transform(),
    )
    val_dataset = KuzushijiDataset(
        dataframe=val_subset,
        image_dir=Config.INPUT_DIR,
        transforms=get_valid_transform(),
    )

    # Verify __getitem__
    img, target = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Image should be a tensor"
    assert img.shape[0] == 3, "Image should have 3 channels"
    assert "boxes" in target and "labels" in target, "Target missing keys"
    assert target["boxes"].shape[1] == 4, "Boxes should be N x 4"
    print("Dataset __getitem__ check: Passed")

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # 4. Model Initialization
    print("\n--- Initializing Model ---")
    # Using default config parameters from library, but explicitly passing num_classes
    model = get_model(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )
    print("Model and Optimizer initialized.")

    # 5. Training Loop Demo
    print("\n--- Running Training Loop (1 Epoch) ---")
    loss = train_one_epoch(
        model, optimizer, train_loader, device, epoch=0, print_freq=2
    )
    assert isinstance(loss, float), "Train function should return a float loss"
    assert np.isfinite(loss), "Loss should be finite"
    print("Training loop completed successfully.")

    # 6. Evaluation Demo
    print("\n--- Running Evaluation ---")
    metrics = evaluate(model, val_loader, device)
    assert "f1" in metrics, "Metrics should contain F1 score"
    assert 0.0 <= metrics["f1"] <= 1.0, "F1 score should be between 0 and 1"
    print(f"Evaluation completed. F1: {metrics['f1']:.4f}")

    # 7. Inference Demo
    print("\n--- Running Inference ---")
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    test_subset = test_df.head(5).copy()

    # Test dataset (no transforms usually, or just resize/normalize)
    # The library's get_valid_transform is suitable for inference (resize + normalize)
    test_dataset = KuzushijiDataset(
        dataframe=test_subset,
        image_dir=Config.INPUT_DIR,
        transforms=get_valid_transform(),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Define output path for demo submission
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    inference(model, test_loader, device, output_path=demo_submission_path)

    # Verify submission file
    assert os.path.exists(demo_submission_path), "Submission file was not created"
    sub_df = pd.read_csv(demo_submission_path)
    assert len(sub_df) == len(
        test_subset
    ), "Submission should have same number of rows as test input"
    assert (
        "image_id" in sub_df.columns and "labels" in sub_df.columns
    ), "Submission columns mismatch"
    print("Inference completed and submission verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
