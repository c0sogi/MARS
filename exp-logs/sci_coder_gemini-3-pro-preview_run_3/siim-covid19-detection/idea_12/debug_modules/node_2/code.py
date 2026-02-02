import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import CovidDataset, load_dataset_metadata
from library.utils import collate_fn, format_prediction_string
from library.model import MultiTaskDINO
from library.loss import MultiTaskCriterion
from library.reasoning_head import DualStreamReasoningModule


def run_demo():
    print("=== Starting Multi-Task DINO Demo ===\n")

    # 1. Configuration Overrides for Speed
    print("--- Step 1: Configuring Environment ---")
    Config.DEBUG = True
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
    Config.EPOCHS = 1

    # Set seeds
    Config.set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Dataset & DataLoader Verification
    print("\n--- Step 2: Verifying Dataset and DataLoader ---")
    # Load metadata first to ensure cache is created/loaded
    try:
        train_meta = load_dataset_metadata("train", load_cached_data=True)
        print(f"Metadata loaded. Rows: {len(train_meta)}")
    except Exception as e:
        print(f"Metadata load warning: {e}")

    # Initialize Dataset
    train_dataset = CovidDataset(split="train", debug=True)
    val_dataset = CovidDataset(split="val", debug=True)

    print(f"Train Dataset Length (Debug): {len(train_dataset)}")
    print(f"Val Dataset Length (Debug): {len(val_dataset)}")

    # Check single item
    img, target, img_id = train_dataset[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Target Keys: {target.keys()}")

    assert img.shape[0] == 3, "Image should have 3 channels"
    assert "boxes" in target, "Target should contain boxes"
    assert "study_labels" in target, "Target should contain study_labels"

    # Initialize DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Fetch one batch
    batch_images, batch_targets, batch_ids = next(iter(train_loader))
    batch_images = batch_images.to(device)
    # Move targets to device
    batch_targets = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in batch_targets
    ]

    print(f"Batch Images Shape: {batch_images.shape}")
    print(f"Batch Size: {len(batch_targets)}")

    # 3. Model Instantiation & Forward Pass
    print("\n--- Step 3: Model Instantiation & Forward Pass ---")
    model = MultiTaskDINO()
    model.to(device)
    model.train()

    # Forward pass
    outputs = model(pixel_values=batch_images)

    print("Model Output Keys:", outputs.keys())

    # Verify Shapes
    # pred_logits: [Batch, Num_Queries, Num_Classes]
    # pred_boxes: [Batch, Num_Queries, 4]
    # study_logits: [Batch, Num_Study_Classes]

    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    study_logits = outputs["study_logits"]

    print(f"Pred Logits Shape: {pred_logits.shape}")
    print(f"Pred Boxes Shape: {pred_boxes.shape}")
    print(f"Study Logits Shape: {study_logits.shape}")

    assert pred_logits.shape[0] == Config.BATCH_SIZE
    assert pred_logits.shape[1] == Config.NUM_QUERIES
    assert pred_logits.shape[2] == Config.NUM_DETECTION_CLASSES

    assert pred_boxes.shape[0] == Config.BATCH_SIZE
    assert pred_boxes.shape[2] == 4

    assert study_logits.shape[0] == Config.BATCH_SIZE
    assert study_logits.shape[1] == Config.NUM_STUDY_CLASSES

    # 4. Loss Computation
    print("\n--- Step 4: Loss Computation ---")
    criterion = MultiTaskCriterion()
    criterion.to(device)

    losses = criterion(outputs, batch_targets)

    print("Loss Components:", losses.keys())
    print(f"Total Loss: {losses['total_loss'].item():.4f}")

    assert "loss_ce" in losses
    assert "loss_bbox" in losses
    assert "loss_giou" in losses
    assert "loss_study" in losses
    assert not torch.isnan(losses["total_loss"]), "Loss is NaN"

    # 5. Optimization Step
    print("\n--- Step 5: Optimization Step ---")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    losses["total_loss"].backward()

    # Check gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradients computed"

    optimizer.step()
    print("Optimizer step completed successfully.")

    # 6. Reasoning Head Logic Verification
    print("\n--- Step 6: Verifying Reasoning Head Logic ---")
    # Isolate the reasoning head to test dimension handling
    reasoning_head = DualStreamReasoningModule().to(device)

    # Create dummy inputs
    # Query embeds: [Batch, Num_Queries, Hidden_Dim]
    dummy_embeds = torch.randn(2, Config.NUM_QUERIES, Config.HIDDEN_DIM).to(device)
    # Pred boxes: [Batch, Num_Queries, 4]
    dummy_boxes = torch.rand(2, Config.NUM_QUERIES, 4).to(device)

    head_output = reasoning_head(dummy_embeds, dummy_boxes)
    print(f"Reasoning Head Output Shape: {head_output.shape}")

    assert head_output.shape == (2, Config.NUM_STUDY_CLASSES)

    # 7. Submission Formatting Verification
    print("\n--- Step 7: Submission Formatting ---")
    # Mock predictions
    study_preds = [
        {"id": "test_001_study", "class_id": 0, "conf": 0.95},
        {
            "id": "test_002",
            "class_id": 1,
            "conf": 0.88,
        },  # Should handle missing _study suffix
    ]

    image_preds = [
        {
            "id": "test_001_image",
            "boxes": [[10, 10, 50, 50]],
            "scores": [0.9],
            "study_neg": False,
        },
        {
            "id": "test_002",  # Should handle missing _image suffix
            "boxes": [],
            "scores": [],
            "study_neg": False,  # No boxes -> none
        },
        {
            "id": "test_003_image",
            "boxes": [[100, 100, 200, 200]],
            "scores": [0.6],
            "study_neg": True,  # Study is negative -> force none
        },
    ]

    submission_lines = format_prediction_string(study_preds, image_preds)

    print("Sample Submission Lines:")
    for line in submission_lines[:5]:
        print(f"  {line}")

    assert len(submission_lines) == 1 + len(study_preds) + len(
        image_preds
    )  # Header + rows
    assert "test_001_study,negative 0.950000 0 0 1 1" in submission_lines
    assert "test_003_image,none 1 0 0 1 1" in submission_lines

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
