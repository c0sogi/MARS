import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# 1. Import Library Components
from library.config import Config
from library.dataset import SIIMDataset, collate_fn
from library.model import build_model
from library.engine import train_one_epoch, evaluate
from library.utils import seed_everything

# ==================================================================================
# Strategy:
# We modify the Config class attributes directly before other modules fully utilize them.
# This ensures that the demo runs quickly (smaller images, debug dataset) and fits
# within the runtime constraints while still exercising all code paths.
# ==================================================================================


def run_demo():
    print("=== Starting Library Demonstration ===\n")

    # ------------------------------------------------------------------------------
    # 1. Setup & Configuration
    # ------------------------------------------------------------------------------
    print("[1] Configuring environment for demo...")

    # Override Config for speed and memory safety on the demo run
    Config.IMG_SIZE = 512  # Reduce from 1024 to 512 for speed
    Config.BATCH_SIZE = 2  # Small batch size
    Config.EPOCHS = 1  # Only run 1 epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEBUG = True  # Use subset of data (50 samples)
    Config.PATIENCE = 1  # Fail fast if no improvement

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Image Size: {Config.IMG_SIZE}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # ------------------------------------------------------------------------------
    # 2. Dataset Loading
    # ------------------------------------------------------------------------------
    print("\n[2] Initializing Datasets...")

    # Initialize Train and Val datasets
    # debug=True loads only the first 50 rows from metadata
    train_dataset = SIIMDataset("train", debug=True)
    val_dataset = SIIMDataset("val", debug=True)

    print(f"    Train Dataset Size: {len(train_dataset)}")
    print(f"    Val Dataset Size: {len(val_dataset)}")

    # Verify Data Loading
    sample_img, sample_target = train_dataset[0]

    # Assertions to verify logic
    assert sample_img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {sample_img.shape}"
    assert "boxes" in sample_target, "Target dict missing 'boxes'"
    assert "study_label" in sample_target, "Target dict missing 'study_label'"

    print("    Sample verification passed.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # ------------------------------------------------------------------------------
    # 3. Model Building
    # ------------------------------------------------------------------------------
    print("\n[3] Building Multi-Task DINO Model...")

    # build_model returns the model and the specific criterion (loss function)
    model, criterion = build_model(Config)
    model.to(device)
    criterion.to(device)

    print("    Model built successfully.")
    print(f"    Backbone: {Config.BACKBONE}")
    print(f"    Num Queries: {Config.NUM_QUERIES}")

    # ------------------------------------------------------------------------------
    # 4. Training Loop Simulation
    # ------------------------------------------------------------------------------
    print("\n[4] Running Training Step (1 Epoch)...")

    # Setup Optimizer (simplified for demo)
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": Config.BACKBONE_LR,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run training for one epoch
    # This exercises the forward pass, loss calculation (Hungarian Matcher + SetCriterion), and backward pass
    train_stats = train_one_epoch(
        model, criterion, train_loader, optimizer, device, epoch=1, accumulation_steps=1
    )

    print(f"    Train Stats: {train_stats}")
    assert "loss" in train_stats, "Train stats missing 'loss'"

    # ------------------------------------------------------------------------------
    # 5. Evaluation Simulation
    # ------------------------------------------------------------------------------
    print("\n[5] Running Evaluation...")

    # Run evaluation on the validation set
    # This exercises the metric calculation (mAP and Study Accuracy)
    val_stats = evaluate(model, criterion, val_loader, device)

    print(f"    Val Stats: {val_stats}")
    assert "map" in val_stats, "Val stats missing 'map'"
    assert "study_acc" in val_stats, "Val stats missing 'study_acc'"

    # ------------------------------------------------------------------------------
    # 6. Inference Demonstration
    # ------------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    # Initialize Test Dataset (using a manual subset to speed up demo)
    test_dataset = SIIMDataset("test", load_cached_data=False)
    # Manually slice dataframe to run inference on just 5 images
    test_dataset.df = test_dataset.df.iloc[:5]

    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model.eval()
    results = []

    with torch.no_grad():
        for samples, targets in test_loader:
            samples = samples.to(device)

            # Forward pass
            outputs = model(samples)

            # Verify Output Shapes
            # study_logits: (B, 4)
            assert outputs["study_logits"].shape == (
                samples.shape[0],
                4,
            ), "Incorrect study_logits shape"
            # pred_boxes: (B, NUM_QUERIES, 4)
            assert outputs["pred_boxes"].shape == (
                samples.shape[0],
                Config.NUM_QUERIES,
                4,
            ), "Incorrect pred_boxes shape"

            # Simulate prediction extraction (simplified from library.engine.inference)
            study_probs = torch.softmax(outputs["study_logits"], dim=1)
            pred_boxes = outputs["pred_boxes"]

            # Just print the first prediction as a sanity check
            print(
                f"    Batch processed. Sample 0 Study Probs: {study_probs[0].cpu().numpy()}"
            )
            print(
                f"    Batch processed. Sample 0 Box 0 (Norm): {pred_boxes[0, 0].cpu().numpy()}"
            )

            results.append(outputs)

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
