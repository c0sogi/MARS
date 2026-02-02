import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# Import provided library modules
from library.configuration import Config
from library.utilities import seed_everything, map5, AverageMeter
from library.data_loader import get_dataloaders, WhaleDataset
from library.architecture import WhaleArcFaceModel
from library.engine import run_training, eval_fn


def main():
    print("=== Starting Whale Identification Demo Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment...")

    # Override Config for a fast demo run
    Config.debug = True  # Use small subset of data
    Config.epochs = 1  # Run only 1 epoch
    Config.batch_size = 8  # Small batch size
    Config.num_workers = 2  # Reduce workers for simple demo
    Config.working_dir = "./working/demo_execution"
    Config.checkpoint_dir = Config.working_dir

    # Clean/Create working directory
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.seed)

    print(f"Debug Mode: {Config.debug}")
    print(f"Device: {Config.device}")
    print(f"Working Directory: {Config.working_dir}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[Step 2] Initializing DataLoaders...")

    # Load data (load_cached_data=False forces re-computation of classes based on debug subset)
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        load_cached_data=False
    )

    print(f"Number of classes in debug set: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Assertions to verify data loading
    assert len(train_loader) > 0, "Train loader is empty"
    assert num_classes > 0, "No classes found"

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Sample Batch - Images: {images.shape}, Labels: {labels.shape}")

    assert images.shape == (
        Config.batch_size,
        3,
        Config.img_size,
        Config.img_size,
    ), f"Incorrect image shape: {images.shape}"
    assert labels.shape == (
        Config.batch_size,
    ), f"Incorrect label shape: {labels.shape}"

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\n[Step 3] Initializing Model...")

    model = WhaleArcFaceModel(num_classes=num_classes)
    model.to(Config.device)

    # Verify Forward Pass (Training Mode - with labels)
    model.train()
    images = images.to(Config.device)
    labels = labels.to(Config.device)

    outputs = model(images, labels)
    print(f"Model Output Shape (Train): {outputs.shape}")

    # ArcFace output should be (Batch_Size, Num_Classes)
    assert outputs.shape == (
        Config.batch_size,
        num_classes,
    ), "Model output shape mismatch in training mode"

    # Verify Forward Pass (Inference Mode - no labels)
    model.eval()
    with torch.no_grad():
        outputs_inf = model(images)
    print(f"Model Output Shape (Inference): {outputs_inf.shape}")
    assert outputs_inf.shape == (
        Config.batch_size,
        num_classes,
    ), "Model output shape mismatch in inference mode"

    # --------------------------------------------------------------------------
    # 4. Metric Verification (MAP@5)
    # --------------------------------------------------------------------------
    print("\n[Step 4] Verifying MAP@5 Metric...")

    # Case 1: Perfect prediction
    # Target: 0, Preds: [0, 1, 2, 3, 4] -> Rank 0 -> Score 1.0
    t1 = np.array([0])
    p1 = np.array([[0, 1, 2, 3, 4]])
    score1 = map5(p1, t1)
    assert abs(score1 - 1.0) < 1e-6, f"Expected 1.0, got {score1}"

    # Case 2: Prediction at rank 2 (3rd item)
    # Target: 2, Preds: [0, 1, 2, 3, 4] -> Rank 2 -> Score 1/3
    t2 = np.array([2])
    p2 = np.array([[0, 1, 2, 3, 4]])
    score2 = map5(p2, t2)
    assert abs(score2 - (1.0 / 3.0)) < 1e-6, f"Expected 0.333..., got {score2}"

    # Case 3: Target not in top 5
    # Target: 9, Preds: [0, 1, 2, 3, 4] -> Score 0.0
    t3 = np.array([9])
    p3 = np.array([[0, 1, 2, 3, 4]])
    score3 = map5(p3, t3)
    assert score3 == 0.0, f"Expected 0.0, got {score3}"

    print("MAP@5 Metric verification passed.")

    # --------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[Step 5] Running Training Loop (1 Epoch)...")

    # Re-initialize model to ensure clean state
    model = WhaleArcFaceModel(num_classes=num_classes)
    model.to(Config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # Execute training engine
    best_map5 = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.device,
        num_epochs=Config.epochs,
    )

    print(f"Training finished. Best Validation MAP@5: {best_map5}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.checkpoint_dir, "model_best.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint verified at: {checkpoint_path}")
    else:
        # It's possible no improvement happened if val score was 0 and best_score init at -1,
        # but logic says best_score starts at -1.0, so 0.0 >= -1.0.
        # Actually, if val_score > best_score. If val_score is 0 and best is -1, it saves.
        # However, run_training initializes best_score = -1.0.
        # Let's check if the file exists.
        print(
            "Warning: Checkpoint not found. (This might happen if validation score didn't improve, though unlikely with init -1.0)"
        )

    # --------------------------------------------------------------------------
    # 6. Inference Demonstration
    # --------------------------------------------------------------------------
    print("\n[Step 6] Running Inference on Test Set...")

    model.eval()
    test_preds = []
    test_imgs = []

    # Run for just one batch to demonstrate
    with torch.no_grad():
        for i, (images, filenames) in enumerate(test_loader):
            images = images.to(Config.device)

            # Get logits
            logits = model(images)

            # Get Top 5
            _, top5_indices = torch.topk(logits, 5, dim=1)

            # Convert to numpy
            top5_indices = top5_indices.cpu().numpy()

            print(
                f"Test Batch {i} - Filenames: {filenames[0]}... Predictions shape: {top5_indices.shape}"
            )

            # Just do one batch for demo speed
            break

    print("\n=== Demo Script Completed Successfully ===")


if __name__ == "__main__":
    main()
