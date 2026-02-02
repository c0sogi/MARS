import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import StegoDataset
from library.model import DRRENet
from library.engine import StegoEngine
from library.utils import seed_everything, alaska_weighted_auc


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Override Config for Fast Baseline Execution
    # We limit epochs and use a larger batch size for the A100 GPU
    Config.NUM_EPOCHS = 3
    Config.BATCH_SIZE = 32
    TRAIN_SUBSET_SIZE = 40000  # Train on a subset to ensure < 2 hours runtime

    print(f"Running on device: {device}")
    print(f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = StegoDataset(split="train")
    val_dataset = StegoDataset(split="val")
    test_dataset = StegoDataset(split="test")

    # Subsample training data for speed
    if len(train_dataset.df) > TRAIN_SUBSET_SIZE:
        print(
            f"Subsampling training data from {len(train_dataset.df)} to {TRAIN_SUBSET_SIZE} samples."
        )
        train_dataset.df = train_dataset.df.sample(
            n=TRAIN_SUBSET_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model (DRRENet)...")
    model = DRRENet().to(device)

    # 4. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )

    # 5. Training
    engine = StegoEngine(model, device, optimizer, scheduler)
    engine.train_model(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # 6. Final Validation & Metric Calculation
    print("\nRunning Final Validation on Best Model...")

    # Load best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model not found, using current weights.")

    model.eval()

    preds_list = []
    targets_list = []

    # Validation Inference (No Gradients)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Forward pass
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)

            preds_list.extend(probs.cpu().numpy())
            targets_list.extend(labels.cpu().numpy())

    preds_arr = np.array(preds_list)
    targets_arr = np.array(targets_list)

    # Compute Metric
    final_metric = alaska_weighted_auc(targets_arr, preds_arr)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(targets_arr - preds_arr)

    # Prepare metadata for analysis
    # We use the validation dataframe to get file paths and labels
    analysis_df = val_dataset.df.copy()
    analysis_df["error"] = errors

    # Extract File Size feature
    file_sizes = []
    for rel_path in analysis_df["file_path"]:
        full_path = os.path.join(Config.INPUT_ROOT, rel_path)
        try:
            file_sizes.append(os.path.getsize(full_path))
        except OSError:
            file_sizes.append(0)
    analysis_df["file_size"] = file_sizes

    # Calculate Correlations
    corr_filesize = analysis_df["error"].corr(analysis_df["file_size"])
    corr_label = analysis_df["error"].corr(analysis_df["label"])

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  File Size: {corr_filesize}")
    print(f"  Label (Class): {corr_label}")

    # 8. Submission Generation
    THRESHOLD = 0.8275809238338193

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        engine.generate_submission(test_loader)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
