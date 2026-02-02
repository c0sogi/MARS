import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import (
    SEED,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    IDEA_DIR,
    INPUT_ROOT,
)
from library.utils import seed_everything, weighted_auc_score
from library.dataset import StegoDataset, get_transforms
from library.model import MonoResidualResNet
from library.engine import run_training

# --- Configuration for Fast Baseline ---
# Limits set to ensure execution within ~30-45 minutes on A100
FAST_EPOCHS = 8
# We now subsample by Image IDs to maintain groups for balancing.
# 30,000 IDs * 2 samples (Cover+Stego) = 60,000 samples per epoch.
TRAIN_SUBSET_IDS = 30000


def main():
    # 1. Setup
    seed_everything(SEED)
    print(f"Device: {DEVICE}")

    # 2. Data Loading
    print("Loading metadata...")
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)

    # Subsample training data by Image ID to preserve groups (Cite solution_lesson_node_00002)
    unique_ids = train_df["image_id"].unique()
    if len(unique_ids) > TRAIN_SUBSET_IDS:
        print(
            f"Subsampling training data from {len(unique_ids)} IDs to {TRAIN_SUBSET_IDS} IDs..."
        )
        # Randomly select IDs
        selected_ids = np.random.choice(unique_ids, TRAIN_SUBSET_IDS, replace=False)
        # Filter dataframe to include all variants (Cover + 3 Stego) for selected IDs
        train_df = train_df[train_df["image_id"].isin(selected_ids)].reset_index(
            drop=True
        )

    # Create Datasets
    # Train uses augmentation and balanced sampling
    train_dataset = StegoDataset(
        train_df, transform=get_transforms("train"), balanced=True
    )
    # Val/Test use deterministic transforms and standard sampling
    val_dataset = StegoDataset(val_df, transform=get_transforms("val"), balanced=False)
    test_dataset = StegoDataset(
        test_df, transform=get_transforms("val"), balanced=False
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = MonoResidualResNet(pretrained=True)
    model.to(DEVICE)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FAST_EPOCHS)

    # 5. Training
    # run_training handles the loop, validation, early stopping, and TTA prediction on test set
    print("Starting training...")
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_epochs=FAST_EPOCHS,
        early_stopping_patience=3,
        label_smoothing=0.05,
    )

    # 6. Final Evaluation & Failure Analysis
    print("\nRunning Final Validation and Failure Analysis...")

    # Load best model weights saved by run_training
    best_model_path = os.path.join(IDEA_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("Warning: Best model not found. Using current weights.")

    model.eval()

    # Collect predictions and targets for the validation set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute and Print Final Metric
    final_score = weighted_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_score}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")

    # Construct analysis dataframe
    # val_loader iterates val_df sequentially (shuffle=False)
    analysis_df = val_df.copy()

    # Ensure lengths match (loader might drop last if configured, but here drop_last=False)
    if len(analysis_df) != len(all_preds):
        print(
            f"Warning: Mismatch in analysis lengths. DF: {len(analysis_df)}, Preds: {len(all_preds)}"
        )
        # Truncate to minimum length to prevent crash
        min_len = min(len(analysis_df), len(all_preds))
        analysis_df = analysis_df.iloc[:min_len]
        all_preds = all_preds[:min_len]
        all_targets = all_targets[:min_len]

    analysis_df["pred"] = all_preds
    analysis_df["target"] = all_targets
    analysis_df["error"] = np.abs(analysis_df["target"] - analysis_df["pred"])

    # Feature Extraction: File Size
    # File size is a proxy for image complexity/entropy in JPEGs
    file_sizes = []
    for path in analysis_df["image_path"]:
        full_path = os.path.join(INPUT_ROOT, path)
        try:
            file_sizes.append(os.path.getsize(full_path))
        except OSError:
            file_sizes.append(0)

    analysis_df["file_size"] = file_sizes

    # 1. Correlation with File Size
    corr_size = analysis_df["error"].corr(analysis_df["file_size"])
    print(f"Correlation between Error and File Size: {corr_size:.4f}")

    # 2. Error by Source Algorithm
    print("\nMean Error by Source:")
    print(analysis_df.groupby("source")["error"].mean())

    # 3. Error by Class
    print("\nMean Error by Class:")
    print(analysis_df.groupby("label")["error"].mean())

    print("\nRun complete.")


if __name__ == "__main__":
    main()
