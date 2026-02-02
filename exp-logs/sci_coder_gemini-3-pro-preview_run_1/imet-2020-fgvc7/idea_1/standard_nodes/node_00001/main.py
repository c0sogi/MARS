import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders, ArtworkDataset, get_transforms
from library.engine import run_training, inference
from library.model import ArtworkResNet


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for a fast baseline execution
    Config.EPOCHS = 5
    Config.DEBUG = False

    # Define subset size for training to ensure speed
    TRAIN_SUBSET_SIZE = 20000

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print(f"Configuration: Epochs={Config.EPOCHS}, Train Subset={TRAIN_SUBSET_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading datasets...")
    # Load full datasets first to get correct validation/test loaders
    train_loader_full, val_loader, test_loader = get_dataloaders(debug=False)

    # Manually subset the training data to limit samples
    full_train_df = train_loader_full.dataset.df
    subset_train_df = full_train_df.sample(
        n=TRAIN_SUBSET_SIZE, random_state=Config.SEED
    ).reset_index(drop=True)

    print(
        f"Subsetting training data from {len(full_train_df)} to {len(subset_train_df)} samples."
    )

    # Create new Dataset and DataLoader for the training subset
    train_dataset_subset = ArtworkDataset(
        subset_train_df, mode="train", transform=get_transforms(mode="train")
    )

    train_loader_subset = DataLoader(
        train_dataset_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # ---------------------------------------------------------
    # 3. Training
    # ---------------------------------------------------------
    print("Starting training...")
    # Train on the subset, validate on the full validation set
    run_training(train_loader_subset, val_loader)

    # ---------------------------------------------------------
    # 4. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("Performing validation and failure analysis...")

    device = Config.DEVICE

    # Load the best model saved during training
    model = ArtworkResNet(num_classes=Config.NUM_CLASSES, pretrained=False)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Error: Model file not found. Training might have failed.")
        sys.exit(1)

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    # We need to iterate over the validation loader to get predictions
    # val_loader is sequential (shuffle=False), so it aligns with val_loader.dataset.df

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Convert targets to integer for bitwise operations later
    all_targets_int = all_targets.astype(int)

    # Calculate Final Metric
    preds_binary = (all_preds > Config.THRESHOLD).astype(int)
    final_f1 = f1_score(all_targets_int, preds_binary, average="micro", zero_division=0)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    print("Running failure analysis...")

    # 1. Calculate Error Magnitude per sample (1 - F1 score of that sample)
    # Manual calculation of per-sample F1 is efficient
    f1_per_sample = []
    for p, t in zip(preds_binary, all_targets_int):
        tp = np.sum(p & t)
        fp = np.sum(p & (1 - t))
        fn = np.sum((1 - p) & t)

        denominator = 2 * tp + fp + fn
        if denominator > 0:
            score = 2 * tp / denominator
        else:
            # If target has no labels and prediction has no labels, score is 1
            # If target has no labels but prediction does, score is 0
            score = 1.0 if np.sum(t) == 0 else 0.0
        f1_per_sample.append(score)

    f1_per_sample = np.array(f1_per_sample)
    error_magnitude = 1.0 - f1_per_sample

    # 2. Feature: Number of Attributes (Ground Truth Complexity)
    label_counts = all_targets_int.sum(axis=1)

    # 3. Feature: File Size (Input Feature)
    # Access metadata from the dataset
    val_df = val_loader.dataset.df
    file_sizes = []

    # Retrieve file sizes
    for _, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            size = os.path.getsize(full_path)
        except OSError:
            size = 0
        file_sizes.append(size)
    file_sizes = np.array(file_sizes)

    # Calculate Correlations
    if len(error_magnitude) > 1:
        corr_labels = np.corrcoef(error_magnitude, label_counts)[0, 1]
        corr_size = np.corrcoef(error_magnitude, file_sizes)[0, 1]
    else:
        corr_labels = 0.0
        corr_size = 0.0

    print(f"Correlation between Error (1-F1) and Label Count: {corr_labels:.4f}")
    print(f"Correlation between Error (1-F1) and File Size: {corr_size:.4f}")

    # ---------------------------------------------------------
    # 5. Inference & Submission
    # ---------------------------------------------------------
    print("Generating submission...")
    inference(test_loader)
    print("Done.")


if __name__ == "__main__":
    main()
