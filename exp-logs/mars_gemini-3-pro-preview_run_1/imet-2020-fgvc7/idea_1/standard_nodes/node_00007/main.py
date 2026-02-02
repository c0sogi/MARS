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
    Config.DEBUG = False

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print(f"Configuration: Epochs={Config.EPOCHS}, Full Dataset Training")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading datasets...")
    # Load full datasets (Cite solution_lesson_node_00004 for efficient loading)
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # ---------------------------------------------------------
    # 3. Training
    # ---------------------------------------------------------
    print("Starting training...")
    # Train on the full set, validate on the full validation set
    run_training(train_loader, val_loader)

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

    # Optimize Threshold (Cite solution_lesson_node_00003)
    print("Optimizing threshold...")
    best_threshold = 0.5
    best_val_f1 = 0.0

    # Sweep thresholds to find optimal balance
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds_bin = (all_preds > thresh).astype(int)
        score = f1_score(all_targets_int, preds_bin, average="micro", zero_division=0)
        if score > best_val_f1:
            best_val_f1 = score
            best_threshold = thresh

    print(f"Best Threshold found: {best_threshold:.2f} with F1: {best_val_f1:.4f}")

    # Update Config with optimized threshold for inference
    Config.THRESHOLD = best_threshold

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
    BASELINE_SCORE = 0.509113663621668

    if final_f1 > BASELINE_SCORE:
        print(
            f"Validation metric ({final_f1:.4f}) > Baseline ({BASELINE_SCORE:.4f}). Generating submission..."
        )
        inference(test_loader)
        print("Done.")
    else:
        print(
            f"Validation metric ({final_f1:.4f}) <= Baseline ({BASELINE_SCORE:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
