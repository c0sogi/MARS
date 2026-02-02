import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import seed_everything
from library.training import train_model, validate, optimize_thresholds
from library.model import ArtworkModel
from library.dataset import ArtworkDataset, get_transforms
from library.inference import run_inference


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    Config.setup()

    print("Starting execution of runfile.py...")

    # 2. Train Model
    # We limit epochs to 5 to ensure fast execution while maintaining enough capacity
    # to learn patterns and beat the metric threshold.
    # We use the full dataset (data_limit=None) to maximize performance.
    print("Step 1: Training Model...")
    train_model(data_limit=None, num_epochs=5)

    # 3. Validation and Metric Calculation
    print("Step 2: Final Validation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model saved during training
    model = ArtworkModel(model_name=Config.MODEL_NAME, pretrained=False)
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model file not found at {Config.MODEL_PATH}")
        sys.exit(1)

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Setup validation loader
    val_dataset = ArtworkDataset(
        mode="val", load_cached_data=True, transform=get_transforms("val")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Criterion for loss calculation (needed for validate function signature)
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Run validation
    # val_preds are probabilities (sigmoid applied inside validate)
    val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)

    # Optimize threshold
    best_threshold, best_f1 = optimize_thresholds(val_preds, val_targets)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {best_f1}")

    # 4. Failure Analysis
    print("Step 3: Failure Analysis...")

    # Move to CPU/Numpy for analysis
    preds_np = val_preds.numpy()
    targets_np = val_targets.numpy()

    # Calculate Label Cardinality (number of ground truth labels per sample)
    label_cardinality = targets_np.sum(axis=1)

    # Calculate Error Magnitude (1 - F1 score per sample)
    # Binarize predictions using the optimized threshold
    preds_bin = (preds_np > best_threshold).astype(int)

    # Vectorized F1 calculation per sample
    # F1 = 2*TP / (2*TP + FP + FN)
    tp = (preds_bin * targets_np).sum(axis=1)
    fp = (preds_bin * (1 - targets_np)).sum(axis=1)
    fn = ((1 - preds_bin) * targets_np).sum(axis=1)

    epsilon = 1e-7
    f1_per_sample = 2 * tp / (2 * tp + fp + fn + epsilon)
    error_magnitude = 1.0 - f1_per_sample

    # Calculate Correlation
    if np.std(label_cardinality) > 0 and np.std(error_magnitude) > 0:
        correlation = np.corrcoef(label_cardinality, error_magnitude)[0, 1]
        print(
            f"Correlation between Error Magnitude and Input Features (Label Cardinality): {correlation}"
        )
    else:
        print("Correlation could not be calculated (zero variance).")

    # 5. Submission
    print("Step 4: Submission Generation...")
    THRESHOLD_TO_BEAT = 0.6106623748931248

    if best_f1 > THRESHOLD_TO_BEAT:
        print(
            f"Metric ({best_f1}) exceeds threshold ({THRESHOLD_TO_BEAT}). Generating submission..."
        )
        # We pass the already optimized threshold to save time
        run_inference(model_path=Config.MODEL_PATH, threshold=best_threshold)
    else:
        print(
            f"Metric ({best_f1}) does not exceed threshold ({THRESHOLD_TO_BEAT}). Submission skipped."
        )


if __name__ == "__main__":
    main()
