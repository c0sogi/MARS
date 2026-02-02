import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from PIL import Image

# Import provided library components
from library.config import Config
from library.train import train_fold
from library.inference import predict_and_submit
from library.dataset import prepare_folds, get_dataloaders
from library.model import DogModel
from library.utils import seed_everything


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Modify Config for a fast baseline execution as required
    Config.DEBUG = True  # Limit training samples (500)
    Config.N_FOLDS = 5  # Must be >= 2 for StratifiedKFold. We only run fold 0 below.
    Config.EPOCHS_WARMUP = 1  # Minimal warmup
    Config.EPOCHS_FINE_TUNE = 1  # Minimal fine-tuning

    # Setup environment (dirs, seeds)
    Config.setup()

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    # Train the first fold. This returns the path to the best "Soup" model.
    soup_model_path = train_fold(fold_idx=0)

    # ---------------------------------------------------------
    # 3. Validation & Failure Analysis
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the validation data for Fold 0
    # Note: get_dataloaders calls prepare_folds internally.
    # We load cached data because train_fold just generated it.
    _, val_loader = get_dataloaders(fold_idx=0, load_cached_data=True)

    # Load the trained model
    model = DogModel(pretrained=False)
    state_dict = torch.load(soup_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Run Inference on Validation Set
    all_probs = []
    all_labels = []
    all_losses = []

    # We compute CrossEntropyLoss per sample to get error magnitude for failure analysis
    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            loss = criterion(logits, labels)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_losses.append(loss.cpu().numpy())

    # Concatenate results
    y_pred = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    errors = np.concatenate(all_losses)

    # Compute Final Validation Metric
    # Using the competition metric: Multi Class Log Loss
    val_metric = log_loss(y_true, y_pred, labels=list(range(Config.NUM_CLASSES)))
    print(f"Final Validation Metric: {val_metric}")

    # Failure Analysis: Correlation with Input Features
    # We need to retrieve metadata for the validation set samples
    # prepare_folds returns the full dataframe with fold assignments
    df_folds, _ = prepare_folds(load_cached_data=True)
    val_df = df_folds[df_folds["fold"] == 0].reset_index(drop=True)

    # Ensure alignment: val_loader iterates sequentially, so val_df should match
    # provided shuffle=False in val_loader (checked in library/dataset.py: it is False)

    file_sizes = []
    widths = []
    heights = []

    # Extract features
    for _, row in val_df.iterrows():
        # Construct full path
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # File Size
        try:
            size = os.path.getsize(full_path)
        except OSError:
            size = 0
        file_sizes.append(size)

        # Image Dimensions
        try:
            with Image.open(full_path) as img:
                w, h = img.size
        except Exception:
            w, h = 0, 0
        widths.append(w)
        heights.append(h)

    # Calculate Correlations
    # Handle cases where variance is 0 (e.g. if all images are same size/dummy) to avoid warnings
    if np.std(file_sizes) > 0:
        corr_size, _ = pearsonr(errors, file_sizes)
    else:
        corr_size = 0.0

    if np.std(widths) > 0:
        corr_width, _ = pearsonr(errors, widths)
    else:
        corr_width = 0.0

    if np.std(heights) > 0:
        corr_height, _ = pearsonr(errors, heights)
    else:
        corr_height = 0.0

    print(f"Correlation (Error vs File Size): {corr_size}")
    print(f"Correlation (Error vs Width): {corr_width}")
    print(f"Correlation (Error vs Height): {corr_height}")

    # ---------------------------------------------------------
    # 4. Submission
    # ---------------------------------------------------------
    target_threshold = 0.14004325100369866

    if val_metric < target_threshold:
        predict_and_submit()


if __name__ == "__main__":
    main()
